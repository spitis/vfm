#!/usr/bin/env python3
"""Train a VFM: CriteriaBank pretraining from a base LM, or downstream finetuning.

  # Downstream finetuning (paper recipe), one seed:
  python scripts/finetune.py --config scripts/configs/finetune.yaml \
      --model spitis/VFM-0.6B --train data/prm800k/train.jsonl \
      --dev data/prm800k/dev300_s1.jsonl --seed 1 --output-dir runs/prm800k_vfm0.6b_s1

  # Then select the checkpoint and threshold on the same dev set and score the test set:
  python scripts/evaluate.py --benchmark prm800k --model "runs/prm800k_vfm0.6b_s1/checkpoint-*" \
      --dev data/prm800k/dev300_s1.jsonl

  # CriteriaBank pretraining with the KL regularizer (multi-GPU via torchrun):
  torchrun --nproc_per_node 4 scripts/finetune.py --config scripts/configs/pretrain_kl_0.6b.yaml

Every option can be set in the YAML config and overridden on the command line.
Effective batch = per_device_batch_size * gradient_accumulation_steps * number of GPUs.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from transformers import AutoTokenizer, EarlyStoppingCallback, TrainingArguments  # noqa: E402

from vfm import VFMCollator, VFMForCriteriaVerification  # noqa: E402
from vfm.data import read_jsonl  # noqa: E402
from vfm.training import (VFMTrainer, calibrate_head_bias, compute_metrics,  # noqa: E402
                          flatten_for_training, init_from_base, load_frozen_base)

CALIBRATION_ROWS = 512  # rows used to mean-center the head bias after the verbalizer warm-start

DEFAULTS = {
    "model": None,                 # VFM checkpoint to finetune (Hub id or path)
    "init_from_base": None,        # or: base LM to start from (pretraining / Direct-FT control)
    "train": None,
    "dev": None,
    "output_dir": "runs/vfm",
    "seed": 42,
    "head_learning_rate": 1e-4,
    "backbone_learning_rate": 5e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,          # head weight only; the backbone and head bias are not decayed
    "max_grad_norm": 1.0,
    "num_train_epochs": 30,
    "max_steps": -1,
    "early_stopping_patience": 5,  # evaluations without dev improvement; 0 disables
    "eval_steps": None,            # evaluate on --dev and save every N steps; null = once per epoch
    "save_total_limit": 12,
    "load_best_model_at_end": True,  # with --dev: keep the checkpoint with the best dev macro-F1
    "per_device_batch_size": 4,
    "gradient_accumulation_steps": 32,
    "max_length": 4096,
    "kl_lambda": 0.0,              # > 0 adds lambda * KL(p_base || p_theta) to the frozen init_from_base
    "report_to": "none",
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    for k, v in DEFAULTS.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            ap.add_argument(flag, default=None,
                            type=int if k == "eval_steps" else (type(v) if v is not None else str))
    ns = ap.parse_args()
    cfg = dict(DEFAULTS)
    if ns.config:
        with open(ns.config) as f:
            loaded = yaml.safe_load(f) or {}
        unknown = sorted(set(loaded) - set(DEFAULTS))
        if unknown:
            raise SystemExit(f"Unknown keys in {ns.config}: {', '.join(unknown)} (valid: {', '.join(DEFAULTS)})")
        cfg.update(loaded)
    cfg.update({k: v for k, v in vars(ns).items() if k != "config" and v is not None})
    for k, default in DEFAULTS.items():  # YAML may read e.g. 5e-5 as a string
        if isinstance(default, (int, float)) and not isinstance(default, bool) and isinstance(cfg[k], str):
            cfg[k] = type(default)(float(cfg[k]))
    return cfg


def main():
    cfg = parse_args()
    if bool(cfg["model"]) == bool(cfg["init_from_base"]):
        raise SystemExit("Set exactly one of --model or --init-from-base")
    if not cfg["train"]:
        raise SystemExit("--train is required")
    torch.manual_seed(cfg["seed"])
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    if cfg["model"]:
        model = VFMForCriteriaVerification.from_pretrained(cfg["model"], dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    else:
        model, tokenizer = init_from_base(cfg["init_from_base"], dtype)
    model.config.max_prefix_length = cfg["max_length"]  # saved with the checkpoint
    model.config.preserve_loss = "kl_base" if cfg["kl_lambda"] > 0 else "none"
    model.config.preserve_lambda = cfg["kl_lambda"]

    collator = VFMCollator(tokenizer, max_length=cfg["max_length"],
                           max_criterion_length=model.config.max_criterion_length)
    train = flatten_for_training(read_jsonl(cfg["train"]))
    dev = flatten_for_training(read_jsonl(cfg["dev"])) if cfg["dev"] else None
    print(f"train rows: {len(train)}" + (f", dev rows: {len(dev)}" if dev else ""))

    if cfg["init_from_base"]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.distributed.is_available() and "LOCAL_RANK" in os.environ:
            device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        model.to(device)
        rows = train[:CALIBRATION_ROWS]  # first rows, identical on every rank
        bs = cfg["per_device_batch_size"]
        bias = calibrate_head_bias(model, (collator(rows[i:i + bs]) for i in range(0, len(rows), bs)))
        print("calibrated head bias (YES, NO, N/A):", [round(x, 4) for x in bias.float().tolist()])

    world = int(os.environ.get("WORLD_SIZE", 1))
    eff_batch = cfg["per_device_batch_size"] * cfg["gradient_accumulation_steps"] * world
    steps_per_epoch = max(1, math.ceil(len(train) / eff_batch))
    eval_steps = cfg["eval_steps"] or steps_per_epoch
    has_dev = dev is not None
    targs = TrainingArguments(
        output_dir=cfg["output_dir"], seed=cfg["seed"],
        per_device_train_batch_size=cfg["per_device_batch_size"],
        per_device_eval_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["head_learning_rate"], weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"], warmup_ratio=cfg["warmup_ratio"],
        max_grad_norm=cfg["max_grad_norm"], num_train_epochs=cfg["num_train_epochs"],
        max_steps=cfg["max_steps"], bf16=dtype == torch.bfloat16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps" if has_dev else "no", eval_steps=eval_steps,
        save_strategy="steps", save_steps=eval_steps, save_total_limit=cfg["save_total_limit"],
        save_only_model=True, load_best_model_at_end=has_dev and cfg["load_best_model_at_end"],
        metric_for_best_model="eval_f1_macro", greater_is_better=True,
        logging_steps=10, report_to=cfg["report_to"],
        dataloader_num_workers=2, remove_unused_columns=False,
        label_names=["labels"], ddp_find_unused_parameters=False,
    )
    callbacks = []
    if has_dev and cfg["early_stopping_patience"] > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"]))
    kl_base = None
    if cfg["kl_lambda"] > 0:
        if not cfg["init_from_base"]:
            raise SystemExit("--kl-lambda > 0 needs --init-from-base")
        kl_base = load_frozen_base(cfg["init_from_base"], dtype)
    trainer = VFMTrainer(
        model=model, args=targs, train_dataset=train, eval_dataset=dev, data_collator=collator,
        processing_class=tokenizer, compute_metrics=compute_metrics if has_dev else None,
        callbacks=callbacks, head_learning_rate=cfg["head_learning_rate"],
        backbone_learning_rate=cfg["backbone_learning_rate"], kl_base_model=kl_base,
        kl_lambda=cfg["kl_lambda"],
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(cfg["output_dir"])
        print("saved", cfg["output_dir"])


if __name__ == "__main__":
    main()
