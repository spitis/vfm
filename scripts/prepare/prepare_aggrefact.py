#!/usr/bin/env python3
"""Prepare the LLM-AggreFact factual-consistency benchmark and its finetuning data.

Evaluation source: Hugging Face dataset ``lytang/LLM-AggreFact``,
https://huggingface.co/datasets/lytang/LLM-AggreFact (Tang et al., 2024). License:
CC BY-ND 4.0, so the benchmark is used for evaluation only. This script downloads the data
from the Hugging Face Hub; this repository does not redistribute it.

Finetuning source (never LLM-AggreFact): ANLI (``facebook/anli``, CC BY-NC 4.0), VitaminC
(``tals/vitaminc``, CC BY-SA 3.0) and FEVER with gold evidence
(``copenlu/fever_gold_evidence``, CC BY-SA 3.0). Entailment/supports -> YES,
contradiction/refutes -> NO, neutral/not-enough-info rows are dropped.

Conversion: context = the grounding document, target = the claim, criterion = "Is this claim
factually supported by the provided document?", label YES if supported else NO. The paper
evaluates on 3000 records sampled at random (seed 0) from the 29320-record test set.

Output splits (sizes as in the paper): train 10000, dev 30420, test 3000, test_full 29320,
plus, from the manifest, the 300-record dev subsets dev300_s1, dev300_s2, dev300_s3 (not
written with --no-manifest).

Usage:
    python scripts/prepare/prepare_aggrefact.py [--out-dir DIR] [--no-manifest] [--strict]
"""
from __future__ import annotations

import argparse
import random

from datasets import load_dataset

from _common import add_common_args, load_manifest, select_by_manifest, write_split
from vfm.data import make_record

CRITERION = "Is this claim factually supported by the provided document?"
TEST_SAMPLES, TEST_SAMPLE_SEED = 3000, 0


def eval_splits():
    ds = load_dataset("lytang/LLM-AggreFact")
    splits = {name: [make_record([{"role": "user", "content": s["doc"]}], s["claim"], CRITERION,
                                 "YES" if s["label"] == 1 else "NO", meta={"dataset": s["dataset"]})
                     for s in ds[name]]
              for name in ("dev", "test")}
    full = splits["test"]
    splits["test"] = random.Random(TEST_SAMPLE_SEED).sample(full, TEST_SAMPLES)
    splits["test_full"] = full
    return splits


def _rec(premise, hypothesis, label, source):
    return make_record([{"role": "user", "content": premise}], hypothesis, CRITERION, label,
                       meta={"source": source})


def load_anli(max_per_round, seed):
    records = []
    for name in ("train_r1", "train_r2", "train_r3"):
        rows = [_rec(s["premise"], s["hypothesis"], "YES" if s["label"] == 0 else "NO",
                     f"anli_{name}")
                for s in load_dataset("facebook/anli", split=name) if s["label"] != 1]
        random.seed(seed)
        random.shuffle(rows)
        records.extend(rows[:max_per_round])
    return records


def load_vitaminc(max_samples, seed):
    rows = []
    for s in load_dataset("tals/vitaminc", split="train"):
        lab = s["label"]
        if isinstance(lab, int):
            if lab == 2:
                continue
            label = "YES" if lab == 0 else "NO"
        else:
            if lab.upper() == "NOT ENOUGH INFO":
                continue
            label = "YES" if lab.upper() == "SUPPORTS" else "NO"
        rows.append(_rec(s["evidence"], s["claim"], label, "vitaminc"))
    random.seed(seed)
    random.shuffle(rows)
    return rows[:max_samples]


def load_fever(max_samples, seed):
    rows = []
    for s in load_dataset("copenlu/fever_gold_evidence", split="train"):
        lab = s.get("label", "")
        if isinstance(lab, int):
            lab = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}.get(lab, "")
        if lab.upper() not in ("SUPPORTS", "REFUTES"):
            continue
        raw = s.get("evidence", "")
        if isinstance(raw, list):
            sents = []
            for e in raw:
                if isinstance(e, list) and len(e) >= 3:
                    sents.append(e[2])
                elif isinstance(e, str):
                    sents.append(e)
            evidence = " ".join(sents)
        else:
            evidence = str(raw)
        claim = s.get("claim", "")
        if not evidence or len(evidence.strip()) < 10 or not claim:
            continue
        rows.append(_rec(evidence, claim, "YES" if lab.upper() == "SUPPORTS" else "NO", "fever"))
    random.seed(seed)
    random.shuffle(rows)
    return rows[:max_samples]


def train_split(seed=42, target_size=10000):
    random.seed(seed)
    per_source = int(target_size * 1.5 / 3)
    pool = (load_anli(per_source // 3, seed) + load_vitaminc(per_source, seed)
            + load_fever(per_source, seed))
    random.shuffle(pool)
    return pool[:target_size]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap, "aggrefact")
    args = ap.parse_args()
    manifest = None if args.no_manifest else load_manifest("aggrefact")
    splits = eval_splits()
    wanted = list(manifest["splits"]) if manifest else list(splits) + ["train"]
    if "train" in wanted:
        splits["train"] = train_split(seed=args.seed)
    if manifest is None:
        for name in wanted:
            write_split(args.out_dir, name, splits[name])
        return
    # dev300_* are subsets of dev; every wanted split is selected from the pooled records.
    pool = [r for name in ("dev", "test_full", "train") for r in splits[name]]
    for name in wanted:
        write_split(args.out_dir, name, select_by_manifest(pool, manifest, name, args.strict))


if __name__ == "__main__":
    main()
