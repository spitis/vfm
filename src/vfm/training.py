"""Training utilities: model initialization from a base LM, the KL regularizer, and a Trainer.

Recipe (paper, Appendix "Training Recipe"): full-parameter training, separate learning
rates for the head and the backbone, AdamW, cosine schedule, bf16 weights and optimizer
states, weight decay 0.01 on the head weight only, gradient clipping 1.0, effective batch
128. CriteriaBank pretraining adds lambda * KL(p_base || p_theta) over the next-token
distributions of the trace (prefix) tokens, with the frozen base model as reference.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoTokenizer, Trainer

from .configuration_vfm import LABEL_NAMES, VFMConfig
from .metrics import ternary_metrics
from .modeling_vfm import VFMForCriteriaVerification

DEFAULT_VERBALIZERS = (" Yes", "No", " Unknown")


def init_from_base(base_model: str, dtype: torch.dtype = torch.bfloat16, max_prefix_length: int = 4096,
                   verbalizers: Sequence[str] = DEFAULT_VERBALIZERS):
    """Build an untrained VFM from a base LM, with the head warm-started from the
    unembedding rows of the verbalizer tokens (first token of each), bias zero."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    backbone = AutoModel.from_pretrained(base_model, dtype=dtype, attn_implementation="sdpa")
    ids = []
    for v in verbalizers:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if not toks:
            raise ValueError(f"verbalizer {v!r} encodes to no tokens")
        ids.append(toks[0])
    config = VFMConfig(backbone_config=backbone.config, max_prefix_length=max_prefix_length,
                       verbalizers=list(verbalizers),
                       verbalizer_token_ids=ids, base_model_name_or_path=base_model,
                       architectures=["VFMForCriteriaVerification"])
    model = VFMForCriteriaVerification(config, backbone=backbone)
    unembed = backbone.get_input_embeddings().weight  # tied output embeddings (Qwen3 0.6B/4B)
    with torch.no_grad():
        model.verification_head.weight.copy_(unembed[ids].to(model.verification_head.weight.dtype))
        model.verification_head.bias.zero_()
    model.to(dtype)
    return model, tokenizer


@torch.no_grad()
def calibrate_head_bias(model: VFMForCriteriaVerification, batches: Iterable[Dict[str, torch.Tensor]]):
    """Mean-center each class logit over unlabeled batches (label-free): bias -= mean logit.

    Applied once after the verbalizer warm-start, before pretraining, so that no
    verbalizer wins by default because of its baseline logit level.
    """
    model.eval()
    device = next(model.parameters()).device
    total, count = torch.zeros(3, dtype=torch.float64, device=device), 0
    for b in batches:
        out = model(**{k: v.to(device) for k, v in b.items() if k != "labels"})
        flat = out.logits.reshape(-1, 3).double()
        total += flat.sum(0)
        count += flat.shape[0]
    if count:
        model.verification_head.bias.sub_((total / count).to(model.verification_head.bias.dtype))
    return model.verification_head.bias.detach().clone()


def load_frozen_base(base_model: str, dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    base = AutoModel.from_pretrained(base_model, dtype=dtype, attn_implementation="sdpa")
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    return base


def kl_to_base(hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
               unembed: torch.Tensor, base: nn.Module, chunk: int = 256) -> torch.Tensor:
    """Mean over valid next-token positions of KL(p_base || p_theta), computed in fp32.

    hidden: (B, L, H) last-layer prefix states of the model being trained.
    Positions whose input token is padding are excluded.
    """
    H = hidden.shape[-1]
    valid = (attention_mask[:, :-1] == 1).reshape(-1)
    cur = hidden[:, :-1].reshape(-1, H)[valid]
    with torch.no_grad():
        base_hidden = base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        ref = base_hidden[:, :-1].reshape(-1, H)[valid]
    base_unembed = base.get_input_embeddings().weight
    if cur.shape[0] == 0:
        return hidden.sum() * 0.0

    def _chunk(h_cur, h_ref):
        with torch.autocast(device_type=h_cur.device.type, enabled=False):
            log_p = torch.log_softmax(h_cur.float() @ unembed.float().T, dim=-1)
            log_q = torch.log_softmax(h_ref.float() @ base_unembed.float().T, dim=-1).detach()
            return nn.functional.kl_div(log_p, log_q, log_target=True, reduction="sum")

    total = cur.new_zeros((), dtype=torch.float32)
    for i in range(0, cur.shape[0], chunk):
        h, r = cur[i:i + chunk], ref[i:i + chunk]
        total = total + (checkpoint(_chunk, h, r, use_reentrant=False) if h.requires_grad else _chunk(h, r))
    return total / cur.shape[0]


class VFMTrainer(Trainer):
    """Trainer with head/backbone learning rates and the optional KL-to-base regularizer."""

    def __init__(self, *args, head_learning_rate: float = 1e-4, backbone_learning_rate: float = 5e-5,
                 kl_base_model: Optional[nn.Module] = None, kl_lambda: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_lr, self.backbone_lr = head_learning_rate, backbone_learning_rate
        self.kl_base_model, self.kl_lambda = kl_base_model, kl_lambda
        if kl_base_model is not None:
            kl_base_model.to(self.args.device)

    def create_optimizer(self):
        if self.optimizer is None:
            head = self.model.verification_head
            backbone = [p for n, p in self.model.named_parameters()
                        if n.startswith("backbone.") and p.requires_grad]
            # As in the paper's runs, weight decay applies to the head weight only.
            groups = [{"params": backbone, "weight_decay": 0.0, "lr": self.backbone_lr},
                      {"params": [head.weight], "weight_decay": self.args.weight_decay, "lr": self.head_lr},
                      {"params": [head.bias], "weight_decay": 0.0, "lr": self.head_lr}]
            opt_cls, opt_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = opt_cls(groups, **opt_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        use_kl = self.kl_base_model is not None and self.kl_lambda > 0 and model.training
        out = model(**inputs, output_prefix_hidden_states=use_kl)
        loss = out.loss  # cross-entropy over the labelled criteria
        if use_kl:
            unwrapped = self.accelerator.unwrap_model(model)
            kl = kl_to_base(out.prefix_hidden_states, inputs["input_ids"], inputs["attention_mask"],
                            unwrapped.get_input_embeddings().weight, self.kl_base_model)
            loss = loss + self.kl_lambda * kl
        return (loss, {"logits": out.logits}) if return_outputs else loss


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Accuracy and three-class macro-F1 of argmax predictions (used for early stopping)."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    keep = labels != -100
    m = ternary_metrics([LABEL_NAMES[i] for i in preds[keep]], [LABEL_NAMES[i] for i in labels[keep]])
    return {"accuracy": m["accuracy"], "f1_macro": m["macro_f1"]}


def flatten_for_training(records: Iterable[Dict]) -> List[Dict]:
    """Single-criterion training rows from VFM records."""
    return [{"context": r["context"], "target": r["target"], "criteria": [c], "labels": [l]}
            for r in records for c, l in zip(r["criteria"], r["labels"])]
