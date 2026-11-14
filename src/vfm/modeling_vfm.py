"""VFM: a criterion verifier with tree attention over a shared (context, target) prefix.

The backbone reads the prefix once, followed by N criterion blocks. A 4D attention mask
lets each criterion attend to the full prefix and causally to itself, but not to other
criteria, and the prefix never attends to criteria. Every criterion block starts at the
position right after the prefix, so verdicts do not depend on how many criteria are
scored or in what order. A linear head maps the hidden state of the last token of each
block to YES / NO / N/A logits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from transformers.utils import ModelOutput

from .collator_vfm import VFMCollator
from .configuration_vfm import LABEL_NAMES, VFMConfig


@dataclass
class VFMOutput(ModelOutput):
    """Outputs of `VFMForCriteriaVerification`.

    logits: (B, N, 3) YES / NO / N/A logits, one row per criterion.
    loss: cross-entropy against `labels` when given (ignore index -100).
    prefix_hidden_states: (B, L, H) last-layer prefix states, only when requested
        (used by the KL regularizer during pretraining).
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    prefix_hidden_states: Optional[torch.FloatTensor] = None


def _criterion_mask(n_blocks: int, block: int, crit_mask: torch.Tensor) -> torch.Tensor:
    """(B, 1, n_blocks*block, n_blocks*block) boolean mask among criterion tokens: causal
    within each block, no attention across blocks."""
    device = crit_mask.device
    idx = torch.arange(n_blocks * block, device=device)
    r, c = idx.unsqueeze(1), idx.unsqueeze(0)
    allowed = (r >= c) & (r // block == c // block)
    return allowed.unsqueeze(0).unsqueeze(0) & (crit_mask.unsqueeze(1).unsqueeze(1) == 1)


def _readout(hidden: torch.Tensor, crit_mask: torch.Tensor, n_blocks: int, block: int) -> torch.Tensor:
    """Hidden state at the last non-padding token of each criterion block: (B, N, H)."""
    B, _, H = hidden.shape
    blocks = hidden.reshape(B, n_blocks, block, H)
    last = (crit_mask.reshape(B, n_blocks, block).sum(-1).long() - 1).clamp(min=0)
    return torch.gather(blocks, 2, last[:, :, None, None].expand(-1, -1, 1, H)).squeeze(2)


class VFMPreTrainedModel(PreTrainedModel):
    config_class = VFMConfig
    base_model_prefix = "vfm"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = False
    _supports_flash_attn_2 = False
    _no_split_modules = ["Qwen3DecoderLayer"]

    def _init_weights(self, module):
        std = getattr(self.config.backbone_config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif hasattr(module, "weight") and module.__class__.__name__.endswith("RMSNorm"):
            module.weight.data.fill_(1.0)


class VFMForCriteriaVerification(VFMPreTrainedModel):
    """Backbone + ternary verification head. See the module docstring for the layout."""

    def __init__(self, config: VFMConfig, backbone: Optional[nn.Module] = None):
        super().__init__(config)
        bcfg = backbone.config if backbone is not None else config.backbone_config
        # The tree mask is a dense 4D mask, which flash-attention kernels do not accept.
        if getattr(bcfg, "_attn_implementation", None) in (None, "flash_attention_2", "flash_attention_3"):
            bcfg._attn_implementation = "sdpa"
        if backbone is None:
            backbone = AutoModel.from_config(bcfg)
        self.backbone = backbone
        self.verification_head = nn.Linear(config.hidden_size, len(LABEL_NAMES), bias=True)
        self.post_init()

    # ------------------------------------------------------------------ helpers
    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def _float_mask(self, mask_bool: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        zero = torch.tensor(0.0, dtype=dtype, device=mask_bool.device)
        neg = torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=mask_bool.device)
        return torch.where(mask_bool, zero, neg)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        input_ids: torch.LongTensor,
        criterion_input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        criterion_attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_prefix_hidden_states: bool = False,
        **kwargs,
    ) -> VFMOutput:
        """Score N criteria per row in one pass.

        input_ids, attention_mask: (B, L) left-padded prefix.
        criterion_input_ids, criterion_attention_mask: (B, N*T) right-padded criterion
            blocks of T = config.max_criterion_length tokens each.
        labels: optional (B, N) with 0=YES, 1=NO, 2=N/A, -100=ignore.
        """
        B, L = input_ids.shape
        T = self.config.max_criterion_length
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if criterion_attention_mask is None:
            criterion_attention_mask = torch.ones_like(criterion_input_ids)
        if criterion_input_ids.shape[1] % T:
            raise ValueError(f"criterion_input_ids length must be a multiple of {T}")
        N = criterion_input_ids.shape[1] // T
        C = N * T

        embed = self.get_input_embeddings()
        inputs_embeds = torch.cat([embed(input_ids), embed(criterion_input_ids)], dim=1)

        # Positions: prefix 0..L-1; every criterion block continues from the prefix end.
        prefix_pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        crit_pos = (L + torch.arange(T, device=device)).repeat(N).unsqueeze(0).expand(B, -1)
        position_ids = torch.cat([prefix_pos, crit_pos], dim=1)

        # Tree attention mask (True = may attend).
        key_ok = attention_mask.unsqueeze(1).unsqueeze(1) == 1                       # (B,1,1,L)
        mask = torch.zeros((B, 1, L + C, L + C), dtype=torch.bool, device=device)
        causal = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))
        mask[:, :, :L, :L] = causal.unsqueeze(0).unsqueeze(0) & key_ok
        mask[:, :, L:, :L] = key_ok.expand(B, 1, C, L)
        mask[:, :, L:, L:] = _criterion_mask(N, T, criterion_attention_mask)

        out = self.backbone(inputs_embeds=inputs_embeds, position_ids=position_ids,
                            attention_mask=self._float_mask(mask, inputs_embeds.dtype),
                            use_cache=False, return_dict=True)
        hidden = out.last_hidden_state
        logits = self.verification_head(_readout(hidden[:, L:], criterion_attention_mask, N, T))

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                               labels.reshape(-1).to(device), ignore_index=-100)
        return VFMOutput(loss=loss, logits=logits,
                         prefix_hidden_states=hidden[:, :L] if output_prefix_hidden_states else None)

    # -------------------------------------------------------- prefix-cached path
    @torch.no_grad()
    def encode_prefix(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = None):
        """Run the backbone on the prefix only; returns (cache, prefix_length)."""
        from transformers import DynamicCache

        L = input_ids.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        cache = DynamicCache(config=self.backbone.config)
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                            past_key_values=cache, use_cache=True, return_dict=True)
        return out.past_key_values, L

    @torch.no_grad()
    def score_with_prefix_cache(self, criterion_input_ids, criterion_attention_mask, cache, prefix_length,
                                prefix_attention_mask=None) -> torch.Tensor:
        """Logits (B, N, 3) for criterion blocks attending to a cached prefix.

        The cache is cropped back to the prefix afterwards, so it can be reused for the
        next chunk of criteria.
        """
        B = criterion_input_ids.shape[0]
        T = self.config.max_criterion_length
        N = criterion_input_ids.shape[1] // T
        C = N * T
        L = prefix_length
        device = criterion_input_ids.device
        if prefix_attention_mask is None:
            prefix_attention_mask = torch.ones((B, L), dtype=torch.long, device=device)
        inputs_embeds = self.get_input_embeddings()(criterion_input_ids)
        position_ids = (L + torch.arange(T, device=device)).repeat(N).unsqueeze(0).expand(B, -1)
        mask = torch.zeros((B, 1, C, L + C), dtype=torch.bool, device=device)
        mask[:, :, :, :L] = (prefix_attention_mask.unsqueeze(1).unsqueeze(1) == 1).expand(B, 1, C, L)
        mask[:, :, :, L:] = _criterion_mask(N, T, criterion_attention_mask)
        try:
            out = self.backbone(inputs_embeds=inputs_embeds, position_ids=position_ids,
                                attention_mask=self._float_mask(mask, inputs_embeds.dtype),
                                past_key_values=cache, cache_position=torch.arange(L, L + C, device=device),
                                use_cache=True, return_dict=True)
        finally:
            extra = cache.get_seq_length() - L
            if extra > 0:
                cache.crop(-extra)
        return self.verification_head(_readout(out.last_hidden_state, criterion_attention_mask, N, T))

    # ------------------------------------------------------------- high level
    def _collator(self, tokenizer, max_length: Optional[int] = None) -> VFMCollator:
        return VFMCollator(tokenizer, max_length=max_length or self.config.max_prefix_length,
                           max_criterion_length=self.config.max_criterion_length)

    @staticmethod
    def _verdicts(criteria: Sequence[str], logits: torch.Tensor) -> List[Dict[str, Any]]:
        logits = logits.float()
        probs = logits.softmax(-1).tolist()
        binary = logits[:, :2].softmax(-1)[:, 0].tolist()
        out = []
        for c, p, pb, lg in zip(criteria, probs, binary, logits.tolist()):
            out.append({"criterion": c, "verdict": LABEL_NAMES[max(range(3), key=p.__getitem__)],
                        "p_yes": p[0], "p_no": p[1], "p_na": p[2], "p_yes_binary": pb, "logits": lg})
        return out

    @torch.no_grad()
    def verify(self, tokenizer, context, target: Optional[str], criteria: Sequence[str],
               chunk_size: int = 64, max_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """Score many criteria against one (context, target).

        The prefix is encoded once and its key-value cache is reused for chunks of
        `chunk_size` criteria, so memory stays flat in the number of criteria.

        Args:
            tokenizer: the model's tokenizer.
            context: a string (one user turn) or a list of {"role", "content"} messages.
            target: the text being verified (e.g. the assistant response), or None.
            criteria: natural-language yes/no questions.
        Returns:
            One dict per criterion: criterion, verdict (YES / NO / N/A), p_yes, p_no,
            p_na, p_yes_binary (P(YES) renormalized over YES and NO), and the raw logits.
        """
        if isinstance(criteria, str):
            criteria = [criteria]
        if not criteria:
            return []
        self.eval()
        device = next(self.parameters()).device
        col = self._collator(tokenizer, max_length)
        pre = col.encode_prefixes([{"context": context, "target": target}])
        cache, L = self.encode_prefix(pre["input_ids"].to(device), pre["attention_mask"].to(device))
        results = []
        for i in range(0, len(criteria), chunk_size):
            chunk = list(criteria[i:i + chunk_size])
            enc = col.encode_criteria(chunk)
            ids = enc["criterion_input_ids"].reshape(1, -1).to(device)
            am = enc["criterion_attention_mask"].reshape(1, -1).to(device)
            logits = self.score_with_prefix_cache(ids, am, cache, L, pre["attention_mask"].to(device))
            results.extend(self._verdicts(chunk, logits[0]))
        return results

    @torch.no_grad()
    def verify_batch(self, tokenizer, rows: Sequence[Dict[str, Any]], batch_size: int = 16,
                     max_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """Score single-criterion rows {context, target, criterion}, batched with padding.

        Use this for many different (context, target) pairs; use `verify` for many
        criteria on one pair.
        """
        self.eval()
        device = next(self.parameters()).device
        col = self._collator(tokenizer, max_length)
        results = []
        for i in range(0, len(rows), batch_size):
            chunk = [{"context": r.get("context"), "target": r.get("target"),
                      "criteria": [r["criterion"]]} for r in rows[i:i + batch_size]]
            batch = col(chunk)
            out = self(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device),
                       criterion_input_ids=batch["criterion_input_ids"].to(device),
                       criterion_attention_mask=batch["criterion_attention_mask"].to(device))
            results.extend(self._verdicts([r["criteria"][0] for r in chunk], out.logits[:, 0]))
        return results


VFMConfig.register_for_auto_class()
VFMForCriteriaVerification.register_for_auto_class("AutoModel")
