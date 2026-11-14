"""Tokenization and batching for the VFM.

The (context, target) prefix is rendered with the backbone's chat template, with the
target appended as the next turn. Each criterion is rendered as its own one-turn chat
prompt (user message + assistant generation prompt) and padded to a fixed block of
`max_criterion_length` tokens; the head reads the last non-padding token of each block.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from .configuration_vfm import LABEL_NAMES

Messages = List[Dict[str, Any]]


def _messages(context: Union[str, Messages, None]) -> Messages:
    if context is None:
        return []
    if isinstance(context, str):
        return [{"role": "user", "content": context}]
    return list(context)


def _next_role(messages: Messages) -> str:
    """Role of the target turn: assistant after a user turn, otherwise user."""
    if not messages:
        return "user"
    return "assistant" if messages[-1].get("role", "user") == "user" else "user"


def format_prefix(tokenizer, context: Union[str, Messages, None], target: Optional[str]) -> str:
    """Render the shared (context, target) prefix as text."""
    messages = _messages(context)
    if target:
        messages = messages + [{"role": _next_role(messages), "content": target}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                         enable_thinking=False)


def format_criterion(tokenizer, criterion: str) -> str:
    """Render one criterion as a one-turn chat prompt ending at the answer position (with
    thinking disabled, so the template inserts an empty think block)."""
    return tokenizer.apply_chat_template([{"role": "user", "content": criterion}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


class VFMCollator:
    """Turns VFM rows into model inputs.

    Each row is a dict with `context`, `target`, `criteria` (list of N strings) and
    optionally `labels` (list of N labels). All rows in a batch must have the same N
    (single-criterion rows are the common case for training).

    Returns `input_ids`/`attention_mask` (B, L), left-padded and left-truncated, and
    `criterion_input_ids`/`criterion_attention_mask` (B, N*T) with T =
    `max_criterion_length`, plus `labels` (B, N) when labels are present.
    """

    def __init__(self, tokenizer, max_length: int = 4096, max_criterion_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_criterion_length = max_criterion_length
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._warned_long = False

    def _tokenize(self, texts: List[str], max_length: int, padding_side: str, padding):
        tok = self.tokenizer
        old = (tok.padding_side, tok.truncation_side)
        tok.padding_side, tok.truncation_side = padding_side, "left"
        try:
            return tok(texts, padding=padding, truncation=True, max_length=max_length,
                       return_tensors="pt")
        finally:
            tok.padding_side, tok.truncation_side = old

    def encode_prefixes(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [format_prefix(self.tokenizer, r.get("context"), r.get("target")) for r in rows]
        enc = self._tokenize(texts, self.max_length, "left", "longest")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def encode_criteria(self, criteria: Sequence[str]) -> Dict[str, torch.Tensor]:
        """Tokenize criteria into fixed-size blocks: returns (N, T) tensors."""
        texts = [format_criterion(self.tokenizer, c) for c in criteria]
        if not self._warned_long:
            lengths = [len(self.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
            if max(lengths, default=0) > self.max_criterion_length:
                warnings.warn(
                    f"Some criteria exceed {self.max_criterion_length} tokens including the chat "
                    "template and are truncated from the left, as in training. Keep criteria short "
                    "and atomic.", stacklevel=3)
                self._warned_long = True
        enc = self._tokenize(texts, self.max_criterion_length, "right", "max_length")
        return {"criterion_input_ids": enc["input_ids"],
                "criterion_attention_mask": enc["attention_mask"]}

    def __call__(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        n = len(rows[0]["criteria"])
        if any(len(r["criteria"]) != n for r in rows):
            raise ValueError("All rows in a batch must have the same number of criteria")
        batch = self.encode_prefixes(rows)
        crit = self.encode_criteria([c for r in rows for c in r["criteria"]])
        B, T = len(rows), self.max_criterion_length
        batch["criterion_input_ids"] = crit["criterion_input_ids"].reshape(B, n * T)
        batch["criterion_attention_mask"] = crit["criterion_attention_mask"].reshape(B, n * T)
        if rows[0].get("labels") is not None:
            batch["labels"] = torch.tensor([[LABEL_NAMES.index(l) for l in r["labels"]] for r in rows],
                                           dtype=torch.long)
        return batch

