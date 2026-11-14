"""High-level inference API.

    from vfm import VFM

    vfm = VFM.from_pretrained("spitis/VFM-4B")
    verdicts = vfm.verify(
        context=[{"role": "user", "content": "I have $50,000 to invest. Any advice?"}],
        target="Put it all in one stock; it is guaranteed to double.",
        criteria=["Does the response guarantee investment returns?",
                  "Does the response ask about the user's risk tolerance?"],
    )
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoTokenizer

from .modeling_vfm import VFMForCriteriaVerification


class VFM:
    """A VFM model and its tokenizer."""

    def __init__(self, model: VFMForCriteriaVerification, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, name_or_path: str, device: Optional[str] = None,
                        dtype: torch.dtype = torch.bfloat16, **kwargs) -> "VFM":
        """Load a VFM checkpoint (Hugging Face Hub id or local directory)."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = VFMForCriteriaVerification.from_pretrained(name_or_path, dtype=dtype, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        return cls(model.to(device), tokenizer)

    def verify(self, context, target: Optional[str], criteria: Sequence[str],
               chunk_size: int = 64, max_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """Score criteria against one (context, target); see `VFMForCriteriaVerification.verify`."""
        return self.model.verify(self.tokenizer, context, target, criteria,
                                 chunk_size=chunk_size, max_length=max_length)

    def verify_batch(self, rows: Sequence[Dict[str, Any]], batch_size: int = 16,
                     max_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """Score rows {context, target, criterion}; see `VFMForCriteriaVerification.verify_batch`."""
        return self.model.verify_batch(self.tokenizer, rows, batch_size=batch_size, max_length=max_length)
