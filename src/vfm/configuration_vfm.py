"""Configuration for the VFM criterion verifier."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from transformers import AutoConfig, PretrainedConfig

LABEL_NAMES = ["YES", "NO", "N/A"]


class VFMConfig(PretrainedConfig):
    r"""Configuration of a VFM: a decoder-only backbone plus a 3-way (YES / NO / N/A) head.

    Args:
        backbone_config (`dict` or `PretrainedConfig`): configuration of the backbone
            (e.g. a `Qwen3Config`). The backbone is built from this config, so loading a
            VFM never downloads the base model.
        max_criterion_length (`int`, defaults to 64): token length of each criterion
            block, including the chat-template tokens. Longer criteria are truncated
            from the left, as in training.
        max_prefix_length (`int`, defaults to 4096): default maximum token length of the
            shared (context, target) prefix. Longer prefixes are truncated from the left.
        verbalizers (`list[str]`): verbalizer strings whose unembedding rows initialized
            the head (training provenance only).
        verbalizer_token_ids (`list[int]`): token ids of `verbalizers`.
        base_model_name_or_path (`str`): the base model the backbone was trained from
            (provenance only; not loaded).
        preserve_loss (`str`): pretraining regularizer, `"kl_base"` or `"none"`
            (training metadata only).
        preserve_lambda (`float`): weight of the pretraining regularizer.
    """

    model_type = "vfm"
    sub_configs = {"backbone_config": AutoConfig}

    def __init__(
        self,
        backbone_config: Optional[Union[Dict[str, Any], PretrainedConfig]] = None,
        max_criterion_length: int = 64,
        max_prefix_length: int = 4096,
        verbalizers: Optional[List[str]] = None,
        verbalizer_token_ids: Optional[List[int]] = None,
        base_model_name_or_path: Optional[str] = None,
        preserve_loss: str = "none",
        preserve_lambda: float = 0.0,
        **kwargs,
    ):
        if isinstance(backbone_config, dict):
            backbone_config = dict(backbone_config)
            model_type = backbone_config.pop("model_type", "qwen3")
            backbone_config = AutoConfig.for_model(model_type, **backbone_config)
        self.backbone_config = backbone_config
        self.max_criterion_length = max_criterion_length
        self.max_prefix_length = max_prefix_length
        self.verbalizers = verbalizers
        self.verbalizer_token_ids = verbalizer_token_ids
        self.base_model_name_or_path = base_model_name_or_path
        self.preserve_loss = preserve_loss
        self.preserve_lambda = preserve_lambda
        kwargs.setdefault("id2label", {i: n for i, n in enumerate(LABEL_NAMES)})
        kwargs.setdefault("label2id", {n: i for i, n in enumerate(LABEL_NAMES)})
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)

    @property
    def hidden_size(self) -> int:
        return self.backbone_config.hidden_size
