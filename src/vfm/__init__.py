"""VFM: runtime verification of many natural-language criteria in one forward pass."""
from transformers import AutoConfig, AutoModel

from .collator_vfm import VFMCollator
from .configuration_vfm import VFMConfig
from .inference import VFM
from .modeling_vfm import VFMForCriteriaVerification, VFMOutput

AutoConfig.register("vfm", VFMConfig, exist_ok=True)
AutoModel.register(VFMConfig, VFMForCriteriaVerification, exist_ok=True)

__version__ = "1.0.0"

__all__ = ["VFM", "VFMConfig", "VFMCollator", "VFMForCriteriaVerification", "VFMOutput"]
