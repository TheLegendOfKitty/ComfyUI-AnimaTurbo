"""
ComfyUI-AnimaTurbo: Anima/cosmos-predict2 int8 inference speedups, packaged as a
single self-contained custom_nodes directory (no comfy-kitchen fork required).

Each patch module below self-applies at import time (installs its hook, or is a
no-op if its target isn't present/importable). AnimaFuseQKV is the only component
that requires explicit placement in a workflow; everything else auto-applies as
soon as this pack loads.
"""
import logging

from . import sage_pv_fp16_patch  # noqa: F401  self-applies: sageattn -> CUDA pv=fp16 dispatch
from . import int8_shared_quant_patch  # noqa: F401  self-applies: dedupes q/k/v activation quantize (plain int8)
from . import int8_convrot_shared_quant_patch  # noqa: F401  self-applies: dedupes q/k/v activation quantize (convrot)
from . import w4a4_compile_patch  # noqa: F401  self-applies: makes ConvRot W4A4 Linear torch.compile-safe
from . import warp_fht  # noqa: F401  self-applies: JIT-builds standalone kernel + installs _C shim

try:
    from .anima_qkv_fusion import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as e:
    logging.warning(f"[ComfyUI-AnimaTurbo] anima_qkv_fusion import failed, AnimaFuseQKV node unavailable: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
