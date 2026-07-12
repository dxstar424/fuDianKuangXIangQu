"""
v0.9.2: Monkey-patch Fp8LinearMethod.apply() to avoid torch._scaled_mm.

Same as vendored vllm/model_executor/layers/quantization/fp8.py patch.
This runs at fdu_vllm.activate() time as belt-and-suspenders if the
vendored source patch somehow didn't take effect (e.g., base image fp8.py).

Qwen3.5-27B is a dense model (no MoE layers), so Fp8MoEMethod is not used.
"""

from __future__ import annotations

import logging
import torch

logger = logging.getLogger("fdu_vllm.fp8_fallback")

_PATCHED = False


def activate_fp8_fallback() -> bool:
    global _PATCHED
    if _PATCHED:
        return True

    try:
        from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

        _orig = Fp8LinearMethod.apply

        def _patched(self, layer, x, bias=None):
            w = layer.weight.to(x.dtype) * layer.weight_scale.to(x.dtype)
            return torch.nn.functional.linear(x, w.t(), bias)

        Fp8LinearMethod.apply = _patched
        _PATCHED = True
        logger.info(
            "FDU fp8_fallback: Fp8LinearMethod.apply() → dequant+matmul (no torch._scaled_mm)"
        )
        return True

    except Exception as e:
        logger.warning("FDU fp8_fallback: patch failed: %s", e)
        return False


def is_patched() -> bool:
    return _PATCHED
