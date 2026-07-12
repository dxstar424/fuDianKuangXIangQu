"""
v0.9.3: No-op — weight quantization is a dead end on this DCU.

torch._scaled_mm requires ROCm MI300+ (not gfx942).
Dequant+matmul fallback has 81GB HBM traffic vs 54GB bf16 (1.5x slower).

This hook point is kept alive for future use.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("fdu_vllm.fp8_fallback")

_PATCHED = False


def activate_fp8_fallback() -> bool:
    global _PATCHED
    _PATCHED = True
    logger.info("FDU fp8_fallback: no-op (v0.9.3 — bf16 stock)")
    return True


def is_patched() -> bool:
    return _PATCHED
