"""
v0.9.3: No-op placeholder — no weight quantization.

Previous attempts:
  v0.8.1 bitsandbytes INT4: matmul_4bit no ROCm HIP kernel → 0 benefit
  v0.9.0 FP8 W8A8: torch._scaled_mm crashes on platform gfx942 DCU
  v0.9.2 FP8 dequant fallback: 81GB HBM traffic (vs 54GB bf16) → 1.5x slower

Weight quantization is a dead end on this platform — all quant methods
either crash or have no HIP-accelerated computation path.

Strategy: stock bf16 + AITER optimizations (FLASH_ATTN, skinny_gemm, rmsnorm).
"""

from __future__ import annotations

import logging

logger = logging.getLogger("fdu_vllm.quant_force")

_PATCHED = False


def activate_quant_force() -> bool:
    """v0.9.3: NO-OP — no weight quantization forced.

    We keep this hook point alive for future use (e.g., AWQ Triton if viable).
    """
    global _PATCHED
    _PATCHED = True
    logger.info("FDU quant_force: no-op (v0.9.3 — bf16 stock, no weight quant)")
    return True


def is_patched() -> bool:
    return _PATCHED
