"""
v0.9.2: Global monkey-patch of torch._scaled_mm to avoid platform crash.

torch._scaled_mm requires ROCm MI300+ which the platform gfx942 DCU doesn't have.
ALL ROCm FP8 kernel classes call torch._scaled_mm — patching just one class
(ROCmFP8ScaledMMLinearKernel) was insufficient because the kernel selector
falls through to PerTensorTorchFP8ScaledMMLinearKernel etc. when on_mi3xx()
returns False.

This patch replaces torch._scaled_mm globally with a dequant+matmul fallback
that works on any GPU. The decode path (M <= 4) in ROCmFP8ScaledMMLinearKernel
uses wvSplitKQ and never calls torch._scaled_mm, so it's unaffected.
"""

from __future__ import annotations

import logging
import torch

logger = logging.getLogger("fdu_vllm.fp8_fallback")

_PATCHED = False
_ORIGINAL_SCALED_MM = None


def _fallback_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    scale_a: torch.Tensor | None = None,
    scale_b: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize FP8→fp32 then torch.matmul. Works on any GPU.

    torch._scaled_mm semantics: C = (A * scale_a) @ (B * scale_b) + bias
    where A, B are FP8, scale_a/b are scaling factors.
    """
    if out_dtype is None:
        out_dtype = A.dtype

    # Dequantize: convert FP8 → fp32, apply scales
    A_dq = A.to(torch.float32)
    if scale_a is not None:
        # scale_a can be scalar, per-token [M,1], etc.
        # Broadcast to match A's shape
        sa = scale_a.to(torch.float32)
        while sa.dim() < A_dq.dim():
            sa = sa.unsqueeze(-1)
        A_dq = A_dq * sa

    B_dq = B.to(torch.float32)
    if scale_b is not None:
        sb = scale_b.to(torch.float32)
        while sb.dim() < B_dq.dim():
            sb = sb.unsqueeze(-1)
        B_dq = B_dq * sb

    # GEMM: A[M, K] @ B[K, N] → [M, N]
    # B is stored as [K, N] in FP8, needs transpose for matmul if column-major
    # torch._scaled_mm expects B in [K, N] (column-major), our matmul wants [K, N].t()
    result = torch.matmul(A_dq, B_dq)

    if bias is not None:
        result = result + bias.to(torch.float32)

    return result.to(out_dtype)


def activate_fp8_fallback() -> bool:
    """Replace torch._scaled_mm with a dequant+matmul fallback.

    Must be called BEFORE any FP8 inference (after vLLM import, before LLM creation).
    """
    global _PATCHED, _ORIGINAL_SCALED_MM
    if _PATCHED:
        return True

    try:
        _ORIGINAL_SCALED_MM = torch._scaled_mm
        torch._scaled_mm = _fallback_scaled_mm
        _PATCHED = True
        logger.info(
            "FDU fp8_fallback: torch._scaled_mm replaced with dequant+matmul fallback "
            "(platform gfx942 DCU doesn't support torch._scaled_mm)"
        )
        return True

    except Exception as e:
        logger.error("FDU fp8_fallback: patch failed: %s", e)
        return False


def is_patched() -> bool:
    return _PATCHED
