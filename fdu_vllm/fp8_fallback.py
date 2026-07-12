"""
v0.9.1: Monkey-patch ROCm FP8 kernel to avoid torch._scaled_mm for prefill.

torch._scaled_mm requires ROCm MI300+ which the platform DCU doesn't support.
The kernel has two paths:
  - M <= 4 (decode): wvSplitKQ — custom HIP kernel, works fine
  - M > 4  (prefill): torch._scaled_mm — CRASHES on platform DCU

This patch intercepts apply_scaled_mm on ROCmFP8ScaledMMLinearKernel:
  - M <= 4: use original (wvSplitKQ)
  - M > 4:  dequantize FP8→bf16 + torch.matmul (no _scaled_mm dependency)
"""

from __future__ import annotations

import logging
import torch

logger = logging.getLogger("fdu_vllm.fp8_fallback")

_PATCHED = False


def activate_fp8_fallback() -> bool:
    """Patch ROCmFP8ScaledMMLinearKernel to avoid torch._scaled_mm for M > 4.

    Must be called BEFORE model loading (after vLLM import, before LLM creation).
    """
    global _PATCHED
    if _PATCHED:
        return True

    try:
        from vllm.model_executor.kernels.linear.scaled_mm.rocm import (
            ROCmFP8ScaledMMLinearKernel,
        )

        _original_apply = ROCmFP8ScaledMMLinearKernel.apply_scaled_mm

        def _patched_apply(
            self,
            *,
            A: torch.Tensor,
            B: torch.Tensor,
            out_dtype: torch.dtype,
            As: torch.Tensor,
            Bs: torch.Tensor,
            bias: torch.Tensor | None,
            output_shape: list,
        ) -> torch.Tensor:
            # M <= 4 (decode): use wvSplitKQ — native HIP kernel, fast path
            if A.shape[0] <= 4:
                return _original_apply(
                    self,
                    A=A,
                    B=B,
                    out_dtype=out_dtype,
                    As=As,
                    Bs=Bs,
                    bias=bias,
                    output_shape=output_shape,
                )

            # M > 4 (prefill): dequantize FP8→bf16 + torch.matmul
            # torch._scaled_mm is NOT available on this DCU
            # Dequant: result = (A_fp8 * As) @ (B_fp8 * Bs).t()
            A_dq = A.to(torch.float32) * As.to(torch.float32)
            B_dq = B.to(torch.float32) * Bs.to(torch.float32)
            # B is [K, N], we need matmul(A_dq[M,K], B_dq.t()[N,K]) → [M, N]
            result = torch.matmul(A_dq, B_dq)
            if bias is not None:
                result = result + bias
            return torch.narrow(result, 0, 0, output_shape[0]).view(
                *output_shape
            ).to(out_dtype)

        ROCmFP8ScaledMMLinearKernel.apply_scaled_mm = _patched_apply
        _PATCHED = True
        logger.info(
            "FDU fp8_fallback: ROCmFP8ScaledMMLinearKernel.apply_scaled_mm patched "
            "(M<=4: wvSplitKQ, M>4: dequant+matmul fallback)"
        )
        return True

    except ImportError:
        logger.warning(
            "FDU fp8_fallback: ROCm FP8 kernel not importable yet, "
            "will retry at hook time"
        )
        return False
    except Exception as e:
        logger.error("FDU fp8_fallback: patch failed: %s", e)
        return False


def is_patched() -> bool:
    return _PATCHED
