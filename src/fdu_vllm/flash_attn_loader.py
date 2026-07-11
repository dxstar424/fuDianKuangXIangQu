"""
HIP FlashAttention kernel loader (ctypes).

Loads the pre-compiled dcu_flash_attn_prefill.so and wraps the
prefill kernel for use in vLLM attention backends.

Kernel capabilities:
- Single-Q-row-per-block design with LDS double buffering
- Multi-head GQA support (n_heads / n_kv_heads groups)
- Causal masking (autoregressive prefill)
- Online softmax (FlashAttention algorithm)

Fallback: PyTorch scaled_dot_product_attention if .so not available.
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger("fdu_vllm.flash_attn")

_HIP_KERNEL: Optional[ctypes.CDLL] = None
_KERNEL_DIR = Path(__file__).resolve().parents[1] / "attention" / "hip_kernels"


def _find_so() -> Optional[Path]:
    """Find the compiled .so file."""
    candidates = [
        _KERNEL_DIR / "build" / "dcu_flash_attn_prefill.so",
        _KERNEL_DIR / "dcu_flash_attn_prefill.so",
        Path("/public/home/xdzs2026_c415/2025pra-fdu-fudiankuangxiangqu")
        / "src/attention/hip_kernels/build/dcu_flash_attn_prefill.so",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_kernel() -> Optional[ctypes.CDLL]:
    global _HIP_KERNEL
    if _HIP_KERNEL is not None:
        return _HIP_KERNEL

    so_path = _find_so()
    if so_path is None:
        logger.info("FlashAttn .so not found; using PyTorch fallback")
        return None

    try:
        lib = ctypes.CDLL(str(so_path))
        lib.dcu_flash_attn_prefill.argtypes = [
            ctypes.c_void_p,  # Q
            ctypes.c_void_p,  # K
            ctypes.c_void_p,  # V
            ctypes.c_void_p,  # O
            ctypes.c_float,   # scale
            ctypes.c_int,     # seq_len
            ctypes.c_int,     # n_heads
            ctypes.c_int,     # n_kv_heads
            ctypes.c_int,     # head_dim
            ctypes.c_void_p,  # hipStream_t
        ]
        lib.dcu_flash_attn_prefill.restype = ctypes.c_int
        _HIP_KERNEL = lib
        logger.info("Loaded FlashAttn kernel: %s", so_path)
        return lib
    except OSError as e:
        logger.warning("Failed to load FlashAttn .so: %s", e)
        return None


def is_available() -> bool:
    """Check if the HIP kernel .so is loadable."""
    return _load_kernel() is not None


def flash_attn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    FlashAttention prefill (contiguous single-sequence, causal).

    Args:
        q: [seq_len, n_heads, head_dim] bf16
        k: [seq_len, n_kv_heads, head_dim] bf16
        v: [seq_len, n_kv_heads, head_dim] bf16
        scale: attention scale (1/sqrt(head_dim))
        output: optional pre-allocated output [seq_len, n_heads, head_dim]

    Returns:
        output tensor [seq_len, n_heads, head_dim] bf16
    """
    lib = _load_kernel()
    if lib is None:
        return _fallback_sdpa(q, k, v, scale, output)

    seq_len, n_heads, head_dim = q.shape
    n_kv_heads = k.shape[1]

    if output is None:
        output = torch.empty_like(q)

    # Ensure bf16, contiguous, on correct device
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    output = output.contiguous()

    stream = torch.cuda.current_stream().cuda_stream

    ret = lib.dcu_flash_attn_prefill(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), output.data_ptr(),
        ctypes.c_float(scale),
        ctypes.c_int(seq_len), ctypes.c_int(n_heads),
        ctypes.c_int(n_kv_heads), ctypes.c_int(head_dim),
        ctypes.c_void_p(stream),
    )

    if ret != 0:
        logger.debug("HIP kernel returned %d; falling back to PyTorch", ret)
        return _fallback_sdpa(q, k, v, scale, output)

    return output


def _fallback_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """PyTorch SDPA fallback with GQA support."""
    from fdu_vllm.gqa_decode import gqa_scaled_dot_product_attention

    n_heads = q.shape[1]
    n_kv_heads = k.shape[1]
    out = gqa_scaled_dot_product_attention(q, k, v, n_heads, n_kv_heads, scale)
    if output is not None:
        output.copy_(out)
        return output
    return out
