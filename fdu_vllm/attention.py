"""Attention backend integration with HIP fallback."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("fdu_vllm.attention")

_BACKEND: Optional[object] = None


def install_attention_hooks(enable_gqa: bool = True, use_fp8: bool = False):
    global _BACKEND
    from attention.dcu_attention import DCUAttentionBackend

    _BACKEND = DCUAttentionBackend(
        num_heads=64,
        num_kv_heads=32,
        head_dim=128,
        use_fp8=use_fp8,
    )

    if _BACKEND.is_rocm():
        loaded = _BACKEND.load_hip_kernel(verbose=True)
        if not loaded:
            logger.info("HIP kernel unavailable; using GQA PyTorch path")
    else:
        logger.info("Non-ROCm env; PyTorch attention fallback")

    if enable_gqa:
        _patch_forward_with_gqa(_BACKEND)

    return _BACKEND


def _patch_forward_with_gqa(backend) -> None:
    from fdu_vllm.gqa_decode import gqa_scaled_dot_product_attention

    original_torch = backend._forward_torch

    def _forward_gqa(query, key, value):
        try:
            return gqa_scaled_dot_product_attention(
                query, key, value,
                backend.num_heads,
                backend.num_kv_heads,
                backend.scale,
            )
        except Exception:
            return original_torch(query, key, value)

    backend._forward_torch = _forward_gqa

    def _forward_safe(query, key, value, block_tables=None, context_lens=None, max_context_len=0, alibi_slopes=None):
        if backend.using_hip_kernel:
            try:
                return backend._forward_hip(query, key, value, block_tables, context_lens, max_context_len)
            except NotImplementedError:
                pass
        return backend._forward_torch(query, key, value)

    backend.forward = _forward_safe
    logger.info("GQA-optimized forward installed")
