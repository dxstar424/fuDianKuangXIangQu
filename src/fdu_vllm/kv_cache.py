"""KV cache block manager hooks (no batch scheduler changes)."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("fdu_vllm.kv_cache")


def install_kv_hooks(
    enable_defrag: bool = True,
    enable_prefix: bool = True,
    defrag_threshold: float = 0.7,
):
    from kv_cache.block_allocator import CustomBlockAllocator
    from kv_cache.cache_manager import CustomCacheManager

    allocator = CustomBlockAllocator(
        num_gpu_blocks=1024,
        block_size=16,
        enable_defrag=enable_defrag,
        defrag_threshold=defrag_threshold,
        enable_tiered=True,
    )
    mgr = CustomCacheManager(
        allocator=allocator,
        watermark=0.05,
        enable_prefix_cache=enable_prefix,
    )
    logger.info(
        "KV hooks: defrag=%s prefix=%s threshold=%.2f",
        enable_defrag,
        enable_prefix,
        defrag_threshold,
    )
    return mgr


def wrap_kv_write(tensor, quant_hooks: Optional[object] = None):
    """Optional FP8 wrap on KV write path."""
    if quant_hooks is not None and getattr(quant_hooks, "enabled", False):
        return quant_hooks.quantize_pair(tensor)
    return tensor, None


def wrap_kv_read(tensor, scale, quant_hooks: Optional[object] = None):
    if quant_hooks is not None and scale is not None:
        return quant_hooks.dequantize(tensor, scale)
    return tensor
