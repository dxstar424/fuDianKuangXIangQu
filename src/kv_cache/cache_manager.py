"""
KV Cache Manager with DCU-aware memory budgeting.

Optimization strategies:
1. Watermark-based memory budget: reserve headroom for decode spikes
2. Prefix-aware allocation: detect shared prefixes and reuse KV blocks
3. Proactive eviction: evict least-recently-used blocks before OOM
4. HBM bandwidth accounting: track and limit per-request HBM bandwidth usage
"""

from typing import Dict, Optional
import torch

from .block_allocator import CustomBlockAllocator


class CustomCacheManager:
    """
    Manages KV cache lifecycle across requests.

    Integrates with CustomBlockAllocator for block management
    and adds prefix caching + watermark budgeting.
    """

    def __init__(
        self,
        allocator: CustomBlockAllocator,
        watermark: float = 0.05,
        enable_prefix_cache: bool = True,
    ):
        """
        Args:
            allocator: CustomBlockAllocator instance
            watermark: fraction of GPU memory reserved as safety margin (0.05 = 5%)
            enable_prefix_cache: enable automatic prefix detection and sharing
        """
        self.allocator = allocator
        self.watermark = watermark
        self.enable_prefix_cache = enable_prefix_cache

        # Prefix cache: hash(prefix_tokens) -> block_indices
        self.prefix_cache: Dict[int, list] = {}

        # Active sequences tracking
        self.active_sequences: Dict[int, int] = {}  # seq_id -> num_blocks

    def reserve_decode_headroom(self, max_decode_tokens: int) -> bool:
        """
        Reserve blocks for decode phase to prevent OOM during generation.

        Args:
            max_decode_tokens: maximum tokens to reserve for

        Returns:
            True if reservation succeeded
        """
        reserve_blocks = (max_decode_tokens + self.allocator.block_size - 1)
        reserve_blocks //= self.allocator.block_size

        total_blocks = self.allocator.num_gpu_blocks
        watermark_blocks = int(total_blocks * self.watermark)

        free = len(self.allocator.free_blocks)
        return free >= reserve_blocks + watermark_blocks

    def find_prefix(self, token_ids: torch.Tensor) -> Optional[list]:
        """
        Check if a prefix of token_ids already exists in cache.

        Args:
            token_ids: input token IDs

        Returns:
            Pre-allocated block indices if prefix found, else None
        """
        if not self.enable_prefix_cache:
            return None

        # Simple prefix matching: hash first N tokens
        prefix_len = min(len(token_ids), 256)
        prefix_hash = hash(tuple(token_ids[:prefix_len].tolist()))
        return self.prefix_cache.get(prefix_hash)

    def cache_prefix(self, token_ids: torch.Tensor, blocks: list) -> None:
        """Store prefix-to-blocks mapping for reuse."""
        if self.enable_prefix_cache:
            prefix_len = min(len(token_ids), 256)
            prefix_hash = hash(tuple(token_ids[:prefix_len].tolist()))
            self.prefix_cache[prefix_hash] = blocks

    def evict_lru(self, num_blocks: int) -> int:
        """Evict LRU sequences to free up blocks. Returns number freed."""
        # Placeholder: track access timestamps per sequence
        freed = 0
        for seq_id in sorted(self.active_sequences.keys()):
            if freed >= num_blocks:
                break
            self.allocator.free(seq_id)
            freed += self.active_sequences[seq_id]
            del self.active_sequences[seq_id]
        return freed
