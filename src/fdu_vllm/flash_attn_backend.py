"""
vLLM V1 AttentionBackend wrapper — routes prefill to HIP FlashAttention.

Strategy:
- Monkey-patch RocmAttentionImpl.forward() to intercept prefill tokens
- Prefill (key/value are non-None, meaning new tokens): dispatch to HIP kernel
- Decode (key/value are None, meaning KV cache read): pass through to original
- Any error: silent fallback to original path (no crash, no perf regression)

Activation:
  export FDU_ENABLE_FLASH_ATTN=1
  export FDU_PHASE=2  (or higher)
  Requires compiled .so at src/attention/hip_kernels/build/

Competition compliance:
- Online-only (no persistent quantized weights)
- Custom HIP kernel (allowed)
- Does not modify model weights/structure
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger("fdu_vllm.flash_attn_backend")

_PATCHED = False
_ORIG_FORWARD = None


def install_flash_attn_backend() -> bool:
    """
    Install HIP FlashAttention backend by patching RocmAttentionImpl.forward().

    Returns True if successfully installed, False if not available.
    """
    global _PATCHED, _ORIG_FORWARD

    if _PATCHED:
        return True

    from fdu_vllm.flash_attn_loader import is_available, flash_attn_prefill

    if not is_available():
        logger.info("FlashAttn kernel not available; skipping backend install")
        return False

    try:
        from vllm.v1.attention.backends.rocm_attn import RocmAttentionImpl
        from vllm.v1.attention.backend import AttentionType
    except ImportError as e:
        logger.warning("Cannot import RocmAttentionImpl: %s", e)
        return False

    _ORIG_FORWARD = RocmAttentionImpl.forward

    def _patched_forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ── Determine if this is pure prefill ──────────────────
        # Prefill condition: key and value are non-None (new tokens being
        # written to KV cache), AND we have multiple query tokens.
        # In vLLM V1 chunked prefill, prefill tokens reach us with key/value
        # provided; decode tokens have key=None, value=None.
        is_prefill = (
            key is not None
            and value is not None
            and attn_metadata is not None
            and getattr(self, "attn_type", None) == AttentionType.DECODER
        )

        if is_prefill:
            try:
                num_tokens = attn_metadata.num_actual_tokens
                max_qlen = attn_metadata.max_query_len

                # Only use HIP kernel for substantial prefill (≥128 tokens)
                if num_tokens >= 128 and output is not None:
                    scale = float(self.scale)

                    # Q: [num_tokens, n_heads, head_dim]
                    # K, V: [num_tokens, n_kv_heads, head_dim]
                    hip_out = flash_attn_prefill(
                        query[:num_tokens],
                        key[:num_tokens],
                        value[:num_tokens],
                        scale=scale,
                        output=output[:num_tokens],
                    )

                    # KV cache update — still needed (write K/V to page table)
                    # The original do_kv_cache_update is called separately by vLLM,
                    # so we only handle the attention computation here.

                    return output

            except Exception as exc:
                logger.debug(
                    "HIP FlashAttn failed (fallback to stock): %s", exc
                )

        # Fall through to original forward
        return _ORIG_FORWARD(
            self, layer, query, key, value, kv_cache,
            attn_metadata, output, output_scale, output_block_scale,
        )

    RocmAttentionImpl.forward = _patched_forward
    RocmAttentionImpl._fdu_flash_attn_patched = True
    _PATCHED = True

    logger.info(
        "FlashAttn backend installed: prefill (≥128 tokens) → HIP kernel, "
        "decode → stock RocmAttention"
    )
    return True
