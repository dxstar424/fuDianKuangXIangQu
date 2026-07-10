"""Wrap a stock vLLM AttentionBackend so GQA avoids KV materialization on torch paths.

PagedAttention / Rocm kernels already handle GQA natively; this wrapper:
  1. Marks the selected backend as FDU-GQA for logging / report evidence
  2. Overrides encoder / torch SDPA fallbacks to use gqa_scaled_dot_product_attention
  3. Never touches batch scheduler
"""

from __future__ import annotations

import logging
from typing import Any, Type

logger = logging.getLogger("fdu_vllm.gqa_wrap")


def wrap_attn_backend(backend_cls: Type) -> Type:
    """Return a subclass of *backend_cls* with GQA-safe Impl."""
    if getattr(backend_cls, "_fdu_gqa_wrapped", False):
        return backend_cls

    try:
        stock_impl = backend_cls.get_impl_cls()
    except Exception as e:
        logger.warning("Cannot wrap backend %s: %s", backend_cls, e)
        return backend_cls

    class FduGqaImpl(stock_impl):  # type: ignore[valid-type,misc]
        """Stock Impl + GQA einsum on torch encoder path when Hq != Hkv."""

        def _forward_encoder_attention(self, query, key, value, output, attn_metadata, layer):
            # Prefer GQA path when shapes differ (avoid repeat_interleave).
            try:
                if (
                    hasattr(self, "num_heads")
                    and hasattr(self, "num_kv_heads")
                    and self.num_heads != self.num_kv_heads
                    and query is not None
                    and key is not None
                    and query.shape[-2] != key.shape[-2]
                ):
                    from fdu_vllm.gqa_decode import gqa_scaled_dot_product_attention

                    out = gqa_scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        self.num_heads,
                        self.num_kv_heads,
                        float(self.scale),
                    )
                    if output is not None:
                        output.copy_(out)
                        return output
                    return out
            except Exception as exc:
                logger.debug("GQA encoder path fallback: %s", exc)

            return super()._forward_encoder_attention(
                query, key, value, output, attn_metadata, layer
            )

    class FduGqaBackend(backend_cls):  # type: ignore[valid-type,misc]
        _fdu_gqa_wrapped = True

        @staticmethod
        def get_name() -> str:
            try:
                base = backend_cls.get_name()
            except Exception:
                base = getattr(backend_cls, "__name__", "ATTN")
            return f"FDU_GQA_{base}"

        @staticmethod
        def get_impl_cls() -> Any:
            return FduGqaImpl

    FduGqaBackend.__name__ = f"FduGqa_{getattr(backend_cls, '__name__', 'Backend')}"
    FduGqaImpl.__name__ = f"FduGqa_{getattr(stock_impl, '__name__', 'Impl')}"
    logger.info("Wrapped attention backend → %s", FduGqaBackend.get_name())
    return FduGqaBackend


def patch_attn_selector() -> bool:
    """Monkey-patch vllm.v1.attention.selector.get_attn_backend (idempotent)."""
    import os

    if os.environ.get("FDU_ENABLE_GQA_OPT", "0") not in ("1", "true", "True"):
        logger.info("GQA selector patch skipped (FDU_ENABLE_GQA_OPT!=1)")
        return False

    try:
        from vllm.v1.attention import selector
    except ImportError as e:
        logger.warning("vLLM attention selector unavailable: %s", e)
        return False

    if getattr(selector, "_fdu_gqa_patched", False):
        return True

    _orig = selector.get_attn_backend

    def _patched(*args, **kwargs):
        cls = _orig(*args, **kwargs)
        return wrap_attn_backend(cls)

    selector.get_attn_backend = _patched  # type: ignore[assignment]
    selector._fdu_gqa_patched = True
    # Clear lru cache so subsequent calls see the wrap
    cached = getattr(selector, "_cached_get_attn_backend", None)
    if cached is not None and hasattr(cached, "cache_clear"):
        cached.cache_clear()
    logger.info("Patched vllm.v1.attention.selector.get_attn_backend for GQA")
    return True
