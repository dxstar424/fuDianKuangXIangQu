# FDU GQA attention wrap — copied into vllm_cscc by apply_vllm_patches.sh
#
# Runtime activation still goes through fdu_vllm.activate() → patch_attn_selector().
# This module is the overlay-side import target for wheel builds.

"""Re-export GQA wrap for patched vLLM trees."""

try:
    from fdu_vllm.gqa_backend_wrap import patch_attn_selector, wrap_attn_backend
except ImportError:
    # When only overlay is present before fdu_vllm sync
    patch_attn_selector = None  # type: ignore
    wrap_attn_backend = None  # type: ignore

__all__ = ["patch_attn_selector", "wrap_attn_backend"]
