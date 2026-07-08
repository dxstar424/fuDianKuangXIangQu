"""Runtime hooks: KV, attention, FP8, HIP graph. No batch scheduler patches."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("fdu_vllm")

_ACTIVE = False
_ATTENTION: Optional[object] = None
_KV_QUANT: Optional[object] = None
_EXEC: Optional[object] = None
_CACHE_MGR: Optional[object] = None


def is_active() -> bool:
    return _ACTIVE


def _ensure_src_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def activate() -> None:
    global _ACTIVE, _ATTENTION, _KV_QUANT, _EXEC, _CACHE_MGR

    from fdu_vllm.config import get_config

    cfg = get_config()
    if not cfg.enable:
        logger.info("FDU optimizations disabled (FDU_ENABLE=0)")
        return

    _ensure_src_path()
    _ACTIVE = True
    logger.info(
        "FDU plugin active: kv=%s attn=%s kv_quant=%s prefix=%s hip_graph=%s gqa=%s",
        cfg.kv_strategy,
        cfg.attention_backend,
        cfg.enable_kv_quant,
        cfg.enable_prefix_cache,
        cfg.enable_hip_graph,
        cfg.enable_gqa_opt,
    )

    if cfg.enable_kv_quant:
        from fdu_vllm.kv_fp8 import KVQuantHooks

        _KV_QUANT = KVQuantHooks(enabled=True, dtype=cfg.kv_quant_dtype)
        _KV_QUANT.install()

    if cfg.kv_strategy in ("defrag", "prealloc", "dynamic"):
        from fdu_vllm.kv_cache import install_kv_hooks

        _CACHE_MGR = install_kv_hooks(
            enable_defrag=cfg.kv_strategy == "defrag",
            enable_prefix=cfg.enable_prefix_cache,
            defrag_threshold=cfg.defrag_threshold,
        )

    if cfg.attention_backend == "dcu_optimized":
        from fdu_vllm.attention import install_attention_hooks

        _ATTENTION = install_attention_hooks(
            enable_gqa=cfg.enable_gqa_opt,
            use_fp8=cfg.enable_kv_quant,
        )

    if cfg.enable_hip_graph:
        from fdu_vllm.hip_graph import install_hip_graph_hooks

        _EXEC = install_hip_graph_hooks(max_batch_size=1)

    _try_patch_vllm_worker(cfg)


def _try_patch_vllm_worker(cfg) -> None:
    """Best-effort vLLM worker hooks after import."""
    try:
        import vllm  # noqa: F401
    except ImportError:
        logger.debug("vLLM not installed yet")
        return

    try:
        from fdu_vllm.vllm_worker import patch_worker_if_available

        patch_worker_if_available(cfg)
    except Exception as e:
        logger.warning("Worker patch skipped: %s", e)
