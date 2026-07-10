"""Optional vLLM worker-level patches (prefix caching flags only)."""

from __future__ import annotations

import logging

logger = logging.getLogger("fdu_vllm.worker")


def patch_worker_if_available(cfg) -> None:
    """Does not modify batch scheduler; logs intended vLLM CLI flags."""
    flags = []
    if cfg.enable_prefix_cache:
        flags.append("--enable-prefix-caching")
    logger.info(
        "vLLM worker: use launch.sh flags %s (scheduler params locked by rules)",
        " ".join(flags) if flags else "(none)",
    )
