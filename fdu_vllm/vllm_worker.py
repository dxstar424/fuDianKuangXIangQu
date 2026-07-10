"""vLLM runtime patches: GQA selector wrap + optional HIP Graph (no scheduler)."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("fdu_vllm.worker")


def patch_worker_if_available(cfg) -> None:
    """Install compliant runtime hooks after vLLM is importable."""
    flags = []
    if cfg.enable_prefix_cache:
        flags.append("--enable-prefix-caching")
    logger.info(
        "vLLM worker: launch flags %s (scheduler params locked)",
        " ".join(flags) if flags else "(none)",
    )

    # Phase 2+: GQA — patch attention selector (real wiring)
    if cfg.phase >= 2 and cfg.enable_gqa_opt:
        try:
            from fdu_vllm.gqa_backend_wrap import patch_attn_selector

            ok = patch_attn_selector()
            logger.info("GQA selector patch: %s", "OK" if ok else "skipped")
        except Exception as e:
            logger.warning("GQA selector patch failed: %s", e)

    # Phase 2+: HIP Graph — opt-in only
    if cfg.phase >= 2 and cfg.enable_hip_graph:
        if os.environ.get("ENFORCE_EAGER", "0") in ("1", "true", "True"):
            logger.warning(
                "FDU_ENABLE_HIP_GRAPH=1 but ENFORCE_EAGER=1 — graph will not run; "
                "set ENFORCE_EAGER=0 for S4"
            )
        try:
            from fdu_vllm.hip_graph import install_hip_graph_hooks, patch_model_runner_graph

            opt = install_hip_graph_hooks(max_batch_size=1)
            patch_model_runner_graph(opt)
            logger.info("HIP Graph model-runner patch installed")
        except Exception as e:
            logger.warning("HIP Graph patch failed: %s", e)
