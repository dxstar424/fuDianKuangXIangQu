"""HIP Graph hooks for decode (opt-in via FDU_ENABLE_HIP_GRAPH=1)."""

from __future__ import annotations

import logging

logger = logging.getLogger("fdu_vllm.hip_graph")


def install_hip_graph_hooks(max_batch_size: int = 1):
    from executor.exec_path import ExecPathOptimizer

    opt = ExecPathOptimizer(
        max_batch_size=max_batch_size,
        enable_hip_graph=True,
        num_steps_per_schedule=1,  # concurrent=1: no batch step batching
    )
    if not opt.is_rocm():
        logger.warning("HIP Graph requested but not on ROCm; disabled")
        opt.enable_hip_graph = False
    else:
        logger.info("HIP Graph hooks ready (capture on first decode step)")
    return opt
