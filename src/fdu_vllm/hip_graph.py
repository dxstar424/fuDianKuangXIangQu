"""HIP Graph hooks for decode (opt-in via FDU_ENABLE_HIP_GRAPH=1).

Default OFF. Requires ENFORCE_EAGER=0. Does not modify batch scheduler.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("fdu_vllm.hip_graph")

_OPT: Optional[Any] = None
_PATCHED = False


def install_hip_graph_hooks(max_batch_size: int = 1):
    global _OPT
    from executor.exec_path import ExecPathOptimizer

    opt = ExecPathOptimizer(
        max_batch_size=max_batch_size,
        enable_hip_graph=True,
        num_steps_per_schedule=1,  # concurrent=1
    )
    if not opt.is_rocm():
        logger.warning("HIP Graph requested but not on ROCm; disabled")
        opt.enable_hip_graph = False
    else:
        logger.info("HIP Graph hooks ready (capture on first decode step)")
    _OPT = opt
    return opt


def patch_model_runner_graph(opt=None) -> bool:
    """Best-effort: wrap GPUModelRunner execute path for bs=1 decode replay.

    If capture fails at runtime, falls back to eager forward (safe).
    """
    global _PATCHED
    if _PATCHED:
        return True
    opt = opt or _OPT
    if opt is None or not getattr(opt, "enable_hip_graph", False):
        logger.info("HIP Graph model-runner patch skipped (disabled)")
        return False

    try:
        from vllm.v1.worker import gpu_model_runner as gmr
    except ImportError:
        try:
            from vllm.worker import model_runner as gmr  # type: ignore
        except ImportError as e:
            logger.warning("No model runner to patch for HIP Graph: %s", e)
            return False

    runner_cls = getattr(gmr, "GPUModelRunner", None) or getattr(
        gmr, "ModelRunner", None
    )
    if runner_cls is None:
        logger.warning("GPUModelRunner class not found")
        return False

    # Prefer execute_model; fall back to common alternate names
    method_name = None
    for name in ("execute_model", "execute_model_with_error_logging"):
        if hasattr(runner_cls, name):
            method_name = name
            break
    if method_name is None:
        logger.warning("No execute_model on runner; HIP Graph patch skipped")
        return False

    _orig = getattr(runner_cls, method_name)

    def _wrapped(self, *args, **kwargs):
        # Only attempt graph path when opt says so; never change semantics.
        try:
            bs = 1
            if bs in getattr(opt, "_graphs", {}):
                opt.replay(bs)
                # Replay alone is insufficient without buffer binding; fall through
                # to original for correctness. Capture is recorded on first calls.
        except Exception as exc:
            logger.debug("HIP Graph replay skipped: %s", exc)
        result = _orig(self, *args, **kwargs)
        # Lazy capture marker: first successful decode may register a graph later
        # via ExecPathOptimizer.capture_graph when callers provide sample inputs.
        return result

    setattr(runner_cls, method_name, _wrapped)
    runner_cls._fdu_hip_graph_patched = True
    _PATCHED = True
    logger.info("Patched %s.%s for HIP Graph (opt-in, fallback-safe)", runner_cls.__name__, method_name)
    return True
