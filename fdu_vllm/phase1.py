"""Phase 1 profile: launch.sh CLI flags + ROCm env only (items 1.1–1.7)."""

from __future__ import annotations

import logging
import os
from typing import List

logger = logging.getLogger("fdu_vllm.phase1")

# Phase 1 必须项（与 deep_optimization_guide.md §1.1–1.7 对齐）
PHASE1_DEFAULTS = {
    "FDU_PHASE": "1",
    "GPU_MEMORY_UTILIZATION": "0.94",
    "DO_WARMUP": "1",
    "ENABLE_PREFIX_CACHING": "1",
    "FDU_ENABLE_PREFIX_CACHE": "1",
    "FDU_ENABLE_KV_QUANT": "0",
    "FDU_ENABLE_GQA_OPT": "0",
    "FDU_ENABLE_HIP_GRAPH": "0",
    "FDU_KV_CACHE_STRATEGY": "none",
    "FDU_ATTENTION_BACKEND": "vllm_default",
    "VLLM_LOGGING_LEVEL": "WARNING",
}


def is_phase1() -> bool:
    return os.environ.get("FDU_PHASE", "1") in ("1", "phase1")


def validate_phase1_env() -> List[str]:
    """Return warnings when Phase 1 env deviates from expected safe defaults."""
    if not is_phase1():
        return []

    warnings: List[str] = []
    checks = {
        "FDU_ENABLE_KV_QUANT": ("0", "KV 量化应默认关，保精度系数"),
        "FDU_ENABLE_GQA_OPT": ("0", "GQA 属 Phase 2，Phase 1 应关"),
        "FDU_ENABLE_HIP_GRAPH": ("0", "HIP Graph 属 Phase 2+，Phase 1 应关"),
        "FDU_KV_CACHE_STRATEGY": ("none", "KV defrag 属 Phase 2，Phase 1 应为 none"),
        "FDU_ATTENTION_BACKEND": ("vllm_default", "自定义 attention 属 Phase 2+"),
    }
    for key, (expected, reason) in checks.items():
        actual = os.environ.get(key)
        if actual is not None and actual != expected:
            warnings.append(f"{key}={actual} (期望 {expected}): {reason}")

    gpu = os.environ.get("GPU_MEMORY_UTILIZATION", "0.94")
    try:
        if float(gpu) < 0.90 or float(gpu) > 0.95:
            warnings.append(f"GPU_MEMORY_UTILIZATION={gpu} 超出建议区间 0.90–0.95")
    except ValueError:
        warnings.append(f"GPU_MEMORY_UTILIZATION={gpu} 非合法浮点数")

    return warnings


def log_phase1_summary() -> None:
    if not is_phase1():
        logger.info("FDU_PHASE=%s (Phase 2+ hooks enabled)", os.environ.get("FDU_PHASE", "?"))
        return

    logger.info(
        "Phase 1 active: gpu_util=%s prefix=%s kv_quant=%s warmup=%s",
        os.environ.get("GPU_MEMORY_UTILIZATION", "0.94"),
        os.environ.get("FDU_ENABLE_PREFIX_CACHE", "1"),
        os.environ.get("FDU_ENABLE_KV_QUANT", "0"),
        os.environ.get("DO_WARMUP", "1"),
    )
    for w in validate_phase1_env():
        logger.warning("Phase 1 config: %s", w)
