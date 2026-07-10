"""在 import vllm 之前设置性能相关环境变量（Phase 1 低风险提分）。

与 scripts/rocm_env.sh 对齐：worker 子进程未必 source shell，故在 Python 侧再 setdefault。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def sanitize_import_paths() -> None:
    """Drop repo root from sys.path so vendored vllm/ does not shadow pip wheel."""

    def _is_repo_root(path: Path) -> bool:
        return (path / "launch.sh").is_file() and (path / "vllm").is_dir()

    cleaned: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            resolved = Path(entry).resolve()
        except OSError:
            cleaned.append(entry)
            continue
        if _is_repo_root(resolved):
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned


def configure_before_vllm_import() -> None:
    """赛题允许：启动参数 / 环境变量；不修改 locked CLI 参数。"""
    sanitize_import_paths()
    # ── 1.4 降低 Python 侧日志开销 ──
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

    # ── 1.5 ROCm / DCU（与 rocm_env.sh 一致，不影响精度）──
    os.environ.setdefault("HIP_PLATFORM", "amd")
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    os.environ.setdefault("HIP_FORCE_DEV_KERNARG", "1")
    os.environ.setdefault("GPU_MAX_HW_QUEUES", "2")
    os.environ.setdefault("HSA_ENABLE_SDMA", "1")
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

    # 避免编译/CPU 线程过多抢占（评测并发=1）
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

    # Phase 1 默认：关 KV 量化（保精度系数）
    os.environ.setdefault("FDU_ENABLE_KV_QUANT", "0")
    os.environ.setdefault("FDU_PHASE", "1")
