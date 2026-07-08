"""在 import vllm 之前设置性能相关环境变量（低风险提分）。"""

from __future__ import annotations

import os


def configure_before_vllm_import() -> None:
    """赛题允许：启动参数 / 环境变量；不修改 locked CLI 参数。"""
    # 降低 Python 侧日志开销
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

    # ROCm / DCU 常用吞吐优化（不影响精度）
    os.environ.setdefault("HIP_FORCE_DEV_KERNARG", "1")
    os.environ.setdefault("GPU_MAX_HW_QUEUES", "2")

    # 避免编译线程过多抢占（单请求并发=1）
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
