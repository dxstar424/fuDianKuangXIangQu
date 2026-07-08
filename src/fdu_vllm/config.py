"""Load config.yaml + environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:
    yaml = None


def _find_config() -> Path:
    for base in (Path("/workspace"), Path(__file__).resolve().parents[2]):
        p = base / "config.yaml"
        if p.exists():
            return p
    return Path(__file__).resolve().parents[2] / "config.yaml"


@dataclass
class FduConfig:
    enable: bool = True
    kv_strategy: str = "defrag"
    attention_backend: str = "dcu_optimized"
    enable_kv_quant: bool = True
    enable_prefix_cache: bool = True
    enable_hip_graph: bool = False
    enable_gqa_opt: bool = True
    gpu_memory_utilization: float = 0.92
    defrag_threshold: float = 0.7
    kv_quant_dtype: str = "fp8"
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_env_and_file(cls) -> "FduConfig":
        raw: dict[str, Any] = {}
        cfg_path = _find_config()
        if yaml and cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

        kv = raw.get("kv_cache", {})
        attn = raw.get("attention", {})
        exe = raw.get("executor", {})

        def _bool(name: str, default: bool) -> bool:
            v = os.environ.get(name)
            if v is None:
                return default
            return v not in ("0", "false", "False", "")

        return cls(
            enable=_bool("FDU_ENABLE", True),
            kv_strategy=os.environ.get("FDU_KV_CACHE_STRATEGY", kv.get("strategy", "defrag")),
            attention_backend=os.environ.get("FDU_ATTENTION_BACKEND", attn.get("backend", "dcu_optimized")),
            enable_kv_quant=_bool("FDU_ENABLE_KV_QUANT", kv.get("enable_kv_quant", True)),
            enable_prefix_cache=_bool(
                "FDU_ENABLE_PREFIX_CACHE", exe.get("enable_prefix_caching", True)
            ),
            enable_hip_graph=_bool("FDU_ENABLE_HIP_GRAPH", exe.get("use_cuda_graph", False)),
            enable_gqa_opt=_bool("FDU_ENABLE_GQA_OPT", True),
            gpu_memory_utilization=float(
                os.environ.get("GPU_MEMORY_UTILIZATION", kv.get("gpu_memory_utilization", 0.92))
            ),
            defrag_threshold=float(kv.get("defrag_threshold", 0.7)),
            kv_quant_dtype=kv.get("kv_quant_dtype", "fp8"),
            raw=raw,
        )


_CONFIG: Optional[FduConfig] = None


def get_config() -> FduConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = FduConfig.from_env_and_file()
    return _CONFIG
