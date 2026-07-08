"""
Configuration loader for config.yaml
=====================================
从 YAML 加载所有可调参数，支持环境变量覆盖（ENV > YAML > default）。
submission 要求所有参数在 config.yaml 中声明，此模块负责读取+校验。
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict


# ---- 默认值（config.yaml 缺失时的 fallback）----

DEFAULTS: Dict[str, Any] = {
    "kv_cache": {
        "strategy": "defrag",
        "block_size": 16,
        "gpu_memory_utilization": 0.92,
        "enable_kv_quant": True,
        "kv_quant_dtype": "fp8",
        "defrag_threshold": 0.7,
    },
    "attention": {
        "backend": "dcu_optimized",
        "use_paged_attention": True,
        "num_kv_heads_override": None,
        "sliding_window": None,
    },
    "scheduler": {
        "policy": "length_aware",
        "max_num_seqs": 256,
        "max_num_batched_tokens": 8192,
        "prefill_batch_size": 4,
    },
    "executor": {
        "reduce_python_overhead": True,
        "use_hip_graph": True,
        "hip_graph_max_bs": 16,
        "enable_prefix_caching": True,
        "num_scheduler_steps": 8,
    },
    "precision": {
        "enforce_accuracy_check": True,
        "accuracy_tolerance": 0.01,
    },
}


# ---- 环境变量映射: ENV_VAR → (yaml.section, yaml.key, type) ----

_ENV_MAP = [
    ("FDU_KV_CACHE_STRATEGY",    "kv_cache",   "strategy",                str),
    ("FDU_ATTENTION_BACKEND",    "attention",  "backend",                 str),
    ("FDU_ENABLE_KV_QUANT",      "kv_cache",   "enable_kv_quant",         lambda v: v == "1"),
    ("FDU_SCHEDULER_POLICY",     "scheduler",  "policy",                  str),
    ("FDU_OPTIMIZE",             "executor",   "reduce_python_overhead",  lambda v: v == "1"),
]


# ---- 加载 ----

_config_cache: Dict[str, Any] | None = None


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """
    加载 config.yaml，合并 默认值 → YAML → 环境变量。

    Args:
        config_path: config.yaml 路径，默认 ../config.yaml

    Returns:
        合并后的配置字典
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"

    config = _deep_copy(DEFAULTS)

    # 1. 加载 YAML
    if os.path.exists(config_path):
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f)
        if yaml_data:
            _deep_merge(config, yaml_data)

    # 2. 环境变量覆盖
    for env_var, section, key, caster in _ENV_MAP:
        val = os.environ.get(env_var)
        if val is not None:
            try:
                config.setdefault(section, {})[key] = caster(val)
            except (ValueError, TypeError):
                pass  # 保持 YAML/default 值

    _config_cache = config
    _validate(config)
    return config


def get(section: str, key: str, default=None):
    """快捷取值: config.get('kv_cache', 'strategy')"""
    cfg = load_config()
    return cfg.get(section, {}).get(key, default)


# ---- 校验 ----

def _validate(config: Dict[str, Any]) -> None:
    """校验关键参数在合法范围内。"""

    # KV Cache strategy
    strat = config.get("kv_cache", {}).get("strategy")
    if strat not in ("defrag", "prealloc", "dynamic", None):
        raise ValueError(f"Invalid kv_cache.strategy: {strat}")

    # GPU memory utilization
    gmu = config.get("kv_cache", {}).get("gpu_memory_utilization", 0.92)
    if not 0.5 <= gmu <= 0.99:
        raise ValueError(f"gpu_memory_utilization out of range: {gmu}")

    # Scheduler policy
    policy = config.get("scheduler", {}).get("policy")
    if policy not in ("length_aware", "fcfs", "priority", None):
        raise ValueError(f"Invalid scheduler.policy: {policy}")

    # Attention backend
    backend = config.get("attention", {}).get("backend")
    if backend not in ("dcu_optimized", "flash_attn", "flashinfer", None):
        raise ValueError(f"Invalid attention.backend: {backend}")

    # Accuracy tolerance
    tol = config.get("precision", {}).get("accuracy_tolerance", 0.01)
    if not 0.0 <= tol <= 1.0:
        raise ValueError(f"accuracy_tolerance out of range: {tol}")


# ---- Helper ----

def _deep_copy(d: Dict[str, Any]) -> Dict[str, Any]:
    """浅层递归拷贝（仅处理嵌套 dict）。"""
    import copy
    return copy.deepcopy(d)


def _deep_merge(base: Dict, override: Dict) -> None:
    """Recursively merge override into base, modifying base in place."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
