"""
vLLM Integration Plugin for FDU SCCSCC26
========================================
单入口接入所有优化模块到 vLLM 运行时。

触发: export FDU_OPTIMIZE=1 && bash launch.sh
或:    from src.plugin import apply; apply()
"""

import os
import logging
from typing import Optional

logger = logging.getLogger("fdu.plugin")


# ---- 开关 / 环境检测 ----

def is_enabled() -> bool:
    return os.environ.get("FDU_OPTIMIZE", "0") == "1"


def is_rocm() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        return hasattr(torch.version, "hip") and torch.version.hip is not None
    except ImportError:
        return False


def _get_vllm_version() -> Optional[str]:
    try:
        import vllm
        return getattr(vllm, "__version__", "0.18.1")
    except ImportError:
        return None


# ============================================================
# Attention Backend 注册（多路径 try/except）
# ============================================================

def _register_attention_backend():
    """向 vLLM 注册 'dcu_optimized' backend。尝试多种 vLLM 0.18.x 路径。"""

    # 路径 1: vllm.attention.selector.get_attn_backend（0.18.x 常见）
    try:
        from vllm.attention import selector
        if not hasattr(selector, "_fdu_patched"):
            _orig = selector.get_attn_backend

            def _patched_get_attn_backend(*args, **kwargs):
                env_backend = os.environ.get("VLLM_ATTENTION_BACKEND", "")
                if env_backend == "dcu_optimized":
                    from src.attention.dcu_attention import DCUAttentionBackend
                    logger.info("[FDU] DCU attention backend via vllm.attention.selector")
                    return DCUAttentionBackend
                return _orig(*args, **kwargs)

            selector.get_attn_backend = _patched_get_attn_backend
            selector._fdu_patched = True
            logger.info("[FDU] Attention selector patched (vllm.attention.selector)")
            return
    except (ImportError, AttributeError):
        pass

    # 路径 2: vllm.attention.backends.registry（备选）
    try:
        from vllm.attention.backends import registry
        if not hasattr(registry, "_fdu_patched"):
            _orig_get = registry.get_backend_class

            def _patched_get_backend(name, *args, **kwargs):
                if name == "dcu_optimized":
                    from src.attention.dcu_attention import DCUAttentionBackend
                    logger.info("[FDU] DCU attention backend via registry")
                    return DCUAttentionBackend
                return _orig_get(name, *args, **kwargs)

            registry.get_backend_class = _patched_get_backend
            registry._fdu_patched = True
            logger.info("[FDU] Attention backend registered (registry)")
            return
    except (ImportError, AttributeError):
        pass

    logger.warning("[FDU] Could not register attention backend — all paths failed.")


# ============================================================
# 调度器替换
# ============================================================

def _replace_scheduler():
    """Monkey-patch vLLM Scheduler。尝试 _schedule / schedule 两种方法名。"""
    try:
        from vllm.core import scheduler as sched_module

        for method_name in ("_schedule", "schedule"):
            if hasattr(sched_module.Scheduler, method_name):
                _orig = getattr(sched_module.Scheduler, method_name)
                _patched_name = f"_fdu_orig_{method_name}"

                if hasattr(sched_module.Scheduler, _patched_name):
                    logger.info("[FDU] Scheduler already patched")
                    return

                def _make_patched(orig_fn, mname):
                    def _patched(self):
                        if os.environ.get("FDU_SCHEDULER_POLICY") == "length_aware":
                            waiting = getattr(self, "waiting", [])
                            if waiting:
                                waiting.sort(
                                    key=lambda r: len(getattr(r, "prompt_token_ids", []))
                                )
                        return orig_fn(self)
                    return _patched

                patched = _make_patched(_orig, method_name)
                setattr(sched_module.Scheduler, method_name, patched)
                setattr(sched_module.Scheduler, _patched_name, _orig)
                logger.info(f"[FDU] Scheduler.{method_name}() patched")
                return

        logger.warning("[FDU] Scheduler patch failed — no known method found")

    except ImportError:
        logger.warning("[FDU] Cannot patch scheduler (vLLM not imported yet)")


# ============================================================
# Block 分配器
# ============================================================

def _replace_block_allocator():
    """注入分级块分配 + defrag 检查。"""
    try:
        from vllm.core import block_manager as bm_module
        cls = bm_module.BlockSpaceManager

        if hasattr(cls, "_fdu_orig_allocate"):
            logger.info("[FDU] Block allocator already patched")
            return

        _orig = cls.allocate
        cls._fdu_orig_allocate = _orig

        def _patched_allocate(self, seq_group):
            result = _orig(self, seq_group)
            strategy = os.environ.get("FDU_KV_CACHE_STRATEGY", "")
            if strategy == "defrag":
                try:
                    total = self.num_total_gpu_blocks
                    free = len(self.gpu_allocator.free_blocks)
                    if total > 0 and free / total > 0.7:
                        logger.debug("[FDU] Defrag threshold reached: %d/%d", free, total)
                except Exception:
                    pass
            return result

        cls.allocate = _patched_allocate
        logger.info("[FDU] Block allocator patched")

    except ImportError:
        logger.warning("[FDU] Cannot patch block allocator")


# ============================================================
# KV 量化注入
# ============================================================

def _inject_kv_quantization():
    if os.environ.get("FDU_ENABLE_KV_QUANT", "0") != "1":
        logger.info("[FDU] KV Quant disabled")
        return
    try:
        from src.quantization.kv_quant import KVCacheQuantizer
        import src.quantization.kv_quant as kv_mod
        kv_mod._global_quantizer = KVCacheQuantizer(dtype="fp8")
        logger.info("[FDU] KV Quant enabled (FP8 E4M3, online, non-persistent)")
    except ImportError:
        logger.warning("[FDU] Cannot inject KV quantization")


# ============================================================
# 单入口
# ============================================================

def apply() -> bool:
    """应用全部 FDU 优化。在 vLLM import 前调用。"""
    if not is_enabled():
        logger.info("[FDU] Plugin disabled (FDU_OPTIMIZE!=1)")
        return False

    ver = _get_vllm_version() or "unknown"
    logger.info("[FDU] ========================================")
    logger.info("[FDU] Applying FDU SCCSCC26 optimizations")
    logger.info("[FDU] vLLM: %s  |  ROCm: %s", ver, is_rocm())
    logger.info("[FDU] ========================================")

    _register_attention_backend()
    _replace_scheduler()
    _replace_block_allocator()
    _inject_kv_quantization()

    logger.info("[FDU] Exec path optimizer: call ExecPathOptimizer.warmup() after model load")
    logger.info("[FDU] All plugins applied")
    return True
