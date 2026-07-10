"""
vLLM Integration Plugin for FDU SCCSCC26 (V1 architecture)
==========================================================
单入口接入所有优化模块到 vLLM V1 运行时。

触发: export FDU_OPTIMIZE=1 && python -m vllm.entrypoints.openai.api_server
"""

import os
import logging

logger = logging.getLogger("fdu.plugin")


def is_enabled() -> bool:
    return os.environ.get("FDU_OPTIMIZE", "0") == "1"


def is_rocm() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and hasattr(torch.version, "hip") and torch.version.hip is not None
    except ImportError:
        return False


# ============================================================
# Attention Backend 注册（V1 路径: vllm.v1.attention.selector）
# ============================================================

def _register_attention_backend():
    try:
        from vllm.v1.attention import selector
        if hasattr(selector, "_fdu_patched"):
            return
        _orig = selector.get_attn_backend

        def _patched(*args, **kwargs):
            env = os.environ.get("VLLM_ATTENTION_BACKEND", "")
            if env == "dcu_optimized":
                from vllm.fdu_optimize.attention.dcu_attention import DCUAttentionBackend
                logger.info("[FDU] DCU attention backend via vllm.v1.attention.selector")
                return DCUAttentionBackend
            return _orig(*args, **kwargs)

        selector.get_attn_backend = _patched
        selector._fdu_patched = True
        logger.info("[FDU] Attention selector patched")
    except Exception as e:
        logger.warning("[FDU] Attention patch skipped: %s", e)


# ============================================================
# 调度器替换（V1 路径: vllm.v1.core.sched.scheduler）
# ============================================================

def _replace_scheduler():
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
        if hasattr(Scheduler, "_fdu_patched"):
            return
        _orig = Scheduler.schedule

        def _patched(self):
            if os.environ.get("FDU_SCHEDULER_POLICY") == "length_aware":
                waiting = getattr(self, "waiting", [])
                if waiting:
                    waiting.sort(key=lambda r: len(getattr(r, "prompt_token_ids", [])))
            return _orig(self)

        Scheduler.schedule = _patched
        Scheduler._fdu_patched = True
        Scheduler._fdu_orig_schedule = _orig
        logger.info("[FDU] Scheduler.schedule() patched (V1)")
    except Exception as e:
        logger.warning("[FDU] Scheduler patch skipped: %s", e)


# ============================================================
# KV 量化注入（独立，非 monkey-patch）
# ============================================================

def _inject_kv_quantization():
    if os.environ.get("FDU_ENABLE_KV_QUANT", "0") != "1":
        logger.info("[FDU] KV Quant disabled")
        return
    try:
        from vllm.fdu_optimize.quantization.kv_quant import KVCacheQuantizer
        import vllm.fdu_optimize.quantization.kv_quant as kv_mod
        kv_mod._global_quantizer = KVCacheQuantizer(dtype="fp8")
        logger.info("[FDU] KV Quant enabled (FP8 E4M3, online, non-persistent)")
    except Exception as e:
        logger.warning("[FDU] KV Quant skipped: %s", e)


# ============================================================
# 单入口
# ============================================================

def apply() -> bool:
    if not is_enabled():
        logger.info("[FDU] Plugin disabled (FDU_OPTIMIZE!=1)")
        return False

    logger.info("[FDU] ========================================")
    logger.info("[FDU] Applying FDU SCCSCC26 optimizations (V1)")
    logger.info("[FDU] ROCm: %s", is_rocm())
    logger.info("[FDU] ========================================")

    _register_attention_backend()
    _replace_scheduler()
    _inject_kv_quantization()

    logger.info("[FDU] All plugins applied")
    return True
