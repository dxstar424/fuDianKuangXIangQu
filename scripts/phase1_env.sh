#!/bin/bash
# Phase 1 专用环境：仅 1.1–1.7 低风险项，禁用 Phase 2+ 运行时钩子
# 由 launch.sh / rocm_env.sh source；评测机默认 FDU_PHASE=1

export FDU_PHASE="${FDU_PHASE:-1}"

# ── 1.1 显存利用率 ──
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"

# ── 1.2 分档 warmup ──
export DO_WARMUP="${DO_WARMUP:-1}"
export WARMUP_ROUNDS="${WARMUP_ROUNDS:-1}"
export WARMUP_TIER="${WARMUP_TIER:-all}"

# ── 1.3 prefix caching ──
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
export FDU_ENABLE_PREFIX_CACHE="${FDU_ENABLE_PREFIX_CACHE:-1}"

# ── 1.6 KV 量化默认关（保精度系数）──
export FDU_ENABLE_KV_QUANT="${FDU_ENABLE_KV_QUANT:-0}"

# ── Phase 1：关闭 Phase 2+ 钩子（GQA / KV defrag / HIP Graph / 自定义 attention）──
if [[ "${FDU_PHASE}" == "1" ]]; then
    export FDU_ENABLE="${FDU_ENABLE:-1}"
    export FDU_KV_CACHE_STRATEGY="${FDU_KV_CACHE_STRATEGY:-none}"
    export FDU_ATTENTION_BACKEND="${FDU_ATTENTION_BACKEND:-vllm_default}"
    export FDU_ENABLE_GQA_OPT="${FDU_ENABLE_GQA_OPT:-0}"
    export FDU_ENABLE_HIP_GRAPH="${FDU_ENABLE_HIP_GRAPH:-0}"
fi
