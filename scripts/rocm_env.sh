#!/bin/bash
# Phase 1: DCU/ROCm 运行时环境（低风险提分优先）
_ROCM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=phase1_env.sh
source "$_ROCM_DIR/phase1_env.sh"

# ── 1.5 ROCm/DCU 带宽与队列（不影响精度）──
export HIP_PLATFORM="${HIP_PLATFORM:-amd}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-1}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export HIP_FORCE_DEV_KERNARG="${HIP_FORCE_DEV_KERNARG:-1}"

# Phase 2+ 默认：只开 GQA（已接线）；defrag/FP8/Graph 保持关直至单独门禁
if [[ "${FDU_PHASE}" != "1" ]]; then
    export FDU_ENABLE="${FDU_ENABLE:-1}"
    export FDU_KV_CACHE_STRATEGY="${FDU_KV_CACHE_STRATEGY:-none}"
    export FDU_ATTENTION_BACKEND="${FDU_ATTENTION_BACKEND:-vllm_default}"
    export FDU_ENABLE_KV_QUANT="${FDU_ENABLE_KV_QUANT:-0}"
    export FDU_ENABLE_PREFIX_CACHE="${FDU_ENABLE_PREFIX_CACHE:-1}"
    export FDU_ENABLE_HIP_GRAPH="${FDU_ENABLE_HIP_GRAPH:-0}"
    export FDU_ENABLE_GQA_OPT="${FDU_ENABLE_GQA_OPT:-1}"
fi
