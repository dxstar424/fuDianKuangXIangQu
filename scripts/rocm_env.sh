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

# ── CSDN 文章优化：DCU 特定 tuning ──
# BLAS: 小 batch decode (M=1 GEMV) 走 rocBLAS 常比 hipBLASLt 快
export TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT:-0}"
# aiter: 部分 gfx9 架构开了反慢，先关
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-0}"
# attention 走 Triton（DCU 默认最优路径）
export VLLM_USE_TRITON_FLASH_ATTN="${VLLM_USE_TRITON_FLASH_ATTN:-1}"
# safetensors 快速搬运到显存
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"

# ── 缓存复用：torch.compile / Triton / MIOPEN ──
CACHE_BASE="${FDU_CACHE_ROOT:-/workspace}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_BASE/vllm_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_BASE/triton_cache}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$CACHE_BASE/miopen_cache}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-$CACHE_BASE/miopen_cache}"

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
