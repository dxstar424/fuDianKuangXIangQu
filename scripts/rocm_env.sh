#!/bin/bash
# v0.4.0: 去掉 FP8/AITER + 激进 ROCm 系统优化
_ROCM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=phase1_env.sh
source "$_ROCM_DIR/phase1_env.sh"

# ── 1. DCU 设备与架构 ──
export HIP_PLATFORM="${HIP_PLATFORM:-amd}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-1}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export HIP_FORCE_DEV_KERNARG="${HIP_FORCE_DEV_KERNARG:-1}"

# gfx942 架构 override（DCU BW1000 / MI300X 级别）
HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.4.2}"
export HSA_OVERRIDE_GFX_VERSION

# ── 2. BLAS 后端选择（关键！）──
# decode (M=1 GEMV) 走 rocBLAS，prefill (大 batch GEMM) 走 hipBLASLt
# TORCH_BLAS_PREFER_HIPBLASLT=0 → rocBLAS 优先（decode 更快）
# 但对大 batch prefill，hipBLASLt 可能更好。这里走 rocBLAS 全局，稳妥。
export TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT:-0}"

# ── 3. vLLM DCU 内核选择 ──
# ★★★ 关 AITER（FP8 路径走不通，纯 bf16 不需要 aiter）★★★
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-0}"
# 开 AITER RMSNorm（非 FP8，纯 bf16 RMSNorm 加速）
export VLLM_ROCM_USE_AITER_RMSNORM="${VLLM_ROCM_USE_AITER_RMSNORM:-1}"
# ★ 确认 skinny_gemm 启用（decode GEMV 手写 HIP kernel）★
export VLLM_ROCM_USE_SKINNY_GEMM="${VLLM_ROCM_USE_SKINNY_GEMM:-1}"
# safetensors 快速搬运
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"

# ── 4. ROCm 微架构自调优 ──
# rocBLAS 自动调优（从预编译的 kernel 中选最优）
export ROCBLAS_LAYER="${ROCBLAS_LAYER:-4}"
# MIOpen 自动寻找最优卷积/注意力算法
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-1}"
# 不阻塞 kernel launch（异步执行）
export HIP_LAUNCH_BLOCKING="${HIP_LAUNCH_BLOCKING:-0}"

# ── 5. 缓存复用：torch.compile / Triton / MIOPEN ──
CACHE_BASE="${FDU_CACHE_ROOT:-/workspace}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_BASE/vllm_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_BASE/triton_cache}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$CACHE_BASE/miopen_cache}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-$CACHE_BASE/miopen_cache}"

# Phase 2+ 默认：只开 GQA；defrag/FP8/Graph 保持关
if [[ "${FDU_PHASE}" != "1" ]]; then
    export FDU_ENABLE="${FDU_ENABLE:-1}"
    export FDU_KV_CACHE_STRATEGY="${FDU_KV_CACHE_STRATEGY:-none}"
    export FDU_ATTENTION_BACKEND="${FDU_ATTENTION_BACKEND:-vllm_default}"
    export FDU_ENABLE_KV_QUANT="${FDU_ENABLE_KV_QUANT:-0}"
    export FDU_ENABLE_PREFIX_CACHE="${FDU_ENABLE_PREFIX_CACHE:-1}"
    export FDU_ENABLE_HIP_GRAPH="${FDU_ENABLE_HIP_GRAPH:-0}"
    export FDU_ENABLE_GQA_OPT="${FDU_ENABLE_GQA_OPT:-1}"
fi
