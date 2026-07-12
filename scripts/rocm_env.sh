#!/bin/bash
# v0.9.3: bf16 stock + AITER HIP FlashAttention (no weight quantization)
_ROCM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── DCU 设备 ──
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.4.2}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-1}"
export HIP_FORCE_DEV_KERNARG="${HIP_FORCE_DEV_KERNARG:-1}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"

# ── ★★★ 关键：AITER=1 但不设 UNIFIED=1 → fall through 到 FLASH_ATTN ★★★
export VLLM_ROCM_USE_AITER=1
# 不设 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION（默认 False）
# VLLM_ROCM_USE_AITER_MHA 默认 True → 选中 FLASH_ATTN 后端
# 该后端用 rocm_aiter_ops.flash_attn_varlen_func() — AITER 编译的 HIP CK kernel

export VLLM_ROCM_USE_SKINNY_GEMM="${VLLM_ROCM_USE_SKINNY_GEMM:-1}"
export VLLM_ROCM_USE_AITER_RMSNORM="${VLLM_ROCM_USE_AITER_RMSNORM:-1}"
export TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT:-0}"

# ── ROCm 自调优 ──
export ROCBLAS_LAYER="${ROCBLAS_LAYER:-4}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-1}"
export HIP_LAUNCH_BLOCKING="${HIP_LAUNCH_BLOCKING:-0}"

# ── 缓存 ──
CACHE_BASE="${FDU_CACHE_ROOT:-/workspace}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_BASE/vllm_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_BASE/triton_cache}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$CACHE_BASE/miopen_cache}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-$CACHE_BASE/miopen_cache}"

# v0.9.3: quant_force 为 no-op（重量化在此 DCU 上不可行）
# FDU_ENABLE=1 激活插件（含 quant_force no-op + fp8_fallback no-op）
export FDU_ENABLE=1

# Phase 2+ 钩子保持关
export FDU_ENABLE_KV_QUANT="${FDU_ENABLE_KV_QUANT:-0}"
export FDU_ENABLE_GQA_OPT="${FDU_ENABLE_GQA_OPT:-0}"
export FDU_ENABLE_HIP_GRAPH="${FDU_ENABLE_HIP_GRAPH:-0}"
