#!/bin/bash
# ============================================================
# FDU SCCSCC26 - vLLM 推理服务启动脚本
# 评测平台通过此脚本启动服务，参数可自行调整
# ============================================================
set -e

MODEL_PATH="${MODEL_PATH:-/data/Qwen3.5-27B}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"

# --- 注入自定义优化模块 ---
# 通过 PYTHONPATH 注入，使 vLLM 加载时自动挂载自定义插件
export PYTHONPATH="/workspace:${PYTHONPATH}"

# --- 自定义环境变量（详见 docs/env_vars.md）---
export FDU_KV_CACHE_STRATEGY="${FDU_KV_CACHE_STRATEGY:-defrag}"
export FDU_ATTENTION_BACKEND="${FDU_ATTENTION_BACKEND:-dcu_optimized}"
export FDU_ENABLE_KV_QUANT="${FDU_ENABLE_KV_QUANT:-1}"
export FDU_SCHEDULER_POLICY="${FDU_SCHEDULER_POLICY:-length_aware}"

# --- 启动 vLLM 服务 ---
exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --trust-remote-code \
    --served-model-name Qwen3.5-27B \
    "$@"
