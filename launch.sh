#!/bin/bash
# ============================================================
# FDU SCCSCC26 - vLLM 推理服务启动脚本（优化版）
# 平台调用格式（固定）:
#   bash launch.sh --model /data/Qwen3.5-27B --port 8000 [--tensor-parallel-size N] [其他]
# ============================================================
set -e

# --- 默认值 ---
MODEL_PATH="/data/Qwen3.5-27B"
PORT=8000
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"

# --- 自定义优化模块 ---
export PYTHONPATH="/workspace:${PYTHONPATH}"

# --- FDU 优化开关 ---
export FDU_OPTIMIZE="${FDU_OPTIMIZE:-1}"
export FDU_KV_CACHE_STRATEGY="${FDU_KV_CACHE_STRATEGY:-defrag}"
export FDU_ATTENTION_BACKEND="${FDU_ATTENTION_BACKEND:-dcu_optimized}"
export FDU_ENABLE_KV_QUANT="${FDU_ENABLE_KV_QUANT:-1}"
export FDU_SCHEDULER_POLICY="${FDU_SCHEDULER_POLICY:-length_aware}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-dcu_optimized}"

# --- 解析 CLI 参数（平台评测固定通过 --model --port 传参）---
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL_PATH="$2";   shift 2 ;;
        --port)    PORT="$2";         shift 2 ;;
        --tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        *)         ARGS+=("$1");      shift ;;
    esac
done

echo "=== FDU SCCSCC26 vLLM Server ==="
echo "Model:         ${MODEL_PATH}"
echo "Port:          ${PORT}"
echo "TP:            ${TENSOR_PARALLEL_SIZE}"
echo "Backend:       ${VLLM_ATTENTION_BACKEND}"
echo "KV Quant:      ${FDU_ENABLE_KV_QUANT}"
echo "Scheduler:     ${FDU_SCHEDULER_POLICY}"
echo "================================="

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --trust-remote-code \
    --served-model-name Qwen3.5-27B \
    "${ARGS[@]}"
