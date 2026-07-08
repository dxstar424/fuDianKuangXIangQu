#!/bin/bash
# ============================================================
# FDU SCCSCC26 - vLLM 推理服务启动脚本（优化版）
# 平台调用: bash launch.sh --model /data/Qwen3.5-27B --port 8000 [--tensor-parallel-size N]
# ============================================================
set -e

MODEL_PATH="/data/Qwen3.5-27B"
PORT=8000
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"

# --- FDU 优化环境变量（供 vLLM 源码 patch 读取）---
export FDU_ENABLE_KV_QUANT="${FDU_ENABLE_KV_QUANT:-1}"
export FDU_ENABLE_HIP_GRAPH="${FDU_ENABLE_HIP_GRAPH:-1}"
export FDU_SCHEDULER_POLICY="${FDU_SCHEDULER_POLICY:-length_aware}"

# --- 解析 CLI 参数 ---
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)              MODEL_PATH="$2";           shift 2 ;;
        --port)               PORT="$2";                 shift 2 ;;
        --tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        *)                    ARGS+=("$1");              shift ;;
    esac
done

echo "=== FDU SCCSCC26 vLLM Server ==="
echo "Model:         ${MODEL_PATH}"
echo "Port:          ${PORT}"
echo "TP:            ${TENSOR_PARALLEL_SIZE}"
echo "Prefix Cache:  enabled"
echo "HIP Graph:     ${FDU_ENABLE_HIP_GRAPH}"
echo "KV Quant:      ${FDU_ENABLE_KV_QUANT}"
echo "Scheduler:     ${FDU_SCHEDULER_POLICY}"
echo "Max Seqs:      ${MAX_NUM_SEQS}"
echo "Max Batched:   ${MAX_NUM_BATCHED_TOKENS}"
echo "GPU Mem Util:  ${GPU_MEMORY_UTILIZATION}"
echo "================================="

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --trust-remote-code \
    --served-model-name Qwen3.5-27B \
    --enable-prefix-caching \
    --compilation-config '{"cudagraph_mode": 2}' \
    "${ARGS[@]}"
