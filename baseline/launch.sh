#!/bin/bash
# ============================================================
# BASELINE - 纯净 vLLM 0.18.1 启动脚本
# 不使用任何自定义优化模块，用于获取基线性能数据
# ============================================================
set -e

MODEL_PATH="${MODEL_PATH:-/data/Qwen3.5-27B}"
PORT="${PORT:-8000}"

echo "=== Baseline vLLM Server ==="
echo "Model:     ${MODEL_PATH}"
echo "Port:      ${PORT}"
echo "Framework: vLLM 0.18.1 (stock)"
echo "=============================="

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --served-model-name Qwen3.5-27B \
    "$@"
