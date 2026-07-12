#!/usr/bin/env bash
# ============================================================
# v0.7.0 — INT4 AWQ 权重量化 + AITER HIP FlashAttention
#
# 突破 60 分瓶颈的唯一路径：减少 decode 权重 HBM IO
#   bf16: 54GB ÷ 1.2TB/s = 45ms/token（瓶颈）
#   INT4: ~14GB ÷ 1.2TB/s = ~12ms/token（3.75x 理论加速）
#
# 使用 mattbucci/Qwen3.5-27B-AWQ（thinking-aware 校准，~18GB）
# AWQ Triton 内核在 ROCm 上 JIT 编译，避免 C++ kernel 兼容问题
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/scripts/rocm_env.sh"

_resolve_model_path() {
    for cand in \
        /root/Qwen3.5-27B-AWQ /data/Qwen3.5-27B-AWQ \
        "${SCNET_HOME:-/public/home/xdzs2026_c415}/Qwen3.5-27B-AWQ" \
        /root/Qwen3.5-27B /data/Qwen3.5-27B; do
        [[ -d "$cand" ]] && echo "$cand" && return
    done
    echo "${MODEL_PATH:-/data/Qwen3.5-27B-AWQ}"
}

MODEL_PATH="$(_resolve_model_path)"
PORT="${PORT:-8000}"

VLLM_ARGS=(
    --model "${MODEL_PATH}"
    --port "${PORT}"
    --tensor-parallel-size 1
    --max-model-len 32768
    --max-num-seqs 256
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.98}"
    --quantization awq
    --trust-remote-code
    --served-model-name Qwen3.5-27B
    --load-format runai_streamer
    --enable-prefix-caching
    --no-enable-log-requests
    --disable-log-stats
    --compilation-config '{"cudagraph_mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}'
)

echo "[launch] === v0.7.0: INT4 AWQ + HIP FlashAttention ==="
echo "[launch]   VLLM_ROCM_USE_AITER=1 (AITER HIP FA for 25% GQA layers)"
echo "[launch]   VLLM_USE_TRITON_AWQ=1 (Triton AWQ dequant kernel, ROCm safe)"
echo "[launch]   --quantization awq (INT4 weights, ~18GB vs 52GB bf16)"
echo "[launch]   GPU 0.98 (35GB freed by INT4 → more KV cache)"
echo "[launch]   model: ${MODEL_PATH}"
echo "[launch]   port:  ${PORT}"
echo "[launch] ====================================="

cd /tmp
exec python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" "$@"
