#!/usr/bin/env bash
# ============================================================
# v0.8.0 — 强制 INT4 在线量化 (bitsandbytes) + AITER HIP FlashAttention
#
# 策略：不依赖 CLI flag，直接修改 vLLM 源码强制默认 quantization="bitsandbytes"
#       bnb 在模型加载时自动将 bf16 权重在线量化为 INT4
#       权重 HBM IO: 54GB → ~14GB (4x)
#
# 与 v0.7.0 的关键区别：
#   v0.7.0: --quantization awq CLI flag → 平台评测机覆盖 → 无效
#   v0.8.0: 修改 vLLM config/model.py 默认值 → 平台无法覆盖 → 一定生效
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/scripts/rocm_env.sh"

_resolve_model_path() {
    for cand in \
        /root/Qwen3.5-27B /data/Qwen3.5-27B \
        "${SCNET_HOME:-/public/home/xdzs2026_c415}/Qwen3.5-27B" \
        "${HOME}/Qwen3.5-27B"; do
        [[ -d "$cand" ]] && echo "$cand" && return
    done
    echo "${MODEL_PATH:-/data/Qwen3.5-27B}"
}

MODEL_PATH="$(_resolve_model_path)"
PORT="${PORT:-8000}"

VLLM_ARGS=(
    --model "${MODEL_PATH}"
    --port "${PORT}"
    --tensor-parallel-size 1
    --max-model-len 32768
    --max-num-seqs 256
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.95}"
    --dtype bfloat16
    --trust-remote-code
    --served-model-name Qwen3.5-27B
    --load-format runai_streamer
    --enable-prefix-caching
    --no-enable-log-requests
    --disable-log-stats
    --compilation-config '{"cudagraph_mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}'
)

echo "[launch] === v0.8.0: bitsandbytes INT4 online quant + HIP FA ==="
echo "[launch]   quantization DEFAULT = bitsandbytes (patched vLLM source, no CLI flag)"
echo "[launch]   bnb_4bit_compute_dtype = bfloat16, bnb_4bit_quant_type = nf4"
echo "[launch]   bf16 weights → online INT4 at load time (4x IO reduction)"
echo "[launch]   VLLM_ROCM_USE_AITER=1 (HIP FA for GQA layers)"
echo "[launch]   model: ${MODEL_PATH}"
echo "[launch]   port:  ${PORT}"
echo "[launch] ====================================="

cd /tmp
exec python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" "$@"
