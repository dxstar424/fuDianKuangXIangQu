#!/usr/bin/env bash
# ============================================================
# v0.6.0 FINAL — 强制 AITER HIP FlashAttention + 全配置
#
# 根因：v0.5.0 设了 UNIFIED_ATTENTION=1 → 走 AITER Triton kernel
#       真正快的 FLASH_ATTN（AITER HIP FlashAttention）在 P2
#       必须不设 UNIFIED_ATTENTION 才能 fall through 到它
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

echo "[launch] === v0.6.0 FINAL: HIP FlashAttention ==="
echo "[launch]   VLLM_ROCM_USE_AITER=1 (enable AITER)"
echo "[launch]   VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 (skip Triton unified)"
echo "[launch]   → should select FLASH_ATTN (AITER HIP CK kernel)"
echo "[launch]   model: ${MODEL_PATH}"
echo "[launch]   port:  ${PORT}"
echo "[launch]   prefix-caching + warmup + cudagraph FULL_DECODE_ONLY"
echo "[launch] ====================================="

cd /tmp
exec python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" "$@"
