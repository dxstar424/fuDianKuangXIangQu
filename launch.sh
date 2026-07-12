#!/usr/bin/env bash
# ============================================================
# v0.9.3 — bf16 stock + AITER optimizations (no weight quantization)
#
# 策略：PYTHONPATH 确保 fdu_vllm 可 import（v0.8.1 的 bug）
#       AITER: FLASH_ATTN + skinny_gemm + rmsnorm（env vars 控制）
#       无重量化 — FP8/bnb 在此平台 DCU 上均不可行
#
# ★ FP8/bnb 总结：
#   bnb INT4: matmul_4bit 无 ROCm HIP kernel（CPU 反量化）
#   FP8 W8A8: torch._scaled_mm 需要 MI300+（gfx942 不支持）
#   FP8 fallback: dequant+matmul = 81GB HBM > 54GB bf16（1.5x 更慢）
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# ★ FIX v0.9.0: 在 cd /tmp 之前把 repo root 加到 PYTHONPATH，让 fdu_vllm 可 import
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
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

echo "[launch] === v0.9.3: bf16 stock + AITER (FLASH_ATTN/skinny_gemm/rmsnorm) ==="
echo "[launch]   quant_force: no-op (weight quantization dead end on this DCU)"
echo "[launch]   AITER=1 → FLASH_ATTN backend (HIP CK FlashAttention)"
echo "[launch]   skinny_gemm=1 → decode GEMV HIP kernel"
echo "[launch]   rmsnorm=1 → AITER RMSNorm"
echo "[launch]   PYTHONPATH=$SCRIPT_DIR (fdu_vllm import fix)"
echo "[launch]   model: ${MODEL_PATH}"
echo "[launch]   port:  ${PORT}"
echo "[launch] ====================================="

cd /tmp
exec python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" "$@"
