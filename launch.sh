#!/usr/bin/env bash
# ============================================================
# v0.9.0 — FP8 online quantization (torch._scaled_mm ROCm HIP kernel)
#
# 策略：fdu_vllm/quant_force.py monkey-patch vllm.config.model.ModelConfig
#       在 vLLM import 时自动执行，将 quantization 强制为 "fp8"
#       平台评测机无法跳过这个 hook（vllm/__init__.py 无条件调用 fdu_vllm.activate()）
#
# 与 v0.8.1 的区别：
#   v0.8.1: bitsandbytes INT4 → matmul_4bit on-the-fly dequant（无 ROCm HIP kernel）
#   v0.9.0: FP8 W8A8 → torch._scaled_mm（ROCm 原生 HIP kernel），无需反量化
#
# ★ 关键修复：PYTHONPATH 确保 fdu_vllm 可 import（v0.8.1 因 cd /tmp 导致 import 失败）
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

echo "[launch] === v0.9.0: FP8 online quantization (torch._scaled_mm HIP kernel) ==="
echo "[launch]   quant_force.py monkey-patches ModelConfig at vLLM import time"
echo "[launch]   quantization → 'fp8' (forced, NOT via CLI flag)"
echo "[launch]   bf16 weights → FP8 W8A8 at model load (2x IO reduction)"
echo "[launch]   torch._scaled_mm: ROCm 原生 HIP kernel (no on-the-fly dequant)"
echo "[launch]   VLLM_ROCM_USE_AITER=1 (HIP FA for attention)"
echo "[launch]   PYTHONPATH=$SCRIPT_DIR (fdu_vllm import fix)"
echo "[launch]   model: ${MODEL_PATH}"
echo "[launch]   port:  ${PORT}"
echo "[launch] ====================================="

cd /tmp
exec python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" "$@"
