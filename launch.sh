#!/usr/bin/env bash
# ============================================================
# v1.1.0 — Pre-quantize bf16→AWQ INT4 at startup, native vLLM AWQ pipeline
#
# 策略：启动时量化模型到 /tmp/awq_model/，vLLM 原生 AWQ 加载
#   无需 monkey-patch 权重加载！vLLM 自带 AWQ Triton kernels
#
#   Step 1: PYTHONPATH → fdu_vllm importable
#   Step 2: python -m fdu_vllm.pre_quantize → bf16→AWQ INT4 量化
#   Step 3: vLLM --model /tmp/awq_model --quantization awq
#   Step 4: vLLM 原生加载 AWQ 格式 → Triton fused dequant+matmul
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
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
AWQ_MODEL_DIR="${AWQ_MODEL_DIR:-/tmp/awq_model}"
PORT="${PORT:-8000}"

echo "[launch] === v1.1.0: Pre-quantize bf16→AWQ INT4 + native vLLM AWQ ==="
echo "[launch]   bf16 model: ${MODEL_PATH}"
echo "[launch]   AWQ output: ${AWQ_MODEL_DIR}"

# ── Step 1: Pre-quantize if not already done ──
if [[ ! -f "${AWQ_MODEL_DIR}/quant_config.json" ]]; then
    echo "[launch]   Pre-quantizing (one-time, ~60-90s)..."
    python -m fdu_vllm.pre_quantize "${MODEL_PATH}" "${AWQ_MODEL_DIR}"
    echo "[launch]   Pre-quantization complete."
else
    echo "[launch]   AWQ model already exists, skipping pre-quantization."
fi

# ── Step 2: Start vLLM with AWQ model ──
VLLM_ARGS=(
    --model "${AWQ_MODEL_DIR}"
    --quantization awq
    --port "${PORT}"
    --tensor-parallel-size 1
    --max-model-len 32768
    --max-num-seqs 256
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.95}"
    --dtype float16
    --trust-remote-code
    --served-model-name Qwen3.5-27B
    --load-format auto
    --enable-prefix-caching
    --no-enable-log-requests
    --disable-log-stats
    --compilation-config '{"cudagraph_mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}'
)

echo "[launch]   Starting vLLM with AWQ model..."
echo "[launch]   port:  ${PORT}"
echo "[launch] ====================================="

cd /tmp
exec python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" "$@"
