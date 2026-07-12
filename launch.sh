#!/usr/bin/env bash
# ============================================================
# v1.0.0 — AWQ INT4 online quantization (Triton fused dequant+matmul)
#
# 策略：fdu_vllm 三部曲
#   1) quant_force.py: ModelConfig monkey-patch → quantization="awq"
#      + 创建 quant_config.json 在模型目录
#   2) awq_online.py: 拦截权重加载, bf16→AWQ INT4 在线量化
#   3) AWQ Triton kernels: 融合 dequant+matmul（纯 Triton, GPU 原生）
#
# Triton on ROCm: VLLM_USE_TRITON_AWQ=1 被 rocm.py 强制开启
# 每次 decode 读取 13.5GB INT4 vs 54GB bf16 → 4x IO reduction
# ★ PYTHONPATH fix 确保 fdu_vllm 可 import
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

echo "[launch] === v1.0.0: AWQ INT4 online quant (Triton fused dequant+matmul) ==="
echo "[launch]   quant_force.py → ModelConfig monkey-patch → quantization='awq'"
echo "[launch]   awq_online.py → intercept weight loading → bf16→AWQ INT4"
echo "[launch]   AWQ Triton kernels → fused dequant+matmul (GPU-native)"
echo "[launch]   4x weight IO: 13.5GB INT4 vs 54GB bf16 per decode step"
echo "[launch]   AITER=1 → FLASH_ATTN + skinny_gemm + rmsnorm"
echo "[launch]   PYTHONPATH=$SCRIPT_DIR"
echo "[launch]   model: ${MODEL_PATH}"
echo "[launch]   port:  ${PORT}"
echo "[launch] ====================================="

cd /tmp
exec python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" "$@"
