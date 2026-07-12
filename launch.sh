#!/usr/bin/env bash
# ============================================================
# v0.5.0 — AITER unified attention + minimal config
#
# 五次实验全部 ~59.7，CLI flag 改动无效。
# v0.5.0 只做一件事：启用 AITER 统一注意力后端。
# AITER unified attention 是 vLLM ROCm 后端优先级最高的路径，
# 可能走 AITER 手写 HIP FlashAttention kernel。
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 最简 ROCm 环境 ──
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.4.2}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export SAFETENSORS_FAST_GPU=1

# ★★★ AITER 统一注意力 — v0.5.0 唯一新尝试 ★★★
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1

# 其他保持默认：AITER_RMSNORM=True (default), SKINNY_GEMM=True (default)
# 不设 TORCH_BLAS_PREFER_HIPBLASLT（让 vLLM 自己决定）

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

# ── 最简 vLLM CLI ──
# 不设 --compilation-config（让 O2 默认值生效）
# 不设 --block-size（让 vLLM 默认 16）
# 不设 --quantization（纯 bf16）
# 不设 --enforce-eager（让 CUDA Graph + torch.compile 生效）
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
)

echo "[launch] === v0.5.0 AITER UNIFIED ATTENTION ==="
echo "[launch]   VLLM_ROCM_USE_AITER=1"
echo "[launch]   VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1"
echo "[launch]   model: ${MODEL_PATH}"
echo "[launch]   port:  ${PORT}"
echo "[launch] ====================================="

# 启动
cd /tmp
exec python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" "$@"
