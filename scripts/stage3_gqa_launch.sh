#!/bin/bash
# 阶段3：GQA 深接启动（须先 verify_token_consistency + 8-16K×5）
# 默认：仅开 GQA，关 defrag / HIP Graph / KV FP8
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(dirname "$SCRIPT_DIR")"

export PROJ
export PORT="${PORT:-8001}"
export USE_FDU_SERVER=1
export FDU_PHASE=2
export FDU_ENABLE_GQA_OPT=1
export FDU_ATTENTION_BACKEND="${FDU_ATTENTION_BACKEND:-vllm_default}"
export FDU_KV_CACHE_STRATEGY=none
export FDU_ENABLE_HIP_GRAPH=0
export FDU_ENABLE_KV_QUANT=0
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export DO_WARMUP="${DO_WARMUP:-0}"
export PYTHONPATH="${PROJ}/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[stage3_gqa] USE_FDU_SERVER=1 FDU_PHASE=2 GQA=1 defrag=off graph=off fp8=off"
echo "[stage3_gqa] Protocol: smoke → verify_token_consistency → 8-16K×5 → quick gate"
exec bash "$PROJ/scripts/scnet_start_optimized.sh"
