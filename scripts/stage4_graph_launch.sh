#!/bin/bash
# 阶段4：HIP Graph opt-in（仅 S3 平台净增且距截止≥48h）
# 要求：ENFORCE_EAGER=0；失败立即回退 ENFORCE_EAGER=1 FDU_ENABLE_HIP_GRAPH=0
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(dirname "$SCRIPT_DIR")"

export PROJ
export PORT="${PORT:-8001}"
export USE_FDU_SERVER=1
export FDU_PHASE=2
export FDU_ENABLE_GQA_OPT="${FDU_ENABLE_GQA_OPT:-1}"
export FDU_ENABLE_HIP_GRAPH=1
export FDU_KV_CACHE_STRATEGY=none
export FDU_ENABLE_KV_QUANT=0
export ENFORCE_EAGER=0
export DO_WARMUP="${DO_WARMUP:-0}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
export PYTHONPATH="${PROJ}/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[stage4_graph] HIP Graph ON + ENFORCE_EAGER=0"
echo "[stage4_graph] Must pass ≥30min SCNet soak; on crash: ENFORCE_EAGER=1 GRAPH=0"
exec bash "$PROJ/scripts/scnet_start_optimized.sh"
