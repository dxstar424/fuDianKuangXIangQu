#!/bin/bash
# Phase2 板块：KV defrag（实验用；须 deep hook 进 vLLM 后才有真实吞吐收益）
# 默认其余 Phase2 关，仅开 defrag 策略标记 + 插件路径
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(dirname "$SCRIPT_DIR")"

export PROJ
export PORT="${PORT:-8001}"
export USE_FDU_SERVER=1
export FDU_PHASE=2
export FDU_KV_CACHE_STRATEGY=defrag
export FDU_ENABLE_GQA_OPT=0
export FDU_ENABLE_HIP_GRAPH=0
export FDU_ENABLE_KV_QUANT=0
export FDU_ATTENTION_BACKEND=vllm_default
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export DO_WARMUP="${DO_WARMUP:-0}"
export PYTHONPATH="${PROJ}/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[stage3_defrag] defrag=ON gqa=off graph=off fp8=off"
echo "[stage3_defrag] NOTE: 若日志有 KV hooks 但吞吐不变 → CacheEngine 未 deep hook，勿合 main"
echo "[stage3_defrag] 主攻观测: 16-32K 吞吐 + TTFT P99"
exec bash "$PROJ/scripts/scnet_start_optimized.sh"
