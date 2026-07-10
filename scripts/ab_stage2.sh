#!/bin/bash
# 阶段2：launch 单变量 A/B（ENFORCE_EAGER / 单档 warmup）
# 用法（服务未起时）：
#   bash scripts/ab_stage2.sh eager-off     # ENFORCE_EAGER=0
#   bash scripts/ab_stage2.sh warmup-816    # DO_WARMUP=1 WARMUP_TIER=8-16K
#   bash scripts/ab_stage2.sh baseline     # 当前 recover 默认
# 另开终端测：cd ~/testdata && ./run_throughput.sh 8-16K 5
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(dirname "$SCRIPT_DIR")"
MODE="${1:-baseline}"

export PROJ
export PORT="${PORT:-8001}"
export FDU_PHASE="${FDU_PHASE:-1}"
export USE_FDU_SERVER="${USE_FDU_SERVER:-0}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
export FDU_ENABLE_KV_QUANT=0

case "$MODE" in
  baseline|recover)
    export ENFORCE_EAGER=1
    export DO_WARMUP=0
    echo "[ab_stage2] A=recover: ENFORCE_EAGER=1 DO_WARMUP=0 gpu=0.94"
    ;;
  eager-off|graph)
    export ENFORCE_EAGER=0
    export DO_WARMUP=0
    echo "[ab_stage2] B=eager-off: ENFORCE_EAGER=0 (vLLM native graph) DO_WARMUP=0"
    ;;
  warmup-816|warmup)
    export ENFORCE_EAGER=1
    export DO_WARMUP=1
    export WARMUP_TIER=8-16K
    export WARMUP_ROUNDS=1
    echo "[ab_stage2] B=warmup-816: DO_WARMUP=1 TIER=8-16K ENFORCE_EAGER=1"
    ;;
  *)
    echo "Usage: ab_stage2.sh [baseline|eager-off|warmup-816]" >&2
    exit 1
    ;;
esac

echo "[ab_stage2] After healthy: ./run_throughput.sh 8-16K 5"
echo "[ab_stage2] go/no-go: +≥0.5 tok/s vs recover tag, SLA not near 1.5×"
exec bash "$PROJ/scripts/scnet_start_optimized.sh"
