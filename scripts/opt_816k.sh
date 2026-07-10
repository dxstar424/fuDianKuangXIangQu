#!/bin/bash
# 8–16K 专用优化启动（一次只开一个实验）
#
# 基于 SCNet 实测（10 req, conc=1）：
#   recover: Output 12.19 | TTFT P99 15.9s | TPOT P99 71.0ms
#   tpot (ENFORCE_EAGER=0): Output 12.17 | TTFT/TPOT 持平 → NO-GO（2026-07-11）
#   时间拆分：TTFT≈11s + decode≈66s → decode 占 ~86%；原生 Graph 不降 TPOT
#
# 用法：
#   bash scripts/opt_816k.sh tpot         # 已证伪：勿合入
#   bash scripts/opt_816k.sh ttft         # 可选：仅 8-16K warmup（压 TTFT P99）
#   bash scripts/opt_816k.sh recover      # 对照：当前恢复默认
#
# 另开终端：
#   cd ~/testdata && ./run_throughput.sh 8-16K 10
#
# go/no-go（相对 recover 的 12.19）：
#   Output ≥ 12.7 tok/s（+0.5）且 TPOT P99 未明显变差、TTFT P99 未逼近熔断
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(dirname "$SCRIPT_DIR")"
MODE="${1:-tpot}"

export PROJ
export PORT="${PORT:-8001}"
export FDU_PHASE=1
export USE_FDU_SERVER=0
export GPU_MEMORY_UTILIZATION=0.94
export FDU_ENABLE_KV_QUANT=0
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
export FDU_ENABLE_PREFIX_CACHE=1

case "$MODE" in
  tpot|graph|eager-off)
    # 主实验：TPOT 占墙钟 ~86%，优先试 vLLM 原生 Graph
    export ENFORCE_EAGER=0
    export DO_WARMUP=0
    echo "[opt_816k] === 8-16K TPOT experiment ==="
    echo "[opt_816k] ENFORCE_EAGER=0 (native graph) DO_WARMUP=0 gpu=0.94"
    echo "[opt_816k] Expect: TPOT ↓ from ~70.5ms → Output ↑ toward ~13+"
    ;;
  ttft|warmup)
    # 次实验：TTFT P99 15.9s vs median 11.4s，尖刺明显
    export ENFORCE_EAGER=1
    export DO_WARMUP=1
    export WARMUP_TIER=8-16K
    export WARMUP_ROUNDS=1
    echo "[opt_816k] === 8-16K TTFT experiment ==="
    echo "[opt_816k] DO_WARMUP=1 TIER=8-16K ENFORCE_EAGER=1 gpu=0.94"
    echo "[opt_816k] Expect: TTFT P99 ↓ toward median (~11s); Output 微升"
    ;;
  recover|baseline)
    export ENFORCE_EAGER=1
    export DO_WARMUP=0
    echo "[opt_816k] === recover control ==="
    echo "[opt_816k] ENFORCE_EAGER=1 DO_WARMUP=0 (match platform recover)"
    ;;
  both)
    echo "[opt_816k] REFUSE: do not combine tpot+ttft in one run (A/B only)" >&2
    echo "Run tpot first, then ttft separately." >&2
    exit 1
    ;;
  *)
    echo "Usage: opt_816k.sh [tpot|ttft|recover]" >&2
    exit 1
    ;;
esac

echo "[opt_816k] After healthy:"
echo "  cd ~/testdata && ./run_throughput.sh 8-16K 10"
echo "[opt_816k] Record: Output tok/s, TTFT P99, TPOT P99"
exec bash "$PROJ/scripts/scnet_start_optimized.sh"
