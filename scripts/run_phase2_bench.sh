#!/bin/bash
# Phase2 单板块 SCNet 快测（服务已 healthy 后另开终端执行）
# 用法: bash scripts/run_phase2_bench.sh [8-16K|16-32K|4-8K] [条数]
set -euo pipefail
TIER="${1:-8-16K}"
N="${2:-5}"
TESTDATA="${TESTDATA:-$HOME/testdata}"
for cand in "$HOME/testdata" "/public/home/xdzs2026_c415/testdata"; do
  [[ -d "$cand" ]] && TESTDATA="$cand" && break
done
if [[ ! -d "$TESTDATA" ]]; then
  echo "testdata not found" >&2
  exit 1
fi
cd "$TESTDATA"
echo "[phase2_bench] tier=$TIER n=$N"
./run_throughput.sh "$TIER" "$N"
