#!/bin/bash
# Phase 0: 记录 baseline 指标到 results/ 并生成 report 片段
set -euo pipefail

TESTDATA="${TESTDATA:-$HOME/testdata}"
OUT="${OUT:-results/baseline_$(date +%Y%m%d_%H%M%S).txt}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

mkdir -p "$(dirname "$OUT")"
cd "$TESTDATA"

{
    echo "FDU Baseline Record $(date -Iseconds)"
    echo "================================"
    echo ""
    echo "--- 4-8K ---"
    ./run_throughput.sh 4-8K 10 2>&1 || true
    echo ""
    echo "--- 8-16K ---"
    ./run_throughput.sh 8-16K 10 2>&1 || true
    echo ""
    echo "--- 16-32K ---"
    ./run_throughput.sh 16-32K 10 2>&1 || true
} | tee "$PROJ_DIR/$OUT"

echo "Saved to $PROJ_DIR/$OUT"
echo "Copy TTFT/TPOT/throughput into report.md section 5"
