#!/bin/bash
# 每阶段精度/性能门禁（SCNet testdata）
set -euo pipefail

MODE="${1:-quick}"
SCNET_HOME="${SCNET_HOME:-/public/home/xdzs2026_c415}"
if [[ -z "${TESTDATA:-}" ]]; then
    for cand in "$HOME/testdata" "$SCNET_HOME/testdata"; do
        if [[ -d "$cand" ]]; then
            TESTDATA="$cand"
            break
        fi
    done
    TESTDATA="${TESTDATA:-$HOME/testdata}"
fi
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"

if [[ ! -d "$TESTDATA" ]]; then
    echo "[gate_check] testdata not found at $TESTDATA"
    echo "Run scripts/scnet_setup.sh first"
    exit 1
fi

cd "$TESTDATA"

case "$MODE" in
  quick)
    echo "=== Gate: throughput 8-16K x20 + accuracy hotpotqa x10 ==="
    ./run_throughput.sh 8-16K 20
    ./run_accuracy.sh hotpotqa 10
    ;;
  full)
    echo "=== Gate: full throughput + all accuracy tasks ==="
    ./run_throughput.sh
    ./run_accuracy.sh
    ./run_accuracy.sh gov_report 10
    ./run_accuracy.sh retrieval_multi_point 10
    ./run_accuracy.sh aggregation_keyword_aggregation 10
    ;;
  throughput)
    ./run_throughput.sh "${@:2}"
    ;;
  accuracy)
    ./run_accuracy.sh "${@:2}"
    ;;
  *)
    echo "Usage: gate_check.sh [quick|full|throughput|accuracy] [args...]"
    exit 1
    ;;
esac

echo "=== Gate check passed (review metrics manually; Δ≤1% required) ==="
