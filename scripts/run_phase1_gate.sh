#!/bin/bash
# Phase 1 完整门禁流程（SCNet）：启动优化版 → quick gate
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Phase 1 gate: verify config ==="
bash "$SCRIPT_DIR/verify_phase1_config.sh"

echo ""
echo "=== Phase 1 gate: start optimized server (port 8001) ==="
echo "Run in background or separate terminal:"
echo "  bash $SCRIPT_DIR/scnet_start_optimized.sh"
echo ""
echo "Then:"
echo "  bash $SCRIPT_DIR/gate_check.sh quick"
echo ""
echo "Phase 1 completion criteria (manual review):"
echo "  - 8-16K throughput >= stock baseline"
echo "  - TTFT/TPOT P99 <= baseline x 1.5"
echo "  - accuracy delta <= 1%"
echo "  - platform submit SLA/accuracy penalty = 0"
