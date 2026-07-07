#!/bin/bash
# ============================================================
# 一键 Baseline 评测流程
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_DIR"

echo "=== Baseline Benchmark Pipeline ==="

# 1. 启动服务
echo "[1/3] Starting baseline server..."
bash baseline/launch.sh &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null" EXIT

# 2. 等待健康检查
echo "[2/3] Waiting for server..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "      Server ready after ${i}s"
        break
    fi
    sleep 2
done

# 3. 运行评测
echo "[3/3] Running benchmark..."
python scripts/benchmark.py --host localhost --port 8000 --output results/

echo "=== Done ==="
