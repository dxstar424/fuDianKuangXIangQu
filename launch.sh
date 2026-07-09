#!/bin/bash
# ============================================================
# 评测启动脚本 — 最容易拿分的低风险优化（合规）
# - 显存利用率 0.94（长上下文 KV 更充裕，主攻 8-16K / 16-32K）
# - prefix caching（降 TTFT）
# - 关闭日志开销
# - 分档 warmup（稳 TTFT P99）
# - KV FP8 默认关闭（保精度系数=1.0）
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/rocm_env.sh
source "$SCRIPT_DIR/scripts/rocm_env.sh"

MODEL_PATH="${MODEL_PATH:-/data/Qwen3.5-27B}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
# 提分：略提高显存利用率（仍低于 OOM 风险区，SCNet 可 A/B 0.93~0.95）
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
DO_WARMUP="${DO_WARMUP:-1}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-1}"
WARMUP_TIER="${WARMUP_TIER:-all}"

export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"
export GPU_MEMORY_UTILIZATION

# 容易拿分：先保精度，KV 在线量化默认关（验证通过后再 FDU_ENABLE_KV_QUANT=1）
export FDU_ENABLE_KV_QUANT="${FDU_ENABLE_KV_QUANT:-0}"
export FDU_ENABLE_PREFIX_CACHE="${FDU_ENABLE_PREFIX_CACHE:-1}"

VLLM_ARGS=(
    --model "${MODEL_PATH}"
    --port "${PORT}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --dtype bfloat16
    --trust-remote-code
    --served-model-name Qwen3.5-27B
    --disable-log-requests
    --disable-log-stats
)

if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]] && [[ "${FDU_ENABLE_PREFIX_CACHE}" == "1" ]]; then
    VLLM_ARGS+=(--enable-prefix-caching)
fi

_log_phase1_config() {
    echo "[launch] === Phase 1 config (FDU_PHASE=${FDU_PHASE:-1}) ==="
    echo "[launch]   1.1 GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
    echo "[launch]   1.2 DO_WARMUP=${DO_WARMUP} WARMUP_TIER=${WARMUP_TIER} ROUNDS=${WARMUP_ROUNDS}"
    echo "[launch]   1.3 ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
    echo "[launch]   1.4 --disable-log-requests --disable-log-stats"
    echo "[launch]   1.5 ROCm env via scripts/rocm_env.sh"
    echo "[launch]   1.6 FDU_ENABLE_KV_QUANT=${FDU_ENABLE_KV_QUANT}"
    echo "[launch]   1.7 --dtype bfloat16 --served-model-name Qwen3.5-27B --max-model-len ${MAX_MODEL_LEN}"
    if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]] && [[ "${FDU_ENABLE_PREFIX_CACHE}" == "1" ]]; then
        echo "[launch]   prefix caching: ON"
    else
        echo "[launch]   prefix caching: OFF"
    fi
    echo "[launch] =========================================="
}

_log_phase1_config

_start_server() {
    exec python -m fdu_vllm.server "${VLLM_ARGS[@]}" "$@"
}

_wait_healthy() {
    local deadline=300
    local i=0
    while (( i < deadline )); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            echo "[launch] Server healthy after ${i}s"
            return 0
        fi
        sleep 2
        ((i += 2))
    done
    echo "[launch] ERROR: server not healthy within ${deadline}s" >&2
    return 1
}

if [[ "${DO_WARMUP}" != "1" ]]; then
    _start_server
fi

python -m fdu_vllm.server "${VLLM_ARGS[@]}" "$@" &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

_wait_healthy

python "$SCRIPT_DIR/scripts/warmup_server.py" \
    --host 127.0.0.1 --port "${PORT}" \
    --rounds "${WARMUP_ROUNDS}" --tier "${WARMUP_TIER}"

echo "[launch] Warmup done; serving (pid=${SERVER_PID})"
wait "${SERVER_PID}"
