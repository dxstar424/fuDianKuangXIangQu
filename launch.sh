#!/bin/bash
# ============================================================
# 评测启动脚本 — Phase 1 最有把握提分项（合规 · 默认开启）
#
# 相对 stock baseline（gpu=0.92、无 warmup、无 prefix、有日志）：
#   1.1 GPU_MEMORY_UTILIZATION=0.94     → 长档 KV 池更大（8-16K / 16-32K）
#   1.2 分档 warmup（主攻档优先）       → 稳 TTFT P99，防 SLA 熔断
#   1.3 --enable-prefix-caching        → 共享前缀降 TTFT
#   1.4 --disable-log-requests/stats   → 减 Python I/O
#   1.5 ROCm/DCU env（rocm_env.sh）    → 带宽/分配稳定性
#   1.6 FDU_ENABLE_KV_QUANT=0          → 保精度系数 1.0
#   1.7 --dtype bfloat16 + 合规接口    → 与官方权重一致
#
# 禁止：改 max-num-seqs / max-num-batched-tokens / batch scheduler
# 文档：docs/easy_scoring.md · docs/deep_optimization_guide.md §三
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/rocm_env.sh
source "$SCRIPT_DIR/scripts/rocm_env.sh"

# 模型路径：优先 /root（SCNet PDF：加载更快），再 /data，再环境变量
_resolve_model_path() {
    if [[ -n "${MODEL_PATH:-}" ]] && [[ -d "${MODEL_PATH}" ]]; then
        echo "${MODEL_PATH}"
        return
    fi
    for cand in \
        /root/Qwen3.5-27B \
        /data/Qwen3.5-27B \
        "${SCNET_HOME:-/public/home/xdzs2026_c415}/Qwen3.5-27B" \
        "${HOME}/Qwen3.5-27B"; do
        if [[ -d "$cand" ]]; then
            echo "$cand"
            return
        fi
    done
    echo "${MODEL_PATH:-/data/Qwen3.5-27B}"
}

MODEL_PATH="$(_resolve_model_path)"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# 与 stock baseline 一致，禁止为「提吞吐」改大（赛题红线）
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

# ── Phase 1 默认（可由环境覆盖；一次只改一个做 A/B）──
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
DO_WARMUP="${DO_WARMUP:-1}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-1}"
WARMUP_TIER="${WARMUP_TIER:-all}"

export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"
export GPU_MEMORY_UTILIZATION
export MODEL_PATH
export FDU_ENABLE_KV_QUANT="${FDU_ENABLE_KV_QUANT:-0}"
export FDU_ENABLE_PREFIX_CACHE="${FDU_ENABLE_PREFIX_CACHE:-1}"
export FDU_PHASE="${FDU_PHASE:-1}"

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
    echo "[launch] === Phase 1 sure-win config (FDU_PHASE=${FDU_PHASE}) ==="
    echo "[launch]   model: ${MODEL_PATH}"
    echo "[launch]   port:  ${PORT}"
    echo "[launch]   1.1 GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}  (stock=0.92)"
    echo "[launch]   1.2 DO_WARMUP=${DO_WARMUP} TIER=${WARMUP_TIER} ROUNDS=${WARMUP_ROUNDS}"
    echo "[launch]   1.3 prefix_caching=${ENABLE_PREFIX_CACHING}/${FDU_ENABLE_PREFIX_CACHE}"
    echo "[launch]   1.4 disable-log-requests + disable-log-stats"
    echo "[launch]   1.5 ROCm: HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-?} HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-?}"
    echo "[launch]   1.6 FDU_ENABLE_KV_QUANT=${FDU_ENABLE_KV_QUANT}"
    echo "[launch]   1.7 dtype=bfloat16 served=Qwen3.5-27B max_model_len=${MAX_MODEL_LEN}"
    echo "[launch]   locked: max-num-seqs=${MAX_NUM_SEQS} (unchanged vs stock)"
    echo "[launch] ========================================================"
}

_log_phase1_config

_start_server() {
    exec python -m fdu_vllm.server "${VLLM_ARGS[@]}" "$@"
}

_wait_healthy() {
    local deadline="${HEALTH_TIMEOUT:-900}"
    local i=0
    echo "[launch] waiting for health (timeout=${deadline}s) ..."
    while (( i < deadline )); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            echo "[launch] Server healthy (/health) after ${i}s"
            return 0
        fi
        if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
            echo "[launch] Server healthy (/v1/models) after ${i}s"
            return 0
        fi
        sleep 2
        ((i += 2)) || true
        if (( i % 30 == 0 )); then
            echo "[launch] still waiting... ${i}s"
        fi
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

echo "[launch] starting tiered warmup (8-16K first when tier=all) ..."
if ! python "$SCRIPT_DIR/scripts/warmup_server.py" \
    --host 127.0.0.1 --port "${PORT}" \
    --rounds "${WARMUP_ROUNDS}" --tier "${WARMUP_TIER}"; then
    echo "[launch] WARNING: warmup failed (non-fatal); serving anyway" >&2
    echo "[launch]   hint: DO_WARMUP=0 to skip, or WARMUP_TIER=8-16K for single tier" >&2
else
    echo "[launch] Warmup done"
fi

echo "[launch] serving (pid=${SERVER_PID})"
# 评测机需要前台进程；去掉 trap 以免 wait 后误杀
trap - EXIT
wait "${SERVER_PID}"
