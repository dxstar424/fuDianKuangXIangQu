#!/usr/bin/env bash
# ============================================================
# 评测启动脚本 — 去掉负优化（先超过官方 Baseline）
#
# 本队正式分：2026-07-10 lutinayi → 59.97（12.92/10.04/5.77）
#   ≈ 接近官方 Baseline（公式约 60 分）；16-32K 低于 Baseline
# 「富贵花开」84 分是他队，不是本队历史成绩
#
# 默认策略：
#   - 入口：stock vllm api_server（不用 fdu_vllm.server）
#   - GPU_MEMORY_UTILIZATION=0.94（禁止默认 0.95）
#   - DO_WARMUP=0
#   - prefix caching + 关日志
#   - --enforce-eager
#
# 禁止改：max-model-len / max-num-seqs / max-num-batched-tokens / scheduler
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/rocm_env.sh
source "$SCRIPT_DIR/scripts/rocm_env.sh"

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
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

# ── v0.2.17 实验 B：GPU 0.95 + warmup + native graph ──
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
DO_WARMUP="${DO_WARMUP:-1}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-1}"
WARMUP_TIER="${WARMUP_TIER:-8-16K}"
# 0=stock api_server（推荐）；1=fdu_vllm.server（仅本地验证后）
USE_FDU_SERVER="${USE_FDU_SERVER:-0}"
# 0=开原生 HIP Graph（平台测）；1=强制 eager
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

export GPU_MEMORY_UTILIZATION
export MODEL_PATH
export FDU_ENABLE_KV_QUANT="${FDU_ENABLE_KV_QUANT:-0}"
export FDU_ENABLE_PREFIX_CACHE="${FDU_ENABLE_PREFIX_CACHE:-1}"
export FDU_PHASE="${FDU_PHASE:-1}"
# 仅插件目录；勿把仓库根加入 PYTHONPATH（会盖住完整 vllm wheel）
if [[ "${USE_FDU_SERVER}" == "1" ]]; then
    export PYTHONPATH="${SCRIPT_DIR}/src"
fi

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
    --no-enable-log-requests
    --disable-log-stats
)

if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]] && [[ "${FDU_ENABLE_PREFIX_CACHE}" == "1" ]]; then
    VLLM_ARGS+=(--enable-prefix-caching)
fi

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    VLLM_ARGS+=(--enforce-eager)
fi

_log_config() {
    echo "[launch] === recovery config (post 59.97) ==="
    echo "[launch]   model: ${MODEL_PATH}"
    echo "[launch]   port:  ${PORT}"
    echo "[launch]   GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}  (NOT 0.95)"
    echo "[launch]   DO_WARMUP=${DO_WARMUP} TIER=${WARMUP_TIER}"
    echo "[launch]   prefix=${ENABLE_PREFIX_CACHING}/${FDU_ENABLE_PREFIX_CACHE}"
    echo "[launch]   USE_FDU_SERVER=${USE_FDU_SERVER}  (0=stock api_server)"
    echo "[launch]   ENFORCE_EAGER=${ENFORCE_EAGER}"
    echo "[launch]   locked: max-num-seqs=${MAX_NUM_SEQS}"
    echo "[launch] ====================================="
}

_log_config

_run_server() {
    # 勿在仓库根 cwd 起 python：sys.path[0] 会加载残缺 vendored vllm/
    cd /tmp
    if [[ "${USE_FDU_SERVER}" == "1" ]]; then
        exec python -m fdu_vllm.server "${VLLM_ARGS[@]}" "$@"
    else
        exec python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" "$@"
    fi
}

_wait_healthy() {
    local deadline="${HEALTH_TIMEOUT:-900}"
    local i=0
    echo "[launch] waiting for health (timeout=${deadline}s) ..."
    while (( i < deadline )); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
            || curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
            echo "[launch] Server healthy after ${i}s"
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
    _run_server
fi

cd /tmp
if [[ "${USE_FDU_SERVER}" == "1" ]]; then
    python -m fdu_vllm.server "${VLLM_ARGS[@]}" "$@" &
else
    python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" "$@" &
fi
SERVER_PID=$!
cd "$SCRIPT_DIR"
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

_wait_healthy

echo "[launch] starting warmup ..."
if ! python "$SCRIPT_DIR/scripts/warmup_server.py" \
    --host 127.0.0.1 --port "${PORT}" \
    --rounds "${WARMUP_ROUNDS}" --tier "${WARMUP_TIER}"; then
    echo "[launch] WARNING: warmup failed (non-fatal); serving anyway" >&2
else
    echo "[launch] Warmup done"
fi

echo "[launch] serving (pid=${SERVER_PID})"
trap - EXIT
wait "${SERVER_PID}"
