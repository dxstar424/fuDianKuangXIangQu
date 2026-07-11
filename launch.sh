#!/usr/bin/env bash
# ============================================================
# v0.4.0 激进冲刺 — 去掉 FP8 负优化 + 最大化系统配置
#
# 历史：四轮实验 A-D 全部 ~60 分（含 --quantization fp8）。
# DCU 讲义实测：FP8 量化在 prefill 是负优化（反量化开销 > 收益），
# 这正是 16-32K 5.77 < baseline 7.75 的根因。
#
# v0.4.0 策略：
#   - 完全移除 FP8 权重量化（最关键！）
#   - GPU 0.97 激进显存（更大 KV cache pool）
#   - cudagraph_mode=3 FULL_DECODE_ONLY + capture_sizes=[1,2,4,8]
#   - BLAS 后端：rocBLAS decode + hipBLASLt prefill 由 TORCH_BLAS_PREFER_HIPBLASLT 控制
#   - Triton FlashAttention prefill（vLLM V1 默认 TRITON_ATTN 后端）
#   - warmup 全档（稳定 TTFT P99）
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

# ── v0.4.0 激进冲刺配置 ──
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
DO_WARMUP="${DO_WARMUP:-1}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-2}"
WARMUP_TIER="${WARMUP_TIER:-all}"
USE_FDU_SERVER="${USE_FDU_SERVER:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
# FULL_DECODE_ONLY: decode 图捕获, prefill 走 eager (最优安全配置)
COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"cudagraph_mode\": 3, \"cudagraph_capture_sizes\": [1, 2, 4, 8]}}"
LOAD_FORMAT="${LOAD_FORMAT:-runai_streamer}"
# ★★★ 关 FP8 —— 这是 16-32K 倒退的根因 ★★★
ENABLE_FP8_WEIGHT_QUANT="${ENABLE_FP8_WEIGHT_QUANT:-0}"

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
    --load-format "${LOAD_FORMAT}"
    --compilation-config "${COMPILATION_CONFIG}"
    --block-size 32
)

if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]] && [[ "${FDU_ENABLE_PREFIX_CACHE}" == "1" ]]; then
    VLLM_ARGS+=(--enable-prefix-caching)
fi

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    VLLM_ARGS+=(--enforce-eager)
fi

# ★ v0.4.0: 不再传 --quantization fp8 ★

_log_config() {
    echo "[launch] === v0.4.0 aggressive (NO FP8) ==="
    echo "[launch]   model: ${MODEL_PATH}"
    echo "[launch]   port:  ${PORT}"
    echo "[launch]   GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
    echo "[launch]   DO_WARMUP=${DO_WARMUP} TIER=${WARMUP_TIER}"
    echo "[launch]   prefix=${ENABLE_PREFIX_CACHING}/${FDU_ENABLE_PREFIX_CACHE}"
    echo "[launch]   ENFORCE_EAGER=${ENFORCE_EAGER}"
    echo "[launch]   FP8_WEIGHT_QUANT=${ENABLE_FP8_WEIGHT_QUANT}  (OFF=关键!)"
    echo "[launch]   compilation=${COMPILATION_CONFIG}"
    echo "[launch]   block_size=32 (大块=少页表开销)"
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
