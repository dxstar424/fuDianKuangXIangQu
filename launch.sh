#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        echo "[launch] ERROR: neither python nor python3 is available" >&2
        exit 2
    fi
elif ! PYTHON_BIN="$(command -v -- "$PYTHON_BIN")"; then
    echo "[launch] ERROR: selected interpreter is not available" >&2
    exit 2
fi

_resolve_model_path() {
    local candidate
    if [[ -n "${MODEL_PATH:-}" ]]; then
        printf '%s\n' "$MODEL_PATH"
        return
    fi
    for candidate in \
        /root/Qwen3.5-27B \
        /data/Qwen3.5-27B \
        "${SCNET_HOME:-/public/home/xdzs2026_c415}/Qwen3.5-27B" \
        "${HOME}/Qwen3.5-27B"
    do
        if [[ -d "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    printf '%s\n' /data/Qwen3.5-27B
}

_is_true() {
    case "${1:-}" in
        1|[Tt][Rr][Uu][Ee]) return 0 ;;
        *) return 1 ;;
    esac
}

MODEL_PATH="$(_resolve_model_path)"
PORT="${PORT:-8000}"

source "$SCRIPT_DIR/scripts/rocm_env.sh"

EXPECTED_PREFIX="$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')"

unset PYTHONPATH
cd /tmp

PREFLIGHT_ARGS=(
    "$SCRIPT_DIR/scripts/preflight_rocm.py"
    --expected-prefix "$EXPECTED_PREFIX"
    --require-arch gfx936
)
if _is_true "$VLLM_ROCM_USE_SKINNY_GEMM" \
    && ! _is_true "$FDU_FORCE_STOCK_GEMM"
then
    PREFLIGHT_ARGS+=(--require-skinny)
fi

"$PYTHON_BIN" "${PREFLIGHT_ARGS[@]}"

VLLM_ARGS=(
    --model "$MODEL_PATH"
    --port "$PORT"
    --tensor-parallel-size 1
    --max-model-len 32768
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.94}"
    --dtype bfloat16
    --trust-remote-code
    --served-model-name Qwen3.5-27B
    --load-format auto
    --no-enable-log-requests
)

exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    "${VLLM_ARGS[@]}" "$@"
