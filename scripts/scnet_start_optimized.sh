#!/usr/bin/env bash
set -euo pipefail

SCNET_HOME="${SCNET_HOME:-/public/home/xdzs2026_c415}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

_resolve_proj() {
    local candidate
    for candidate in \
        "${PROJ:-}" \
        "$HOME/2025pra-fdu-fudiankuangxiangqu" \
        "$SCNET_HOME/2025pra-fdu-fudiankuangxiangqu" \
        "$(cd "$SCRIPT_DIR/.." && pwd -P)"
    do
        if [[ -n "$candidate" && -f "$candidate/launch.sh" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    printf '%s\n' "$(cd "$SCRIPT_DIR/.." && pwd -P)"
}

PROJ="$(_resolve_proj)"
DEFAULT_PYTHON_BIN="/public/home/xdzs2026_c415/venvs/vllm_gfx936/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -n "${VLLM_ENV:-}" ]]; then
        PYTHON_BIN="${VLLM_ENV%/}/bin/python"
    else
        PYTHON_BIN="$DEFAULT_PYTHON_BIN"
    fi
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[scnet_start] ERROR: interpreter is not executable: $PYTHON_BIN" >&2
    exit 2
fi

if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ -f /root/Qwen3.5-27B/config.json ]]; then
        MODEL_PATH=/root/Qwen3.5-27B
    elif [[ -f "$SCNET_HOME/Qwen3.5-27B/config.json" ]]; then
        MODEL_PATH="$SCNET_HOME/Qwen3.5-27B"
    else
        MODEL_PATH="$HOME/Qwen3.5-27B"
    fi
fi

export PYTHON_BIN MODEL_PATH
export PORT="${PORT:-8001}"

exec bash "$PROJ/launch.sh" "$@"
