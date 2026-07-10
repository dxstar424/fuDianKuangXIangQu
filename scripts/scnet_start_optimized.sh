#!/bin/bash
# SCNet testdata 目录下一键启动优化版（对齐 PDF 端口 8001）
set -euo pipefail

SCNET_HOME="${SCNET_HOME:-/public/home/xdzs2026_c415}"

_resolve_proj() {
    local cand
    for cand in \
        "${PROJ:-}" \
        "$HOME/2025pra-fdu-fudiankuangxiangqu" \
        "$SCNET_HOME/2025pra-fdu-fudiankuangxiangqu" \
        "$(cd "$(dirname "$0")/.." && pwd)"
    do
        [[ -n "$cand" && -f "$cand/launch.sh" ]] && { echo "$cand"; return; }
    done
    echo "$(cd "$(dirname "$0")/.." && pwd)"
}

PROJ="$(_resolve_proj)"

# 模型优先 /root（PDF copy 后加载快），否则 SCNet 家目录 / 当前 HOME
if [[ -d /root/Qwen3.5-27B ]]; then
    export MODEL_PATH="${MODEL_PATH:-/root/Qwen3.5-27B}"
elif [[ -d "$SCNET_HOME/Qwen3.5-27B" ]]; then
    export MODEL_PATH="${MODEL_PATH:-$SCNET_HOME/Qwen3.5-27B}"
else
    export MODEL_PATH="${MODEL_PATH:-$HOME/Qwen3.5-27B}"
fi
export PORT="${PORT:-8001}"
export FDU_PHASE="${FDU_PHASE:-1}"
export DO_WARMUP="${DO_WARMUP:-1}"
export WARMUP_TIER="${WARMUP_TIER:-all}"

# PDF：从家目录 copy 到 /root 加速加载
if [[ -d "$MODEL_PATH" ]] && [[ ! -d /root/Qwen3.5-27B ]]; then
    echo "[scnet_start] Copy model to /root/Qwen3.5-27B ..."
    cp -r "$MODEL_PATH" /root/Qwen3.5-27B
    export MODEL_PATH=/root/Qwen3.5-27B
elif [[ -d /root/Qwen3.5-27B ]]; then
    export MODEL_PATH=/root/Qwen3.5-27B
fi

cd "$PROJ"
exec bash launch.sh
