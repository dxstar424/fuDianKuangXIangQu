#!/bin/bash
# SCNet testdata 目录下一键启动优化版（对齐 PDF 端口 8001）
set -euo pipefail

PROJ="${PROJ:-$HOME/2025pra-fdu-fudiankuangxiangqu}"
if [[ ! -f "$PROJ/launch.sh" ]]; then
    PROJ="$(cd "$(dirname "$0")/.." && pwd)"
fi

export MODEL_PATH="${MODEL_PATH:-$HOME/Qwen3.5-27B}"
export PORT="${PORT:-8001}"
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
