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
# 与平台 launch.sh 一致：默认关 warmup；需要时 DO_WARMUP=1
export DO_WARMUP="${DO_WARMUP:-0}"
export WARMUP_TIER="${WARMUP_TIER:-8-16K}"

# 与 baseline 相同：模型应在 /root/Qwen3.5-27B（用户先手动 cp，见 docs/SCNET_RUN.md）
# 勿 cp 到已存在的 /root/Qwen3.5-27B 目录内（会嵌套成 Qwen3.5-27B/Qwen3.5-27B）
if [[ -f /root/Qwen3.5-27B/config.json ]]; then
    export MODEL_PATH=/root/Qwen3.5-27B
elif [[ -f "$SCNET_HOME/Qwen3.5-27B/config.json" ]]; then
    echo "[scnet_start] WARN: /root/Qwen3.5-27B missing; run:" >&2
    echo "  cp -r $SCNET_HOME/Qwen3.5-27B /root/Qwen3.5-27B" >&2
    export MODEL_PATH="$SCNET_HOME/Qwen3.5-27B"
fi

# 与 baseline 相同用 vllm_cscc wheel；PYTHONPATH 只能 src，不能 $PROJ
export PYTHONPATH="$PROJ/src"

# 勿 cd 到仓库根再起 python（见 launch.sh）；launch.sh 内会 cd /tmp
cd /tmp
exec bash "$PROJ/launch.sh"
