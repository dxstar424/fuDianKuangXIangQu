#!/usr/bin/env bash
# ============================================================
# FDU SCCSCC26 — vLLM 推理服务启动脚本（提交入口）
# 平台调用: bash launch.sh --model /data/Qwen3.5-27B --port 8000 [--tensor-parallel-size N]
#
# 本脚本 = 官方 start_vllm.sh 锁定命令的可参数化版本。
# 锁定参数（不得改动，须与评测命令一致）:
#   dtype / max-num-seqs / max-num-batched-tokens / gpu-memory-utilization /
#   enable_thinking / reasoning-parser / served-model-name。
#
# 优化 flag 在本地验证通过（吞吐↑ 且 SLA 达标 且 精度 Δ≤1%）后，
# 才追加到命令末尾（见文末占位）。
# ============================================================
set -u
set -o pipefail

MODEL_DIR="/data/Qwen3.5-27B"
PORT=8000
TP=1
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)                 MODEL_DIR="$2"; shift 2 ;;
        --port)                  PORT="$2";      shift 2 ;;
        --tensor-parallel-size)  TP="$2";        shift 2 ;;
        *)                       EXTRA+=("$1");  shift ;;
    esac
done

# --- 官方锁定命令 ---
# 优化 flag 验证通过后，在下面追加对应行，例如：
#     --kv-cache-dtype fp8 \
exec vllm serve "$MODEL_DIR" \
    --served-model-name Qwen3.5-27B \
    --port "$PORT" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --max-num-seqs 128 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.95 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    ${EXTRA[@]+"${EXTRA[@]}"}
