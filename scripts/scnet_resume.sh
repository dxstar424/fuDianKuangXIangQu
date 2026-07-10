#!/bin/bash
# SCNet 容器重启后最短恢复流程（路径对齐团队调试指南）
#
# 持久化目录（4h 关机后仍保留，无需重下）:
#   /public/home/xdzs2026_c415/{Qwen3.5-27B,vllm_cscc,testdata,2025pra-fdu-fudiankuangxiangqu}
# 每次新容器仅需: pip install wheel → cp 模型到 /root → 启动服务 → 冒烟
#
# 用法:
#   bash scripts/scnet_resume.sh              # baseline (start_vllm.sh)
#   bash scripts/scnet_resume.sh optimized    # lutinayi Phase 1 (launch.sh)
#   bash scripts/scnet_resume.sh smoke        # 仅冒烟（服务已起）
#   bash scripts/scnet_resume.sh bench 8-16K  # 冒烟 + 单档吞吐 x10
set -euo pipefail

SCNET_HOME="${SCNET_HOME:-/public/home/xdzs2026_c415}"
VLLM_DIST="${VLLM_DIST:-$SCNET_HOME/vllm_cscc/dist}"
MODEL_SRC="${MODEL_SRC:-$SCNET_HOME/Qwen3.5-27B}"
MODEL_ROOT="${MODEL_ROOT:-/root/Qwen3.5-27B}"
TESTDATA="${TESTDATA:-$SCNET_HOME/testdata}"
PROJ="${PROJ:-$SCNET_HOME/2025pra-fdu-fudiankuangxiangqu}"
PORT="${PORT:-8001}"
MODE="${1:-baseline}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY=127.0.0.1,localhost

_install_vllm_wheel() {
    log "pip install vLLM wheel from $VLLM_DIST"
    cd "$VLLM_DIST"
    local wheel
    wheel="$(ls -1 vllm-*.whl 2>/dev/null | head -1 || true)"
    if [[ -z "$wheel" ]]; then
        echo "ERROR: 无 wheel，请先: cd $SCNET_HOME/vllm_cscc && python setup.py bdist_wheel" >&2
        exit 1
    fi
    pip install "$wheel" --no-deps
    python -c "import vllm; print('vLLM', vllm.__version__)"
}

_copy_model_to_root() {
    if [[ ! -f "$MODEL_SRC/config.json" ]]; then
        echo "ERROR: 模型不在 $MODEL_SRC，请先 modelscope download" >&2
        exit 1
    fi
    if [[ ! -f "$MODEL_ROOT/config.json" ]]; then
        log "cp 模型到 $MODEL_ROOT（约 1–3 min）"
        cp -r "$MODEL_SRC" "$MODEL_ROOT"
    else
        log "模型已在 $MODEL_ROOT，跳过 cp"
    fi
}

_smoke_test() {
    log "冒烟 curl :$PORT"
    curl -sf "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"Qwen3.5-27B","messages":[{"role":"user","content":"你好，回复一句话"}],"temperature":0.0,"max_tokens":16}'
    echo ""
    log "冒烟 OK"
}

_wait_port() {
    local deadline="${WAIT_TIMEOUT:-600}"
    local i=0
    log "等待 :$PORT 就绪（最多 ${deadline}s）..."
    while (( i < deadline )); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
            || curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
            log "服务就绪 (${i}s)"
            return 0
        fi
        sleep 5
        ((i += 5)) || true
    done
    echo "ERROR: :$PORT 超时未就绪" >&2
    return 1
}

_start_baseline() {
    log "启动 baseline: testdata/start_vllm.sh"
    cd "$TESTDATA"
    MODEL_DIR="$MODEL_ROOT" ./start_vllm.sh
}

_start_optimized() {
    if [[ ! -f "$PROJ/launch.sh" ]]; then
        echo "ERROR: 优化仓库不在 $PROJ，请先 git clone lutinayi_branch" >&2
        exit 1
    fi
    export MODEL_PATH="$MODEL_ROOT"
    export PORT
    export FDU_PHASE="${FDU_PHASE:-1}"
    export PROJ
    log "启动优化版 Phase ${FDU_PHASE}: $PROJ/launch.sh"
    bash "$PROJ/scripts/scnet_start_optimized.sh"
}

case "$MODE" in
  baseline|optimized|opt)
    _install_vllm_wheel
    _copy_model_to_root
    if [[ "$MODE" == "optimized" || "$MODE" == "opt" ]]; then
        echo ""
        echo ">>> 优化版需前台占用终端；请另开终端跑冒烟/测评:"
        echo "    bash $PROJ/scripts/scnet_resume.sh smoke"
        echo "    bash $PROJ/scripts/scnet_resume.sh bench 8-16K"
        echo ""
        _start_optimized
    else
        echo ""
        echo ">>> baseline 需前台占用终端；请另开终端:"
        echo "    bash $SCNET_HOME/2025pra-fdu-fudiankuangxiangqu/scripts/scnet_resume.sh smoke"
        echo ""
        _start_baseline
    fi
    ;;
  smoke)
    _smoke_test
    ;;
  bench)
    tier="${2:-8-16K}"
    _smoke_test
    log "吞吐 $tier x10"
    cd "$TESTDATA"
    ./run_throughput.sh "$tier" 10
    ;;
  prereq|prepare)
    _install_vllm_wheel
    _copy_model_to_root
    log "前置完成；手动启动:"
    echo "  baseline:   cd $TESTDATA && MODEL_DIR=$MODEL_ROOT ./start_vllm.sh"
    echo "  optimized:  bash $PROJ/scripts/scnet_start_optimized.sh"
    ;;
  *)
    echo "Usage: scnet_resume.sh [baseline|optimized|smoke|bench <tier>|prereq]"
    exit 1
    ;;
esac
