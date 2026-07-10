#!/bin/bash
# 镜像评测机 /coursegrader/submit 的 vLLM 编译步骤，完整 log 落盘。
# 官方平台只显示「vLLM build failed」时，在 SCNet 跑本脚本拿详细错误。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="${PROJ_DIR:-$(dirname "$SCRIPT_DIR")}"
RESULTS_DIR="${RESULTS_DIR:-$PROJ_DIR/results}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$RESULTS_DIR/platform_build_${STAMP}.log}"

mkdir -p "$RESULTS_DIR"

log() { echo "[platform_build] $*" | tee -a "$LOG_FILE"; }

_preflight() {
    local ok=1
    if [[ ! -f "$PROJ_DIR/setup.py" ]]; then
        log "FAIL: missing $PROJ_DIR/setup.py"
        ok=0
    fi
    if [[ ! -f "$PROJ_DIR/vllm/__init__.py" ]]; then
        log "FAIL: missing $PROJ_DIR/vllm/__init__.py"
        ok=0
    fi
    if [[ ! -d "$PROJ_DIR/requirements" ]]; then
        log "FAIL: missing $PROJ_DIR/requirements/ (get_requirements 会报错)"
        ok=0
    fi
    if [[ ! -f "$PROJ_DIR/requirements/rocm.txt" ]]; then
        log "FAIL: missing $PROJ_DIR/requirements/rocm.txt"
        ok=0
    fi
    if (( ok == 0 )); then
        log "Preflight 未通过 — 与官方「missing setup.py / build failed」同类问题"
        log "LOG=$LOG_FILE"
        exit 2
    fi
    log "Preflight OK (setup.py + vllm/ + requirements/)"
}

_rocm_env() {
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
    export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
    export HIPCC_COMPILE_FLAGS_APPEND="${HIPCC_COMPILE_FLAGS_APPEND:--O3}"
    if [[ -z "${GPU_ARCH:-}" ]] && command -v rocminfo &>/dev/null; then
        GPU_ARCH="$(rocminfo 2>/dev/null | grep -m1 'Name:' | grep -oE 'gfx[0-9a-z]+' | head -1 || true)"
    fi
    if [[ -n "${GPU_ARCH:-}" ]]; then
        export HIPCC_COMPILE_FLAGS_APPEND="$HIPCC_COMPILE_FLAGS_APPEND --offload-arch=$GPU_ARCH"
        log "GPU_ARCH=$GPU_ARCH"
    fi
    if [[ -n "${ROCM_PATH:-}" ]]; then
        log "ROCM_PATH=$ROCM_PATH"
    fi
}

_run_build() {
    cd "$PROJ_DIR"
    log "cwd=$PWD"
    log "python=$(command -v python3 || command -v python)"
    python3 -c "import torch; print('torch', torch.__version__, 'hip', getattr(torch.version,'hip',None))" 2>&1 | tee -a "$LOG_FILE" || true
    log ">>> python setup.py bdist_wheel (评测机同款)"
    set +e
    python3 setup.py bdist_wheel 2>&1 | tee -a "$LOG_FILE"
    local rc=${PIPESTATUS[0]}
    set -e
    if (( rc != 0 )); then
        log "BUILD FAILED exit=$rc"
        log "完整日志: $LOG_FILE"
        log "请把该文件内容发给队友排查（官方平台通常不展示细节）"
        exit "$rc"
    fi
    log "BUILD OK"
    if ls -1 dist/vllm-*.whl 2>/dev/null | head -1 | tee -a "$LOG_FILE"; then
        log "wheel 已生成于 $PROJ_DIR/dist/"
    fi
    log "LOG=$LOG_FILE"
}

_preflight
_rocm_env
_run_build
