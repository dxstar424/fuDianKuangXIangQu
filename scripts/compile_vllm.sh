#!/bin/bash
# 克隆 vllm_cscc、应用 FDU 补丁、编译 wheel
set -euo pipefail

VLLM_REPO="${VLLM_REPO:-http://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc.git}"
VLLM_BRANCH="${VLLM_BRANCH:-v0.18.1}"
VLLM_DIR="${VLLM_DIR:-$HOME/vllm_cscc}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

# ROCm 编译优化
export HIPCC_COMPILE_FLAGS_APPEND="${HIPCC_COMPILE_FLAGS_APPEND:--O3}"
if [[ -z "${GPU_ARCH:-}" ]]; then
    if command -v rocminfo &>/dev/null; then
        GPU_ARCH="$(rocminfo | grep -m1 'Name:' | grep gfx | awk '{print $2}' || true)"
    fi
fi
if [[ -n "${GPU_ARCH:-}" ]]; then
    export HIPCC_COMPILE_FLAGS_APPEND="$HIPCC_COMPILE_FLAGS_APPEND --offload-arch=$GPU_ARCH"
fi

echo "[compile_vllm] VLLM_DIR=$VLLM_DIR branch=$VLLM_BRANCH"

if [[ ! -d "$VLLM_DIR/.git" ]]; then
    git clone -b "$VLLM_BRANCH" --depth 1 "$VLLM_REPO" "$VLLM_DIR"
fi

bash "$SCRIPT_DIR/apply_vllm_patches.sh" "$VLLM_DIR"

cd "$VLLM_DIR"
python setup.py bdist_wheel
pip install --force-reinstall dist/vllm-*.whl --no-deps

echo "[compile_vllm] Installed: $(pip show vllm | grep Version)"
