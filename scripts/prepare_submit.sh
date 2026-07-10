#!/bin/bash
# 将 vllm_cscc 源码纳入仓库，供评测机离线编译（无网络时必需）
set -euo pipefail

VLLM_REPO="${VLLM_REPO:-http://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc.git}"
VLLM_BRANCH="${VLLM_BRANCH:-v0.18.1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
VLLM_DIR="${VLLM_DIR:-$PROJ_DIR/vllm_cscc}"

echo "[prepare_submit] PROJ=$PROJ_DIR"
echo "[prepare_submit] VLLM_DIR=$VLLM_DIR branch=$VLLM_BRANCH"

if [[ -d "$VLLM_DIR/.git" ]]; then
    echo "[prepare_submit] reuse existing $VLLM_DIR"
elif [[ -d "${HOME}/vllm_cscc/.git" ]]; then
    echo "[prepare_submit] copy from ${HOME}/vllm_cscc"
    rm -rf "$VLLM_DIR"
    cp -a "${HOME}/vllm_cscc" "$VLLM_DIR"
else
    echo "[prepare_submit] cloning $VLLM_REPO"
    rm -rf "$VLLM_DIR"
    git clone -b "$VLLM_BRANCH" --depth 1 "$VLLM_REPO" "$VLLM_DIR"
fi

bash "$SCRIPT_DIR/apply_vllm_patches.sh" "$VLLM_DIR"

if [[ ! -f "$PROJ_DIR/setup.py" ]]; then
    echo "[prepare_submit] ERROR: missing root setup.py" >&2
    exit 1
fi

echo "[prepare_submit] OK — next on dev machine:"
echo "  git add setup.py vllm_cscc scripts/apply_vllm_patches.sh src/fdu_vllm patches/"
echo "  git commit -m 'chore: vendor vllm_cscc for platform submit'"
echo "  git push origin lutinayi_branch"
