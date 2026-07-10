#!/bin/bash
# 将 vllm_cscc 源码铺到仓库根目录，满足评测机 /coursegrader/submit/setup.py + vllm/
set -euo pipefail

VLLM_REPO="${VLLM_REPO:-http://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc.git}"
VLLM_BRANCH="${VLLM_BRANCH:-v0.18.1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
STAGING="${STAGING:-$PROJ_DIR/.vllm_cscc_staging}"

echo "[prepare_submit] PROJ=$PROJ_DIR"

if [[ -d "${HOME}/vllm_cscc/.git" ]]; then
    echo "[prepare_submit] reuse ${HOME}/vllm_cscc"
    rm -rf "$STAGING"
    cp -a "${HOME}/vllm_cscc" "$STAGING"
elif [[ -d "$STAGING/.git" ]]; then
    echo "[prepare_submit] reuse staging $STAGING"
else
    echo "[prepare_submit] cloning $VLLM_REPO ($VLLM_BRANCH)"
    rm -rf "$STAGING"
    git clone -b "$VLLM_BRANCH" --depth 1 "$VLLM_REPO" "$STAGING"
fi

bash "$SCRIPT_DIR/apply_vllm_patches.sh" "$STAGING"

for item in setup.py pyproject.toml CMakeLists.txt MANIFEST.in use_existing_torch.py vllm cmake csrc fdu_vllm; do
    rm -rf "$PROJ_DIR/$item"
    cp -a "$STAGING/$item" "$PROJ_DIR/$item"
done

echo "[prepare_submit] OK: root has setup.py + vllm/"
echo "  git add setup.py pyproject.toml CMakeLists.txt MANIFEST.in use_existing_torch.py vllm cmake csrc fdu_vllm"
echo "  git commit -m 'chore: vendor vllm_cscc source at repo root for platform'"
echo "  git push origin <your-branch>"
