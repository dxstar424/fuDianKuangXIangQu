#!/bin/bash
# 将 patches/vllm_cscc 覆盖层合入 vllm_cscc 源码树
set -euo pipefail

VLLM_DIR="${1:?Usage: apply_vllm_patches.sh <vllm_cscc_dir>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
OVERLAY="$PROJ_DIR/patches/vllm_cscc/overlay"

if [[ ! -d "$OVERLAY" ]]; then
    echo "[apply_vllm_patches] No overlay at $OVERLAY, skip"
    exit 0
fi

echo "[apply_vllm_patches] Copying overlay into $VLLM_DIR"
cp -r "$OVERLAY/." "$VLLM_DIR/"

# 注册 FDU 插件：在 vllm/__init__.py 末尾追加 import（幂等）
INIT_PY="$VLLM_DIR/vllm/__init__.py"
MARKER="# FDU_CSCC_PLUGIN"
if [[ -f "$INIT_PY" ]] && ! grep -q "$MARKER" "$INIT_PY"; then
    cat >> "$INIT_PY" <<'EOF'

# FDU_CSCC_PLUGIN
try:
    import fdu_vllm  # noqa: F401
    fdu_vllm.activate()
except Exception as _fdu_err:
    import logging
    logging.getLogger("fdu_vllm").warning("FDU plugin not activated: %s", _fdu_err)
EOF
fi

# 将 fdu_vllm 包链接到 vllm 可 import 路径
FDU_SRC="$PROJ_DIR/src/fdu_vllm"
if [[ -d "$FDU_SRC" ]]; then
    mkdir -p "$VLLM_DIR/fdu_vllm"
    rsync -a --delete "$FDU_SRC/" "$VLLM_DIR/fdu_vllm/" 2>/dev/null || cp -r "$FDU_SRC/." "$VLLM_DIR/fdu_vllm/"
fi

# v0.9.0: patch pyproject.toml to include fdu_vllm in pip install
PYPROJECT="$VLLM_DIR/pyproject.toml"
if [[ -f "$PYPROJECT" ]]; then
    if grep -q 'include = \["vllm\*"\]' "$PYPROJECT"; then
        sed -i 's/include = \["vllm\*"\]/include = ["vllm*", "fdu_vllm*"]/' "$PYPROJECT"
        echo "[apply_vllm_patches] pyproject.toml: added fdu_vllm* to packages"
    fi
fi

echo "[apply_vllm_patches] Done"
