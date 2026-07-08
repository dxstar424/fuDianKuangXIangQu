#!/bin/bash
# ============================================================
# HIP Kernel 编译脚本
# 用 hipcc 独立编译 .cpp → .so，再由 Python ctypes 加载
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
KERNEL_DIR="$PROJ_DIR/src/attention/hip_kernels"
BUILD_DIR="$PROJ_DIR/build/kernels"

# --- DCU 架构选择 ---
# 默认 gfx942 (CDNA3), 如需 CDNA2 改为 gfx90a
GFX_ARCH="${GFX_ARCH:-gfx942}"

echo "[FDU] HIP Kernel Compiler"
echo "[FDU]   Source:  $KERNEL_DIR"
echo "[FDU]   Build:   $BUILD_DIR"
echo "[FDU]   Arch:    $GFX_ARCH"

# --- 检查 hipcc ---
if ! command -v hipcc &>/dev/null; then
    echo "[FDU] ERROR: hipcc not found. Is DTK installed?"
    echo "[FDU]   Path: /opt/rocm/bin/hipcc  or  /opt/dtk/bin/hipcc"
    exit 1
fi

echo "[FDU]   hipcc:   $(which hipcc)"
echo "[FDU]   version: $(hipcc --version 2>&1 | head -1)"

mkdir -p "$BUILD_DIR"

# --- 编译每个 kernel 源文件 ---
KERNELS=(
    "dcu_flash_attn"
)

for kernel in "${KERNELS[@]}"; do
    SRC="$KERNEL_DIR/${kernel}.cpp"
    OUT="$BUILD_DIR/${kernel}.so"

    if [ ! -f "$SRC" ]; then
        echo "[FDU] WARNING: Source not found: $SRC"
        continue
    fi

    echo "[FDU] Compiling: ${kernel}.cpp → ${kernel}.so"

    hipcc \
        -O3 \
        -D__HIP_PLATFORM_AMD__ \
        --offload-arch="$GFX_ARCH" \
        -std=c++17 \
        -fPIC \
        -shared \
        -Wno-deprecated-declarations \
        -o "$OUT" \
        "$SRC"

    if [ -f "$OUT" ]; then
        SIZE=$(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT" 2>/dev/null)
        echo "[FDU]   ✓  Built: ${kernel}.so ($SIZE bytes)"
    else
        echo "[FDU]   ✗  Build failed for ${kernel}"
        exit 1
    fi
done

echo "[FDU] All kernels compiled successfully."
echo "[FDU] Output: $BUILD_DIR/"
ls -la "$BUILD_DIR"/
