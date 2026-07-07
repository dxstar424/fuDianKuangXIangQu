#!/bin/bash
# ============================================================
# 编译自定义 HIP Kernel（在 Docker build 时运行）
# 实际 JIT 在运行时由 PyTorch 完成，此脚本做预检查
# ============================================================
set -e

echo "[FDU] Checking HIP compiler availability..."
which hipcc || echo "[FDU] WARNING: hipcc not found - kernels will JIT at runtime"

echo "[FDU] Kernel source validation..."
for f in src/attention/hip_kernels/*.{cpp,hip,cu}; do
    [ -f "$f" ] && echo "[FDU]   Found: $f"
done

echo "[FDU] Compile check complete."
