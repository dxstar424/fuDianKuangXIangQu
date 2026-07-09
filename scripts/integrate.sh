#!/bin/bash
# ============================================================
# 一键集成 FDU 优化 → vLLM DCU fork
# 1. 编译 HIP FlashAttention kernel
# 2. 注册为 vLLM attention backend
# 3. 修改 scheduler 加入长度感知
# 4. 重编译 vLLM
# ============================================================
set -e
VLLM=/public/home/xdzs2026_c415/vllm_cscc
SRC=/public/home/xdzs2026_c415/src
BUILD=$VLLM/build/fdu_kernels

echo "=== Step 1: 编译 HIP kernel ==="
mkdir -p $BUILD
hipcc -O3 --offload-arch=gfx942 -std=c++17 -fPIC -shared \
    -o $BUILD/dcu_flash_attn.so \
    $SRC/attention/hip_kernels/dcu_flash_attn.cpp
echo "Kernel compiled: $BUILD/dcu_flash_attn.so"

echo "=== Step 2: 创建 DCU attention backend ==="
cat > $VLLM/vllm/v1/attention/backends/dcu_attn.py << 'PYEOF'
"""DCU FlashAttention backend for vLLM V1."""
import ctypes
import torch
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

class DCUFlashAttentionBackend(FlashAttentionBackend):
    """FlashAttention + DCU HIP kernel fallback."""

    @staticmethod
    def get_name() -> str:
        return "dcu_flash_attn"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kernel = None
        self._try_load_kernel()

    def _try_load_kernel(self):
        try:
            import os
            path = os.path.join(os.path.dirname(__file__), "../../../build/fdu_kernels/dcu_flash_attn.so")
            if os.path.exists(path):
                lib = ctypes.CDLL(path)
                lib.dcu_flash_attn_forward.argtypes = [ctypes.c_void_p]*9 + [ctypes.c_void_p]
                lib.dcu_flash_attn_forward.restype = ctypes.c_int
                self._kernel = lib
                print(f"[FDU] DCU FlashAttention kernel loaded: {path}")
        except Exception as e:
            print(f"[FDU] Kernel load skipped: {e}")
PYEOF
echo "Backend created"

echo "=== Step 3: 注册 backend ==="
SEL=$VLLM/vllm/v1/attention/selector.py
if ! grep -q "dcu_attn" $SEL; then
    python3 -c "
lines = open('$SEL').readlines()
new = []
for line in lines:
    new.append(line)
    if 'from vllm.v1.attention.backends.flash_attn' in line and 'dcu' not in line:
        new.append('from vllm.v1.attention.backends.dcu_attn import DCUFlashAttentionBackend  # FDU\n')
open('$SEL', 'w').writelines(new)
"
    echo "Backend registered in selector"
fi

echo "=== Step 4: Scheduler 长度感知 ==="
SCHED=$VLLM/vllm/v1/core/sched/scheduler.py
if ! grep -q "FDU_LENGTH_AWARE" $SCHED; then
    python3 -c "
lines = open('$SCHED').readlines()
new = []
for line in lines:
    if 'def schedule(self' in line and 'FDU' not in line:
        indent = line[:len(line) - len(line.lstrip())]
        new.append(f'{indent}# FDU_LENGTH_AWARE: sort waiting by prompt length (short first)\n')
        new.append(f'{indent}if __import__(\"os\").environ.get(\"FDU_SCHEDULER_POLICY\") == \"length_aware\":\n')
        new.append(f'{indent}    self.waiting.sort(key=lambda r: len(getattr(r, \"prompt_token_ids\", [])))\n')
    new.append(line)
open('$SCHED', 'w').writelines(new)
"
    echo "Scheduler patched"
fi

echo "=== Step 5: 重编译 vLLM ==="
cd $VLLM
python setup.py bdist_wheel
pip install dist/vllm-*.whl --no-deps --force-reinstall

echo "=== 集成完成 ==="
echo "启动: FDU_SCHEDULER_POLICY=length_aware vllm serve /root/Qwen3.5-27B ..."
