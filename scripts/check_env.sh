#!/bin/bash
# ============================================================
# 环境检查脚本 — SSH 进容器后运行
# 用法: bash check_env.sh
# ============================================================
set -e

PASS=0
FAIL=0

check() {
    local desc="$1"
    shift
    if "$@" > /dev/null 2>&1; then
        echo "✅ $desc"
        ((PASS++))
    else
        echo "❌ $desc"
        ((FAIL++))
    fi
}

echo "============================================"
echo "  Pra2026 环境检查"
echo "============================================"
echo ""

# --- 基础环境 ---
echo "--- 基础环境 ---"
check "当前用户家目录可写" test -w "$HOME"
check "Python 3.10" python3 --version
check "pip 可用" pip --version
which python3 2>/dev/null && echo "   python3: $(which python3)"

# --- DCU/ROCm ---
echo ""
echo "--- DCU/ROCm ---"
check "ROCm 已安装 (rocm-smi)" which rocm-smi
check "ROCm Profiler (rocprof)" which rocprof
check "hy-smi (DCU 监控)" which hy-smi

# --- HIP 编译器 ---
echo ""
echo "--- HIP 编译工具链 ---"
check "hipcc (HIP 编译器)" which hipcc

# --- Python 深度学习库 ---
echo ""
echo "--- Python 库 ---"
check "torch 已安装" python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
check "torch ROCm 后端" python3 -c "import torch; assert torch.cuda.is_available(); print(f'DCU: {torch.cuda.get_device_name(0)}')"
check "vLLM 已安装" python3 -c "import vllm; print(f'vLLM {vllm.__version__}')"

# --- 模型 ---
echo ""
echo "--- 模型文件 ---"
check "模型存在于 ~/Qwen3.5-27B" test -d "$HOME/Qwen3.5-27B"
check "模型存在于 /root/Qwen3.5-27B" test -d "/root/Qwen3.5-27B"

# --- vLLM 源码 ---
echo ""
echo "--- vLLM 源码 ---"
check "vLLM 源码目录 ~/vllm_cscc" test -d "$HOME/vllm_cscc"

# --- 测试数据 ---
echo ""
echo "--- 测试数据 ~/testdata ---"
check "testdata 目录存在" test -d "$HOME/testdata"

# --- 显存 ---
echo ""
echo "--- DCU 显存状态 ---"
hy-smi 2>/dev/null || echo "   hy-smi 不可用，请检查 ROCm 状态"

echo ""
echo "============================================"
echo "  结果: ${PASS} 通过 / ${FAIL} 失败"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "⚠️  请先在容器内完成竞赛文档第 7-9 步的环境搭建"
    exit 1
else
    echo ""
    echo "🎉 环境就绪，可以开始跑 Baseline！"
fi
