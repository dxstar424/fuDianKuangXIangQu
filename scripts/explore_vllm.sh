#!/bin/bash
# ============================================================
# 探索 vLLM DCU fork 内部结构，找出所有需要 patch 的位置
# 在容器上运行一次，输出贴给 Codex 即可精确定位
# ============================================================
set -e
VLLM="/public/home/xdzs2026_c415/vllm_cscc/vllm"

echo "========== 1. attention backend 注册机制 =========="
find "$VLLM" -path "*/attention*" -name "*.py" | head -20
echo "---"
grep -rn "get_attn_backend\|register_backend\|AttentionBackend" "$VLLM" --include="*.py" -l | head -10

echo ""
echo "========== 2. KV cache 写入/读取路径 =========="
grep -rn "kv_cache\|key_cache\|value_cache\|k_cache\|v_cache" "$VLLM/v1" --include="*.py" -l | head -15

echo ""
echo "========== 3. Scheduler 类定义 =========="
grep -rn "class Scheduler" "$VLLM" --include="*.py" | head -5

echo ""
echo "========== 4. Block allocator / BlockSpaceManager =========="
grep -rn "class.*Block.*Manager\|class.*Allocator\|class.*Cache.*Manager" "$VLLM" --include="*.py" | head -10

echo ""
echo "========== 5. Attention forward 入口 =========="
grep -rn "def forward" "$VLLM/v1/attention" --include="*.py" | head -10

echo ""
echo "========== 6. GPU model runner (prefill/decode 路径) =========="
find "$VLLM" -name "gpu_model_runner*" -o -name "model_runner*" | head -5
grep -rn "def execute_model\|def _execute_model\|def run" "$VLLM/v1" --include="*.py" | grep -v test | head -10

echo ""
echo "========== 7. FlashAttention / PagedAttention 实现 =========="
find "$VLLM" -name "*flash*" -o -name "*paged*" | head -10

echo ""
echo "========== 8. 编译配置 / CUDA graph 支持 =========="
grep -rn "cudagraph\|cuda_graph\|hip_graph" "$VLLM" --include="*.py" -l | head -10

echo ""
echo "========== 9. __init__.py 模块导出 =========="
head -3 "$VLLM/__init__.py"
echo "---"
grep "from vllm" "$VLLM/v1/__init__.py" 2>/dev/null | head -10

echo ""
echo "========== 10. 平台 start_vllm.sh 内容 =========="
cat /public/home/xdzs2026_c415/testdata/start_vllm.sh 2>/dev/null || echo "not found"
