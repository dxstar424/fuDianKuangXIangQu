#!/bin/bash
# Phase 0: SCNet 环境一键初始化（选手测试调试文档）
set -euo pipefail

VLLM_REPO="${VLLM_REPO:-http://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc.git}"
VLLM_BRANCH="${VLLM_BRANCH:-v0.18.1}"
MODEL_DIR="${MODEL_DIR:-$HOME/Qwen3.5-27B}"
TESTDATA_URL="${TESTDATA_URL:-https://zzefile.scnet.cn:65011/efile/s/d/c2N5MTE1OTkxMDU1OQ==/a927e65672549b46}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Phase 0: SCNet Setup ==="

# 1. 编译 vLLM（含补丁）
bash "$SCRIPT_DIR/compile_vllm.sh"

# 2. 下载模型
if [[ ! -d "$MODEL_DIR/config.json" ]] && [[ ! -f "$MODEL_DIR/config.json" ]]; then
    pip install -q modelscope
    modelscope download --model Qwen/Qwen3.5-27B --local_dir "$MODEL_DIR"
fi

# 3. 下载 testdata
if [[ ! -f "$HOME/testdata/start_vllm.sh" ]]; then
    cd "$HOME"
    curl -f -C - -o testdata.tar.gz "$TESTDATA_URL"
    mkdir -p ./testdata
    tar -xzf testdata.tar.gz -C ./testdata --strip-components=1
    chmod +x testdata/*.sh 2>/dev/null || true
fi

# 4. 复制项目 launch 到 testdata（可选，使用优化版启动）
cp "$PROJ_DIR/launch.sh" "$HOME/testdata/launch_optimized.sh" 2>/dev/null || true

echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. cp -r $MODEL_DIR /root/Qwen3.5-27B   # 每次新容器启动后"
echo "  2. cd ~/testdata && ./start_vllm.sh       # 或 launch_optimized.sh"
echo "  3. bash $SCRIPT_DIR/gate_check.sh quick"
