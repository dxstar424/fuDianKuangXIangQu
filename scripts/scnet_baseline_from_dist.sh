#!/bin/bash
# 从 vllm_cscc/dist 目录继续完成 PDF 文档步骤 7-11 的 baseline 测评
# 用法: 在 SCNet 容器内，cd ~/vllm_cscc/dist 后执行:
#   bash ~/2025pra-fdu-fudiankuangxiangqu/scripts/scnet_baseline_from_dist.sh
# 或复制本脚本内容到容器执行
set -euo pipefail

VLLM_DIST="${VLLM_DIST:-$HOME/vllm_cscc/dist}"
MODEL_DIR="${MODEL_DIR:-$HOME/Qwen3.5-27B}"
TESTDATA="${TESTDATA:-$HOME/testdata}"
TESTDATA_URL="${TESTDATA_URL:-https://zzefile.scnet.cn:65011/efile/s/d/c2N5MTE1OTkxMDU1OQ==/a927e65672549b46}"
RESULT_FILE="${RESULT_FILE:-$HOME/baseline_results_$(date +%Y%m%d_%H%M%S).txt}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Step 7: 安装 vLLM wheel ──
log "Step 7: 安装 vLLM wheel ..."
cd "$VLLM_DIST"
WHEEL=$(ls -1 vllm-*.whl 2>/dev/null | head -1)
if [[ -z "$WHEEL" ]]; then
    echo "ERROR: 未找到 vllm-*.whl，请先执行 python setup.py bdist_wheel"
    exit 1
fi
pip install "$WHEEL" --no-deps
python -c "import vllm; print('vLLM version:', vllm.__version__)"

# ── Step 8: 下载模型 ──
log "Step 8: 检查/下载 Qwen3.5-27B ..."
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    pip install -q modelscope
    modelscope download --model Qwen/Qwen3.5-27B --local_dir "$MODEL_DIR"
fi
log "模型路径: $MODEL_DIR"

# ── Step 9: 下载 testdata ──
log "Step 9: 检查/下载 testdata ..."
if [[ ! -f "$TESTDATA/start_vllm.sh" ]]; then
    cd "$HOME"
    curl -f -C - -o testdata.tar.gz "$TESTDATA_URL"
    mkdir -p ./testdata
    tar -xzf testdata.tar.gz -C ./testdata --strip-components=1
    chmod +x testdata/*.sh
fi
ls "$TESTDATA"/*.jsonl "$TESTDATA"/*.sh

# ── Step 10: 启动 vLLM 服务 ──
log "Step 10: 复制模型到 /root 并启动服务 ..."
sudo mkdir -p /root 2>/dev/null || true
if [[ ! -d /root/Qwen3.5-27B ]]; then
    cp -r "$MODEL_DIR" /root/Qwen3.5-27B
fi

cd "$TESTDATA"
# 若已有服务在跑则跳过启动
if curl -sf http://127.0.0.1:8001/health > /dev/null 2>&1; then
    log "服务已在 8001 端口运行，跳过启动"
else
    log "启动 start_vllm.sh（后台），首次约需 10 分钟 ..."
    nohup ./start_vllm.sh > "$HOME/vllm_server.log" 2>&1 &
    SERVER_PID=$!
    log "服务 PID=$SERVER_PID，日志: $HOME/vllm_server.log"
fi

# 等待健康检查（最多 20 分钟）
log "等待服务就绪 ..."
for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8001/v1/models > /dev/null 2>&1; then
        log "服务就绪 (${i}×5s)"
        break
    fi
    if [[ $i -eq 120 ]]; then
        echo "ERROR: 服务启动超时，查看 tail -50 $HOME/vllm_server.log"
        tail -50 "$HOME/vllm_server.log" || true
        exit 1
    fi
    sleep 5
done

# 单次推理验证
log "curl 验证 ..."
curl -s http://127.0.0.1:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
  "model": "Qwen3.5-27B",
  "messages": [{"role": "user", "content": "你好，简单回复一句话。"}],
  "temperature": 0.0,
  "max_tokens": 64
}' | head -c 500
echo ""

# ── Step 10: 吞吐测试（三档各 10 条快速 baseline）──
log "Step 10b: 吞吐 baseline 测试 ..."
{
    echo "========================================"
    echo "Baseline 吞吐测评 $(date -Iseconds)"
    echo "vLLM: $(python -c 'import vllm; print(vllm.__version__)')"
    echo "========================================"
    echo ""
    echo "--- 4-8K (10条) ---"
    ./run_throughput.sh 4-8K 10
    echo ""
    echo "--- 8-16K (10条) ---"
    ./run_throughput.sh 8-16K 10
    echo ""
    echo "--- 16-32K (10条) ---"
    ./run_throughput.sh 16-32K 10
} | tee "$RESULT_FILE"

log "吞吐结果已保存: $RESULT_FILE"

# ── Step 11: 精度测试（各数据集 10 条）──
log "Step 11: 精度 baseline 测试 ..."
{
    echo ""
    echo "========================================"
    echo "Baseline 精度测评 $(date -Iseconds)"
    echo "========================================"
    ./run_accuracy.sh hotpotqa 10
    ./run_accuracy.sh gov_report 10
    ./run_accuracy.sh retrieval_multi_point 10
    ./run_accuracy.sh aggregation_keyword_aggregation 10
} | tee -a "$RESULT_FILE"

log "全部 baseline 结果: $RESULT_FILE"
echo ""
echo "请将以下指标填入 report.md:"
echo "  吞吐: Output throughput, TTFT P99, TPOT P99 (三档)"
echo "  精度: hotpotqa F1, gov_report ROUGE, retrieval/aggregation Accuracy"
