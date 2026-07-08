# 最容易拿分的改进（已合入 launch.sh）

按官方跑分权重与 SLA 约束，优先做**低风险、保精度**的项：

| 优先级 | 改动 | 作用 | 风险 |
|--------|------|------|------|
| P0 | `GPU_MEMORY_UTILIZATION=0.94` | 8-16K/16-32K 更多 KV，提吞吐 | OOM 时需降到 0.92 |
| P0 | 分档 warmup (`warmup_server.py`) | 稳 TTFT P99，防首条熔断 | 启动慢几分钟 |
| P0 | `FDU_ENABLE_KV_QUANT=0` 默认 | 精度系数保持 1.0 | 长档显存收益延后 |
| P1 | `--enable-prefix-caching` | 长 prefill 降 TTFT | 盲测收益有限但成本低 |
| P1 | `--disable-log-requests/stats` | 减 Python 开销 | 无 |
| P1 | `vllm_env.py` ROCm 变量 | 带宽/launch 微优化 | 无 |
| P2 | `FDU_ENABLE_KV_QUANT=1` | 长档吞吐 +15% 潜力 | 须 `gate_check full` |

## SCNet 快速验证（主攻 50% 权重档）

```bash
# 终端1：启动
bash scripts/scnet_start_optimized.sh

# 终端2：门禁（先看 8-16K）
cd ~/testdata
./run_throughput.sh 8-16K 20
./run_accuracy.sh hotpotqa 10
```

## 评测机

平台执行 `launch.sh` 即可；无需改 locked 参数。
