# 最容易拿分的改进（已合入 launch.sh）

> 深度指南（必须 vs 冲刺）：[deep_optimization_guide.md](./deep_optimization_guide.md)  
> 总规划：[optimization_roadmap.md](./optimization_roadmap.md)  
> 官方解读：[official_guidance_interpretation.md](./official_guidance_interpretation.md)  
> **DCU 实测**：[dcu_decode_benchmark_interpretation.md](./dcu_decode_benchmark_interpretation.md)

按官方 **2026-07-09 指导** + 跑分权重，优化分两阶段打：

- **Prefill → TTFT**：warmup、prefix cache、显存预算
- **Decode → TPOT/吞吐**：GQA、KV 块策略、Graph capture、KV FP8（须融合）

## P0 — 显存与 TTFT（立即）

| 改动 | 作用 | 对应官方方向 |
|------|------|--------------|
| `GPU_MEMORY_UTILIZATION=0.94` | 长档 KV 更充裕，提吞吐 | 显存精细化管理 |
| 分档 warmup | 稳 TTFT P99，防 SLA 熔断 | Prefill / 首字体验 |
| `FDU_ENABLE_KV_QUANT=0` 默认 | 精度系数 1.0 | KV 量化双刃剑，先不开 |

## P1 — KV 与 Decode 访存（baseline 后）

| 改动 | 作用 | 对应官方方向 |
|------|------|--------------|
| KV defrag / tiered blocks | 降碎片、稳长档 | **PagedAttention 块分配（官方最强调）** |
| GQA einsum decode | 减 KV 读放大，降 TPOT | Decode 访存瓶颈 |
| 长档 TPOT profiling | 定位 KV 读取路径 | 不能只看吞吐 |
| `--enable-prefix-caching` | 降重复 prefill 的 TTFT | KV 命中率 |
| `--disable-log-requests/stats` | 减 Python 开销 | 执行路径 |
| `vllm_env.py` ROCm 变量 | 带宽/launch 微优化 | DCU 适配 |

## P2 — 调度与 KV 量化（有数据后再开）

| 改动 | 作用 | 门禁 |
|------|------|------|
| `FDU_ENABLE_HIP_GRAPH=1` | Graph capture 减 token 调度开销 | 官方推荐；TPOT↓ 且 SLA 过 |
| KV FP8 **算子融合版** | 降 KV 显存 | **禁止独立反量化**；精度 Δ≤1% + 净 TPOT 收益 |

## P3 — 仅 profiling 证明需要时

| 改动 | 条件 |
|------|------|
| HIP FlashAttention（**prefill / TTFT**） | 长档 TTFT 主因是 O(S²) 注意力（见 DCU 实测 §2.2） |

## 停止投入（DCU 实测证伪）

| 方向 | 原因 |
|------|------|
| Decode GEMV 双缓冲 / 提 wave 占用 | 已在 HBM 101%，双缓冲 ±1.5% |
| 持久化权重量化（讲义 49→40ms） | 赛题红线 |

## SCNet 验证（主攻 8-16K · 50% 权重）

```bash
# 终端1：启动
bash scripts/scnet_start_optimized.sh

# 终端2：baseline 吞吐 + 精度
cd ~/testdata
./run_throughput.sh 8-16K 20
./run_accuracy.sh hotpotqa 10

# 终端3：长档 TPOT 分析（baseline 跑通后）
# 记录 decode 步 KV read 时间占比，填入 report.md
```

## 提交前核对（官方强调）

- [ ] `docs/env_vars.md` 与 `launch.sh` 实际 env 一致
- [ ] `report.md` 每项优化有 **±X% 吞吐 / ±Y ms TPOT** 量化贡献
- [ ] KV 量化未删历史 token、未生成低精度权重缓存
- [ ] 未改 batch scheduler / 自回归解码语义

## 评测机

平台执行 `launch.sh` 即可；无需改 locked 参数。
