# 优化方案说明文档

> 赛程规划：[docs/optimization_roadmap.md](docs/optimization_roadmap.md)  
> 冲刺攻略：[docs/sprint_strategy_0711.md](docs/sprint_strategy_0711.md)  
> 官方解读：[docs/official_guidance_interpretation.md](docs/official_guidance_interpretation.md)  
> 最有把握清单：[docs/easy_scoring.md](docs/easy_scoring.md)

## 1. 技术路线概述

以 **vLLM 0.18.1 (vllm_cscc)** 为基础，通过 `fdu_vllm` 插件合入优化，**不修改 batch scheduler**。

优化维度（对齐官方 2026-07-09 指导 + 7/11 少次多阶段）：

| 阶段 | 瓶颈 | 优化方向 |
|------|------|----------|
| Prefill | 算力（GEMM） | prefix cache、显存 0.94 → **TTFT** |
| Decode | 带宽 + KV 读 | **GQA wrap**、可选 Graph → **TPOT / 吞吐** |

**资料过滤**：禁用投机解码 / 剪枝 / 换框架；可借 FlashQLA/FA 融合思想与官方 KV/Graph。

## 2. 多阶段实施状态

| 阶段 | 内容 | 状态 | SCNet / 平台 |
|------|------|------|--------------|
| S1 | recover：stock + gpu=0.94 + eager + warmup=0 | **代码已锁** | 平台 59.97，待重提 ≥65 |
| S2 | `ENFORCE_EAGER=0` / 单档 warmup A/B | 脚本就绪 | 待测 |
| S3 | GQA selector wrap（深接） | **已接线**，默认关（须 USE_FDU_SERVER=1） | 待实机 |
| S4 | HIP Graph model-runner 补丁 | 默认关，opt-in | 待 S3 净增 |
| — | KV FP8 / defrag / FlashQLA 整库 | **默认不做** | — |

## 3. 优化点与预期贡献

> 官方要求：对每项优化做**量化贡献分析**。SCNet/平台实测后填入「实测贡献」列。

### 3.1 S1 Recover（相对 7/10 负优化配置）

| 序号 | 优化项 | 默认 | 主攻指标 | 预期 | 实测贡献 |
|------|--------|------|----------|------|----------|
| 1.1 | `gpu_memory_utilization` **0.94** | 开 | 16–32K 纠负 | 避免 OOM/负提升 | 待填 |
| 1.2 | stock `api_server` | 开 | 稳定性 | 去掉未验证插件 | 待填 |
| 1.3 | `--enforce-eager` | 开 | 避 Graph 负优化 | S2 再 A/B 关 | 待填 |
| 1.4 | `DO_WARMUP=0` | 开 | 启动时间 | 平台不依赖长 dummy | — |
| 1.5 | `--enable-prefix-caching` | 开 | TTFT | 共享前缀 | 待填 |
| 1.6 | disable-log-requests/stats | 开 | TPOT 微降 | 减 Python I/O | 待填 |
| 1.7 | `FDU_ENABLE_KV_QUANT=0` | 强制 | 精度系数 | 保 k=1.0 | — |
| 1.8 | bf16 + 合规 served-name | 开 | 稳定性 | 与官方权重一致 | — |

### 3.2 S2–S4（门禁后逐项开）

| 序号 | 优化项 | 阶段 | 主攻指标 | 预期 | 实测贡献 |
|------|--------|------|----------|------|----------|
| 2a | `ENFORCE_EAGER=0`（原生 graph） | S2 | TPOT | 减 launch | 待 A/B |
| 2b | 单档 warmup 8–16K | S2 | TTFT P99 | 防尖刺 | 待 A/B |
| 3 | GQA selector wrap | S3 | TPOT / 8–16K | 减 KV 物化 | 待填 |
| 4 | HIP Graph（自研钩子） | S4 | TPOT P99 | 调度开销↓ | 待填 |
| — | KV FP8 / defrag | — | — | 默认不做 | — |

## 4. Baseline 数据记录

> SCNet stock baseline 待重跑。平台实测见 [`docs/baseline_result.pdf`](docs/baseline_result.pdf)。

### 4.1 SCNet stock baseline（start_vllm.sh）

| 档位 | TTFT P99 | TPOT P99 | 吞吐 tok/s | SLA |
|------|----------|----------|------------|-----|
| 4-8K (20%) | — | — | — | 待测 |
| 8-16K (50%) | — | — | — | 待测 |
| 16-32K (30%) | — | — | — | 待测 |

### 4.2 本队平台实测（lutinayi_branch · 2026-07-10 22:14）

| 档位 | 吞吐 tok/s | SLA扣分 | 精度扣分 |
|------|------------|---------|----------|
| 4-8K (20%) | 12.92 | 0 | 0 |
| 8-16K (50%) | 10.04 | 0 | 0 |
| 16-32K (30%) | 5.77 | 0 | 0 |
| **最终得分** | **59.97** | 0 | 0 |

> 深度解读见 [`docs/scoring_and_results_interpretation.md`](docs/scoring_and_results_interpretation.md)。  
> 「富贵花开」84.74 **不是本队账号**，仅作排行榜外部参考。

### 4.3 与榜首差距（相对本队 7/10）

| 档位 | 榜首 | 本队 | 差距 |
|------|------|------|------|
| 8K-16K (50%) | 19.51 | 10.04 | -9.47 |
| 4K-8K (20%) | 21.42 | 12.92 | -8.50 |
| 16K-32K (30%) | 15.05 | 5.77 | -9.28 |

## 5. 评测与门禁命令

```bash
bash scripts/verify_phase1_config.sh
bash scripts/scnet_start_optimized.sh
bash scripts/gate_check.sh quick
bash launch.sh
```

## 6. 合规声明

- 未修改 `max-num-seqs`、`max-num-batched-tokens`、batch scheduler
- 未使用投机解码、权重持久化量化、低精度权重缓存
- KV FP8 仅推理期在线量化，不写盘；**保留全部历史 KV**
- Graph capture 不改变自回归解码语义
- 未启用 `custom_scheduler`
- 自定义环境变量见 [docs/env_vars.md](docs/env_vars.md)
