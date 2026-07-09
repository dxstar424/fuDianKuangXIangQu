# 优化方案说明文档

> 赛程规划：[docs/optimization_roadmap.md](docs/optimization_roadmap.md)  
> 官方解读：[docs/official_guidance_interpretation.md](docs/official_guidance_interpretation.md)

## 1. 技术路线概述

以 **vLLM 0.18.1 (vllm_cscc)** 为基础，通过 `fdu_vllm` 插件合入优化，**不修改 batch scheduler**。

优化维度（对齐官方 2026-07-09 指导）：

| 阶段 | 瓶颈 | 优化方向 |
|------|------|----------|
| Prefill | 算力（GEMM） | warmup、prefix cache、显存预算 → **TTFT** |
| Decode | 带宽 + KV 读 | GQA、KV 块/defrag、Graph capture、KV FP8 融合 → **TPOT / 吞吐** |

## 2. 多阶段实施状态

| 阶段 | 内容 | 状态 | SCNet 指标 |
|------|------|------|------------|
| 0 | SCNet 环境 + baseline | 脚本就绪 | 待实机填充 |
| 1 | launch/ROCm/compile/warmup | **代码完成** | 待 SCNet 门禁 |
| 2 | GQA + prefix + KV block | 已实现 | 待实机填充 |
| 3 | KV FP8 在线量化 | 已实现 | 待精度门禁 |
| 4 | HIP attention + token 验证 | 骨架+fallback | 待实机填充 |
| 5 | HIP Graph | 默认关闭 | opt-in |
| 6 | 文档/提交 | 已更新 | 待平台提交 |

## 3. 优化点与预期贡献

> 官方要求：对每项优化做**量化贡献分析**。SCNet 实测后填入「实测贡献」列。  
> **最有把握项清单**：[docs/easy_scoring.md](docs/easy_scoring.md)（Phase 1 默认开启）。

### 3.1 Phase 1 最有把握项（相对 stock gpu=0.92）

| 序号 | 优化项 | 默认 | 主攻指标 | 预期 | 实测贡献 |
|------|--------|------|----------|------|----------|
| 1.1 | `gpu_memory_utilization` 0.94 | 开 | 8–16K / 16–32K 吞吐 | 长档 KV 更充裕 | 待填 |
| 1.2 | 分档 warmup（8–16K 优先） | 开 | TTFT P99 | 防首条 SLA 熔断 | 待填 |
| 1.3 | `--enable-prefix-caching` | 开 | TTFT | 共享前缀降 prefill | 待填 |
| 1.4 | disable-log-requests/stats | 开 | TPOT 微降 | 减 Python I/O | 待填 |
| 1.5 | ROCm env（SDMA 等） | 开 | Decode 稳定 | 带宽/分配 | 待填 |
| 1.6 | `FDU_ENABLE_KV_QUANT=0` | 强制 | 精度系数 | 保 k=1.0 | 待填 |
| 1.7 | bf16 + 合规 served-name | 开 | 稳定性 | 与官方权重一致 | — |

### 3.2 Phase 2+（门禁后再开，默认关）

| 序号 | 优化项 | 阶段 | 主攻指标 | 预期 | 实测贡献 |
|------|--------|------|----------|------|----------|
| 2 | GQA einsum decode | 2 | TPOT P99 | TPOT −5~10% | 待填 |
| 3 | KV defrag/tiered blocks | 2 | TPOT、长档吞吐 | 降碎片、稳 KV 读 | 待填 |
| 4 | HIP Graph capture | 2b | TPOT P99 | 调度开销 −5~15% | 待填 |
| 5 | KV FP8 融合（非独立反量化） | 3 | 长档吞吐 | 显存↓ → 吞吐↑ | 待 A/B |
| 6 | HIP FlashAttention | 4 | Prefill/TTFT | profiling 后 | 暂缓 |

## 4. Baseline 数据记录

> SCNet stock baseline 待重跑（2026-07-08 自测失败：服务未启动）。  
> 平台实测见 [`docs/baseline_result.pdf`](docs/baseline_result.pdf)。

### 4.1 SCNet stock baseline（start_vllm.sh）

| 档位 | TTFT P99 | TPOT P99 | 吞吐 tok/s | SLA |
|------|----------|----------|------------|-----|
| 4-8K (20%) | — | — | — | 待测 |
| 8-16K (50%) | — | — | — | 待测 |
| 16-32K (30%) | — | — | — | 待测 |

### 4.2 竞赛平台实测（富贵花开 · 2026-07-06）

| 档位 | 吞吐 tok/s | SLA扣分 | 精度扣分 |
|------|------------|---------|----------|
| 4-8K (20%) | 18.37 | 0 | 0 |
| 8-16K (50%) | 16.65 | 0 | 0 |
| 16-32K (30%) | 13.49 | 0 | 0 |
| **最终得分** | **84.74 (#26)** | 0 | 0 |

### 4.3 与榜首差距（豆包F4 · #1）

| 档位 | 榜首 | 我们 | 差距 |
|------|------|------|------|
| 8K-16K (50%) | 19.51 | 16.65 | -2.86 |
| 4K-8K (20%) | 21.42 | 18.37 | -3.05 |
| 16K-32K (30%) | 15.05 | 13.49 | -1.56 |

## 5. 评测与门禁命令

```bash
# SCNet Phase 0
bash scripts/scnet_setup.sh
bash scripts/record_baseline.sh

# 每阶段门禁
bash scripts/gate_check.sh quick
bash scripts/gate_check.sh full

# 编译 vLLM + 补丁
bash scripts/compile_vllm.sh

# 启动（评测机 / 本地）
bash launch.sh

# Token 一致性（Phase 4）
python scripts/verify_token_consistency.py --baseline-port 8000 --opt-port 8001
```

## 6. 合规声明

- 未修改 `max-num-seqs`、`max-num-batched-tokens`、batch scheduler
- 未使用投机解码、权重持久化量化、低精度权重缓存
- KV FP8 仅推理期在线量化，不写盘；**保留全部历史 KV，不 skip/delete**
- KV 量化目标为算子融合，避免独立反量化抵消收益
- Graph capture 不改变自回归解码语义
- 未启用 `custom_scheduler`
- 自定义环境变量见 [docs/env_vars.md](docs/env_vars.md)
