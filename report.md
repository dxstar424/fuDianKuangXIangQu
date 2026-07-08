# 优化方案说明文档

## 1. 技术路线概述

以 **vLLM 0.18.1 (vllm_cscc)** 为基础，通过 `fdu_vllm` 插件合入优化，**不修改 batch scheduler**。

优化维度：KV Cache / Decode 算子(GQA+HIP) / 执行路径(warmup+可选 HIP Graph)

## 2. 多阶段实施状态

| 阶段 | 内容 | 状态 | SCNet 指标 |
|------|------|------|------------|
| 0 | SCNet 环境 + baseline | 脚本就绪 | 待实机填充 |
| 1 | launch/ROCm/compile/warmup | 已实现 | 待实机填充 |
| 2 | GQA + prefix + KV block | 已实现 | 待实机填充 |
| 3 | KV FP8 在线量化 | 已实现 | 待精度门禁 |
| 4 | HIP attention + token 验证 | 骨架+fallback | 待实机填充 |
| 5 | HIP Graph | 默认关闭 | opt-in |
| 6 | 文档/提交 | 已更新 | 待平台提交 |

## 3. 优化点与预期贡献

| 序号 | 优化项 | 阶段 | 预期 |
|------|--------|------|------|
| 1 | launch warmup + rocm env | 1 | TTFT P99 稳定 |
| 2 | GQA einsum decode | 2 | TPOT −5~10% |
| 3 | prefix caching | 2 | 长档 TTFT |
| 4 | KV defrag/tiered blocks | 2 | 长档显存 |
| 5 | KV FP8 在线 | 3 | 长档吞吐 +15% |
| 6 | HIP FlashAttention | 4 | TPOT −15~25% |
| 7 | HIP Graph | 5 | TPOT −5~15% |

## 4. Baseline 数据记录

> 在 SCNet 执行 `bash scripts/record_baseline.sh` 后填入。

| 档位 | TTFT P99 | TPOT P99 | 吞吐 tok/s | SLA |
|------|----------|----------|------------|-----|
| 4-8K (20%) | - | - | - | - |
| 8-16K (50%) | - | - | - | - |
| 16-32K (30%) | - | - | - | - |

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
- 未使用投机解码、权重持久化量化
- KV FP8 仅推理期在线量化，不写盘
- 未启用 `custom_scheduler`
