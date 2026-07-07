# 优化方案说明文档

## 1. 技术路线概述

本文档描述针对 Qwen3.5-27B 在 DCU 加速卡上的推理服务优化方案。

### 总体思路

以 vLLM 0.18.1 为基础框架，从 KV Cache 管理、算子优化、执行路径三个维度
进行系统性优化，在保证 SLA 门限和模型精度的前提下最大化吞吐量。

## 2. 优化措施与贡献分析

### 2.1 KV Cache 与显存管理优化

**策略**：分级块分配 + 碎片整理 + Prefix 缓存 + KV FP8 量化

- **分级块分配**：根据上下文长度选择不同块大小（16/64/256），减少短上下文
  的内碎片和长上下文的块表开销
- **碎片整理**：在空闲块占比超过阈值时主动整理，回收 HBM 连续空间
- **Prefix 缓存**：检测共享前缀，复用 KV 块，减少重复 prefill
- **KV FP8 量化**：在写入 KV Cache 时量化为 FP8（E4M3），读取时反量化，
  节省约 40-50% KV Cache 显存

### 2.2 Decode 阶段算子优化

**策略**：自研 DCU HIP Attention Kernel + 算子融合

- **HIP FlashAttention**：针对 DCU CDNA 架构调优 tile 大小（64×64），
  利用 64-wide wavefront 和 LDS（Local Data Share）
- **GQA 优化**：针对 Qwen 的 32 KV heads 减少 KV 重复扩展开销
- **线性层融合**：QKV 投影融合为单次 GEMM

### 2.3 执行路径优化

**策略**：CUDA Graph 捕获 + 调度批量化 + 异步传输

- **HIP Graph 捕获**：对 Decode 阶段固定 batch size 路径进行图捕获，消除
  逐 step 的 kernel launch 开销
- **调度批量化**：每 4-8 步调度一次，减少 Python 层开销
- **异步 HBM ↔ Host 传输**：利用 DCU 异步 DMA 引擎重叠数据传输

## 3. 优化点汇总表

| 序号 | 优化项 | 类别 | 预期收益 |
|------|--------|------|----------|
| 1 | 分级块分配 | KV Cache | 减少内碎片 15%+ |
| 2 | 碎片整理 | KV Cache | 回收 HBM 碎片 |
| 3 | Prefix 缓存 | KV Cache | Prefill 加速 20%+ |
| 4 | KV FP8 量化 | KV Cache | 节省 KV 显存 40%+ |
| 5 | HIP FlashAttention | 算子 | TPOT 降低 15-25% |
| 6 | GQA 优化 | 算子 | Attention 加速 10% |
| 7 | 算子融合 | 算子 | Kernel launch 减少 |
| 8 | HIP Graph 捕获 | 执行路径 | Decode 开销降低 30%+ |
| 9 | 调度批量化 | 执行路径 | Python 开销降低 |
| 10 | 异步传输 | 执行路径 | 隐藏数据传输延迟 |

## 4. 实验记录

| 日期 | 版本 | TTFT P99 | TPOT P99 | 吞吐量 | SLA | 精度系数 |
|------|------|----------|----------|--------|-----|----------|
| - | baseline | - | - | - | - | 1.00 |

## 5. Baseline 数据记录

> 📋 以下数据将在竞赛平台 DCU 实机上首次运行后填充。

| 档位 | TTFT P50 | TTFT P99 | TPOT P50 | TPOT P99 | 吞吐量 | SLA |
|------|----------|----------|----------|----------|--------|-----|
| short (20%) | - | - | - | - | - | - |
| medium (50%) | - | - | - | - | - | - |
| long (30%) | - | - | - | - | - | - |
| **加权总分** | | | | | **-** | |

### 评测方式

```bash
# 一键 Baseline 评测
bash scripts/run_baseline.sh

# 手动分步
bash baseline/launch.sh &
python scripts/benchmark.py --host localhost --port 8000 --output results/

# 优化版评测
bash launch.sh &
python scripts/benchmark.py --host localhost --port 8000 --output results/

# 对比
python scripts/compare.py results/baseline_xxx.json results/optimized_xxx.json
```
