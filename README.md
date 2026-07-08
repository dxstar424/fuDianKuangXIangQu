# Pra2026 — 基于国产加速卡的 Qwen 大模型推理服务优化

> 🏆 2026 年全国大学生计算机系统能力大赛 · 智能计算创新设计赛（先导杯）
>
> 赛题：基于国产加速卡（DCU）的千问大模型推理服务优化

[![Python](https://img.shields.io/badge/Python-3.10.12-blue)](https://www.python.org/)
[![vLLM](https://img.shields.io/badge/vLLM-0.18.1-green)](https://github.com/vllm-project/vllm)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-orange)](https://pytorch.org/)
[![ROCm](https://img.shields.io/badge/ROCm-DCU-red)](https://www.amd.com/en/products/software/rocm.html)

## 概述

本项目以 **Qwen3.5-27B (bf16)** 模型为优化对象，在 **vLLM 0.18.1** 推理框架与 **DCU 加速卡**（单卡）的统一容器环境下，进行系统级推理性能优化。核心目标：**在满足 TTFT / TPOT SLA 硬性约束的前提下，最大化输出吞吐量**。

### 优化策略全景

```
最终得分 = 吞吐量得分 × 精度系数
    │
    ├── KV Cache 优化 ──── 分级块分配 · 碎片整理 · KV FP8 量化 · Prefix 缓存
    ├── Decode 算子优化 ── HIP FlashAttention · GQA 优化 · MFMA 加速 · LDS 利用
    └── 执行路径优化 ──── HIP Graph 捕获 · 调度批量化 · 异步 SDMA 传输
```

## 项目结构

```
Pra2026/
├── baseline/                         # Baseline 纯净环境（对照基准）
│   ├── launch.sh                     #    stock vLLM，无自定义模块
│   └── config.yaml                   #    默认参数
├── launch.sh                         # 优化版启动脚本（含自定义模块注入）
├── config.yaml                       # 可调参数集中管理（10+ 项）
├── Dockerfile                        # 容器构建文件
├── requirements.txt                  # 额外 Python 依赖
├── scripts/
│   ├── benchmark.py                  # 评测工具（三档负载 · SLA 判定 · 纯 stdlib）
│   ├── compare.py                    # Baseline vs Optimized 对比报告
│   ├── run_baseline.sh               # 一键 Baseline 评测流程
│   └── compile_kernels.sh            # HIP kernel 编译脚本
├── src/
│   ├── kv_cache/                     # KV Cache 管理
│   │   ├── block_allocator.py        #   分级块分配 + 碎片整理
│   │   └── cache_manager.py          #   Watermark 预算 + Prefix 缓存
│   ├── attention/                    # DCU Attention 后端
│   │   ├── dcu_attention.py          #   双路径：HIP JIT kernel / PyTorch fallback
│   │   └── hip_kernels/              #   HIP C++ kernel 源码
│   │       └── dcu_flash_attn.cpp    #     FlashAttention（LDS + MFMA + online softmax）
│   ├── scheduler/                    # 自定义调度器
│   │   └── custom_scheduler.py       #   长度感知调度（prefill/decode 分离）
│   ├── quantization/                 # KV Cache 量化
│   │   └── kv_quant.py              #    FP8 在线量化（非持久化）
│   ├── executor/                     # 执行路径优化
│   │   └── exec_path.py             #    HIP Graph 捕获 · 调度批量化 · 预热
│   └── utils/                        # 工具
│       └── profiling.py             #    RequestProfiler + DCUHardwareProfiler
├── docs/
│   └── env_vars.md                   # 环境变量说明
├── results/                          # 评测结果输出（.gitignored）
├── changelog.md                      # 提交变更日志
└── report.md                         # 优化方案说明文档
```

## 快速开始

### 前置条件

- 竞赛指定 DCU 加速卡（单卡）
- Python 3.10.12
- vLLM 0.18.1
- PyTorch 2.10.0 (ROCm)
- DTK（含 hipcc）

### Baseline 评测（上平台第一步）

```bash
# 一键运行
bash scripts/run_baseline.sh

# 或分步
bash baseline/launch.sh &
python scripts/benchmark.py --host localhost --port 8000 --output results/
```

### 优化版评测

```bash
bash launch.sh &
python scripts/benchmark.py --host localhost --port 8000 --output results/
```

### 对比报告

```bash
python scripts/compare.py results/baseline_xxx.json results/optimized_xxx.json
```

### 单档快速迭代

```bash
# 只跑短上下文（100 请求，8 并发）
python scripts/benchmark.py --tier short
```

## 评测体系

| 负载档位 | 上下文长度 | 并发数 | 权重 | 场景 |
|----------|-----------|--------|------|------|
| short | ~512 tokens | 8 | 20% | 对话 / 简单问答 |
| medium | ~4096 tokens | 32 | 50% | 代码生成 / 文档摘要 |
| long | ~16384 tokens | 8 | 30% | 长文档理解 |

### SLA 硬性约束（违反则该档吞吐量得分清零）

- **TTFT P99** ≤ Baseline TTFT P99 × 1.5
- **TPOT P99** ≤ Baseline TPOT P99 × 1.5

### 精度系数

四类任务精度下降幅度 Δ 映射为系数（0~1.00），Δ ≤ 1% 时系数 = 1.00。最终得分 = 吞吐量得分 × 精度系数。

## 优化技术路线

### KV Cache 与显存管理
- **分级块分配**：根据上下文长度（短/中/长）选择不同块大小（16/64/256 tokens），减少内碎片
- **碎片整理**：空闲块占比超过阈值时主动整理，回收 HBM 连续空间
- **KV FP8 量化**：写入时量化为 FP8 (E4M3)，读取时反量化，节省约 40-50% KV Cache 显存
- **Prefix 缓存**：检测共享前缀，复用 KV 块，避免重复 prefill

### Decode 阶段算子优化（HIP/DTK）
- **HIP FlashAttention**：针对 DCU CDNA 架构手写 HIP C++ kernel，利用 LDS（64KB/CU）做 double buffering，MFMA 指令（16×16×16）加速矩阵乘法
- **GQA 优化**：Qwen 的 64 Q heads / 32 KV heads = 2 queries per KV head，减少 KV 扩展开销
- **128B HBM burst 对齐**：全局内存访问对齐到 HBM burst 边界，避免跨 burst 额外开销

### 执行路径优化
- **HIP Graph 捕获**：对 Decode 阶段固定 batch 路径进行图捕获，消除逐 step 的 kernel launch 开销
- **调度批量化**：每 4-8 步调度一次，减少 Python 层开销
- **异步 SDMA 传输**：利用 DCU 异步 DMA 引擎重叠 HBM ↔ Host 数据搬运
- **预热**：服务初始化时运行 5 轮 dummy 推理，填充 ROCm kernel 缓存

## 竞赛约束

| ✅ 允许 | ❌ 禁止 |
|---------|---------|
| 推理过程中的算子级低精度计算 | 修改模型结构、权重或推理语义 |
| KV Cache 在线量化（非持久化） | 持久化量化、剪枝、蒸馏、微调 |
| 自定义 HIP kernel | 预缓存答案、截断输入、跳层 |
| vLLM 插件 / 调度器定制 | 外挂辅助模型（含投机采样） |
| 环境变量自定义 | 评测期间下载外部依赖 |

## 团队成员

- 复旦大学 — dx

## 提交清单

- [x] `launch.sh` — 服务启动脚本（Baseline + Optimized 双版本）
- [x] `config.yaml` — 可调参数声明
- [x] `Dockerfile` — 容器构建
- [x] `changelog.md` — 变更日志
- [x] `report.md` — 优化方案说明
- [x] `docs/env_vars.md` — 环境变量说明
- [x] `src/` — 完整源代码
- [ ] `checksum.txt` — 权重 SHA256（待平台生成）

## 参考资源

- [竞赛平台](https://pra.xtnl.org.cn/)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [ROCm HIP 编程指南](https://rocm.docs.amd.com/en/latest/reference/hip.html)
- [DTK 软件使用说明](https://pra.xtnl.org.cn/)（竞赛平台提供）
