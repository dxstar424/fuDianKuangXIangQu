# Pra2026 — 基于国产加速卡的 Qwen 大模型推理服务优化

> **其他 Agent 请先读**：[../AGENTS.md](../AGENTS.md)（工作区必读，含目录地图、红线、优先级）

> 🏆 2026 年全国大学生计算机系统能力大赛 · 智能计算创新设计赛（先导杯）
>
> 赛题：基于国产加速卡（DCU）的千问大模型推理服务优化

[Python](https://www.python.org/)
[vLLM](https://github.com/vllm-project/vllm)
[PyTorch](https://pytorch.org/)
[ROCm](https://www.amd.com/en/products/software/rocm.html)

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
compute/                              # 工作区根（非 git）
├── AGENTS.md                           # Agent 必读
├── baseline/                           # ← 本仓库（竞赛提交）
│   ├── baseline/                       # stock vLLM 对照（launch.sh + config.yaml）
│   ├── launch.sh                       # 优化版启动（fdu_vllm）
│   ├── config.yaml                     # 可调参数
│   ├── src/fdu_vllm/                   # vLLM 插件入口（主要改动区）
│   ├── src/{kv_cache,attention,...}/   # 优化模块
│   ├── scripts/                        # 评测、编译、SCNet 脚本
│   ├── docs/                           # 队内战术文档（roadmap 为总入口）
│   └── patches/vllm_cscc/              # vLLM 补丁
├── infra/                              # SSH/容器连接（不提交）
└── docs/official/                      # 赛题原文参考（不提交）
```



## 快速开始



### SCNet 调试（选手 PDF 流程）

```bash
bash scripts/scnet_setup.sh          # Phase 0: vLLM 编译 + 模型 + testdata
bash scripts/record_baseline.sh      # 记录三档 baseline
bash scripts/gate_check.sh quick     # 精度/性能门禁
bash scripts/compile_vllm.sh         # 重编译 vLLM + FDU 补丁
bash launch.sh                       # 优化版服务（含 warmup）
```



### 本地 / 评测机

```bash
bash baseline/launch.sh &            # stock baseline
bash launch.sh                       # fdu_vllm 优化版（python -m fdu_vllm.server）
python scripts/benchmark.py --host localhost --port 8000 --output results/
python scripts/compare.py results/baseline_xxx.json results/optimized_xxx.json
```



### 项目结构（v0.2.0）

```
├── src/fdu_vllm/          # vLLM 插件入口（activate + server）
├── patches/vllm_cscc/     # vllm_cscc 补丁
├── scripts/
│   ├── scnet_setup.sh     # SCNet 一键初始化
│   ├── compile_vllm.sh    # 编译官方 vllm_cscc + 补丁
│   ├── gate_check.sh      # 每阶段门禁
│   └── verify_token_consistency.py
├── launch.sh              # 评测启动（合规，无 scheduler env）
└── docs/submit_checklist.md
```



## 评测体系

| 负载档位 | 上下文长度 | 并发数 | 权重 | 场景 |
|----------|------------|--------|------|------|
| 4-8K | 4K–8K | 1 | 20% | 短上下文 |
| 8-16K | 8K–16K | 1 | 50% | **主攻档** |
| 16-32K | 16K–32K | 1 | 30% | 长上下文 |

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
- **预热**：服务初始化时运行 dummy 推理，填充 ROCm kernel 缓存

## 竞赛约束

| ✅ 允许 | ❌ 禁止 |
|---------|---------|
| 推理过程中的算子级低精度计算 | 修改模型结构、权重或推理语义 |
| KV Cache 在线量化（非持久化） | 持久化量化、剪枝、蒸馏、微调 |
| 自定义 HIP kernel | 预缓存答案、截断输入、跳层 |
| vLLM 插件 / 环境变量 | 外挂辅助模型（含投机采样） |
| 显存利用率等非锁定启动参数 | 改 max-num-seqs / max-num-batched-tokens / batch scheduler |

## 团队成员

- 复旦大学 — 待补充



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

