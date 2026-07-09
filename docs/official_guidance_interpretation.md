# 官方技术指导解读（2026-07-09 · zhaorq）

> 本文档整理今日官方技术指导的核心观点，并映射到本队代码路径与行动项。  
> 配套文档：[optimization_roadmap.md](./optimization_roadmap.md) · [easy_scoring.md](./easy_scoring.md) · [report.md](../report.md)

---

## 一、官方明确了什么（总览）

官方将本次初赛的核心考核方向概括为三条主线：

| 主线 | 官方表述 | 对比赛的实际含义 |
|------|----------|------------------|
| **KV Cache 优化** | 推理加速引擎的核心机制 | 长上下文下吞吐与 TPOT 的第一抓手 |
| **显存精细化管理** | PagedAttention 块分配、碎片、量化 | 决定 8–16K / 16–32K 能否跑满、是否 OOM |
| **并发与调度** | token 调度、graph capture 等 | 初赛并发=1，**不是改 batch scheduler**，而是减 decode 逐步调度开销 |

优化总目标一句话：**在 SLA 硬约束下，最大化输出吞吐量，同时控制 TTFT / TPOT**——即在用户体验（首字、逐字延迟）与系统效率（吞吐）之间找最优解，而不是单押吞吐。

---

## 二、硬性约束与指标边界（不能跑偏）

### 2.1 优化目标（双目标 + 硬门槛）

```
                    ┌─────────────────────────────────┐
                    │  最大化：输出吞吐量（计分主体）   │
                    └─────────────────────────────────┘
                                    ↑
                    必须在 SLA 内才能计分
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │  TTFT P99（分档）≤ Baseline × 1.5                      │
        │  TPOT P99（全局）≤ Baseline × 1.5                      │
        │  精度系数：四类任务 Δ ≤ 1% 时 k = 1.0                  │
        └─────────────────────────────────────────────────────────┘
```

**解读：**

- 吞吐是**唯一计分项**，但 SLA 是**熔断器**——某一档 TTFT/TPOT 超标，该档吞吐得分直接清零。
- 官方强调「降低响应延迟」与「最大化吞吐」并重：只提吞吐把 TTFT/TPOT 顶穿，等于白干。
- 性能分析**不能只看最终吞吐数字**，必须能解释 TTFT（Prefill 路径）和 TPOT（Decode 访存路径）各自变了什么。

### 2.2 指标口径（官方再次强调）

| 指标 | 统计范围 | 常见误区 |
|------|----------|----------|
| **吞吐量** | 仅 **输出 token** / 时间窗；**不含** prompt token | 把 prefill 算进分子 |
| **TTFT** | 客户端发请求 → 收到**第一个生成 token**；含 tokenizer + prefill | 只算 GPU prefill、不含 HTTP |
| **TPOT** | 单请求：(末 token 时间 − 首 token 时间) / (输出 token 数 − 1)；全局 P99 | 把 TTFT 混进 TPOT；单 token 请求参与 TPOT |
| **并发** | 初赛固定 **1** | 误以为可以靠 batch 并发提分 |

### 2.3 技术红线（踩线 = 成绩作废）

官方在技术说明中**再次划红线**，与赛题 PDF 一致，重点包括：

| 红线 | 说明 |
|------|------|
| **禁止持久化低精度权重** | 不得生成可复用的量化/剪枝权重文件或缓存 |
| **禁止改变自回归解码语义** | Graph capture 可以，但不能改解码逻辑、投机解码、跳层 |
| **KV 量化不能删历史** | KV Cache 量化必须**保留全部历史 token 的 KV 位**，不得删除、跳过、裁剪 |
| **模型权重不可量化** | 允许算子内临时低精度、KV 在线量化；**不允许**权重持久化量化 |
| **环境变量必须文档化** | 用了自定义 env 必须提交说明（变量名、取值、原因），否则可能判无效 |
| **优化说明须可量化** | `report.md` 须写清技术路线，并对**每项优化的吞吐/时延贡献**做量化分析 |

---

## 三、Prefill vs Decode：瓶颈不同，打法必须分开

官方用两阶段模型帮选手定方向——**这是今天指导里最有行动价值的部分**。

### 3.1 Prefill 阶段（决定 TTFT）

| 属性 | 内容 |
|------|------|
| **计算特征** | 计算密集型；大规模矩阵乘法（GEMM） |
| **硬件瓶颈** | DCU **浮点算力**（MFMA 等） |
| **体验影响** | 首 token 过慢 → 用户感知「卡住」 |
| **主要指标** | **TTFT P99**（分档独立考核） |
| **优化方向** | 提高 Prefill 矩阵乘效率；减少 Python/调度在首 token 前的开销；长上下文下 KV 写入与 prefill 协同 |

**本队映射：**

- `warmup_server.py` 分档 warmup → 稳 TTFT P99，防首条熔断
- `--enable-prefix-caching` → 共享前缀时减少重复 prefill
- `GPU_MEMORY_UTILIZATION=0.94` → 长 prefill 时 KV 空间更充裕，减少因 OOM/reallocate 导致的 TTFT 尖刺
- **暂不优先**：手写 HIP Prefill kernel（ROI 低、周期长）

### 3.2 Decode 阶段（决定 TPOT 与吞吐）

| 属性 | 内容 |
|------|------|
| **计算特征** | **访存密集型**；每步反复读权重 + 读/write KV |
| **硬件瓶颈** | **HBM 带宽**、KV Cache 读取路径 |
| **体验影响** | 逐字生成慢 → 流式输出「一顿一顿」 |
| **主要指标** | **TPOT P99**（全局）；输出吞吐 |
| **优化方向** | 优化 Attention/Linear 算子；GQA 减 KV 读放大；KV 布局与块分配；Graph capture 减 launch 开销 |

**官方特别强调：**

> 长输入下 TPOT 恶化，要重点分析 **KV 的读取路径**——长文本卡顿往往卡在显存管理，而不是算力不够。

**本队映射：**

- `gqa_decode.py` → 减少 GQA 下 KV 重复读取（64Q/32KV）
- `kv_cache/` 块分配 + defrag → 降碎片、稳长档 KV
- `FDU_ENABLE_HIP_GRAPH=1` → Graph capture 减 token 步调度/sync（**不改变自回归语义**）
- KV FP8 → 降 KV 显存占用，但**必须与算子融合**（见 §四）

### 3.3 两阶段对照表（队内决策用）

| 阶段 | 瓶颈 | 主攻指标 | 权重档关联 | 本队优先动作 |
|------|------|----------|------------|--------------|
| Prefill | 算力 + 首路径开销 | TTFT P99 | 各档独立，长档更敏感 | warmup、prefix cache、显存预算 |
| Decode | 带宽 + KV 读路径 | TPOT P99、吞吐 | **8–16K 占 50%** | GQA、KV 块/量化融合、HIP Graph |

---

## 四、KV Cache：官方最看重的优化抓手

### 4.1 为什么 KV 是核心

- 长上下文下 KV 随序列长度**线性膨胀**，占显存大头。
- PagedAttention 的**块分配策略**直接影响：碎片率、能否装下更长上下文、KV 读取是否连续高效。
- 长请求易引发**显存碎片**，进而 OOM 或服务不稳定——官方点出这是「算力战」之外的**调度极限战**。

### 4.2 官方建议的 KV 优化方向

1. **提高 KV Cache 命中率 / 利用率**
   - 优化 PagedAttention **块大小、对齐、分配与回收**
   - 针对长上下文选合适 block granularity
2. **长输入下分析 TPOT 的 KV 读取路径**
   - 用 profiler 看 decode 每步 KV read 是否成为瓶颈
   - 不要只看 aggregate throughput
3. **KV 在线量化（合规范围内）**
   - 可以降显存、换更长上下文或更高 `gpu_memory_utilization` 空间
   - **关键警告**：若反量化作为**独立算子**执行，收益会被额外访存抵消
   - **必须**：量化/反量化与 Attention 或 KV read **算子融合**
   - **必须**：保留全部历史 token 的 KV，不得 skip/delete

### 4.3 对本队 KV FP8 方案的调整

当前 `kv_fp8.py` 为独立 quantize/dequantize 钩子，**符合赛题合规**，但官方提示可能存在**性能陷阱**：

| 现状 | 官方要求 | 调整计划 |
|------|----------|----------|
| 独立 quantize/dequantize | 融合进 Attention kernel | Phase 3 开启 FP8 前，SCNet 对比「独立反量化 vs 融合路径」 |
| 默认 `FDU_ENABLE_KV_QUANT=0` | 量化是把双刃剑 | **维持默认关**，精度过关 + 证明有净收益再开 |
| `kv_cache/` defrag 骨架 | 块分配是重点 | **提升优先级**：验证是否 deep hook vLLM，否则 Phase 2 无实际收益 |

---

## 五、调度与 Graph Capture

官方针对 **token 调度开销大** 的建议：

- 使用 **Graph capture** 减少逐步 decode 的同步与 kernel launch 开销
- **前提**：不能改变标准自回归解码流程

**与本队方案对应：**

- `fdu_vllm/hip_graph.py` + `FDU_ENABLE_HIP_GRAPH=1` — **与官方方向一致**
- 优先级：在 KV / GQA 有实测收益后开启（P3），长测确认 TPOT 下降且 SLA 不回归
- **禁止**：`custom_scheduler.py`、修改 vLLM batch scheduler（赛题红线，且并发=1 无 batch 收益）

---

## 六、性能分析方法论（官方强调的「科学调优」）

官方实质要求：把「玄学调参」变成「可复现、可量化」的优化。

### 6.1 不能只看吞吐

分析时至少分三档（4–8K / 8–16K / 16–32K）分别看：

- Output throughput
- TTFT P99（分档）
- TPOT P99（全局池）

并记录：**哪一阶段（Prefill / Decode）是主矛盾**。

### 6.2 建议的分析顺序

```
1. SCNet stock baseline（start_vllm.sh）→ 三档 TTFT/TPOT/吞吐
2. 每次只改一类优化 → A/B 对比
3. 长档（8–16K、16–32K）专项看 TPOT 与 KV 相关指标
4. gate_check 精度 → 再平台提交
5. report.md 回填「该项优化 ±X% 吞吐 / ±Y ms TPOT」
```

### 6.3 提交物要求（官方再次强调）

| 材料 | 要求 |
|------|------|
| `report.md` | 技术路线 + **每项优化贡献量化** |
| `docs/env_vars.md` | 变量名、取值、配置原因 |
| 源码 | 规范、可编译、**注释清晰**（便于复核） |
| `changelog.md` | 版本变更可追溯 |

---

## 七、映射到本队行动（相对原计划的调整）

### 7.1 优先级调整（2026-07-09 起生效）

原路线图 Phase 1→6 整体保留，但**权重重新分配**：

| 新优先级 | 方向 | 原 Phase | 调整说明 |
|----------|------|----------|----------|
| **P0** | 显存预算 + TTFT 稳定 | 1 | 不变；官方确认显存是长档基础 |
| **P1** | KV 块分配 / defrag / 碎片 | 2 | **上调**：与官方「PagedAttention 块策略」直接对齐 |
| **P1** | GQA decode（TPOT 访存） | 2 | 不变；对准 decode 带宽瓶颈 |
| **P2** | KV FP8 **融合版** | 3 | **加门禁**：独立反量化无净收益则不做 |
| **P2** | HIP Graph（调度） | 5→提前 | 官方明确推荐；可与 GQA 并行验证 |
| **P3** | HIP FlashAttention | 4 | **维持最低**：除非 profiling 证明 attention 是 TPOT 主因 |

### 7.2 新增必做项（原规划缺失）

1. **SCNet stock baseline 必须先跑通** — 无 baseline 无法做「量化贡献分析」
2. **长档 TPOT profiling** — 至少一次 decode 步的 KV read / kernel 时间分解
3. **report.md 每项优化填贡献表** — 官方倒逼，也是答辩材料
4. **env_vars.md 与 launch.sh 同步** — 已有，提交前再核对

### 7.3 明确不做（官方红线 + 低 ROI）

- 持久化权重量化 / 低精度权重缓存
- 删/Skip 历史 KV token
- 改 batch scheduler / max-num-seqs
- 投机解码、跳层、early-exit
- 未 profiling 直接 all-in HIP kernel 两周

---

## 八、与当前代码库的对照检查

| 官方方向 | 本队代码 | 差距 / 下一步 |
|----------|----------|---------------|
| PagedAttention 块优化 | `src/kv_cache/block_allocator.py` | 需验证是否接入 vLLM 运行时 |
| KV 量化 + 融合 | `kv_fp8.py` 独立钩子 | 需融合或证伪 |
| Decode 访存（GQA） | `gqa_decode.py` | 需 token 一致性 + TPOT 数据 |
| Graph capture | `hip_graph.py` | 默认关，待长测 |
| TTFT（Prefill） | warmup + prefix cache | prefix 默认值已修复 |
| 环境可复现 | `env_vars.md` | 保持同步 |
| 贡献量化 | `report.md` §3 | **待填实测数据** |

---

## 九、队内速记（一页纸）

1. **计分看吞吐，生死看 SLA，口碑看 TTFT/TPOT。**
2. **Prefill 拼算力 → TTFT；Decode 拼带宽 + KV 读 → TPOT。**
3. **KV Cache + 显存块策略是官方最强调的抓手，不是边角料。**
4. **KV 量化必须融合，独立反量化可能白做。**
5. **Graph capture 减调度开销，但不能改解码语义。**
6. **并发=1，调度优化 ≠ 改 batch scheduler。**
7. **每项优化要有 A/B 数字写进 report，env 要写进 env_vars。**
8. **先跑通 SCNet baseline，再谈优化；先 quick gate，再平台提交。**

---

## 十、文档索引

| 文档 | 用途 |
|------|------|
| [deep_optimization_guide.md](./deep_optimization_guide.md) | **必须 vs 冲刺 · 深度提分总指南** |
| [optimization_roadmap.md](./optimization_roadmap.md) | 赛程、分工、优先级 |
| [easy_scoring.md](./easy_scoring.md) | 短期改动清单 |
| [baseline_result.pdf](./baseline_result.pdf) | 实测数据汇总 |
| [env_vars.md](./env_vars.md) | 环境变量提交说明 |
| [../report.md](../report.md) | 官方优化方案说明（须含量化贡献） |
