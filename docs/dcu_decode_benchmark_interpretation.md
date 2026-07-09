# DCU Decode 访存实测解读与提分调整

> **来源**：`大模型decode访存瓶颈与双缓冲_DCU实测(1).html`（gfx936 / wave64 微基准）  
> **关联**：[official_guidance_interpretation.md](./official_guidance_interpretation.md) · [optimization_roadmap.md](./optimization_roadmap.md) · [team_division.md](./team_division.md)

---

## 一、实测硬件基线（gfx936）

| 指标 | 实测值 | 含义 |
|------|--------|------|
| HBM 峰值带宽 | **1206 GB/s** | decode GEMV 已能跑到 92–101% |
| bf16 算力峰值 | **≈395 TFLOPS** | prefill 大 GEMM 可贴顶 |
| Roofline 拐点 | **≈327 FLOP/byte** | 低于此 = 访存受限；高于 = 计算受限 |

**对我们比赛的含义**：在 DCU 上，decode 与 prefill 的优化手段**完全相反**，不能混为一谈。

---

## 二、核心结论（用数据说话）

### 2.1 Decode：权重 IO 一家独大（≈95%）

实验 A（gate_up 356.5MB，M 从 1 扫到 4096）：

| M (token) | 场景 | dt (μs) | TFLOPS | 权重 GB/s | 瓶颈 |
|-----------|------|---------|--------|-----------|------|
| 1 | **decode** | 292 | **1.22** | **1221** | 访存 |
| 32 | 过渡 | 271 | 42 | 1318 | 过渡 |
| 2048 | **prefill** | 1836 | **398** | 194 | 计算 |

- M=1→32 时，**耗时几乎不变**（292→271 μs），因为时间 ≈ 权重字节 ÷ 带宽，与 token 数无关。
- 算力从 1.2 TFLOPS 涨到 42 TFLOPS，但 dt 不动 → **decode 端算力 99.7% 时间在等权重**。

**换算到 Qwen3.5-27B 整模型一次 decode：**

| 资源 | 估算 | 占 decode 时间 |
|------|------|----------------|
| 权重 IO（54 GB bf16） | 54÷1.2 TB/s ≈ **45 ms/token** | **≈95%** |
| 计算（54 GFLOP/token） | ÷395 TFLOPS ≈ **0.14 ms** | **≈0.3%** |
| KV IO（8K 上下文） | ≈ **0.4 ms** | ≈4% |
| KV IO（32K 上下文） | ≈ **1.7 ms** | 仍远小于权重 IO |

**与平台 baseline 对照**：实测 bf16 TPOT ≈ **49 ms**，与 45 ms 权重 IO 模型高度吻合 → **TPOT 主因是搬权重，不是算力不够**。

### 2.2 Prefill：计算主导，长档注意力 O(S²) 抬头

实验 C（投影 GEMM vs 单层注意力）：

| S | 投影 GEMM | 注意力(单层) | attn/GEMM |
|---|-----------|--------------|-----------|
| 512 | 0.29 ms | 0.18 ms | 0.61 |
| 8192 | 3.65 ms | **23.7 ms** | **6.50** |

- S 翻倍：GEMM ≈×2（O(S)），注意力 ≈×3.2–4（O(S²)）。
- **8192 时单层注意力已是投影 GEMM 的 6.5 倍** → 长 prompt TTFT 超线性膨胀的主因。

**实测 TTFT：**

| 上下文 | TTFT |
|--------|------|
| 8K | 2790 ms |
| 16K | 5371 ms（8→16K 约 ×1.93） |
| 32K | 11423 ms（16→32K 约 ×2.13，**超线性**） |

**混合架构缓冲**：64 层中仅 **16 层 full-attention**，48 层 GDN 线性注意力 O(S) → TTFT 增长比纯 Transformer 温和，但 **16–32K 档 TTFT 仍是 SLA 风险点**。

### 2.3 双缓冲 / 高占用：对 decode GEMV **几乎无效**

实验 B（gate_up / down GEMV，有无双缓冲）：

- 无双缓冲已达 HBM 峰值 **92–101%**
- 双缓冲 vs 无双缓冲差异 **±1.5%**（测量噪声）
- VGPR≈58、LDS≈32 → 占用率已高，**硬件 wave 切换已藏掉延迟**

**原因**：GEMV 是**带宽墙**不是**延迟墙**。双缓冲/提占用是治延迟的；管子已满，加 wave 只排队更长。

**Amdahl 尾项**：即使 GEMV 到顶，端到端 TPOT 还有 RMSNorm、RoPE、逐元素、sampling、**数十次 kernel launch** → 图捕获/融合仍有空间。

### 2.4 文档作者的 TPOT 49→40 ms

讲义提到 **权重 int8/int4 量化** 将 TPOT 从 ~49 ms 压到 ~40 ms。

**赛题红线**：禁止**持久化**权重量化、低精度权重缓存。  
**合规边界（需法务级谨慎）**：允许「算子内临时低精度、低精度矩阵乘法」——若做**读 bf16、kernel 内动态量化、不写盘**，理论上在边界内，但工程量大、精度风险高、评测审核不确定。

**当前策略**：**不把权重量化作为必须路径**；优先合规项（Graph、融合、GQA、KV 管理）。

---

## 三、对比赛优化方向的修正（相对原 roadmap）

### 3.1 应该上调的优先级

| 优化项 | 依据 | 主攻 | 负责 |
|--------|------|------|------|
| **HIP Graph / 算子融合** | Amdahl：launch×几十次/token | TPOT −5~15% | P2 |
| **GQA decode** | KV IO 长档上升；减 KV 读放大 | TPOT（长档更明显） | P2 |
| **Prefix cache + warmup** | Prefill O(S²)；稳 TTFT | **TTFT P99**（16–32K） | P1 |
| **FlashAttention 类 prefill** | 8192 单层 attn 23.7ms | **TTFT**（8–16K/16–32K） | P2（若 vLLM 路径可 hook） |
| **显存 0.94 + KV 块策略** | 长档 KV 写/碎片；非 decode 权重主因但防 OOM/抖动 | 吞吐稳定 | P1 |
| **rocprof / omniperf 诊断** | MemUnitBusy≈100% 确认带宽墙 | 避免无效优化 | I + G |

### 3.2 应该下调或停止投入的

| 方向 | 原因 |
|------|------|
| **Decode GEMV 双缓冲 / 提 wave 占用** | 实测 ±1.5%，已在带宽墙 |
| **手写 HIP GEMV 榨带宽** | 无双缓冲已 101% 峰值，边际 ≈0 |
| **Decode 侧 FlashAttention 指望 2×** | decode 瓶颈是**权重 IO**不是 attention 算力 |
| **以为「算力优化」能救 TPOT** | decode 算力只用峰值 0.3% |
| **Prefill 权重量化** | 权重被 S 摊薄；讲义称 down 量化**拖累 16–32K** |

### 3.3 KV FP8 的定位（修正预期）

| 预期（原） | 修正（实测后） |
|------------|----------------|
| decode TPOT −15% | **unlikely** — KV IO 8K 仅 ~4% decode 时间 |
| 长档吞吐 +15% | **可能** — 32K KV ~1.7ms 且显存压力↑；更多 KV 槽 → 少碎片/OOM |
| 必须融合 | **更必须** — 独立反量化增加访存，与带宽墙相悖 |

**结论**：KV FP8 主要价值在 **显存预算 + 长档稳定性**，不是 decode 权重带宽的替代品。

---

## 四、分档提分策略（结合权重 20/50/30%）

### 4–8K（20%）

- Decode 权重 IO 主导 → **Graph capture、减 launch** 是少数合规增量
- KV IO 占比小 → KV FP8 收益有限
- TTFT 压力中等 → warmup 即可

### 8–16K（50%，主攻）

- TTFT：注意力开始反超 GEMM（S≈1K+）→ **prefix cache**、prefill 路径优化
- TPOT：仍 ≈45 ms 权重墙 → **Graph/融合** > HIP GEMV 双缓冲
- **profiling 必做**：确认端到端 TPOT 中 launch 占比

### 16–32K（30%）

- TTFT **超线性**（16→32K ×2.13）→ **SLA 熔断高风险**；prefill/flash 优先于 decode
- KV IO 占比上升 → **GQA + KV 块/defrag + FP8（融合）** 有价值
- 避免 prefill 路径上的权重量化（讲义实测拖累长档）

---

## 五、自查清单（profiler 驱动，避免死胡同）

跑 `rocprof` / `omniperf` 后对照：

| 观测 | 诊断 | 有效方向 | 无效方向 |
|------|------|----------|----------|
| MemUnitBusy≈100%，VALUBusy 低 | 带宽墙 | 减字节（合规内）、Graph、融合 | 双缓冲、提占用 |
| MemUnitStalled 高，Busy 不满 | 延迟墙 | 双缓冲、预取、提占用 | — |
| 达成带宽 ÷ 峰值 ≥85% | 已在墙 | 换优化域 | 继续榨 GEMV |
| VGPR 低、wave 多 | 占用已够 | 同上 | 软件流水 |

**Roofline 快算**：算术强度 = FLOPs/byte；decode GEMV ≈1 FLOP/byte ≪ 327 → **必然访存受限**。

---

## 六、对本队代码的具体改进建议

### 6.1 立即（必须）

1. **P2 验证 `FDU_ENABLE_HIP_GRAPH=1`** — 对准 Amdahl launch 尾项；长测 TPOT + SLA
2. **P1 跑通 SCNet baseline** — 拿到 49ms 量级 TPOT 与 TTFT 2790/5371/11423 量级对照
3. **G 记录分档 TTFT/TPOT** — 验证 16→32K 是否超线性恶化

### 6.2 短期（主攻 8–16K）

1. **Prefix + warmup**（已有）— 对准 prefill O(S²)
2. **GQA**（已有）— 长档 KV 读路径
3. **评估 vLLM/patch 是否可开 prefill flash-attn** — 对准 TTFT，非 decode GEMV

### 6.3 中期（16–32K + 合规）

1. **KV FP8 仅在上板 fusion 路径后开启**
2. **KV defrag deep hook** — 防长请求碎片（官方 + 实测双强调）

### 6.4 暂停 / 降级

1. `dcu_flash_attn.cpp` decode 方向大投入 → 除非 profiling 证明 attention 非权重 IO 主因
2. GEMV 双缓冲实验 → 实测已证伪
3. 权重量化方案 → 合规与精度风险高，仅作冲刺备选且须 G 否决权

---

## 七、与官方 zhaorq 指导的对齐

| 官方说法 | 实测印证 |
|----------|----------|
| Decode 访存密集型 | 权重 IO 95%，带宽墙 |
| 长输入 TPOT 看 KV 读路径 | KV 占比随长度升，但 decode 仍权重主导 |
| KV 量化须融合 | 带宽墙下独立反量化必亏 |
| Graph capture 减调度 | Amdahl：launch×N 次/token |
| Prefill vs Decode 分开优化 | Roofline 拐点 M≈32–128，两阶段瓶颈相反 |

---

## 八、队内一句话

> **Decode 慢是因为 54GB 权重每 token 搬一遍，不是算力不够；双缓冲救不了带宽墙。合规提 TPOT 靠 Graph/融合减 launch，合规提 TTFT 靠 prefill/注意力；KV 优化主攻长档显存与 KV 读，别在 GEMV 双缓冲上浪费时间。**

---

## 九、待填：与 SCNet 实测对齐表

| 指标 | 讲义实测 | 我们 SCNet | 我们平台 |
|------|----------|------------|----------|
| TPOT (bf16) | ~49 ms | 待填 | — |
| TTFT 8K | 2790 ms | 待填 | — |
| TTFT 16K | 5371 ms | 待填 | — |
| TTFT 32K | 11423 ms | 待填 | — |
| 8-16K 吞吐 | — | 待填 | 16.65 tok/s |

SCNet baseline 跑通后，I 填入 [`baseline_result.pdf`](./baseline_result.pdf)，G 核对是否接近讲义数量级。
