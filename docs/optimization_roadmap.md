# 优化方向总规划（Optimization Roadmap）

> **截止**：2026-07-15 12:00  
> **团队**：4 人 · 复旦大学  
> **赛题**：基于国产加速卡的 Qwen 大模型推理服务优化（初赛）  
> **排行榜**：[希冀平台](https://course.educg.net/sv2/indexexp/contest/contest_rank.jsp?contestID=rF1452h6En0&my=false&contestCID=0#contestSubAn)

本文档是**唯一总规划入口**，整合战术优先级、技术阶段、赛程与门禁。细节见：


| 文档                                                                 | 用途                   |
| ------------------------------------------------------------------ | -------------------- |
| [**deep_optimization_guide.md**](./deep_optimization_guide.md)     | **必须 vs 冲刺 · 深度提分总指南** |
| [easy_scoring.md](./easy_scoring.md)                               | P0/P1/P2 短期改动清单      |
| [../report.md](../report.md)                                       | 官方提交用优化方案说明          |
| [env_vars.md](./env_vars.md)                                       | 环境变量与开关              |
| [submit_checklist.md](./submit_checklist.md)                       | 平台提交前检查              |
| [baseline_result.pdf](./baseline_result.pdf)                       | Baseline / 平台实测数据汇总  |
| [official_guidance_interpretation.md](./official_guidance_interpretation.md) | **官方技术指导解读（2026-07-09）** |
| [team_division.md](./team_division.md) | **四人分工 · Git · 门禁 · 协作流程** |
| [parameter_tuning.md](./parameter_tuning.md) | 参数调优手册（整合负责人维护） |
| [dcu_decode_benchmark_interpretation.md](./dcu_decode_benchmark_interpretation.md) | **DCU decode 访存实测解读（gfx936）** |
| [../../infra/docs/scnet-baseline-runbook.md](../../infra/docs/scnet-baseline-runbook.md) | SCNet 连容器与跑 baseline |

---

## 0. 官方指导对齐（2026-07-09 · zhaorq）

> 完整解读见 [official_guidance_interpretation.md](./official_guidance_interpretation.md)

官方明确三大考核方向：**KV Cache 优化 · 显存精细化管理 · 调度（token/graph，非 batch scheduler）**。

| 推理阶段 | 瓶颈 | 主攻指标 | 本队动作 |
|----------|------|----------|----------|
| **Prefill** | 算力（GEMM） | TTFT P99 | warmup、prefix cache、显存 0.94 |
| **Decode** | 带宽 + KV 读取 | TPOT P99、吞吐 | GQA、KV 块/defrag、Graph capture、KV FP8 融合 |

**相对原计划的调整（2026-07-09 官方指导 + DCU 实测）：**

1. **KV 块分配 / defrag 上调为 P1** — PagedAttention + 长档 KV 读
2. **HIP Graph / 算子融合上调为 P2** — 实测 decode 带宽墙下，launch 尾项（Amdahl）是少数合规 TPOT 增量
3. **GEMV 双缓冲 / 提占用 → 停止投入** — DCU 实测无双缓冲已达 HBM 101%，双缓冲 ±1.5%
4. **Decode HIP FlashAttention 维持 P4** — decode 95% 是权重 IO，不是 attention 算力
5. **Prefill flash / prefix / warmup 上调** — O(S²) 注意力主导长档 TTFT（8K→32K 超线性）
6. **KV FP8 修正预期** — 主要价值在显存/长档，非 decode 权重带宽；**必须融合**
7. **权重量化（int8）不作必须路径** — 赛题禁持久化；讲义 49→40ms 方案合规风险高

> 实测详情：[dcu_decode_benchmark_interpretation.md](./dcu_decode_benchmark_interpretation.md)

**新增分析门禁（每次优化后）：**

```
□ 三档吞吐是否变化？
□ TTFT P99（分档）是否仍 ≤ Baseline × 1.5？
□ TPOT P99（全局）是否仍 ≤ Baseline × 1.5？
□ 能否在 report.md 写出「±X% 吞吐 / ±Y ms TPOT」？
```

## 1. 目标与当前位置

### 1.1 评分公式（牢记）

```
最终得分 = 吞吐量得分 × 精度系数

吞吐量得分 = 4-8K(20%) + 8-16K(50%) + 16-32K(30%)
SLA 熔断：任一档 TTFT P99 或全局 TPOT P99 超标 → 该档吞吐得分清零
精度系数：四类任务 Δ≤1% 时 k=1.00（见参赛说明）
```

**主攻 8-16K 档（50% 权重）**。SLA 和精度是硬门槛，吞吐优化必须在二者通过后才有意义。

### 1.2 排行榜快照（2026-07-08）


| 对比项        | 榜首（豆包F4）    | 复旦最佳（富贵花开 #26） | 差距        |
| ---------- | ----------- | -------------- | --------- |
| 最终得分       | 89.2        | 84.7           | −4.5      |
| 8K-16K 吞吐  | 19.51 tok/s | 16.65 tok/s    | **−2.86** |
| 4K-8K 吞吐   | 21.42       | 18.37          | −3.05     |
| 16K-32K 吞吐 | 15.05       | 13.49          | −1.56     |
| SLA 扣分     | 0           | 0              | —         |
| 精度扣分       | 0           | 0              | —         |


**阶段目标：**


| 时间节点       | 目标分数 | 关键动作                                  |
| ---------- | ---- | ------------------------------------- |
| 7/9–7/10   | ≥85  | SCNet baseline + v0.2.1 launch 优化平台提交 |
| 7/11–7/12  | ≥87  | GQA / KV FP8（精度过关后）实机验证               |
| 7/13–7/14  | ≥89  | HIP Graph / 16K-32K 专项 + 最终提交         |
| 7/15 12:00 | 封榜   | 留 buffer 应对编译失败                       |


### 1.3 团队提交策略

- **统一用一个主账号提交**（建议「富贵花开」），**main 分支仅门禁通过后更新**。
- **每次平台提交前**必须由 **G** 跑过 `gate_check.sh quick`（至少）或 `full`（大合并 / KV FP8 后）。
- **合并与 Git 由 I 负责**：合并前打 tag，跑不通立即回滚，禁止把不可运行的 main 留到截止前。
- 反面案例：LUX. 7/8 未经充分验证提交，分数从 ~84 跌到 77。

---

## 2. 优化路线图（按 ROI + 官方导向排序）

```
Phase 0  baseline 数据 + profiling 基线
   ↓
Phase 1  显存预算 + TTFT 稳定（P0）─────── 当前重点
   ↓
Phase 2  KV 块/defrag + GQA decode（P1）── 官方最强调
   ↓
Phase 2b HIP Graph / 调度（P2）────────── 官方推荐 graph capture
   ↓
Phase 3  KV FP8 融合版（P2，须精度+净收益门禁）
   ↓
Phase 4  HIP FlashAttention（P4，profiling 证明后再做）
   ↓
Phase 6  文档量化贡献 + 平台提交
```

### 2.1 各阶段详情


| Phase | 优化项                         | 代码路径                                           | 预期收益                    | 状态          | 下一动作                   |
| ----- | --------------------------- | ---------------------------------------------- | ----------------------- | ----------- | ---------------------- |
| 0     | SCNet 环境 + baseline         | `scripts/scnet_setup.sh`, `record_baseline.sh` | 建立对照基线                  | 脚本就绪        | **SCNet 实机跑 baseline** |
| 1     | **最有把握 1.1–1.7**（见 easy_scoring） | `launch.sh` + `warmup` + `rocm_env` + `vllm_env` | 保 SLA + 小幅吞吐↑           | **v0.2.6 加固完成** | gate_check + 平台提交      |
| 1     | 显存 0.94 + warmup（8–16K 优先） | `launch.sh`, `warmup_server.py`                | 8-16K +1~3 tok/s，稳 TTFT | 已实现         | SCNet A/B vs stock 0.92 |
| 1     | prefix + 关日志 + ROCm + bf16 | `launch.sh`, `vllm_env.py`                     | TTFT / 微 TPOT            | 已实现         | 随 Phase 1 提交           |
| 2     | GQA einsum decode           | `fdu_vllm/gqa_decode.py`, `attention.py`       | TPOT −5~10%             | 默认关（PHASE=1） | `FDU_PHASE=2` 后门禁      |
| 2     | KV defrag / tiered blocks   | `fdu_vllm/kv_cache.py`, `src/kv_cache/`        | 长档显存、降碎片（**官方 P1**） | 骨架          | **验证 deep hook + 长档 TPOT** |
| 2     | TPOT profiling（长档 KV 读路径） | `src/utils/profiling.py`                       | 定位 decode 瓶颈            | 待做          | baseline 后首次 profiling   |
| 2b    | HIP Graph（graph capture）  | `fdu_vllm/hip_graph.py`                        | TPOT −5~15%，减调度开销       | 默认关         | GQA 验证后并行长测             |
| 3     | KV FP8 **算子融合**           | `fdu_vllm/kv_fp8.py` + attention             | 长档吞吐（须净收益）              | 默认关         | 证伪独立反量化后再开             |
| 4     | HIP FlashAttention          | `attention/dcu_attention.py`, `hip_kernels/`   | Prefill TTFT（非 decode 主因） | 骨架+fallback | **仅 prefill/O(S²) 证伪后再做** |
| 6     | 文档 / 提交                     | `report.md`, `changelog.md`, `easy_scoring.md` | —                       | 文档已对齐       | 回填三档数据                 |


### 2.2 优先级速查（日常执行看这里）

与 [easy_scoring.md](./easy_scoring.md) 一致：


| 优先级    | 改动                              | 何时做                      |
| ------ | ------------------------------- | ------------------------ |
| **P0** | `GPU_MEMORY_UTILIZATION=0.94`   | 立即                       |
| **P0** | 分档 warmup                       | 立即                       |
| **P0** | `FDU_ENABLE_KV_QUANT=0`         | 立即（保精度）                  |
| **P1** | prefix caching / 关日志 / ROCm env | Phase 1 一并提交             |
| **P1** | KV defrag / tiered blocks       | baseline 后，与 GQA 并行 |
| **P1** | GQA decode + 长档 TPOT profiling | Phase 2               |
| **P2** | `FDU_ENABLE_HIP_GRAPH=1`        | GQA 验证后，官方推荐 graph capture |
| **P2** | KV FP8 **融合版**（非独立反量化）   | 精度 + 净 TPOT 收益双门禁 |
| **P3** | HIP FlashAttention（**prefill / TTFT**） | profiling 证明长档 TTFT 主因是 O(S²) attn |
| **停止** | GEMV 双缓冲 / 提 wave 占用 | DCU 实测已在带宽墙，±1.5% |


---

## 3. 7 天赛程（7/9 → 7/15）

### Day 1–2（7/9–7/10）：保底拿分


| 任务                                                    | 负责人   | 产出                     |
| ----------------------------------------------------- | ----- | ---------------------- |
| SCNet 连容器，跑 `record_baseline.sh` + TPOT profiling | **I + P1** | 三档 baseline + decode 瓶颈报告 |
| 启动 `scnet_start_optimized.sh`，跑 `gate_check.sh quick` | **G** | 8-16K 吞吐 + hotpotqa 精度 |
| 确认 launch 参数在评测机等价生效 | **I** | 日志 + parameter_tuning.md |
| 平台提交 v0.2.1 | **G** | 一次有效提交，分数 ≥85 |


**Day 2 结束门禁：** baseline 表有数；至少 1 次平台提交；SLA=0、精度扣分=0。

### Day 3–4（7/11–7/12）：冲 8-16K


| 任务                                        | 负责人 | 产出                               |
| ----------------------------------------- | --- | -------------------------------- |
| GQA 路径 token 一致性验证 | **P2** | `verify_token_consistency.py` 通过 |
| KV 块/defrag deep hook + FP8 融合 A/B | **P1 + P2** | 独立反量化 vs 融合路径数据 |
| HIP Graph 长测 | **P2** | TPOT 下降且 SLA 不回归 |
| 16K-32K 显存 / 吞吐专项 | **P1** | 16K-32K 提升 ≥1 tok/s |
| 每日 integrate 合并 + tag | **I** | 合并后 smoke 通过 |
| 回填 `report.md` 优化贡献 | **G** | 文档与门禁数据一致 |


**Day 4 结束门禁：** 8-16K 较 baseline 提升可量化；若开 KV FP8，四类精度 Δ≤1%。

### Day 5–6（7/13–7/14）：冲刺


| 任务                                     | 负责人 | 产出                  |
| -------------------------------------- | --- | ------------------- |
| 评估 HIP Graph | **P2** | 若 Day 3–4 未完成则此处补测 |
| 16K-32K 档专项 | **P1** | 缩小与榜首 1.56 tok/s 差距 |
| **7/13 main 大合并** | **I** | 四人最优 branch → main |
| `gate_check.sh full` | **G** | 提交前最后一道门 |
| 更新 `report.md` + `parameter_tuning.md` | **G + I** | 提交材料完整 |


### Day 7（7/15 上午）：封榜 buffer

- 仅修复编译失败或 SLA 回归，**不做未验证的大改**。
- 12:00 前停止提交。

---

## 4. 四人分工

> 完整流程（Git、合并窗口、回滚、门禁节奏）：[team_division.md](./team_division.md)

| 代号 | 角色 | 核心职责 | 关键产出 |
|------|------|----------|----------|
| **I** | **整合与 Git** | 分支合并、**跑通优先**、tag 回滚、参数 A/B、[parameter_tuning.md](./parameter_tuning.md) | 可运行的 main、`integrate-pre-*` / `gate-pass-*` tag |
| **P1** | **性能 · KV/显存/Prefill** | 显存 0.94、KV 块/defrag、warmup、prefix、16–32K | TTFT 稳、长档吞吐↑ |
| **P2** | **性能 · Decode/算子** | GQA、HIP Graph、KV FP8 融合、HIP Attention（按需） | TPOT↓、token 一致性 |
| **G** | **门禁与定期评测** | **SLA P99 ≤1.5×**、**精度 Δ≤1%**、红线审查、平台提交 | gate 报告、指标表、否决权 |

### 4.1 为什么这样分

- **整合 I 单独一人**：两人 branch 各自能跑、合并就崩 — 需要专人 diff 冲突区（launch/hooks/config）、冒烟、回滚。
- **门禁 G 单独一人**：冲吞吐时容易牺牲 TTFT/TPOT/精度；G **有权否决**「看起来快但熔断」的改动。
- **性能 P1/P2 按 Prefill vs Decode 分**：对齐官方瓶颈分析，减少两人改同一文件。

### 4.2 Git 要点（I 执行）

- 个人：`feat/kv-*`（P1）、`feat/decode-*`（P2）
- 合并前：`git tag integrate-pre-YYYYMMDD-HHMM`
- 合并后：启动 + curl 冒烟；失败即 `reset` / `revert`
- **7/13**：四人最优 → **main 大合并** → 7/14 终提交

### 4.3 评测节奏（G 执行，非每行代码）

| 时机 | 动作 |
|------|------|
| I 每晚合并后 | 次日 G 跑 `gate_check quick`（若 SCNet 可用） |
| 每 2 天 | 三档吞吐 + TTFT/TPOT 记入 §5 |
| KV FP8 / 7/13 大合并前 | `gate_check full` |
| 7/14 | full + 平台提交 |

每日站会四问：

1. **I**：昨晚合并跑通了吗？回滚了吗？
2. **P1/P2**：今天动哪条指标、哪个 branch？
3. **G**：最近门禁 SLA/精度过了吗？有没有红线风险？
4. **全员**：有没有未经验证就想上平台？

---

## 5. 实机数据回填表

> SCNet 跑完后填入；平台得分出来后更新「平台实测」列。

### 5.1 Baseline（stock vLLM · SCNet）


| 档位     | 权重  | TTFT P99 | TPOT P99 | 吞吐 tok/s | SLA          |
| ------ | --- | -------- | -------- | -------- | ------------ |
| 4-8K   | 20% | —        | —        | —        | 待测（7/8 自测失败） |
| 8-16K  | 50% | —        | —        | —        | 待测           |
| 16-32K | 30% | —        | —        | —        | 待测           |


### 5.2 竞赛平台实测（富贵花开 · 2026-07-06）


| 档位     | TTFT P99 | TPOT P99 | 吞吐 tok/s | vs 榜首 | SLA  |
| ------ | -------- | -------- | -------- | ----- | ---- |
| 4-8K   | —        | —        | 18.37    | -3.05 | 0 扣分 |
| 8-16K  | —        | —        | 16.65    | -2.86 | 0 扣分 |
| 16-32K | —        | —        | 13.49    | -1.56 | 0 扣分 |


### 5.3 平台得分记录


| 日期         | 提交账号 | 8K-16K | 4K-8K | 16K-32K | 最终得分  | SLA | 精度扣分 | 备注       |
| ---------- | ---- | ------ | ----- | ------- | ----- | --- | ---- | -------- |
| 2026-07-06 | 富贵花开 | 16.65  | 18.37 | 13.49   | 84.74 | 0   | 0    | 平台基线     |
| 2026-07-08 | LUX. | 13.96  | 15.68 | 10.02   | 77.27 | 0   | 0    | 回归，勿再盲提交 |


---

## 6. 决策门禁（什么时候开什么开关）


| 开关                            | 开启条件                      | 关闭/回退条件            |
| ----------------------------- | ------------------------- | ------------------ |
| `GPU_MEMORY_UTILIZATION=0.94` | 默认开                       | OOM → 降到 0.92      |
| `FDU_ENABLE_KV_QUANT=1`       | `gate_check full` 四类 Δ≤1% | 任一任务 Δ>1% → 关      |
| `FDU_ENABLE_HIP_GRAPH=1`      | SCNet 长测 8-16K SLA 通过     | TPOT P99 回归 → 关    |
| HIP FlashAttention 深入开发       | Phase 1–3 完成且分数 ≥87       | 编译失败 / 2 天无收益 → 停  |
| 平台提交                          | SCNet quick gate 通过       | 未跑 gate → **禁止提交** |


---

## 7. 禁止项（死胡同清单）


| 禁止                                                             | 原因           |
| -------------------------------------------------------------- | ------------ |
| 修改 batch scheduler / `max-num-seqs` / `max-num-batched-tokens` | 赛题明确禁止       |
| 投机解码 / draft model / early-exit                                | 违规，直接取消成绩    |
| 权重持久化量化 / 剪枝 / 微调                                              | 违规           |
| 多账号无门禁乱提交                                                      | 浪费次数，可能降分    |
| 跳过 baseline 直接写 HIP kernel                                     | ROI 低，编译周期长  |
| decode GEMV 双缓冲 / 占用率调优                                        | DCU 实测已证伪（±1.5%） |
| 持久化权重量化（讲义 49→40ms 路径）                                     | 赛题红线 + 精度风险   |
| 启用 `custom_scheduler.py`                                       | 违规且无收益（并发=1） |
| KV 量化未融合、独立反量化抵消收益                              | 官方警告：量化可能是负优化 |
| 未写 env_vars / report 量化贡献                            | 官方：可复现性要求，可能判无效 |


---

## 8. 已知问题与待办


| #   | 问题                                   | 影响              | 负责人 | 状态              |
| --- | ------------------------------------ | --------------- | --- | --------------- |
| 1   | ~~`launch.sh` prefix caching 默认值缺失~~ | TTFT            | I   | **已修复（v0.2.2）** |
| 2   | `report.md` baseline 三档为空            | 无法量化优化收益        | I+G | 待 SCNet         |
| 3   | KV hooks 是否 deep integrate vLLM      | Phase 2 实际收益不确定 | P1   | 待验证             |
| 4   | 队内角色 I/P1/P2/G 姓名未填               | 协作不清            | 全员  | 填 team_division §10 |
| 5   | KV FP8 须算子融合，非独立反量化         | Phase 3 可能无净收益 | P2   | 待 A/B 证伪/证实   |
| 6   | report.md 缺量化贡献表              | 官方提交要求         | G    | 每项优化填 ±%     |
| 7   | parameter_tuning.md A/B 表为空      | 调参无据可查         | I    | baseline 后填     |
| 8   | 与 DCU 讲义 TPOT/TTFT 未对齐        | 无法验证优化假设      | G    | SCNet 后填 §九对照表 |


---

## 9. 快速命令索引

```bash
# ── SCNet Phase 0：环境与 baseline ──
bash scripts/scnet_setup.sh
bash scripts/record_baseline.sh

# ── 启动优化版（端口 8001）──
bash scripts/scnet_start_optimized.sh

# ── 门禁 ──
bash scripts/gate_check.sh quick          # 8-16K x20 + hotpotqa x10
bash scripts/gate_check.sh full           # 全档吞吐 + 四类精度

# ── 评测机等价启动 ──
bash launch.sh

# ── Token 一致性（开 GQA / KV FP8 后）──
python scripts/verify_token_consistency.py --baseline-port 8000 --opt-port 8001

# ── 编译 vLLM（改 patches 后）──
bash scripts/compile_vllm.sh
```

---

## 10. 文档维护

- **谁改代码，谁更新** §5 数据表 + `report.md` + `changelog.md`。
- **每次平台提交后**，D 更新 §5.3 平台得分记录。
- **优先级冲突时**：以本文档 §2.2 P0→P3 为准；分工见 [team_division.md](./team_division.md)；参数见 [parameter_tuning.md](./parameter_tuning.md)。

