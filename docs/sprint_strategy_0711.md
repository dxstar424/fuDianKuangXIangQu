# 冲刺攻略 7/11–7/15（资料过滤 + 少次多阶段 + Phase2 三板）

> **总规划**：[optimization_roadmap.md](./optimization_roadmap.md)  
> **四人分工**：[team_division.md](./team_division.md)（可一人一块）  
> **评分解读**：[scoring_and_results_interpretation.md](./scoring_and_results_interpretation.md)  
> **本队正式分**：lutinayi **59.97**（12.92 / 10.04 / **5.77**）

---

## 0. 硬前提（全队必读）

| 事实 | 含义 |
|------|------|
| 16–32K **5.77** < Baseline ~7.8 | S1 必须先纠负，再开 Phase2 |
| SCNet ≠ 平台 | 自测看方向；**得分只信平台** |
| 平台提交全赛程 **≤4 次** | 每阶段最多 1 次 |
| go/no-go | 8–16K 相对上一 tag **+≥0.5 tok/s**，且 TTFT/TPOT 未逼近 Baseline×1.5 |

**资料禁用**：投机解码 / 剪枝 / 换框架 / GEMV 双缓冲。

---

## 1. 总路线（先 S1，再 Phase2 三板）

```
S1 recover（stock 0.94 eager）→ 平台 1 次，总分≥65
        ↓
S2 launch A/B（eager-off / 单档 warmup）→ 只合赢家
        ↓
Phase2 三板在 SCNet 各跑一轮（一次只开一块）：
   ① GQA      ② KV defrag     ③ Graph（含 S2 原生 graph 对照）
        ↓
有净收益的才合入 main → quick gate → 平台 ≤1 次/阶段
```

**不要三板同一天全开、不要四人同时改 launch 默认。**

---

## 2. 一人一块（推荐认领）

| 板块 | 负责人（队内填名） | 分支示例 | SCNet 启动脚本 |
|------|-------------------|----------|----------------|
| **A · Launch/长档** | | `feat/block-launch` | 默认 `scnet_start_optimized.sh` / `ab_stage2.sh` |
| **B · GQA** | | `feat/block-gqa` | `stage3_gqa_launch.sh` |
| **C · KV defrag** | | `feat/block-defrag` | `stage3_defrag_launch.sh` |
| **D · Graph** | | `feat/block-graph` | `ab_stage2.sh eager-off` 或 `stage4_graph_launch.sh` |
| **当值 · 合并/gate/提交** | 轮值 | `integrate/*` | `gate_check.sh` |

合并顺序：**A（S1/S2 赢家）→ B → C → D**，每天最多合 **1 块** 进 `integrate`。

---

## 3. S1 — Recover（必做，未完成不开 Phase2）

```bash
bash scripts/verify_recover_config.sh
git tag recover-pre-platform    # 整合负责人打 tag

# 终端1
bash scripts/scnet_start_optimized.sh

# 终端2：至少 8-16K + 16-32K
cd ~/testdata
./run_throughput.sh 8-16K 10
./run_throughput.sh 16-32K 10
```

| 过关 | 标准 |
|------|------|
| 16–32K | ≥ **7.8** tok/s（纠负） |
| 总分 | 平台 **>60**，目标 **≥65** |
| 禁止 | 0.95、插件、关 eager、全档 warmup |

→ **平台提交第 1 次**（本阶段唯一）。

---

## 4. S2 — Launch 单变量（最多 2 实验 → 合 1 个赢家）

```bash
bash scripts/ab_stage2.sh baseline      # A：recover 默认
bash scripts/ab_stage2.sh eager-off     # B：vLLM 原生 graph（也算 Graph 对照）
bash scripts/ab_stage2.sh warmup-816    # C：仅 TTFT 尖刺时试
```

每轮后（服务已起）：

```bash
bash scripts/run_phase2_bench.sh 8-16K 5
# defrag/长档再加：
bash scripts/run_phase2_bench.sh 16-32K 5
```

| 变量 | 主看 | 合入条件 |
|------|------|----------|
| `ENFORCE_EAGER=0` | TPOT、8–16K 吞吐 | +≥0.5 tok/s，16–32K 不崩 |
| 单档 warmup | TTFT P99 | 仅 TTFT 尖刺时合入 |

记录写入 [parameter_tuning.md](./parameter_tuning.md) §4。

---

## 5. Phase2 三板 — SCNet 跑数协议（核心）

**共用规则**

- 一次只测 **一个板块**（关其它 Phase2 开关）
- 开发启动：`DO_WARMUP=0`（~10 min，别 full 三档 warmup）
- 快测：`bash scripts/run_phase2_bench.sh 8-16K 5`
- 长档：`bash scripts/run_phase2_bench.sh 16-32K 5`（defrag 必做）
- 对照 baseline：相对 tag `recover-pre-platform` 或 S1 平台后 tag

### 5.1 板块 B — GQA（已接线，优先测）

```bash
# 终端1
bash scripts/stage3_gqa_launch.sh

# 终端2
curl -sf http://127.0.0.1:8001/health
python scripts/verify_token_consistency.py --opt-port 8001
bash scripts/run_phase2_bench.sh 8-16K 5
```

| 看啥 | 过关线 |
|------|--------|
| 日志 | `GQA selector patch` / `FDU_GQA_` backend 名 |
| 精度 | token 一致 |
| 吞吐 | 8–16K **+≥0.5 tok/s** |
| SLA | TPOT P99 别逼近 1.5× |

代码位置：`src/fdu_vllm/gqa_backend_wrap.py`、`stage3_gqa_launch.sh`

### 5.2 板块 C — KV defrag（实验；deep hook 前数字可能不变）

```bash
bash scripts/stage3_defrag_launch.sh
bash scripts/run_phase2_bench.sh 16-32K 5
bash scripts/run_phase2_bench.sh 8-16K 5
```

| 看啥 | 说明 |
|------|------|
| 日志 | `KV hooks: defrag=True` |
| 吞吐 | **16–32K 主看**；若无变化 → 仅记录「未接线」，**不合 main** |
| 风险 | 同步 defrag 可能 TTFT 尖刺 → 盯 TTFT P99 |

> defrag 对象在 `src/kv_cache/`，尚未 deep hook 进 vLLM `CacheEngine`。  
> **无净收益 = 正常，直接放弃 C 板块，不占平台配额。**

### 5.3 板块 D — Graph（两档测法）

**D1 · 轻量（S2 已含）**：`ab_stage2.sh eager-off` — vLLM **原生** graph，不改插件。

**D2 · 自研钩子（S4，仅 GQA 平台有收益后）**：

```bash
bash scripts/stage4_graph_launch.sh   # ENFORCE_EAGER=0 + FDU_ENABLE_HIP_GRAPH=1
bash scripts/run_phase2_bench.sh 8-16K 5
# 连续压测 ≥30min 看 TPOT P99 稳不稳
```

| 看啥 | 过关线 |
|------|--------|
| 8–16K 吞吐 | +≥0.5 vs 当前 main 候选 |
| 稳定性 | 无 crash；TPOT P99 不熔断 |
| 失败 | 立即 `ENFORCE_EAGER=1` `FDU_ENABLE_HIP_GRAPH=0` |

---

## 6. 三板结果记录表（合 main 前必填）

复制到 `parameter_tuning.md` §4 或队内飞书/群公告：

| 日期 | 板块 | 8–16K tok/s | 16–32K tok/s | TTFT P99 | TPOT P99 | token/精度 | 结论 |
|------|------|-------------|--------------|----------|----------|------------|------|
| | S1 recover | | | | | — | |
| | S2 eager-off | | | | | — | |
| | **GQA** | | | | | 一致？ | 合 / 弃 |
| | **defrag** | | | | | — | 合 / 弃 |
| | **Graph** | | | | | — | 合 / 弃 |

**合入 main 规则**：只合 **结论=合** 的项；**一天最多合 1 块**；合并前打 `integrate-pre-YYYYMMDD-HHMM`。

---

## 7. 合入 main + 平台（少次）

```bash
# 整合当值
git tag integrate-pre-$(date +%Y%m%d-%H%M)
git checkout main
git merge --no-ff feat/block-xxx

# 门禁当值（有 GQA/Graph 改动时）
bash scripts/gate_check.sh quick
# 大合并或开新钩子后
bash scripts/gate_check.sh full

# 过关 → tag gate-pass-YYYYMMDD → 平台提交（G 授权）
```

| 阶段 | 平台次数 | 触发条件 |
|------|----------|----------|
| S1 | 1 | 16–32K SCNet 不崩 |
| S2 赢家 | ≤1 | 8–16K +≥0.5 |
| Phase2 任一块 | ≤1 | quick 过 + 有净收益 |
| 冻结前 | ≤1 | 最后一轮整合 |

---

## 8. 日常节奏（解决重启 20 分钟）

| 场景 | 做法 | 时长 |
|------|------|------|
| 开发重启 | `DO_WARMUP=0` + 对应 `stage*.sh` | ~10 min |
| 热机 | 2–3 次 curl 短请求 | 1 min |
| 快测 | `run_phase2_bench.sh 8-16K 5` | 2–5 min |
| 门禁 | `gate_check.sh quick` | ~45–60 min |

**一天最多 2 次重启**；写代码不占卡，**跑数才占卡**（上午/下午轮流）。

---

## 9. 止损（别钻死胡同）

| 情况 | 动作 |
|------|------|
| GQA 无 +0.5 | 弃 B，直接试 D1（原生 graph） |
| defrag 数字不变 | 弃 C（预期内，未 deep hook） |
| Graph crash / TPOT 爆 | 关 Graph，冻结 S2/GQA 赢家 |
| 三板都没净收益 | **冻结 launch 赢家**，只修 S1，不再开新钩子 |
| 距 7/15 <48h | 禁止新开 FP8 / FlashQLA / defrag deep hook |

---

## 10. 你队现在怎么做（最短 checklist）

**整合当值**

1. 确认 S1 `verify_recover_config.sh` 过 → 打 tag → 平台 1 次  
2. 发认领表：谁负责 B GQA / C defrag / D Graph  

**各板块负责人（SCNet 已连）**

3. 只起自己板块的 `stage*.sh` 或 `ab_stage2.sh`  
4. `run_phase2_bench.sh 8-16K 5`（C 再加 16–32K）  
5. 填 §6 表一行  

**门禁当值**

6. 有「合」的板块 → 1 次 `gate_check quick`  
7. 过 → 授权平台；不过 → 否决，整合回滚  

---

## 附录：脚本速查

| 脚本 | 用途 |
|------|------|
| `verify_recover_config.sh` | S1 默认锁定检查 |
| `scnet_start_optimized.sh` | S1 默认启动 |
| `ab_stage2.sh` | S2 A/B |
| `stage3_gqa_launch.sh` | 板块 B |
| `stage3_defrag_launch.sh` | 板块 C |
| `stage4_graph_launch.sh` | 板块 D（自研） |
| `run_phase2_bench.sh` | 快测吞吐 |
| `verify_token_consistency.py` | GQA 精度 |
| `gate_check.sh quick/full` | 门禁 |
