# 四人分工与协作流程（按 TTFT / TPOT + 三档优先级）

> 总规划：[optimization_roadmap.md](./optimization_roadmap.md)  
> 冲刺攻略：[sprint_strategy_0711.md](./sprint_strategy_0711.md)  
> 参数手册：[parameter_tuning.md](./parameter_tuning.md)

**分工原则**：指标拆开（TTFT ≠ TPOT）、档位有主次（主攻 8–16K），**不要三档各做一套大工程**。

---

## 0. 可分开优化的维度 → 谁管

```
                    ┌──────────────────────────────────────┐
                    │  计分：吞吐 × 精度；SLA 熔断清零档分   │
                    └──────────────────────────────────────┘
           ┌────────────────────┴────────────────────┐
           ▼                                         ▼
    【TTFT P99 · Prefill】                    【TPOT P99 · Decode】
    分档熔断 · 保底不炸                        全局熔断 · 主攻涨吞吐
           │                                         │
           ▼                                         ▼
         角色 P1                                   角色 P2
    prefix / warmup / 显存 0.94              GQA / Graph /（默认不做 FP8）
    16–32K 纠负 + 防 TTFT 尖刺               主看 8–16K 吞吐 + TPOT
           │                                         │
           └────────────────────┬────────────────────┘
                                ▼
                    角色 I：合并 / tag / launch 默认 / 平台节奏
                                ▼
                    角色 G：TTFT+TPOT+精度+红线 一票否决
```

| 维度 | 分开优化？ | 主责 | 说明 |
|------|------------|------|------|
| **TTFT** | ✅ 与 TPOT 分开想 | **P1** | Prefill：prefix、单档 warmup、显存；目标「别熔断」 |
| **TPOT** | ✅ 与 TTFT 分开想 | **P2** | Decode：GQA、Graph；目标「涨 8–16K 吞吐」 |
| **8–16K** | ✅ **唯一主攻档** | **P2 主 + P1 协** | 50% 权重；A/B 与 gate 以此为准 |
| **16–32K** | ⚠️ 只纠负/防熔断 | **P1** | 30%；勿单独开 FP8/defrag 大工程 |
| **4–8K** | ❌ 不单独优化 | 跟跑 | 20%；主攻 8–16K 通常会带上 |
| **精度 / 红线** | ✅ 独立守门 | **G** | 与吞吐优化解耦 |
| **合并 / 提交配额** | ✅ 独立节奏 | **I** | 平台 ≤4 次；S1–S4 各 ≤1 |

---

## 1. 角色一览

| 代号 | 角色 | 核心 KPI | 一句话 |
|------|------|----------|--------|
| **I** | 整合 · Git · 提交节奏 | main 可跑；平台配额不超支 | 跑通 + 少次提交比单点炫技重要 |
| **P1** | **TTFT / Prefill / 长档保底** | TTFT P99 不过线；16–32K ≥ Baseline | 守首字、纠长档负分 |
| **P2** | **TPOT / Decode / 8–16K** | 8–16K 吞吐↑；TPOT P99 下降且不过线 | 冲分主刀 |
| **G** | 门禁 · SLA · 精度 | SLA=0、精度 Δ≤1%、红线清 | 吞吐再高，熔断=零分 |

> 队内填名：`@I @P1 @P2 @G`

---

## 2. 角色 I — 整合与提交节奏

### 负责

| 职责 | 对齐冲刺 |
|------|----------|
| `launch.sh` 默认值、S1 recover 锁定 | `verify_recover_config.sh` |
| tag / merge / 回滚 | `recover-pre-*` / `gate-pass-*` / `platform-submit-*` |
| S2 `ab_stage2` 结果写入 `parameter_tuning.md` §4 | 只合赢家进默认 |
| 平台提交配额（全赛程约 ≤4） | 每阶段最多 1 次；**仅 G 授权后提交** |
| 7/13 前大合并 | 留 7/14–7/15 buffer |

### 不负责

- 不写 GQA/Graph 大块（那是 P2）
- 不擅自开 `gpu=0.95` / 全档 warmup / KV FP8
- 不牺牲 SLA 换吞吐（G 否决）

### 本周日程（I）

| 日 | 动作 |
|----|------|
| 7/11 | S1：tag + 协助平台第 1 次提交 |
| 7/12 | S2：收 P1/P2 的 A/B 数，合赢家 |
| 7/13 | S3 GQA 合 main（P2 交付后） |
| 7/14 | S4 仅门禁通过才合；否则冻结 |
| 7/15 上午 | 只修编译/启动 |

---

## 3. 角色 P1 — TTFT / Prefill / 16–32K 保底

### 主攻（与 P2 边界清晰）

| 做 | 不做 |
|----|------|
| prefix caching 是否生效、日志确认 | 不改 GQA / HIP Graph 核心 |
| `DO_WARMUP` + **仅** `WARMUP_TIER=8-16K`（S2-2b，仅 TTFT 尖刺时） | 不开全档 warmup（启动太慢） |
| 显存 `0.94` 稳定性；OOM 回退链 | **禁止默认改回 0.95**（除非 G+I 书面同意） |
| **16–32K**：吞吐 ≥~7.8、TTFT P99 不炸 | 不为 16–32K 单独开 defrag/FP8（默认不做） |
| 4–8K：合并后扫一眼即可 | 不为 4–8K 单独调参 |

### 分支与自测

```bash
git checkout -b feat/prefill-ttft   # 或 feat/launch-warmup
# 自测重点：TTFT P99（分档），其次 16–32K 吞吐
cd ~/testdata && ./run_throughput.sh 16-32K 5
# 主档也看一眼，避免拖累 8–16K
./run_throughput.sh 8-16K 5
```

### 交付物

- MR 说明：改了哪个 env / 预期 TTFT ±X ms
- 给 I 一行 A/B 表数据（`parameter_tuning.md` §4）

---

## 4. 角色 P2 — TPOT / Decode / 8–16K 主攻

### 主攻

| 做 | 不做 |
|----|------|
| **S3 GQA**：`stage3_gqa_launch.sh`、token 一致性 | 不改 scheduler / 投机解码 |
| **S4 Graph**（仅 S3 平台净增后）：`stage4_graph_launch.sh` | S3 无收益则 **跳过 S4** |
| 主看 **8–16K 吞吐 + 全局 TPOT P99** | 不为 4–8K 单独开实验 |
| `verify_token_consistency.py` | 默认 **不开** KV FP8 / FlashQLA 整库 |

### 分支与自测

```bash
git checkout -b feat/gqa-decode
bash scripts/stage3_gqa_launch.sh
# 协议：smoke → token 一致 → 8-16K ×5 → 有收益再 quick
python scripts/verify_token_consistency.py --opt-port 8001
cd ~/testdata && ./run_throughput.sh 8-16K 5
```

### 交付物

- 接线可运行（日志可见 GQA selector patch）
- 8–16K：吞吐 / TPOT 相对 recover tag 的数字
- 无收益 → 主动申请回滚，不硬上平台

---

## 5. 角色 G — 双指标守门（TTFT + TPOT）+ 精度

### 守门表（每次 gate 必填）

| 检查项 | 标准 | 谁的改动易踩 |
|--------|------|--------------|
| TTFT P99（**分档**） | ≤ Baseline×1.5 | P1 warmup/显存；P2 长测 |
| TPOT P99（**全局**） | ≤ Baseline×1.5 | P2 GQA/Graph |
| 8–16K 吞吐 | 相对上一 tag +≥0.5 才建议提交 | P2 |
| 16–32K 吞吐 | ≥ Baseline（~7.8），勿再负提升 | P1 |
| 精度四门 | Δ≤1% | P2 开 GQA/Graph 后 |
| 红线 | 无 scheduler/投机/持久化量化 | 全员 |

**一票否决**：吞吐好看但 TTFT 或 TPOT 逼近/超过 1.5× → **不得合 main、不得上平台**。

### 评测节奏（少次）

| 时机 | 命令 | 看什么 |
|------|------|--------|
| S1 提交前 | 三档×10 或 quick | 16–32K 纠负 + SLA |
| S2 合赢家前 | `8-16K ×5` + 记 TTFT/TPOT | 是否真有 +0.5 |
| S3 / S4 | `gate_check.sh quick`；大改 `full` | 精度 + SLA |
| 每日 | 不必每 commit 都测 | 跟 I 合并窗口 |

### 平台提交

- **仅 G 授权**（或 G 本人）触发；账号用本队，勿用他队「富贵花开」
- 条件：`gate-pass-*` tag + 当日门禁通过 + 未超阶段配额

---

## 6. 优化项 × 角色对照（执行版）

| 优化项 | 指标 | 档位侧重 | 主攻 | 协助 | 门禁 |
|--------|------|----------|------|------|------|
| S1 recover（0.94/eager/stock） | 三档纠负 | 尤其 16–32K | **I** | P1 | G |
| prefix + 关日志 | TTFT / 微 TPOT | 全档 | P1 | I | G |
| S2 `ENFORCE_EAGER=0` | **TPOT** | 8–16K | **P2** | I | G |
| S2 单档 warmup | **TTFT** | 8–16K | **P1** | I | G |
| S3 GQA wrap | **TPOT** + 吞吐 | **8–16K** | **P2** | — | G + token |
| S4 HIP Graph | **TPOT** | 8–16K | **P2** | I | G 长测 |
| 16–32K 扫尾 | TTFT+吞吐保底 | 16–32K | **P1** | G 记录 | G |
| 4–8K | 跟跑 | 4–8K | — | G 扫一眼 | G |
| KV FP8 / defrag / FlashQLA | — | — | **默认不做** | — | — |
| 合并 / 回滚 / 文档 | — | — | **I** | 全员 | G |
| 平台提交 | 得分 | — | **G** | I | — |

---

## 7. 协作流程（少次调试）

```
P1（TTFT）          P2（TPOT / 8-16K）
   │                      │
   │  feat/prefill-*      │  feat/gqa-* / feat/graph-*
   └──────────┬───────────┘
              ▼
         I 每日整合窗口
         tag → merge → smoke（DO_WARMUP=0）
              │
              ├─ 失败 → 回滚，P1/P2 修
              ▼
         G 排期 gate（每阶段 1 次 quick 为主）
              │
              ├─ TTFT 或 TPOT 熔断风险 / 精度 FAIL → 否决
              ▼
         gate-pass-* → 平台（该阶段第 1 次，G 授权）
```

**并行不打架**：

- P1 白天改 warmup/prefix 文档与 env 实验设计；P2 白天改 GQA 代码
- **同一时刻 SCNet 只有一台服务**：约定上午 P2 占卡（Decode 实验），下午 P1 占卡（TTFT/长档），或隔日轮换
- I 不占卡写合并与文档；G 用已起服务跑 gate

---

## 8. 每日站会（15 min）

1. **I**：默认配置是否仍是 S1 recover？今日平台配额还剩几次？
2. **P1**：TTFT / 16–32K 有没有尖刺或负提升？
3. **P2**：8–16K 吞吐、TPOT 相对上一 tag 的数字？
4. **G**：最近 gate 的 TTFT/TPOT/精度？能否授权提交？

---

## 9. 截止里程碑（对齐 S1–S4）

| 日期 | 里程碑 | 负责人 |
|------|--------|--------|
| **7/11** | S1 平台提交，目标总分 ≥65 | I + G（P1 协 16–32K） |
| **7/12** | S2 A/B 出赢家（eager 或 warmup） | P2 主 eager；P1 主 warmup；I 合入 |
| **7/13** | S3 GQA 合 main + quick；平台 ≤1 | **P2** + G + I |
| **7/14** | S4 Graph 仅可选；否则冻结 S3 | P2 + G |
| **7/15 上午** | 只修编译 | I |
| **7/15 12:00** | 封榜 | — |

---

## 10. 队内填写

| 代号 | 姓名 | 主指标 | 分支命名 | 联系方式 |
|------|------|--------|----------|----------|
| I | | 合并/提交节奏 | `integrate/*`，管 `main` | |
| P1 | | **TTFT** + 16–32K | `feat/prefill-*`, `feat/launch-*` | |
| P2 | | **TPOT** + 8–16K | `feat/gqa-*`, `feat/graph-*` | |
| G | | SLA + 精度 | 一般不长期开发分支 | |
