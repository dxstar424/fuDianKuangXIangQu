# 四人分工与协作流程

> 总规划：[optimization_roadmap.md](./optimization_roadmap.md)  
> 官方导向：[official_guidance_interpretation.md](./official_guidance_interpretation.md)  
> 参数手册：[parameter_tuning.md](./parameter_tuning.md)（整合负责人维护）

---

## 1. 角色定义（四人各一块，边界清晰）

```
                    ┌─────────────────────────────────────┐
                    │  main（评测提交分支，整合负责人管）   │
                    └─────────────────────────────────────┘
                           ↑ 定期合并          ↑ 门禁通过才合
              ┌────────────┼────────────┐      │
         feat/kv      feat/decode   feat/launch  │
              │            │            │      │
         角色 P1         角色 P2      角色 P1/P2
              └────────────┴────────────┘
                           │
                    角色 G 定期评测
                    （SLA / 精度 / 红线）
                           │
                    角色 I 整合 + Git + 调参文档
```

| 代号 | 角色 | 核心 KPI | 一句话 |
|------|------|----------|--------|
| **I** | **整合与 Git 负责人** | main 可编译、可启动、可评测 | 多人代码能跑通比单点高分更重要 |
| **P1** | **性能 · KV / 显存 / Prefill** | 8–16K / 16–32K 吞吐↑，TTFT 不爆 | PagedAttention、显存、warmup、prefix |
| **P2** | **性能 · Decode / 算子** | TPOT↓，吞吐↑ | GQA、Graph capture、KV FP8 融合、HIP |
| **G** | **门禁与定期评测** | SLA=0、精度扣分=0 | 吞吐再高，熔断或精度崩 = 零分 |

> **名字占位**：队内自行填 `@I @P1 @P2 @G`，本文档用代号。

---

## 2. 角色 I — 整合与 Git 负责人（最关键的新增职责）

### 2.1 负责什么

| 职责 | 说明 |
|------|------|
| **代码整合** | 各成员 branch 合并到 `integrate/*` 或 `main`；解决冲突；**合并后必须跑通启动 + curl + 至少 quick gate** |
| **跑通优先** | 两人单独都好、合起来崩 → 整合负责人 bisect / 回滚，不允许把「跑不通的 main」留给截止前 |
| **Git 版本控制** | 打 tag、回滚、分支策略；**每次合并前打 tag**，避免「跑不通找不到上一版」 |
| **参数调优** | 维护 [parameter_tuning.md](./parameter_tuning.md)；单变量 A/B；解释「莫名其妙的指标」 |
| **参数文档** | 与 [env_vars.md](./env_vars.md) 同步，保证提交合规 |
| **截止前大合并** | **7/13 前** 将四人最优改动合入 `main`，留 7/13–7/14 在 main 上最后一轮 |

### 2.2 不负责什么

- 不代替 P1/P2 写大块新优化（但可以修合并冲突、修 import、修 launch 默认值）
- 不单独决定「牺牲 SLA 换吞吐」（G 一票否决）

### 2.3 Git 工作流（整合负责人执行）

```bash
# ── 个人开发 ──
git checkout -b feat/kv-defrag        # P1
git checkout -b feat/gqa-graph        # P2
git checkout -b feat/launch-tune      # 可 P1 或 I

# ── 合并前（I 做）──
git tag integrate-pre-$(date +%Y%m%d-%H%M)   # 必打，方便回滚
git checkout main
git merge --no-ff feat/xxx -m "integrate: kv defrag from P1"

# ── 合并后冒烟（I 做，约 15–30 min）──
bash launch.sh   # 或 SCNet 等价
curl http://127.0.0.1:8000/health
# 失败 → git reset --hard integrate-pre-XXXX 或 revert merge commit

# ── 冒烟过 → 通知 G 排期评测 ──
```

**Tag 命名规范：**

| Tag | 含义 |
|-----|------|
| `integrate-pre-YYYYMMDD-HHMM` | 合并前快照，回滚用 |
| `gate-pass-YYYYMMDD` | G 门禁通过的可提交版本 |
| `platform-submit-YYYYMMDD` | 已推平台评测 |

**跑不通时：**

```bash
git tag -l 'integrate-pre-*'   # 找最近好版本
git checkout integrate-pre-20260712-2030
# 或
git revert -m 1 <merge_commit_sha>
```

### 2.4 整合窗口（建议固定，减少互相等待）

| 时间 | 动作 |
|------|------|
| 每日 **20:00** | P1/P2 push 当日 branch → I 合并到 `integrate/daily` |
| 合并后 | I 冒烟；过则 G **次日** 跑门禁（不必每行代码都测） |
| **7/12 晚** | 四人最优 branch 清单确认 |
| **7/13 全天** | **大合并 → main** |
| **7/14** | main 上只做小修 + G full gate + 平台提交 |

---

## 3. 角色 P1 — 性能 · KV / 显存 / Prefill

### 主攻方向（对齐官方指导）

- `GPU_MEMORY_UTILIZATION`、KV 块分配 / defrag（`src/kv_cache/`）
- warmup、prefix caching（**TTFT P99**）
- 16K–32K 长档显存与吞吐

### 工作方式

- 在 `feat/kv-*` 或 `feat/prefill-*` branch 开发
- 自测：`run_throughput.sh 8-16K 10` + 看一眼 TTFT
- **合并前**在 MR/群里说明：改了哪些参数、预期影响哪条指标
- 不擅自改 `max-num-seqs`、batch scheduler、max_tokens

### 交付物

- 代码 + `parameter_tuning.md` §4 一行 A/B（可由 I 代填）
- SCNet baseline 跑通（与 I 协作，前 2 天优先）

---

## 4. 角色 P2 — 性能 · Decode / 算子

### 主攻方向

- GQA decode（`gqa_decode.py`）→ **TPOT**
- HIP Graph capture（`FDU_ENABLE_HIP_GRAPH`）
- KV FP8 **融合**（非独立反量化）
- HIP Attention（**仅** profiling 证明需要时）

### 工作方式

- 在 `feat/decode-*` branch 开发
- 开 GQA/FP8/Graph 后跑 `verify_token_consistency.py`
- 合并依赖 I；**禁止**未经 G 门禁直接 push main 上平台

### 交付物

- 代码 + TPOT profiling 简要结论（哪条 KV 读路径是瓶颈）

---

## 5. 角色 G — 门禁与定期评测（SLA / 精度守门员）

### 5.1 核心职责

**别人冲吞吐时，G 守两条生命线 + 红线：**

| 守门项 | 标准 | 后果 |
|--------|------|------|
| **TTFT P99** | 各档 ≤ Baseline × **1.5** | 该档吞吐得分 **清零** |
| **TPOT P99** | 全局 ≤ Baseline × **1.5** | 该档吞吐得分 **清零** |
| **精度系数** | 四类任务 Δ ≤ **1%** | k < 1，大幅掉分 |
| **合规红线** | 见 §5.2 | 取消成绩 / 提交无效 |

**有权否决**：吞吐看起来很好，但 SLA 或精度不过的改动，**不得合 main、不得上平台**。

### 5.2 红线检查清单（每次合并后必过）

- [ ] 未改 `max_tokens`、`temperature`、`max-num-seqs`、`max-num-batched-tokens`
- [ ] 未改 batch scheduler / 投机解码 / 跳层
- [ ] 未持久化权重量化、未生成低精度权重缓存
- [ ] KV 量化未删除历史 token
- [ ] `launch.sh` 在评测机可启动
- [ ] `docs/env_vars.md` 与实际 env 一致
- [ ] AI 生成代码无「调低 max_tokens 刷分」类幼稚错误

### 5.3 定期评测节奏（不是每改一行就测）

| 频率 | 内容 | 命令 |
|------|------|------|
| **I 合并后 +1 天** | quick 门禁 | `gate_check.sh quick` |
| **每 2 天** | 三档吞吐 + TTFT/TPOT 记录 | `record_baseline.sh` 对比 |
| **开 KV FP8 / 大合并前** | full 门禁 | `gate_check.sh full` |
| **7/14 提交前** | full + 文档核对 | 同上 + report/env_vars |

### 5.4 维护的指标表

- 更新 [optimization_roadmap.md](./optimization_roadmap.md) §5
- 更新 [baseline_result.pdf](./baseline_result.pdf)（或通知 I 重跑生成脚本）
- 更新 [report.md](../report.md) §3「实测贡献」列
- 平台提交后记录得分（SLA / 精度扣分）

### 5.5 平台提交

- **仅 G 或 G 授权 I** 触发平台提交（统一账号「富贵花开」）
- 提交条件：`gate-pass-*` tag 存在 + 当日 quick/full 通过

---

## 6. 协作流程（一张图）

```
P1/P2 各自 branch 开发
        │
        ▼
   每日整合窗口（I）
   tag → merge → 冒烟启动
        │
        ├─ 失败 → 回滚 tag，P1/P2 _fix
        │
        ▼ 成功
   G 排期评测（非实时）
   quick / full gate
        │
        ├─ SLA/精度/红线 FAIL → 否决，I 回滚或 fix
        │
        ▼ PASS
   tag gate-pass-* → 可选平台提交
        │
        ▼
   7/13 四人最优 → main 大合并 → 7/14 终提交
```

---

## 7. 优化方向分工对照（谁做什么）

| 优化项 | 主攻 | 协助 | 门禁 |
|--------|------|------|------|
| SCNet baseline | I + P1 | P2 | G 记录 |
| 显存 0.94 / warmup | P1 | I 调参文档 | G |
| KV defrag / 块策略 | P1 | — | G |
| prefix caching | P1 | I | G |
| GQA decode | P2 | — | G + token 验证 |
| HIP Graph | P2 | I | G 长测 |
| KV FP8 融合 | P2 | P1 | G full |
| HIP FlashAttention | P2 | — | G + profiling |
| 分支合并 / 回滚 | **I** | 全员 | G |
| 参数 A/B 文档 | **I** | P1/P2 提供数据 | G |
| 平台提交 | **G** | I | — |

---

## 8. 每日站会（15 min，四人）

1. **I**：昨晚合并 main/integrate 是否跑通？回滚了吗？
2. **P1/P2**：今天在哪个 branch、动哪条指标？
3. **G**：最近一次门禁结果？SLA/精度有没有红线？
4. **全员**：有没有人在改 locked 参数 / 盲目上平台？

---

## 9. 截止前里程碑

| 日期 | 里程碑 | 负责人 |
|------|--------|--------|
| 7/9–7/10 | SCNet baseline 跑通 + 首次 quick gate | I+P1, G |
| 7/11 | 至少 1 次 integrate 合并成功 + gate-pass tag | I, G |
| 7/12 | 四人 branch 清单 + 参数文档初版 | I, P1, P2 |
| **7/13** | **main 大合并完成**，main 上 smoke + full gate | **I**, G |
| 7/14 | 平台终提交 | G |
| 7/15 12:00 | 封榜 | — |

---

## 10. 队内填写

| 代号 | 姓名 | 分支命名 | 联系方式 |
|------|------|----------|----------|
| I | | `integrate/*`, 管 `main` | |
| P1 | | `feat/kv-*`, `feat/prefill-*` | |
| P2 | | `feat/decode-*`, `feat/graph-*` | |
| G | | 一般不长期开发分支 | |
