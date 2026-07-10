# 最有把握提分清单（去掉负优化 · 先超过官方 Baseline）

> **冲刺攻略**：[sprint_strategy_0711.md](./sprint_strategy_0711.md)（S1–S4 少次多阶段）  
> **深度解读**：[scoring_and_results_interpretation.md](./scoring_and_results_interpretation.md)  
> **分支**：`lutinayi_branch` · 入口：`launch.sh`

## 先认清事实（2026-07-10 22:14 · 本队 lutinayi_branch）

```
得分 59.97
吞吐 12.92 / 10.04 / 5.77
SLA=0 精度=0 → 能跑、未熔断；吞吐接近官方 Baseline（公式上约 60 分）
16-32K 5.77 < Baseline~7.75 → 该档为负提升
```

| 对比 | 4–8K | 8–16K | 16–32K | 得分 |
|------|------|-------|--------|------|
| **本队 7/10 lutinayi** | 12.92 | 10.04 | 5.77 | **59.97** |
| 官方 Baseline（估） | ~10.6 | ~9.6 | ~7.8 | **~60** |
| 榜上他队参考（富贵花开，**非本队**） | 18.37 | 16.65 | 13.49 | 84.74 |

**结论**：本队正式分 ≈「接近官方 Baseline」。排行榜 84 分账号**不是我们的**，不能当「曾经达到过」或回血目标。

---

## 当前默认（S1 Recover · 已锁）

| 项 | 7/10 提交配置 | **现在默认** | 原因 |
|----|---------------|--------------|------|
| 入口 | `fdu_vllm.server` | **stock `api_server`** | 去掉未验证插件层 |
| 显存 | **0.95** | **0.94** | 长档负优化主嫌疑 |
| warmup | 开（全档） | **关** | 平台评测不依赖长 dummy |
| Graph | 默认开 | **`--enforce-eager`** | 避 DCU Graph 负优化 |
| prefix + 关日志 | 开 | **仍开** | 低风险保留 |

校验：`bash scripts/verify_recover_config.sh`

---

## 提交顺序（少次多阶段）

1. **S1**：推恢复版 → 目标：**稳定高于官方 Baseline**（总分 ≥65）— 平台 **1 次**
2. **S2**：`ab_stage2.sh` 对 `ENFORCE_EAGER=0` / 单档 warmup 做 A/B → 平台 **≤1**
3. **S3**：`stage3_gqa_launch.sh`（GQA 已接线）→ token 一致 + quick → 平台 **≤1**
4. **S4**：仅 S3 净增后 `stage4_graph_launch.sh`；**禁止**未验证开 0.95 / KV FP8 / defrag
5. **不要**用他队 84 分当本队历史成绩

全赛程平台提交约 **≤4 次**。详见 [sprint_strategy_0711.md](./sprint_strategy_0711.md)。
