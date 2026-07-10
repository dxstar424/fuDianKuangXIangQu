# 参数调优手册（由整合负责人维护）

> **维护人**：角色 I（整合与 Git）  
> **冲刺阶段**：[sprint_strategy_0711.md](./sprint_strategy_0711.md)  
> **提交对齐**：[env_vars.md](./env_vars.md)  
> **调参原则**：一次只动一个变量；每次改动记录 A/B；疑似负优化立即回滚。

---

## 1. 怎么用本文档

| 场景 | 做法 |
|------|------|
| S1 恢复提交 | `verify_recover_config.sh` → SCNet 三档 → 平台 1 次 |
| S2 launch A/B | `ab_stage2.sh` → 8–16K×5 → 只合赢家 |
| S3 GQA | `stage3_gqa_launch.sh` → token 一致 → quick → 平台 |
| S4 Graph | 仅 S3 净增后；`stage4_graph_launch.sh` |
| 指标好但 SLA 熔断 | 角色 G 否决；回滚 tag |

---

## 2. 参数总览（按影响面）

### 2.1 启动与显存（S1 默认）

| 参数 | 默认值 | 作用 | 调大/开 | 调小/关 | 风险 |
|------|--------|------|---------|---------|------|
| `GPU_MEMORY_UTILIZATION` | **`0.94`** | KV 池占比 | 长档 KV↑ | OOM↓ | **禁止默认 0.95**（7/10 负优化） |
| `DO_WARMUP` | **`0`** | 分档 dummy | TTFT 稳 | 启动快 | 平台默认关；S2 可试单档 |
| `WARMUP_ROUNDS` | `1` | warmup 轮数 | 更稳 | 更快 | 多轮慢数分钟 |
| `WARMUP_TIER` | `8-16K` | 预热档 | 主档稳 | — | 仅 DO_WARMUP=1 |
| `ENABLE_PREFIX_CACHING` | `1` | prefix CLI | TTFT↓ | — | 低风险保留 |
| `USE_FDU_SERVER` | **`0`** | 插件入口 | S3 起用 | stock | S1/S2 保持 0 |
| `ENFORCE_EAGER` | **`1`** | 禁原生 graph | 稳 | `0`=开 graph | S2 主 A/B 项 |

### 2.2 FDU 优化开关

| 参数 | S1 默认 | S3（Phase2） | 建议 |
|------|---------|--------------|------|
| `FDU_PHASE` | `1` | `2` | S3 才升 |
| `FDU_ENABLE_GQA_OPT` | `0` | **`1`** | **唯一优先深接项** |
| `FDU_ENABLE_HIP_GRAPH` | `0` | `0` | 仅 S4；须 `ENFORCE_EAGER=0` |
| `FDU_ENABLE_KV_QUANT` | `0` | `0` | 默认不做 |
| `FDU_KV_CACHE_STRATEGY` | `none` | **`none`** | defrag 未接线，勿开 |
| `FDU_ATTENTION_BACKEND` | `vllm_default` | `vllm_default` | GQA wrap stock selector |

### 2.3 赛题锁定（禁止改）

- `--max-num-seqs`、`--max-num-batched-tokens`、batch scheduler
- `max_tokens`、`temperature=0`、模型权重

---

## 3. 推荐调参顺序（少次 · Phase2 三板）

```
【S1】verify_recover_config → 三档 SCNet → 平台 1 次（≥65）
【S2】ab_stage2: eager-off / warmup-816 → 只合赢家
【Phase2 三板各跑一轮，一次只开一块】
  B GQA:      stage3_gqa_launch.sh → token 一致 → 8-16K×5
  C defrag:   stage3_defrag_launch.sh → 16-32K×5（无收益可弃）
  D Graph:    ab_stage2 eager-off 或 stage4_graph_launch.sh
【合 main】仅「合」项 + quick gate → 平台 ≤1/阶段
```

详见 [sprint_strategy_0711.md](./sprint_strategy_0711.md) §5–§7。

脚本：`run_phase2_bench.sh`、`stage3_defrag_launch.sh`

**判据**：8–16K +≥0.5 tok/s 且 SLA 未逼近 Baseline×1.5。

---

## 4. 实测 A/B 记录表

| 日期 | 变量 | A 值 | B 值 | 8-16K 吞吐 | TTFT P99 | TPOT P99 | 精度 Δ | 结论 |
|------|------|------|------|------------|----------|----------|--------|------|
| | | | | | | | | |
| | | | | | | | | |

---

## 5. 难以解释的现象 → 排查

| 现象 | 可能原因 | 排查 |
|------|----------|------|
| 开 FDU_PHASE=2 无变化 | 未 `USE_FDU_SERVER=1` / 未 activate | 查日志 GQA selector patch |
| 吞吐 0 | 服务未起 | curl :8001/health |
| TTFT 爆 | 未 warmup | S2 试 warmup-816 |
| Graph 崩 | DCU graph 不稳 | `ENFORCE_EAGER=1` `FDU_ENABLE_HIP_GRAPH=0` |
| 平台降 SCNet 升 | env 不一致 | 对齐 launch.sh |

---

## 6. 合并冲突高发区

- `launch.sh`、`config.yaml`、`hooks.py`、`gqa_backend_wrap.py`、`env_vars.md` / 本文

冲突时优先保留「G 上次门禁通过」版本。
