# 参数调优手册（由整合负责人维护）

> **维护人**：角色 I（整合与 Git）  
> **提交对齐**：[env_vars.md](./env_vars.md)（评测必填，须与本文一致）  
> **调参原则**：一次只动一个变量；每次改动记录 A/B 数据；疑似负优化立即回滚。

---

## 1. 怎么用本文档

| 场景 | 做法 |
|------|------|
| 想提 8–16K 吞吐 | 先查 §3「显存/KV」→ 改一个参数 → 交给角色 G 跑门禁 |
| 合并后跑不通 | 整合负责人查 §6「合并冲突参数」→ 回滚 tag |
| 指标好但 SLA 熔断 | 角色 G 否决；整合负责人回滚该 MR 引入的参数 |
| 搞不清某参数作用 | 整合负责人做单变量 A/B，结果写入 §4 实测表 |

---

## 2. 参数总览（按影响面）

### 2.1 启动与显存（影响 TTFT + 长档吞吐）

| 参数 | 默认值 | 作用 | 调大/开 | 调小/关 | 风险 |
|------|--------|------|---------|---------|------|
| `GPU_MEMORY_UTILIZATION` | `0.94` | vLLM KV 池占 GPU 显存比例 | 长档 KV 更多，吞吐↑ | OOM 风险↓ | >0.95 易 OOM；<0.92 长档吞吐掉 |
| `DO_WARMUP` | `1` | 启动后分档 dummy 推理 | TTFT P99 更稳 | 启动快 | 关可能导致首条 SLA 熔断 |
| `WARMUP_ROUNDS` | `1` | warmup 轮数 | TTFT 更稳 | 启动更快 | 多轮启动慢数分钟 |
| `WARMUP_TIER` | `all` | 预热档位 | 全档 TTFT 稳 | 只预热指定档 | 未预热档首条可能尖刺 |
| `ENABLE_PREFIX_CACHING` | `1` | vLLM prefix 缓存 CLI | 共享前缀 TTFT↓ | 略减内存开销 | 盲测收益有限 |
| `FDU_ENABLE_PREFIX_CACHE` | `1` | 与上配合生效 | 同上 | prefix 可能不生效 | launch 须 export |

### 2.2 FDU 优化开关（影响 TPOT / 吞吐 / 精度）

| 参数 | 默认值 | 作用 | 建议 |
|------|--------|------|------|
| `FDU_ENABLE` | `1` | 总开关 | 调试 stock 时设 0 |
| `FDU_ENABLE_KV_QUANT` | `0` | KV 在线 FP8 | **默认关**；G 确认精度+净 TPOT 后再开 |
| `FDU_ENABLE_GQA_OPT` | `1` | GQA einsum decode | 降 TPOT；合并后须 token 一致性 |
| `FDU_ENABLE_HIP_GRAPH` | `0` | Graph capture | TPOT↓；长测 SLA 不过则关 |
| `FDU_KV_CACHE_STRATEGY` | `defrag` | KV 块策略 | defrag / prealloc / dynamic |
| `FDU_ATTENTION_BACKEND` | `dcu_optimized` | Attention 路径 | 非 DCU 环境自动 fallback |

### 2.3 ROCm 环境（微优化，一般不动）

| 参数 | 默认 | 作用 |
|------|------|------|
| `GPU_MAX_HW_QUEUES` | `2` | 硬件队列 |
| `HSA_ENABLE_SDMA` | `1` | 异步 DMA |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True` | 显存分配器 |
| `OMP_NUM_THREADS` | `8` | CPU 线程，防抢占 |

### 2.4 赛题锁定（禁止改）

以下参数**不得**在优化中修改，门禁角色 G 每次合并必查：

- `--max-num-seqs`、`--max-num-batched-tokens`
- `max_tokens`（评测脚本锁定）
- `temperature`（必须为 0）
- batch scheduler 相关代码
- 模型权重 / 结构

---

## 3. 推荐调参顺序（整合负责人执行）

```
1. GPU_MEMORY_UTILIZATION  0.92 → 0.93 → 0.94（OOM 则回退）
2. 确认 warmup + prefix 生效（看 launch 日志）
3. FDU_ENABLE_GQA_OPT=1 合并后验证
4. KV defrag / FP8（须 G 门禁）
5. FDU_ENABLE_HIP_GRAPH=1（须 G 长测）
```

**一次只改一项**，改完提交到集成分支，等 G 评测后再动下一项。

---

## 4. 实测 A/B 记录表（整合负责人填写）

| 日期 | 变量 | A 值 | B 值 | 8-16K 吞吐 | TTFT P99 | TPOT P99 | 精度 Δ | 结论 |
|------|------|------|------|------------|----------|----------|--------|------|
| | | | | | | | | |
| | | | | | | | | |

---

## 5. 难以解释的现象 → 排查清单

| 现象 | 可能原因 | 排查 |
|------|----------|------|
| 吞吐 0 / 全 failed | 服务未启动、模型路径错 | curl :8001/health |
| TTFT 突然爆 | 未 warmup、prefix 未开 | 查 launch env |
| TPOT 变差但吞吐不变 | KV 量化独立反量化 | 关 FP8 对比 |
| 合并后编译过、启动崩 | 两分支改了同一 hook | bisect / 回滚 tag |
| 平台分降、SCNet 升 | 评测机 env 与 SCNet 不一致 | 对齐 launch.sh |

---

## 6. 合并冲突高发区

合并前整合负责人重点 diff 这些文件：

- `launch.sh`（env 默认值）
- `config.yaml`
- `src/fdu_vllm/hooks.py`（开关加载顺序）
- `src/fdu_vllm/config.py`
- `docs/env_vars.md` / 本文档

**规则**：冲突时优先保留「G 上次门禁通过」的版本，再逐项 cherry-pick 新优化。

---

## 7. 与 env_vars.md 的关系

- **env_vars.md**：官方提交用，变量名 + 取值 + 原因（精简）
- **本文档**：队内调参实验记录 + 详细作用 + A/B 数据（可长）

整合负责人：**改参数 → 先更本文 §4 → 定稿后同步 env_vars.md**。
