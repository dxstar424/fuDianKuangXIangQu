# dx_branch gfx936 平台交接

> 更新：2026-07-14
>
> 分支：`dx_branch`
>
> 本轮分支审计前代码锚点：`8b31f18a793dda9b6b9067a305975207b767c52e`
>
> 最终交付提交以 `git rev-parse HEAD` 以及远端 `dx_branch` / GitHub `main` 为准。

## 一页结论

- 已记录的安全参照仍是 **66.8175 分**：4–8K / 8–16K / 16–32K 分别为 `15.03 / 12.00 / 6.09 tok/s`，SLA 与精度扣分均为 0。评测提交 hash 尚未随结果记录，不能把全部增益独立归因于 LLMM1。
- 当前平台候选默认使用 `FDU_GFX936_QUANT_MODE=w8`：五个已通过 gfx936 microbenchmark 的精确 shape 使用 W8A16 JIT GEMV，`(5120,17408)` 因只有 `0.505x` 自动保留 stock BF16。
- Decode 侧保留 BF16 LLMM1/stock 回退，并增加 KV block table 脏提交；本轮再借鉴 `lutinayi_branch` 的两个低风险点：vLLM 原生 prefix caching（默认开启，可用 `ENABLE_PREFIX_CACHING=0` 关闭）和 `--disable-log-stats`。
- **不合入**两个分支的自定义 Attention、KV FP8、分层 allocator、defrag、HIP Graph、scheduler 参数和长 warmup。它们要么未接入真实执行链，要么已有负优化/无收益证据，要么会扩大本次平台盲测风险。
- 不再追加 SCNet 测试。基于已有 shape microbenchmark 的主观平台预估为 **70–74 分，中位约 71 分**；这不是实测保证，W8 的端到端 TTFT、TPOT、长档吞吐和四项精度仍由平台决定。

## 当前提交实际生效的路径

```text
原始 Qwen3.5-27B BF16 checkpoint
  -> 原生 gfx936 wheel
  -> native vLLM prefix caching（默认开，可回滚）
  -> quiet request/stat logging
  -> FDU_GFX936_QUANT_MODE=w8
       -> 5 个接纳 shape: W8A16 N=1 JIT GEMV
       -> (5120,17408): stock BF16
       -> JIT / ABI / smoke / admission 失败: BF16 LLMM1 或 stock
  -> KV block table 仅 dirty 时 H2D
```

没有设置平台锁定的 `--max-num-seqs`、`--max-num-batched-tokens` 或自定义 scheduler 参数。默认 block size 仍由 vLLM 保持为 16，不修改 Attention 数值或 KV 物理布局。

## 已记录的保底参照

| 指标 | 平台实测 |
|---|---:|
| 4–8K 吞吐 | 15.03 tok/s |
| 8–16K 吞吐 | 12.00 tok/s |
| 16–32K 吞吐 | 6.09 tok/s |
| SLA 扣分 | 0 |
| 精度扣分 | 0 |
| 最终得分 | **66.8175** |

如果该结果对应 `88b7d10` 或其后继，则包含原生 gfx936 + BF16 + 五个 N=1 LLMM1 shape；由于缺少精确提交和同环境 stock A/B，只把它当回滚参照。

## lutinayi_branch 与 wyb 审计

| 分支锚点 | 可核验记录 | 实现状态 | 本轮处理 |
|---|---|---|---|
| `origin/lutinayi_branch@5fec8013754dee2fbc3acb6274ef52e6418d8c00` | 最新记录 `13.04 / 10.08 / 5.78 tok/s`、**60.19 分**、无 SLA/精度扣分；相对 59.85 的组合改动为 `+0.34` 分 | 根目录可提交；默认 BF16、prefix cache、关日志、warmup×2、eager。KV FP8 记录为 8–16K `7.81 tok/s`，约 `-36%`；GQA/Graph/defrag 默认关闭 | 只采用原生 prefix cache 与关统计日志；不整段 cherry-pick launch，因为其中含锁定的 `--max-num-seqs 256` 和本轮不需要的 warmup/plugin 分支 |
| `origin/wyb@f7dac25201f34435d27e25c25b5574fa1b6c251f` | `report.md` 吞吐表为空，`changelog.md` 明确写“待 DCU 平台验证” | 整个工程嵌套在子目录；`plugin.py` 不进入 launch；Attention 是标量 correctness baseline；KV allocator 明确是 metadata-only，defrag 不搬 KV tensor | 不合入代码；只保留“水位线、连续空闲块、受限 prefix 元数据”作为以后真实 vLLM allocator A/B 的设计提示 |

注意：`lutinayi_branch` 的 `+0.34` 是 warmup、缓存和运行配置的组合差异，不是 prefix caching 或关日志的独立 A/B，因此本交接不虚构单项收益。`wyb` 没有平台成绩，不能用于分数预测。

## Attention：为什么不借代码

1. 当前 vendored ROCm backend 已读取 `num_kv_heads`，计算 `num_queries_per_kv`；HIP paged-attention 内核也直接使用 `gqa_ratio = num_heads / num_kv_heads`。Qwen 的 GQA 并不是未支持状态。
2. `lutinayi_branch` 的 wrapper 只覆写 `_forward_encoder_attention`。Qwen3.5-27B 是 decoder-only，主 Decode paged-attention 不会进入该路径，所以默认关闭是正确的。
3. `wyb` 的 HIP Attention 文件自称 scalar correctness baseline，没有 MFMA/LDS 双缓冲等面向 DCU 的高速实现；它也未由 launch/plugin 接入 vLLM，测试仅覆盖孤立 Python 接口。

结论：本轮继续使用 vLLM 原生 ROCm paged attention。移植任一分支的 Attention 只会增加 dispatch、正确性和构建风险，没有可核验提分证据。

## KV Cache：借机制，不替换 allocator

当前采用两项真实执行链优化：

- `--enable-prefix-caching` 使用 vLLM 自身的 block ownership、hash、refcount 和 eviction 机制；只有评测请求存在完整 block 公共前缀时才会命中，语义不变。默认开，`ENABLE_PREFIX_CACHING=0` 可独立关闭。
- `vllm/v1/worker/block_table.py` 只在新增、移动、交换 block 或暴露可写 CPU 表后标记 dirty；无变化的 decode 步跳过重复 block-table H2D。

不采用 `wyb` 的 16/64/256 “分层 block”与 defrag：其代码只改变 Python 元数据，未复制真实 KV tensor，也未原子更新 vLLM block table；把不同 token 容量映射到同一物理 block id 会破坏真实容量语义。单请求评测下，自定义 LRU/碎片整理的潜在收益也明显低于接入风险。

KV FP8 继续关闭。`lutinayi_branch` 已记录 8–16K 吞吐降至 `7.81 tok/s`、TPOT 约 `117 ms`，说明该 ROCm 路径的量化/反量化成本超过 KV 带宽收益。

## Linear、执行路径与其他取舍

- **保留**：五个 BF16 LLMM1 shape，作为 W8 admission 基准与 fail-open 后备。
- **保留**：选择性 W8；六个 shape 中只接纳 `1.19x–1.53x` 的五个，拒绝 `0.505x` 的 MLP shape。
- **新增**：`--disable-log-stats`，配合已有 `--no-enable-log-requests`，只减少 Python 日志/统计开销，不改变请求语义。
- **不新增 warmup**：`lutinayi_branch` 的两轮全档 warmup 只有组合 `+0.34` 分记录；当前 W8 prefill 还会临时还原 BF16 weight，长 warmup 会放大启动时间和显存峰值风险。
- **不新增 Graph/AITER/scheduler 调参**：缺少 gfx936 当前候选的隔离收益，并可能触碰平台锁定项或改变 SLA 尾延迟。

## W8 已有 microbenchmark

| `(M,K)` | 最终选择 | speedup | 数值结果 |
|---|---|---:|---|
| `(16384,5120)` | W8 | 1.284x | 通过 |
| `(96,5120)` | W8 | 1.332x | 通过 |
| `(14336,5120)` | W8 | 1.219x | 通过 |
| `(5120,6144)` | W8 | 1.529x | 通过 |
| `(34816,5120)` | W8 | 1.192x | 通过 |
| `(5120,17408)` | BF16 | 0.505x | W8 拒绝 |

接纳项 NRMSE 约 `0.00447–0.00458`、cosine 约 `0.99999`。JIT 冷编译约 7 秒，ABI、GPU smoke 和 cache path 已通过。这些数据只证明单 kernel，不证明平台端到端得分。

## 回滚矩阵

| 目的 | 环境变量 |
|---|---|
| 当前平台候选 | `FDU_GFX936_QUANT_MODE=w8 ENABLE_PREFIX_CACHING=1` |
| 只关闭 prefix cache | `ENABLE_PREFIX_CACHING=0` |
| 回到 66.8175 类 BF16/LLMM1 路径 | `FDU_GFX936_QUANT_MODE=off` |
| 完全 stock BF16 linear | `FDU_GFX936_QUANT_MODE=off FDU_FORCE_STOCK_GEMM=1` |

JIT、ABI 或 GPU smoke 失败时 `launch.sh` 自动把量化模式改为 `off`。`--disable-log-stats` 无数值影响，不单独设置回滚变量。

## 提交前只做本地检查

```bash
python3 -m unittest discover -s tests/fdu -p 'test_*.py' -v
python3 -m py_compile \
  scripts/build_gfx936_quant_jit.py \
  scripts/preflight_gfx936_quant.py \
  scripts/bench_gfx936_quant.py \
  vllm/model_executor/layers/gfx936_online_quant.py
bash -n launch.sh scripts/rocm_env.sh scripts/scnet_ab_gfx936.sh
git diff --check
git status --short
git rev-parse HEAD
```

平台提交时绑定 `dx_branch`；GitLab 不稳定时，使用已同步同一提交的 GitHub `main`。平台返回后必须记录：精确 commit、三档 throughput、TTFT P99、TPOT P99、四项精度与最终得分。

## 历史 no-go

- `wvSplitK`：gfx936 benchmark 未通过；BF16 保底继续使用已测 LLMM1。
- AWQ、bitsandbytes、持久 INT4/FP8：规则或 gfx936 kernel 风险，不进入当前路径。
- vendor FP8 / `torch._scaled_mm`：当前设备类没有已证明稳定快路径。
- AITER、KV FP8、旧 GQA/HIP Graph、metadata-only defrag、scheduler 参数：无当前 gfx936 净收益证据或已有负优化。
- 历史 `fdu_vllm/` 插件链：`FDU_ENABLE=0`，不作为平台激活机制。

完整 SCNet 历史操作仅查 [SCNET_RUN.md](SCNET_RUN.md)；本轮不再执行 SCNet。
