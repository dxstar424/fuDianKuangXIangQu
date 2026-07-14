# 优化方案说明

## 当前提交路径（2026-07-15）

目标平台为 vLLM 0.18.1、Qwen3.5-27B BF16、单张原生 `gfx936` DCU。当前提交保留一个已验证保底路径，并加入两个必须经过实机门禁的在线量化候选：

```text
BF16 checkpoint
  -> native gfx936 wheel
  -> native prefix caching + quiet request/stat logging
  -> FDU_GFX936_QUANT_MODE=w8（本次平台盲测默认）
       -> HIP 源随 wheel 编入 vllm._rocm_C
       -> 5 个 microbenchmark 接纳 shape: W8A16 GEMV
       -> (5120,17408): admission 拒绝，stock BF16
       -> ABI/GPU smoke 或零量化 layer: 启动失败
  -> 可选 off / hybrid_w4
       -> load 后逐 shape 在线量化、数值/速度 admission
       -> N=1 自定义 GEMV；prefill 临时还原 BF16
       -> 单 shape 拒绝逐级回退 BF16
```

原运行时 JIT 候选的平台结果为 66.7878，与上一轮 66.8175 在 `0.33%` 内一致。平台没有提供启动日志或 checkout commit，无法判断 W8 是否激活，因此不把它归因为 W8 成绩。下一候选仅消除激活链不确定性：内核编入 wheel、Python 默认 `w8`、自动定位 `_rocm_C`，请求量化但零 layer 激活时停止启动。

## 已记录结果与证据边界

`dx_branch` 最近记录的平台结果：

| 档位/指标 | 实测值 | SLA 扣分 | 精度扣分 |
|---|---:|---:|---:|
| 4–8K 吞吐 | 15.00 tok/s | 0 | 0 |
| 8–16K 吞吐 | 11.97 tok/s | 0 | 0 |
| 16–32K 吞吐 | 6.11 tok/s | 0 | 0 |
| 最终得分 | **66.7878** | 0 | 0 |

评测提交 hash 尚未随结果记录。上一轮为 `66.8175`、`15.03 / 12.00 / 6.09 tok/s`，与本轮的差值是正常波动；两个结果共同构成约 66.8 分的保底参照。缺少同环境 stock/LLMM1 A/B 和 W8 启动证据，不能将增益独立归因于 LLMM1，也不能声称运行时 JIT W8 已生效。

## 优化 1：原生 gfx936 BF16 LLMM1 保底

- wheel 以 `PYTORCH_ROCM_ARCH=gfx936` 构建，不使用 `HSA_OVERRIDE_GFX_VERSION` 伪装架构。
- `vllm/model_executor/layers/rocm_skinny_shapes.py` 只列出 SCNet 数值与性能通过的五个 `(N,M,K,dtype,bias)`。
- dispatch 只在 `N=1`、BF16、无 bias、连续 weight 和 exact shape 下调用 LLMM1。
- `(1,5120,17408)` 因旧 LLMM1 数值失败继续使用 stock BF16。
- `FDU_FORCE_STOCK_GEMM=1` 可无需重编立即回到 stock linear。

当前代码中的该路径是 W8/W4 每个 shape 的速度基准；约 66.8 分仅作为分支保底参照，待补齐评测 commit 后再确认对应关系。

## 优化 2：wheel 内置 W8A16 候选

`FDU_GFX936_QUANT_MODE=w8` 时，启动链执行以下步骤：

1. Docker/wheel 构建将 `csrc/fdu/gfx936_quant_gemv.hip` 编入 `vllm._rocm_C`；
2. 启动时自动定位已安装的 `_rocm_C*.so`，检查四个 C ABI 符号并执行真实 HIP smoke test；
3. BF16 checkpoint 正常加载后，只对六类精确 linear shape 分块生成 per-row W8 与 scale；
4. 每个 `(M,K,W8)` 的第一个真实 layer 对 BF16 基准做同步数值/性能 admission；
5. 接纳后不再保留该 layer 的 BF16 parameter；拒绝则保持原 BF16/LLMM1 路径；
6. N=1 decode 调用 shape-specialized W8A16 HIP GEMV，其他 N 临时还原一个 BF16 weight 后调用现有 linear；
7. 请求量化但模型加载后零 layer 激活时抛错，禁止静默变成 BF16 提交。

W8 门禁：输出有限、NRMSE `<=0.015`、cosine `>=0.999`、相对 BF16/LLMM1 中位延迟至少 `1.10x`。

已有 gfx936 六 shape microbenchmark：

| `(M,K)` | 选择 | speedup | 结果 |
|---|---|---:|---|
| `(16384,5120)` | W8 | 1.284x | 接纳 |
| `(96,5120)` | W8 | 1.332x | 接纳 |
| `(14336,5120)` | W8 | 1.219x | 接纳 |
| `(5120,6144)` | W8 | 1.529x | 接纳 |
| `(34816,5120)` | W8 | 1.192x | 接纳 |
| `(5120,17408)` | BF16 | 0.505x | 拒绝 W8 |

各 W8 项 NRMSE 约 `0.00447–0.00458`、cosine 约 `0.99999`。这只证明逐 shape 数值与 N=1 kernel 速度，不代表端到端 TTFT、TPOT 或任务精度。

## 优化 3：selective group-32 W4 候选

`FDU_GFX936_QUANT_MODE=hybrid_w4` 仅对 `(34816,5120)` 与 `(5120,17408)` 两个 MLP shape 优先使用 group-32 W4A16，其他四类仍尝试 W8。W4 拒绝时依次回退 W8 和 BF16，不会跳过 layer。

W4 门禁：输出有限、NRMSE `<=0.08`、cosine `>=0.995`、相对 BF16 至少 `1.10x`。进入端到端 hybrid 测试前还要求两个 MLP W4 行各自比 W8 microbenchmark 至少快 `1.05x`。

## 优化 4：KV block table 脏提交

`vllm/v1/worker/block_table.py` 原先在每个模型步都执行完整 block table H2D，即使本 token 没有分配新 KV block。当前实现只在 `append_row`、`add_row`、`move_row` 或 `swap_row` 后标记 dirty，并在首次 `commit_block_table` 后清除标记；无变化时直接跳过复制。

默认 16-token block 下，稳定单请求 decode 通常每 16 token 才追加一个 block，因此理论上约 15/16 步可省去该元数据 H2D。它不修改 block size、KV 地址、slot mapping 或 Attention kernel，只减少冗余执行路径；没有 DCU 端到端数据，不声明实际 TPOT 提升。

## 优化 5：原生 prefix cache 与静默统计

`launch.sh` 默认追加 vLLM 官方 `--enable-prefix-caching` 和 `--disable-log-stats`，并保留已有 `--no-enable-log-requests`。prefix cache 使用 vLLM 自身的 block hash、ownership、refcount 与 eviction，不替换 KV allocator；`ENABLE_PREFIX_CACHING=0` 可独立关闭。

该选择来自对 `origin/lutinayi_branch@5fec801` 的审计：该分支最新记录为 `13.04 / 10.08 / 5.78 tok/s`、60.19 分且无 SLA/精度扣分，但 prefix cache、关日志与 warmup 等没有独立 A/B，不能把其组合结果归因于单项。`origin/wyb@f7dac25` 的 KV manager 明确是 metadata-only，Attention 为未接入 launch 的 scalar correctness baseline，报告没有平台吞吐，因此不合入其代码。

当前 ROCm paged-attention 已原生处理 `num_heads / num_kv_heads` 的 GQA ratio；`lutinayi_branch` 的 wrapper 只覆盖 encoder attention，对 decoder-only Qwen 主路径无效。故本轮不移植两个分支的 Attention、KV FP8、defrag、Graph、scheduler 参数或长 warmup。

## 构建、证据与失败处理

HIP 源码不包含 Torch、ATen、pybind11 或 vLLM C++ 头文件，可直接加入 `_rocm_C` 的 HIP source list。正式服务不再运行 hipcc；独立 benchmark 仍可用原 JIT builder 生成 `/tmp` `.so`，用于复现已有 kernel 数据，不进入平台启动链。

六 shape benchmark JSON 记录：

- 原仓库精确 40 位 commit、HIP 源 SHA-256、PyTorch/ROCm 和架构；
- 每个 shape 的请求/选择 kind、packing 时间、NRMSE、cosine；
- BF16 与候选延迟、speedup、峰值显存、admission 原因；
- `allow_nan=False`，任何非有限指标都会失败。

服务门禁不仅检查 `/health`。请求 W8/hybrid 时，启动前必须通过 wheel ABI 与 GPU smoke；模型加载完成后必须至少激活一个量化 layer。单 shape admission 仍可安全回退 BF16，但候选整体不能零层静默成功。

## SCNet 快速验证矩阵

| 阶段 | 最小样本 | 继续条件 |
|---|---:|---|
| wheel ABI | 一次启动 | `_rocm_C` 含四个符号；GPU smoke 通过 |
| W8 六 shape | warmup 2、重复 8 | 六行全过；快速部分路径至少两个 MLP + 两个其他 shape |
| W8 8–16K | 3 条 | `>=12.60 tok/s`；TTFT/TPOT `< baseline×1.45`；无失败 |
| hybrid microbench | 六 shape | 两个 MLP W4 通过，且各自比 W8 `>=1.05x` |
| hybrid 8–16K | 3 条 | 比 W8 `>=1.03x`；同一 SLA 余量 |
| 胜者短验 | 三档各 3 条 | 无档位回退；加权投影高于 66.7878 |
| 抽样精度 | HotpotQA、Retrieval MultiPoint 各 3 条 | 相对保底下降 `<=1%` |

快速样本只用于节省 SCNet 周转时间。平台完整吞吐、P99 和四项精度是最终判断。

## 可复现性与回滚

- 唯一操作手册：[docs/SCNET_RUN.md](docs/SCNET_RUN.md)
- 环境变量：[docs/env_vars.md](docs/env_vars.md)
- 当前交接：[docs/GFX936_HANDOFF.md](docs/GFX936_HANDOFF.md)
- 本次平台候选：`FDU_GFX936_QUANT_MODE=w8`
- 原生 prefix cache：`ENABLE_PREFIX_CACHING=1`；独立回滚为 `0`
- 约 66.8 分回滚：`FDU_GFX936_QUANT_MODE=off`
- 全 stock linear：再设置 `FDU_FORCE_STOCK_GEMM=1`

`scripts/rocm_env.sh`、Dockerfile 与 Python 缺省值同步为 `w8`；若平台 build、SLA 或精度失败，下一提交只需显式恢复 `off`，无需修改 checkpoint。

## 合规声明

- 使用原始 Qwen3.5-27B BF16 checkpoint；不修改、裁剪、跳层或生成持久化量化模型。
- packed weight 与 scale 仅在当前推理进程显存中产生，不写模型目录或持久磁盘；wheel 只包含可执行内核，不包含量化权重。
- 不使用投机解码、预缓存答案、评测期下载或另一模型。
- 不修改平台锁定的 scheduler/batch、温度、最大 token 或请求语义。
- 不伪装 GPU 架构；仅在真实 gfx936 上启用候选。
- 失败回退执行完整 BF16 linear，不改变模型拓扑、层数或自回归语义。

## 历史方案（非当前启动路径）

AWQ/预量化模型、bitsandbytes INT4、vendor FP8、gfx936 `wvSplitK`、AITER、KV FP8、旧 GQA/HIP Graph、metadata-only KV defrag 与 `fdu_vllm` 插件链均已退出当前路线。它们存在规则风险、设备支持不足、数值失败、负优化、未接入真实执行链或激活链不可靠等问题，只保留在 `changelog.md` 与历史设计资料中，不应再用于启动或提交。
