# gfx936 在线量化路线交接

> 更新：2026-07-14
> 目标：Qwen3.5-27B BF16、vLLM 0.18.1、单张原生 `gfx936` DCU。

## 一页结论

- `dx_branch` 已记录平台得分 66.8175，SLA 与精度扣分均为 0；评测提交 hash 尚未随结果记录。若对应 `88b7d10` 或其后继，则包含原生 gfx936 + BF16 + 5 个 N=1 LLMM1 shape，但不能量化 LLMM1 的独立贡献。
- 在线 W8 与 selective W4 的实现已合入主工作区，代码锚点为 `1dc9f46`；交付提交请以 `git rev-parse HEAD` 为准。
- `FDU_GFX936_QUANT_MODE` 默认仍是 `off`。W8/W4 尚无 SCNet 端到端数据，不能声称已提速。
- 最快验证顺序是：复用 wheel → 45 秒 JIT 门禁 → 六 shape W8 → 一次 8–16K 三样本 → 有收益才试 hybrid → 胜者短验。
- 完整可复制命令只认 [SCNET_RUN.md](SCNET_RUN.md)。

## 已记录的保底参照

| 指标 | 平台实测 |
|---|---:|
| 4–8K 吞吐 | 15.03 tok/s |
| 8–16K 吞吐 | 12.00 tok/s |
| 16–32K 吞吐 | 6.09 tok/s |
| SLA 扣分 | 0 |
| 精度扣分 | 0 |
| 最终得分 | 66.8175 |

该结果可用于本轮快速候选的方向比较。由于缺少平台评测 commit 与同环境 stock/LLMM1 A/B，不能把 66.8175 或相对旧结果的差值独立归因于 LLMM1。

`vllm/model_executor/layers/rocm_skinny_shapes.py` 中有五个 SCNet 已验证 BF16 shape。`(N=1, M=5120, K=17408)` 的旧 LLMM1 数值测试失败，因此保底模式对它使用 stock BF16 linear。

## 当前候选设计

| 模式 | Decode `N=1` | Prefill/其他 N | 失败处理 |
|---|---|---|---|
| `off` | 已测 BF16 LLMM1/stock | BF16 | 保底 |
| `w8` | 六类精确 shape 逐项尝试 W8A16 HIP GEMV | 临时还原 BF16 后走现有 linear | 逐 shape 回退 BF16 |
| `hybrid_w4` | 两个 MLP shape 先尝试 group-32 W4，其他尝试 W8 | 临时还原 BF16 | W4 → W8 → BF16 |

候选 shape 为：

```text
(16384, 5120)  (96, 5120)  (14336, 5120)
(5120, 6144)   (34816, 5120)  (5120, 17408)
```

`scripts/build_gfx936_quant_jit.py` 将不依赖 Torch C++ 头文件的小型 HIP 源码编译到 `/tmp/fdu_gfx936_quant/<hash>.so`，超时为 45 秒。packed weight 与 scale 只存在当前推理进程的显存中，不写 checkpoint、模型目录或持久缓存。

## 关键安全合同

1. 只接受真实 `gfx936`；`scripts/rocm_env.sh` 主动取消 `HSA_OVERRIDE_GFX_VERSION`。
2. 启动仍固定 `--dtype bfloat16`，checkpoint 不转换、不替换。
3. JIT、ABI 或 GPU smoke 失败时 `launch.sh` 回退 `off`。
4. `scripts/scnet_ab_gfx936.sh` 要求服务日志声明的模式与请求模式完全一致；如果日志包含 `keeping BF16 path`，即使 `/health` 成功也判候选失败并停止进程。
5. 首个真实 layer 对每个 `(M,K,kind)` 做数值与速度门禁，之后才复用决策；未通过的 layer 保留原 BF16 parameter。
6. benchmark JSON 记录原仓库精确 40 位提交号和 HIP 源文件 SHA-256；实验副本不含 `.git` 也不会丢失 provenance。
7. `FDU_GFX936_QUANT_MODE=off` 无需重编即可回到 66.8175 路径；`FDU_FORCE_STOCK_GEMM=1` 可进一步禁用 BF16 LLMM1。

## 实现地图

| 文件 | 责任 |
|---|---|
| `csrc/fdu/gfx936_quant_gemv.hip` | W8/W4 GEMV 与 BF16 重构 C ABI |
| `scripts/build_gfx936_quant_jit.py` | 45 秒超时、哈希、原子 `/tmp` JIT 缓存 |
| `scripts/preflight_gfx936_quant.py` | 四符号 ABI 与小 shape GPU smoke |
| `scripts/bench_gfx936_quant.py` | 六 shape 正确性、速度、显存和 provenance JSON |
| `vllm/model_executor/layers/gfx936_online_quant.py` | mode/shape policy、packing、admission、ctypes runtime |
| `vllm/model_executor/layers/linear.py` | load 后在线转换与 apply 分派 |
| `vllm/model_executor/layers/utils.py` | opaque Torch custom op 和 BF16 LLMM1 fallback |
| `vllm/model_executor/model_loader/utils.py` | 转换完成后释放 allocator cache |
| `launch.sh` | JIT/preflight/fail-open 启动合同 |
| `scripts/scnet_ab_gfx936.sh` | 可复现的 off/W8/hybrid 服务与评测 wrapper |

## SCNet 快速决策

先执行 [SCNET_RUN.md](SCNET_RUN.md) 的 0–4 节。W8 六 shape 全通过最理想；为了快速筛选，至少要求两个 MLP shape 和另外两个 shape 通过，其他 shape 必须明确保持 BF16。

W8 的第一轮端到端门槛：

- 8–16K 三样本吞吐 `>= 12.60 tok/s`；
- TTFT/TPOT P99 均 `< baseline × 1.45`；
- 无 OOM、Traceback、非有限指标或静默回退；
- 生成成功的服务 probe。

只有 W8 有净收益时才测 hybrid。hybrid 需要两个 MLP W4 行通过 W4 数值门禁、各自比 W8 microbenchmark 快至少 1.05x，并且 8–16K 端到端再提高至少 3%。

最终候选再跑三档各 3 条，以及 HotpotQA、Retrieval MultiPoint 各 3 条。选项只有：

```text
hybrid_w4 明确胜出 -> 选 hybrid_w4
否则 w8 通过       -> 选 w8
否则               -> 保持 off
```

在完成以上证据前，不修改 Dockerfile 与 `scripts/rocm_env.sh` 的默认 `off`。

## 需要带回的证据

- `/tmp/fdu_gfx936_quant_compile.log`
- `/tmp/fdu_gfx936_quant_w8.json`
- `/tmp/fdu_gfx936_quant_w8.log`
- `/tmp/fdu_gfx936_w8.log`
- `results/gfx936_skinny/throughput/w8-fast/8-16K.json`
- 若测 hybrid，再带回 hybrid JSON、服务日志和 8–16K result JSON

JSON 中必须能看到 `git_commit`、`hip_source_sha256`、设备/ROCm 信息、每个 shape 的 kind、NRMSE、cosine、BF16/candidate latency、speedup、峰值显存与 admission 原因。

## 历史 no-go

- `wvSplitK`：gfx936 直接 benchmark 未通过，当前不再扩展；BF16 保底用已测 LLMM1。
- AWQ/预量化模型、bitsandbytes、持久 INT4/FP8：不属于当前路径，存在规则或 gfx936 kernel 风险。
- vendor FP8 / `torch._scaled_mm`：本设备类没有已证明可用的稳定快路径。
- AITER、KV FP8、旧 GQA/HIP Graph、scheduler 调参：未证明净收益且会污染本轮单变量判断。
- 历史 `fdu_vllm/` 插件链：当前 `FDU_ENABLE=0`，不作为平台激活机制。

旧实验仍保留在 `changelog.md` 和 `docs/superpowers/` 供复盘，但不应再用于 SCNet 操作。
