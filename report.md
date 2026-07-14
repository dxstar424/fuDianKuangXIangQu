# 优化方案说明

## 当前提交路径（2026-07-14）

目标平台为 vLLM 0.18.1、Qwen3.5-27B BF16、单张原生 `gfx936` DCU。当前提交保留一个已验证保底路径，并加入两个必须经过实机门禁的在线量化候选：

```text
BF16 checkpoint
  -> native gfx936 wheel
  -> FDU_GFX936_QUANT_MODE=off（默认）
       -> 5 个实测 N=1 shape: BF16 LLMM1
       -> 其他 shape: stock BF16 linear
  -> 可选 w8 / hybrid_w4
       -> 启动时 JIT 小型 HIP 库
       -> load 后逐 shape 在线量化、数值/速度 admission
       -> N=1 自定义 GEMV；prefill 临时还原 BF16
       -> 任一失败逐级回退 BF16
```

默认值保持 `off`，因为 W8/W4 尚未完成 SCNet 端到端验证。未取得实测数据前，本报告不把理论带宽收益写成实际贡献。

## 已验证结果

当前 BF16/LLMM1 保底路径的平台结果：

| 档位/指标 | 实测值 | SLA 扣分 | 精度扣分 |
|---|---:|---:|---:|
| 4–8K 吞吐 | 15.03 tok/s | 0 | 0 |
| 8–16K 吞吐 | 12.00 tok/s | 0 | 0 |
| 16–32K 吞吐 | 6.09 tok/s | 0 | 0 |
| 最终得分 | **66.8175** | 0 | 0 |

与历史 BF16 平台记录 12.92 / 10.04 / 5.77 tok/s、59.97 分相比，当前已测路径分别提高约 16.3%、19.5%、5.5%，得分提高约 11.4%。该对比仅用于记录迭代贡献；在线量化候选的贡献当前记为“待 SCNet”，不是已验证提升。

## 优化 1：原生 gfx936 BF16 LLMM1 保底

- wheel 以 `PYTORCH_ROCM_ARCH=gfx936` 构建，不使用 `HSA_OVERRIDE_GFX_VERSION` 伪装架构。
- `vllm/model_executor/layers/rocm_skinny_shapes.py` 只列出 SCNet 数值与性能通过的五个 `(N,M,K,dtype,bias)`。
- dispatch 只在 `N=1`、BF16、无 bias、连续 weight 和 exact shape 下调用 LLMM1。
- `(1,5120,17408)` 因旧 LLMM1 数值失败继续使用 stock BF16。
- `FDU_FORCE_STOCK_GEMM=1` 可无需重编立即回到 stock linear。

该路径是 66.8175 分结果的运行基础，也是 W8/W4 每个 shape 的速度基准。

## 优化 2：运行时 W8A16 候选

`FDU_GFX936_QUANT_MODE=w8` 时，启动链执行以下步骤：

1. 在 45 秒超时内将 `csrc/fdu/gfx936_quant_gemv.hip` 编译到 `/tmp/fdu_gfx936_quant/<hash>.so`；
2. 检查四个 C ABI 符号并执行真实 HIP smoke test；
3. BF16 checkpoint 正常加载后，只对六类精确 linear shape 分块生成 per-row W8 与 scale；
4. 每个 `(M,K,W8)` 的第一个真实 layer 对 BF16 基准做同步数值/性能 admission；
5. 接纳后不再保留该 layer 的 BF16 parameter；拒绝则保持原 BF16/LLMM1 路径；
6. N=1 decode 调用 shape-specialized W8A16 HIP GEMV，其他 N 临时还原一个 BF16 weight 后调用现有 linear。

W8 门禁：输出有限、NRMSE `<=0.015`、cosine `>=0.999`、相对 BF16/LLMM1 中位延迟至少 `1.10x`。

## 优化 3：selective group-32 W4 候选

`FDU_GFX936_QUANT_MODE=hybrid_w4` 仅对 `(34816,5120)` 与 `(5120,17408)` 两个 MLP shape 优先使用 group-32 W4A16，其他四类仍尝试 W8。W4 拒绝时依次回退 W8 和 BF16，不会跳过 layer。

W4 门禁：输出有限、NRMSE `<=0.08`、cosine `>=0.995`、相对 BF16 至少 `1.10x`。进入端到端 hybrid 测试前还要求两个 MLP W4 行各自比 W8 microbenchmark 至少快 `1.05x`。

## 编译、证据与失败处理

JIT 源码不包含 Torch、ATen、pybind11 或 vLLM C++ 头文件；编译参数固定为原生 gfx936，并以源码、编译器身份和 flags 的哈希作为 `/tmp` 缓存键。临时文件成功后原子 rename，超时/失败会清理半成品。

六 shape benchmark JSON 记录：

- 原仓库精确 40 位 commit、HIP 源 SHA-256、PyTorch/ROCm 和架构；
- 每个 shape 的请求/选择 kind、packing 时间、NRMSE、cosine；
- BF16 与候选延迟、speedup、峰值显存、admission 原因；
- `allow_nan=False`，任何非有限指标都会失败。

服务门禁不仅检查 `/health`。请求 W8/hybrid 时，日志必须声明完全相同的 `quant_mode`，且不能出现 Traceback、OOM、非有限 admission 或 `keeping BF16 path`。否则脚本停止服务并返回失败，避免把静默回退当成候选结果。

## SCNet 快速验证矩阵

| 阶段 | 最小样本 | 继续条件 |
|---|---:|---|
| JIT | 首次 + 2 次 cache hit | 首次 `<=50s`；同一 `.so` 路径；ABI/smoke 通过 |
| W8 六 shape | warmup 2、重复 8 | 六行全过；快速部分路径至少两个 MLP + 两个其他 shape |
| W8 8–16K | 3 条 | `>=12.60 tok/s`；TTFT/TPOT `< baseline×1.45`；无失败 |
| hybrid microbench | 六 shape | 两个 MLP W4 通过，且各自比 W8 `>=1.05x` |
| hybrid 8–16K | 3 条 | 比 W8 `>=1.03x`；同一 SLA 余量 |
| 胜者短验 | 三档各 3 条 | 无档位回退；加权投影高于 66.8175 |
| 抽样精度 | HotpotQA、Retrieval MultiPoint 各 3 条 | 相对保底下降 `<=1%` |

快速样本只用于节省 SCNet 周转时间。平台完整吞吐、P99 和四项精度是最终判断。

## 可复现性与回滚

- 唯一操作手册：[docs/SCNET_RUN.md](docs/SCNET_RUN.md)
- 环境变量：[docs/env_vars.md](docs/env_vars.md)
- 当前交接：[docs/GFX936_HANDOFF.md](docs/GFX936_HANDOFF.md)
- 在线量化关闭：`FDU_GFX936_QUANT_MODE=off`
- 全 stock linear：再设置 `FDU_FORCE_STOCK_GEMM=1`

在 SCNet 证据确认胜者前，`scripts/rocm_env.sh` 与 Dockerfile 都保持 `FDU_GFX936_QUANT_MODE=off`。

## 合规声明

- 使用原始 Qwen3.5-27B BF16 checkpoint；不修改、裁剪、跳层或生成持久化量化模型。
- packed weight 与 scale 仅在当前推理进程显存中产生，不写模型目录或持久磁盘；JIT `.so` 仅存 `/tmp`。
- 不使用投机解码、预缓存答案、评测期下载或另一模型。
- 不修改平台锁定的 scheduler/batch、温度、最大 token 或请求语义。
- 不伪装 GPU 架构；仅在真实 gfx936 上启用候选。
- 失败回退执行完整 BF16 linear，不改变模型拓扑、层数或自回归语义。

## 历史方案（非当前启动路径）

AWQ/预量化模型、bitsandbytes INT4、vendor FP8、gfx936 `wvSplitK`、AITER、KV FP8、旧 GQA/HIP Graph 与 `fdu_vllm` 插件链均已退出当前路线。它们存在规则风险、设备支持不足、数值失败、负优化或激活链不可靠等问题，只保留在 `changelog.md` 与历史设计资料中，不应再用于启动或提交。
