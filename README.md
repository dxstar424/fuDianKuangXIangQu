# 先导杯 2026：Qwen3.5-27B 单 DCU 推理优化

本仓库面向 vLLM 0.18.1、Qwen3.5-27B BF16 和单张原生 `gfx936` DCU。目标是在 TTFT/TPOT SLA 与精度门槛内提高输出 token 吞吐。

## 当前路线

当前平台盲测候选是：

```text
原始 BF16 checkpoint
  -> 原生 gfx936 wheel
  -> FDU_GFX936_QUANT_MODE=w8
       -> 5 个 SCNet microbenchmark 已接纳 shape 使用 W8A16 JIT GEMV
       -> (5120, 17408) 自动拒绝并保留 stock BF16 linear
  -> JIT/ABI/smoke/admission 任一失败时回退 BF16/LLMM1
```

`dx_branch` 最近一次记录的平台结果为：

| 指标 | 实测值 |
|---|---:|
| 4–8K 吞吐 | 15.03 tok/s |
| 8–16K 吞吐 | 12.00 tok/s |
| 16–32K 吞吐 | 6.09 tok/s |
| SLA 扣分 | 0 |
| 精度扣分 | 0 |
| 最终得分 | 66.8175 |

评测提交 hash 尚未随结果记录。如果该次评测对应 `88b7d10` 或其后继，则结果包含 5-shape LLMM1；当前将它作为保底参照，但不能把增益独立归因于 LLMM1，后续仍需同环境 stock/LLMM1 A/B。

当前新增候选是在评测机启动时 JIT 编译的小型 HIP 库，并对六类精确 shape 做在线、进程内权重量化：

| `FDU_GFX936_QUANT_MODE` | 行为 |
|---|---|
| `off` | 当前 BF16/LLMM1 保底实现，66.8175 回滚路径 |
| `w8`（默认） | 通过门禁的 shape 使用 W8A16 N=1 GEMV |
| `hybrid_w4` | 两个 MLP shape 优先 group-32 W4，其余使用 W8；逐 shape 失败回退 |

量化候选只在运行进程的显存中生成 packed weight 和 scale，不修改 checkpoint，也不把量化权重写入模型目录或持久目录。N=1 decode 使用自定义 HIP GEMV；prefill 临时还原 BF16 后走现有 linear。编译、ABI、数值或速度门禁失败时保持 BF16 路径。

W8 六 shape microbenchmark 已得到 5 个接纳、1 个拒绝；接纳项相对当前 BF16/LLMM1 为 `1.19x–1.53x`，`(5120,17408)` 仅 `0.505x`，因此明确保持 BF16。按本轮“停止 SCNet、直接平台盲测”的决策，默认改为选择性 `w8`；尚无端到端吞吐、SLA 或精度结论，不能把 microbenchmark 写成平台提分。

Decode 执行路径另加 KV block table 脏提交：只有新增、移动或交换 KV block 时才把 CPU block table 复制到 GPU。默认 block size 为 16，稳定单请求 decode 通常可跳过约 15/16 次重复 H2D；不改变 KV 内容、块大小、Attention 数值或分配策略。该项只有静态/状态机测试，平台收益仍待跑分。

## 关键实现

```text
launch.sh
scripts/rocm_env.sh
scripts/scnet_ab_gfx936.sh
scripts/build_gfx936_quant_jit.py
scripts/preflight_gfx936_quant.py
scripts/bench_gfx936_quant.py

csrc/fdu/gfx936_quant_gemv.hip
vllm/model_executor/layers/gfx936_online_quant.py
vllm/model_executor/layers/linear.py
vllm/model_executor/layers/utils.py
vllm/model_executor/layers/rocm_skinny_shapes.py
vllm/v1/worker/block_table.py

docs/SCNET_RUN.md
docs/GFX936_HANDOFF.md
docs/env_vars.md
report.md
```

仓库根目录的 `fdu_vllm/` 和 `src/` 保留历史实验代码；当前启动链设置 `FDU_ENABLE=0`，不依赖这些插件钩子。

## 本地静态验证

macOS 没有 ROCm/DCU，只运行不依赖 GPU 的契约测试：

```bash
python3 -m unittest discover -s tests/fdu -p 'test_*.py' -v
python3 -m py_compile \
  scripts/build_gfx936_quant_jit.py \
  scripts/preflight_gfx936_quant.py \
  scripts/bench_gfx936_quant.py \
  vllm/model_executor/layers/gfx936_online_quant.py
bash -n launch.sh scripts/rocm_env.sh scripts/scnet_ab_gfx936.sh
```

真实 kernel、模型加载、吞吐、SLA 和精度只能在 SCNet `gfx936` 上判断。

## 安全边界

- 不修改或持久化量化 Qwen3.5-27B checkpoint。
- 不使用投机解码、剪枝、层跳过、预缓存答案或评测期下载。
- 不修改平台锁定的 batch/scheduler、采样和 token 参数。
- 不设置 `HSA_OVERRIDE_GFX_VERSION`，只编译和运行原生 `gfx936`。
- JIT 产物只在 `/tmp/fdu_gfx936_quant/`，按源文件、编译器和参数哈希，进程/容器结束后可丢弃。
- `FDU_FORCE_STOCK_GEMM=1` 可立即禁用 BF16 LLMM1；`FDU_GFX936_QUANT_MODE=off` 可立即禁用在线量化。

## 已放弃的活跃方向

AWQ/预量化模型、bitsandbytes INT4、FP8 vendor 路径、gfx936 `wvSplitK`、AITER、KV FP8、自定义 GQA/HIP Graph 和 scheduler 调参都不是当前提交路径。原因包括规则风险、gfx936 支持不足、数值失败、实测负优化或无法隔离变量。它们只保留在 [changelog.md](changelog.md) 与历史设计文档中供复盘。

## 文档入口

- [SCNet 最快测试流程](docs/SCNET_RUN.md)
- [gfx936 当前状态与交接](docs/GFX936_HANDOFF.md)
- [环境变量](docs/env_vars.md)
- [平台提交检查清单](docs/submit_checklist.md)
- [优化方案与实测记录](report.md)
- [变更历史](changelog.md)
