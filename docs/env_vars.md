# gfx936 当前环境变量

当前提交以原生 `gfx936`、原始 BF16 checkpoint 和已测 BF16 LLMM1 路径为保底。在线量化必须经过 SCNet 门禁，默认保持关闭。

## 启动变量

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `HIP_VISIBLE_DEVICES` | `0` | 选择单张 DCU |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True` | 降低显存碎片风险 |
| `SAFETENSORS_FAST_GPU` | `1` | 加速 BF16 checkpoint 读取 |
| `VLLM_ROCM_USE_SKINNY_GEMM` | `1` | 允许已验证的 BF16 LLMM1 shape |
| `FDU_FORCE_STOCK_GEMM` | `0` | 设为 `1` 时所有 BF16 linear 回退 stock |
| `VLLM_ROCM_USE_AITER` | `0` | 当前 A/B 禁用 AITER |
| `FDU_ENABLE` | `0` | 禁用历史 `fdu_vllm` 插件钩子 |
| `FDU_GFX936_QUANT_MODE` | `off` | 在线量化模式，见下表 |
| `FDU_CACHE_ROOT` | `/public/home/xdzs2026_c415/cache` | vLLM/Triton/MIOpen 持久缓存根目录 |

`FDU_GFX936_QUANT_MODE` 的允许值：

| 值 | 行为 |
|---|---|
| `off` | 默认；保持已测 BF16/LLMM1 路径 |
| `w8` | 六类精确 shape 逐项做 W8 数值和速度门禁；不通过的 shape 保持 BF16 |
| `hybrid_w4` | 两个 MLP shape 先尝试 group-32 W4，再回退 W8/BF16；其余 shape 尝试 W8 |

未知值会被改为 `off`。JIT 编译、预检或 ABI 失败也会在加载模型前回退 `off`；SCNet 启动脚本会把这种 fail-open 视为候选失败，不会误报候选健康。

## 内部证据变量

| 变量 | 来源 | 作用 |
|---|---|---|
| `FDU_GFX936_QUANT_SO` | `launch.sh` / JIT builder | 指向 `/tmp/fdu_gfx936_quant/<hash>.so`；不应手工固化 |
| `FDU_SOURCE_COMMIT` | `scripts/scnet_ab_gfx936.sh` | 把原仓库的 40 位提交号写入 benchmark JSON；不改变运行逻辑 |

量化 `.so`、packed weight 和 scale 都不是提交模型的一部分：`.so` 仅存 `/tmp`，权重表示仅驻留当前进程显存，不写 checkpoint、模型目录或 SCNet 持久结果目录。

## 主动取消的变量

`scripts/rocm_env.sh` 会执行：

```bash
unset HSA_OVERRIDE_GFX_VERSION ROCBLAS_LAYER
```

不得伪装成 `gfx942`，也不在正式 A/B 中开启 rocBLAS profiling。

## 启动前门禁与回滚

`launch.sh` 在读入 Qwen3.5-27B 前验证当前 venv、原生 `gfx936`、vLLM 扩展以及 BF16 LLMM1 必需符号。量化模式非 `off` 时，还要求 JIT 在 45 秒内完成并通过四个 ABI 符号和 GPU smoke test。

立即回到已测保底路径：

```bash
export FDU_GFX936_QUANT_MODE=off
export FDU_FORCE_STOCK_GEMM=0
bash launch.sh
```

完全回到 stock BF16 linear：

```bash
export FDU_GFX936_QUANT_MODE=off
export FDU_FORCE_STOCK_GEMM=1
bash launch.sh
```
