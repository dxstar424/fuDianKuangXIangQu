# gfx936 当前环境变量

当前提交以原生 `gfx936`、原始 BF16 checkpoint 和已测 BF16 LLMM1 路径为保底，并默认启用选择性 W8 平台盲测与 vLLM 原生 prefix caching。W8/W4 HIP ABI 已编入 `vllm._rocm_C`；每个 shape 仍必须通过进程内数值/速度 admission，失败项保持 BF16。

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
| `FDU_GFX936_QUANT_MODE` | `w8` | 选择性在线 W8 平台候选，见下表 |
| `ENABLE_PREFIX_CACHING` | `1` | 启用 vLLM 原生 prefix cache；设为 `0` 可独立回滚 |
| `FDU_CACHE_ROOT` | `/public/home/xdzs2026_c415/cache` | vLLM/Triton/MIOpen 持久缓存根目录 |

`FDU_GFX936_QUANT_MODE` 的允许值：

| 值 | 行为 |
|---|---|
| `off` | 立即回到已测 BF16/LLMM1 路径 |
| `w8` | 默认；六类精确 shape 逐项做 W8 数值和速度门禁；不通过的 shape 保持 BF16 |
| `hybrid_w4` | 两个 MLP shape 先尝试 group-32 W4，再回退 W8/BF16；其余 shape 尝试 W8 |

未知值会被改为 `off`。显式请求 `w8`/`hybrid_w4` 时，wheel ABI 或 GPU smoke 失败会终止启动；模型加载后零量化 layer 也会报错。单个 shape 未通过 admission 时仍安全保持 BF16。

已有 gfx936 microbenchmark 中，W8 接纳 `(16384,5120)`、`(96,5120)`、`(14336,5120)`、`(5120,6144)`、`(34816,5120)`，速度比分别约为 `1.284x`、`1.332x`、`1.219x`、`1.529x`、`1.192x`。`(5120,17408)` 仅 `0.505x`，运行时 admission 会拒绝它并保留 stock BF16。该证据不是端到端平台结果。

## 内部证据变量

| 变量 | 来源 | 作用 |
|---|---|---|
| `FDU_GFX936_QUANT_SO` | 独立 benchmark 可选覆盖 | 正式服务不设置时自动使用已安装的 `vllm._rocm_C`；JIT 微基准可显式传入 `/tmp` `.so` |
| `FDU_SOURCE_COMMIT` | `scripts/scnet_ab_gfx936.sh` | 把原仓库的 40 位提交号写入 benchmark JSON；不改变运行逻辑 |

正式量化内核属于提交 wheel 的 `_rocm_C`，但不包含模型权重。packed weight 和 scale 仅驻留当前进程显存，不写 checkpoint、模型目录或 SCNet 持久结果目录；JIT `.so` 只用于独立 kernel benchmark。

## 主动取消的变量

`scripts/rocm_env.sh` 会执行：

```bash
unset HSA_OVERRIDE_GFX_VERSION ROCBLAS_LAYER
```

不得伪装成 `gfx942`，也不在正式 A/B 中开启 rocBLAS profiling。

## 启动前门禁与回滚

`launch.sh` 在读入 Qwen3.5-27B 前验证当前 venv、原生 `gfx936`、vLLM 扩展以及 BF16 LLMM1 必需符号。量化模式非 `off` 时，还要求 wheel 内 `_rocm_C` 暴露四个 ABI 符号并通过 GPU smoke test；不再启动 hipcc。模型加载后再检查实际量化 layer 数大于零。

`ENABLE_PREFIX_CACHING=1` 只追加 vLLM 官方 `--enable-prefix-caching`，复用引擎自身的 block hash、refcount 与 eviction；不使用历史 `fdu_vllm/kv_cache.py` 或 `wyb` 的 metadata-only allocator。是否提速取决于评测请求是否存在完整 block 公共前缀，当前没有独立 A/B。启动固定追加 `--disable-log-stats`，与已有 `--no-enable-log-requests` 一起减少 Python 日志开销。

立即回到已测保底路径：

```bash
export FDU_GFX936_QUANT_MODE=off
export FDU_FORCE_STOCK_GEMM=0
export ENABLE_PREFIX_CACHING=1
bash launch.sh
```

完全回到 stock BF16 linear：

```bash
export FDU_GFX936_QUANT_MODE=off
export FDU_FORCE_STOCK_GEMM=1
export ENABLE_PREFIX_CACHING=0
bash launch.sh
```
