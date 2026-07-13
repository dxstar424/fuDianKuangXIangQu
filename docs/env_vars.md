# gfx936 BF16 运行环境变量

> 当前提交路径只使用原生 `gfx936` 和 BF16 权重。历史量化模块保留于仓库作实验记录，启动链不会激活它们。

## 活跃变量

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `HIP_VISIBLE_DEVICES` | `0` | 选择单 DCU |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True` | 减少显存碎片 |
| `SAFETENSORS_FAST_GPU` | `1` | 快速读取模型权重 |
| `VLLM_ROCM_USE_SKINNY_GEMM` | `1` | 允许 ROCm skinny GEMM 调度 |
| `FDU_FORCE_STOCK_GEMM` | `0` | 设为 `1` 立即回退原生 BF16 linear |
| `VLLM_ROCM_USE_AITER` | `0` | 隔离 AITER，保持 A/B 单变量 |
| `FDU_ENABLE` | `0` | 关闭历史 FDU 插件钩子 |
| `FDU_CACHE_ROOT` | `/public/home/xdzs2026_c415/cache` | SCNet 持久缓存根目录 |

`scripts/rocm_env.sh` 会主动 `unset HSA_OVERRIDE_GFX_VERSION ROCBLAS_LAYER`。不允许架构伪装，也不开 rocBLAS profiling。

## 启动前门禁

`launch.sh` 在模型加载前调用 `scripts/preflight_rocm.py`，并要求：

- Python、`vllm` 和扩展模块都来自当前安装 venv；
- 原生架构为 `gfx936`；
- `vllm._C` 与 `vllm._rocm_C` 可加载；
- `wvSplitK` 和 `LLMM1` 符号存在。

任一项不符合都在读取 Qwen3.5-27B 前失败。`launch.sh` 在启动服务前清空 `PYTHONPATH`，避免仓库内源码覆盖已安装 wheel。

## 回滚

```bash
export FDU_FORCE_STOCK_GEMM=1
bash launch.sh
```

该回滚不需要重编 wheel，且不改变 BF16 模型、attention 或 scheduler。
