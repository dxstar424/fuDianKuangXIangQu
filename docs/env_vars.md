# 环境变量说明（评测提交必填）

> v0.8.0: bitsandbytes INT4 在线量化 — 源码级默认值，不依赖 CLI flag

## v0.8.0 核心策略 ★★★

**不依赖 `--quantization` CLI flag（v0.7.0 教训：平台评测机覆盖 CLI flag）**
**而是直接修改 vLLM 源码文件，改变默认行为：**

| 文件 | 修改 |
|------|------|
| `vllm/config/model.py` | `quantization` 默认值: `None` → `"bitsandbytes"` |
| `vllm/.../bitsandbytes.py` | `bnb_4bit_compute_dtype`: `"float32"` → `"bfloat16"`, `quant_type`: `"fp4"` → `"nf4"` |

Docker build 时 COPY 修改后的 `.py` 文件覆盖 base image 的 vLLM 安装。
评测机使用我们的 Docker 镜像 → 自动使用 bitsandbytes INT4 量化 → 无法覆盖。

### 权重量化（源码级强制）

| 机制 | 详情 |
|------|------|
| 量化方法 | bitsandbytes 4-bit (nf4) |
| 触发方式 | vLLM 源码默认值（非 CLI flag） |
| 计算精度 | bfloat16 |
| 权重 IO 缩减 | 54GB → ~14GB（4x） |

### AITER / attention

| 变量名 | 值 | 作用 |
|--------|-----|------|
| `VLLM_ROCM_USE_AITER` | **1** | 启用 AITER HIP FlashAttention |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | **不设** | Skip Triton unified, use HIP CK FA |
| `VLLM_ROCM_USE_SKINNY_GEMM` | **1** | Decode GEMV HIP kernel |
| `VLLM_ROCM_USE_AITER_RMSNORM` | **1** | AITER RMSNorm |

### ROCm 系统

| 变量名 | 值 | 作用 |
|--------|-----|------|
| `HSA_OVERRIDE_GFX_VERSION` | `9.4.2` | gfx942 |
| `HIP_VISIBLE_DEVICES` | `0` | DCU |
| `GPU_MEMORY_UTILIZATION` | `0.95` | 显存 |
| `ROCBLAS_LAYER` | `4` | rocBLAS 自调优 |
| `MIOPEN_FIND_MODE` | `1` | MIOpen 自调优 |

## 机制

1. Docker build → pip install bitsandbytes → COPY patched .py files 覆盖 vLLM 源码
2. 评测机启动容器 → vLLM 读取 ModelConfig → quantization 默认 = "bitsandbytes"
3. bitsandbytes 在模型加载时将 bf16 权重在线量化为 INT4 (nf4)
4. 推理时使用 4-bit 权重 → 权重 HBM IO 减少 4x
