# 环境变量说明（评测提交必填）

> v0.7.0: INT4 AWQ 权重量化 — 突破 60 分瓶颈

## v0.7.0 核心配置

### 权重量化（★ 最关键 ★）

| 变量名 | 值 | 作用 |
|--------|-----|------|
| `VLLM_USE_TRITON_AWQ` | **1** | ★★★ 强制 Triton AWQ dequant kernel（ROCm 安全） |
| `--quantization awq` | CLI flag | ★★★ 启用 AWQ INT4 权重量化 |

权重 HBM IO: 54GB (bf16) → ~14GB (INT4) = **3.75x 理论加速**

### AITER / attention

| 变量名 | 值 | 作用 |
|--------|-----|------|
| `VLLM_ROCM_USE_AITER` | **1** | 启用 AITER HIP FlashAttention（25% GQA 层） |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | **不设** | 跳过 Triton 统一后端，fall through 到 FLASH_ATTN |
| `VLLM_ROCM_USE_SKINNY_GEMM` | **1** | Decode GEMV HIP kernel |
| `VLLM_ROCM_USE_AITER_RMSNORM` | **1** | AITER RMSNorm |

### ROCm 系统

| 变量名 | 值 | 作用 |
|--------|-----|------|
| `HSA_OVERRIDE_GFX_VERSION` | `9.4.2` | gfx942 架构 |
| `HIP_VISIBLE_DEVICES` | `0` | 可见 DCU |
| `GPU_MEMORY_UTILIZATION` | `0.98` | 显存利用率（INT4 释放 35GB 后可提高） |
| `ROCBLAS_LAYER` | `4` | rocBLAS 自调优 |
| `MIOPEN_FIND_MODE` | `1` | MIOpen 自调优 |
| `SAFETENSORS_FAST_GPU` | `1` | 快速 GPU safetensors 加载 |
| `LOAD_FORMAT` | `runai_streamer` | 快速权重加载 |

## 模型

使用 `mattbucci/Qwen3.5-27B-AWQ`（thinking-aware 校准，~18GB）。
在 SCNet 持久存储上通过 huggingface_hub 下载。

## 机制

1. AWQ INT4 权重加载 → vLLM AWQLinearMethod
2. `VLLM_USE_TRITON_AWQ=1` → Triton JIT 编译 dequant+GEMM kernel（避免 C++ kernel ROCm 兼容问题）
3. AITER HIP FlashAttention 处理 GQA 层的 prefill/decode attention
4. DeltaNet 层（75%）走 AWQ INT4 GEMV，GQA 层（25%）走 AITER FlashAttention
