# 环境变量说明（评测提交必填）

> v0.5.0: AITER unified attention — 极简配置

## v0.5.0 核心配置

| 变量名 | 值 | 作用 |
|--------|-----|------|
| `VLLM_ROCM_USE_AITER` | **1** | ★ 启用 AMD AITER 优化算子库 |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | **1** | ★★ 使用 AITER 统一注意力后端（替代默认 TRITON_ATTN） |
| `HSA_OVERRIDE_GFX_VERSION` | `9.4.2` | gfx942 架构 override |
| `HIP_VISIBLE_DEVICES` | `0` | 可见 DCU |
| `GPU_MEMORY_UTILIZATION` | `0.95` | 显存利用率 |

## vLLM CLI 参数

所有参数保持 vLLM 默认，不设置特殊 compilation-config / block-size / quantization 等。

## 机制

AITER unified attention 是 vLLM ROCm 平台的最高优先级 attention 后端。
启用后 vLLM 自动：
1. 选择 `ROCM_AITER_UNIFIED_ATTN` 后端（替代 TRITON_ATTN）
2. 设置 `block_size=64`（AITER backend 要求）
3. 使用 AITER 的 FlashAttention HIP kernel 处理 prefill/decode attention
