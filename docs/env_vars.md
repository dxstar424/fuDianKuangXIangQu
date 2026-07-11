# 环境变量说明（评测提交必填）

> v0.4.0 激进冲刺：去 FP8 + 最大化系统配置

## Phase 阶段

| 变量名 | 默认 | 作用 |
|--------|------|------|
| `FDU_PHASE` | `1` | `1`=仅 launch/ROCm；`2`=启用 GQA 等钩子 |

## FDU 优化开关

| 变量名 | v0.4.0 | 作用 |
|--------|--------|------|
| `FDU_ENABLE` | `1` | 总开关 |
| `FDU_KV_CACHE_STRATEGY` | `none` | defrag 未接线，保持 none |
| `FDU_ATTENTION_BACKEND` | `vllm_default` | 走 vLLM 原生后端选择 |
| `FDU_ENABLE_KV_QUANT` | `0` | KV FP8 关 |
| `FDU_ENABLE_PREFIX_CACHE` | `1` | Prefix 缓存 |
| `FDU_ENABLE_GQA_OPT` | `0` | GQA wrap（已证实为死代码路径） |
| `FDU_ENABLE_HIP_GRAPH` | `0` | 不启用（vLLM 原生 HIP Graph 已由 cudagraph_mode 控制） |
| `ENABLE_FP8_WEIGHT_QUANT` | **`0`** | ★ 关 FP8 权重量化——v0.4.0 最关键改动 |

## 启动参数（launch.sh · v0.4.0）

| 变量名 | 默认 | 说明 |
|--------|------|------|
| `MODEL_PATH` | 自动检测 | 模型路径 |
| `PORT` | `8000` | 服务端口 |
| `GPU_MEMORY_UTILIZATION` | **`0.97`** | ★ 激进显存（更大 KV cache 池） |
| `DO_WARMUP` | **`1`** | 启动 warmup |
| `WARMUP_ROUNDS` | `2` | warmup 轮数 |
| `WARMUP_TIER` | **`all`** | 全档 warmup（稳 TTFT P99） |
| `ENABLE_PREFIX_CACHING` | `1` | prefix caching |
| `USE_FDU_SERVER` | `0` | stock api_server |
| `ENFORCE_EAGER` | `0` | ★ 不强制 eager（让 vLLM 用原生 HIP Graph） |
| `COMPILATION_CONFIG` | `{"cudagraph_mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}` | FULL_DECODE_ONLY + 小 batch 图捕获 |
| `LOAD_FORMAT` | `runai_streamer` | 流式快速加载 |
| `HEALTH_TIMEOUT` | `900` | 健康检查超时 |

## ROCm/DCU 系统优化（scripts/rocm_env.sh）

| 变量名 | v0.4.0 | 说明 |
|--------|--------|------|
| `HSA_OVERRIDE_GFX_VERSION` | `9.4.2` | gfx942 kernel 选择 |
| `HIP_VISIBLE_DEVICES` | `0` | 可见 DCU |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True` | 缓解显存碎片 |
| `HSA_ENABLE_SDMA` | `1` | 异步 SDMA |
| `GPU_MAX_HW_QUEUES` | `2` | 硬件队列数 |
| `HIP_FORCE_DEV_KERNARG` | `1` | kernel 参数直传 |
| `TORCH_BLAS_PREFER_HIPBLASLT` | **`0`** | ★ 走 rocBLAS（decode GEMV 更快） |
| `VLLM_ROCM_USE_AITER` | **`0`** | ★★ 关 AITER（无 FP8 时不需要） |
| `VLLM_ROCM_USE_AITER_RMSNORM` | **`1`** | 开 AITER RMSNorm（纯 bf16 加速） |
| `VLLM_ROCM_USE_SKINNY_GEMM` | **`1`** | ★ decode GEMV 手写 HIP kernel（wvSplitK） |
| `SAFETENSORS_FAST_GPU` | `1` | safetensors 快速搬运 |
| `ROCBLAS_LAYER` | **`4`** | rocBLAS 内部自动调优 |
| `MIOPEN_FIND_MODE` | **`1`** | MIOpen 自动寻找最优算法 |
| `HIP_LAUNCH_BLOCKING` | `0` | 异步 kernel launch |

## vLLM CLI 新增参数（v0.4.0）

| 参数 | 值 | 说明 |
|------|-----|------|
| `--block-size` | **`32`** | ★ 页表块大小加倍（16→32），减少 32K 长序列页表遍历开销 |
| `--compilation-config` | `cudagraph_mode=3` | FULL_DECODE_ONLY（decode 图捕获，prefill eager） |

## 已移除

- `ENABLE_FP8_WEIGHT_QUANT=1` + `--quantization fp8` → 16-32K 倒退的根因
- `VLLM_USE_TRITON_FLASH_ATTN` → 非 vLLM 原生 env var，无效 cargo cult 配置
- `VLLM_ROCM_USE_AITER=1` → FP8 GEMM 后端，无 FP8 时不需要
