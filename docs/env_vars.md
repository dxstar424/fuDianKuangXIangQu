# 环境变量说明（评测提交必填）

> 冲刺阶段配额见 [sprint_strategy_0711.md](./sprint_strategy_0711.md)

## Phase 阶段

| 变量名 | 默认 | 作用 |
|--------|------|------|
| `FDU_PHASE` | `1` | `1`=仅 launch/ROCm（S1/S2）；`2`=启用 GQA 等钩子（S3+） |

## FDU 优化开关

| 变量名 | Phase 1 / S1 | Phase 2 / S3 | 作用 |
|--------|--------------|--------------|------|
| `FDU_ENABLE` | `1` | `1` | 总开关 |
| `FDU_KV_CACHE_STRATEGY` | `none` | **`none`** | defrag **未接线**，保持 none |
| `FDU_ATTENTION_BACKEND` | `vllm_default` | `vllm_default` | GQA wrap stock selector |
| `FDU_ENABLE_KV_QUANT` | `0` | `0` | KV FP8 默认关 |
| `FDU_ENABLE_PREFIX_CACHE` | `1` | `1` | Prefix 缓存 |
| `FDU_ENABLE_GQA_OPT` | `0` | **`1`** | GQA selector wrap（已接线） |
| `FDU_ENABLE_HIP_GRAPH` | `0` | `0` | 仅 S4；须 `ENFORCE_EAGER=0` |
| `FDU_ENABLE_FLASH_ATTN` | `0` | **`1`** | HIP FlashAttention prefill kernel（v0.2.19+） |
| `ENABLE_FP8_WEIGHT_QUANT` | **`1`** | **`1`** | **v0.3.0** FP8 W8A8 在线权重量化（权重 HBM IO 减半） |

## 启动参数（launch.sh · S1 Recover）

| 变量名 | 默认 | 说明 | 配置原因 |
|--------|------|------|----------|
| `MODEL_PATH` | 自动：`/root`→`/data`→`$HOME` | 模型路径 | SCNet `/root` 加载更快 |
| `PORT` | `8000` | 服务端口 | 评测机默认 |
| `GPU_MEMORY_UTILIZATION` | **`0.94`** | 显存利用率 | **禁止默认 0.95** |
| `DO_WARMUP` | **`0`** | 启动 warmup | 平台默认关；S2 可试 |
| `WARMUP_ROUNDS` | `1` | warmup 轮数 | — |
| `WARMUP_TIER` | `8-16K` | warmup 档位 | 仅 DO_WARMUP=1 |
| `ENABLE_PREFIX_CACHING` | `1` | prefix caching | 低风险 |
| `USE_FDU_SERVER` | **`0`** | `1`=fdu_vllm.server | S1/S2 stock；**S3 起用 1** |
| `ENFORCE_EAGER` | **`1`** | `--enforce-eager` | S2 可 A/B 关 |
| `ENABLE_FP8_WEIGHT_QUANT` | **`1`** | v0.3.0 FP8 W8A8 在线权重量化 | 权重 HBM IO 减半 (45ms→22.5ms) |
| `HEALTH_TIMEOUT` | `900` | 健康检查超时 | 大模型加载慢 |

## ROCm/DCU（scripts/rocm_env.sh）

| 变量名 | 默认 | 说明 |
|--------|------|------|
| `HIP_PLATFORM` | `amd` | HIP 平台 |
| `HIP_VISIBLE_DEVICES` | `0` | 可见 DCU |
| `GPU_MAX_HW_QUEUES` | `2` | 硬件队列 |
| `HSA_ENABLE_SDMA` | `1` | 异步 SDMA |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True` | 显存分配 |
| `HIPCC_COMPILE_FLAGS_APPEND` | `-O3` | vLLM 编译优化 |
| `GPU_ARCH` | 自动 | hipcc `--offload-arch` |
| `VLLM_ROCM_USE_AITER` | **`1`** | v0.3.0 开（FP8 W8A8 GEMM 走 AITER Triton BMM） |
| `VLLM_ROCM_USE_SKINNY_GEMM` | `1` | ROCm FP8 scaled_mm kernel（decode skinny GEMM） |

## 已移除（违规/无效）

- `FDU_SCHEDULER_POLICY` — 赛题禁止修改 batch scheduler；并发=1 无收益
