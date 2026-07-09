# 环境变量说明（评测提交必填）

## Phase 阶段

| 变量名 | 默认 | 作用 |
|--------|------|------|
| `FDU_PHASE` | `1` | `1`=仅 launch/ROCm（Phase 1）；`2+`=启用 GQA/KV/attention 钩子 |

## FDU 优化开关（Phase 2+，Phase 1 默认关）

| 变量名 | Phase 1 默认 | Phase 2+ 默认 | 作用 |
|--------|-------------|--------------|------|
| `FDU_ENABLE` | `1` | `1` | 总开关 |
| `FDU_KV_CACHE_STRATEGY` | `none` | `defrag` | KV 块分配策略 |
| `FDU_ATTENTION_BACKEND` | `vllm_default` | `dcu_optimized` | Attention 路径 |
| `FDU_ENABLE_KV_QUANT` | `0` | `0` | KV 在线 FP8（默认关，保精度） |
| `FDU_ENABLE_PREFIX_CACHE` | `1` | `1` | Prefix 缓存（配合 launch CLI） |
| `FDU_ENABLE_GQA_OPT` | `0` | `1` | GQA einsum 路径 |
| `FDU_ENABLE_HIP_GRAPH` | `0` | `0` | HIP Graph decode |

## 启动参数（launch.sh）

| 变量名 | 默认 | 说明 |
|--------|------|------|
| `MODEL_PATH` | `/data/Qwen3.5-27B` | 模型路径 |
| `PORT` | `8000` | 服务端口 |
| `GPU_MEMORY_UTILIZATION` | `0.94` | 显存利用率（提分关键） |
| `DO_WARMUP` | `1` | 启动后分档 warmup |
| `WARMUP_ROUNDS` | `1` | warmup 轮数 |
| `WARMUP_TIER` | `all` | `all` / `4-8K` / `8-16K` / `16-32K` |
| `ENABLE_PREFIX_CACHING` | `1` | vLLM prefix caching CLI |

## ROCm/DCU（scripts/rocm_env.sh）

| 变量名 | 默认 | 说明 |
|--------|------|------|
| `HIP_PLATFORM` | `amd` | HIP 平台 |
| `HIP_VISIBLE_DEVICES` | `0` | 可见 DCU |
| `GPU_MAX_HW_QUEUES` | `2` | 硬件队列 |
| `HSA_ENABLE_SDMA` | `1` | 异步 SDMA |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True` | 显存分配 |
| `HIPCC_COMPILE_FLAGS_APPEND` | `-O3` | vLLM 编译优化（compile_vllm.sh） |
| `GPU_ARCH` | 自动 | hipcc `--offload-arch` |

## 已移除（违规/无效）

- `FDU_SCHEDULER_POLICY` — 赛题禁止修改 batch scheduler；并发=1 无收益
