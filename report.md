# 优化方案说明文档

> 赛程规划：[docs/optimization_roadmap.md](docs/optimization_roadmap.md)  
> 官方解读：[docs/official_guidance_interpretation.md](docs/official_guidance_interpretation.md)

## 1. 技术路线概述

<<<<<<< HEAD
以 **vLLM 0.18.1 (vllm_cscc)** 为基础，通过 `fdu_vllm` 插件合入优化，**不修改 batch scheduler**。
=======
以 **vLLM 0.18.1** 为基础框架，在 **Qwen3.5-27B (bf16)** 单卡 DCU 环境下，
从 KV Cache 管理、算子优化、执行路径三个维度进行系统级推理优化。
核心目标：**在满足 TTFT / TPOT SLA 硬性约束的前提下，最大化输出吞吐量**。
>>>>>>> 47eb201a21f0eb422c50c45ccf05692b555313c7

优化维度（对齐官方 2026-07-09 指导）：

<<<<<<< HEAD
| 阶段 | 瓶颈 | 优化方向 |
|------|------|----------|
| Prefill | 算力（GEMM） | warmup、prefix cache、显存预算 → **TTFT** |
| Decode | 带宽 + KV 读 | GQA、KV 块/defrag、Graph capture、KV FP8 融合 → **TPOT / 吞吐** |
=======
```
最终得分 = 吞吐量得分 × 精度系数
    │
    ├── KV Cache 优化 ──── 分级块分配 · 碎片整理 · KV FP8 量化 · Prefix 缓存
    ├── Decode 算子优化 ── HIP FlashAttention · GQA 优化 · MFMA 加速 · LDS 利用
    └── 执行路径优化 ──── HIP Graph 捕获 · 调度批量化 · 异步 SDMA 传输
```
>>>>>>> 47eb201a21f0eb422c50c45ccf05692b555313c7

## 2. 多阶段实施状态

| 阶段 | 内容 | 状态 | SCNet 指标 |
|------|------|------|------------|
| 0 | SCNet 环境 + baseline | 脚本就绪 | 待实机填充 |
| 1 | launch/ROCm/compile/warmup | **代码完成** | 待 SCNet 门禁 |
| 2 | GQA + prefix + KV block | 已实现 | 待实机填充 |
| 3 | KV FP8 在线量化 | 已实现 | 待精度门禁 |
| 4 | HIP attention + token 验证 | 骨架+fallback | 待实机填充 |
| 5 | HIP Graph | 默认关闭 | opt-in |
| 6 | 文档/提交 | 已更新 | 待平台提交 |

## 3. 优化点与预期贡献

<<<<<<< HEAD
> 官方要求：对每项优化做**量化贡献分析**。SCNet 实测后填入「实测贡献」列。  
> **最有把握项清单**：[docs/easy_scoring.md](docs/easy_scoring.md)（Phase 1 默认开启）。
=======
- **分级块分配**：根据上下文长度选择不同块大小（16/64/256），减少短上下文的内碎片和长上下文的块表开销
- **碎片整理**：空闲块占比超过 70% 时触发整理，回收 HBM 连续空间
- **KV FP8 量化**：写入时量化为 FP8 (E4M3)，读取时反量化为 bf16。按 Qwen3.5-27B 参数估算，KV Cache 显存节省约 40-50%，可支撑更大 batch size
- **Prefix 缓存**：检测共享前缀并复用 KV 块，避免重复 prefill
>>>>>>> 47eb201a21f0eb422c50c45ccf05692b555313c7

### 3.1 Phase 1 最有把握项（相对 stock gpu=0.92）

<<<<<<< HEAD
| 序号 | 优化项 | 默认 | 主攻指标 | 预期 | 实测贡献 |
|------|--------|------|----------|------|----------|
| 1.1 | `gpu_memory_utilization` 0.94 | 开 | 8–16K / 16–32K 吞吐 | 长档 KV 更充裕 | 待填 |
| 1.2 | 分档 warmup（8–16K 优先） | 开 | TTFT P99 | 防首条 SLA 熔断 | 待填 |
| 1.3 | `--enable-prefix-caching` | 开 | TTFT | 共享前缀降 prefill | 待填 |
| 1.4 | disable-log-requests/stats | 开 | TPOT 微降 | 减 Python I/O | 待填 |
| 1.5 | ROCm env（SDMA 等） | 开 | Decode 稳定 | 带宽/分配 | 待填 |
| 1.6 | `FDU_ENABLE_KV_QUANT=0` | 强制 | 精度系数 | 保 k=1.0 | 待填 |
| 1.7 | bf16 + 合规 served-name | 开 | 稳定性 | 与官方权重一致 | — |

### 3.2 Phase 2+（门禁后再开，默认关）
=======
**策略**：手写 HIP FlashAttention kernel（基于 DTK/hipcc + CDNA 架构）

- **LDS double buffering**：利用 64KB/CU 片上共享内存，分 tile 加载 Q/K/V，隐藏 HBM 延迟
- **MFMA 加速**：QK^T 和 PV 计算使用 DCU Matrix Core 的 MFMA 指令（16×16×16），替代标量 dot product
- **Online softmax**：避免写出中间注意力矩阵，降低显存占用
- **128B HBM burst 对齐**：全局内存访问对齐到 HBM burst 边界
- **GQA 优化**：Qwen 的 64 Q heads / 32 KV heads，直接利用 2 queries per KV head
>>>>>>> 47eb201a21f0eb422c50c45ccf05692b555313c7

| 序号 | 优化项 | 阶段 | 主攻指标 | 预期 | 实测贡献 |
|------|--------|------|----------|------|----------|
| 2 | GQA einsum decode | 2 | TPOT P99 | TPOT −5~10% | 待填 |
| 3 | KV defrag/tiered blocks | 2 | TPOT、长档吞吐 | 降碎片、稳 KV 读 | 待填 |
| 4 | HIP Graph capture | 2b | TPOT P99 | 调度开销 −5~15% | 待填 |
| 5 | KV FP8 融合（非独立反量化） | 3 | 长档吞吐 | 显存↓ → 吞吐↑ | 待 A/B |
| 6 | HIP FlashAttention | 4 | Prefill/TTFT | profiling 后 | 暂缓 |

<<<<<<< HEAD
## 4. Baseline 数据记录

> SCNet stock baseline 待重跑（2026-07-08 自测失败：服务未启动）。  
> 平台实测见 [`docs/baseline_result.pdf`](docs/baseline_result.pdf)。
=======
- **HIP Graph 捕获**：对 Decode 阶段固定 batch 路径进行图捕获，消除逐 step kernel launch 开销
- **调度批量化**：每 4-8 步调度一次，减少 Python 层开销
- **异步 SDMA 传输**：利用 DCU 异步 DMA 引擎重叠数据搬运
- **预热**：服务初始化时运行 dummy 推理，填充 ROCm kernel 缓存
>>>>>>> 47eb201a21f0eb422c50c45ccf05692b555313c7

### 4.1 SCNet stock baseline（start_vllm.sh）

<<<<<<< HEAD
| 档位 | TTFT P99 | TPOT P99 | 吞吐 tok/s | SLA |
|------|----------|----------|------------|-----|
| 4-8K (20%) | — | — | — | 待测 |
| 8-16K (50%) | — | — | — | 待测 |
| 16-32K (30%) | — | — | — | 待测 |

### 4.2 竞赛平台实测（富贵花开 · 2026-07-06）

| 档位 | 吞吐 tok/s | SLA扣分 | 精度扣分 |
|------|------------|---------|----------|
| 4-8K (20%) | 18.37 | 0 | 0 |
| 8-16K (50%) | 16.65 | 0 | 0 |
| 16-32K (30%) | 13.49 | 0 | 0 |
| **最终得分** | **84.74 (#26)** | 0 | 0 |

### 4.3 与榜首差距（豆包F4 · #1）

| 档位 | 榜首 | 我们 | 差距 |
|------|------|------|------|
| 8K-16K (50%) | 19.51 | 16.65 | -2.86 |
| 4K-8K (20%) | 21.42 | 18.37 | -3.05 |
| 16K-32K (30%) | 15.05 | 13.49 | -1.56 |

## 5. 评测与门禁命令

```bash
# SCNet Phase 0
bash scripts/scnet_setup.sh
bash scripts/record_baseline.sh

# 每阶段门禁
bash scripts/gate_check.sh quick
bash scripts/gate_check.sh full

# 编译 vLLM + 补丁
bash scripts/compile_vllm.sh

# 启动（评测机 / 本地）
bash launch.sh

# Token 一致性（Phase 4）
python scripts/verify_token_consistency.py --baseline-port 8000 --opt-port 8001
```

## 6. 合规声明

- 未修改 `max-num-seqs`、`max-num-batched-tokens`、batch scheduler
- 未使用投机解码、权重持久化量化、低精度权重缓存
- KV FP8 仅推理期在线量化，不写盘；**保留全部历史 KV，不 skip/delete**
- KV 量化目标为算子融合，避免独立反量化抵消收益
- Graph capture 不改变自回归解码语义
- 未启用 `custom_scheduler`
- 自定义环境变量见 [docs/env_vars.md](docs/env_vars.md)
=======
| 类别 | 优化项 | 方法 | 预期收益 | 状态 |
|------|--------|------|----------|------|
| KV Cache | 分级块分配 | 16/64/256 三级 | 减少内碎片 20-30% | ✅ 已实现 |
| KV Cache | 碎片整理 | 空闲块 >70% 触发 | HBM 连续性提升 | ✅ 已实现 |
| KV Cache | Prefix 缓存 | hash 匹配共享前缀 | 减少 prefill 30-50% | ✅ 已实现 |
| KV Cache | FP8 在线量化 | E4M3 写入/读取 | KV 显存 -40-50% | ✅ 已实现 |
| Decode | HIP FlashAttention | LDS+MFMA+online softmax | TPOT -20-30% | ⚠️ 待 DCU 验证 |
| 调度 | 长度感知调度 | 短请求优先 + prefill/decode 分离 | TTFT P99 改善 | ✅ 已实现 |
| 执行 | HIP Graph | 图捕获 + 回放 | kernel launch 开销 -30% | ✅ 已实现 |
| 推理 | KV 量化 | FP8 在线量化 | 显存利用提升 | ✅ 已实现 |

## 4. 关键代码路径

- `src/plugin.py` — vLLM 集成入口，`apply()` 函数注入全部优化
- `src/config.py` — config.yaml 加载器 + 参数校验
- `src/attention/dcu_attention.py` — DCU attention 后端（HIP + PyTorch fallback）
- `src/attention/hip_kernels/dcu_flash_attn.cpp` — HIP FlashAttention kernel
- `src/kv_cache/block_allocator.py` — 分级块分配器
- `src/kv_cache/cache_manager.py` — Watermark 预算 + Prefix 缓存
- `src/scheduler/custom_scheduler.py` — 长度感知调度器
- `src/quantization/kv_quant.py` — KV FP8 在线量化
- `src/executor/exec_path.py` — HIP Graph 捕获 + 预热
- `src/utils/profiling.py` — RequestProfiler + DCUHardwareProfiler

## 5. 性能数据

（待 DCU 平台 Baseline 评测完成后填入）

| 档位 | Baseline TTFT P99 | Optimized TTFT P99 | Baseline TPOT | Optimized TPOT | Baseline 吞吐 | Optimized 吞吐 | SLA |
|------|------------------|--------------------|---------------|----------------|--------------|----------------|-----|
| 4-8K | - | - | - | - | - | - | - |
| 8-16K | - | - | - | - | - | - | - |
| 16-32K | - | - | - | - | - | - | - |

## 6. 参考资源

- 竞赛官网：https://pra.xtnl.org.cn/（HIP 编程指南、DTK 使用说明）
- vLLM 0.18.1 源码（平台镜像）：http://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc.git
- ROCm 官方文档：https://rocm.docs.amd.com/
>>>>>>> 47eb201a21f0eb422c50c45ccf05692b555313c7
