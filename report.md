# 优化方案说明文档

## 1. 技术路线概述

以 **vLLM 0.18.1** 为基础框架，在 **Qwen3.5-27B (bf16)** 单卡 DCU 环境下，
从 KV Cache 管理、算子优化、执行路径三个维度进行系统级推理优化。
核心目标：**在满足 TTFT / TPOT SLA 硬性约束的前提下，最大化输出吞吐量**。

### 总体思路

```
最终得分 = 吞吐量得分 × 精度系数
    │
    ├── KV Cache 优化 ──── 分级块分配 · 碎片整理 · KV FP8 量化 · Prefix 缓存
    ├── Decode 算子优化 ── HIP FlashAttention · GQA 优化 · MFMA 加速 · LDS 利用
    └── 执行路径优化 ──── HIP Graph 捕获 · 调度批量化 · 异步 SDMA 传输
```

## 2. 优化措施与贡献分析

### 2.1 KV Cache 与显存管理优化

**策略**：分级块分配 + 碎片整理 + Prefix 缓存 + KV FP8 量化

- **分级块分配**：根据上下文长度选择不同块大小（16/64/256），减少短上下文的内碎片和长上下文的块表开销
- **碎片整理**：空闲块占比超过 70% 时触发整理，回收 HBM 连续空间
- **KV FP8 量化**：写入时量化为 FP8 (E4M3)，读取时反量化为 bf16。按 Qwen3.5-27B 参数估算，KV Cache 显存节省约 40-50%，可支撑更大 batch size
- **Prefix 缓存**：检测共享前缀并复用 KV 块，避免重复 prefill

### 2.2 Decode 阶段算子优化

**策略**：手写 HIP FlashAttention kernel（基于 DTK/hipcc + CDNA 架构）

- **LDS double buffering**：利用 64KB/CU 片上共享内存，分 tile 加载 Q/K/V，隐藏 HBM 延迟
- **MFMA 加速**：QK^T 和 PV 计算使用 DCU Matrix Core 的 MFMA 指令（16×16×16），替代标量 dot product
- **Online softmax**：避免写出中间注意力矩阵，降低显存占用
- **128B HBM burst 对齐**：全局内存访问对齐到 HBM burst 边界
- **GQA 优化**：Qwen 的 64 Q heads / 32 KV heads，直接利用 2 queries per KV head

### 2.3 执行路径优化

- **HIP Graph 捕获**：对 Decode 阶段固定 batch 路径进行图捕获，消除逐 step kernel launch 开销
- **调度批量化**：每 4-8 步调度一次，减少 Python 层开销
- **异步 SDMA 传输**：利用 DCU 异步 DMA 引擎重叠数据搬运
- **预热**：服务初始化时运行 dummy 推理，填充 ROCm kernel 缓存

## 3. 优化点汇总表

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
