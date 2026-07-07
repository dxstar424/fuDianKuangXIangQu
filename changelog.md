# 变更日志

## [v0.1.1] - 2026-07-07

### 改动
- `exec_path.py`: 添加完整的 HIP/ROCm API 映射注释，消除 CUDA 语义歧义
- `dcu_attention.py`: 重写为双路径架构（HIP JIT kernel + PyTorch fallback），
  加入 ROCm 环境检测和 kernel JIT 编译框架
- `hip_kernels/dcu_flash_attn.cpp`: 新增 HIP FlashAttention kernel 骨架，
  基于 LDS + MFMA 指令 + online softmax
- `profiling.py`: 新增 DCUHardwareProfiler（rocprof + rocm-smi 集成）
- `env_vars.md`: 新增 ROCm/DCU 环境变量说明

### 预期效果
- 代码语义从 CUDA 切换为 HIP/ROCm/DTK 原生
- 补齐 DCU profiling 工具链
- HIP kernel 框架就绪，待实机编译验证

### 实际结果
- 本地 Python 编译通过，HIP C++ 待 DCU 实机 hipcc 编译验证

---

## [v0.1.0] - 2026-07-06

### 改动
- 项目初始化：目录结构、Dockerfile、模块骨架、Baseline 评测工具

### 预期效果
- 基础框架就绪
