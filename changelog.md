# 变更日志

## [v0.2.0] - 2026-07-08

### 改动
- `dcu_flash_attn.cpp`: 修复 `extern "C" __global__` HIP 语法错误。改为 C++ name-mangling kernel + C-linkage host wrapper，Python 通过 `ctypes` 调用 wrapper 而非 kernel
- `dcu_attention.py`: 改用 ctypes 调用 host wrapper，传递 hipStream_t 参数，增加错误码检查
- `plugin.py`: 重写 monkey-patch 逻辑，增加 vLLM 0.18.x 多路径兼容 fallback（selector / registry / scheduler._schedule / scheduler.schedule）
- `launch.sh`: 新增 CLI 参数解析（`--model` / `--port` / `--tensor-parallel-size`），匹配平台评测启动格式
- `config.yaml`: 重命名 `use_cuda_graph`→`use_hip_graph`，`cuda_graph_max_bs`→`hip_graph_max_bs`
- `requirements.txt`: 删除 `nvidia-ml-py`（NVIDIA 专用，DCU 不可用）
- `Dockerfile`: 更新注释、新增 `COPY docs/` 以符合提交规范
- `compile_kernels.sh`: 切换到 hipcc 独立编译 + ctypes 加载方案
- `config.py`: 新增 config.yaml 加载器 + 参数校验

### 预期效果
- HIP kernel 语法符合 DTK 编译要求
- launch.sh 可通过平台 CLI 正常调用
- vLLM 集成兼容多种 0.18.x 路径

### 实际结果
- 本地 Python 语法编译通过，待 DCU 实机验证

---

## [v0.1.1] - 2026-07-07

### 改动
- `exec_path.py`: 添加 HIP/ROCm API 映射注释
- `dcu_attention.py`: 双路径架构（HIP JIT + PyTorch fallback）
- `profiling.py`: 新增 DCUHardwareProfiler

---

## [v0.1.0] - 2026-07-06

### 改动
- 项目初始化：目录结构、Dockerfile、模块骨架、Baseline 评测工具
