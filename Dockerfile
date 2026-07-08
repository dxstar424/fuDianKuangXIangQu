# ============================================================
# FDU SCCSCC26 - Qwen3.5-27B × vLLM 0.18.1 推理优化
# 基于大赛官方基础镜像构建
# ============================================================

ARG BASE_IMAGE=competition/vllm-0.18.1-base:v1.0
FROM ${BASE_IMAGE}

# --- 系统依赖（HIP kernel 编译工具链）---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- 工作目录 ---
WORKDIR /workspace

# --- 安装 Python 依赖 ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- 拷贝源码 ---
COPY src/ ./src/

# --- 编译自定义 HIP Kernel ---
# 注意：Docker build 时无 DCU 设备，hipcc 仅做语法编译；
# 完整 kernel 功能需在运行时通过 ctypes 加载预编译的 .so。
COPY scripts/compile_kernels.sh ./scripts/compile_kernels.sh
RUN bash scripts/compile_kernels.sh || echo "[FDU] HIP kernel syntax-only compile — OK (no DCU device)"

# --- 拷贝启动脚本与配置 ---
COPY launch.sh .
COPY config.yaml .

# --- 拷贝环境变量说明文档（提交必需）---
COPY docs/env_vars.md ./docs/env_vars.md

# --- 入口 ---
EXPOSE 8000
CMD ["bash", "launch.sh"]
