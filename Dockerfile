# ============================================================
# FDU SCCSCC26 - Qwen3.5-27B × vLLM 0.18.1 推理优化
# ============================================================

ARG BASE_IMAGE=competition/vllm-0.18.1-base:v1.0
FROM ${BASE_IMAGE}

<<<<<<< HEAD
=======
# --- 系统依赖（HIP kernel 编译工具链）---
>>>>>>> 47eb201a21f0eb422c50c45ccf05692b555313c7
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    rsync \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
<<<<<<< HEAD
COPY patches/ ./patches/
COPY scripts/ ./scripts/
=======

# --- 编译自定义 HIP Kernel ---
# 注意：Docker build 时无 DCU 设备，hipcc 仅做语法编译；
# 完整 kernel 功能需在运行时通过 ctypes 加载预编译的 .so。
COPY scripts/compile_kernels.sh ./scripts/compile_kernels.sh
RUN bash scripts/compile_kernels.sh || echo "[FDU] HIP kernel syntax-only compile — OK (no DCU device)"

# --- 拷贝启动脚本与配置 ---
COPY launch.sh .
>>>>>>> 47eb201a21f0eb422c50c45ccf05692b555313c7
COPY config.yaml .
COPY launch.sh .

RUN chmod +x scripts/*.sh launch.sh \
    && bash scripts/compile_kernels.sh

# 若基础镜像未预装 vllm wheel，在构建阶段编译（需 build-arg ENABLE_VLLM_BUILD=1）
ARG ENABLE_VLLM_BUILD=0
RUN if [ "$ENABLE_VLLM_BUILD" = "1" ]; then bash scripts/compile_vllm.sh; fi

<<<<<<< HEAD
=======
# --- 拷贝环境变量说明文档（提交必需）---
COPY docs/env_vars.md ./docs/env_vars.md

# --- 入口 ---
>>>>>>> 47eb201a21f0eb422c50c45ccf05692b555313c7
EXPOSE 8000
ENV FDU_PHASE=1
ENV GPU_MEMORY_UTILIZATION=0.94
ENV FDU_ENABLE_KV_QUANT=0
ENV FDU_ENABLE_PREFIX_CACHE=1
ENV ENABLE_PREFIX_CACHING=1
ENV DO_WARMUP=1
ENV WARMUP_TIER=all
CMD ["bash", "launch.sh"]
