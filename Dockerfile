# ============================================================
# FDU SCCSCC26 - Qwen3.5-27B × vLLM 0.18.1 推理优化
# ============================================================

ARG BASE_IMAGE=competition/vllm-0.18.1-base:v1.0
FROM ${BASE_IMAGE}

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
RUN pip install --no-cache-dir runai-model-streamer || echo "[FDU] runai_streamer skipped (non-fatal)"

COPY src/ ./src/
COPY patches/ ./patches/
COPY scripts/ ./scripts/
COPY config.yaml .
COPY launch.sh .
COPY docs/env_vars.md ./docs/env_vars.md

RUN chmod +x scripts/*.sh launch.sh \
    && bash scripts/compile_kernels.sh || echo "[FDU] compile_kernels skipped/non-fatal"

# 若基础镜像未预装 vllm wheel，在构建阶段编译（需 build-arg ENABLE_VLLM_BUILD=1）
ARG ENABLE_VLLM_BUILD=0
RUN if [ "$ENABLE_VLLM_BUILD" = "1" ]; then bash scripts/compile_vllm.sh; fi

EXPOSE 8000
# v0.5.0: AITER unified attention
ENV VLLM_ROCM_USE_AITER=1
ENV VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
ENV HSA_OVERRIDE_GFX_VERSION=9.4.2
ENV HIP_VISIBLE_DEVICES=0
ENV GPU_MEMORY_UTILIZATION=0.95
ENV LOAD_FORMAT=runai_streamer
CMD ["bash", "launch.sh"]
