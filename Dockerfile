# ============================================================
# FDU SCCSCC26 — v0.6.0 FINAL: HIP FlashAttention
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
RUN pip install --no-cache-dir runai-model-streamer || echo "[FDU] runai_streamer skipped"

COPY src/ ./src/
COPY patches/ ./patches/
COPY scripts/ ./scripts/
COPY config.yaml .
COPY launch.sh .
COPY docs/env_vars.md ./docs/env_vars.md

RUN chmod +x scripts/*.sh launch.sh

ARG ENABLE_VLLM_BUILD=0
RUN if [ "$ENABLE_VLLM_BUILD" = "1" ]; then bash scripts/compile_vllm.sh; fi

EXPOSE 8000
# v0.6.0 FINAL: AITER=1, NO unified → FLASH_ATTN (HIP CK kernel)
ENV VLLM_ROCM_USE_AITER=1
ENV VLLM_ROCM_USE_SKINNY_GEMM=1
ENV VLLM_ROCM_USE_AITER_RMSNORM=1
ENV TORCH_BLAS_PREFER_HIPBLASLT=0
ENV HSA_OVERRIDE_GFX_VERSION=9.4.2
ENV HIP_VISIBLE_DEVICES=0
ENV GPU_MEMORY_UTILIZATION=0.95
ENV LOAD_FORMAT=runai_streamer
ENV SAFETENSORS_FAST_GPU=1
ENV ROCBLAS_LAYER=4
ENV MIOPEN_FIND_MODE=1
CMD ["bash", "launch.sh"]
