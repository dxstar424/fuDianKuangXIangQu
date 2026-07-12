# ============================================================
# FDU SCCSCC26 — v0.8.0: bitsandbytes INT4 + HIP FlashAttention
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
# v0.8.0: bitsandbytes for online INT4 quantization (bf16 → INT4 at model load)
RUN pip install --no-cache-dir bitsandbytes>=0.49.2 || echo "[FDU] bitsandbytes skipped"

COPY src/ ./src/
COPY patches/ ./patches/
COPY scripts/ ./scripts/
COPY config.yaml .
COPY launch.sh .
COPY docs/env_vars.md ./docs/env_vars.md

RUN chmod +x scripts/*.sh launch.sh

# v0.8.0: Overwrite vLLM source with patched defaults
#   model.py: quantization default None → "bitsandbytes"
#   bitsandbytes.py: compute_dtype float32 → bfloat16, quant_type fp4 → nf4
RUN python -c "
import vllm.config.model
import shutil, os
# Patch model.py
dst_model = vllm.config.model.__file__
shutil.copy('/workspace/patches/vllm_cscc_modified/model.py', dst_model)
pycache = os.path.join(os.path.dirname(dst_model), '__pycache__')
if os.path.exists(pycache):
    for f in os.listdir(pycache):
        if 'model' in f:
            os.remove(os.path.join(pycache, f))
print('[FDU] vLLM model.py patched: quantization default → bitsandbytes')
# Patch bitsandbytes.py (best-effort: requires bnb installed)
try:
    import vllm.model_executor.layers.quantization.bitsandbytes as bnb_mod
    shutil.copy('/workspace/patches/vllm_cscc_modified/bitsandbytes.py', bnb_mod.__file__)
    pycache = os.path.join(os.path.dirname(bnb_mod.__file__), '__pycache__')
    if os.path.exists(pycache):
        for f in os.listdir(pycache):
            if 'bitsandbytes' in f:
                os.remove(os.path.join(pycache, f))
    print('[FDU] bitsandbytes.py patched: compute_dtype → bfloat16, quant_type → nf4')
except Exception as e:
    print(f'[FDU] bitsandbytes.py patch skipped (bnb may not be installed): {e}')
"

ARG ENABLE_VLLM_BUILD=0
RUN if [ "$ENABLE_VLLM_BUILD" = "1" ]; then bash scripts/compile_vllm.sh; fi

EXPOSE 8000
# v0.8.0: bitsandbytes INT4 (via patched vLLM source, no CLI flag needed)
#         + AITER HIP FlashAttention + skinny_gemm + RMSNorm
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
