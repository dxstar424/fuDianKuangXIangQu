# ============================================================
# FDU SCCSCC26 — v0.9.0: FP8 online quantization (torch._scaled_mm HIP kernel)
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
# v0.8.1 (legacy fallback): bitsandbytes for INT4 path (not used by v0.9.0 FP8)
RUN pip install --no-cache-dir bitsandbytes>=0.49.2 || echo "[FDU] bnb skipped (FP8 is primary)"

COPY src/ ./src/
COPY patches/ ./patches/
COPY scripts/ ./scripts/
COPY fdu_vllm/ ./fdu_vllm/
COPY config.yaml .
COPY launch.sh .
COPY docs/env_vars.md ./docs/env_vars.md

RUN chmod +x scripts/*.sh launch.sh

# v0.9.0: FP8 online quantization via fdu_vllm monkey-patch
#   quant_force.py → quantization="fp8" (ModelConfig.__init__ patched)
#   Fp8OnlineLinearMethod → bf16 → FP8 at model load
#   torch._scaled_mm → ROCm native HIP kernel (no on-the-fly dequant)
#   PYTHONPATH set in launch.sh ensures fdu_vllm is importable
# v0.9.0: belt-and-suspenders — also patch installed vLLM source default
#   model.py: quantization default None → "fp8"
#   This covers the Dockerfile scenario where fdu_vllm may or may not be importable
RUN python -c "
import vllm.config.model
import shutil, os
dst_model = vllm.config.model.__file__
shutil.copy('/workspace/patches/vllm_cscc_modified/model.py', dst_model)
pycache = os.path.join(os.path.dirname(dst_model), '__pycache__')
if os.path.exists(pycache):
    for f in os.listdir(pycache):
        if 'model' in f:
            os.remove(os.path.join(pycache, f))
print('[FDU] vLLM model.py patched: quantization default → fp8')
"

ARG ENABLE_VLLM_BUILD=0
RUN if [ "$ENABLE_VLLM_BUILD" = "1" ]; then bash scripts/compile_vllm.sh; fi

EXPOSE 8000
# v0.9.0: FP8 online quantization via fdu_vllm monkey-patch (not CLI flag)
# Fp8OnlineLinearMethod: bf16 weights → FP8 at model load time
# torch._scaled_mm: ROCm native HIP kernel for W8A8 matmul
# PYTHONPATH set in launch.sh ensures fdu_vllm is importable
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
