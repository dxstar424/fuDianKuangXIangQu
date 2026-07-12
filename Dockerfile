# ============================================================
# FDU SCCSCC26 — v1.0.0: AWQ INT4 online quantization
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

# v1.0.0: AWQ INT4 online quantization
#   quant_force.py: forces quantization="awq" + creates quant_config.json
#   awq_online.py: intercepts weight loading, bf16→AWQ INT4 on-the-fly
#   AWQ Triton kernels: fused dequant+matmul (VLLM_USE_TRITON_AWQ=1)
RUN python -c "
import vllm
# Patch __init__.py — add FDU hook (harmless, quant_force is no-op)
init_py = vllm.__file__
marker = '# FDU_CSCC_PLUGIN'
with open(init_py, 'r') as f:
    content = f.read()
if marker not in content:
    with open(init_py, 'a') as f:
        f.write('\n')
        f.write(marker + '\n')
        f.write('try:\n')
        f.write('    import fdu_vllm  # noqa: F401\n')
        f.write('    fdu_vllm.activate()\n')
        f.write('except Exception as _fdu_err:\n')
        f.write('    import logging\n')
        f.write('    logging.getLogger(\"fdu_vllm\").warning(\"FDU plugin not activated: %s\", _fdu_err)\n')
    print('[FDU] vLLM __init__.py patched: FDU hook appended')
else:
    print('[FDU] vLLM __init__.py: FDU hook already present')
"

ARG ENABLE_VLLM_BUILD=0
RUN if [ "$ENABLE_VLLM_BUILD" = "1" ]; then bash scripts/compile_vllm.sh; fi

EXPOSE 8000
# v0.9.3: bf16 stock + AITER optimizations (no weight quantization)
# FLASH_ATTN + skinny_gemm + rmsnorm via AITER
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
