ARG BASE_IMAGE=competition/vllm-0.18.1-base:v1.0
FROM ${BASE_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

RUN python -m pip install --no-cache-dir -r requirements/build.txt
RUN PYTORCH_ROCM_ARCH=gfx936 python setup.py bdist_wheel
RUN python -m pip install --no-deps --force-reinstall dist/vllm-*.whl

RUN chmod +x launch.sh scripts/*.sh scripts/preflight_rocm.py

ENV HIP_VISIBLE_DEVICES=0
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
ENV SAFETENSORS_FAST_GPU=1
ENV VLLM_ROCM_USE_AITER=0
ENV VLLM_ROCM_USE_SKINNY_GEMM=1
ENV FDU_FORCE_STOCK_GEMM=0
ENV FDU_ENABLE=0
ENV FDU_GFX936_QUANT_MODE=w8
ENV ENABLE_PREFIX_CACHING=1

EXPOSE 8000
CMD ["bash", "launch.sh"]
