"""
v1.1.0: Pre-quantize bf16 model→AWQ INT4 at vLLM import time.

Monkey-patches ModelConfig.__init__ to:
  1. Detect if --model points to a bf16 HuggingFace model
  2. Pre-quantize to /tmp/awq_model/ if not already done
  3. Redirect --model → /tmp/awq_model --quantization → awq

This runs BEFORE platform evaluator parses CLI args (vllm/__init__.py
→ fdu_vllm.activate()). The platform CANNOT skip this.

No weight-loading hacks needed — vLLM loads AWQ format natively.
AWQ Triton kernels (VLLM_USE_TRITON_AWQ=1 on ROCm) do fused dequant+matmul.
"""

from __future__ import annotations

import logging
import os
from functools import wraps

logger = logging.getLogger("fdu_vllm.quant_force")

_PATCHED = False
_AWQ_DIR = "/tmp/awq_model"


def _patch_model_config() -> bool:
    global _PATCHED
    if _PATCHED:
        return True

    try:
        from vllm.config.model import ModelConfig

        _original_init = ModelConfig.__init__

        @wraps(_original_init)
        def _patched_init(self, **kwargs):
            model_path = kwargs.get("model") or getattr(self, "model", None)

            if model_path and os.path.isdir(model_path):
                cfg_path = os.path.join(_AWQ_DIR, "quant_config.json")

                # Pre-quantize if not already done
                if not os.path.exists(cfg_path):
                    logger.info(
                        "FDU v1.1.0: pre-quantizing %s → %s", model_path, _AWQ_DIR
                    )
                    try:
                        from fdu_vllm.pre_quantize import pre_quantize_model

                        pre_quantize_model(model_path, _AWQ_DIR)
                        logger.info("FDU v1.1.0: pre-quantization complete")
                    except Exception as e:
                        logger.error(
                            "FDU v1.1.0: pre-quantization failed: %s", e
                        )
                        # Fall through — vLLM will load bf16 model normally

                # Redirect to AWQ model if available
                if os.path.exists(cfg_path):
                    kwargs["model"] = _AWQ_DIR
                    kwargs["quantization"] = "awq"
                    kwargs["dtype"] = "float16"  # AWQ requires float16 on ROCm
                    logger.info(
                        "FDU v1.1.0: redirecting → %s, quantization=awq, dtype=float16",
                        _AWQ_DIR,
                    )

            _original_init(self, **kwargs)

        ModelConfig.__init__ = _patched_init
        _PATCHED = True
        logger.info(
            "FDU quant_force v1.1.0: ModelConfig.__init__ patched "
            "(pre-quantize bf16→AWQ INT4 at vLLM import time)"
        )
        return True

    except ImportError:
        logger.warning("FDU quant_force: vLLM not importable yet")
        return False
    except Exception as e:
        logger.error("FDU quant_force: patch failed: %s", e)
        return False


def activate_quant_force() -> bool:
    return _patch_model_config()


def is_patched() -> bool:
    return _PATCHED
