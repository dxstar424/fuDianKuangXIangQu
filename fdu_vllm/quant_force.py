"""
v0.8.1: Monkey-patch vLLM ModelConfig to force bitsandbytes INT4 quantization.

This runs at vLLM import time (via vllm/__init__.py → fdu_vllm.activate()),
BEFORE any model config is parsed. The platform CANNOT skip this hook.

Unlike CLI flags (overridden by platform evaluator) or Dockerfile patches
(may not be used by platform), this monkey-patch is guaranteed to execute
as long as vLLM imports our fdu_vllm plugin — which it always does.
"""

from __future__ import annotations

import logging
from functools import wraps

logger = logging.getLogger("fdu_vllm.quant_force")

_PATCHED = False


def _patch_model_config() -> bool:
    """Monkey-patch vllm.config.model.ModelConfig to force quantization='bitsandbytes'.

    The patch intercepts ModelConfig.__init__ to ensure quantization is always
    set to 'bitsandbytes', regardless of CLI args or config files.

    Returns True if patched successfully.
    """
    global _PATCHED
    if _PATCHED:
        return True

    try:
        from vllm.config.model import ModelConfig

        _original_init = ModelConfig.__init__

        @wraps(_original_init)
        def _patched_init(self, **kwargs):
            # Force quantization before original init processes it
            kwargs["quantization"] = "bitsandbytes"
            _original_init(self, **kwargs)

        ModelConfig.__init__ = _patched_init
        _PATCHED = True
        logger.info(
            "FDU quant_force: ModelConfig.__init__ patched — "
            "quantization forced to 'bitsandbytes'"
        )
        return True

    except ImportError:
        logger.warning(
            "FDU quant_force: vLLM not importable yet, will retry at hook time"
        )
        return False
    except Exception as e:
        logger.error("FDU quant_force: patch failed: %s", e)
        return False


def activate_quant_force() -> bool:
    """Public entry point — called from hooks.py."""
    return _patch_model_config()


def is_patched() -> bool:
    return _PATCHED
