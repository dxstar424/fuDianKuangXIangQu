"""
v1.0.0: Force AWQ INT4 quantization + create quant_config.json.

Monkey-patches ModelConfig.__init__ to force quantization="awq" and
creates quant_config.json in the model directory so vLLM's AWQ config
loader can find it.

Combined with awq_online.py (which intercepts weight loading to do
bf16→INT4 on-the-fly), this enables online AWQ INT4 quantization
using Triton fused dequant+matmul kernels on ROCm gfx942.

Previous attempts:
  v0.8.1 bitsandbytes INT4: matmul_4bit no ROCm HIP kernel → 0 benefit
  v0.9.0 FP8 W8A8: torch._scaled_mm crashes (needs MI300+)
  v0.9.2 FP8 fallback: dequant+matmul = 81GB HBM > 54GB bf16 → slower
"""

from __future__ import annotations

import json
import logging
import os
from functools import wraps
from pathlib import Path

logger = logging.getLogger("fdu_vllm.quant_force")

_PATCHED = False
_GROUP_SIZE = 128


def _create_quant_config(model_dir: str) -> None:
    """Create quant_config.json so vLLM's AWQ config loader finds it."""
    if not model_dir or not os.path.isdir(model_dir):
        return
    cfg_path = Path(model_dir) / "quant_config.json"
    if cfg_path.exists():
        return  # already created
    cfg = {
        "quant_method": "awq",
        "bits": 4,
        "group_size": _GROUP_SIZE,
        "zero_point": True,
    }
    cfg_path.write_text(json.dumps(cfg))
    logger.info("FDU quant_force: created %s → %s", cfg_path, cfg)


def _patch_model_config() -> bool:
    """Monkey-patch ModelConfig.__init__ to force quantization='awq'."""
    global _PATCHED
    if _PATCHED:
        return True

    try:
        from vllm.config.model import ModelConfig

        _original_init = ModelConfig.__init__

        @wraps(_original_init)
        def _patched_init(self, **kwargs):
            kwargs["quantization"] = "awq"
            _original_init(self, **kwargs)
            # After init, model path is resolved — create quant_config.json
            try:
                _create_quant_config(self.model)
            except Exception:
                pass

        ModelConfig.__init__ = _patched_init
        _PATCHED = True
        logger.info(
            "FDU quant_force: ModelConfig.__init__ patched — "
            "quantization forced to 'awq' (online INT4)"
        )
        return True

    except ImportError:
        logger.warning(
            "FDU quant_force: vLLM not importable yet"
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
