"""Non-persistent KV FP8 hooks (competition-compliant)."""

from __future__ import annotations

import logging
from typing import Tuple

import torch

logger = logging.getLogger("fdu_vllm.kv_fp8")


class KVQuantHooks:
    FP8_MAX = 448.0
    FP8_MIN = -448.0

    def __init__(self, enabled: bool = True, dtype: str = "fp8"):
        self.enabled = enabled
        self.dtype = dtype

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled or self.dtype != "fp8":
            return tensor, torch.tensor(1.0, device=tensor.device, dtype=torch.float32)
        amax = tensor.abs().max()
        scale = torch.clamp(amax / self.FP8_MAX, min=1e-12)
        q = torch.clamp(tensor / scale, self.FP8_MIN, self.FP8_MAX)
        if hasattr(torch, "float8_e4m3fn"):
            q = q.to(torch.float8_e4m3fn)
        return q, scale

    def dequantize(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if hasattr(torch, "float8_e4m3fn") and quantized.dtype == torch.float8_e4m3fn:
            return quantized.to(torch.bfloat16) * scale
        return quantized

    def quantize_pair(self, tensor: torch.Tensor):
        return self.quantize(tensor)

    def install(self) -> None:
        logger.info("KV FP8 hooks installed (online, non-persistent)")
