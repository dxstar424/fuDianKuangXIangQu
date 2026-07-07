"""
KV Cache Online Quantization (non-persistent, per-inference).

Allowed by competition rules: "推理过程中的非持久化、算子级低精度计算优化"
Not allowed: "持久化量化...或生成可复用量化权重缓存"

Strategy:
- Quantize KV cache to FP8 at write time (store phase)
- Dequantize back to bf16 at read time (load phase)
- No persistent quantized weights; quantization happens per-token, per-request
"""

import torch
from typing import Tuple


class KVCacheQuantizer:
    """
    Online KV Cache quantization / dequantization.

    Uses dynamic per-tensor scaling (FP8 E4M3 format).
    Quantization and dequantization happen inline during inference;
    no persistent quantized state is stored.
    """

    # FP8 E4M3: exponent=4, mantissa=3, range ~[-448, 448]
    FP8_MAX = 448.0
    FP8_MIN = -448.0

    def __init__(self, dtype: str = "fp8"):
        """
        Args:
            dtype: quantization type - "fp8" (E4M3) or "int8"
        """
        self.dtype = dtype
        self.enabled = True

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a bf16/fp16 tensor to FP8.

        Args:
            tensor: input in bf16 [..., head_dim]

        Returns:
            (quantized_tensor, scale)
                quantized_tensor: FP8 values as uint8
                scale: per-tensor scale factor in bf16
        """
        if not self.enabled or self.dtype != "fp8":
            return tensor, torch.tensor(1.0, device=tensor.device)

        # Per-tensor dynamic scaling
        amax = tensor.abs().max()
        scale = amax / self.FP8_MAX
        scale = torch.clamp(scale, min=1e-12)

        # Quantize: x_fp8 = round(clip(x_bf16 / scale, -FP8_MAX, FP8_MAX))
        quantized = tensor / scale
        quantized = torch.clamp(quantized, self.FP8_MIN, self.FP8_MAX)
        quantized = quantized.to(torch.float8_e4m3fn)

        return quantized, scale

    def dequantize(
        self, quantized: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Dequantize FP8 KV cache back to bf16 for attention computation.

        Args:
            quantized: FP8 tensor
            scale: per-tensor scale factor from quantize()

        Returns:
            Dequantized bf16 tensor
        """
        if quantized.dtype != torch.float8_e4m3fn:
            return quantized

        return quantized.to(torch.bfloat16) * scale

    def estimate_memory_savings(self, num_blocks: int, block_size: int,
                                 num_layers: int, num_kv_heads: int,
                                 head_dim: int) -> float:
        """
        Estimate HBM savings from KV cache quantization.

        Returns:
            Memory savings in GiB
        """
        bytes_bf16 = 2  # bf16 = 2 bytes
        bytes_fp8 = 1    # fp8 = 1 byte

        kv_cache_size_bf16 = (
            num_blocks * block_size * num_layers * 2 * num_kv_heads * head_dim
            * bytes_bf16
        )
        kv_cache_size_fp8 = (
            num_blocks * block_size * num_layers * 2 * num_kv_heads * head_dim
            * bytes_fp8
        )
        return (kv_cache_size_bf16 - kv_cache_size_fp8) / (1024 ** 3)
