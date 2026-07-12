"""
v1.0.0: Online AWQ INT4 quantization via weight iterator monkey-patch.

Intercepts DefaultModelLoader.get_all_weights() to on-the-fly quantize
bf16 weights → AWQ INT4 format (packed int32 + scales). The existing
AWQ Triton kernels (awq_gemm_triton, awq_dequantize_triton) handle the
fused dequant+matmul forward pass.

Strategy:
  1. quant_force.py forces quantization="awq" + creates quant_config.json
  2. This module wraps the weight iterator to intercept bf16 .weight entries
  3. Each bf16 weight is quantized per-group-of-128 to 4-bit AWQ format
  4. Yields qweight/qzeros/scales entries that vLLM's AWQ pipeline expects
  5. AWQ Triton kernels (pure Triton, GPU-native) do fused dequant+matmul

Why this works on gfx942:
  - AWQ on ROCm uses Triton (VLLM_USE_TRITON_AWQ=1 forced by platform)
  - Triton compiles to HIP on ROCm — no CUDA dependency
  - Fused dequant+matmul in a single kernel → no intermediate HBM write
  - 4x weight IO reduction (13.5GB vs 54GB) → 2-3x decode throughput
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger("fdu_vllm.awq_online")

_PATCHED = False
_GROUP_SIZE = 128


def _create_quant_config(model_dir: str) -> Path:
    """Create quant_config.json in model directory for AWQ config loading."""
    cfg_path = Path(model_dir) / "quant_config.json"
    cfg = {
        "quant_method": "awq",
        "bits": 4,
        "group_size": _GROUP_SIZE,
        "zero_point": True,
    }
    cfg_path.write_text(json.dumps(cfg))
    logger.info("FDU awq: created %s → %s", cfg_path, cfg)
    return cfg_path


def _quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a single bf16/fp16 weight tensor to AWQ INT4 format.

    Args:
        weight: [N_out, K_in] linear weight (vLLM/PyTorch convention)

    Returns:
        qweight:  [K_in, N_out_padded//8] int32 (AWQ reverse-order packed)
        scales:   [K_in_padded//128, N_out_padded] float16
        qzeros:   [K_in_padded//128, N_out_padded//8] int32
    """
    N_out, K_in = weight.shape
    group_size = _GROUP_SIZE

    # --- Step 1: transpose to AWQ layout [K_in, N_out] ---
    w = weight.t().contiguous()  # [K_in, N_out]

    # --- Step 2: pad K_in to multiple of group_size ---
    num_groups = (K_in + group_size - 1) // group_size
    K_pad = num_groups * group_size
    N_pad = ((N_out + 7) // 8) * 8  # pad N to multiple of 8

    if K_pad > K_in or N_pad > N_out:
        w_padded = torch.zeros(K_pad, N_pad, dtype=w.dtype, device=w.device)
        w_padded[:K_in, :N_out] = w
        w = w_padded
    # --- Step 3: group-wise asymmetric quantization ---
    # Reshape to [G, gs, N_pad]
    w_groups = w.view(num_groups, group_size, N_pad)

    w_min = w_groups.amin(dim=1, keepdim=True).to(torch.float32)  # [G, 1, N]
    w_max = w_groups.amax(dim=1, keepdim=True).to(torch.float32)  # [G, 1, N]

    # Avoid division by zero
    range_val = w_max - w_min
    range_val = torch.where(range_val < 1e-9, torch.ones_like(range_val), range_val)

    scales_fp32 = range_val / 15.0
    scales = scales_fp32.squeeze(1).to(torch.float16)  # [G, N_pad]

    zero_points = torch.round(-w_min / scales_fp32)
    zero_points = torch.clamp(zero_points.squeeze(1), 0, 15).to(torch.int32)  # [G, N_pad]

    # Quantize: q = round(w / scale + zero_point), clamp [0, 15]
    q_float = (
        w_groups.to(torch.float32) / scales_fp32 + zero_points.unsqueeze(1)
    )
    q = torch.clamp(torch.round(q_float), 0, 15).to(torch.uint8)  # [G, gs, N_pad]

    # --- Step 4: reshape to [K_pad, N_pad] and pack into int32 ---
    q = q.view(K_pad, N_pad)  # [K_pad, N_pad]
    q = q.view(K_pad, N_pad // 8, 8)  # [K_pad, N_pad//8, 8]

    # AWQ reverse order: [0, 4, 1, 5, 2, 6, 3, 7]
    # Corresponds to shifts: [0, 16, 4, 20, 8, 24, 12, 28]
    awq_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=q.device)
    q = q[:, :, awq_order]  # reorder

    # Pack: each 4-bit value shifted by (pos * 4) bits, summed into int32
    shifts = (torch.arange(8, device=q.device) * 4).to(torch.int32)
    qweight = (q.to(torch.int32) << shifts).sum(dim=-1)  # [K_pad, N_pad//8]

    # --- Step 5: pack qzeros ---
    # Zero points: [G, N_pad] → pack into [G, N_pad//8]
    zp = zero_points.view(num_groups, N_pad // 8, 8)  # [G, N_pad//8, 8]
    zp = zp[:, :, awq_order]  # reorder
    qzeros = (zp.to(torch.int32) << shifts).sum(dim=-1)  # [G, N_pad//8]

    return qweight, scales, qzeros


def _wrap_weights_iter(
    weights_iter,
    quant_prefixes: list[str] | None = None,
):
    """Wrap a weight iterator to on-the-fly quantize bf16 weights to AWQ INT4.

    For each (name, tensor) from weights_iter:
      - If name matches '*.weight' and is bf16/fp16 → quantize, yield AWQ entries
      - If name is already AWQ format (*.qweight, *.scales, *.qzeros) → skip
      - Otherwise → yield unchanged

    This lets us feed bf16 safetensors data into vLLM's AWQ pipeline seamlessly.
    """
    for name, tensor in weights_iter:
        # Skip already-quantized entries (shouldn't appear for bf16 model)
        if name.endswith(".qweight") or name.endswith(".qzeros") or name.endswith(".scales"):
            continue

        # Intercept bf16/fp16 linear weights and quantize them
        if name.endswith(".weight") and tensor.dtype in (torch.bfloat16, torch.float16):
            prefix = name.rsplit(".weight", 1)[0]

            try:
                qweight, scales, qzeros = _quantize_weight(tensor)
                yield (f"{prefix}.qweight", qweight)
                yield (f"{prefix}.scales", scales)
                yield (f"{prefix}.qzeros", qzeros)
            except Exception as e:
                logger.error("FDU awq: failed to quantize %s (shape=%s): %s", name, tensor.shape, e)
                raise

            del tensor  # free bf16 memory
        else:
            yield (name, tensor)


def _patch_default_loader():
    """Monkey-patch DefaultModelLoader.get_all_weights to quantize on-the-fly."""
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    _orig_get_all = DefaultModelLoader.get_all_weights

    def _patched_get_all(self, model_config, model):
        weights_iter = _orig_get_all(self, model_config, model)
        return _wrap_weights_iter(weights_iter)

    DefaultModelLoader.get_all_weights = _patched_get_all
    return True


def activate_awq_online() -> bool:
    """Activate online AWQ INT4 quantization.

    Called from hooks.py after quant_force monkey-patch and before model loading.
    """
    global _PATCHED
    if _PATCHED:
        return True

    try:
        _patch_default_loader()
        _PATCHED = True
        logger.info(
            "FDU awq_online v1.0.0: AWQ INT4 online quantization active "
            "(bf16→INT4 at load time, Triton fused dequant+matmul)"
        )
        return True
    except Exception as e:
        logger.error("FDU awq_online: activation failed: %s", e)
        return False


def is_patched() -> bool:
    return _PATCHED
