"""
v1.1.0: Pre-quantize bf16 model → AWQ INT4 at startup.

Strategy: instead of monkey-patching weight loaders (fragile, complex),
quantize the entire model to a temporary directory at container startup,
then point vLLM to the quantized directory. vLLM's native AWQ pipeline
handles everything — no weight loading hacks needed.

Flow:
  1. Read bf16 safetensors
  2. For each linear weight: quantize to AWQ INT4 + scales + zeros
  3. Save as new safetensors + quant_config.json in /tmp/awq_model/
  4. vLLM starts with --model /tmp/awq_model/
  5. vLLM loads AWQ format natively → AWQ Triton kernels → speed

Competition compliance: quantization is done online at startup, the
/tmp/ directory is ephemeral (non-persistent).
"""

from __future__ import annotations

import json
import logging
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger("fdu_vllm.pre_quantize")

GROUP_SIZE = 128
WEIGHT_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def _quant_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize bf16 weight [N_out, K_in] → AWQ INT4 packed format.

    Returns:
        qweight: [K, N//8] int32 (AWQ reverse-order packed)
        scales:  [K//128, N] float16
        qzeros:  [K//128, N//8] int32
    """
    N_out, K_in = weight.shape

    # Pad K_in to multiple of group_size
    num_groups = (K_in + GROUP_SIZE - 1) // GROUP_SIZE
    K_pad = num_groups * GROUP_SIZE
    N_pad = ((N_out + 7) // 8) * 8

    w = torch.zeros(K_pad, N_pad, dtype=weight.dtype, device=weight.device)
    w[:K_in, :N_out] = weight.t()

    # Per-group asymmetric quantization
    w_groups = w.view(num_groups, GROUP_SIZE, N_pad)
    w32 = w_groups.to(torch.float32)

    w_min = w32.amin(dim=1, keepdim=True)
    w_max = w32.amax(dim=1, keepdim=True)
    rng = w_max - w_min
    rng = torch.where(rng < 1e-9, torch.ones_like(rng), rng)

    scales_fp32 = rng.squeeze(1) / 15.0  # [G, N]
    zp_fp32 = torch.clamp(torch.round(-w_min.squeeze(1) / scales_fp32), 0, 15)

    q = torch.clamp(torch.round(w32.squeeze(1) / scales_fp32.unsqueeze(1) + zp_fp32.unsqueeze(1)), 0, 15)
    q = q.to(torch.uint8).view(K_pad, N_pad)

    scales = scales_fp32.to(torch.float16)
    zp = zp_fp32.to(torch.int32)

    # AWQ reverse-order packing: [0,4,1,5,2,6,3,7]
    q = q.view(K_pad, N_pad // 8, 8)
    order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=q.device)
    q = q[:, :, order]
    shifts = (torch.arange(8, device=q.device) * 4).to(torch.int32)
    qweight = (q.to(torch.int32) << shifts).sum(dim=-1)

    # Pack zero points
    zp_packed = zp.view(num_groups, N_pad // 8, 8)
    zp_packed = zp_packed[:, :, order]
    qzeros = (zp_packed.to(torch.int32) << shifts).sum(dim=-1)

    # Trim padding
    qweight = qweight[:K_in, :((N_out + 7) // 8)]
    scales = scales[:, :N_out]
    qzeros = qzeros[:, :((N_out + 7) // 8)]

    return qweight.cpu(), scales.cpu(), qzeros.cpu()


def pre_quantize_model(model_dir: str, output_dir: str) -> None:
    """Quantize a bf16 HuggingFace model to AWQ INT4 format.

    Reads all safetensors from model_dir, quantizes each linear weight,
    writes new safetensors + configs to output_dir.

    Args:
        model_dir: Path to bf16 HuggingFace model directory
        output_dir: Path to write AWQ-quantized model
    """
    model_path = Path(model_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Copy non-weight files (config, tokenizer, etc.)
    for f in model_path.iterdir():
        if f.name.endswith(".json") or f.name.endswith(".txt"):
            import shutil
            shutil.copy2(f, out_path / f.name)
        elif f.name == "tokenizer.model" or f.name.endswith(".py"):
            import shutil
            shutil.copy2(f, out_path / f.name)

    # Create quant_config.json
    quant_cfg = {
        "quant_method": "awq",
        "bits": 4,
        "group_size": GROUP_SIZE,
        "zero_point": True,
    }
    (out_path / "quant_config.json").write_text(json.dumps(quant_cfg))
    logger.info("Created quant_config.json: %s", quant_cfg)

    # Find all safetensors files
    st_files = sorted(f for f in model_path.glob("*.safetensors")
                      if not f.name.startswith("."))

    total_size = 0
    quant_size = 0
    total_files = len(st_files)

    for idx, st_file in enumerate(st_files):
        logger.info("[%d/%d] Processing %s", idx + 1, total_files, st_file.name)
        new_tensors: dict[str, torch.Tensor] = {}

        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                total_size += tensor.numel() * tensor.element_size()

                # Check if this is a linear weight to quantize
                should_quantize = (
                    tensor.dtype in (torch.bfloat16, torch.float16)
                    and tensor.ndim == 2
                    and any(key.endswith(f".{s}.weight") for s in WEIGHT_SUFFIXES)
                    and tensor.shape[1] % GROUP_SIZE == 0
                    and tensor.shape[0] % 8 == 0
                )

                if should_quantize:
                    qweight, scales, qzeros = _quant_weight(tensor)
                    prefix = key.rsplit(".weight", 1)[0]
                    new_tensors[f"{prefix}.qweight"] = qweight
                    new_tensors[f"{prefix}.scales"] = scales
                    new_tensors[f"{prefix}.qzeros"] = qzeros
                    quant_size += (
                        qweight.numel() * 4
                        + scales.numel() * 2
                        + qzeros.numel() * 4
                    )
                else:
                    new_tensors[key] = tensor

        out_file = out_path / st_file.name
        save_file(new_tensors, str(out_file))
        del new_tensors

    logger.info(
        "Quantization complete: %.1fGB bf16 → %.1fGB INT4 (%.1fx reduction)",
        total_size / 1e9, quant_size / 1e9, total_size / quant_size
    )

    logger.info("Model ready at: %s", output_dir)
    logger.info("Start vLLM with: --model %s --quantization awq --dtype float16", output_dir)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FDU pre-quantize bf16→AWQ INT4")
    parser.add_argument("model_dir", help="Path to bf16 HuggingFace model")
    parser.add_argument("output_dir", nargs="?", default="/tmp/awq_model",
                       help="Output directory (default: /tmp/awq_model)")
    args = parser.parse_args()
    pre_quantize_model(args.model_dir, args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
