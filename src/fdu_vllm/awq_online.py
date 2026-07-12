"""
v1.0.0: Online AWQ INT4 quantization — custom weight_loader approach.

Strategy (simplest possible — no process_weights_after_loading dependency):
  1. quant_force.py forces quantization="awq" + creates quant_config.json
  2. This module patches AWQLinearMethod.create_weights to create a "weight"
     param with a CUSTOM weight_loader that quantizes bf16→AWQ INT4 inline.
  3. When vLLM loads the bf16 weight from safetensors, the custom loader:
     a) Copies the bf16 data
     b) Quantizes it to AWQ INT4 (per-group-128 asymmetric)
     c) Registers qweight/qzeros/scales as nn.Parameters
     d) Removes the temporary "weight" param
  4. process_weights_after_loading is no-op'd (already done)
  5. AWQ Triton kernels handle fused dequant+matmul on forward

Why this approach:
  - No dependency on when process_weights_after_loading fires
  - Quantization happens inline during weight loading (guaranteed to execute)
  - Only affects AWQ-configured layers (no embedding/lm_head interference)
  - Handles TP correctly (each shard's weight_loader called independently)
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("fdu_vllm.awq_online")

_PATCHED = False
_GROUP_SIZE = 128


def _quantize_and_register(layer: torch.nn.Module, weight_bf16: torch.Tensor):
    """Quantize loaded bf16 weight → AWQ INT4, register as layer params.

    Called from the custom weight_loader. At this point, weight_bf16 is
    on the correct device, in the correct dtype, with TP sharding applied.

    Args:
        layer: The linear layer module (e.g., q_proj)
        weight_bf16: [N_out, K_in] bf16 weight (TP-sharded if applicable)
    """
    N_out, K_in = weight_bf16.shape
    group_size = _GROUP_SIZE
    num_groups = K_in // group_size

    # --- Quantize: per-group asymmetric INT4 ---
    w = weight_bf16.t().contiguous()  # → [K_in, N_out]
    w_groups = w.view(num_groups, group_size, N_out)

    w_min = w_groups.amin(dim=1, keepdim=True).to(torch.float32)  # [G, 1, N]
    w_max = w_groups.amax(dim=1, keepdim=True).to(torch.float32)  # [G, 1, N]

    rng = w_max - w_min
    rng = torch.where(rng < 1e-9, torch.ones_like(rng), rng)

    scales_fp32 = rng / 15.0
    zp = torch.clamp(
        torch.round(-w_min / scales_fp32), 0, 15
    ).to(torch.int32)  # [G, 1, N]

    # Quantize
    q_float = w_groups.to(torch.float32) / scales_fp32 + zp
    q = torch.clamp(torch.round(q_float), 0, 15).to(torch.uint8)  # [G, gs, N]

    scales = scales_fp32.squeeze(1).to(weight_bf16.dtype)  # [G, N]
    zp = zp.squeeze(1)  # [G, N]

    # --- Pack into AWQ int32 format ---
    q = q.view(K_in, N_out)  # [K, N]
    q = q.view(K_in, N_out // 8, 8)  # [K, N/8, 8]

    # AWQ reverse order: [0,4,1,5,2,6,3,7]
    awq_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=q.device)
    q = q[:, :, awq_order]

    shifts = (torch.arange(8, device=q.device) * 4).to(torch.int32)
    qweight = (q.to(torch.int32) << shifts).sum(dim=-1)  # [K, N/8]

    # Pack zero points
    zp_data = zp.view(num_groups, N_out // 8, 8)  # [G, N/8, 8]
    zp_data = zp_data[:, :, awq_order]
    qzeros = (zp_data.to(torch.int32) << shifts).sum(dim=-1)  # [G, N/8]

    # --- Remove temporary weight, register AWQ parameters ---
    del layer.weight

    layer.register_parameter(
        "qweight", torch.nn.Parameter(qweight, requires_grad=False)
    )
    layer.register_parameter(
        "qzeros", torch.nn.Parameter(qzeros, requires_grad=False)
    )
    layer.register_parameter(
        "scales", torch.nn.Parameter(scales, requires_grad=False)
    )


def _activate_patches():
    """Apply all AWQ online quantization monkey-patches."""
    from vllm.model_executor.layers.quantization.awq import (
        AWQConfig,
        AWQLinearMethod,
    )

    # --- Patch 1: AWQ create_weights → create "weight" with custom loader ---
    _orig_create = AWQLinearMethod.create_weights

    def _fdu_create_weights(self, layer, input_size_per_partition,
                            output_partition_sizes, input_size, output_size,
                            params_dtype, **extra):
        """Create a temporary 'weight' param that triggers quantization on load."""
        output_size_per_partition = sum(output_partition_sizes)

        weight_param = torch.nn.Parameter(
            torch.empty(output_size_per_partition, input_size_per_partition,
                        dtype=params_dtype),
            requires_grad=False,
        )

        # Custom weight_loader: quantizes inline when vLLM loads the weight
        def _fdu_weight_loader(param, loaded_weight):
            param.data.copy_(loaded_weight)       # store bf16
            _quantize_and_register(layer, param.data)  # quantize + register AWQ params
            # layer.weight is now deleted, qweight/qzeros/scales are registered

        weight_param.weight_loader = _fdu_weight_loader
        layer.register_parameter("weight", weight_param)

    AWQLinearMethod.create_weights = _fdu_create_weights

    # --- Patch 2: process_weights_after_loading → no-op ---
    # Quantization already happened in the weight_loader.
    def _fdu_process_weights(self, layer):
        pass

    AWQLinearMethod.process_weights_after_loading = _fdu_process_weights

    # --- Patch 3: disable maybe_update_config ---
    # Stock vLLM scans safetensors to auto-detect unquantized layers. For a bf16
    # model, ALL weights are bf16, so it marks EVERYTHING as not-to-convert,
    # meaning get_quant_method returns UnquantizedLinearMethod for all layers.
    def _noop_maybe_update(self, model_name, revision=None):
        pass

    AWQConfig.maybe_update_config = _noop_maybe_update

    return True


def activate_awq_online() -> bool:
    global _PATCHED
    if _PATCHED:
        return True

    try:
        _activate_patches()
        _PATCHED = True
        logger.info(
            "FDU awq_online v1.0.0: AWQ online INT4 active "
            "(custom weight_loader → bf16→AWQ INT4 at load time)"
        )
        return True
    except Exception as e:
        logger.error("FDU awq_online: activation failed: %s", e)
        return False


def is_patched() -> bool:
    return _PATCHED
