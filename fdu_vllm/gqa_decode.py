"""GQA-optimized attention without full KV repeat (memory bandwidth saving)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gqa_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """
    GQA attention: [T, Hq, D] x [T, Hkv, D] without materializing repeated KV.

    Uses einsum grouping when Hq % Hkv == 0.
    """
    if num_q_heads == num_kv_heads:
        return F.scaled_dot_product_attention(
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
            scale=scale,
        ).transpose(0, 1)

    if num_q_heads % num_kv_heads != 0:
        key = key.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
        value = value.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
        return F.scaled_dot_product_attention(
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
            scale=scale,
        ).transpose(0, 1)

    g = num_q_heads // num_kv_heads
    # query: [T, Hq, D] -> [T, Hkv, g, D]
    t, _, d = query.shape
    q = query.view(t, num_kv_heads, g, d)
    k = key.unsqueeze(2)   # [T, Hkv, 1, D]
    v = value.unsqueeze(2)

    # scores: [T, Hkv, g, T] — decode-friendly small T
    scores = torch.einsum("thgd,thd->thgt", q, k.squeeze(2)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("thgt,thd->thgd", attn, v.squeeze(2))
    return out.view(t, num_q_heads, d)
