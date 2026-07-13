from __future__ import annotations

from typing import AbstractSet

from .rocm_skinny_shapes import VALIDATED_GFX936_SHAPES, SkinnyShape


SUPPORTED_BATCH_SIZES = frozenset({1, 2, 4})
SUPPORTED_DTYPES = frozenset({"bfloat16", "float16"})


def is_gfx936_skinny_eligible(
    *,
    n: int,
    m: int,
    k: int,
    dtype_name: str,
    bias_present: bool,
    weight_contiguous: bool,
    activation_reshapeable: bool,
    validated_shapes: AbstractSet[SkinnyShape] = VALIDATED_GFX936_SHAPES,
) -> bool:
    shape = (n, m, k, dtype_name, bias_present)
    return (
        n in SUPPORTED_BATCH_SIZES
        and dtype_name in SUPPORTED_DTYPES
        and not bias_present
        and weight_contiguous
        and activation_reshapeable
        and k % 8 == 0
        and shape in validated_shapes
    )
