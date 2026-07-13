from __future__ import annotations


SkinnyShape = tuple[int, int, int, str, bool]

# SCNet gfx936 BF16 LLMM1 results, measured 2026-07-14. The mlp_down
# (1, 5120, 17408) shape is intentionally absent because it failed numerics.
VALIDATED_GFX936_SHAPES: frozenset[SkinnyShape] = frozenset(
    {
        (1, 16384, 5120, "bfloat16", False),
        (1, 96, 5120, "bfloat16", False),
        (1, 14336, 5120, "bfloat16", False),
        (1, 5120, 6144, "bfloat16", False),
        (1, 34816, 5120, "bfloat16", False),
    }
)
