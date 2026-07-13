from __future__ import annotations


SkinnyShape = tuple[int, int, int, str, bool]

# This remains empty until scripts/bench_gfx936_skinny.py admits measured rows.
VALIDATED_GFX936_SHAPES: frozenset[SkinnyShape] = frozenset()
