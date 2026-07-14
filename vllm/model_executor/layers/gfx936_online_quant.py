from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch


QuantMode = Literal["off", "w8", "hybrid_w4"]
QuantKind = Literal["w8", "w4"]

QUANT_SHAPES: frozenset[tuple[int, int]] = frozenset(
    {
        (16384, 5120),
        (96, 5120),
        (14336, 5120),
        (5120, 6144),
        (34816, 5120),
        (5120, 17408),
    }
)
W4_MLP_SHAPES: frozenset[tuple[int, int]] = frozenset(
    {(34816, 5120), (5120, 17408)}
)
PACK_CHUNK_BYTES = 64 << 20


def parse_quant_mode(value: str | None = None) -> QuantMode:
    normalized = (value if value is not None else os.getenv("FDU_GFX936_QUANT_MODE", "off"))
    normalized = normalized.strip().lower()
    if normalized in {"off", "w8", "hybrid_w4"}:
        return normalized  # type: ignore[return-value]
    return "off"


def is_quant_shape(m: int, k: int) -> bool:
    return (m, k) in QUANT_SHAPES


def candidate_kinds(mode: str, m: int, k: int) -> tuple[QuantKind, ...]:
    parsed = parse_quant_mode(mode)
    if parsed == "off" or not is_quant_shape(m, k):
        return ()
    if parsed == "hybrid_w4" and (m, k) in W4_MLP_SHAPES:
        return ("w4", "w8")
    return ("w8",)


def iter_row_chunks(
    rows: int, columns: int, byte_limit: int = PACK_CHUNK_BYTES
) -> Iterator[tuple[int, int]]:
    if rows <= 0 or columns <= 0 or byte_limit <= 0:
        raise ValueError("rows, columns, and byte_limit must be positive")
    rows_per_chunk = max(1, byte_limit // (columns * 2))
    for start in range(0, rows, rows_per_chunk):
        yield start, min(rows, start + rows_per_chunk)


def _symmetric_quantize(values: Sequence[float], limit: int) -> tuple[list[int], float]:
    maximum = max((abs(float(value)) for value in values), default=0.0)
    scale = maximum / limit if maximum > 0.0 else 1.0
    quantized = [max(-limit, min(limit, round(float(value) / scale))) for value in values]
    return quantized, scale


def quantize_row_w8_reference(values: Sequence[float]) -> tuple[list[int], float]:
    if not values:
        raise ValueError("W8 row must not be empty")
    return _symmetric_quantize(values, 127)


def quantize_group_w4_reference(
    values: Sequence[float], group_size: int = 32
) -> tuple[bytes, list[float]]:
    if group_size <= 0 or len(values) == 0 or len(values) % group_size != 0:
        raise ValueError("W4 row length must be a positive multiple of group_size")
    packed = bytearray()
    scales: list[float] = []
    for group_start in range(0, len(values), group_size):
        quantized, scale = _symmetric_quantize(
            values[group_start : group_start + group_size], 7
        )
        scales.append(scale)
        for index in range(0, group_size, 2):
            low = quantized[index] & 0x0F
            high = quantized[index + 1] & 0x0F
            packed.append(low | (high << 4))
    return bytes(packed), scales


def unpack_group_w4_reference(packed: bytes, value_count: int) -> list[int]:
    if value_count < 0 or value_count % 2 != 0 or len(packed) * 2 != value_count:
        raise ValueError("packed W4 length does not match value_count")
    values: list[int] = []
    for byte in packed:
        for nibble in (byte & 0x0F, byte >> 4):
            values.append(nibble - 16 if nibble >= 8 else nibble)
    return values
