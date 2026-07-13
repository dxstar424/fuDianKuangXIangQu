from __future__ import annotations


ROCM_SKINNY_GEMM_ARCHES = frozenset({"gfx90a", "gfx936", "gfx942", "gfx950"})


def canonical_rocm_arch(gcn_arch: str | None) -> str:
    return (gcn_arch or "").split(":", 1)[0].strip().lower()


def is_gfx936_arch(gcn_arch: str | None) -> bool:
    return canonical_rocm_arch(gcn_arch) == "gfx936"


def supports_rocm_skinny_gemm_arch(gcn_arch: str | None) -> bool:
    return canonical_rocm_arch(gcn_arch) in ROCM_SKINNY_GEMM_ARCHES
