#!/usr/bin/env python3
"""Fail-fast validation for the installed native ROCm vLLM wheel."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _module_file(module: ModuleType) -> str | None:
    value = getattr(module, "__file__", None)
    return str(value) if value is not None else None


def collect_report() -> dict[str, object]:
    """Collect runtime provenance and registered ROCm operator capabilities."""
    report: dict[str, object] = {
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "vllm_file": None,
        "vllm_import_error": None,
        "_C_file": None,
        "_C_import_error": None,
        "_rocm_C_file": None,
        "_rocm_C_import_error": None,
        "torch_hip_version": None,
        "torch_import_error": None,
        "gcn_arch_name": None,
        "device_error": None,
        "has_wvSplitK": False,
        "has_LLMM1": False,
        "ops_error": None,
    }

    try:
        vllm = importlib.import_module("vllm")
        report["vllm_file"] = _module_file(vllm)
    except Exception as error:
        report["vllm_import_error"] = _error_text(error)

    for module_name, file_key, error_key in (
        ("vllm._C", "_C_file", "_C_import_error"),
        ("vllm._rocm_C", "_rocm_C_file", "_rocm_C_import_error"),
    ):
        try:
            module = importlib.import_module(module_name)
            report[file_key] = _module_file(module)
        except Exception as error:
            report[error_key] = _error_text(error)

    try:
        torch = importlib.import_module("torch")
        report["torch_hip_version"] = getattr(torch.version, "hip", None)
    except Exception as error:
        report["torch_import_error"] = _error_text(error)
        return report

    try:
        properties = torch.cuda.get_device_properties(0)
        report["gcn_arch_name"] = getattr(properties, "gcnArchName", None)
    except Exception as error:
        report["device_error"] = _error_text(error)

    try:
        namespace = torch.ops._rocm_C
        report["has_wvSplitK"] = hasattr(namespace, "wvSplitK")
        report["has_LLMM1"] = hasattr(namespace, "LLMM1")
    except Exception as error:
        report["ops_error"] = _error_text(error)

    return report


def _canonical_arch(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().split(":", 1)[0]


def _is_within(path_value: object, prefix: Path) -> bool:
    if not isinstance(path_value, (str, bytes)) or not path_value:
        return False
    try:
        path = Path(path_value).expanduser().resolve(strict=False)
        path.relative_to(prefix)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _missing_detail(report: dict[str, object], error_key: str) -> str:
    return str(report.get(error_key) or "no module path reported")


def validate_report(
    report: dict[str, object],
    *,
    expected_prefix: str,
    required_arch: str | None,
    require_skinny: bool,
) -> list[str]:
    """Return deterministic contract violations for a collected report."""
    errors: list[str] = []
    prefix = Path(expected_prefix).expanduser().resolve(strict=False)
    prefix_text = str(prefix)

    sys_prefix = report.get("sys_prefix")
    if not _is_within(sys_prefix, prefix):
        errors.append(
            f"sys.prefix is outside expected prefix {prefix_text}: {sys_prefix}"
        )

    vllm_file = report.get("vllm_file")
    if not _is_within(vllm_file, prefix):
        errors.append(
            f"vllm.__file__ is outside expected prefix {prefix_text}: "
            f"{vllm_file}"
        )

    core_file = report.get("_C_file")
    if not core_file:
        errors.append(
            "Required extension vllm._C is unavailable: "
            f"{_missing_detail(report, '_C_import_error')}"
        )
    elif not _is_within(core_file, prefix):
        errors.append(
            f"vllm._C is outside expected prefix {prefix_text}: {core_file}"
        )

    if required_arch is not None:
        expected_arch = _canonical_arch(required_arch)
        actual_arch = _canonical_arch(report.get("gcn_arch_name"))
        if actual_arch != expected_arch:
            errors.append(
                "ROCm architecture mismatch: expected "
                f"{expected_arch or '<missing>'}, got "
                f"{actual_arch or '<missing>'}"
            )

    if require_skinny:
        rocm_file = report.get("_rocm_C_file")
        if not rocm_file:
            errors.append(
                "Required extension vllm._rocm_C is unavailable: "
                f"{_missing_detail(report, '_rocm_C_import_error')}"
            )
        elif not _is_within(rocm_file, prefix):
            errors.append(
                f"vllm._rocm_C is outside expected prefix {prefix_text}: "
                f"{rocm_file}"
            )
        if not report.get("has_wvSplitK"):
            errors.append(
                "Required ROCm op torch.ops._rocm_C.wvSplitK is unavailable"
            )
        if not report.get("has_LLMM1"):
            errors.append(
                "Required ROCm op torch.ops._rocm_C.LLMM1 is unavailable"
            )

    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the installed vLLM ROCm wheel before model load."
    )
    parser.add_argument("--expected-prefix", required=True)
    parser.add_argument("--require-arch")
    parser.add_argument("--require-skinny", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = collect_report()
    errors = validate_report(
        report,
        expected_prefix=args.expected_prefix,
        required_arch=args.require_arch,
        require_skinny=args.require_skinny,
    )
    print(json.dumps({"report": report, "errors": errors}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
