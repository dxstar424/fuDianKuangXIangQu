from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_PATH = ROOT / "scripts/preflight_rocm.py"


def _load_preflight() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "fdu_preflight_rocm_under_test", PREFLIGHT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load preflight module from {PREFLIGHT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _valid_report() -> dict[str, object]:
    prefix = "/opt/vllm-gfx936"
    site_packages = f"{prefix}/lib/python3.11/site-packages"
    return {
        "sys_executable": f"{prefix}/bin/python",
        "sys_prefix": prefix,
        "vllm_file": f"{site_packages}/vllm/__init__.py",
        "vllm_import_error": None,
        "_C_file": f"{site_packages}/vllm/_C.so",
        "_C_import_error": None,
        "_rocm_C_file": f"{site_packages}/vllm/_rocm_C.so",
        "_rocm_C_import_error": None,
        "torch_hip_version": "7.1.0",
        "torch_import_error": None,
        "gcn_arch_name": "gfx936",
        "device_error": None,
        "has_wvSplitK": True,
        "has_LLMM1": True,
    }


class PreflightValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.preflight = _load_preflight()

    def _validate(
        self,
        report: dict[str, object],
        *,
        required_arch: str | None = "gfx936",
        require_skinny: bool = True,
    ) -> list[str]:
        return self.preflight.validate_report(
            report,
            expected_prefix="/opt/vllm-gfx936",
            required_arch=required_arch,
            require_skinny=require_skinny,
        )

    def test_accepts_valid_report(self) -> None:
        self.assertEqual(self._validate(_valid_report()), [])

    def test_canonicalizes_feature_suffixed_architecture(self) -> None:
        report = _valid_report()
        report["gcn_arch_name"] = "  GFX936:sramecc+:xnack-  "
        self.assertEqual(self._validate(report), [])

    def test_rejects_wrong_architecture(self) -> None:
        report = _valid_report()
        report["gcn_arch_name"] = "gfx942"
        self.assertEqual(
            self._validate(report),
            ["ROCm architecture mismatch: expected gfx936, got gfx942"],
        )

    def test_rejects_vllm_path_outside_expected_prefix(self) -> None:
        report = _valid_report()
        report["vllm_file"] = "/opt/vllm-gfx936-old/vllm/__init__.py"
        self.assertEqual(
            self._validate(report),
            [
                "vllm.__file__ is outside expected prefix "
                "/opt/vllm-gfx936: /opt/vllm-gfx936-old/vllm/__init__.py"
            ],
        )

    def test_rejects_sys_prefix_outside_expected_prefix(self) -> None:
        report = _valid_report()
        report["sys_prefix"] = "/usr"
        self.assertEqual(
            self._validate(report),
            [
                "sys.prefix is outside expected prefix "
                "/opt/vllm-gfx936: /usr"
            ],
        )

    def test_requires_core_extension(self) -> None:
        report = _valid_report()
        report["_C_file"] = None
        report["_C_import_error"] = "synthetic core import failure"
        self.assertEqual(
            self._validate(report),
            [
                "Required extension vllm._C is unavailable: "
                "synthetic core import failure"
            ],
        )

    def test_rejects_core_extension_outside_expected_prefix(self) -> None:
        report = _valid_report()
        report["_C_file"] = "/usr/lib/python3.11/site-packages/vllm/_C.so"
        self.assertEqual(
            self._validate(report),
            [
                "vllm._C is outside expected prefix /opt/vllm-gfx936: "
                "/usr/lib/python3.11/site-packages/vllm/_C.so"
            ],
        )

    def test_requires_rocm_extension_when_skinny_is_required(self) -> None:
        report = _valid_report()
        report["_rocm_C_file"] = None
        report["_rocm_C_import_error"] = "synthetic ROCm import failure"
        self.assertEqual(
            self._validate(report),
            [
                "Required extension vllm._rocm_C is unavailable: "
                "synthetic ROCm import failure"
            ],
        )

    def test_requires_wvsplitk_when_skinny_is_required(self) -> None:
        report = _valid_report()
        report["has_wvSplitK"] = False
        self.assertEqual(
            self._validate(report),
            [
                "Required ROCm op torch.ops._rocm_C.wvSplitK is unavailable"
            ],
        )

    def test_requires_llmm1_when_skinny_is_required(self) -> None:
        report = _valid_report()
        report["has_LLMM1"] = False
        self.assertEqual(
            self._validate(report),
            ["Required ROCm op torch.ops._rocm_C.LLMM1 is unavailable"],
        )

    def test_skinny_components_are_optional_when_not_required(self) -> None:
        report = _valid_report()
        report["_rocm_C_file"] = None
        report["_rocm_C_import_error"] = "not installed"
        report["has_wvSplitK"] = False
        report["has_LLMM1"] = False
        self.assertEqual(self._validate(report, require_skinny=False), [])


if __name__ == "__main__":
    unittest.main()
