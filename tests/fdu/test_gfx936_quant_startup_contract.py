from __future__ import annotations

import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = ROOT / "scripts/preflight_gfx936_quant.py"
LAUNCH = ROOT / "launch.sh"
ROCM_ENV = ROOT / "scripts/rocm_env.sh"
DOCKERFILE = ROOT / "Dockerfile"
CMAKE = ROOT / "CMakeLists.txt"

REQUIRED_SYMBOLS = (
    "fdu_gfx936_w8a16_gemv",
    "fdu_gfx936_w4a16_gemv",
    "fdu_gfx936_w8_dequant",
    "fdu_gfx936_w4_dequant",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_preflight():
    if not PREFLIGHT.is_file():
        raise AssertionError(f"missing startup preflight: {PREFLIGHT}")
    spec = importlib.util.spec_from_file_location(
        "gfx936_quant_preflight_under_test", PREFLIGHT
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot create import spec for {PREFLIGHT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeLibrary:
    def __init__(self, missing: str) -> None:
        for name in REQUIRED_SYMBOLS:
            if name != missing:
                setattr(self, name, object())


class Gfx936QuantStartupContractTest(unittest.TestCase):
    def test_preflight_imports_without_torch_and_exposes_symbol_validation(self) -> None:
        preflight = _load_preflight()
        self.assertTrue(callable(preflight.validate_symbols))

    def test_preflight_has_no_module_scope_torch_or_vllm_import(self) -> None:
        self.assertTrue(PREFLIGHT.is_file(), f"missing {PREFLIGHT}")
        tree = ast.parse(_read(PREFLIGHT))
        imported_roots: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertNotIn("torch", imported_roots)
        self.assertNotIn("vllm", imported_roots)

    def test_preflight_declares_exact_native_abi_and_rejects_each_missing_symbol(
        self,
    ) -> None:
        preflight = _load_preflight()
        self.assertEqual(preflight.REQUIRED_SYMBOLS, REQUIRED_SYMBOLS)
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "quant.so"
            library.write_bytes(b"not-empty")
            for missing in REQUIRED_SYMBOLS:
                fake = _FakeLibrary(missing)
                with self.subTest(missing=missing), mock.patch.object(
                    preflight.ctypes, "CDLL", return_value=fake
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        rf"missing required ABI symbol.*{missing}",
                    ):
                        preflight.validate_symbols(library)

    def test_bundled_extension_loads_torch_dependencies_before_cdll(self) -> None:
        preflight = _load_preflight()
        events: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "_rocm_C.abi3.so"
            library.write_bytes(b"not-empty")
            fake = _FakeLibrary(missing="")
            with (
                mock.patch.object(
                    preflight.importlib,
                    "import_module",
                    side_effect=lambda name: events.append(name),
                ),
                mock.patch.object(
                    preflight.ctypes,
                    "CDLL",
                    side_effect=lambda path: events.append("cdll") or fake,
                ),
            ):
                preflight.validate_symbols(library)
        self.assertEqual(events, ["torch", "cdll"])

    def test_rocm_wheel_embeds_quant_kernel(self) -> None:
        text = _read(CMAKE)
        rocm_sources = text.split("set(VLLM_ROCM_EXT_SRC", 1)[1].split(")", 1)[0]
        self.assertIn('"csrc/fdu/gfx936_quant_gemv.hip"', rocm_sources)

    def test_launch_smoke_checks_bundled_quant_before_changing_directory(self) -> None:
        text = _read(LAUNCH)
        self.assertNotIn("build_gfx936_quant_jit.py", text)
        self.assertIn("preflight_gfx936_quant.py", text)
        self.assertNotIn("gfx936 quant JIT/preflight failed", text)

        native_preflight = text.index('"$PYTHON_BIN" "${PREFLIGHT_ARGS[@]}"')
        quant_preflight = text.index("preflight_gfx936_quant.py")
        change_directory = text.index("cd /tmp")
        self.assertLess(native_preflight, quant_preflight)
        self.assertLess(quant_preflight, change_directory)

    def test_preflight_can_resolve_bundled_rocm_extension(self) -> None:
        preflight = _load_preflight()
        self.assertTrue(callable(preflight.find_bundled_library))

    def test_quant_mode_defaults_w8_in_environment_and_image(self) -> None:
        self.assertIn(
            'FDU_GFX936_QUANT_MODE="${FDU_GFX936_QUANT_MODE:-w8}"',
            _read(ROCM_ENV),
        )
        self.assertIn("FDU_GFX936_QUANT_MODE=w8", _read(DOCKERFILE))


if __name__ == "__main__":
    unittest.main()
