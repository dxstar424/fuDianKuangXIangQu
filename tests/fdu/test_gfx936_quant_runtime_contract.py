from __future__ import annotations

import ast
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "vllm/model_executor/layers/gfx936_online_quant.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "gfx936_online_quant_runtime_under_test", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Symbol:
    def __init__(self) -> None:
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return 0


class _Library:
    def __init__(self) -> None:
        self.fdu_gfx936_w8a16_gemv = _Symbol()
        self.fdu_gfx936_w4a16_gemv = _Symbol()
        self.fdu_gfx936_w8_dequant = _Symbol()
        self.fdu_gfx936_w4_dequant = _Symbol()


class Gfx936QuantRuntimeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = MODULE_PATH.read_text()
        cls.tree = ast.parse(cls.source)
        cls.quant = _load_module()

    def test_torch_is_not_imported_at_module_scope(self) -> None:
        imports = [
            node
            for node in self.tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        self.assertFalse(
            any(
                (
                    isinstance(node, ast.Import)
                    and any(alias.name == "torch" for alias in node.names)
                )
                or (isinstance(node, ast.ImportFrom) and node.module == "torch")
                for node in imports
            )
        )

    def test_exact_abi_symbols_are_required(self) -> None:
        self.assertEqual(
            self.quant.REQUIRED_SYMBOLS,
            (
                "fdu_gfx936_w8a16_gemv",
                "fdu_gfx936_w4a16_gemv",
                "fdu_gfx936_w8_dequant",
                "fdu_gfx936_w4_dequant",
            ),
        )

    def test_loader_binds_pointer_and_integer_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_path = Path(directory) / "kernel.so"
            library_path.write_bytes(b"fixture")
            fake = _Library()
            with mock.patch.object(self.quant.ctypes, "CDLL", return_value=fake):
                loaded = self.quant.load_kernel_library(library_path)
            self.assertIs(loaded.library, fake)
            self.assertEqual(len(fake.fdu_gfx936_w8a16_gemv.argtypes), 7)
            self.assertEqual(len(fake.fdu_gfx936_w8_dequant.argtypes), 6)

    def test_loader_discovers_bundled_rocm_extension_without_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "vllm"
            package.mkdir()
            extension = package / "_rocm_C.abi3.so"
            extension.write_bytes(b"fixture")
            spec = mock.Mock(submodule_search_locations=[str(package)])
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(
                    self.quant.importlib.util, "find_spec", return_value=spec
                ),
            ):
                self.assertEqual(
                    self.quant.resolve_kernel_library_path(), extension.resolve()
                )

    def test_explicit_library_path_overrides_bundled_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            explicit = Path(directory) / "explicit.so"
            explicit.write_bytes(b"fixture")
            with mock.patch.dict(
                os.environ, {"FDU_GFX936_QUANT_SO": str(explicit)}, clear=True
            ):
                self.assertEqual(
                    self.quant.resolve_kernel_library_path(), explicit.resolve()
                )

    def test_missing_abi_symbol_is_reported_as_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_path = Path(directory) / "kernel.so"
            library_path.write_bytes(b"fixture")
            fake = _Library()
            del fake.fdu_gfx936_w4_dequant
            with mock.patch.object(self.quant.ctypes, "CDLL", return_value=fake):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"missing required ABI symbol.*fdu_gfx936_w4_dequant",
                ):
                    self.quant.load_kernel_library(library_path)

    def test_signature_binding_error_identifies_abi_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_path = Path(directory) / "kernel.so"
            library_path.write_bytes(b"fixture")
            fake = _Library()
            fake.fdu_gfx936_w8a16_gemv = object()
            with mock.patch.object(self.quant.ctypes, "CDLL", return_value=fake):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"failed to bind.*fdu_gfx936_w8a16_gemv",
                ):
                    self.quant.load_kernel_library(library_path)

    def test_kernel_argument_error_is_normalized_for_runtime_fallback(self) -> None:
        def reject_arguments(*args):
            raise self.quant.ctypes.ArgumentError("bad pointer")

        with self.assertRaisesRegex(
            RuntimeError, r"w8_gemv ABI invocation failed.*bad pointer"
        ):
            self.quant._call_kernel(reject_arguments, "w8_gemv")

    def test_missing_library_fails_without_loading_torch(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(FileNotFoundError):
                self.quant.load_kernel_library()


if __name__ == "__main__":
    unittest.main()
