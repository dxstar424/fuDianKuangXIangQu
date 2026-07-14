from __future__ import annotations

import importlib.util
import stat
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build_gfx936_quant_jit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gfx936_jit_builder_under_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Gfx936JitBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = _load_module()

    def _fake_compiler(self, directory: Path, *, sleep: float = 0.0) -> Path:
        compiler = directory / "hipcc"
        compiler.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, sys, time\n"
            f"time.sleep({sleep!r})\n"
            "if '--version' in sys.argv:\n"
            "    print('fake hipcc 1.0')\n"
            "    raise SystemExit(0)\n"
            "out = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
            "out.write_bytes(b'fake-so')\n"
        )
        compiler.chmod(compiler.stat().st_mode | stat.S_IXUSR)
        return compiler

    def test_command_targets_only_gfx936_and_has_no_torch_include(self) -> None:
        command = self.builder.build_command(
            Path("/opt/rocm/bin/hipcc"), Path("kernel.hip"), Path("kernel.so"), "gfx936"
        )
        self.assertIn("--offload-arch=gfx936", command)
        self.assertIn("-shared", command)
        self.assertNotIn("torch", " ".join(command).lower())

    def test_cache_key_changes_with_source_or_compiler(self) -> None:
        first = self.builder.cache_key(b"one", "hipcc-a", "gfx936")
        second = self.builder.cache_key(b"two", "hipcc-a", "gfx936")
        third = self.builder.cache_key(b"one", "hipcc-b", "gfx936")
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)

    def test_compile_is_atomic_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            compiler = self._fake_compiler(directory)
            source = directory / "kernel.hip"
            source.write_text("extern \"C\" int kernel() { return 0; }\n")
            output = self.builder.compile_kernel(
                source=source,
                cache_root=directory / "cache",
                compiler=compiler,
                arch="gfx936",
                timeout_s=1.0,
            )
            self.assertEqual(output.read_bytes(), b"fake-so")
            self.assertEqual(
                output,
                self.builder.compile_kernel(source, directory / "cache", compiler, "gfx936", 1.0),
            )
            self.assertEqual(list((directory / "cache").glob("*.tmp.*")), [])

    def test_timeout_removes_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            compiler = self._fake_compiler(directory, sleep=0.2)
            source = directory / "kernel.hip"
            source.write_text("extern \"C\" int kernel() { return 0; }\n")
            with self.assertRaises(self.builder.BuildError):
                self.builder.compile_kernel(
                    source, directory / "cache", compiler, "gfx936", 0.02
                )
            self.assertEqual(list((directory / "cache").glob("*")), [])


if __name__ == "__main__":
    unittest.main()
