from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/bench_gfx936_quant.py"
EXPECTED_SHAPES = (
    (16384, 5120),
    (96, 5120),
    (14336, 5120),
    (5120, 6144),
    (34816, 5120),
    (5120, 17408),
)
W4_MLP_SHAPES = frozenset({(34816, 5120), (5120, 17408)})


def _load_benchmark():
    if not SCRIPT.is_file():
        raise AssertionError(f"missing benchmark: {SCRIPT}")
    spec = importlib.util.spec_from_file_location(
        "gfx936_quant_benchmark_under_test", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot create import spec for {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"torch": None, "vllm": None}):
        spec.loader.exec_module(module)
    return module


class Gfx936QuantBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.benchmark = _load_benchmark()

    def test_row_admission_uses_runtime_thresholds_and_rejects_nonfinite(self) -> None:
        good = {"nrmse": 0.01, "cosine": 0.9995, "speedup": 1.11}
        self.assertTrue(self.benchmark.row_is_admitted("w8", good))
        self.assertFalse(
            self.benchmark.row_is_admitted("w8", {**good, "speedup": 1.09})
        )
        self.assertFalse(
            self.benchmark.row_is_admitted(
                "w8", {**good, "nrmse": float("nan")}
            )
        )
        self.assertTrue(
            self.benchmark.row_is_admitted(
                "w4", {"nrmse": 0.079, "cosine": 0.995, "speedup": 1.10}
            )
        )

    def test_declares_exact_six_shapes_and_only_two_w4_shapes(self) -> None:
        self.assertEqual(self.benchmark.QUANT_SHAPES, EXPECTED_SHAPES)
        self.assertEqual(self.benchmark.W4_MLP_SHAPES, W4_MLP_SHAPES)

    def test_cli_defaults_to_two_warmups_and_eight_repetitions(self) -> None:
        args = self.benchmark._parse_args(
            [
                "--mode",
                "w8",
                "--library",
                "/tmp/quant.so",
                "--output",
                "/tmp/report.json",
            ]
        )
        self.assertEqual(args.warmup, 2)
        self.assertEqual(args.repetitions, 8)

    def test_cli_rejects_nonpositive_counts(self) -> None:
        base = [
            "--mode",
            "w8",
            "--library",
            "/tmp/quant.so",
            "--output",
            "/tmp/report.json",
        ]
        with self.assertRaises(SystemExit):
            self.benchmark._parse_args([*base, "--warmup", "0"])
        with self.assertRaises(SystemExit):
            self.benchmark._parse_args([*base, "--repetitions", "-1"])

    def test_has_no_module_scope_torch_or_vllm_import(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        imported_roots: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertNotIn("torch", imported_roots)
        self.assertNotIn("vllm", imported_roots)


if __name__ == "__main__":
    unittest.main()
