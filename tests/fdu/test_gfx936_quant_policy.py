from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "vllm/model_executor/layers/gfx936_online_quant.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "gfx936_online_quant_policy_under_test", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Gfx936QuantPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.quant = _load_module()

    def test_mode_parser_defaults_w8_but_invalid_values_fail_closed(self) -> None:
        self.assertEqual(self.quant.parse_quant_mode(None), "w8")
        self.assertEqual(self.quant.parse_quant_mode("W8"), "w8")
        self.assertEqual(self.quant.parse_quant_mode(" hybrid_w4 "), "hybrid_w4")
        self.assertEqual(self.quant.parse_quant_mode("unexpected"), "off")

    def test_exact_six_shapes_are_admitted(self) -> None:
        self.assertEqual(
            self.quant.QUANT_SHAPES,
            frozenset(
                {
                    (16384, 5120),
                    (96, 5120),
                    (14336, 5120),
                    (5120, 6144),
                    (34816, 5120),
                    (5120, 17408),
                }
            ),
        )
        self.assertFalse(self.quant.is_quant_shape(5120, 5120))

    def test_hybrid_tries_w4_then_w8_only_for_mlp(self) -> None:
        self.assertEqual(
            self.quant.candidate_kinds("hybrid_w4", 34816, 5120),
            ("w4", "w8"),
        )
        self.assertEqual(
            self.quant.candidate_kinds("hybrid_w4", 16384, 5120),
            ("w8",),
        )
        self.assertEqual(self.quant.candidate_kinds("off", 34816, 5120), ())

    def test_row_chunks_bound_original_bf16_bytes(self) -> None:
        chunks = list(self.quant.iter_row_chunks(34816, 5120, 64 << 20))
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], 34816)
        self.assertTrue(all((end - start) * 5120 * 2 <= 64 << 20 for start, end in chunks))

    def test_w8_reference_round_trip(self) -> None:
        packed, scale = self.quant.quantize_row_w8_reference([-1.0, 0.0, 1.0])
        self.assertEqual(packed, [-127, 0, 127])
        self.assertAlmostEqual(scale, 1.0 / 127.0)

    def test_w4_reference_packs_low_nibble_first(self) -> None:
        row = [-1.0, 1.0] + [0.0] * 30
        packed, scales = self.quant.quantize_group_w4_reference(row, group_size=32)
        unpacked = self.quant.unpack_group_w4_reference(packed, 32)
        self.assertEqual(unpacked[:2], [-7, 7])
        self.assertEqual(len(packed), 16)
        self.assertEqual(len(scales), 1)
        self.assertAlmostEqual(scales[0], 1.0 / 7.0)


if __name__ == "__main__":
    unittest.main()
