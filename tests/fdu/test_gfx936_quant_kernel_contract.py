from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "csrc/fdu/gfx936_quant_gemv.hip"


class Gfx936QuantKernelContractTest(unittest.TestCase):
    def test_source_is_torch_header_free(self) -> None:
        text = SOURCE.read_text()
        lowered = text.lower()
        self.assertNotIn("torch/", lowered)
        self.assertNotIn("aten/", lowered)
        self.assertNotIn("pybind", lowered)

    def test_exports_all_four_c_launchers(self) -> None:
        text = SOURCE.read_text()
        for symbol in (
            "fdu_gfx936_w8a16_gemv",
            "fdu_gfx936_w4a16_gemv",
            "fdu_gfx936_w8_dequant",
            "fdu_gfx936_w4_dequant",
        ):
            with self.subTest(symbol=symbol):
                self.assertRegex(text, rf'extern\s+"C"\s+int\s+{symbol}\s*\(')

    def test_decode_uses_fixed_threads_and_real_k_specializations(self) -> None:
        text = SOURCE.read_text()
        self.assertRegex(text, r"BLOCK_THREADS\s*=\s*256")
        self.assertRegex(text, r"ROWS_PER_BLOCK\s*=\s*4")
        for k in (5120, 6144, 17408):
            self.assertIn(f"case {k}:", text)
        self.assertNotIn("gfx942", text.lower())


if __name__ == "__main__":
    unittest.main()
