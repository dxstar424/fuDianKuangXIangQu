from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class Gfx936BuildContractTest(unittest.TestCase):
    def test_hip_build_requests_rocm_extension(self) -> None:
        text = (ROOT / "setup.py").read_text()
        self.assertRegex(
            text,
            re.compile(
                r"if _is_hip\(\):\s+"
                r"ext_modules\.append\(CMakeExtension\(name=\"vllm\._rocm_C\"\)\)"
            ),
        )

    def test_gfx936_compiles_only_the_gfx9_skinny_family(self) -> None:
        text = (ROOT / "csrc/rocm/skinny_gemms.cu").read_text()
        gfx9_block, mi3xx_tail = text.split("#if defined(__HIPCC__)", 2)[1:]
        self.assertIn("defined(__gfx936__)", gfx9_block)
        mi3xx_block = mi3xx_tail.split("#endif", 1)[0]
        self.assertNotIn("defined(__gfx936__)", mi3xx_block)


if __name__ == "__main__":
    unittest.main()
