import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/score_gfx936.py"


def _load_module():
    name = "score_gfx936_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load score module")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


class ScoreGfx936Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def _write(self, root, run, tier, throughput, ttft=100.0, tpot=10.0):
        path = root / "throughput" / run / f"{tier}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "output_throughput": throughput,
                    "p99_ttft_ms": ttft,
                    "p99_tpot_ms": tpot,
                }
            )
        )

    def test_medians_weights_formula_and_coefficient(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for tier in self.module.TIERS:
                self._write(root, "c1", tier, 10.0)
                self._write(root, "c2", tier, 12.0)
                self._write(root, "x1", tier, 16.0)
                self._write(root, "x2", tier, 18.0)
            result = self.module.score_results(
                results_root=root,
                control_runs=["c1", "c2"],
                candidate_runs=["x1", "x2"],
                accuracy_coefficient=0.9,
            )
        gain = (17.0 - 11.0) / 11.0
        expected = 100.0 * (0.6 + 0.4 * (1.0 - math.exp(-1.3 * gain)))
        self.assertAlmostEqual(result["tiers"]["8-16K"]["relative_gain"], gain)
        self.assertAlmostEqual(result["weighted_raw_score"], expected)
        self.assertAlmostEqual(result["final_score"], expected * 0.9)
        self.assertEqual(
            [result["tiers"][tier]["weight"] for tier in self.module.TIERS],
            [0.2, 0.5, 0.3],
        )

    def test_worst_p99_sla_failure_zeros_only_failed_tier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for tier in self.module.TIERS:
                self._write(root, "control", tier, 10.0, ttft=100.0, tpot=10.0)
                self._write(
                    root,
                    "candidate",
                    tier,
                    20.0,
                    ttft=151.0 if tier == "8-16K" else 100.0,
                    tpot=10.0,
                )
            result = self.module.score_results(
                results_root=root,
                control_runs=["control"],
                candidate_runs=["candidate"],
                accuracy_coefficient=1.0,
            )
        failed = result["tiers"]["8-16K"]
        self.assertFalse(failed["sla_pass"])
        self.assertEqual(failed["raw_tier_score"], 0.0)
        self.assertGreater(result["tiers"]["4-8K"]["raw_tier_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
