import importlib.util
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/bench_gfx936_skinny.py"


def _load_module():
    name = "bench_gfx936_skinny_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
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


class SkinnyBenchmarkMathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_derives_qwen35_shapes_from_config(self) -> None:
        config = {
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 8,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "attn_output_gate": True,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]
            * 2,
        }

        rows = self.module.derive_qwen35_shapes(config)

        self.assertEqual(
            {(row["m"], row["k"]) for row in rows},
            {
                (12288, 4096),
                (64, 4096),
                (10240, 4096),
                (4096, 4096),
                (24576, 4096),
                (4096, 12288),
            },
        )
        by_family = {row["family"]: row for row in rows}
        self.assertEqual(by_family["gdn_qkvz"]["layer_count"], 6)
        self.assertEqual(by_family["full_attention_qkv_gate"]["layer_count"], 2)
        self.assertEqual(by_family["attention_output"]["layer_count"], 8)

    def test_derives_nested_text_config_and_default_layer_pattern(self) -> None:
        rows = self.module.derive_qwen35_shapes(
            {
                "text_config": {
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 4,
                    "full_attention_interval": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 8,
                    "attn_output_gate": False,
                    "linear_key_head_dim": 4,
                    "linear_value_head_dim": 8,
                    "linear_num_key_heads": 2,
                    "linear_num_value_heads": 4,
                }
            }
        )
        by_family = {row["family"]: row for row in rows}
        self.assertEqual(by_family["gdn_qkvz"]["m"], 80)
        self.assertEqual(by_family["full_attention_qkv_gate"]["m"], 64)
        self.assertEqual(by_family["gdn_ba"]["layer_count"], 2)
        self.assertEqual(by_family["full_attention_qkv_gate"]["layer_count"], 2)

    def test_vector_metrics(self) -> None:
        self.assertAlmostEqual(self.module.cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(self.module.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertEqual(self.module.cosine_similarity([0.0], [0.0]), 1.0)
        self.assertAlmostEqual(self.module.relative_l2([2.0, 0.0], [1.0, 0.0]), 0.5)
        self.assertEqual(self.module.relative_l2([0.0], [0.0]), 0.0)
        self.assertEqual(self.module.relative_l2([0.0], [1.0]), float("inf"))
        with self.assertRaises(ValueError):
            self.module.cosine_similarity([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            self.module.relative_l2([], [])

    def test_percentile_uses_linear_interpolation(self) -> None:
        self.assertEqual(self.module.percentile([4.0, 1.0, 3.0, 2.0], 0.0), 1.0)
        self.assertEqual(self.module.percentile([4.0, 1.0, 3.0, 2.0], 1.0), 4.0)
        self.assertAlmostEqual(self.module.percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)
        with self.assertRaises(ValueError):
            self.module.percentile([], 0.5)
        with self.assertRaises(ValueError):
            self.module.percentile([1.0], 1.01)

    def test_projects_total_latency_not_mean_ratio(self) -> None:
        families = {
            "gdn_qkvz": 6,
            "gdn_ba": 6,
            "full_attention_qkv_gate": 2,
            "attention_output": 8,
            "mlp_gate_up": 8,
            "mlp_down": 8,
        }
        rows = []
        for n in (1, 2, 4):
            for family, layer_count in families.items():
                rows.append(
                    {
                        "n": n,
                        "family": family,
                        "layer_count": layer_count,
                        "stock_median_ms": 2.0,
                        "candidate_median_ms": 1.0,
                        "admitted": True,
                    }
                )
        self.assertEqual(self.module.project_linear_speedup(rows), {1: 2.0, 2: 2.0, 4: 2.0})
        rows.pop()
        self.assertEqual(self.module.project_linear_speedup(rows)[4], 0.0)

    def test_render_whitelist_is_strict_sorted_and_deduplicated(self) -> None:
        good = {
            "n": 2,
            "m": 4096,
            "k": 4096,
            "dtype": "bfloat16",
            "bias_present": False,
            "finite": True,
            "assert_close": True,
            "cosine_similarity": 0.9999,
            "relative_l2": 0.001,
            "stock_median_ms": 2.0,
            "candidate_median_ms": 1.0,
            "stock_p99_ms": 2.2,
            "candidate_p99_ms": 1.2,
            "speedup": 2.0,
        }
        second = dict(good, n=1, m=12288)
        rejected = dict(good, n=4, speedup=1.149)

        rendered = self.module.render_whitelist_module([good, rejected, second, good])

        self.assertIn('(1, 12288, 4096, "bfloat16", False)', rendered)
        self.assertIn('(2, 4096, 4096, "bfloat16", False)', rendered)
        self.assertNotIn("(4,", rendered)
        self.assertEqual(rendered.count('(2, 4096, 4096, "bfloat16", False)'), 1)
        self.assertLess(rendered.index("(1, 12288"), rendered.index("(2, 4096"))
        self.assertEqual(
            self.module.render_whitelist_module([]).splitlines()[-1],
            "VALIDATED_GFX936_SHAPES: frozenset[SkinnyShape] = frozenset()",
        )

    def test_module_has_no_top_level_gpu_imports(self) -> None:
        source = SCRIPT.read_text()
        top_level_prefix = source.split("def run_gpu_benchmark", 1)[0]
        self.assertNotIn("import torch", top_level_prefix)
        self.assertNotIn("import vllm", top_level_prefix)


if __name__ == "__main__":
    unittest.main()
