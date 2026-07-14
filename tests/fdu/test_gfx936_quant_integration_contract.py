from __future__ import annotations

import ast
import importlib.util
import inspect
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
QUANT_PATH = ROOT / "vllm/model_executor/layers/gfx936_online_quant.py"
LAYER_UTILS_PATH = ROOT / "vllm/model_executor/layers/utils.py"
LINEAR_PATH = ROOT / "vllm/model_executor/layers/linear.py"
LOADER_UTILS_PATH = ROOT / "vllm/model_executor/model_loader/utils.py"


def _load_quant_module():
    spec = importlib.util.spec_from_file_location(
        "gfx936_online_quant_integration_under_test", QUANT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {QUANT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text()
    return source, ast.parse(source)


def _top_level_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one function named {name}, found {len(matches)}")
    return matches[0]


def _class_method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise AssertionError(
            f"expected one class named {class_name}, found {len(classes)}"
        )
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]
    if len(methods) != 1:
        raise AssertionError(
            f"expected one {class_name}.{method_name}, found {len(methods)}"
        )
    return methods[0]


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        if prefix is not None:
            return f"{prefix}.{node.attr}"
    return None


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call) and _dotted_name(candidate.func) == name
    ]


class Gfx936QuantIntegrationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.quant_source, cls.quant_tree = _parse(QUANT_PATH)
        cls.layer_utils_source, cls.layer_utils_tree = _parse(LAYER_UTILS_PATH)
        cls.linear_source, cls.linear_tree = _parse(LINEAR_PATH)
        cls.loader_source, cls.loader_tree = _parse(LOADER_UTILS_PATH)
        cls.quant = _load_quant_module()

    def test_admission_constants_and_public_benchmark_signature(self) -> None:
        expected = {
            "W8_NRMSE_LIMIT": 0.015,
            "W8_COSINE_LIMIT": 0.999,
            "W4_NRMSE_LIMIT": 0.08,
            "W4_COSINE_LIMIT": 0.995,
            "MIN_SPEEDUP": 1.10,
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertTrue(hasattr(self.quant, name), f"missing {name}")
                self.assertEqual(getattr(self.quant, name), value)

        self.assertTrue(
            hasattr(self.quant, "benchmark_candidate"),
            "missing benchmark_candidate",
        )
        signature = inspect.signature(self.quant.benchmark_candidate)
        self.assertEqual(
            list(signature.parameters),
            [
                "weight",
                "packed",
                "scale",
                "kind",
                "warmup_repetitions",
                "timed_repetitions",
            ],
        )
        self.assertEqual(
            signature.parameters["warmup_repetitions"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )
        self.assertEqual(
            signature.parameters["timed_repetitions"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )

    def test_quant_custom_op_has_real_fake_registration_and_wrapper(self) -> None:
        for name in (
            "gfx936_quant_linear_impl",
            "gfx936_quant_linear_fake",
            "gfx936_quant_linear",
        ):
            function = _top_level_function(self.layer_utils_tree, name)
            self.assertIsNotNone(function.returns)
            self.assertTrue(all(argument.annotation is not None for argument in function.args.args))

        registrations = [
            call
            for call in _calls(self.layer_utils_tree, "direct_register_custom_op")
            if any(
                keyword.arg == "op_name"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "gfx936_quant_linear"
                for keyword in call.keywords
            )
        ]
        self.assertEqual(len(registrations), 1)
        keyword_values = {
            keyword.arg: keyword.value
            for keyword in registrations[0].keywords
            if keyword.arg is not None
        }
        self.assertEqual(
            _dotted_name(keyword_values["op_func"]), "gfx936_quant_linear_impl"
        )
        self.assertEqual(
            _dotted_name(keyword_values["fake_impl"]), "gfx936_quant_linear_fake"
        )
        self.assertEqual(ast.literal_eval(keyword_values["mutates_args"]), [])

        implementation = _top_level_function(
            self.layer_utils_tree, "gfx936_quant_linear_impl"
        )
        wrapper = _top_level_function(self.layer_utils_tree, "gfx936_quant_linear")
        self.assertEqual(len(_calls(implementation, "quant_linear_impl")), 1)
        self.assertEqual(
            len(_calls(wrapper, "torch.ops.vllm.gfx936_quant_linear")), 1
        )

    def test_linear_converts_after_loading_and_dispatches_quant_first(self) -> None:
        process = _class_method(
            self.linear_tree,
            "UnquantizedLinearMethod",
            "process_weights_after_loading",
        )
        conversion_calls = _calls(process, "maybe_quantize_gfx936_layer")
        self.assertEqual(len(conversion_calls), 1)

        apply = _class_method(self.linear_tree, "UnquantizedLinearMethod", "apply")
        quantized_checks = _calls(apply, "is_gfx936_quantized_layer")
        quantized_dispatches = _calls(apply, "gfx936_quant_linear")
        batch_invariant_checks = _calls(apply, "vllm_is_batch_invariant")
        self.assertEqual(len(quantized_checks), 1)
        self.assertEqual(len(quantized_dispatches), 1)
        self.assertEqual(len(batch_invariant_checks), 1)
        self.assertLess(quantized_checks[0].lineno, quantized_dispatches[0].lineno)
        self.assertLess(quantized_dispatches[0].lineno, batch_invariant_checks[0].lineno)

    def test_loader_releases_allocator_once_before_attention_postprocessing(self) -> None:
        process = _top_level_function(
            self.loader_tree, "process_weights_after_loading"
        )
        online_calls = _calls(process, "online_quantization_active")
        empty_cache_calls = _calls(process, "torch.cuda.empty_cache")
        self.assertEqual(len(online_calls), 1)
        self.assertEqual(len(empty_cache_calls), 1)

        top_level_loops = [node for node in process.body if isinstance(node, ast.For)]
        self.assertGreaterEqual(len(top_level_loops), 2)
        first_loop, attention_loop = top_level_loops[:2]
        self.assertLess(first_loop.end_lineno, online_calls[0].lineno)
        self.assertLess(online_calls[0].lineno, empty_cache_calls[0].lineno)
        self.assertLess(empty_cache_calls[0].lineno, attention_loop.lineno)

    def test_w8_admission_accepts_boundary_and_rejects_worse_metrics(self) -> None:
        self.assertTrue(
            hasattr(self.quant, "evaluate_admission"), "missing evaluate_admission"
        )
        accepted = self.quant.evaluate_admission("w8", 0.015, 0.999, 11.0, 10.0)
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.speedup, 1.10)

        rejected_inputs = (
            (0.015001, 0.999, 11.0, 10.0),
            (0.015, 0.998999, 11.0, 10.0),
            (0.015, 0.999, 10.999, 10.0),
        )
        for arguments in rejected_inputs:
            with self.subTest(arguments=arguments):
                self.assertFalse(
                    self.quant.evaluate_admission("w8", *arguments).accepted
                )

    def test_w4_admission_accepts_boundary_and_rejects_non_finite_metrics(self) -> None:
        self.assertTrue(
            hasattr(self.quant, "evaluate_admission"), "missing evaluate_admission"
        )
        accepted = self.quant.evaluate_admission("w4", 0.08, 0.995, 11.0, 10.0)
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.speedup, 1.10)

        rejected_inputs = (
            (math.nan, 0.995, 11.0, 10.0),
            (0.08, math.inf, 11.0, 10.0),
            (0.08, 0.995, math.inf, 10.0),
        )
        for arguments in rejected_inputs:
            with self.subTest(arguments=arguments):
                self.assertFalse(
                    self.quant.evaluate_admission("w4", *arguments).accepted
                )

    def test_non_positive_candidate_time_rejects_without_division(self) -> None:
        self.assertTrue(
            hasattr(self.quant, "evaluate_admission"), "missing evaluate_admission"
        )
        for candidate_ms in (0.0, -1.0):
            with self.subTest(candidate_ms=candidate_ms):
                result = self.quant.evaluate_admission(
                    "w8", 0.0, 1.0, 11.0, candidate_ms
                )
                self.assertFalse(result.accepted)
                self.assertEqual(result.speedup, 0.0)


if __name__ == "__main__":
    unittest.main()
