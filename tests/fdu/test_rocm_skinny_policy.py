from __future__ import annotations

import ast
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
_MISSING = object()


def _snapshot_modules(names: tuple[str, ...]) -> dict[str, object]:
    return {name: sys.modules.get(name, _MISSING) for name in names}


def _restore_modules(saved_modules: dict[str, object]) -> None:
    for name, saved in reversed(saved_modules.items()):
        if saved is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved  # type: ignore[assignment]


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    saved = _snapshot_modules((name,))
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        _restore_modules(saved)
        raise
    return module


class ModuleIsolationTest(unittest.TestCase):
    @staticmethod
    def _restore_entries(saved_modules: dict[str, object]) -> None:
        for name, saved in saved_modules.items():
            if saved is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved  # type: ignore[assignment]

    def test_failed_load_restores_preexisting_module(self) -> None:
        name = "fdu_failing_test_module"
        saved = {name: sys.modules.get(name, _MISSING)}
        sentinel = ModuleType(name)
        sys.modules[name] = sentinel
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "raises.py"
                path.write_text("raise RuntimeError('expected')\n")
                with self.assertRaisesRegex(RuntimeError, "expected"):
                    _load_module(name, path)
            self.assertIs(sys.modules.get(name), sentinel)
        finally:
            self._restore_entries(saved)

    def test_failed_load_removes_new_module(self) -> None:
        name = "fdu_new_failing_test_module"
        saved = {name: sys.modules.get(name, _MISSING)}
        sys.modules.pop(name, None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "raises.py"
                path.write_text("raise RuntimeError('expected')\n")
                with self.assertRaisesRegex(RuntimeError, "expected"):
                    _load_module(name, path)
            self.assertNotIn(name, sys.modules)
        finally:
            self._restore_entries(saved)

    def test_capability_lifecycle_does_not_leak_module(self) -> None:
        name = "rocm_capabilities_under_test"
        saved = {name: sys.modules.get(name, _MISSING)}
        sys.modules.pop(name, None)

        class LifecycleCase(RocmCapabilitiesTest):
            pass

        LifecycleCase._class_cleanups = []
        try:
            LifecycleCase.setUpClass()
            LifecycleCase.doClassCleanups()
            self.assertNotIn(name, sys.modules)
        finally:
            LifecycleCase.doClassCleanups()
            self._restore_entries(saved)

    def test_policy_lifecycle_restores_preexisting_child_modules(self) -> None:
        module_names = (
            "vllm",
            "vllm.model_executor",
            "vllm.model_executor.layers",
            "vllm.model_executor.layers.rocm_skinny_shapes",
            "vllm.model_executor.layers.rocm_skinny_policy",
        )
        saved = {
            name: sys.modules.get(name, _MISSING) for name in module_names
        }
        child_names = module_names[-2:]
        sentinels = {name: ModuleType(name) for name in child_names}
        sys.modules.update(sentinels)

        class LifecycleCase(RocmSkinnyPolicyTest):
            @classmethod
            def setUpClass(cls) -> None:
                super().setUpClass()
                raise RuntimeError("expected setup failure")

            def runTest(self) -> None:
                self.fail("setUpClass failure should prevent this test")

        LifecycleCase._class_cleanups = []
        try:
            result = unittest.TestResult()
            unittest.TestSuite((LifecycleCase(),)).run(result)
            self.assertEqual(len(result.errors), 1)
            self.assertIn("expected setup failure", result.errors[0][1])
            for name, sentinel in sentinels.items():
                self.assertIs(sys.modules.get(name), sentinel)
        finally:
            LifecycleCase.doClassCleanups()
            self._restore_entries(saved)


class RocmCapabilitiesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module_name = "rocm_capabilities_under_test"
        saved_modules = _snapshot_modules((module_name,))
        cls.addClassCleanup(_restore_modules, saved_modules)
        cls.capabilities = _load_module(
            module_name,
            ROOT / "vllm/platforms/rocm_capabilities.py",
        )

    def test_canonical_arch_strips_feature_suffix(self) -> None:
        self.assertEqual(
            self.capabilities.canonical_rocm_arch("gfx936:sramecc+:xnack-"),
            "gfx936",
        )

    def test_gfx936_arch_accepts_feature_suffix(self) -> None:
        self.assertTrue(
            self.capabilities.is_gfx936_arch("gfx936:sramecc+:xnack-")
        )

    def test_skinny_gemm_arch_allowlist(self) -> None:
        self.assertTrue(
            self.capabilities.supports_rocm_skinny_gemm_arch("gfx936")
        )
        self.assertFalse(
            self.capabilities.supports_rocm_skinny_gemm_arch("gfx1100")
        )


class RocmSkinnyPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        parent_paths = {
            "vllm": ROOT / "vllm",
            "vllm.model_executor": ROOT / "vllm/model_executor",
            "vllm.model_executor.layers": ROOT / "vllm/model_executor/layers",
        }
        child_module_names = (
            "vllm.model_executor.layers.rocm_skinny_shapes",
            "vllm.model_executor.layers.rocm_skinny_policy",
        )
        module_names = (*parent_paths, *child_module_names)
        saved_modules = _snapshot_modules(module_names)
        cls.addClassCleanup(_restore_modules, saved_modules)
        for name, path in parent_paths.items():
            parent = ModuleType(name)
            parent.__path__ = [str(path)]
            sys.modules[name] = parent

        cls.shapes = _load_module(
            child_module_names[0],
            ROOT / "vllm/model_executor/layers/rocm_skinny_shapes.py",
        )
        cls.policy = _load_module(
            child_module_names[1],
            ROOT / "vllm/model_executor/layers/rocm_skinny_policy.py",
        )

    def _eligible(self, **overrides: object) -> bool:
        arguments: dict[str, object] = {
            "n": 1,
            "m": 4096,
            "k": 4096,
            "dtype_name": "bfloat16",
            "bias_present": False,
            "weight_contiguous": True,
            "activation_reshapeable": True,
        }
        arguments.update(overrides)
        if "validated_shapes" not in arguments:
            arguments["validated_shapes"] = frozenset(
                {
                    (
                        arguments["n"],
                        arguments["m"],
                        arguments["k"],
                        arguments["dtype_name"],
                        arguments["bias_present"],
                    )
                }
            )
        return self.policy.is_gfx936_skinny_eligible(**arguments)

    def test_validated_shape_is_eligible(self) -> None:
        passing = frozenset({(1, 4096, 4096, "bfloat16", False)})
        self.assertTrue(
            self.policy.is_gfx936_skinny_eligible(
                n=1,
                m=4096,
                k=4096,
                dtype_name="bfloat16",
                bias_present=False,
                weight_contiguous=True,
                activation_reshapeable=True,
                validated_shapes=passing,
            )
        )

    def test_committed_whitelist_contains_only_measured_llmm1_shapes(self) -> None:
        self.assertEqual(
            self.shapes.VALIDATED_GFX936_SHAPES,
            frozenset(
                {
                    (1, 16384, 5120, "bfloat16", False),
                    (1, 96, 5120, "bfloat16", False),
                    (1, 14336, 5120, "bfloat16", False),
                    (1, 5120, 6144, "bfloat16", False),
                    (1, 34816, 5120, "bfloat16", False),
                }
            ),
        )

    def test_rejects_unmeasured_multirow_batches(self) -> None:
        self.assertFalse(self._eligible(n=2))
        self.assertFalse(self._eligible(n=4))

    def test_rejects_unsupported_batch_size(self) -> None:
        self.assertFalse(self._eligible(n=5))

    def test_rejects_unsupported_dtype(self) -> None:
        self.assertFalse(self._eligible(dtype_name="float32"))

    def test_rejects_bias(self) -> None:
        self.assertFalse(self._eligible(bias_present=True))

    def test_rejects_noncontiguous_weight(self) -> None:
        self.assertFalse(self._eligible(weight_contiguous=False))

    def test_rejects_nonreshapeable_activation(self) -> None:
        self.assertFalse(self._eligible(activation_reshapeable=False))

    def test_rejects_unaligned_k(self) -> None:
        self.assertFalse(self._eligible(k=4095))

    def test_rejects_shape_absent_from_validated_shapes(self) -> None:
        self.assertFalse(self._eligible(validated_shapes=frozenset()))


class RocmPlatformIsolationTest(unittest.TestCase):
    def test_gfx936_is_not_in_broad_architecture_predicates(self) -> None:
        text = (ROOT / "vllm/platforms/rocm.py").read_text()
        tree = ast.parse(text)
        for assignment in ("_ON_MI3XX", "_ON_GFX9"):
            node = next(
                node
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == assignment
                    for target in node.targets
                )
            )
            expression = ast.get_source_segment(text, node.value)
            self.assertIsNotNone(expression)
            self.assertNotIn("gfx936", expression)


if __name__ == "__main__":
    unittest.main()
