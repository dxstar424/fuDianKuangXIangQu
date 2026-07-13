from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RocmCapabilitiesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.capabilities = _load_module(
            "rocm_capabilities_under_test",
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
        cls.saved_modules = {
            name: sys.modules.get(name) for name in parent_paths
        }
        for name, path in parent_paths.items():
            parent = ModuleType(name)
            parent.__path__ = [str(path)]
            sys.modules[name] = parent

        shapes = _load_module(
            "vllm.model_executor.layers.rocm_skinny_shapes",
            ROOT / "vllm/model_executor/layers/rocm_skinny_shapes.py",
        )
        cls.policy = _load_module(
            "vllm.model_executor.layers.rocm_skinny_policy",
            ROOT / "vllm/model_executor/layers/rocm_skinny_policy.py",
        )
        cls.loaded_module_names = [shapes.__name__, cls.policy.__name__]

    @classmethod
    def tearDownClass(cls) -> None:
        for name in cls.loaded_module_names:
            sys.modules.pop(name, None)
        for name, saved in cls.saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

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
        lines = (ROOT / "vllm/platforms/rocm.py").read_text().splitlines()
        for assignment in ("_ON_MI3XX", "_ON_GFX9"):
            line = next(
                line for line in lines if line.startswith(f"{assignment} =")
            )
            self.assertNotIn("gfx936", line)


if __name__ == "__main__":
    unittest.main()
