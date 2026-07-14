from __future__ import annotations

import ast
import os
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
ENVS_PATH = ROOT / "vllm/envs.py"
UTILS_PATH = ROOT / "vllm/model_executor/layers/utils.py"
ROCM_PATH = ROOT / "vllm/platforms/rocm.py"
_MISSING = object()
_STUBBED_MODULE_NAMES = (
    "vllm",
    "vllm.platforms",
    "vllm.platforms.rocm",
    "vllm.model_executor",
    "vllm.model_executor.layers",
    "vllm.model_executor.layers.rocm_skinny_policy",
    "aiter",
    "aiter.ops",
    "aiter.ops.triton",
    "aiter.ops.triton.gemm_a16w16",
)


def _parse(path: Path) -> tuple[str, ast.Module]:
    text = path.read_text()
    return text, ast.parse(text)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one function named {name}, found {len(matches)}"
        )
    return matches[0]


def _assignment_index(function: ast.FunctionDef, name: str) -> int:
    for index, node in enumerate(function.body):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return index
    raise AssertionError(f"missing top-level assignment to {name}")


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


def _expression(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _assignment_value(source: str) -> ast.expr:
    assignment = ast.parse(source).body[0]
    if not isinstance(assignment, ast.Assign):
        raise AssertionError("test helper expected an assignment")
    return assignment.value


def _snapshot_modules(names: tuple[str, ...]) -> dict[str, object]:
    return {name: sys.modules.get(name, _MISSING) for name in names}


def _restore_modules(saved_modules: dict[str, object]) -> None:
    for name, saved in reversed(saved_modules.items()):
        if saved is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved  # type: ignore[assignment]


class _FakeDtype:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return f"torch.{self.name}"


class _FakeTensor:
    def __init__(
        self,
        label: str,
        shape: tuple[int, ...],
        dtype: _FakeDtype,
        *,
        contiguous: bool = True,
    ) -> None:
        self.label = label
        self.shape = shape
        self.dtype = dtype
        self._contiguous = contiguous

    def numel(self) -> int:
        product = 1
        for dimension in self.shape:
            product *= dimension
        return product

    def size(self, dimension: int) -> int:
        return self.shape[dimension]

    def reshape(self, *shape: int) -> _FakeTensor:
        return self

    def is_contiguous(self) -> bool:
        return self._contiguous


class _FakeResult:
    def __init__(self, origin: str) -> None:
        self.origin = origin
        self.reshape_calls: list[tuple[int, ...]] = []

    def reshape(self, *shape: int) -> _FakeResult:
        self.reshape_calls.append(shape)
        return self


@dataclass(frozen=True)
class _DispatchScenario:
    force_stock: bool = False
    use_skinny: bool = True
    supports_skinny: bool = True
    gfx936: bool = False
    gfx9: bool = False
    gfx950: bool = False
    policy_result: bool | BaseException = False
    aiter_result: bool = False
    n: int = 1
    m: int = 4096
    k: int = 4096
    dtype_name: str = "bfloat16"
    weight_dtype_name: str | None = None
    bias: object | None = None
    weight_contiguous: bool = True
    cu_count: int = 120


@dataclass(frozen=True)
class _DispatchRun:
    result: _FakeResult
    events: list[tuple[object, ...]]
    x: _FakeTensor
    weight: _FakeTensor


def _fake_package(name: str) -> ModuleType:
    package = ModuleType(name)
    package.__path__ = []  # type: ignore[attr-defined]
    return package


def _compile_dispatch(namespace: dict[str, object]):
    text, tree = _parse(UTILS_PATH)
    function = _function(tree, "rocm_unquantized_gemm_impl")
    function_source = ast.get_source_segment(text, function)
    if function_source is None:
        raise AssertionError("could not extract ROCm GEMM dispatch source")
    source = "from __future__ import annotations\n" + function_source
    exec(compile(source, UTILS_PATH, "exec"), namespace)
    return namespace["rocm_unquantized_gemm_impl"]


def _run_dispatch(scenario: _DispatchScenario) -> _DispatchRun:
    events: list[tuple[object, ...]] = []
    bfloat16 = _FakeDtype("bfloat16")
    float16 = _FakeDtype("float16")
    dtypes = {"bfloat16": bfloat16, "float16": float16}
    x_dtype = dtypes[scenario.dtype_name]
    weight_dtype = dtypes[scenario.weight_dtype_name or scenario.dtype_name]
    x = _FakeTensor("x", (scenario.n, scenario.k), x_dtype)
    weight = _FakeTensor(
        "weight",
        (scenario.m, scenario.k),
        weight_dtype,
        contiguous=scenario.weight_contiguous,
    )

    def record(name: str, *arguments: object) -> None:
        events.append((name, *arguments))

    def stock_linear(
        actual_x: _FakeTensor,
        actual_weight: _FakeTensor,
        actual_bias: object | None,
    ) -> _FakeResult:
        record("stock", actual_x, actual_weight, actual_bias)
        return _FakeResult("stock")

    def num_compute_units() -> int:
        record("cu")
        return scenario.cu_count

    def use_aiter_triton_gemm(
        n: int, m: int, k: int, dtype: _FakeDtype
    ) -> bool:
        record("aiter_select", n, m, k, str(dtype))
        return scenario.aiter_result

    def wvsplitkrc(
        actual_x: _FakeTensor,
        actual_weight: _FakeTensor,
        cu_count: int,
        bias: object | None,
    ) -> _FakeResult:
        record("wvSplitKrc", actual_x, actual_weight, cu_count, bias)
        return _FakeResult("wvSplitKrc")

    def wvsplitk(
        actual_weight: _FakeTensor,
        actual_x: _FakeTensor,
        cu_count: int,
        bias: object | None,
    ) -> _FakeResult:
        record("wvSplitK", actual_weight, actual_x, cu_count, bias)
        return _FakeResult("wvSplitK")

    def llmm1(
        actual_weight: _FakeTensor,
        actual_x: _FakeTensor,
        split_count: int,
    ) -> _FakeResult:
        record("LLMM1", actual_weight, actual_x, split_count)
        return _FakeResult("LLMM1")

    def platform_flag(name: str, result: bool):
        def getter() -> bool:
            record(name)
            return result

        return getter

    def policy(**arguments: object) -> bool:
        record("policy", arguments)
        if isinstance(scenario.policy_result, BaseException):
            raise scenario.policy_result
        return scenario.policy_result

    def aiter_gemm(
        actual_x: _FakeTensor,
        actual_weight: _FakeTensor,
        actual_bias: object | None,
    ) -> _FakeResult:
        record("aiter", actual_x, actual_weight, actual_bias)
        return _FakeResult("aiter")

    vllm = _fake_package("vllm")
    platforms = _fake_package("vllm.platforms")
    rocm = ModuleType("vllm.platforms.rocm")
    rocm.on_gfx936 = platform_flag("on_gfx936", scenario.gfx936)
    rocm.on_gfx9 = platform_flag("on_gfx9", scenario.gfx9)
    rocm.on_gfx950 = platform_flag("on_gfx950", scenario.gfx950)
    rocm.supports_rocm_skinny_gemm = platform_flag(
        "supports_skinny", scenario.supports_skinny
    )
    vllm.platforms = platforms
    platforms.rocm = rocm

    model_executor = _fake_package("vllm.model_executor")
    layers = _fake_package("vllm.model_executor.layers")
    policy_module = ModuleType(
        "vllm.model_executor.layers.rocm_skinny_policy"
    )
    policy_module.is_gfx936_skinny_eligible = policy
    vllm.model_executor = model_executor
    model_executor.layers = layers
    layers.rocm_skinny_policy = policy_module

    aiter = _fake_package("aiter")
    aiter_ops = _fake_package("aiter.ops")
    aiter_triton = _fake_package("aiter.ops.triton")
    aiter_gemm_module = ModuleType("aiter.ops.triton.gemm_a16w16")
    aiter_gemm_module.gemm_a16w16 = aiter_gemm
    aiter.ops = aiter_ops
    aiter_ops.triton = aiter_triton
    aiter_triton.gemm_a16w16 = aiter_gemm_module

    fake_modules = {
        "vllm": vllm,
        "vllm.platforms": platforms,
        "vllm.platforms.rocm": rocm,
        "vllm.model_executor": model_executor,
        "vllm.model_executor.layers": layers,
        "vllm.model_executor.layers.rocm_skinny_policy": policy_module,
        "aiter": aiter,
        "aiter.ops": aiter_ops,
        "aiter.ops.triton": aiter_triton,
        "aiter.ops.triton.gemm_a16w16": aiter_gemm_module,
    }
    saved_modules = _snapshot_modules(_STUBBED_MODULE_NAMES)
    try:
        sys.modules.update(fake_modules)
        namespace = {
            "torch": SimpleNamespace(
                float16=float16,
                bfloat16=bfloat16,
                nn=SimpleNamespace(
                    functional=SimpleNamespace(linear=stock_linear)
                ),
            ),
            "envs": SimpleNamespace(
                FDU_FORCE_STOCK_GEMM=scenario.force_stock,
                VLLM_ROCM_USE_SKINNY_GEMM=scenario.use_skinny,
            ),
            "ops": SimpleNamespace(
                wvSplitKrc=wvsplitkrc,
                wvSplitK=wvsplitk,
                LLMM1=llmm1,
            ),
            "num_compute_units": num_compute_units,
            "use_aiter_triton_gemm": use_aiter_triton_gemm,
        }
        dispatch = _compile_dispatch(namespace)
        result = dispatch(x, weight, scenario.bias)
        return _DispatchRun(result=result, events=events, x=x, weight=weight)
    finally:
        _restore_modules(saved_modules)


def _event_names(run: _DispatchRun) -> list[object]:
    return [event[0] for event in run.events]


class EnvContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text, cls.tree = _parse(ENVS_PATH)

    def test_stock_gemm_rollback_flag_has_typed_false_default(self) -> None:
        declarations = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "FDU_FORCE_STOCK_GEMM"
        ]
        self.assertEqual(len(declarations), 1)
        declaration = declarations[0]
        self.assertIsInstance(declaration.annotation, ast.Name)
        self.assertEqual(declaration.annotation.id, "bool")
        self.assertIsInstance(declaration.value, ast.Constant)
        self.assertIs(declaration.value.value, False)

    def test_stock_gemm_rollback_flag_registry_contract(self) -> None:
        registries = [
            node
            for node in self.tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "environment_variables"
        ]
        self.assertEqual(len(registries), 1)
        registry = registries[0]
        self.assertIsInstance(registry.value, ast.Dict)
        entries = {
            key.value: value
            for key, value in zip(registry.value.keys, registry.value.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        self.assertTrue(
            "FDU_FORCE_STOCK_GEMM" in entries,
            "missing FDU_FORCE_STOCK_GEMM environment registry entry",
        )
        getter = entries["FDU_FORCE_STOCK_GEMM"]
        expected = _expression(
            'lambda: os.getenv("FDU_FORCE_STOCK_GEMM", "False").lower() '
            'in ("true", "1")'
        )
        self.assertEqual(
            ast.dump(getter, include_attributes=False),
            ast.dump(expected, include_attributes=False),
        )

        compiled_getter = eval(
            compile(
                ast.fix_missing_locations(ast.Expression(getter)),
                ENVS_PATH,
                "eval",
            ),
            {"os": os},
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(compiled_getter())
        for enabled_value in ("true", "1"):
            with self.subTest(enabled_value=enabled_value):
                with mock.patch.dict(
                    os.environ,
                    {"FDU_FORCE_STOCK_GEMM": enabled_value},
                    clear=True,
                ):
                    self.assertTrue(compiled_getter())


class DispatchContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text, cls.tree = _parse(UTILS_PATH)
        cls.function = _function(cls.tree, "rocm_unquantized_gemm_impl")

    def test_dispatch_imports_narrow_rocm_capabilities(self) -> None:
        imports = [
            node
            for node in self.function.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "vllm.platforms.rocm"
        ]
        self.assertEqual(len(imports), 1)
        self.assertEqual(
            {alias.name for alias in imports[0].names},
            {"on_gfx936", "on_gfx9", "on_gfx950", "supports_rocm_skinny_gemm"},
        )

    def test_stock_rollback_precedes_all_custom_dispatch(self) -> None:
        shape_indices = {
            name: _assignment_index(self.function, name) for name in ("n", "m", "k")
        }
        guard_indices = [
            index
            for index, node in enumerate(self.function.body)
            if isinstance(node, ast.If)
            and ast.dump(node.test, include_attributes=False)
            == ast.dump(
                _expression("envs.FDU_FORCE_STOCK_GEMM"),
                include_attributes=False,
            )
        ]
        self.assertEqual(len(guard_indices), 1)
        guard_index = guard_indices[0]
        self.assertEqual(guard_index, max(shape_indices.values()) + 1)
        guard = self.function.body[guard_index]
        self.assertIsInstance(guard, ast.If)
        self.assertEqual(len(guard.body), 1)
        self.assertIsInstance(guard.body[0], ast.Return)
        expected_return = _expression("torch.nn.functional.linear(x, weight, bias)")
        self.assertEqual(
            ast.dump(guard.body[0].value, include_attributes=False),
            ast.dump(expected_return, include_attributes=False),
        )

        later_calls = []
        for name in (
            "num_compute_units",
            "ops.wvSplitKrc",
            "gemm_a16w16",
            "ops.wvSplitK",
            "ops.LLMM1",
        ):
            later_calls.extend(_calls(self.function, name))
        self.assertTrue(later_calls)
        self.assertLess(guard.lineno, min(call.lineno for call in later_calls))

    def test_skinny_base_predicate_is_capability_and_dtype_guarded(self) -> None:
        use_skinny_index = _assignment_index(self.function, "use_skinny")
        assignment = self.function.body[use_skinny_index]
        self.assertIsInstance(assignment, ast.Assign)
        expected = _assignment_value(
            """
use_skinny = (
    envs.VLLM_ROCM_USE_SKINNY_GEMM
    and supports_rocm_skinny_gemm()
    and x.dtype in [torch.float16, torch.bfloat16]
    and weight.dtype == x.dtype
    and k % 8 == 0
)
"""
        )
        self.assertEqual(
            ast.dump(assignment.value, include_attributes=False),
            ast.dump(expected, include_attributes=False),
        )

    def test_gfx936_policy_guards_wvsplitk(self) -> None:
        expected_test = _expression("use_skinny and on_gfx936()")
        gfx936_branches = [
            node
            for node in self.function.body
            if isinstance(node, ast.If)
            and ast.dump(node.test, include_attributes=False)
            == ast.dump(expected_test, include_attributes=False)
        ]
        self.assertEqual(len(gfx936_branches), 1)
        gfx936_branch = gfx936_branches[0]

        policy_imports = [
            node
            for node in gfx936_branch.body
            if isinstance(node, ast.ImportFrom)
            and node.module
            == "vllm.model_executor.layers.rocm_skinny_policy"
        ]
        self.assertEqual(len(policy_imports), 1)
        self.assertEqual(
            [alias.name for alias in policy_imports[0].names],
            ["is_gfx936_skinny_eligible"],
        )
        policy_calls = _calls(gfx936_branch, "is_gfx936_skinny_eligible")
        self.assertEqual(len(policy_calls), 1)
        policy_call = policy_calls[0]
        expected_policy_call = _expression(
            """
is_gfx936_skinny_eligible(
    n=n,
    m=m,
    k=k,
    dtype_name=str(x.dtype).removeprefix("torch."),
    bias_present=bias is not None,
    weight_contiguous=weight.is_contiguous(),
    activation_reshapeable=x.size(-1) == k,
)
"""
        )
        self.assertEqual(
            ast.dump(policy_call, include_attributes=False),
            ast.dump(expected_policy_call, include_attributes=False),
        )

        wvsplitk_calls = _calls(self.function, "ops.wvSplitK")
        self.assertEqual(len(wvsplitk_calls), 1)
        self.assertLess(policy_call.lineno, wvsplitk_calls[0].lineno)

    def test_non_gfx936_skinny_dispatch_remains_on_gfx9(self) -> None:
        expected_test = _expression("use_skinny and on_gfx936()")
        gfx936_branches = [
            node
            for node in self.function.body
            if isinstance(node, ast.If)
            and ast.dump(node.test, include_attributes=False)
            == ast.dump(expected_test, include_attributes=False)
        ]
        self.assertEqual(len(gfx936_branches), 1)
        gfx936_branch = gfx936_branches[0]
        self.assertEqual(len(gfx936_branch.orelse), 1)
        original_arch_branch = gfx936_branch.orelse[0]
        self.assertIsInstance(original_arch_branch, ast.If)
        self.assertEqual(
            ast.dump(original_arch_branch.test, include_attributes=False),
            ast.dump(_expression("use_skinny"), include_attributes=False),
        )
        self.assertEqual(len(original_arch_branch.body), 1)
        expected_assignment = ast.parse("use_skinny = on_gfx9()").body[0]
        self.assertEqual(
            ast.dump(original_arch_branch.body[0], include_attributes=False),
            ast.dump(expected_assignment, include_attributes=False),
        )

    def test_dispatch_does_not_swallow_gpu_exceptions(self) -> None:
        self.assertFalse(
            any(isinstance(node, ast.Try) for node in ast.walk(self.function))
        )


class DispatchSemanticsTest(unittest.TestCase):
    def test_force_stock_short_circuits_all_accelerated_dispatch(self) -> None:
        run = _run_dispatch(
            _DispatchScenario(
                force_stock=True,
                gfx936=True,
                policy_result=True,
                aiter_result=True,
            )
        )

        self.assertEqual(run.result.origin, "stock")
        self.assertEqual(_event_names(run), ["stock"])
        self.assertIs(run.events[0][1], run.x)
        self.assertIs(run.events[0][2], run.weight)
        self.assertIsNone(run.events[0][3])

    def test_gfx936_empty_policy_falls_back_without_custom_ops(self) -> None:
        run = _run_dispatch(
            _DispatchScenario(gfx936=True, policy_result=False)
        )

        self.assertEqual(run.result.origin, "stock")
        self.assertEqual(
            _event_names(run),
            [
                "cu",
                "on_gfx950",
                "aiter_select",
                "supports_skinny",
                "on_gfx936",
                "policy",
                "stock",
            ],
        )
        policy_event = run.events[5]
        self.assertEqual(
            policy_event[1],
            {
                "n": 1,
                "m": 4096,
                "k": 4096,
                "dtype_name": "bfloat16",
                "bias_present": False,
                "weight_contiguous": True,
                "activation_reshapeable": True,
            },
        )
        self.assertNotIn("wvSplitK", _event_names(run))
        self.assertNotIn("LLMM1", _event_names(run))

    def test_gfx936_true_policy_reaches_llmm1_for_measured_n1(self) -> None:
        run = _run_dispatch(
            _DispatchScenario(gfx936=True, policy_result=True, n=1)
        )

        self.assertEqual(run.result.origin, "LLMM1")
        self.assertEqual(
            _event_names(run),
            [
                "cu",
                "on_gfx950",
                "aiter_select",
                "supports_skinny",
                "on_gfx936",
                "policy",
                "LLMM1",
            ],
        )
        self.assertEqual(run.events[6], ("LLMM1", run.weight, run.x, 4))
        self.assertEqual(run.result.reshape_calls, [(1, 4096)])

    def test_unsupported_capability_returns_stock(self) -> None:
        run = _run_dispatch(
            _DispatchScenario(
                supports_skinny=False,
                gfx936=False,
                gfx9=False,
                policy_result=True,
            )
        )

        self.assertEqual(run.result.origin, "stock")
        self.assertEqual(
            _event_names(run),
            [
                "cu",
                "on_gfx950",
                "aiter_select",
                "supports_skinny",
                "stock",
            ],
        )

    def test_legacy_gfx9_path_reaches_existing_wvsplitk(self) -> None:
        run = _run_dispatch(
            _DispatchScenario(gfx936=False, gfx9=True, policy_result=False)
        )

        self.assertEqual(run.result.origin, "wvSplitK")
        self.assertEqual(
            _event_names(run),
            [
                "cu",
                "on_gfx950",
                "aiter_select",
                "supports_skinny",
                "on_gfx936",
                "on_gfx9",
                "cu",
                "wvSplitK",
            ],
        )
        self.assertEqual(
            run.events[7], ("wvSplitK", run.weight, run.x, 120, None)
        )

    def test_aiter_precedes_gfx936_small_batch_dispatch(self) -> None:
        run = _run_dispatch(
            _DispatchScenario(
                gfx936=True,
                policy_result=True,
                aiter_result=True,
            )
        )

        self.assertEqual(run.result.origin, "aiter")
        self.assertEqual(
            _event_names(run),
            ["cu", "on_gfx950", "aiter_select", "aiter"],
        )

    def test_wvsplitkrc_precedes_aiter_dispatch(self) -> None:
        run = _run_dispatch(
            _DispatchScenario(
                gfx950=True,
                aiter_result=True,
                n=16,
                m=64,
                k=1024,
            )
        )

        self.assertEqual(run.result.origin, "wvSplitKrc")
        self.assertEqual(
            _event_names(run), ["cu", "on_gfx950", "wvSplitKrc"]
        )
        self.assertEqual(
            run.events[2], ("wvSplitKrc", run.x, run.weight, 120, None)
        )

    def test_stub_modules_are_restored_after_dispatch_exception(self) -> None:
        before = _snapshot_modules(_STUBBED_MODULE_NAMES)

        with self.assertRaisesRegex(RuntimeError, "expected policy failure"):
            _run_dispatch(
                _DispatchScenario(
                    gfx936=True,
                    policy_result=RuntimeError("expected policy failure"),
                )
            )

        after = _snapshot_modules(_STUBBED_MODULE_NAMES)
        for name in _STUBBED_MODULE_NAMES:
            with self.subTest(module=name):
                self.assertIs(after[name], before[name])


class PlatformIsolationContractTest(unittest.TestCase):
    def test_gfx936_stays_out_of_global_gfx9_and_mi300_predicates(self) -> None:
        text, tree = _parse(ROCM_PATH)
        for assignment_name in ("_ON_GFX9", "_ON_MI3XX"):
            assignments = [
                node
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == assignment_name
                    for target in node.targets
                )
            ]
            self.assertEqual(len(assignments), 1)
            expression = ast.get_source_segment(text, assignments[0].value)
            self.assertIsNotNone(expression)
            self.assertNotIn("gfx936", expression)


if __name__ == "__main__":
    unittest.main()
