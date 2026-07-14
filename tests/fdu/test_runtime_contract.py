from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "launch.sh"
ROCM_ENV = ROOT / "scripts/rocm_env.sh"
SCNET_START = ROOT / "scripts/scnet_start_optimized.sh"
DOCKERFILE = ROOT / "Dockerfile"
HOOKS = ROOT / "fdu_vllm/hooks.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class RuntimeContractTest(unittest.TestCase):
    def test_active_runtime_files_have_no_legacy_quantization_path(self) -> None:
        forbidden = ("awq", "bitsandbytes", "pre_quantize", "/tmp/awq_model")
        for path in (LAUNCH, ROCM_ENV, DOCKERFILE):
            text = _read(path).lower()
            for token in forbidden:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, text)

    def test_architecture_override_and_rocblas_profiling_are_only_unset(self) -> None:
        safety_line = "unset HSA_OVERRIDE_GFX_VERSION ROCBLAS_LAYER"
        launch_text = _read(LAUNCH)
        docker_text = _read(DOCKERFILE)
        rocm_text = _read(ROCM_ENV)

        for name in ("HSA_OVERRIDE_GFX_VERSION", "ROCBLAS_LAYER"):
            with self.subTest(name=name):
                self.assertNotIn(name, launch_text)
                self.assertNotIn(name, docker_text)
                self.assertEqual(rocm_text.count(name), 1)
        self.assertIn(safety_line, rocm_text.splitlines())

    def test_launch_uses_installed_bf16_runtime_and_preflight(self) -> None:
        text = _read(LAUNCH)
        self.assertNotRegex(text, r"(?m)^\s*export\s+PYTHONPATH=")
        self.assertNotIn("$SCRIPT_DIR:${PYTHONPATH", text)
        self.assertRegex(text, r"(?m)^unset\s+PYTHONPATH\s*$")
        self.assertRegex(
            text,
            r'"\$\{?SCRIPT_DIR\}?/scripts/preflight_rocm\.py"',
        )
        self.assertRegex(text, r"--dtype\s+bfloat16\b")
        self.assertNotRegex(text, r"--quantization(?:\s|=)")
        self.assertNotIn("quant_force", text)
        self.assertNotIn("--max-num-seqs", text)
        self.assertNotIn("--max-num-batched-tokens", text)
        self.assertIn("--disable-log-stats", text)
        self.assertRegex(
            text,
            r'if\s+_is_true\s+"\$\{ENABLE_PREFIX_CACHING:-1\}";\s+then'
            r'[\s\S]*VLLM_ARGS\+=\(--enable-prefix-caching\)',
        )

    def test_launch_truth_parser_is_portable_to_bash_3(self) -> None:
        text = _read(LAUNCH)
        self.assertNotRegex(text, r"\$\{[^}]*,,\}")
        self.assertRegex(text, r"1\|\[Tt\]\[Rr\]\[Uu\]\[Ee\]")

    def test_rocm_env_uses_selective_w8_platform_candidate(self) -> None:
        text = _read(ROCM_ENV)
        expected_defaults = {
            "FDU_ENABLE": "0",
            "VLLM_ROCM_USE_AITER": "0",
            "VLLM_ROCM_USE_SKINNY_GEMM": "1",
            "FDU_FORCE_STOCK_GEMM": "0",
            "FDU_GFX936_QUANT_MODE": "w8",
            "ENABLE_PREFIX_CACHING": "1",
        }
        for name, default in expected_defaults.items():
            with self.subTest(name=name):
                self.assertRegex(
                    text,
                    rf'export\s+{name}="\$\{{{name}:-{default}\}}"',
                )

    def test_disabled_plugin_returns_before_optional_runtime_imports(self) -> None:
        text = _read(HOOKS)
        tree = ast.parse(text)
        activate = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "activate"
        )

        self.assertNotIn("quant_force", text)
        config_import = next(
            node
            for node in ast.walk(activate)
            if isinstance(node, ast.ImportFrom)
            and node.module == "fdu_vllm.config"
        )
        config_call = next(
            node
            for node in activate.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "get_config"
        )
        disabled_guard = next(
            node
            for node in activate.body
            if isinstance(node, ast.If)
            and any(isinstance(child, ast.Return) for child in ast.walk(node))
        )
        optional_imports = [
            node
            for node in ast.walk(activate)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and node is not config_import
        ]

        self.assertLess(config_import.lineno, config_call.lineno)
        self.assertLess(config_call.lineno, disabled_guard.lineno)
        self.assertTrue(optional_imports)
        for optional_import in optional_imports:
            with self.subTest(line=optional_import.lineno):
                self.assertGreater(optional_import.lineno, disabled_guard.end_lineno)

    def test_dockerfile_builds_and_installs_submitted_wheel(self) -> None:
        text = _read(DOCKERFILE)
        self.assertRegex(text, r"(?m)^WORKDIR\s+/workspace\s*$")
        self.assertRegex(text, r"(?m)^COPY\s+\.\s+/workspace\s*$")
        self.assertRegex(
            text,
            r"PYTORCH_ROCM_ARCH=gfx936\s+python\s+setup\.py\s+bdist_wheel",
        )
        self.assertRegex(
            text,
            r"python\s+-m\s+pip\s+install\s+"
            r"--no-deps\s+--force-reinstall\s+dist/vllm-\*\.whl",
        )
        self.assertNotIn("vllm.__file__", text)
        self.assertNotIn("FDU hook appended", text)
        self.assertRegex(text, r"(?m)^ENV\s+ENABLE_PREFIX_CACHING=1\s*$")

    def test_scnet_helper_selects_installed_venv_interpreter(self) -> None:
        text = _read(SCNET_START)
        self.assertNotIn("PYTHONPATH", text)
        self.assertIn("VLLM_ENV", text)
        self.assertIn(
            "/public/home/xdzs2026_c415/venvs/vllm_gfx936/bin/python",
            text,
        )
        self.assertRegex(text, r"\[\[\s*!\s+-x\s+\"\$PYTHON_BIN\"\s*\]\]")
        self.assertRegex(text, r'exec\s+bash\s+"\$PROJ/launch\.sh"')


if __name__ == "__main__":
    unittest.main()
