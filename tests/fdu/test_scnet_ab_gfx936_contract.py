from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/scnet_ab_gfx936.sh"


class ScnetAbGfx936FastPathContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_has_all_nine_fast_path_commands(self) -> None:
        for command in (
            "quant-bench-w8",
            "quant-bench-hybrid",
            "sync-candidate-python",
            "start-candidate-off",
            "start-candidate-w8",
            "start-candidate-hybrid",
            "probe-candidate-off",
            "probe-candidate-w8",
            "probe-candidate-hybrid",
        ):
            self.assertIn(f"{command})", self.text)

    def test_start_server_uses_submission_entrypoint_and_distinct_mode_logs(self) -> None:
        self.assertIn('exec bash "$SOURCE_ROOT/launch.sh"', self.text)
        self.assertIn('FDU_GFX936_QUANT_MODE="$quant_mode"', self.text)
        for mode in ("off", "w8", "hybrid_w4"):
            self.assertIn(f" 0 {mode} ;;", self.text)
            self.assertIn(f"/tmp/fdu_gfx936_{mode}.log", self.text)
        self.assertIn('echo "[gfx936:start] quant_mode=$quant_mode"', self.text)

    def test_sync_overlays_only_required_candidate_python_files(self) -> None:
        self.assertIn("import pathlib,vllm", self.text)
        for relative in (
            "envs.py",
            "model_executor/layers/gfx936_online_quant.py",
            "model_executor/layers/linear.py",
            "model_executor/layers/utils.py",
            "model_executor/model_loader/utils.py",
            "model_executor/layers/rocm_skinny_policy.py",
            "model_executor/layers/rocm_skinny_shapes.py",
        ):
            self.assertIn(relative, self.text)
        self.assertIn("py_compile", self.text)
        self.assertIn("vllm._custom_ops", self.text)
        self.assertIn("LLMM1", self.text)
        self.assertIn("run build-candidate once", self.text)

    def test_quant_bench_preflights_and_uses_exact_settings_and_outputs(self) -> None:
        self.assertIn("build_gfx936_quant_jit.py", self.text)
        self.assertIn("preflight_gfx936_quant.py", self.text)
        self.assertIn("--smoke", self.text)
        self.assertIn("bench_gfx936_quant.py", self.text)
        self.assertIn("--warmup 2", self.text)
        self.assertIn("--repetitions 8", self.text)
        for mode in ("w8", "hybrid_w4"):
            self.assertIn(f"/tmp/fdu_gfx936_quant_{mode}.json", self.text)
            self.assertIn(f"/tmp/fdu_gfx936_quant_{mode}.log", self.text)

    def test_mode_probes_check_health_response_and_log_failures(self) -> None:
        self.assertIn('http://127.0.0.1:$PORT/health', self.text)
        self.assertIn('"max_tokens": 16', self.text)
        self.assertIn("choices", self.text)
        self.assertIn("quant_mode=$quant_mode", self.text)
        for marker in ("Traceback", "non-finite admission", "OOM", "out of memory"):
            self.assertIn(marker, self.text)
        self.assertIn('$RESULTS_ROOT/probes/', self.text)

    def test_preserves_every_legacy_command(self) -> None:
        for command in (
            "init",
            "build-control",
            "bench",
            "build-candidate",
            "start-control",
            "start-candidate-stock",
            "start-candidate",
            "stop",
            "probe",
            "throughput",
            "accuracy",
        ):
            self.assertIn(f"{command})", self.text)

    def test_shell_parses(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
