from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/scnet_ab_gfx936.sh"


class ScnetAbContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text()

    def test_uses_fixed_persistent_isolation_paths(self) -> None:
        for value in (
            "/public/home/xdzs2026_c415/experiments/gfx936_skinny",
            "/public/home/xdzs2026_c415/venvs/vllm_baseline",
            "/public/home/xdzs2026_c415/venvs/vllm_gfx936",
            "/public/home/xdzs2026_c415/results/gfx936_skinny",
        ):
            self.assertIn(value, self.text)

    def test_builds_native_wheels_without_ensurepip(self) -> None:
        self.assertIn(
            '"$SYSTEM_PYTHON" -m venv --without-pip --system-site-packages',
            self.text,
        )
        self.assertIn('import pip, torch', self.text)
        self.assertIn('[gfx936:init] creating control venv', self.text)
        self.assertIn('[gfx936:init] creating candidate venv', self.text)

    def test_builds_native_wheels_for_gfx936(self) -> None:
        self.assertIn("PYTORCH_ROCM_ARCH=gfx936", self.text)
        self.assertIn("sha256sum", self.text)

    def test_benchmark_runs_unbuffered_with_progress(self) -> None:
        self.assertIn('"$CONTROL_VENV/bin/python" -u', self.text)
        benchmark = (ROOT / "scripts/bench_gfx936_skinny.py").read_text()
        self.assertIn("[gfx936:bench]", benchmark)

    def test_has_every_pipeline_mode(self) -> None:
        for mode in (
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
            self.assertIn(f"{mode})", self.text)

    def test_stopping_is_pid_specific(self) -> None:
        self.assertNotIn("pkill", self.text)
        self.assertNotIn("killall", self.text)
        self.assertIn('kill -TERM "$pid"', self.text)
        self.assertIn('kill -KILL "$pid"', self.text)

    def test_eval_uses_scratch_copy_not_testdata_working_directory(self) -> None:
        self.assertIn('rsync -a --exclude test', self.text)
        self.assertNotIn('cd "$TESTDATA_ROOT"', self.text)
        self.assertIn('$RESULTS_ROOT/eval_work/', self.text)

    def test_server_modes_lock_bf16_and_safe_runtime_switches(self) -> None:
        self.assertIn("FDU_ENABLE=0", self.text)
        self.assertIn("VLLM_ROCM_USE_AITER=0", self.text)
        self.assertIn("VLLM_ROCM_USE_SKINNY_GEMM=1", self.text)
        self.assertIn("--dtype bfloat16", self.text)
        self.assertIn("--require-arch gfx936", self.text)
        self.assertIn("unset HSA_OVERRIDE_GFX_VERSION ROCBLAS_LAYER PYTHONPATH", self.text)


if __name__ == "__main__":
    unittest.main()
