from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class NoActiveQuantizationTest(unittest.TestCase):
    def test_active_runtime_files_are_bf16_without_quantization(self) -> None:
        paths = (
            "launch.sh",
            "Dockerfile",
            "scripts/rocm_env.sh",
            "config.yaml",
        )
        combined = "\n".join((ROOT / path).read_text().lower() for path in paths)
        for forbidden in (
            "/tmp/awq_model",
            "--quantization",
            "bitsandbytes",
            "pre_quantize",
            "hsa_override_gfx_version=",
        ):
            self.assertNotIn(forbidden, combined)
        self.assertIn('dtype: "bfloat16"', combined)
        self.assertIn("fdu_enable=0", combined)

    def test_config_defaults_every_optional_hook_off(self) -> None:
        text = (ROOT / "config.yaml").read_text().lower()
        self.assertIn('strategy: "none"', text)
        self.assertIn('backend: "vllm_default"', text)
        self.assertIn("enable_kv_quant: false", text)
        self.assertIn("enable_gqa_opt: false", text)
        self.assertIn("enable_hip_graph: false", text)
        self.assertIn("use_cuda_graph: false", text)

    def test_current_docs_describe_only_native_bf16_path(self) -> None:
        env_doc = (ROOT / "docs/env_vars.md").read_text()
        run_doc = (ROOT / "docs/SCNET_RUN.md").read_text()
        self.assertIn("gfx936", env_doc)
        self.assertIn("BF16", env_doc)
        self.assertIn("FDU_FORCE_STOCK_GEMM", env_doc)
        self.assertIn("dx_branch", run_doc)
        self.assertIn("scnet_ab_gfx936.sh", run_doc)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION=", env_doc)
        self.assertNotIn("pkill -f", run_doc)

    def test_active_docs_describe_selective_w8_and_bf16_rollback(self) -> None:
        readme = (ROOT / "README.md").read_text()
        env_doc = (ROOT / "docs/env_vars.md").read_text()
        run_doc = (ROOT / "docs/SCNET_RUN.md").read_text()
        handoff = (ROOT / "docs/GFX936_HANDOFF.md").read_text()
        report = (ROOT / "report.md").read_text()
        for document in (readme, env_doc, run_doc, handoff, report):
            self.assertIn("FDU_GFX936_QUANT_MODE", document)
        self.assertIn("quant-bench-w8", run_doc)
        self.assertIn("默认", env_doc)
        self.assertIn("`w8`", env_doc)
        self.assertIn("`off`", env_doc)
        self.assertNotIn("src/fdu_vllm/                   # vLLM 插件入口（主要改动区）", readme)

    def test_report_and_changelog_mark_old_routes_historical(self) -> None:
        report = (ROOT / "report.md").read_text()
        changelog = (ROOT / "changelog.md").read_text()
        self.assertIn("当前提交路径（2026-07-14）", report)
        self.assertIn("历史方案（非当前启动路径）", report)
        self.assertIn("v1.3.0-gfx936-online-quant", changelog)
        self.assertIn("历史实验（非当前启动路径）", changelog)

    def test_docs_do_not_overattribute_unpinned_platform_result(self) -> None:
        readme = (ROOT / "README.md").read_text()
        handoff = (ROOT / "docs/GFX936_HANDOFF.md").read_text()
        report = (ROOT / "report.md").read_text()
        for document in (readme, handoff, report):
            self.assertIn("评测提交 hash 尚未随结果记录", document)
        self.assertIn("不能将增益独立归因于 LLMM1", report)


if __name__ == "__main__":
    unittest.main()
