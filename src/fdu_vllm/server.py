"""vLLM OpenAI API server entry with FDU plugin activation."""

import sys
from pathlib import Path


def _setup_plugin_path() -> None:
    """Ensure fdu_vllm loads from src/, never repo-root fdu_vllm/."""
    plugin_dir = Path(__file__).resolve().parent
    src_dir = plugin_dir.parent
    if src_dir.name != "src":
        nested = src_dir / "src"
        if (nested / "fdu_vllm").is_dir():
            src_dir = nested
    src = str(src_dir)
    if src not in sys.path:
        sys.path.insert(0, src)


_setup_plugin_path()


def main() -> None:
    import uvloop
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.entrypoints.utils import cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from fdu_vllm.vllm_env import configure_before_vllm_import

    configure_before_vllm_import()

    from fdu_vllm import activate
    from fdu_vllm.phase1 import validate_phase1_env

    activate()
    for warning in validate_phase1_env():
        print(f"[fdu_vllm] WARNING: {warning}", file=sys.stderr)

    # vLLM 0.18+：api_server 无 main()，与 `vllm serve` / __main__ 块对齐
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
