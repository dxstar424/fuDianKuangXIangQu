"""vLLM OpenAI API server entry with FDU plugin activation."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def main() -> None:
    from fdu_vllm.vllm_env import configure_before_vllm_import

    configure_before_vllm_import()

    from fdu_vllm import activate
    from fdu_vllm.phase1 import validate_phase1_env

    activate()
    for warning in validate_phase1_env():
        print(f"[fdu_vllm] WARNING: {warning}", file=sys.stderr)

    from vllm.entrypoints.openai.api_server import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
