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

    activate()
    from vllm.entrypoints.openai.api_server import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
