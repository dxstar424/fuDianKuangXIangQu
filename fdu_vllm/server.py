"""Shim: delegate to src/fdu_vllm/server.py (avoid repo-root path shadowing)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_src_server = Path(__file__).resolve().parents[1] / "src" / "fdu_vllm" / "server.py"
if not _src_server.is_file():
    raise ImportError(f"missing plugin entry: {_src_server}")

sys.modules.pop(__name__, None)
runpy.run_path(str(_src_server), run_name="__main__")
