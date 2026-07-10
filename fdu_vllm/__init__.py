"""FDU vLLM optimization plugin — hooks into vLLM without batch scheduler changes."""

from fdu_vllm.config import FduConfig
from fdu_vllm.hooks import activate, is_active

__all__ = ["activate", "is_active", "FduConfig"]
