#!/usr/bin/env python3
"""Run a deterministic sequential smoke probe against an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence
import urllib.request


PROMPTS = (
    "用一句话介绍复旦大学。",
    "计算 37*19，只输出整数。",
    "Return the word BLUE exactly.",
)


def request_completion(
    *, host: str, port: int, model: str, prompt: str
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "seed": 20260714,
        "max_tokens": 64,
        "stream": False,
    }
    request = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def run_probe(*, host: str, port: int, model: str, label: str) -> dict[str, Any]:
    responses = []
    finish_reasons = []
    usage = []
    for prompt in PROMPTS:
        response = request_completion(host=host, port=port, model=model, prompt=prompt)
        choice = response["choices"][0]
        responses.append(choice["message"]["content"])
        finish_reasons.append(choice.get("finish_reason"))
        usage.append(response.get("usage", {}))
    return {
        "label": label,
        "model": model,
        "prompts": list(PROMPTS),
        "responses": responses,
        "finish_reasons": finish_reasons,
        "usage": usage,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default="Qwen3.5-27B")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_probe(
        host=args.host,
        port=args.port,
        model=args.model,
        label=args.label,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
