#!/usr/bin/env python3
"""
分档 warmup — 对齐官方三档上下文，稳定 TTFT P99（尤其 8-16K 50% 权重档）。

用法:
  python scripts/warmup_server.py --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# 近似 token 规模：中文约 1 字 ≈ 1 token
_PROFILES = [
    ("4-8K", 6000, 128),    # prefill ~6k, decode 128
    ("8-16K", 12000, 256),  # 主攻档：prefill ~12k
    ("16-32K", 20000, 512), # 长档 prefill
]


def _prompt_chars(n: int) -> str:
    base = (
        "请根据以下材料简要回答问题。材料内容涵盖人工智能、系统优化与长文本推理。"
        "我们需要验证服务在长上下文下的首字时延稳定性。"
    )
    if n <= len(base):
        return base[:n]
    reps = (n // len(base)) + 1
    return (base * reps)[:n]


def _request(host: str, port: int, prompt: str, max_tokens: int) -> None:
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": "Qwen3.5-27B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        resp.read()


def warmup(host: str, port: int, rounds: int, tier: str) -> None:
    profiles = _PROFILES
    if tier != "all":
        profiles = [p for p in _PROFILES if p[0] == tier]
        if not profiles:
            print(f"[warmup] unknown tier {tier}, use all|4-8K|8-16K|16-32K", file=sys.stderr)
            sys.exit(1)

    for r in range(rounds):
        for name, chars, max_tok in profiles:
            label = f"round {r + 1}/{rounds} tier {name}"
            try:
                _request(host, port, _prompt_chars(chars), max_tok)
                print(f"[warmup] {label} ok (prefill~{chars} chars, max_tokens={max_tok})")
            except urllib.error.URLError as e:
                print(f"[warmup] {label} failed: {e}", file=sys.stderr)
                sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--tier", default="all", choices=["all", "4-8K", "8-16K", "16-32K"])
    args = p.parse_args()
    warmup(args.host, args.port, args.rounds, args.tier)


if __name__ == "__main__":
    main()
