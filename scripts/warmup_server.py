#!/usr/bin/env python3
"""
分档 warmup — 对齐官方三档上下文，稳定 TTFT P99。

策略（最有把握提分）：
  - tier=all 时 **先跑 8-16K（50% 权重）**，再 4-8K / 16-32K
  - 以 prefill 为主、decode 短输出（触发 JIT + 显存池，不拖垮启动）
  - chat 失败时回退 /v1/completions

用法:
  python scripts/warmup_server.py --host 127.0.0.1 --port 8000
  python scripts/warmup_server.py --tier 8-16K --rounds 1
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# (name, prefill_chars≈tokens, max_tokens)
# all 顺序：主攻档优先
_PROFILES = [
    ("8-16K", 12000, 64),   # 50% 权重，优先预热
    ("4-8K", 6000, 32),
    ("16-32K", 16000, 64),   # 略低于 20k，降低 0.94 显存下 warmup OOM 风险
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


def _post_json(url: str, payload: dict, timeout: int = 1800) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()


def _request(host: str, port: int, prompt: str, max_tokens: int) -> None:
    base = f"http://{host}:{port}"
    chat_payload = {
        "model": "Qwen3.5-27B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        _post_json(f"{base}/v1/chat/completions", chat_payload)
        return
    except urllib.error.URLError:
        pass

    # 回退 completions（部分评测/自测脚本走此路径）
    _post_json(
        f"{base}/v1/completions",
        {
            "model": "Qwen3.5-27B",
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        },
    )


def warmup(host: str, port: int, rounds: int, tier: str) -> None:
    profiles = _PROFILES
    if tier != "all":
        profiles = [p for p in _PROFILES if p[0] == tier]
        if not profiles:
            print(
                f"[warmup] unknown tier {tier}, use all|4-8K|8-16K|16-32K",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"[warmup] host={host}:{port} rounds={rounds} tier={tier}")
    print(f"[warmup] order: {' → '.join(p[0] for p in profiles)}")

    for r in range(rounds):
        for name, chars, max_tok in profiles:
            label = f"round {r + 1}/{rounds} tier {name}"
            try:
                _request(host, port, _prompt_chars(chars), max_tok)
                print(
                    f"[warmup] {label} ok "
                    f"(prefill~{chars} chars, max_tokens={max_tok})"
                )
            except urllib.error.URLError as e:
                print(f"[warmup] {label} failed: {e}", file=sys.stderr)
                sys.exit(1)

    print("[warmup] all tiers done")


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 1 tiered warmup for TTFT P99")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument(
        "--tier",
        default="all",
        choices=["all", "4-8K", "8-16K", "16-32K"],
    )
    args = p.parse_args()
    warmup(args.host, args.port, args.rounds, args.tier)


if __name__ == "__main__":
    main()
