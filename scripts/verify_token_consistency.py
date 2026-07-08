#!/usr/bin/env python3
"""Phase 4: temperature=0 下对比 baseline 与优化版 token 一致性（抽样）。"""
import argparse
import json
import sys
import urllib.request


def complete(host: str, port: int, prompt: str, model: str) -> str:
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 64,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-host", default="127.0.0.1")
    p.add_argument("--baseline-port", type=int, default=8000)
    p.add_argument("--opt-host", default="127.0.0.1")
    p.add_argument("--opt-port", type=int, default=8001)
    p.add_argument("--model", default="Qwen3.5-27B")
    p.add_argument("--prompt", default="用一句话介绍复旦大学。")
    args = p.parse_args()

    b = complete(args.baseline_host, args.baseline_port, args.prompt, args.model)
    o = complete(args.opt_host, args.opt_port, args.prompt, args.model)

    if b == o:
        print("[verify] Token output match (exact string)")
        return 0
    print("[verify] MISMATCH", file=sys.stderr)
    print("baseline:", b[:200], file=sys.stderr)
    print("optimized:", o[:200], file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
