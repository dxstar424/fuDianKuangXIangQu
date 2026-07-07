#!/usr/bin/env python3
"""
Baseline vs Optimized 对比工具
===============================
读取两次 benchmark 的 JSON 输出，生成对比报告。

用法:
    python scripts/compare.py results/baseline_xxx.json results/optimized_xxx.json
"""

import json
import sys
from pathlib import Path


def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compare(baseline_path: str, optimized_path: str) -> str:
    baseline = load_report(baseline_path)
    optimized = load_report(optimized_path)

    lines = []
    lines.append("=" * 70)
    lines.append("  Baseline vs Optimized 对比报告")
    lines.append("=" * 70)

    # 按 tier 对比
    b_tiers = {t["name"]: t for t in baseline["tiers"]}
    o_tiers = {t["name"]: t for t in optimized["tiers"]}

    all_gains = []

    for name in ["short", "medium", "long"]:
        b = b_tiers.get(name)
        o = o_tiers.get(name)
        if not b or not o:
            continue

        ttft_change = ((o["ttft_p99_ms"] - b["ttft_p99_ms"]) / b["ttft_p99_ms"] * 100) if b["ttft_p99_ms"] else 0
        tpot_change = ((o["tpot_p99_ms"] - b["tpot_p99_ms"]) / b["tpot_p99_ms"] * 100) if b["tpot_p99_ms"] else 0
        throughput_change = ((o["throughput_tok_s"] - b["throughput_tok_s"]) / b["throughput_tok_s"] * 100) if b["throughput_tok_s"] else 0

        b_sla = "✅" if b["sla_pass"] else "❌"
        o_sla = "✅" if o["sla_pass"] else "❌"

        lines.append(f"\n--- {name.upper()} (weight={b.get('weight', '?')}) ---")
        lines.append(f"  {'Metric':<20} {'Baseline':>12} {'Optimized':>12} {'Change':>10}")
        lines.append(f"  {'-'*54}")
        lines.append(f"  {'TTFT P99 (ms)':<20} {b['ttft_p99_ms']:>12.1f} {o['ttft_p99_ms']:>12.1f} {ttft_change:>+9.1f}%")
        lines.append(f"  {'TPOT P99 (ms)':<20} {b['tpot_p99_ms']:>12.1f} {o['tpot_p99_ms']:>12.1f} {tpot_change:>+9.1f}%")
        lines.append(f"  {'Throughput (tok/s)':<20} {b['throughput_tok_s']:>12.1f} {o['throughput_tok_s']:>12.1f} {throughput_change:>+9.1f}%")
        lines.append(f"  {'SLA':<20} {b_sla:>12} {o_sla:>12}")

        all_gains.append(throughput_change)

    # 总分对比
    lines.append(f"\n{'='*70}")
    lines.append(f"  Total Score: {baseline.get('total_score', 0):.1f} → {optimized.get('total_score', 0):.1f}")
    if all_gains:
        avg_gain = sum(all_gains) / len(all_gains)
        lines.append(f"  Average Throughput Improvement: {avg_gain:+.1f}%")
    lines.append("=" * 70)

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/compare.py <baseline.json> <optimized.json>")
        sys.exit(1)
    print(compare(sys.argv[1], sys.argv[2]))
