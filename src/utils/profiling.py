"""
Lightweight profiling utilities for inference performance measurement.

Monitors:
- TTFT (Time to First Token)
- TPOT (Time per Output Token) / ITL (Inter-Token Latency)
- Throughput (output tokens/second)
- GPU memory usage
"""

import time
from contextlib import contextmanager
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class RequestMetrics:
    """Per-request timing metrics."""
    seq_id: int
    prompt_len: int
    arrival_time: float
    first_token_time: float = 0.0
    completion_time: float = 0.0
    output_tokens: int = 0
    token_times: List[float] = field(default_factory=list)

    @property
    def ttft(self) -> float:
        """Time to First Token in milliseconds."""
        return (self.first_token_time - self.arrival_time) * 1000

    @property
    def tpot(self) -> float:
        """Time per Output Token in milliseconds."""
        if self.output_tokens < 2:
            return 0.0
        return (
            (self.completion_time - self.first_token_time)
            / (self.output_tokens - 1)
            * 1000
        )

    @property
    def itl_p99(self) -> float:
        """P99 inter-token latency in milliseconds."""
        if len(self.token_times) < 2:
            return 0.0
        sorted_times = sorted(
            self.token_times[i + 1] - self.token_times[i]
            for i in range(len(self.token_times) - 1)
        )
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)] * 1000


class Profiler:
    """
    Collects and aggregates performance metrics across requests.

    Usage:
        profiler = Profiler()
        profiler.on_request_start(seq_id, prompt_len)
        profiler.on_first_token(seq_id)
        profiler.on_token(seq_id)
        profiler.on_request_end(seq_id)
        print(profiler.summary())
    """

    def __init__(self):
        self.requests: Dict[int, RequestMetrics] = {}
        self.start_time: float = time.time()

    def on_request_start(self, seq_id: int, prompt_len: int) -> None:
        self.requests[seq_id] = RequestMetrics(
            seq_id=seq_id,
            prompt_len=prompt_len,
            arrival_time=time.time(),
        )

    def on_first_token(self, seq_id: int) -> None:
        if seq_id in self.requests:
            self.requests[seq_id].first_token_time = time.time()

    def on_token(self, seq_id: int) -> None:
        if seq_id in self.requests:
            req = self.requests[seq_id]
            req.output_tokens += 1
            req.token_times.append(time.time())

    def on_request_end(self, seq_id: int) -> None:
        if seq_id in self.requests:
            self.requests[seq_id].completion_time = time.time()

    def summary(self) -> dict:
        """Compute aggregate metrics."""
        completed = [r for r in self.requests.values()
                     if r.completion_time > 0 and r.first_token_time > 0]
        if not completed:
            return {"error": "No completed requests"}

        # TTFT
        ttfts = sorted([r.ttft for r in completed])
        ttft_p50 = ttfts[len(ttfts) // 2]
        ttft_p99 = ttfts[int(len(ttfts) * 0.99)]

        # TPOT
        tpots = sorted([r.tpot for r in completed if r.tpot > 0])
        tpot_p50 = tpots[len(tpots) // 2] if tpots else 0.0
        tpot_p99 = tpots[int(len(tpots) * 0.99)] if tpots else 0.0

        # Throughput
        elapsed = time.time() - self.start_time
        total_output_tokens = sum(r.output_tokens for r in completed)
        throughput = total_output_tokens / elapsed if elapsed > 0 else 0.0

        # SLA checks
        sla_ttft_ok = ttft_p99 <= 2000  # <= 2000ms
        sla_tpot_ok = tpot_p99 <= 100   # <= 100ms

        return {
            "ttft_p50_ms": round(ttft_p50, 2),
            "ttft_p99_ms": round(ttft_p99, 2),
            "tpot_p50_ms": round(tpot_p50, 2),
            "tpot_p99_ms": round(tpot_p99, 2),
            "throughput_tok_s": round(throughput, 2),
            "total_output_tokens": total_output_tokens,
            "num_requests": len(completed),
            "elapsed_sec": round(elapsed, 2),
            "sla_ttft": "PASS" if sla_ttft_ok else "FAIL",
            "sla_tpot": "PASS" if sla_tpot_ok else "FAIL",
        }


@contextmanager
def Timer(name: str = "operation"):
    """Context manager for timing code blocks."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[Timer] {name}: {elapsed * 1000:.2f} ms")
