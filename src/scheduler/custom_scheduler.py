"""
Custom Scheduler for vLLM with length-aware batching.

DEPRECATED — 赛题禁止修改 vLLM batch scheduler；初赛并发=1，此模块不启用。
保留仅供架构参考，launch.sh / fdu_vllm 不会加载本模块。

原设计目标（未启用）：
1. Length-aware batching
2. Prefill-decode decoupling
3. Priority inversion prevention
4. DCU occupancy optimization
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Deque
import heapq


@dataclass(order=True)
class ScheduledRequest:
    """Request with scheduling metadata."""
    priority: int  # Lower value = higher priority
    seq_id: int = field(compare=False)
    prompt_len: int = field(compare=False)
    arrival_time: float = field(compare=False)
    is_decode: bool = field(compare=False)


class CustomScheduler:
    """
    Length-aware scheduler that balances throughput and latency.

    Policy: "length_aware"
    - Short requests (< 1K tokens): high priority, batched aggressively
    - Medium requests (1K-8K): normal priority
    - Long requests (> 8K): limited concurrency to avoid HBM thrashing
    """

    # DCU compute unit limits
    MAX_WAVEFRONTS_PER_CU = 8

    def __init__(
        self,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 8192,
        policy: str = "length_aware",
    ):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.policy = policy

        # Separate queues
        self.prefill_queue: Deque[ScheduledRequest] = deque()
        self.decode_queue: Deque[ScheduledRequest] = deque()

    def add_request(self, req: ScheduledRequest) -> None:
        """Add a request to the appropriate queue."""
        if req.is_decode:
            self.decode_queue.append(req)
        else:
            self.prefill_queue.append(req)

    def schedule(self) -> List[ScheduledRequest]:
        """
        Select the next batch of requests to execute.

        Strategy:
        1. Prioritize decode requests (low latency for ongoing generations)
        2. Fill remaining budget with prefill requests
        3. Apply length-aware constraints
        """
        if self.policy == "length_aware":
            return self._schedule_length_aware()
        elif self.policy == "fcfs":
            return self._schedule_fcfs()
        else:
            return self._schedule_length_aware()

    def _schedule_length_aware(self) -> List[ScheduledRequest]:
        """Length-aware batching: prevent long requests from starving short ones."""
        batch: List[ScheduledRequest] = []
        batch_tokens = 0

        # Phase 1: Schedule decode requests first (low latency priority)
        while self.decode_queue and len(batch) < self.max_num_seqs:
            req = self.decode_queue.popleft()
            batch.append(req)
            batch_tokens += 1  # Decode: 1 token per step

        # Phase 2: Fill with prefill requests, length-ordered
        # Sort prefill queue by prompt_len ascending (short first)
        sorted_prefill = sorted(self.prefill_queue, key=lambda r: r.prompt_len)
        for req in sorted_prefill:
            if len(batch) >= self.max_num_seqs:
                break
            if batch_tokens + req.prompt_len > self.max_num_batched_tokens:
                # Skip this long request; try next shorter one
                continue
            batch.append(req)
            batch_tokens += req.prompt_len
            self.prefill_queue.remove(req)

        return batch

    def _schedule_fcfs(self) -> List[ScheduledRequest]:
        """Simple first-come-first-served scheduling."""
        batch = []
        batch_tokens = 0
        combined = list(self.decode_queue) + list(self.prefill_queue)

        for req in combined:
            if len(batch) >= self.max_num_seqs:
                break
            tokens = 1 if req.is_decode else req.prompt_len
            if batch_tokens + tokens <= self.max_num_batched_tokens:
                batch.append(req)
                batch_tokens += tokens

        return batch

    @property
    def queue_depth(self) -> dict:
        return {
            "prefill": len(self.prefill_queue),
            "decode": len(self.decode_queue),
        }
