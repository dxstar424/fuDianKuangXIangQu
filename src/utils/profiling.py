"""
DCU 性能分析工具集
==================
提供两类监控：
  A. 软件层 (Python)：TTFT, TPOT, 吞吐量，SLA 判定
  B. 硬件层 (rocprof/rocm-smi)：HBM 带宽, 显存用量, 温度, 功耗

注意：硬件层工具仅在 DCU 实机上可用，本地开发时自动降级为空操作。
"""

import json
import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ============================================================
# A. 软件层 Profiler（跨平台）
# ============================================================

@dataclass
class RequestMetrics:
    """单次请求的时序度量"""
    seq_id: int
    prompt_len: int
    arrival_time: float
    first_token_time: float = 0.0
    completion_time: float = 0.0
    output_tokens: int = 0
    token_times: List[float] = field(default_factory=list)

    @property
    def ttft(self) -> float:
        return (self.first_token_time - self.arrival_time) * 1000

    @property
    def tpot(self) -> float:
        if self.output_tokens < 2:
            return 0.0
        return (self.completion_time - self.first_token_time) / (self.output_tokens - 1) * 1000


class RequestProfiler:
    """跨请求的软件层指标收集器（无外部依赖）"""

    def __init__(self):
        self.requests: Dict[int, RequestMetrics] = {}
        self.start_time = time.time()

    def on_start(self, seq_id: int, prompt_len: int) -> None:
        self.requests[seq_id] = RequestMetrics(seq_id, prompt_len, time.time())

    def on_first_token(self, seq_id: int) -> None:
        if seq_id in self.requests:
            self.requests[seq_id].first_token_time = time.time()

    def on_token(self, seq_id: int) -> None:
        if seq_id in self.requests:
            r = self.requests[seq_id]
            r.output_tokens += 1
            r.token_times.append(time.time())

    def on_end(self, seq_id: int) -> None:
        if seq_id in self.requests:
            self.requests[seq_id].completion_time = time.time()

    def summary(self) -> dict:
        completed = [r for r in self.requests.values() if r.completion_time > 0]
        if not completed:
            return {"error": "No completed requests"}

        ttft = sorted([r.ttft for r in completed])
        tpot = sorted([r.tpot for r in completed if r.tpot > 0])
        elapsed = time.time() - self.start_time
        total_out = sum(r.output_tokens for r in completed)

        def p(data, pct):
            i = min(int(len(data) * pct / 100), len(data) - 1)
            return data[i] if data else 0

        return {
            "ttft_p50_ms": round(p(ttft, 50), 2),
            "ttft_p99_ms": round(p(ttft, 99), 2),
            "tpot_p50_ms": round(p(tpot, 50), 2),
            "tpot_p99_ms": round(p(tpot, 99), 2),
            "throughput_tok_s": round(total_out / elapsed, 2) if elapsed > 0 else 0,
            "num_requests": len(completed),
            "elapsed_sec": round(elapsed, 2),
        }


@contextmanager
def Timer(name: str = "op"):
    start = time.time()
    yield
    print(f"[Timer] {name}: {(time.time() - start) * 1000:.1f} ms")


# ============================================================
# B. DCU 硬件层 Profiler（rocprof / rocm-smi）
# ============================================================

class DCUHardwareProfiler:
    """
    DCU 硬件性能监控。

    封装 rocm-smi（显存/温度/功耗）和 rocprof（kernel 级 profiling），
    本地无 DCU 时自动降级为空操作。
    """

    def __init__(self):
        self._rocm_smi_available = self._check_cmd("rocm-smi --showmeminfo vram 2>/dev/null")
        self._rocprof_available = self._check_cmd("rocprof --help 2>/dev/null")

    @staticmethod
    def _check_cmd(cmd: str) -> bool:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    @property
    def available(self) -> bool:
        return self._rocm_smi_available

    # ---- rocm-smi: 显存 + 功耗 ----

    def get_memory_info(self) -> dict:
        """获取 DCU 显存使用情况（VRAM used / total）"""
        if not self._rocm_smi_available:
            return {"available": False, "note": "rocm-smi not found"}

        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=10
            )
            data = json.loads(result.stdout)
            # rocm-smi JSON 结构因版本而异，此处做通用解析
            gpu_info = {}
            for card_id, card_data in data.items():
                vram = card_data.get("VRAM", {})
                gpu_info[card_id] = {
                    "total_mb": vram.get("Total Memory (MB)", 0),
                    "used_mb": vram.get("Used Memory (MB)", 0),
                }
            return {"available": True, "gpus": gpu_info}
        except Exception as e:
            return {"available": False, "error": str(e)}

    def get_power_temp(self) -> dict:
        """获取 DCU 功耗和温度"""
        if not self._rocm_smi_available:
            return {"available": False}

        try:
            result = subprocess.run(
                ["rocm-smi", "--showpower", "--showtemp", "--json"],
                capture_output=True, text=True, timeout=10
            )
            return {"available": True, "data": json.loads(result.stdout)}
        except Exception:
            return {"available": False}

    # ---- rocprof: kernel 级性能分析 ----

    def profile_kernel(
        self, output_file: str, duration_sec: int = 30
    ) -> Optional[str]:
        """
        使用 rocprof 采集 kernel 级性能数据。

        启动 rocprof 计数器收集（HBM 带宽、L2 cache hit rate、
        wavefront occupancy 等），在指定时长后停止并输出 CSV。

        Args:
            output_file: 输出 CSV 文件路径
            duration_sec: 采集时长（秒）

        Returns:
            输出文件路径，失败返回 None
        """
        if not self._rocprof_available:
            print("[DCUProfiler] rocprof not available, skipping kernel profiling.")
            return None

        metrics = [
            "FETCH_SIZE",              # HBM 读取量
            "WRITE_SIZE",             # HBM 写入量
            "SQ_WAVES",               # wavefront 启动数
            "VALUUtilization",        # 向量 ALU 利用率
            "MemUnitBusy",            # 内存单元忙碌比例
            "L2CacheHit",             # L2 cache 命中
        ]

        cmd = [
            "rocprof",
            "--metrics", ",".join(metrics),
            "-o", output_file,
            "--timestamp", "on",
            "--basenames", "on",
        ]

        try:
            proc = subprocess.Popen(cmd)
            time.sleep(duration_sec)
            proc.terminate()
            proc.wait(timeout=10)
            return output_file
        except Exception as e:
            print(f"[DCUProfiler] rocprof failed: {e}")
            return None

    # ---- 综合快照 ----

    def snapshot(self) -> dict:
        """获取 DCU 当前状态快照（用于日志记录）"""
        return {
            "memory": self.get_memory_info(),
            "power_temp": self.get_power_temp(),
            "rocprof_available": self._rocprof_available,
        }
