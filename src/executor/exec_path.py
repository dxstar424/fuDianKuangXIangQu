"""
Execution Path Optimizer for vLLM on DCU (ROCm/HIP).

DCU 上的 API 映射（PyTorch ROCm 兼容层）：
  torch.cuda.CUDAGraph   → 底层 hipGraphCreate / hipGraphLaunch
  torch.cuda.graph()     → 底层 hipGraphBeginCapture / hipGraphEndCapture
  torch.cuda.empty_cache() → hipDeviceSynchronize + 显存缓存刷新
  tensor.is_cuda          → ROCm 设备上返回 True

注意：PyTorch ROCm 刻意保留 torch.cuda.* 命名空间以保持 API 兼容，
没有 torch.hip 模块。以下代码中的 torch.cuda.* 在 DCU 上运行时会
自动映射到 HIP 底层调用。

Optimization strategies:
1. HIP Graph capture for decode (reduce kernel launch overhead)
2. Fused kernel launches (reduce driver dispatch latency)
3. Async HBM ↔ host transfers (overlap compute and data movement)
4. Warmup & persistent kernel cache
"""

import torch
from typing import Optional, Dict


class ExecPathOptimizer:
    """
    优化模型执行路径，降低时延、提高吞吐。

    DCU 关键优化：
    - HIP Graph 捕获：对 Decode 阶段固定 batch size 路径进行图捕获，
      消除逐 step 的 kernel launch 开销。在 DCU 上通过 ROCm 的
      hipGraph API 实现（由 PyTorch torch.cuda.CUDAGraph 封装）。
    - 调度批量化：每 N 步调度一次，减少 Python 层开销。
    - 异步 HBM↔Host 传输：利用 DCU 异步 DMA 引擎 (SDMA) 重叠数据搬运。
    """

    def __init__(
        self,
        max_batch_size: int = 16,
        enable_hip_graph: bool = True,     # 在 DCU 上等同于 HIP Graph
        num_steps_per_schedule: int = 8,
    ):
        self.max_batch_size = max_batch_size
        self.enable_hip_graph = enable_hip_graph
        self.num_steps_per_schedule = num_steps_per_schedule

        # HIP Graph 缓存（PyTorch torch.cuda.CUDAGraph 在 ROCm 上映射为 HIP Graph）
        # batch_size → captured graph
        self._graphs: Dict[int, torch.cuda.CUDAGraph] = {}

        self._warmed_up = False

    # ---- ROCm/HIP 设备检测 ----

    @staticmethod
    def is_rocm() -> bool:
        """检测当前是否运行在 ROCm/DCU 环境"""
        if not torch.cuda.is_available():
            return False
        try:
            # torch.version.hip 仅在 ROCm 构建时存在
            return hasattr(torch.version, 'hip') and torch.version.hip is not None
        except Exception:
            return False

    @staticmethod
    def get_device_name() -> str:
        """获取当前加速卡名称（DCU 型号或 NVIDIA GPU 型号）"""
        if not torch.cuda.is_available():
            return "CPU"
        try:
            props = torch.cuda.get_device_properties(0)
            return props.name
        except Exception:
            return "Unknown"

    # ---- HIP Graph 捕获与回放 ----

    def capture_graph(
        self,
        batch_size: int,
        model_forward_fn,
        sample_inputs: tuple,
    ) -> None:
        """
        捕获 HIP Graph（通过 PyTorch torch.cuda.CUDAGraph 封装）。

        DCU 底层流程：
          1. hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal)
          2. 执行 model_forward_fn → 所有 kernel 被记录
          3. hipStreamEndCapture(stream, &graph)
          4. hipGraphInstantiate(&instance, graph, ...)

        PyTorch 将以上封装为 torch.cuda.graph() 上下文管理器。

        Args:
            batch_size: 要捕获的 batch size
            model_forward_fn: 模型前向函数
            sample_inputs: 代表性输入（用于捕获）
        """
        if not self.enable_hip_graph:
            return

        # Warmup: 填充 ROCm kernel 缓存（避免捕获未优化的 kernel）
        for _ in range(3):
            model_forward_fn(*sample_inputs)

        # 开始 HIP Graph 捕获
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):      # ← 底层: hipStreamBeginCapture / hipStreamEndCapture
            model_forward_fn(*sample_inputs)

        self._graphs[batch_size] = graph

    def replay(self, batch_size: int) -> None:
        """
        回放已捕获的 HIP Graph。

        底层: hipGraphLaunch(instance, stream)
        """
        if batch_size in self._graphs:
            self._graphs[batch_size].replay()

    # ---- 调度批量化 ----

    def optimize_kernel_launch(self, num_scheduled_requests: int) -> int:
        """
        确定最优调度批量步数。

        更少的 kernel launch → 更低的 dispatch 延迟，
        但过长会降低调度灵活性。按请求数自适应调整。

        Args:
            num_scheduled_requests: 当前活跃请求数

        Returns:
            推荐调度步数
        """
        if num_scheduled_requests <= 4:
            return min(self.num_steps_per_schedule, 2)
        elif num_scheduled_requests <= 32:
            return min(self.num_steps_per_schedule, 4)
        else:
            return self.num_steps_per_schedule

    # ---- 预热 ----

    def warmup(self, model, tokenizer, device: torch.device) -> None:
        """
        预热 ROCm kernel 缓存。

        DCU 上 kernel 首次调用会触发 JIT 编译（通过 HIP RT 或 hipModule），
        后续调用直接从缓存加载。预热确保正式评测时无编译抖动。

        调用时机：服务初始化阶段，在接受真实请求之前。
        """
        if self._warmed_up:
            return

        dummy_input = tokenizer("Warmup pass", return_tensors="pt").to(device)

        for _ in range(5):
            with torch.no_grad():
                _ = model(**dummy_input)

        # 同步 + 刷新缓存：等价于 hipDeviceSynchronize + 显存整理
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        self._warmed_up = True

    @property
    def device_info(self) -> dict:
        """返回当前设备信息（用于日志和调试）"""
        return {
            "device_name": self.get_device_name(),
            "is_rocm": self.is_rocm(),
            "hip_graph_enabled": self.enable_hip_graph,
            "max_batch_size": self.max_batch_size,
        }
