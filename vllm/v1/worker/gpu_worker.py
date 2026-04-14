# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""

import gc
import os
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from datetime import timedelta
from types import NoneType
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)
from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized
from vllm.distributed.eplb.eplb_utils import override_envs_for_eplb
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.parallel_state import (
    Handle,
    get_pp_group,
    get_tp_group,
)
from vllm.distributed.weight_transfer import WeightTransferEngineFactory
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
from vllm.platforms import current_platform
from vllm.profiler.wrapper import CudaProfilerWrapper, TorchProfilerWrapper
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.tracing import instrument
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import MemorySnapshot, format_gib, memory_profiling
from vllm.utils.torch_utils import is_quantized_kv_cache, set_random_seed
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
)
from vllm.v1.utils import compute_iteration_details, report_usage_stats
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.worker_base import WorkerBase
from vllm.v1.worker.workspace import init_workspace_manager

from ...model_executor.model_loader import TensorizerLoader
from .gpu.warmup import warmup_kernels
from .utils import request_memory

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class AsyncIntermediateTensors(IntermediateTensors):
    """IntermediateTensors with lazy comm synchronization"""

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        comm_handles: list[Handle] | None = None,
        comm_postprocess: list[Callable[[], None]] | None = None,
    ) -> None:
        super().__init__(tensors)
        self._comm_handles = comm_handles
        self._comm_postprocess = comm_postprocess
        self._comm_waited = False

    def wait_for_comm(self) -> None:
        if self._comm_waited:
            return
        if self._comm_handles:
            for handle in self._comm_handles:
                handle.wait()
        if self._comm_postprocess:
            for fn in self._comm_postprocess:
                fn()
        self._comm_waited = True

    def __getattribute__(self, name: str):
        # ensure `.tensors` is ready before use
        if name == "tensors" and not object.__getattribute__(self, "_comm_waited"):
            object.__getattribute__(self, "wait_for_comm")()
        return object.__getattribute__(self, name)


class Worker(WorkerBase):
    """
    vLLM 官方默认的 Worker 实现。
    它继承自 WorkerBase，主要负责 GPU 设备的管理、张量运算精度的设置以及各种高级硬件特性的初始化。
    在不指定特殊硬件（如 TPU/Neuron）的情况下，系统默认拉起的就是这个 Worker。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        # ---------------------------------------------------------
        # 1. 继承并初始化基类配置 (WorkerBase)
        # ---------------------------------------------------------
        # 调用父类构造函数，将所有的 config 配置文件“解压缩”绑定到 self 身上。
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        # ---------------------------------------------------------
        # 2. PyTorch 矩阵乘法精度优化 (Float32 Matmul Precision)
        # ---------------------------------------------------------
        # 大模型中哪怕是浮点运算的尾数差异也会影响性能。
        # 现代 NVIDIA GPU (如 Ampere 架构以后的 Tensor Cores) 允许使用 TF32 来加速 FP32 矩阵乘法。
        # 这里从环境变量获取设置（通常为 'high' 或 'highest'），然后告诉 PyTorch 应该使用多高的精度来计算。
        precision = envs.VLLM_FLOAT32_MATMUL_PRECISION
        torch.set_float32_matmul_precision(precision)

        # ---------------------------------------------------------
        # 3. 弹性专家并行 (Elastic EP) 缩放支持
        # ---------------------------------------------------------
        # 这是一个针对混合专家模型 (MoE，如 Mixtral/DeepSeek) 的前沿特性。
        # 当处理 MoE 模型时，它允许集群在运行中动态地增加或减少用来计算不同“专家”的 GPU 节点。
        from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor
        self.elastic_ep_executor = ElasticEPScalingExecutor(self)

        # ---------------------------------------------------------
        # 4. 睡眠模式缓存 (Sleep State Buffers)
        # ---------------------------------------------------------
        # 用于在资源受限的环境下（比如后台闲置的大模型服务），
        # 当模型进入“睡眠”状态时，把一些关键的张量 (Tensor) 保存起来，避免被回收。
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}

        # ---------------------------------------------------------
        # 5. 模型权重传输引擎 (Weight Transfer Engine)
        # ---------------------------------------------------------
        # 这是一个用于高可用/快速弹性部署的设计。
        # 当启动新的 Worker 时，如果不走磁盘加载，而是直接通过网络（如 RDMA/NCCL）
        # 把权重张量从现有的 Worker 拷贝过来，启动速度会提升数倍。
        # 这是一个懒加载机制，只在配置开启时才初始化引擎。
        self.weight_transfer_engine = (
            WeightTransferEngineFactory.create_engine(
                self.vllm_config.weight_transfer_config,
                self.vllm_config.parallel_config,
            )
            if self.vllm_config.weight_transfer_config is not None
            else None
        )

        # ---------------------------------------------------------
        # 6. PyTorch/CUDA 性能分析器 (Profiler) 配置
        # ---------------------------------------------------------
        # Profiler 用于追踪每一行代码、每一个 CUDA 算子到底耗时多少（分析性能瓶颈）。
        self.profiler: Any | None = None
        self.profiler_config = vllm_config.profiler_config

        # 安全检查：只允许 "torch" (PyTorch自带)、"cuda" (NVIDIA Nsight) 或者 None。
        # 注意：这里只是验证配置合法，真正的实例化会在调用 profile() 时懒加载（因为开销很大）。
        if self.profiler_config.profiler not in ("torch", "cuda", None):
            raise ValueError(f"Unknown profiler type: {self.profiler_config.profiler}")

        # ---------------------------------------------------------
        # 7. V2 架构升级标记与流水线并行状态
        # ---------------------------------------------------------
        # vLLM 正在经历底层 ModelRunner (模型执行器) 从 V1 到 V2 的架构大升级。
        # 这个环境变量标志决定了稍后它会实例化新版还是旧版的核心算子引擎。
        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER
        
        # 针对流水线并行 (Pipeline Parallelism, PP) 的优化。
        # 记录上一轮迭代中还未完成的“发送 (Send)”任务句柄，用于实现通信与计算的重叠 (Overlap)。
        self._pp_send_work: list[Handle] = []

    def sleep(self, level: int = 1) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator

        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone() for name, buffer in model.named_buffers()
            }

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %s GiB memory, %s GiB memory is still in use.",
            format_gib(freed_bytes),
            format_gib(used_bytes),
        )

    def wake_up(self, tags: list[str] | None = None) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

        # If the KV cache has just been woken up,
        # the internal state of cache_engine must be reset,
        # especially the FP8 scaling factor.
        if (
            (tags is None or "kv_cache" in tags)
            and is_quantized_kv_cache(self.cache_config.cache_dtype)
            and hasattr(self.model_runner, "init_fp8_kv_scales")
        ):
            self.model_runner.init_fp8_kv_scales()

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if not self.vllm_config.model_config.enable_sleep_mode:
            return nullcontext()

        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        if tag == "weights":
            assert allocator.get_current_usage() == 0, (
                "Sleep mode can only be used for one instance per process."
            )
        return allocator.use_memory_pool(tag=tag)

    @instrument(span_name="Init device") # 性能探针，记录这整个初始化过程耗时
    def init_device(self):
        # ---------------------------------------------------------
        # 阶段 1：物理设备绑定与“工位”计算 (Hardware Binding & Rank Math)
        # ---------------------------------------------------------
        if self.device_config.device_type == "cuda":
            # 踢掉 Ray 带来的一个恶心 BUG：Ray 默认注入的环境变量会干扰 CUDA Graph 的静态图捕获。
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            
            parallel_config = self.parallel_config
            # 【极度复杂的分布式坐标换算】
            # 如果不是用 Ray 或者外部启动器（说明可能是原生 Multiprocessing 启动），
            # 且启用了 DP (数据并行) + TP (张量并行) + PP (流水线并行) 的混合 3D 并行时，
            # 必须重新计算当前进程应该霸占这台机器上的哪一张卡 (local_rank)。
            if (
                parallel_config.distributed_executor_backend not in ("ray", "external_launcher")
                and parallel_config.data_parallel_backend != "ray"
                and parallel_config.nnodes_within_dp == 1
            ):
                # 获取局部的数据并行 Rank
                dp_local_rank = self.parallel_config.data_parallel_rank_local
                if dp_local_rank is None:
                    dp_local_rank = self.parallel_config.data_parallel_index

                # 计算一个完整的模型副本需要多少张卡 (TP * PP)
                tp_pp_world_size = (
                    self.parallel_config.pipeline_parallel_size
                    * self.parallel_config.tensor_parallel_size
                )

                # 【公式】：当前卡的物理槽位 = DP组号 * 单个模型所需卡数 + 组内卡号
                self.local_rank += dp_local_rank * tp_pp_world_size
                
                # 防御性断言：防止算出来的物理卡号超出了机器上实际插入的显卡总数
                assert self.local_rank < torch.accelerator.device_count(), (...)
                visible_device_count = torch.accelerator.device_count() if torch.cuda.is_available() else 0
                assert self.parallel_config.local_world_size <= visible_device_count, (...)

            # 【核心物理动作】：将当前 Python 进程死死绑定到算出来的这块 GPU 上！
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.accelerator.set_device_index(self.device)

            # 检查当前这块卡是不是太老了，支不支持用户要求的精度 (比如老卡跑不了 bfloat16 或 fp8)
            current_platform.check_if_supports_dtype(self.model_config.dtype)

            # ---------------------------------------------------------
            # 阶段 2：底层硬件建网 (Distributed Networking Initialization)
            # ---------------------------------------------------------
            # 【极其关键的顺序】：必须在“测量可用显存”之前，初始化 NCCL (分布式环境)！
            # 因为 NCCL 建群时，为了保证通信速度，会强行向这块 GPU 申请一笔“公款”（内部通信 Buffer，通常几百MB）。
            # 如果先测显存再建群，算出来的 KV Cache 可用空间就会偏大，导致后期直接 OOM。
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )

            if self.use_v2_model_runner:
                logger.info_once("Using V2 Model Runner", scope="local")

            # 设定随机种子，保证所有卡在某些需要随机的操作（比如某些采样或初始化）时步调一致
            set_random_seed(self.model_config.seed)

            # ---------------------------------------------------------
            # 阶段 3：显存大清查 (Memory Profiling & Snapshot)
            # ---------------------------------------------------------
            # NCCL 建群完毕，现在强制做一次深度的垃圾回收和显存清理，把碎渣倒掉。
            gc.collect()
            torch.accelerator.empty_cache()

            # 【拍快照】：记录当前 GPU 显存的精确状态（总共多少，还剩多少）。
            self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
            # 【预留口粮】：基于当前快照，系统会计算出“框架自身和各种杂项需要预留多少显存”，
            # 剩下的所有显存，稍后将被 PagedAttention 全部霸占！
            self.requested_memory = request_memory(init_snapshot, self.cache_config)
            logger.debug("worker init memory snapshot: %r", self.init_snapshot)
            logger.debug("worker requested memory: %sGiB", format_gib(self.requested_memory))
            
        else:
            # 如果不是 CUDA/兼容架构，直接罢工
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # ---------------------------------------------------------
        # 阶段 4：组装计算引擎 (Engine Instantiation)
        # ---------------------------------------------------------
        # 初始化 Workspace 管理器：用于为底层自定义的 CUDA 算子分配临时内存池。
        # 如果开启了 DBO (微批处理优化)，需要给它双倍的临时池 (num_ubatches=2)。
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        # 【终极武器出库】：实例化 ModelRunner！
        # 注意：此时模型权重仍然还没有加载。这里只是把 Runner 组装好。
        # 稍后外层会调用 `worker.load_model()`，Runner 才会真正使用 Meta Device 和 mmap 去吸入权重。
        if self.use_v2_model_runner:
            from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
            self.model_runner: GPUModelRunner = GPUModelRunnerV2(  # type: ignore
                self.vllm_config, self.device
            )
        else:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1
            self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)

        # ---------------------------------------------------------
        # 阶段 5：主节点打卡 (Telemetry)
        # ---------------------------------------------------------
        if self.rank == 0:
            # 如果是全局 0 号卡（总指挥），负责向官方发送一下匿名使用统计数据
            report_usage_stats(self.vllm_config)
        
        # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
        # to hijack tensor allocation.
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """
        加载模型权重到 GPU。
        
        参数:
        - load_dummy_weights: 如果为 True，则不加载真实权重，而是初始化随机/空权重。
        这通常用于显存分析（Profiling）或测试，以避免昂贵的磁盘 I/O。
        """
        
        # 使用括号语法同时开启多个上下文管理器（Python 3.10+ 支持）
        with (
            # 1. 显存池上下文管理
            # 尝试获取一个专门标记为 "weights" 的显存池上下文。
            # 在多卡或复杂显存管理环境下，这可以确保模型权重的分配发生在特定的显存池中，
            # 减少显存碎片，并方便后续对权重占用的显存进行统一追踪或回收。
            self._maybe_get_memory_pool_context(tag="weights"),
            
            # 2. 全局配置上下文绑定
            # set_current_vllm_config 是一个线程局部（Thread-local）的上下文管理器。
            # 它可以将当前的 vllm_config 设为全局可用，这样模型深层的各种算子（Layer/Kernel）
            # 就不需要一层层传递配置对象，直接通过全局 Hook 即可读取所需的架构参数。
            set_current_vllm_config(self.vllm_config),
        ):
            # 3. 核心加载动作
            # 实际的加载逻辑委托给了内部的 self.model_runner 对象。
            # model_runner 会根据前面的配置实例化真正的神经网络类（如 LlamaForCausalLM），
            # 并将模型参数从磁盘（或 CPU 内存）搬运到当前 GPU 的显存中。
            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)

    def update_config(self, overrides: dict[str, Any]) -> None:
        self.model_runner.update_config(overrides)

    def reload_weights(self, *args, **kwargs) -> None:
        self.model_runner.reload_weights(*args, **kwargs)

    @torch.inference_mode() # 强制关闭梯度计算，这在推理阶段是必须的，能节省大量显存和算力
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.
        【整体逻辑】先对现有的显存使用情况进行一次摸底（Profiling），
        然后通过做减法，算出还有多少字节（bytes）可以分给 KV Cache。
        """
        
        # =========================================================================
        # 阶段 1：手动接管模式（Manual Override）
        # =========================================================================
        # 如果用户在启动配置中强行指定了 KV Cache 的绝对大小（以字节为单位）
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            # 即使强行指定了大小，依然需要跑一次“假推理（Profile Run）”。
            # 因为底层的 CUDA 算子或计算图（CUDA Graphs）需要借助这次假推理来完成编译和预热。
            self.model_runner.profile_run()

            # 打印日志警告用户：你正在使用手动控制模式，
            # 引擎将忽略 --gpu-memory-utilization（默认 0.9）这个参数。
            # 如果爆显存了，请自己对比两次运行的日志来微调这个值。
            msg = ( ... ) # 省略长字符串提示
            logger.info(msg)
            return kv_cache_memory_bytes

        # =========================================================================
        # 阶段 2：模拟压测（Dummy Profiling）- 获取动态激活显存峰值
        # =========================================================================
        # Execute a forward pass with dummy inputs to profile the memory usage of the model.
        # 开启显存追踪的上下文管理器。它会记录压测开始前的显存快照。
        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            # 【核心动作】：用极限大小的假数据（最大 Batch Size，最大长度）跑一次完整的前向传播。
            self.model_runner.profile_run()

            # 向 PyTorch 底层查询：在刚才那次假推理中，显存占用“瞬间飙到的最高点（Peak）”是多少？
            profile_torch_peak = torch.accelerator.memory_stats(self.device).get(
                "allocated_bytes.all.peak", 0
            )

            # =========================================================================
            # 阶段 3：CUDA Graph 开销预估
            # =========================================================================
            # CUDA Graph 会把计算图刻录在显卡上，这本身也会吃掉一部分显存。
            # AMD 的 ROCm 平台在这方面统计不准，所以跳过。Nvidia CUDA 平台则进行预估。
            cudagraph_memory_estimate = 0
            if not self.model_config.enforce_eager and not current_platform.is_rocm():
                cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()

        # =========================================================================
        # 阶段 4：算总账（精准的减法运算）
        # =========================================================================
        # 计算假推理带来的纯增量：峰值显存 - 压测前的基础显存
        profile_result.torch_peak_increase = (
            profile_torch_peak - profile_result.before_profile.torch_peak
        )
        
        # 算出模型运行时“雷打不动”必须占用的总显存量（Non KV Cache Memory）：
        # = 非 Torch 框架占用的额外开销 + 刚才测出的激活值峰值增量 + 模型权重的死体积
        profile_result.non_kv_cache_memory = (
            profile_result.non_torch_increase
            + profile_result.torch_peak_increase
            + profile_result.weights_memory
        )

        # 是否要把 CUDA Graph 的预估开销算进账本里（基于环境变量控制）
        cudagraph_memory_estimate_applied = (
            cudagraph_memory_estimate
            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
            else 0
        )

        self.non_torch_memory = profile_result.non_torch_increase
        self.peak_activation_memory = profile_result.torch_peak_increase
        self.cudagraph_memory_estimate = cudagraph_memory_estimate

        # =========================================================================
        # 阶段 5：环境防污染安全校验（极其严谨的设计）
        # =========================================================================
        free_gpu_memory = profile_result.after_profile.free_memory
        
        # 假设前提：在 vLLM 测显存的这几秒钟里，显卡上的其他进程没有乱动显存。
        # 如果压测完之后，发现当前显卡总剩余量，居然比压测前（init_snapshot）还要多，
        # 说明有个其他程序在这期间释放了显存，导致我们的测量基准被污染了！直接报错阻断。
        assert self.init_snapshot.free_memory >= free_gpu_memory, ( ... )

        # 【最终得数】：
        # 理论上留给 KV Cache 的空间 = 系统允许你用的总显存限制（比如总显存的 0.9） - 雷打不动的固定开销 - CUDA Graph 预留
        self.available_kv_cache_memory_bytes = (
            self.requested_memory
            - profile_result.non_kv_cache_memory
            - cudagraph_memory_estimate_applied
        )

        # =========================================================================
        # 阶段 6：详细日志与未来版本警告
        # =========================================================================
        unrequested_memory = self.init_snapshot.free_memory - self.requested_memory
        # 打印各种 debug 级别的日志，方便开发者排查显存分配细节
        logger.debug(...)
        logger.debug(...)
        logger.debug(profile_result)
        logger.info_once(...) # 打印最终可用的 KV Cache 容量

        # 针对即将在 v0.19 版本默认开启的 CUDA Graph 显存预估功能，给出平滑过渡提示：
        # 如果把 CUDA Graph 开销算进来，会导致留给 KV Cache 的空间变小（可能引起原来能跑的模型现在跑不了）。
        # 所以系统会贴心地计算出一个 suggested_util（建议的显存利用率比例），
        # 告诉你：“如果你想保持和旧版本一样的 KV 容量，请把启动参数从 0.90 提高到 0.92”。
        if cudagraph_memory_estimate > 0:
            total_mem = self.init_snapshot.total_memory
            current_util = self.cache_config.gpu_memory_utilization
            cg_util_delta = cudagraph_memory_estimate / total_mem
            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:
                # 已经开启该特性的提示
                equiv_util = round(current_util - cg_util_delta, 4)
                suggested_util = min(round(current_util + cg_util_delta, 4), 1.0)
                logger.info(...)
            else:
                # 尚未开启该特性的警告提示
                suggested_util = min(round(current_util + cg_util_delta, 4), 1.0)
                logger.info(...)

        # 带着最终算出的“完美余额”，交差！
        return int(self.available_kv_cache_memory_bytes)

    def get_kv_connector_handshake_metadata(self) -> dict | None:
        """Get KV connector metadata from this worker if available."""

        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None

        tp_rank = get_tp_group().rank_in_group
        return {tp_rank: metadata}

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def update_max_model_len(self, max_model_len: int) -> None:
        """Update max_model_len after auto-fit to GPU memory.
        This is called when max_model_len=-1 is used and the engine
        automatically determines the maximum context length that fits
        in GPU memory. Workers need to update their cached max_model_len
        to match the engine's decision.
        """
        self.model_config.max_model_len = max_model_len
        if self.model_runner is not None:
            self.model_runner.update_max_model_len(max_model_len)
        logger.debug("Updated max_model_len to %d", max_model_len)

    # 分布式追踪：在性能监控大盘中记录这部分（真正划拨物理显存）的耗时
    @instrument(span_name="Allocate KV cache")
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config.
        使用刚刚（经过 Profiling 和 Scheduler 确认过的）最终配置，正式分配物理显存。
        """

        # 【步骤 1：同步本地账本】
        # Update local config with adjusted num blocks after profiling,
        # so that it's available to the warmup stage.
        # 经过上一步的极限压测和 Auto-fit 裁剪，num_blocks 可能被缩小了。
        # 这里把最终敲定的“真实 Block 数量”写回给 Worker 本地的配置中，
        # 这样接下来的 Warmup（预热）阶段，系统才知道到底在多大的池子里游泳。
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks

        # 【步骤 2：KV 缓存传输（分布式分离推理）初始化】
        # Init kv cache connector here, because it requires `kv_cache_config`.
        # vLLM 的一个前沿特性：支持跨节点的 KV 传输（比如一台机器专门做 Prefill 算提示词，
        # 算完把 KV Cache 传给另一台机器做 Decode 生成）。
        # 作者留下的坑位提示：必须在真正划拨显存前初始化这个 Connector，以免后面被其他缓存组干扰。
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        # 【步骤 3：核心动作 —— 真正划拨物理显存池】
        # 是否启用了“睡眠模式”（Sleep Mode，用于多租户或弹性扩缩容以节省算力成本）
        if self.vllm_config.model_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            # 拿到 CUDA 底层内存分配器的实例
            allocator = CuMemAllocator.get_instance()
            
            # 💡【核心呼应】：还记得我们之前讨论的 "weights" (权重) 显存池吗？
            # 这里它又出现了！这次打的标签是 "kv_cache"！
            # 开启上下文管理器，这意味着接下来底层划拨的几十 GB 显存，
            # 全部被划归到了名叫 "kv_cache" 的专属隔离区。
            # 这样在触发“睡眠模式”时，系统可以一键清空或挂起 "kv_cache" 池，而绝对不会误伤 "weights" 里的模型参数！
            with allocator.use_memory_pool(tag="kv_cache"):
                self.model_runner.initialize_kv_cache(kv_cache_config)
        else:
            # 常规模式：直接在 PyTorch 默认的公共显存池里划出一大块连续内存作为 KV Cache。
            self.model_runner.initialize_kv_cache(kv_cache_config)

        # 【步骤 4：MoE 模型专属附加设置】
        # 如果当前跑的是 MoE 模型（混合专家模型，如 Mixtral/DeepSeek），
        # 并且用户开启了“返回专家路由信息”的开关，则在这里初始化路由捕获器。
        if self.model_config.enable_return_routed_experts:
            self.model_runner.init_routed_experts_capturer()

        # 【步骤 5：安全与隔离清理（KV Zeroing）】
        # Build KV-zero metadata outside the CuMem pool so the bookkeeping
        # GPU tensors (seg_addrs, block-id buffers) use the standard PyTorch
        # allocator and are not discarded during sleep/wake cycles.
        
        # 在多租户云环境中，为了防止后一个用户读取到前一个用户残留在显卡里的 Prompt 信息，
        # 需要对废弃的 KV Cache 进行“零值清空”（KV Zeroing）。
        if kv_cache_config.needs_kv_cache_zeroing and hasattr(
            self.model_runner, "_init_kv_zero_meta"
        ):
            # 💡【精妙的设计】：注意作者上面的英文注释。
            # 记录哪些 Block 需要被清零的“记账元数据（Metadata）”，必须脱离刚才那个 "kv_cache" 的池子来创建！
            # 因为如果系统进入睡眠模式，"kv_cache" 里的物理内容会被全部丢弃，
            # 如果把记账的本子也放在那里，醒来后系统就找不到哪些内存块是干净的了。
            # 所以这段代码写在了 with allocator 之外，强制使用标准的 PyTorch 显存分配。
            self.model_runner._init_kv_zero_meta()

    @instrument(span_name="Warmup (GPU)")
    def compile_or_warm_up_model(self) -> float:
        warmup_sizes: list[int] = []

        if self.vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:
            # warm up sizes that are not in cudagraph capture sizes,
            # but users still want to compile for better performance,
            # e.g. for the max-num-batched token size in chunked prefill.
            compile_sizes = self.vllm_config.compilation_config.compile_sizes
            warmup_sizes = compile_sizes.copy() if compile_sizes is not None else []  # type: ignore[assignment]
            cg_capture_sizes: list[int] = []

            if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                cg_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
                cg_capture_sizes = [] if cg_sizes is None else cg_sizes
                warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]

            compile_ranges = self.vllm_config.compilation_config.get_compile_ranges()
            # For each compile_range, if none of the batch sizes
            # in warmup_sizes or cudagraph_capture_sizes are in the range,
            # add the end of the range to ensure compilation/warmup.
            all_sizes = set(cg_capture_sizes)
            all_sizes.update([x for x in warmup_sizes if isinstance(x, int)])
            for compile_range in compile_ranges:
                if not any(x in compile_range for x in all_sizes):
                    warmup_sizes.append(compile_range.end)

        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

        # Warmup and tune the kernels used during model execution before
        # cuda graph capture.
        kernel_warmup(self)

        cuda_graph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            cuda_graph_memory_bytes = self.model_runner.capture_model()

        # Compare actual vs estimated CUDA graph memory (if we did profiling)
        if (
            hasattr(self, "cudagraph_memory_estimate")
            and self.cudagraph_memory_estimate > 0
        ):
            GiB = lambda b: round(b / GiB_bytes, 2)
            diff = abs(cuda_graph_memory_bytes - self.cudagraph_memory_estimate)
            logger.info(
                "CUDA graph pool memory: %s GiB (actual), %s GiB (estimated), "
                "difference: %s GiB (%.1f%%).",
                GiB(cuda_graph_memory_bytes),
                GiB(self.cudagraph_memory_estimate),
                GiB(diff),
                100 * diff / max(cuda_graph_memory_bytes, 1),
            )

        if self.cache_config.kv_cache_memory_bytes is None and hasattr(
            self, "peak_activation_memory"
        ):
            # Suggests optimal kv cache memory size if we rely on
            # memory_profiling to guess the kv cache memory size which
            # provides peak_activation_memory and a few other memory
            # consumption. `memory_profiling` does not consider
            # CUDAGraph memory size and may not utilize all gpu memory.
            # Users may want fine-grained control to specify kv cache
            # memory size.

            # empirically observed that the memory profiling may
            # slightly underestimate the memory consumption.
            # So leave a small buffer (=150MiB) to avoid OOM.
            redundancy_buffer_memory = 150 * (1 << 20)

            non_kv_cache_memory = (
                self.model_runner.model_memory_usage
                + self.peak_activation_memory
                + self.non_torch_memory
                + cuda_graph_memory_bytes
            )
            kv_cache_memory_bytes_to_gpu_limit = (
                self.init_snapshot.free_memory
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )
            kv_cache_memory_bytes_to_requested_limit = (
                int(self.requested_memory)
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )

            msg = (
                f"Free memory on device "
                f"({format_gib(self.init_snapshot.free_memory)}/"
                f"{format_gib(self.init_snapshot.total_memory)} GiB) on startup. "
                f"Desired GPU memory utilization is "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{format_gib(self.requested_memory)} GiB). "
                f"Actual usage is {format_gib(self.model_runner.model_memory_usage)} "
                f"GiB for weight, {format_gib(self.peak_activation_memory)} GiB "
                f"for peak activation, {format_gib(self.non_torch_memory)} GiB "
                f"for non-torch memory, and {format_gib(cuda_graph_memory_bytes)} "
                f"GiB for CUDAGraph memory. Replace gpu_memory_utilization "
                f"config with `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_requested_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_requested_limit)} GiB) to fit "
                f"into requested memory, or `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_gpu_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_gpu_limit)} GiB) to fully "
                f"utilize gpu memory. Current kv cache memory in use is "
                f"{format_gib(self.available_kv_cache_memory_bytes)} GiB."
            )

            logger.debug(msg)

        if self.use_v2_model_runner:
            # V2: Run full execute_model + sample_tokens to JIT compile triton kernels.
            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
        elif get_pp_group().is_last_rank:
            # V1: Warm up sampler and preallocate memory buffer for logits and other
            # sampling related tensors of max possible shape to avoid memory
            # fragmentation issue.
            # NOTE: This is called after `capture_model` on purpose to prevent
            # memory buffers from being cleared by `torch.accelerator.empty_cache`.
            max_num_reqs = min(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
            )

            # We skip EPLB here since we don't want to record dummy metrics
            hidden_states, last_hidden_states = self.model_runner._dummy_run(
                num_tokens=max_num_reqs,
                skip_eplb=True,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

        return self.compilation_config.compilation_time

    def reset_mm_cache(self) -> None:
        self.model_runner.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        self.model_runner.reset_encoder_cache()

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def get_compilation_match_table(self) -> dict[str, int]:
        from vllm.compilation.passes.vllm_inductor_pass import get_match_table

        return get_match_table()

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        """Get encoder timing stats from model runner."""
        return self.model_runner.get_encoder_timing_stats()

    def annotate_profile(self, scheduler_output):
        # add trace annotation so that we can easily distinguish
        # context/generation request numbers in each iteration.
        # A context request is a request that has not yet generated any tokens
        if not self.profiler:
            return nullcontext()

        self.profiler.step()

        iteration_details = compute_iteration_details(scheduler_output)

        annotation = "".join(
            [
                "execute_context_",
                str(iteration_details.num_ctx_requests),
                "(",
                str(iteration_details.num_ctx_tokens),
                ")_generation_",
                str(iteration_details.num_generation_requests),
                "(",
                str(iteration_details.num_generation_tokens),
                ")",
            ]
        )
        return self.profiler.annotate_context_manager(annotation)

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)

    # 强制禁用 PyTorch 的梯度计算引擎。
    # 在 LLM 推理场景下，这能显著降低显存占用并提升前向传播速度。
    @torch.inference_mode()
    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        
        # ---------------------------------------------------------
        # 1. 扫尾工作：确保上一轮的流水线发送 (PP Send) 已完成
        # ---------------------------------------------------------
        # 在流水线并行中，当前节点计算完后会“异步”把结果发给下一个节点。
        # 这里必须 wait() 确保上一轮的数据已经彻底发完，
        # 否则当前轮次的新计算可能会覆盖底层的通信 buffer，导致数据损坏。
        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []

        intermediate_tensors = None
        # 检查这轮是否真的有 Token 需要算（还是单纯的空转调度）
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        all_gather_tensors = {}
        compilation_config = self.vllm_config.compilation_config
        parallel_config = self.vllm_config.parallel_config

        # ---------------------------------------------------------
        # 2. 序列并行 (Sequence Parallelism, SP) 与 PP 的融合预处理
        # ---------------------------------------------------------
        # 如果同时开启了流水线并行 (PP > 1) 和序列并行 (SP)，情况会非常复杂。
        # 因为 SP 会把长序列切碎分散在多个 GPU 上。在跨流水线节点传输前，
        # 我们需要决定是否要先把这些碎片 All-Gather 拼起来。
        if (
            parallel_config.pipeline_parallel_size > 1
            and compilation_config.pass_config.enable_sp
            and forward_pass
        ):
            # 目前只有 V1 架构的 GPUModelRunner 完美支持这种复杂的 SP+PP 混合逻辑
            assert not self.use_v2_model_runner
            
            # 提取本批次每个请求被调度的 Token 数量
            num_scheduled_tokens_np = np.array(
                list(scheduler_output.num_scheduled_tokens.values()),
                dtype=np.int32,
            )
            
            # TODO: 这是一个技术债。这里提前调用 _determine_batch_execution_and_padding
            # 是为了算出当前 batch 的结构 (batch_desc)，从而判断底层张量是否是 scattered（分散的）。
            _, batch_desc, _, _, _ = (
                self.model_runner._determine_batch_execution_and_padding(
                    num_tokens=num_scheduled_tokens,
                    num_reqs=len(num_scheduled_tokens_np),
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    max_num_scheduled_tokens=num_scheduled_tokens_np.max(),
                    use_cascade_attn=False,
                )
            )
            # 确定哪些张量需要在接收/发送时进行 All-Gather
            all_gather_tensors = {
                "residual": not is_residual_scattered_for_sp(
                    self.vllm_config, batch_desc.num_tokens
                )
            }

        # ---------------------------------------------------------
        # 3. 流水线接收 (Pipeline Receive): 承接上游数据
        # ---------------------------------------------------------
        # 如果当前节点不是流水线的第一级 (not is_first_rank)，它不能凭空算出结果，
        # 它必须先接收上一个 GPU 节点传过来的隐藏层状态 (Hidden States)。
        if forward_pass and not get_pp_group().is_first_rank:
            # 发起异步接收 (irecv)。使用异步是为了尽早发起网络请求，隐藏通信延迟。
            tensor_dict, comm_handles, comm_postprocess = (
                get_pp_group().irecv_tensor_dict(
                    all_gather_group=get_tp_group(),
                    all_gather_tensors=all_gather_tensors,
                )
            )
            assert tensor_dict is not None
            # 将收到的原始 Tensor 包装成 AsyncIntermediateTensors，
            # 底层的 ModelRunner 在真正需要用这个张量之前，会自动调用 wait()。
            intermediate_tensors = AsyncIntermediateTensors(
                tensor_dict,
                comm_handles=comm_handles,
                comm_postprocess=comm_postprocess,
            )

        # ---------------------------------------------------------
        # 4. 核心计算：驱动 Model Runner (The Core Engine)
        # ---------------------------------------------------------
        # 把图纸 (scheduler_output) 和上游传来的半成品 (intermediate_tensors)
        # 一起扔给底层的算子引擎。这里才是真正调用 FlashAttention 等 CUDA 算子的地方。
        with self.annotate_profile(scheduler_output):
            output = self.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )
            
            # 针对 V2 架构 + Pooling 模型 (如 Embedding/Reward 模型) 的特殊逻辑
            if (
                self.use_v2_model_runner
                and self.model_runner.is_pooling_model
                and output is None
            ):
                output = self.model_runner.pool()  # type: ignore
                
            # 如果输出的是最终的 Token 或 Logits (说明这是流水线的最后一级)，
            # 直接将结果层层返回给最外层的主进程。
            if isinstance(
                output, ModelRunnerOutput | AsyncModelRunnerOutput | NoneType
            ):
                return output

        # ---------------------------------------------------------
        # 5. 流水线发送 (Pipeline Send): 传递给下游
        # ---------------------------------------------------------
        # 如果代码走到了这里，说明当前节点是流水线的“中间节点”。
        # output 的类型必定是 IntermediateTensors（即中间层的 Hidden States）。
        assert isinstance(output, IntermediateTensors)
        parallel_config = self.vllm_config.parallel_config
        
        # 确保中间节点不是最后一个 rank (否则它应该返回最终的 Token)，并且未使用外部启动器。
        assert (
            parallel_config.distributed_executor_backend != "external_launcher"
            and not get_pp_group().is_last_rank
        )

        # 发起非阻塞发送 (isend)。
        # 把当前 GPU 算出来的中间层状态，通过 NCCL/P2P 发给流水线里的下一个 GPU。
        # 句柄 (Handle) 保存在 self._pp_send_work 中，供下一轮 step 开始时检查。
        self._pp_send_work = get_pp_group().isend_tensor_dict(
            output.tensors,
            all_gather_group=get_tp_group(),
            all_gather_tensors=all_gather_tensors,
        )

        # 中间节点不需要向主进程返回具体的 Token 结果，所以返回 None。
        return None

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.model_runner.take_draft_token_ids()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # Check if profiling is enabled
        if self.profiler_config is None or self.profiler_config.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. Please set --profiler-config to enable "
                "profiling. Example: "
                "'--profiler-config.profiler=torch --profiler-config.torch_profiler_dir"
                "=YOUR_DIR_PATH_TO_DUMP_TRACE'"
            )

        if is_start:
            # Generate the trace name by combining prefix with comprehensive rank suffix
            from vllm.distributed.utils import get_worker_rank_suffix

            rank_suffix = get_worker_rank_suffix(global_rank=self.rank)

            # Build the full trace name
            if profile_prefix:
                trace_name = f"{profile_prefix}_{rank_suffix}"
            else:
                trace_name = rank_suffix

            # Create the profiler wrapper only on the first start call
            if self.profiler is None:
                profiler_type = self.profiler_config.profiler
                if profiler_type == "torch":
                    self.profiler = TorchProfilerWrapper(
                        self.profiler_config,
                        worker_name=trace_name,
                        local_rank=self.local_rank,
                        activities=["CPU", "CUDA"],
                    )
                    logger.debug(
                        "Starting torch profiler with trace name: %s", trace_name
                    )
                elif profiler_type == "cuda":
                    self.profiler = CudaProfilerWrapper(self.profiler_config)
                    logger.debug("Starting CUDA profiler")
                else:
                    # Config validation should prevent this code being reached
                    raise ValueError(
                        f"Invalid profiler value of {self.profiler_config.profiler}"
                    )

            # If profiler already initialized, restart profiling but keep
            # the original trace name from the first initialization.
            self.profiler.start()
        else:
            if self.profiler is None:
                logger.warning("Profiler was not started, nothing to stop.")
                return
            self.profiler.stop()

    def execute_dummy_batch(self) -> None:
        num_tokens = getattr(self.model_runner, "uniform_decode_query_len", 1)
        self.model_runner._dummy_run(num_tokens, uniform_decode=True)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        from vllm.model_executor.model_loader import ShardedStateLoader

        ShardedStateLoader.save_model(
            self.model_runner.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(self, tensorizer_config: "TensorizerConfig") -> None:
        TensorizerLoader.save_model(
            self.get_model(),
            tensorizer_config=tensorizer_config,
            model_config=self.model_config,
        )

    def init_weight_transfer_engine(self, init_info: dict) -> None:
        """
        Initialize weight transfer mechanism.
        For NCCL backend, this creates a process group with the trainer.

        Args:
            init_info: Dictionary containing backend-specific initialization info
        """
        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. "
                "Please set weight_transfer_config to enable weight transfer."
            )
        # Parse dict into backend-specific typed dataclass
        typed_init_info = self.weight_transfer_engine.parse_init_info(init_info)
        self.weight_transfer_engine.init_transfer_engine(typed_init_info)

    def update_weights(self, update_info: dict) -> None:
        """
        Batched weight update from the trainer.

        Args:
            update_info: Dictionary containing backend-specific update info
        """
        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. "
                "Please set weight_transfer_config to enable weight transfer."
            )

        # Parse dict into backend-specific typed dataclass
        typed_update_info = self.weight_transfer_engine.parse_update_info(update_info)

        model = self.model_runner.model

        if typed_update_info.is_checkpoint_format:
            from vllm.model_executor.model_loader.reload import (
                finalize_layerwise_reload,
                initialize_layerwise_reload,
            )

            # Use layerwise reload pattern for checkpoint format weights
            with torch.device(self.device):
                initialize_layerwise_reload(model)
                self.weight_transfer_engine.receive_weights(
                    typed_update_info,
                    load_weights=model.load_weights,
                )
                finalize_layerwise_reload(model, self.model_config)
        else:
            # Weights are already in kernel format, copy directly
            def load_weights_direct(
                weights: list[tuple[str, torch.Tensor]],
            ) -> None:
                for name, weight in weights:
                    param = model.get_parameter(name)
                    param.copy_(weight)

            self.weight_transfer_engine.receive_weights(
                typed_update_info,
                load_weights=load_weights_direct,
            )

        # NCCL broadcast/packed path are asynchronous.
        # Sync here so the next step uses the new weights.
        torch.accelerator.synchronize()

    def shutdown(self) -> None:
        # has_kv_transfer_group can be None during interpreter shutdown.
        if ensure_kv_transfer_shutdown is not None:
            ensure_kv_transfer_shutdown()
        if self.profiler is not None:
            self.profiler.shutdown()

        if weight_transfer_engine := getattr(self, "weight_transfer_engine", None):
            weight_transfer_engine.shutdown()

    def elastic_ep_execute(self, execute_method: str, *args, **kwargs):
        return self.elastic_ep_executor.execute(execute_method, *args, **kwargs)


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialize the distributed environment."""
    attention_config = vllm_config.attention_config
    parallel_config = vllm_config.parallel_config
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance

    init_batch_invariance(attention_config.backend)
    override_envs_for_eplb(parallel_config)
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_method = distributed_init_method or "env://"

    timeout = None
    if parallel_config.distributed_timeout_seconds is not None:
        timeout = timedelta(seconds=parallel_config.distributed_timeout_seconds)

    init_distributed_environment(
        parallel_config.world_size,
        rank,
        init_method,
        local_rank,
        backend,
        timeout,
    )

    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.decode_context_parallel_size,
    )

    # Init ec connector here before KV caches init
    # NOTE: We do not init KV caches for Encoder-only instance in EPD disagg mode
    ensure_ec_transfer_initialized(vllm_config)
