# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tracing import instrument
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.system_utils import update_environment_variables
from vllm.v1.kv_cache_interface import KVCacheSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
else:
    SchedulerOutput = object
    GrammarOutput = object
    AsyncModelRunnerOutput = object
    ModelRunnerOutput = object

logger = init_logger(__name__)

_R = TypeVar("_R")


class WorkerBase:
    """
    Worker 接口类。
    它的核心价值在于：允许 vLLM 干净利落地将不同硬件（GPU、TPU、CPU等）的实现隔离开来。
    同时，它也抽象了控制面 (Control Plane) 的通信逻辑，比如用来在不同 Worker 之间同步请求的元数据。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        """
        初始化所有 Worker 都必须具备的基础组件。

        参数 (Args):
            vllm_config: 完整的 vLLM 巨型配置对象。
            local_rank: 当前节点（物理机）上的设备索引（比如机器上有 8 张卡，这个值就是 0-7）。
            rank: 全局分布式集群中的唯一标识（比如 2 台 8 卡机，这个值就是 0-15）。
            distributed_init_method: 底层分布式通信库（通常是 PyTorch 的 nccl_init_method 或 tcp 地址）的初始化方式。
            is_driver_worker: 身份标识。当前 Worker 是否需要承担“驱动者”的责任（即是否负责汇总结果并和主进程通信）。
        """
        
        # ---------------------------------------------------------
        # 1. 背包整理：将全局大配置拆解为各个子模块配置
        # ---------------------------------------------------------
        # 士兵上前线前，需要把背包里的指令、弹药、补给分门别类放好。
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config               # 模型权重路径、架构、精度等
        self.cache_config = vllm_config.cache_config               # PagedAttention 的 Block 大小等
        self.lora_config = vllm_config.lora_config                 # 动态 LoRA 适配器配置
        self.load_config = vllm_config.load_config                 # 权重下载、加载格式 (safetensors等)
        self.parallel_config = vllm_config.parallel_config         # TP/PP/EP 分布式并行切分策略
        self.scheduler_config = vllm_config.scheduler_config       # 调度器最大并发数等
        self.device_config = vllm_config.device_config             # 硬件设备类型配置
        self.speculative_config = vllm_config.speculative_config   # 推测解码配置
        self.observability_config = vllm_config.observability_config # 监控打点配置
        self.kv_transfer_config = vllm_config.kv_transfer_config   # 跨节点 KV Cache 传输配置
        self.compilation_config = vllm_config.compilation_config   # Torch Compile 图编译配置

        # ---------------------------------------------------------
        # 2. 硬件平台探测
        # ---------------------------------------------------------
        # 动态导入当前的平台环境（比如探测当前是 NVIDIA 环境、AMD 环境还是 Apple Silicon 环境）
        from vllm.platforms import current_platform
        self.current_platform = current_platform

        # ---------------------------------------------------------
        # 3. 分发身份铭牌 (Dog Tags)
        # ---------------------------------------------------------
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        
        # 标记是否为“班长” (Driver)
        # 在张量并行 (TP) 中，所有卡一起算，但只有 Driver Worker 负责把大家算好的结果收集起来，返回给外面的 Scheduler。
        self.is_driver_worker = is_driver_worker

        # ---------------------------------------------------------
        # 4. 预留“武器”插槽 (Lazy Initialization)
        # ---------------------------------------------------------
        # 注意：这里仅仅是占位（赋值为 None）。
        # 真正的显卡绑定 (device) 和 PyTorch 模型实例化 (model_runner) 
        # 会留到具体的子类（如 GPUWorker）调用 init_device() 和 load_model() 时才去执行。
        # 这样做是为了严格防止 CUDA 在多进程 Fork 之前被意外初始化。
        self.device: torch.device | None = None
        self.model_runner: nn.Module | None = None

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation."""
        raise NotImplementedError

    def compile_or_warm_up_model(self) -> float:
        """Prepare model for execution through compilation/warmup.

        Returns:
            The accumulated compilation time in seconds.
        """
        raise NotImplementedError

    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        return

    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        raise NotImplementedError

    def reset_mm_cache(self) -> None:
        reset_fn = getattr(self.model_runner, "reset_mm_cache", None)
        if callable(reset_fn):
            reset_fn()

    def get_model(self) -> nn.Module:
        raise NotImplementedError

    def apply_model(self, fn: Callable[[nn.Module], _R]) -> _R:
        """Apply a function on the model inside this worker."""
        return fn(self.get_model())

    def get_model_inspection(self) -> str:
        """Return a transformers-style hierarchical view of the model."""
        from vllm.model_inspection import format_model_inspection

        return format_model_inspection(self.get_model())

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """Load model onto target device."""
        raise NotImplementedError

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """If this method returns None, sample_tokens should be called immediately after
        to obtain the ModelRunnerOutput.

        Note that this design may be changed in future if/when structured outputs
        parallelism is re-architected.
        """
        raise NotImplementedError

    def sample_tokens(
        self, grammar_output: GrammarOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """Should be called immediately after execute_model iff it returned None."""
        raise NotImplementedError

    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from model configuration."""
        return self.model_config.get_vocab_size()

    def shutdown(self) -> None:
        """Clean up resources held by the worker."""
        return


class WorkerWrapperBase:
    """
    此类代表执行器 (Executor) / 引擎 (Engine) 中的一个独立进程（通常对应一张 GPU）。
    它的核心职责是：【延迟初始化 (Lazy Initialization)】工作节点，并管理该节点的生命周期。
    
    工作流程设计：
    1. 首先实例化这个 Wrapper（此时它只记住了目标 Worker 的模块名和类名，但不真正加载 PyTorch 模型）。
    2. 接着，主进程可以安全地调用 `update_environment_variables` 来设置 CUDA_VISIBLE_DEVICES、NCCL 等环境变量。
    3. 最后，当环境彻底干净、就绪后，调用 `init_worker` 才真正触发底层 GPU 显存的分配和模型的加载。
    """

    def __init__(
        self,
        rpc_rank: int = 0,
        global_rank: int | None = None,
    ) -> None:
        """
        使用给定的 rpc_rank 初始化工作节点包装器。
        
        【重要概念区分】：
        - rpc_rank: 该节点在当前“执行器 (Executor)”管辖范围内的局部编号。
        - global_rank (隐含在分布式组中): 该节点在整个全局分布式通信组 (如 NCCL) 中的真实编号。
        
        在绝大多数标准部署下（比如单机 8 卡启动 1 个 Engine），rpc_rank 和 global_rank 是完全相等的 (0 到 7)。
        
        但是，在多引擎协同工作的极端场景下，它们会不同：
        例如：在 SPMD（单程序多数据）风格的离线推理中，使用 TP=2（两卡张量并行）。
        用户可能会手动启动 2 个独立的脚本（即 2 个引擎/执行器），每个引擎只管 1 个 Worker。
        此时：
        - 对于引擎 A，它的 Worker 的 rpc_rank = 0，但全局 TP rank = 0。
        - 对于引擎 B，它的 Worker 的 rpc_rank = 0，但全局 TP rank = 1。
        """
        
        # 记录局部通信编号（主控进程用这个编号通过 RPC 找它）
        self.rpc_rank: int = rpc_rank
        
        # 记录全局拓扑编号（底层的 PyTorch/NCCL 进行 AllReduce 时用这个编号）
        # 如果没有显式提供 global_rank，默认它等于 rpc_rank
        self.global_rank: int = self.rpc_rank if global_rank is None else global_rank

        # 以下两个核心属性在 Wrapper 初始化时是【悬空】的。
        # 它们只有在后续真正调用了 `init_worker` 方法后才会被赋值。
        
        # self.worker: 指向真正干活的底层 PyTorch 实例 (如 GPUWorker)
        self.worker: WorkerBase 
        
        # self.vllm_config: 保存当前引擎的全局配置
        self.vllm_config: VllmConfig

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.shutdown()

    def update_environment_variables(
        self,
        envs_list: list[dict[str, str]],
    ) -> None:
        envs = envs_list[self.rpc_rank]
        update_environment_variables(envs)

    # 使用 instrument 装饰器进行性能分析打点，记录 Worker 初始化耗时
    @instrument(span_name="Worker init")
    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None:
        """
        在真正实例化 worker 之前，这里会注入一些通用逻辑。
        传入的参数 (kwargs) 最终会被传递给真正的 worker 类的构造函数。
        """
        # ---------------------------------------------------------
        # 1. 参数提取与配置绑定
        # ---------------------------------------------------------
        # all_kwargs 是一个列表，包含了所有 GPU 的配置。
        # 当前进程只取出属于自己 (rpc_rank) 的那一份参数。
        kwargs = all_kwargs[self.rpc_rank]

        vllm_config: VllmConfig | None = kwargs.get("vllm_config")
        assert vllm_config is not None, "初始化 worker 必须提供 vllm_config"
        self.vllm_config = vllm_config

        # 为当前线程开启函数调用追踪（常用于 profiling 性能分析）
        vllm_config.enable_trace_function_call_for_thread()

        # 加载 vLLM 的通用插件（可能包含自定义算子或调度策略）
        from vllm.plugins import load_general_plugins
        load_general_plugins()

        # ---------------------------------------------------------
        # 2. 动态解析 Worker 类 (Dynamic Class Resolution)
        # ---------------------------------------------------------
        # vLLM 支持极高自由度的定制，你可以在配置文件里指定用什么 Worker 类。
        parallel_config = vllm_config.parallel_config
        
        if isinstance(parallel_config.worker_cls, str):
            # 将字符串路径（如 "vllm.worker.GPUWorker"）通过反射解析为真实的 Python 类。
            worker_class: type[WorkerBase] = resolve_obj_by_qualname(
                parallel_config.worker_cls
            )
        else:
            # 安全与规范化约束：不再允许直接传递类对象，必须传全限定名的字符串。
            # 这是为了防止在多进程/Ray 传输配置时发生序列化 (Pickling) 失败。
            raise ValueError(
                "不再支持直接传递 worker_cls 对象。"
                "请将该类保存在独立模块中，并以字符串形式传递其全限定名。"
            )

        # ---------------------------------------------------------
        # 3. 极其强大的“动态混入” (Dynamic Mixin/Inheritance)
        # ---------------------------------------------------------
        # 如果用户配置了 Worker 扩展类 (Worker Extension Class)，
        # vLLM 会在运行时“强行”让原版的 Worker 继承这个扩展类！
        if parallel_config.worker_extension_cls:
            worker_extension_cls = resolve_obj_by_qualname(
                parallel_config.worker_extension_cls
            )
            extended_calls = []
            
            # 如果原版的 worker_class 还没有继承这个扩展类：
            if worker_extension_cls not in worker_class.__bases__:
                # 3.1 冲突检查：确保扩展类里没有覆盖/重写原版类的核心方法
                for attr in dir(worker_extension_cls):
                    if attr.startswith("__"): # 跳过魔术方法（如 __init__）
                        continue
                    assert not hasattr(worker_class, attr), (
                        f"Worker 类 {worker_class} 已经拥有属性 {attr}，"
                        f"这与扩展类 {worker_extension_cls} 发生了冲突。"
                    )
                    # 记录被注入的可调用方法（这些方法后续可以被 collective_rpc 调用）
                    if callable(getattr(worker_extension_cls, attr)):
                        extended_calls.append(attr)
                        
                # 3.2 Python 黑魔法：动态修改类的继承元组 (__bases__)
                # 这等同于在代码里动态地把 class MyWorker(WorkerBase)
                # 变成了 class MyWorker(WorkerBase, WorkerExtension)
                worker_class.__bases__ = worker_class.__bases__ + (
                    worker_extension_cls,
                )
                logger.info(
                    "已将 %s 注入到 %s，扩展的 collective_rpc 调用包含: %s",
                    worker_extension_cls, worker_class, extended_calls,
                )

        # ---------------------------------------------------------
        # 4. 多模态与共享内存锁配置
        # ---------------------------------------------------------
        # shared_worker_lock 用于在单机多卡 (Multiprocessing) 模式下，
        # 防止多个进程同时读写多模态输入（如图片、视频）的共享内存 (shm)。
        shared_worker_lock = kwargs.pop("shared_worker_lock", None)
        if shared_worker_lock is None:
            msg = (
                "执行器中缺失 `shared_worker_lock` 参数。"
                "当 mm_processor_cache_type='shm' (共享内存) 时需要此参数。"
            )
            mm_config = vllm_config.model_config.multimodal_config
            # 如果模型是多模态的且用了共享内存缓存，但没传锁，这是致命错误
            if mm_config and mm_config.mm_processor_cache_type == "shm":
                raise ValueError(msg)
            else:
                logger.warning_once(msg)
            self.mm_receiver_cache = None
        else:
            # 从注册表中获取多模态接收器缓存，并绑定锁
            self.mm_receiver_cache = (
                MULTIMODAL_REGISTRY.worker_receiver_cache_from_config(
                    vllm_config, shared_worker_lock,
                )
            )

        # ---------------------------------------------------------
        # 5. 见证奇迹的时刻：真正的实例化
        # ---------------------------------------------------------
        # 使用上下文管理器，确保在实例化期间，全局能够访问到当前的 vllm_config。
        with set_current_vllm_config(self.vllm_config):
            # 将 kwargs 解包，传入我们刚才动态拼装好的 worker_class 构造函数中。
            # 从这一行代码执行开始，PyTorch/CUDA 的初始化流程才正式打响！
            self.worker = worker_class(**kwargs)

    def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
        kv_cache_config = kv_cache_configs[self.global_rank]
        assert self.vllm_config is not None
        with set_current_vllm_config(self.vllm_config):
            self.worker.initialize_from_config(kv_cache_config)  # type: ignore

    def init_device(self):
        """
        初始化物理设备（GPU/TPU）。
        这个方法属于 WorkerWrapper（工头），它的核心任务是安全地唤醒底层的真正 Worker，
        并确保在唤醒的过程中，底层的 C++/CUDA 代码能随时读取到全局配置。
        """
        
        # 【安全防线 (Sanity Check)】
        # 确保在点火前，引擎（Engine）已经把包含了万事万物的 vllm_config 塞给当前这个 Wrapper 了。
        # 如果是 None，说明系统初始化顺序乱了，直接报错拦截，防止后续出现空指针引发的玄学崩溃。
        assert self.vllm_config is not None

        # 【开启全局虫洞 (Context Manager)】
        # `set_current_vllm_config` 是一个上下文管理器 (Context Manager)。
        # 它的作用是：在当前线程中，临时把 `self.vllm_config` 挂载为一个“全局可见”的变量。
        # 为什么要这样？因为等会儿执行 init_device 时，可能会调用极其底层的 C++ 算子或者第三方库。
        # 如果不用上下文，你就得把 config 当作参数，一层一层地穿透传递下去，代码会非常臃肿。
        with set_current_vllm_config(self.vllm_config):
            # 官方注释：为了在设备初始化期间，让底层的代码能够随时获取到 vLLM 的配置。
            
            # 【真正点火 (Delegation)】
            # Wrapper 本身是个空壳，这里正式把命令下发给真正干活的 self.worker（比如 GPUWorker）。
            # 底层的 worker.init_device() 会真正去执行 `torch.cuda.set_device()` 和 NCCL 的建群动作。
            # `# type: ignore` 是告诉 Mypy 类型检查器：“我知道基类里可能没声明这个方法，但我保证运行时底层的子类一定有，你别给我报黄线警告了。”
            self.worker.init_device()  # type: ignore

    def __getattr__(self, attr: str):
        return getattr(self.worker, attr)

    def _apply_mm_cache(self, scheduler_output: SchedulerOutput) -> None:
        mm_cache = self.mm_receiver_cache
        if mm_cache is None:
            return

        for req_data in scheduler_output.scheduled_new_reqs:
            req_data.mm_features = mm_cache.get_and_update_features(
                req_data.mm_features
            )

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        self._apply_mm_cache(scheduler_output)

        return self.worker.execute_model(scheduler_output)

    def reset_mm_cache(self) -> None:
        mm_receiver_cache = self.mm_receiver_cache
        if mm_receiver_cache is not None:
            mm_receiver_cache.clear_cache()

        self.worker.reset_mm_cache()
