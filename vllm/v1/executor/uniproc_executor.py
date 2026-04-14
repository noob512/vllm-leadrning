# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cached_property
from multiprocessing import Lock
from typing import Any

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.serial_utils import run_method
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class UniProcExecutor(Executor):
    def _init_executor(self) -> None:
        """
        初始化工作节点 (Worker) 并将模型权重加载到 GPU 显存中。
        这是单卡执行器启动时执行的第一波“重体力活”。
        """
        
        # ---------------------------------------------------------
        # 1. 实例化驱动工作节点 (Driver Worker)
        # ---------------------------------------------------------
        # 即使是单卡单进程，vLLM 依然保留了 Worker 的抽象概念。
        # rpc_rank=0 表示这是主节点（也是这里唯一的一个节点）。
        # WorkerWrapperBase 是对真正负责计算的 GPU Worker 的一层包装。
        self.driver_worker = WorkerWrapperBase(rpc_rank=0)
        
        # 获取分布式参数（即使是单卡，为了兼容底层 PyTorch 统一的接口，
        # 依然需要模拟出 rank=0, local_rank=0 的分布式环境上下文）
        distributed_init_method, rank, local_rank = self._distributed_args()
        
        # 打包初始化参数
        kwargs = dict(
            vllm_config=self.vllm_config,             # 全局配置
            local_rank=local_rank,                    # 本机 GPU 编号 (通常是 0)
            rank=rank,                                # 全局进程编号 (通常是 0)
            distributed_init_method=distributed_init_method, # 分布式初始化方法
            is_driver_worker=True,                    # 标记为主驱动节点
            shared_worker_lock=Lock(),                # 线程锁，保证单进程下并发调用的线程安全
        )

        # ---------------------------------------------------------
        # 2. 异步输出处理线程 (Pipeline/Overlap 优化)
        # ---------------------------------------------------------
        self.async_output_thread: ThreadPoolExecutor | None = None
        # max_concurrent_batches > 1 通常发生在开启了流水线并行 (Pipeline Parallelism) 
        # 或者启用了特定的异步调度优化时。
        if self.max_concurrent_batches > 1:
            # 开启一个独立的后台线程。
            # 作用：当 GPU 还在算下一个 batch 时，这个线程可以提前把上一个 batch 
            # 算好的输出 (Logits/Tokens) 从 GPU 搬回 CPU 并进行处理。
            # 这实现了 CPU 和 GPU 的时间重叠 (Overlap)，极大提升了吞吐量。
            self.async_output_thread = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="WorkerAsyncOutput"
            )

        # ---------------------------------------------------------
        # 3. 真正初始化 Worker 并绑定显卡
        # ---------------------------------------------------------
        # 将上面打包好的参数传给 Worker 进行内部状态的初始化
        self.driver_worker.init_worker(all_kwargs=[kwargs])
        
        # 初始化设备：底层会调用 torch.cuda.set_device()，
        # 确保当前的 Python 进程死死绑定在指定的这一块 GPU 上。
        self.driver_worker.init_device()

        # ---------------------------------------------------------
        # 4. 加载模型权重 (The Heavy Lifting)
        # ---------------------------------------------------------
        # 这是一个针对弹性专家并行 (Elastic Expert Parallelism, 常用于 MoE 模型) 的特殊分支。
        # 如果启用了弹性扩展，使用特殊的加载逻辑。
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.driver_worker.elastic_ep_execute("load_model")
        else:
            # 常规路径：从磁盘或 HuggingFace 缓存中读取几十 GB 的权重，
            # 并搬运到刚才绑定的 GPU 显存中。这通常是启动过程中最耗时的一步。
            self.driver_worker.load_model()
            
        # ---------------------------------------------------------
        # 5. 平台相关的底层对齐
        # ---------------------------------------------------------
        # 不同的硬件平台（NVIDIA CUDA vs AMD ROCm）对显存块 (Block) 
        # 有不同的底层对齐要求。这一步会根据当前硬件更新 PagedAttention 的 Block 大小。
        current_platform.update_block_size_for_backend(self.vllm_config)
        
    def _distributed_args(self) -> tuple[str, int, int]:
        """Return (distributed_init_method, rank, local_rank)."""
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
        # set local rank as the device index if specified
        device_info = self.vllm_config.device_config.device.__str__().split(":")
        local_rank = int(device_info[1]) if len(device_info) > 1 else 0
        return distributed_init_method, 0, local_rank

    @cached_property
    def max_concurrent_batches(self) -> int:
        return 2 if self.scheduler_config.async_scheduling else 1

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,       # 要让底层 Worker 执行的方法名（比如 "execute_model"）
        timeout: float | None = None, # 超时时间（本地调用基本用不上）
        args: tuple = (),             # 传递的参数（比如 scheduler_output 施工图纸）
        kwargs: dict | None = None,   # 传递的关键字参数
        non_block: bool = False,      # 是否开启非阻塞（异步）模式
        single_value: bool = False,   # 是否只返回一个单一结果
    ) -> Any:
        """
        单卡环境下的“伪·分布式集体调用”。
        它的核心作用是：抹平单机和多机分布式的 API 差异，并处理底层的异步 CUDA 同步逻辑。
        """
        if kwargs is None:
            kwargs = {}

        # ---------------------------------------------------------
        # 1. 阻塞模式 (Sync Mode)
        # ---------------------------------------------------------
        # 如果要求阻塞（比如初始化模型权重时），直接在主线程执行。
        if not non_block:
            # run_method 本质上就是: getattr(self.driver_worker, method)(*args, **kwargs)
            # 因为只有 1 张卡，所以直接调唯一的那个 driver_worker。
            result = run_method(self.driver_worker, method, args, kwargs)
            # 伪装成分布式的返回值格式：如果不是 single_value，就把它包成列表 [result]
            return result if single_value else [result]

        # ---------------------------------------------------------
        # 2. 非阻塞模式 (Async / Non-blocking Mode) - 核心性能区
        # ---------------------------------------------------------
        try:
            # 即使是非阻塞，我们依然先直接调用本地的 worker 方法。
            # 这里的巧妙之处在于，对于 execute_model，底层是将任务塞进 CUDA Stream（GPU 异步流），
            # 所以 Python 代码这一步几乎是瞬间完成的，返回一个代表“未决状态”的 AsyncModelRunnerOutput 对象。
            result = run_method(self.driver_worker, method, args, kwargs)
            
            # 判断返回值是不是一个“需要等待 GPU 同步的异步对象”
            if isinstance(result, AsyncModelRunnerOutput):
                # 检查系统有没有开启专门用来收尾的后台线程
                if (async_thread := self.async_output_thread) is not None:
                    
                    # 【核心解耦】：将“死等 GPU 算完”这个动作 (result.get_output)，
                    # 扔给后台的专属线程去阻塞等待。
                    # async_thread.submit 会立刻返回一个标准的 Python Future（期权凭证）。
                    # 这样当前的大堂经理 (主线程) 就可以立刻脱身去算 Grammar Mask 语法掩码了！
                    if single_value:
                        return async_thread.submit(result.get_output)

                    def get_output_list() -> list[Any]:
                        return [result.get_output()]

                    return async_thread.submit(get_output_list)
                
                # 如果没开后台线程，那没办法，只能在这个线程里硬等（降级为阻塞）
                result = result.get_output()
                
            # ---------------------------------------------------------
            # 3. 包装 Future 凭证
            # ---------------------------------------------------------
            # 如果调用的不是异步方法，瞬间拿到了真实数据，
            # 为了 API 的统一，依然强行用 Future 包装一层，并立刻设为 set_result。
            future = Future[Any]()
            future.set_result(result if single_value else [result])
            
        except Exception as e:
            # ---------------------------------------------------------
            # 4. 瞬间异常捕获 (Fast-fail)
            # ---------------------------------------------------------
            # 如果上面的调用瞬间报错（比如压根没找到方法，或者传参错误），
            # 不要让程序直接崩溃，而是把报错信息装进 Future 盒子里。
            # 这样上一层 (execute_model) 检查 future.done() 时，就能优雅地把异常抛出来。
            future = Future[Any]()
            future.set_exception(e)
            
        return future

    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        """
        通知所有底层 Worker 执行一次模型的前向传播（Forward Pass）。
        
        参数:
        - scheduler_output: 调度器刚生成的“施工图纸”，包含要计算哪些 Token、用哪些物理显存块。
        - non_block: 是否使用非阻塞模式。如果是 True，函数会立刻返回一个 Future，而不用等 GPU 算完。
        
        返回值:
        - 可能是直接的模型输出 (ModelRunnerOutput)
        - 或者是包含模型输出的异步对象 (Future)
        """

        # ---------------------------------------------------------
        # 1. 发起集体远程调用 (Collective RPC)
        # ---------------------------------------------------------
        # 这是 vLLM 分布式架构的核心！
        # 无论你底层是单机多卡 (mp) 还是多机多卡 (Ray)，collective_rpc 会把这道命令
        # 【同时】广播给当前 Tensor Parallelism (张量并行) 组里的所有 GPU 进程。
        # 
        # 指令内容：调用目标进程的 "execute_model" 方法。
        # 携带参数：把 scheduler_output 传过去。
        output = self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            single_value=True, # 表明所有 worker 算出的是同一个逻辑结果（虽然各算各的碎片），我们只需要拿 Rank 0 的返回值即可。
        )
        
        # ---------------------------------------------------------
        # 2. 非阻塞模式下的快速失败检测 (Fast-fail mechanism)
        # ---------------------------------------------------------
        # 在非阻塞模式下，output 是一个 Future 对象（异步凭证）。
        # 这里进行一次极速的探查：如果这个任务在发出的瞬间就已经完成了（通常是因为抛出了异常退出），
        # 我们就立刻调用 .result()。
        if non_block and output.done():
            # 调用 .result() 的目的是让异常在“这里”就抛出来（in-line raise），
            # 而不是等到外层代码去 await 这个 Future 时才报错，这有助于获得更精准的报错堆栈。
            output.result()
            
        # 把计算结果（或 Future 凭证）交还给外层的 step() 方法
        return output

    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            non_block=non_block,
            single_value=True,
        )

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.collective_rpc("take_draft_token_ids", single_value=True)

    def check_health(self) -> None:
        # UniProcExecutor will always be healthy as long as
        # it's running.
        return

    def shutdown(self) -> None:
        if worker := self.driver_worker:
            worker.shutdown()

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        return True


class ExecutorWithExternalLauncher(UniProcExecutor):
    """An executor that uses external launchers to launch engines,
    specially designed for torchrun-compatible launchers, for
    offline inference with tensor parallelism.

    see https://github.com/vllm-project/vllm/issues/11400 for
    the motivation, and examples/offline_inference/torchrun_example.py
    for the usage example.

    The key idea: although it is tensor-parallel inference, we only
    create one worker per executor, users will launch multiple
    engines with torchrun-compatible launchers, and all these engines
    work together to process the same prompts. When scheduling is
    deterministic, all the engines will generate the same outputs,
    and they don't need to synchronize the states with each other.
    """

    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        assert not envs.VLLM_ENABLE_V1_MULTIPROCESSING, (
            "To get deterministic execution, "
            "please set VLLM_ENABLE_V1_MULTIPROCESSING=0"
        )
        super()._init_executor()

    def _distributed_args(self) -> tuple[str, int, int]:
        # engines are launched in torchrun-compatible launchers
        # so we can use the env:// method.
        # required env vars:
        # - RANK
        # - LOCAL_RANK
        # - MASTER_ADDR
        # - MASTER_PORT
        distributed_init_method = "env://"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        return distributed_init_method, rank, local_rank

    def determine_available_memory(self) -> list[int]:  # 返回单位为 bytes 的列表
        # 作者留下的核心注释：我们需要获取所有显卡（ranks）中，可用显存最小的那个值。
        # we need to get the min across all ranks.
        
        # 【步骤 1：本地极限压测】
        # 调用父类（或底层实现）的方法。
        # 这里就是上节我们提到的：清空本地缓存、跑 Dummy Forward 假推理、查验 max_memory_allocated，
        # 最终算出“我这张卡”当前还剩多少显存（比如 memory = 15 GB）。
        memory = super().determine_available_memory()
        
        # 导入 vLLM 自己封装的分布式状态管理模块
        from vllm.distributed.parallel_state import get_world_group

        # 【步骤 2：获取通讯局域网】
        # 获取专用的 CPU 分布式通信组 (Process Group)。
        # 为什么用 CPU 组而不是 GPU？因为当前我们正在“精确测量 GPU 显存”，
        # 如果此时为了通信又在 GPU 上创建张量，会污染测量结果。所以把通信任务交给 CPU。
        cpu_group = get_world_group().cpu_group
        
        # 【步骤 3：数据打包准备通信】
        # 把刚刚算出来的、自己这张卡的显存剩余量（Python 的 int 数字），
        # 打包成一个 PyTorch 的 CPU 张量 (Tensor)，因为底层的分布式通信只认 Tensor。
        memory_tensor = torch.tensor([memory], device="cpu", dtype=torch.int64)
        
        # 【步骤 4：核心魔法！分布式规约 (All-Reduce)】
        # 这是一行极其强大的 PyTorch 分布式操作：
        # 1. 它会让当前所有的 Worker 进程在这里“集合（Barrier）”停下。
        # 2. 把大家手里的 memory_tensor 全部收上来。
        # 3. 执行 ReduceOp.MIN 操作（挑出所有人里面最小的那个数字）。
        # 4. 把挑出来的这个最小值，【重新覆盖】写入到每个 Worker 自己的 memory_tensor 中。
        dist.all_reduce(memory_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        
        # 【步骤 5：解包返回】
        # 经过上一步，此时每个 Worker 手里的 memory_tensor.item() 都变成了那个“全网最低值”。
        # 拆包成 Python 数字，放在列表里返回。
        return [memory_tensor.item()]
