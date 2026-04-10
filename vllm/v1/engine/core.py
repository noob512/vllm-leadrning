# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import queue
import signal
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Generator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from enum import IntEnum
from functools import partial
from inspect import isclass, signature
from logging import DEBUG
from multiprocessing.queues import Queue
from typing import Any, TypeVar, cast

import msgspec
import zmq

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.tracing import instrument, maybe_init_worker_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import (
    EEP_NOTIFICATION_CALL_ID,
    EEPNotificationType,
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
    PauseMode,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
    UtilityResult,
)
from vllm.v1.engine.tensor_ipc import TensorIpcReceiver
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    SignalCallback,
    get_device_indices,
)
from vllm.v1.executor import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import compute_iteration_details
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar("_R")  # Return type for collective_rpc

class EngineCore:

    """Inner loop of vLLM's Engine."""
    def __init__(
            self,
            vllm_config: VllmConfig,
            executor_class: type[Executor],
            log_stats: bool,
            executor_fail_callback: Callable | None = None,
            include_finished_set: bool = False,
        ):
        # ---------------------------------------------------------
        # 1. 插件与基础配置
        # ---------------------------------------------------------
        # 加载可能存在的自定义插件（如自定义的注意力机制或算子）
        from vllm.plugins import load_general_plugins
        load_general_plugins()

        self.vllm_config = vllm_config
        # 如果是主节点 (rank_local == 0)，打印初始化日志
        if not vllm_config.parallel_config.data_parallel_rank_local:
            logger.info(
                "Initializing a V1 LLM engine (v%s) with config: %s",
                VLLM_VERSION, vllm_config,
            )
        self.log_stats = log_stats

        # ---------------------------------------------------------
        # 2. 核心大件之一：实例化模型执行器 (Model Executor)
        # ---------------------------------------------------------
        # executor_class 会根据配置实例化对应的执行器（如单卡、NCCL 多卡或 Ray 执行器）。
        # 这里会在后台真正去实例化 PyTorch 模型并将几十 GB 的权重加载到 GPU 上！
        self.model_executor = executor_class(vllm_config)
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(executor_fail_callback)

        self.available_gpu_memory_for_kv_cache = -1

        # 弹性扩展预处理（处理动态增减显卡节点的情况）
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self._eep_scale_up_before_kv_init()

        # ---------------------------------------------------------
        # 3. 核心大件之二：初始化 KV Cache (显存规划)
        # ---------------------------------------------------------
        # 【极其重要】：在这里，vLLM 会用“假数据”跑一次前向传播（Profile），
        # 测试出模型权重到底占了多少显存，然后把剩余的显存全部切分成 PagedAttention 需要的“块 (Blocks)”。
        kv_cache_config = self._initialize_kv_caches(vllm_config)
        
        # 结构化输出管理器（用于强制模型按照 JSON Schema 或正则格式输出）
        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # ---------------------------------------------------------
        # 4. 核心大件之三：初始化调度器 (Scheduler)
        # ---------------------------------------------------------
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

        # 针对无 KV Cache 模型（如仅编码器模型）的特殊处理：禁用分块预填充 (Chunked Prefill)
        if len(kv_cache_config.kv_cache_groups) == 0:  
            if vllm_config.scheduler_config.enable_chunked_prefill:
                logger.warning("Disabling chunked prefill for model without KVCache")
                vllm_config.scheduler_config.enable_chunked_prefill = False

        # 计算调度器的块大小（考虑到可能存在的张量并行和上下文并行）
        scheduler_block_size = (
            vllm_config.cache_config.block_size
            * vllm_config.parallel_config.decode_context_parallel_size
            * vllm_config.parallel_config.prefill_context_parallel_size
        )

        # 正式实例化调度器：它是决定“谁能用显卡、谁要被踢出显卡”的最高指挥官
        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=include_finished_set,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
        )
        
        # 记录是否开启推测解码 (Speculative Decoding)
        self.use_spec_decode = vllm_config.speculative_config is not None
        
        # ---------------------------------------------------------
        # 5. 高阶通信与多模态设置
        # ---------------------------------------------------------
        # 分离式推理架构 (Disaggregated Serving)：用于跨机器传递 KV Cache
        if self.scheduler.connector is not None:  
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  

        # 多模态接收器缓存（用于接收图像/视频的 Tensor）
        mm_registry = MULTIMODAL_REGISTRY
        self.mm_receiver_cache = mm_registry.engine_receiver_cache_from_config(vllm_config)

        # 收集各个 Worker 的 KV Connector 握手元数据（用于多机集群）
        kv_connector = self.scheduler.get_kv_connector()
        if kv_connector is not None:
            xfer_handshake_metadata = self.model_executor.get_kv_connector_handshake_metadata()
            if xfer_handshake_metadata:
                content: dict[int, Any] = {}
                for worker_dict in xfer_handshake_metadata:
                    if worker_dict is not None:
                        content.update(worker_dict)
                kv_connector.set_xfer_handshake_metadata(content)

        # ---------------------------------------------------------
        # 6. 流水线并行与异步计算 (Pipeline Parallelism & Async)
        # ---------------------------------------------------------
        # 如果模型被切分到多台机器的先后顺序上（流水线并行），为了不让机器闲着（消除气泡），
        # 引入异步的 batch_queue。
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: deque | None = None
        if self.batch_queue_size > 1:
            logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        self.is_ec_consumer = (vllm_config.ec_transfer_config is None
                            or vllm_config.ec_transfer_config.is_ec_consumer)
        self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"

        # ---------------------------------------------------------
        # 7. 前缀缓存 (Prefix Caching)
        # ---------------------------------------------------------
        # 开启后，系统会为每一个 System Prompt 或长前缀计算 Hash 值。
        # 下次同样的 Prompt 进来，直接复用显存里的 KV Cache，免去计算。
        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
        if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
            caching_hash_fn = get_hash_fn_by_name(vllm_config.cache_config.prefix_caching_hash_algo)
            init_none_hash(caching_hash_fn)
            self.request_block_hasher = get_request_block_hasher(scheduler_block_size, caching_hash_fn)

        # 动态指定单步执行函数（是否有流水线队列）
        self.step_fn = self.step if self.batch_queue is None else self.step_with_batch_queue
        self.async_scheduling = vllm_config.scheduler_config.async_scheduling

        self.aborts_queue = queue.Queue[list[str]]()
        self._idle_state_callbacks: list[Callable] = []

        # ---------------------------------------------------------
        # 8. 极致性能优化 (Python GC 黑科技)
        # ---------------------------------------------------------
        # 冻结 Python 垃圾回收器的老生代堆 (Freeze GC Heap)。
        # 因为在初始化阶段，我们创建了海量的内部对象（如几十万个 KV Block 对象）。
        # 如果不冻结，Python 的垃圾回收器会定期扫描这些永远不会被释放的对象，
        # 导致生成 Token 时出现毫秒级的卡顿（Latency Spikes）。冻结后可大幅降低卡顿。
        freeze_gc_heap()
        maybe_attach_gc_debug_callback()
        
        # 缓存环境变量，避免运行时高频调用 os.environ（这是慢速系统调用）
        enable_envs_cache()

    @instrument(span_name="Prepare model")
    def _initialize_kv_caches(self, vllm_config: VllmConfig) -> KVCacheConfig:
        start = time.time()

        # Get all kv cache needed by the model
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
                # NOTE(yongji): should already be set
                # during _eep_scale_up_before_kv_init
                assert self.available_gpu_memory_for_kv_cache > 0
                available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                    kv_cache_specs
                )
            else:
                # Profiles the peak memory usage of the model to determine how
                # much memory can be allocated for kv cache.
                available_gpu_memory = self.model_executor.determine_available_memory()
                self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = [0] * len(kv_cache_specs)

        assert len(kv_cache_specs) == len(available_gpu_memory)

        # Track max_model_len before KV cache config to detect auto-fit changes
        max_model_len_before = vllm_config.model_config.max_model_len

        kv_cache_configs = get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_gpu_memory
        )

        # If auto-fit reduced max_model_len, sync the new value to workers.
        # This is needed because workers were spawned before memory profiling
        # and have the original (larger) max_model_len cached.
        max_model_len_after = vllm_config.model_config.max_model_len
        if max_model_len_after != max_model_len_before:
            self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
        if kv_cache_groups:
            vllm_config.cache_config.block_size = min(
                g.kv_cache_spec.block_size for g in kv_cache_groups
            )

        vllm_config.validate_block_size()

        # Initialize kv cache and warmup the execution
        self.model_executor.initialize_from_config(kv_cache_configs)

        elapsed = time.time() - start
        logger.info_once(
            "init engine (profile, create kv cache, warmup model) took %.2f seconds",
            elapsed,
            scope="local",
        )
        return scheduler_kv_cache_config

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_executor.supported_tasks

    def add_request(self, request: Request, request_wave: int = 0):
        """
        将经过反序列化和预处理的请求，正式添加到底层调度器中。
        
        参数解析：
        - request: 包含 Token ID、生成参数等所有核心信息的请求对象。
        - request_wave: 这是一个针对数据并行 (DP) 或 混合专家模型 (MoE) 的高级参数。
          在大规模分布式推理中，任务可能是一波一波 (wave) 到来的，这个参数用来对齐不同卡上的波次。
        """
        
        # ---------------------------------------------------------
        # 1. 最基础的身份校验
        # ---------------------------------------------------------
        # 确保订单号必须是字符串类型。
        # 分布式系统里，ID 的类型不一致会导致非常隐蔽的 Hash 查找失败或内存泄漏Bug。
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )

        # ---------------------------------------------------------
        # 2. 判别“非生成类”任务 (Pooling Tasks Validations)
        # ---------------------------------------------------------
        # 大模型不光能“生成文本 (Generation)”，还能用来提取“向量特征 (Embedding)” 
        # 或进行“文本分类 (Classification)”。这些统称为 Pooling（池化）任务。
        if pooling_params := request.pooling_params:
            # 去系统里查一下：当前挂载的模型，到底支不支持用户想要的池化任务？
            # 比如，你加载了一个纯对话的 Llama 3 模型，用户却非要提取 Embedding 向量，这就属于不支持。
            supported_pooling_tasks = [
                task for task in self.get_supported_tasks() if task in POOLING_TASKS
            ]

            # 如果不支持，立刻打回，拒绝接单，防止 GPU 跑到一半报错崩溃。
            if pooling_params.task not in supported_pooling_tasks:
                raise ValueError(
                    f"Unsupported task: {pooling_params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

        # ---------------------------------------------------------
        # 3. 前沿架构校验：KV Cache 分离架构 (KV Transfer Validation)
        # ---------------------------------------------------------
        # 【黑科技预警】：KV Transfer（KV 缓存转移）是 vLLM 非常前沿的功能。
        # 在极致优化的超大集群中，我们会把“阅读长文本 (Prefill)”和“逐字生成 (Decode)”拆到不同的机器上跑。
        # 此时，Prefill 机器算完的 KV Cache，需要通过网络直接传给 Decode 机器。
        # 这就需要用到 kv_transfer_params 和底层的 KVConnector 组件。
        if request.kv_transfer_params is not None and (
            not self.scheduler.get_kv_connector()
        ):
            # 如果用户传了 KV 转移参数，但当前这台机器根本没开启这个功能组件：
            # 这里不报错，只是发个警告，并默默把这个高级参数忽略掉（降级为普通单机运算）。
            logger.warning(
                "Got kv_transfer_params, but no KVConnector found. "
                "Disabling KVTransfer for this request."
            )

        # ---------------------------------------------------------
        # 4. 终极一跃：推入调度器
        # ---------------------------------------------------------
        # 所有的安检全部通过！
        # 这个任务被正式交给了 Scheduler（调度器）。
        # 调度器会立刻把它塞进 `waiting`（等待区）队列中。
        # 在下一次 `_process_engine_step()`（主厨开火）时，调度器就会根据显存容量，
        # 决定要不要把它捞出来丢进 GPU。
        self.scheduler.add_request(request)

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    @contextmanager
    def log_error_detail(self, scheduler_output: SchedulerOutput):
        """Execute the model and log detailed info on failure."""
        try:
            yield
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(
                self.vllm_config, scheduler_output, self.scheduler.make_stats()
            )
            raise err

    @contextmanager
    def log_iteration_details(self, scheduler_output: SchedulerOutput):
        if not self.vllm_config.observability_config.enable_logging_iteration_details:
            yield
            return
        self._iteration_index = getattr(self, "_iteration_index", 0)
        iteration_details = compute_iteration_details(scheduler_output)
        before = time.monotonic()
        yield
        logger.info(
            "".join(
                [
                    "Iteration(",
                    str(self._iteration_index),
                    "): ",
                    str(iteration_details.num_ctx_requests),
                    " context requests, ",
                    str(iteration_details.num_ctx_tokens),
                    " context tokens, ",
                    str(iteration_details.num_generation_requests),
                    " generation requests, ",
                    str(iteration_details.num_generation_tokens),
                    " generation tokens, iteration elapsed time: ",
                    format((time.monotonic() - before) * 1000, ".2f"),
                    " ms",
                ]
            )
        )
        self._iteration_index += 1

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """
        核心调度、执行与产出方法。
        
        返回值:
        - outputs: 一个字典，包含各个客户端/请求生成的最新 Token 和状态。
        - model_executed: 布尔值，表示这次 Step 是否真正在 GPU 上算出了 Token。
        """

        # ---------------------------------------------------------
        # 1. 守门员：空载检查
        # ---------------------------------------------------------
        # 如果调度器里连一个处于 unfinished (未完成) 状态的请求都没有，
        # 直接返回空结果，避免白白调用昂贵的底层函数。
        if not self.scheduler.has_requests():
            return {}, False

        # ---------------------------------------------------------
        # 2. 调度器 (The Brain): 决定谁上车，并分配显存
        # ---------------------------------------------------------
        # 这是 PagedAttention 发挥作用的地方。
        # schedule() 会遍历等待队列和运行队列，决定：
        # - 哪些请求可以进入这一批次 (Batch) 进行计算？
        # - 它们各自的 KV Cache 块 (Blocks) 在显存的物理地址是什么？
        # 产出 scheduler_output，这是给 GPU 的“施工图纸”。
        scheduler_output = self.scheduler.schedule()

        # ---------------------------------------------------------
        # 3. 执行器 (The Muscle): 将图纸扔给 GPU 开始前向传播
        # ---------------------------------------------------------
        # 将施工图交到底层的 CUDA 算子。
        # 【关键优化：non_block=True】
        # 这里是非阻塞调用，它把矩阵乘法任务推入 GPU 流 (CUDA Stream) 后，
        # 立即返回一个 Future 对象，而不会让 CPU 在这里死等 GPU 算完。
        future = self.model_executor.execute_model(scheduler_output, non_block=True)

        # ---------------------------------------------------------
        # 4. 结构化生成支持 (Grammar Masking) -> 与 GPU 计算并行！
        # ---------------------------------------------------------
        # 就在 GPU 轰鸣着计算矩阵的同时，CPU 也没闲着。
        # 它利用这段时间差，计算那些要求输出 JSON 或正则格式的请求的“语法掩码”。
        # 把不允许生成的 Token 的概率强制设为 0。
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)

        # ---------------------------------------------------------
        # 5. 等待 GPU 交卷 (Blocking & Result Collection)
        # ---------------------------------------------------------
        # 使用上下文管理器捕获在等待期间可能发生的任何 CUDA OOM 或计算错误。
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            # CPU 在这里正式挂起，等待 GPU 计算完毕并把 Logits/Tokens 传回内存。
            model_output = future.result()
            
            # 某些特定的架构（如完全分离的模型）可能只返回了 Logits 而没采样。
            # 如果是这种情况，就在这里带着语法掩码进行最终的 Token 采样 (Sampling)。
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)

        # ---------------------------------------------------------
        # 6. 处理突发事件：取消请求 (Abort Check)
        # ---------------------------------------------------------
        # 【防御性编程】：在 GPU 算这几十毫秒的时间里，用户可能按下了网页上的“停止生成”。
        # 所以在把算出来的 Token 保存之前，必须先检查一下 `aborts_queue`。
        # 如果有被取消的请求，赶紧把它们的状态标记为废弃，避免浪费资源。
        self._process_aborts_queue()

        # ---------------------------------------------------------
        # 7. 闭环反馈：更新请求状态与显存
        # ---------------------------------------------------------
        # 把刚算出来的新 Token (model_output) 喂回给调度器。调度器会做几件事：
        # - 把新 Token 拼接到请求的历史序列中。
        # - 检查这个 Token 是不是 </s> (EOS 结束符)，如果是，标记请求完成，并释放其占用的 KV Cache 显存。
        # - 组装最终的 engine_core_outputs，准备发给前台。
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # 返回结果，并告诉上层这次是否真正调度了 Token 进行计算
        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    def post_step(self, model_executed: bool) -> None:
        # When using async scheduling we can't get draft token ids in advance,
        # so we update draft token ids in the worker process and don't
        # need to update draft token ids here.
        if not self.async_scheduling and self.use_spec_decode and model_executed:
            # Take the draft token ids.
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """

        batch_queue = self.batch_queue
        assert batch_queue is not None

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule()
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
            if self.is_ec_consumer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if self.is_pooling_model or not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                batch_queue.appendleft((future, scheduler_output, exec_future))
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, True

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                # None from sample_tokens() implies that the original execute_model()
                # call failed - raise that exception.
                exec_model_fut.result()
                raise RuntimeError("unexpected error")

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # If we are doing speculative decoding with structured output,
            # we need to get the draft token ids from the prior step before
            # we can compute the grammar bitmask for the deferred request.
            if self.use_spec_decode:
                draft_token_ids = self.model_executor.take_draft_token_ids()
                assert draft_token_ids is not None
                # Update the draft token ids in the scheduler output to
                # filter out the invalid spec tokens, which will be padded
                # with -1 and skipped by the grammar bitmask computation.
                self.scheduler.update_draft_token_ids_in_output(
                    draft_token_ids, deferred_scheduler_output
                )
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

        return engine_core_outputs, model_executed

    def _process_aborts_queue(self):
        if not self.aborts_queue.empty():
            request_ids = []
            while not self.aborts_queue.empty():
                ids = self.aborts_queue.get_nowait()
                # Should be a list here, but also handle string just in case.
                request_ids.extend((ids,) if isinstance(ids, str) else ids)
            # More efficient to abort all as a single batch.
            self.abort_requests(request_ids)

    def shutdown(self):
        self.structured_output_manager.clear_backend()
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        self.model_executor.profile(is_start, profile_prefix)

    def reset_mm_cache(self):
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the multi-modal cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # The cache either exists in EngineCore or WorkerWrapperBase
        if self.mm_receiver_cache is not None:
            self.mm_receiver_cache.clear_cache()

        self.model_executor.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings computed with old weights are not reused.
        Clears both the scheduler's cache manager and the GPU model runner's cache.
        """
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the encoder cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # Reset the scheduler's encoder cache manager (logical state)
        self.scheduler.reset_encoder_cache()
        # Reset the GPU model runner's encoder cache (physical storage)
        self.model_executor.reset_encoder_cache()

    def _reset_caches(self, reset_running_requests=True) -> None:
        self.reset_prefix_cache(reset_running_requests=reset_running_requests)
        self.reset_mm_cache()
        self.reset_encoder_cache()

    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None:
        """Pause generation; behavior depends on mode.

        All pause modes queue new adds -- "abort" and "keep" skip step();
        "wait" allows step() so in-flight requests can drain.

        - ``abort``: Set PAUSED_NEW, abort all requests, wait for abort
          outputs to be sent (when running with output_queue), optionally
          clear caches, then complete the returned Future.
        - ``wait``: Set PAUSED_NEW (queue adds, keep stepping); when drained,
          optionally clear caches, then complete the returned Future.
        - ``keep``: Set PAUSED_ALL; return a Future that completes when the
          output queue is empty.
        """
        if mode not in ("keep", "abort", "wait"):
            raise ValueError(f"Invalid pause mode: {mode}")
        if mode == "wait":
            raise ValueError("'wait' mode can't be used in inproc-engine mode")

        if mode == "abort":
            self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        self.scheduler.set_pause_state(pause_state)
        if clear_cache:
            self._reset_caches()

        return None

    def resume_scheduler(self) -> None:
        """Resume the scheduler and flush any requests queued while paused."""
        self.scheduler.set_pause_state(PauseState.UNPAUSED)

    def is_scheduler_paused(self) -> bool:
        """Return whether the scheduler is in any pause state."""
        return self.scheduler.pause_state != PauseState.UNPAUSED

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None | Future:
        """Put the engine to sleep at the specified level.

        Args:
            level: Sleep level.
                - Level 0: Pause scheduling only. Requests are still accepted
                           but not processed. No GPU memory changes.
                - Level 1: Offload model weights to CPU, discard KV cache.
                - Level 2: Discard all GPU memory.
            mode: Pause mode - how to deal with any existing requests, see
                documentation of pause_scheduler method.
        """

        # Pause scheduler before sleeping.
        clear_prefix_cache = level >= 1
        pause_future = self.pause_scheduler(mode=mode, clear_cache=clear_prefix_cache)
        if level < 1:
            return pause_future

        # Level 1+: Delegate to executor for GPU memory management
        model_executor = self.model_executor
        if pause_future is None:
            model_executor.sleep(level)
            return None

        future = Future[Any]()

        def pause_complete(f: Future):
            try:
                f.result()  # propagate any exception
                future.set_result(model_executor.sleep(level))
            except Exception as e:
                future.set_exception(e)

        logger.info("Waiting for in-flight requests to complete before sleeping...")
        pause_future.add_done_callback(pause_complete)
        return future

    def wake_up(self, tags: list[str] | None = None):
        """Wake up the engine from sleep.

        Args:
            tags: Tags to wake up. Use ["scheduling"] for level 0 wake up.
        """
        if tags is not None and "scheduling" in tags:
            # Remove "scheduling" from tags if there are other tags to process.
            tags = [t for t in tags if t != "scheduling"]

        if tags is None or tags:
            self.model_executor.wake_up(tags)

        # Resume scheduling (applies to all levels)
        self.resume_scheduler()

    def is_sleeping(self) -> bool:
        """Check if engine is sleeping at any level."""
        return self.is_scheduler_paused() or self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.model_executor.save_sharded_state(
            path=path, pattern=pattern, max_size=max_size
        )

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args, kwargs)

    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # Note on thread safety: no race condition.
        # `mm_receiver_cache` is reset at the end of LLMEngine init,
        # and will only be accessed in the input processing thread afterwards.
        if self.mm_receiver_cache is not None and request.mm_features:
            request.mm_features = self.mm_receiver_cache.get_and_update_features(
                request.mm_features
            )

        req = Request.from_engine_core_request(request, self.request_block_hasher)
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave

    def _eep_scale_up_before_kv_init(self):
        raise NotImplementedError

    def _eep_send_engine_core_notification(
        self,
        notification_type: EEPNotificationType,
        vllm_config: VllmConfig | None = None,
    ):
        raise NotImplementedError


class EngineShutdownState(IntEnum):
    RUNNING = 0
    REQUESTED = 1
    SHUTTING_DOWN = 2


class EngineCoreProc(EngineCore):
    """
    EngineCore 的 ZMQ 包装类，负责在后台进程中运行推理引擎。
    它不仅持有模型，还持有负责与主进程通信的 ZMQ 套接字。
    """

    # 当引擎崩溃时发送的特殊字节码
    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"
    addresses: EngineZmqAddresses

    @instrument(span_name="EngineCoreProc init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
        *,
        engine_index: int = 0,
    ):
        # 1. 【核心缓冲区】：建立内部队列
        # input_queue: 存放从主进程/协调器收到的请求
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        # output_queue: 存放 GPU 算完、准备发回给主进程的结果
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()

        # 如果 Executor（执行器）崩了，立刻往输入队列塞一个失败信号，触发自杀逻辑
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
        )

        self.engine_index = engine_index
        # 身份标识：将 index 转为 2 字节的小端序，用于 ZMQ 路由识别
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False
        self.shutdown_state = EngineShutdownState.RUNNING

        # 2. 【多模态零拷贝】：设置张量 IPC 接收器
        self.tensor_ipc_receiver = None
        if tensor_queue is not None:
            # 如果有共享内存队列，专门启动一个接收器处理大图/视频张量
            self.tensor_ipc_receiver = TensorIpcReceiver(tensor_queue)
            logger.info("Using tensor IPC queue for multimodal tensor sharing")

        # 3. 【建立联系】：执行启动握手
        # 这里会阻塞，直到通过 handshake_address 拿到该引擎专属的 ZMQ 通信地址
        with self._perform_handshakes(
            handshake_address,
            identity,
            local_client,
            vllm_config,
            client_handshake_address,
        ) as addresses:
            
            # 判断是否存在数据并行协调器（DP Coordinator）
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address
            )
            
            # 是否由内部协调器负责负载均衡（LB）
            internal_dp_balancing = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb
            )
            # 只有在内部或混合负载均衡模式下，才向协调器上报自己的队列长度
            self.publish_dp_lb_stats = internal_dp_balancing

            self.addresses = addresses
            self.process_input_queue_block = True
            
            # 如果是弹性并行扩展（MoE 动态加卡），先发个通知说我准备好了
            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
                self._eep_send_engine_core_notification(
                    EEPNotificationType.NEW_CORE_ENGINES_INIT_READY,
                    vllm_config=vllm_config,
                )
            
            # 初始化分布式环境（如 NCCL 等）
            self._init_data_parallel(vllm_config)

            # 4. 【正式变身】：调用父类 EngineCore 初始化模型
            # 这一步会触发加载权重、初始化显存块等极耗时的重体力活
            super().__init__(
                vllm_config,
                executor_class,
                log_stats,
                executor_fail_callback,
                internal_dp_balancing,
            )

            # 5. 【后台流水线】：开启 I/O 线程
            # 作用：由于 ZMQ 处理和序列化会占用 CPU，开启独立线程可以实现：
            #      GPU 算当前 batch 时，CPU 在反序列化下一个请求。
            ready_event = threading.Event()
            
            # 线程 A：输入线程。负责从 ZMQ 读取 Request，反序列化后塞进 self.input_queue
            input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(
                    addresses.inputs,
                    addresses.coordinator_input,
                    identity,
                    ready_event,
                ),
                daemon=True,
            )
            input_thread.start()

            # 线程 B：输出线程。负责从 self.output_queue 拿结果，序列化后通过 ZMQ 发回
            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(
                    addresses.outputs,
                    addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            self.output_thread.start()

            # 6. 【最后的验证】：等待协调器就绪信号
            # 除非收到 READY 消息，否则不完成初始化
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    # 如果后台线程直接崩了（如端口占用），抛出错误
                    raise RuntimeError("Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        """
        Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal load-balancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external or hybrid load-balancing, two handshakes are
        performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 and colocated engines themselves which
        don't require the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        input_ctx = zmq.Context()
        is_local = local_client and client_handshake_address is None
        headless = not local_client
        handshake = self._perform_handshake(
            input_ctx,
            handshake_address,
            identity,
            is_local,
            headless,
            vllm_config,
            vllm_config.parallel_config,
        )
        if client_handshake_address is None:
            with handshake as addresses:
                yield addresses
        else:
            assert local_client
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, True, False, vllm_config
            )
            with handshake as addresses, local_handshake as client_addresses:
                addresses.inputs = client_addresses.inputs
                addresses.outputs = client_addresses.outputs
                yield addresses

        # Update config which may have changed from the handshake
        vllm_config.__post_init__()

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        vllm_config: VllmConfig,
        parallel_config_to_update: ParallelConfig | None = None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        with make_zmq_socket(
            ctx,
            handshake_address,
            zmq.DEALER,
            identity=identity,
            linger=5000,
            bind=False,
        ) as handshake_socket:
            # Register engine with front-end.
            addresses = self.startup_handshake(
                handshake_socket, local_client, headless, parallel_config_to_update
            )
            yield addresses

            # Send ready message.
            num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks
            # We pass back the coordinator stats update address here for the
            # external LB case for our colocated front-end to use (coordinator
            # only runs with rank 0).
            dp_stats_address = self.frontend_stats_publish_address

            # Include config hash for DP configuration validation
            ready_msg = {
                "status": "READY",
                "local": local_client,
                "headless": headless,
                "num_gpu_blocks": num_gpu_blocks,
                "dp_stats_address": dp_stats_address,
            }
            if vllm_config.parallel_config.data_parallel_size > 1:
                ready_msg["parallel_config_hash"] = (
                    vllm_config.parallel_config.compute_hash()
                )

            handshake_socket.send(msgspec.msgpack.encode(ready_msg))

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: ParallelConfig | None = None,
    ) -> EngineZmqAddresses:
        # Send registration message.
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "HELLO",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

        # Receive initialization message.
        logger.debug("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError(
                "Did not receive response from front-end "
                f"process within {HANDSHAKE_TIMEOUT_MINS} "
                f"minutes"
            )
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        logger.debug("Received init message: %s", init_message)

        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """
        在后台进程中启动 EngineCore 的忙循环（Busy Loop）。
        这是子进程真正的入口函数。
        """

        # ---------------------------------------------------------
        # 1. 序列化准备
        # ---------------------------------------------------------
        # 确保在进程间传输 Transformer 配置时能够正确地按值序列化。
        maybe_register_config_serialize_by_value()

        engine_core: EngineCoreProc | None = None
        signal_callback: SignalCallback | None = None
        
        try:
            # 2. 基础配置提取
            vllm_config: VllmConfig = kwargs["vllm_config"]
            parallel_config: ParallelConfig = vllm_config.parallel_config
            
            # 判断是否处于数据并行 (DP) 模式
            data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
            
            # ---------------------------------------------------------
            # 3. 设置进程身份（在 ps 或 top 命令中可见）
            # ---------------------------------------------------------
            if data_parallel:
                # 绑定本地显卡 Rank 索引
                parallel_config.data_parallel_rank_local = local_dp_rank
                process_title = f"EngineCore_DP{dp_rank}"
            else:
                process_title = "EngineCore"
            
            set_process_title(process_title) # 修改进程名，方便系统监控
            maybe_init_worker_tracer("vllm.engine_core", "engine_core", process_title)
            decorate_logs() # 装饰日志输出，让日志带上进程身份前缀

            # ---------------------------------------------------------
            # 4. KV Cache 传输优化配置
            # ---------------------------------------------------------
            # 如果开启了跨进程 KV 传输，需要为每个 DP Rank 生成唯一的 Engine ID，防止冲突。
            if data_parallel and vllm_config.kv_transfer_config is not None:
                vllm_config.kv_transfer_config.engine_id = (
                    f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
                )
                logger.debug("将 kv_transfer_config.engine_id 设置为 %s",
                             vllm_config.kv_transfer_config.engine_id)

            # ---------------------------------------------------------
            # 5. 实例化真正的推理核心
            # ---------------------------------------------------------
            parallel_config.data_parallel_index = dp_rank
            
            # 场景 A：如果是混合专家模型 (MoE) 且有数据并行，使用特殊的 DPEngineCoreProc。
            # 这涉及到跨卡的专家协作同步。
            if data_parallel and vllm_config.model_config.is_moe:
                parallel_config.data_parallel_rank = dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                # 场景 B：非 MoE 的 DP Rank 在计算上是独立的，按单副本处理以简化逻辑。
                parallel_config.data_parallel_size = 1
                parallel_config.data_parallel_size_local = 1
                parallel_config.data_parallel_rank = 0
                # 创建标准的引擎核心
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

            assert engine_core is not None

            # ---------------------------------------------------------
            # 6. 优雅停机逻辑：信号处理
            # ---------------------------------------------------------
            def wakeup_engine():
                """
                当收到停机请求时，强制唤醒正在休眠等待任务的引擎。
                """
                # 向输入队列发送一个 WAKEUP 信号，打破 recv() 的阻塞。
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            # 使用 SignalCallback 包装，确保信号处理的线程安全
            signal_callback = SignalCallback(wakeup_engine)

            def signal_handler(signum, frame):
                # 标记状态为“已请求停机”
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()

            # 注册系统信号：Ctrl+C (SIGINT) 和 kill (SIGTERM)
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            # ---------------------------------------------------------
            # 7. 激活心脏：进入忙循环
            # ---------------------------------------------------------
            # 这是一个阻塞调用，直到引擎关闭。
            # 它会不停地：监听请求 -> 调度 -> 推理 -> 返回结果。
            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore 正在正常退出。")
            raise
        except Exception as e:
            # 8. 灾难通报
            if engine_core is None:
                logger.exception("EngineCore 启动失败。")
            else:
                logger.exception("EngineCore 遇到致命错误。")
                # 【关键】：如果自己要死了，一定要给主进程发个“丧信”，防止主进程在那死等。
                engine_core._send_engine_dead()
            raise e
        finally:
            # 9. 资源清理
            # 恢复默认信号处理，停止回调，关闭引擎
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if engine_core is not None:
                engine_core.shutdown()

    def _init_data_parallel(self, vllm_config: VllmConfig):
        pass

    def has_work(self) -> bool:
        """Returns true if the engine should be stepped."""
        return (
            self.engines_running
            or self.scheduler.has_requests()
            or bool(self.batch_queue)
        )

    def is_running(self) -> bool:
        """Returns true if shutdown has not been requested."""
        return self.shutdown_state == EngineShutdownState.RUNNING

    def run_busy_loop(self):
        """
        EngineCore 的核心忙循环（Busy Loop）。
        这是 GPU 进程的‘主引擎’，负责协调指令接收与模型推理。
        """
        
        # ---------------------------------------------------------
        # 循环条件：_handle_shutdown()
        # ---------------------------------------------------------
        # 这是一个状态检查函数。它会检查主进程是否发来了关闭信号。
        # 如果没有停机请求，它返回 True，循环继续；
        # 一旦检测到停机，它会执行清理逻辑并返回 False，打破循环。
        while self._handle_shutdown():
            
            # ---------------------------------------------------------
            # 1) 处理输入队列 (The Listener)
            # ---------------------------------------------------------
            # 这一步是‘听指令’。
            # 它会检查 ZMQ Socket 或 IPC 队列，看看主进程有没有发来：
            # - 新的推理请求 (Add Request)
            # - 取消某个请求 (Abort Request)
            # - 性能分析指令 (Profile)
            # 它会将这些请求放入引擎内部的待处理池（Scheduler）中。
            self._process_input_queue()

            # ---------------------------------------------------------
            # 2) 执行推理步进 (The Executor)
            # ---------------------------------------------------------
            # 这一步是‘干重活’。
            # 这是真正的 GPU 计算环节，包含以下核心逻辑：
            # - 调度 (Scheduler)：决定现在该处理哪些 Token，KV Cache 够不够。
            # - 执行 (Model Forward)：调用 CUDA 算子进行矩阵乘法。
            # - 收集结果：拿到生成的 Token ID，并通过 Output Socket 发回给主进程。
            # 注意：在 LLM 推理中，每执行一次这个 step，通常只产生 1 个新的 Token。
            self._process_engine_step()

        # ---------------------------------------------------------
        # 退出循环：抛出系统退出异常
        # ---------------------------------------------------------
        # 当 _handle_shutdown 返回 False 时，循环结束。
        # 这里显式抛出 SystemExit，告知操作系统该 Python 进程已完成使命，正常关闭。
        raise SystemExit

    def _process_input_queue(self):
        """
        处理输入队列：当需要执行一个引擎步进（Engine Step）时退出该方法。
        简单说：这里负责接收‘加任务’或‘停任务’的指令。
        """

        waited = False
        # ---------------------------------------------------------
        # 第一阶段：空闲等待循环
        # 只有在“当前没活干 (not has_work)”且“引擎还在跑 (is_running)”时进入。
        # ---------------------------------------------------------
        while not self.has_work() and self.is_running():
            # 通知那些正在等待引擎变为空闲状态的回调函数（比如性能分析工具）。
            self._notify_idle_state_callbacks()

            # 如果输入队列完全是空的：
            if self.input_queue.empty():
                # 清空放弃请求队列 (aborts_queue)。
                # 因为此时引擎已经没有任何任务在跑了，之前积压的“放弃指令”已经没意义了。
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
                
                if logger.isEnabledFor(DEBUG):
                    logger.debug("EngineCore 正在等待新任务...")
                    waited = True

            # 获取阻塞标志：如果引擎没事干，通常会设为阻塞 (True)，让进程休眠省电。
            block = self.process_input_queue_block
            try:
                # 【核心】：尝试从队列中获取客户端（主进程）发来的请求。
                # 如果 block=True，进程会在这里停住，直到主进程发来新消息。
                req = self.input_queue.get(block=block)
                
                # 处理请求（例如：ADD_REQUEST, ABORT_REQUEST 等）。
                self._handle_client_request(*req)
            except queue.Empty:
                # 如果是非阻塞模式且队列为空，跳出循环。
                break
            
            # 如果当前是“非阻塞模式”，处理完一个请求就走，不在这里死等。
            if not block:
                break

        if waited:
            logger.debug("EngineCore 循环已激活（收到新任务）。")

        # ---------------------------------------------------------
        # 第二阶段：快速清空队列 (Drain)
        # 此时引擎可能有活干（正在生成 Token），我们需要快速把队列里的剩余指令处理完。
        # ---------------------------------------------------------
        while not self.input_queue.empty():
            # 使用 non-blocking (get_nowait) 快速把积压的指令全拿出来。
            # 这是为了防止在计算 Token 的间隙，错过了主进程发来的“取消请求”或“新请求插队”。
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self) -> bool:
        """
        仅在有未完成的本地请求时被调用。
        执行一次推理引擎的“步进 (Step)”。
        """

        # ---------------------------------------------------------
        # 1. 核心执行 (The Heavy Lifting)
        # ---------------------------------------------------------
        # step_fn() 是一个被封装的核心逻辑，它内部主要干了这几件事：
        #   1. 调度器 (Scheduler) 介入：决定这一步该跑哪些请求（Prefill 还是 Decode）。
        #   2. 分配显存：为这些请求在 PagedAttention 中分配或锁定 KV Cache 块。
        #   3. 准备 Tensor：把输入拼成连续的矩阵 (Batching)。
        #   4. GPU 前向传播 (Forward Pass)：调用 CUDA 算子进行真正的矩阵乘法。
        #   5. 打包结果：将算出的 Token 封装成对象。
        # 
        # 返回值：
        # - outputs: 生成的结果字典（包含 Token、概率等）。
        # - model_executed: 布尔值，表示这次 Step 是否真正在 GPU 上跑了模型计算。
        outputs, model_executed = self.step_fn()
        
        # ---------------------------------------------------------
        # 2. 结果回传 (Return to Sender)
        # ---------------------------------------------------------
        # 遍历生成的输出结果。outputs 通常是一个包含多个客户端 ID 和对应结果的字典。
        for output in outputs.items() if outputs else ():
            # output_queue 是一个 ZMQ 的 PUSH 套接字（或者 multiprocessing.Queue）。
            # 使用 put_nowait 非阻塞地将算出的 Token 立刻“推”回给主进程（API Server）。
            # 这样用户就能看到打字机一样的流式输出。
            self.output_queue.put_nowait(output)
            
        # ---------------------------------------------------------
        # 3. 步进后清理 (Post-step Hook)
        # ---------------------------------------------------------
        # 执行清理和统计工作。比如：
        # - 释放已经生成完毕（遇到 </s> 停止符）的请求的 KV Cache 显存。
        # - 更新内部的性能指标（Tokens per second 等）。
        self.post_step(model_executed)

        # ---------------------------------------------------------
        # 4. 防“饿死”机制 (GIL Yielding / Deadlock Prevention)
        # ---------------------------------------------------------
        # 【关键设计】：有时候引擎有请求在等，但模型却没有执行计算 (not model_executed)。
        # 为什么会这样？
        # 例如，在多机分布式推理中，当前请求可能正在等待从另一台机器传输 KV Cache 过来。
        # 如果此时继续疯狂执行 while True 的空转循环，
        # Python 的全局解释器锁 (GIL) 会被这个无限循环死死霸占！
        # 这会导致负责接收网络数据（如 NIXL 握手）的后台线程被“饿死”，数据永远传不过来，系统死锁。
        #
        # 解决方案：强制休眠 1 毫秒 (0.001秒)。
        # time.sleep 在 Python 底层会主动释放 GIL，让其他后台线程有机会跑一下，接收数据。
        if not model_executed and self.scheduler.has_unfinished_requests():
            time.sleep(0.001)

        return model_executed

    def _notify_idle_state_callbacks(self) -> None:
        while self._idle_state_callbacks:
            callback = self._idle_state_callbacks.pop()
            callback(self)

    def _handle_shutdown(self) -> bool:
        # Check if shutdown was requested and handle it
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return True

        if self.shutdown_state == EngineShutdownState.REQUESTED:
            shutdown_timeout = self.vllm_config.shutdown_timeout

            logger.info("Shutdown initiated (timeout=%d)", shutdown_timeout)

            if shutdown_timeout == 0:
                num_requests = self.scheduler.get_num_unfinished_requests()
                if num_requests > 0:
                    logger.info("Aborting %d requests", num_requests)
                aborted_reqs = self.scheduler.finish_requests(
                    None, RequestStatus.FINISHED_ABORTED
                )
                self._send_abort_outputs(aborted_reqs)
            else:
                num_requests = self.scheduler.get_num_unfinished_requests()
                if num_requests > 0:
                    logger.info(
                        "Draining %d in-flight requests (timeout=%ds)",
                        num_requests,
                        shutdown_timeout,
                    )

            self.shutdown_state = EngineShutdownState.SHUTTING_DOWN

        # Exit when no work remaining
        if not self.has_work():
            logger.info("Shutdown complete")
            return False

        return True

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """
        处理客户端（主进程）发来的请求的分发器 (Dispatcher)。
        根据 request_type 的不同枚举值，执行不同的逻辑。
        """

        # ---------------------------------------------------------
        # 1. 唤醒指令 (WAKEUP)
        # ---------------------------------------------------------
        if request_type == EngineCoreRequestType.WAKEUP:
            # 这是一个“虚假”的请求，专门用来把引擎从 _process_input_queue 的阻塞休眠中叫醒。
            # 叫醒后不需要做任何事，所以直接 return。
            # 通常发生在需要优雅停机 (shutdown) 的时候。
            return
            
        # ---------------------------------------------------------
        # 2. 新增推理请求 (ADD)
        # ---------------------------------------------------------
        elif request_type == EngineCoreRequestType.ADD:
            # 解包：拿到具体的请求对象 (req) 和 请求波次 (request_wave，用于 MoE 协调)
            req, request_wave = request
            
            # 防御性检查：如果引擎正在关闭中，拒绝接收新任务，并通知主进程失败。
            if self._reject_add_in_shutdown(req):
                return
            
            # 将新请求正式放入引擎的 Scheduler（调度器）中排队等待显存。
            self.add_request(req, request_wave)
            
        # ---------------------------------------------------------
        # 3. 放弃推理请求 (ABORT)
        # ---------------------------------------------------------
        elif request_type == EngineCoreRequestType.ABORT:
            # 这里的 request 通常是一个包含 request_id 的列表。
            # 引擎会立刻在 Scheduler 和正在运行的队列中寻找这些请求，并释放它们占用的显存。
            self.abort_requests(request)
            
        # ---------------------------------------------------------
        # 4. 通用工具调用 (UTILITY)
        # ---------------------------------------------------------
        # 这是一个 RPC（远程过程调用）机制。用于执行非计算类的指令，比如：
        # - 获取性能分析状态 (start_profile / stop_profile)
        # - 检查引擎是否卡死 (check_health)
        elif request_type == EngineCoreRequestType.UTILITY:
            # 解包 RPC 调用信息
            client_idx, call_id, method_name, args = request
            
            # 防御性检查：如果是关闭状态，拒绝某些调用。
            if self._reject_utility_in_shutdown(client_idx, call_id, method_name):
                return
            
            output = UtilityOutput(call_id)
            
            # 延迟查找执行函数：使用 getattr 动态从 self (EngineCore) 上获取方法并执行。
            # 这样做可以确保如果 method_name 不存在，抛出的异常能被妥善捕获并返回给主进程，
            # 而不是让当前 GPU 进程直接崩溃。
            get_result = lambda: (method := getattr(self, method_name)) and method(
                # 处理基于 msgspec 序列化的参数转换
                *self._convert_msgspec_args(method, args)
            )
            
            # 定义将结果推回主进程的闭包函数（放入 output_queue 发送）
            enqueue_output = lambda out: self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=out))
            )
            
            # 执行工具方法，并将结果或报错信息放入输出队列。
            self._invoke_utility_method(method_name, get_result, output, enqueue_output)
            
        # ---------------------------------------------------------
        # 5. 执行器崩溃通报 (EXECUTOR_FAILED)
        # ---------------------------------------------------------
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            # 如果是分布式环境，其他 Rank 的底层算子（Executor）如果崩溃了，
            # 协调器会发这个指令通知当前引擎。
            # 此时引擎必须立刻抛出异常自杀，防止分布式集群陷入数据不同步的死锁。
            raise RuntimeError("Executor failed.")
            
        # ---------------------------------------------------------
        # 6. 未知指令处理
        # ---------------------------------------------------------
        else:
            logger.error(
                "遇到无法识别的输入请求类型：%s", request_type
            )

    def _reject_add_in_shutdown(self, request: Request) -> bool:
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False

        logger.info("Rejecting request %s (server shutting down)", request.request_id)
        self._send_abort_outputs_to_client([request.request_id], request.client_index)
        return True

    def _reject_utility_in_shutdown(
        self, client_idx: int, call_id: int, method_name: str
    ) -> bool:
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False

        logger.warning("Rejecting utility call %s (server shutting down)", method_name)
        output = UtilityOutput(call_id, failure_message="Server shutting down")
        self.output_queue.put_nowait(
            (client_idx, EngineCoreOutputs(utility_output=output))
        )
        return True

    @staticmethod
    def _invoke_utility_method(
        name: str, get_result: Callable, output: UtilityOutput, enqueue_output: Callable
    ):
        try:
            result = get_result()
            if isinstance(result, Future):
                # Defer utility output handling until future completion.
                callback = lambda future: EngineCoreProc._invoke_utility_method(
                    name, future.result, output, enqueue_output
                )
                result.add_done_callback(callback)
                return
            output.result = UtilityResult(result)
        except Exception as e:
            logger.exception("Invocation of %s method failed", name)
            output.failure_message = f"Call to {name} method failed: {str(e)}"
        enqueue_output(output)

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
        arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation)
            if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation)
            else v
            for v, p in zip(args, arg_types)
        )

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal(
                "vLLM shutdown signal from EngineCore failed "
                "to send. Please report this issue."
            )

    def process_input_sockets(
        self,
        input_addresses: list[str],     # 前台发送数据的 ZMQ 管道地址
        coord_input_address: str | None,# 用于数据并行 (DP) 状态同步的协调器地址
        identity: bytes,                # 当前 GPU 的身份证号 (如 b'\x00\x00')
        ready_event: threading.Event,   # 用于通知主线程“小工已就位”的事件锁
    ):
        """Input socket IO thread. 专门处理底层网络接收的后台守护线程"""

        # ---------------------------------------------------------
        # 1. 准备拆包工具 (Msgpack Decoders)
        # ---------------------------------------------------------
        # Msgpack 是一种比 JSON 快得多、体积小得多的二进制序列化格式。
        # oob_tensor_provider: 这是处理“带外数据 (Out-of-band)”的机制。
        # 如果包裹里带有极其庞大的多模态 Tensor，解码器会去共享内存 (IPC) 里提货，而不是在网络管道里死磕。
        add_request_decoder = MsgpackDecoder(
            EngineCoreRequest, oob_tensor_provider=self.tensor_ipc_receiver
        )
        generic_decoder = MsgpackDecoder(oob_tensor_provider=self.tensor_ipc_receiver)

        # 使用 ExitStack 和 Context 确保线程崩溃时，网络端口能被干净地释放
        with ExitStack() as stack, zmq.Context() as ctx:
            
            # ---------------------------------------------------------
            # 2. 建立通信基站 (Socket Initialization)
            # ---------------------------------------------------------
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ] # 创建 DEALER 套接字，并挂上自己的 identity 铭牌，主动连接 (bind=False) 到前台
            
            # (可选) 如果有多机/多卡协同，连接到全局状态协调器
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx, coord_input_address, zmq.XSUB, identity=identity, bind=False,
                    )
                )
                # 向协调器发送 \x01 (订阅消息)，表示我要监听全局调度的指令
                coord_socket.send(b"\x01")

            # ---------------------------------------------------------
            # 3. 开启多路复用雷达 (ZMQ Poller & Handshake)
            # ---------------------------------------------------------
            # 为什么用 Poller？因为小工可能同时盯着好几根管道（主管道、协调器管道等）。
            # Poller 可以让线程在没有数据时休眠，任何一根管道有动静就立刻唤醒它，不浪费一点 CPU。
            poller = zmq.Poller()
            for input_socket in input_sockets:
                # 【ZMQ 底层强制握手协议】：
                # 因为前台是 ROUTER，后台是 DEALER。如果 DEALER 不主动发条消息，
                # ROUTER 根本不知道你上线了。发一个空的 b"" 仅仅是为了激活连接。
                input_socket.send(b"")
                # 把管道注册到雷达上，POLLIN 表示监听“是否有数据进来”
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                # 等待大堂经理（协调器）喊“开始营业 (READY)”
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)

            # 告诉启动这个线程的老大：“雷达已开启，小工就位，可以开始干活了！”
            ready_event.set()
            del ready_event
            
            # ---------------------------------------------------------
            # 4. 核心死循环：接单与分拣 (The Event Loop)
            # ---------------------------------------------------------
            while True:
                # 阻塞在这里，直到雷达发现有管道传来了数据
                for input_socket, _ in poller.poll():
                    
                    # ---------------------------------------------------------
                    # 4.1 零拷贝接收与解帧
                    # ---------------------------------------------------------
                    # 还记得前台发的是三段式包裹吗？[地址, 指令, 数据]。
                    # 地址在路由时被剥离了。这里直接收到：[指令, 数据]。
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    
                    if type_frame.buffer == b"READY":
                        assert input_socket == coord_socket
                        continue # 忽略协调器发来的冗余 READY 信号
                        
                    # 将第一段二进制转换为枚举类型（例如 ADD 或 ABORT）
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # ---------------------------------------------------------
                    # 4.2 反序列化 (解码)
                    # ---------------------------------------------------------
                    request: Any
                    if request_type == EngineCoreRequestType.ADD:
                        # 这是一个新订单，用专用的解码器还原成 EngineCoreRequest 对象
                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                        try:
                            # 预处理（比如检查多模态指针是否有效）
                            request = self.preprocess_add_request(req)
                        except Exception:
                            # 如果订单本身是坏的，处理报错并丢弃，继续接下一单
                            self._handle_request_preproc_error(req)
                            continue
                    else:
                        # 对于取消任务等简单指令，用通用解码器
                        request = generic_decoder.decode(data_frames)

                        # ---------------------------------------------------------
                        # 4.3 优先级双通道分发 (ABORT 的特权)
                        # ---------------------------------------------------------
                        if request_type == EngineCoreRequestType.ABORT:
                            # 【绝妙的并发设计】：
                            # 如果用户取消了请求，这个指令会被**同时**塞进两个队列！
                            # 1. 塞进 aborts_queue：这是一个紧急通道。GPU 调度器会优先看这个，
                            #    立刻把作废的任务从显存里踢出去，释放宝贵的资源。
                            # 2. 依然塞进 input_queue：为了保证时间顺序的完整性，防止内部状态机混乱。
                            self.aborts_queue.put_nowait(request)

                    # ---------------------------------------------------------
                    # 4.4 挂单入厨
                    # ---------------------------------------------------------
                    # 拆包、校验完毕，把标准化的指令和请求对象，塞进后台的核心队列。
                    # GPU 主线程 (run_busy_loop) 正盯着这个队列，拿走就直接开火计算！
                    self.input_queue.put_nowait((request_type, request))

    def process_output_sockets(
        self, output_paths: list[str], coord_output_path: str | None, engine_index: int
    ):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Keep references to outputs and buffers until zmq is finished
        # with them (outputs may contain tensors/np arrays whose
        # backing buffers were extracted for zero-copy send).
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths
            ]
            coord_socket = (
                stack.enter_context(
                    make_zmq_socket(
                        ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000
                    )
                )
                if coord_output_path is not None
                else None
            )
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                if client_index == -1:
                    # Don't reuse buffer for coordinator message
                    # which will be very small.
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = sockets[client_index].send_multipart(
                    buffers, copy=False, track=True
                )
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)

    def _handle_request_preproc_error(self, request: EngineCoreRequest) -> None:
        """Log and return a request-scoped error response for exceptions raised
        from the add request preprocessing in the input socket processing thread.
        """
        logger.exception(
            "Unexpected error pre-processing request %s", request.request_id
        )
        self._send_error_outputs_to_client([request.request_id], request.client_index)

    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None:
        """Pause generation; behavior depends on mode.

        All pause modes queue new adds -- "abort" and "keep" skip step();
        "wait" allows step() so in-flight requests can drain.

        - ``abort``: Set PAUSED_NEW, abort all requests, wait for abort
          outputs to be sent (when running with output_queue), optionally
          clear caches, then complete the returned Future.
        - ``wait``: Set PAUSED_NEW (queue adds, keep stepping); when drained,
          optionally clear caches, then complete the returned Future.
        - ``keep``: Set PAUSED_ALL; return a Future that completes when the
          output queue is empty.
        """
        if mode not in ("keep", "abort", "wait"):
            raise ValueError(f"Invalid pause mode: {mode}")

        def engine_idle_callback(engine: "EngineCoreProc", future: Future[Any]) -> None:
            if clear_cache:
                engine._reset_caches()
            future.set_result(None)

        if mode == "abort":
            aborted_reqs = self.scheduler.finish_requests(
                None, RequestStatus.FINISHED_ABORTED
            )
            self._send_abort_outputs(aborted_reqs)

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        self.scheduler.set_pause_state(pause_state)
        if not self.has_work():
            if clear_cache:
                self._reset_caches()
            return None

        future = Future[Any]()
        self._idle_state_callbacks.append(partial(engine_idle_callback, future=future))
        return future

    def _send_finish_outputs_to_client(
        self, req_ids: list[str], client_index: int, finish_reason: FinishReason
    ) -> None:
        outputs = [
            EngineCoreOutput(req_id, [], finish_reason=finish_reason)
            for req_id in req_ids
        ]
        eco = EngineCoreOutputs(finished_requests=req_ids, outputs=outputs)
        self.output_queue.put_nowait((client_index, eco))

    def _send_abort_outputs_to_client(
        self, req_ids: list[str], client_index: int
    ) -> None:
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ABORT)

    def _send_error_outputs_to_client(
        self, req_ids: list[str], client_index: int
    ) -> None:
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ERROR)

    def _send_abort_outputs(self, aborted_reqs: list[tuple[str, int]]) -> None:
        # TODO(nick) this will be moved inside the scheduler
        if aborted_reqs:
            # Map client_index to list of request_ids that belong to that client.
            by_client = defaultdict[int, set[str]](set)
            for req_id, client_index in aborted_reqs:
                by_client[client_index].add(req_id)
            for client_index, req_ids in by_client.items():
                self._send_abort_outputs_to_client(list(req_ids), client_index)


class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
    ):
        assert vllm_config.model_config.is_moe, (
            "DPEngineCoreProc should only be used for MoE models"
        )

        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.step_counter = 0
        self.current_wave = 0
        self.last_counts = (0, 0)

        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        self.eep_scaling_state: ElasticEPScalingState | None = None

        # Initialize the engine.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        super().__init__(
            vllm_config,
            local_client,
            handshake_address,
            executor_class,
            log_stats,
            client_handshake_address,
            engine_index=dp_rank,
            tensor_queue=tensor_queue,
        )

    def _init_data_parallel(self, vllm_config: VllmConfig):
        # Configure GPUs and stateless process group for data parallel.
        parallel_config = vllm_config.parallel_config
        dp_rank = parallel_config.data_parallel_rank
        dp_size = parallel_config.data_parallel_size
        local_dp_rank = parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        self.dp_rank = dp_rank
        dp_group, dp_store = parallel_config.stateless_init_dp_group(return_store=True)
        self.dp_group, self.dp_store = dp_group, dp_store

    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def add_request(self, request: Request, request_wave: int = 0):
        super().add_request(request, request_wave)
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif (
                not self.engines_running
                and self.scheduler.pause_state == PauseState.UNPAUSED
            ):
                self.engines_running = True
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )

    def resume_scheduler(self):
        super().resume_scheduler()
        if (
            self.has_coordinator
            and not self.engines_running
            and self.scheduler.has_unfinished_requests()
        ):
            # Wake up other DP engines.
            self.output_queue.put_nowait(
                (-1, EngineCoreOutputs(start_wave=self.current_wave))
            )

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and (
                new_wave >= self.current_wave
            ):
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug("EngineCore starting idle loop for wave %d.", new_wave)
                    self.engines_running = True
        else:
            super()._handle_client_request(request_type, request)

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed).
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts, step_counter=self.step_counter, current_wave=self.current_wave
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))

    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()

            if self.eep_scaling_state is not None:
                _ = self.eep_scaling_state.progress()
                if self.eep_scaling_state.is_complete():
                    if self.eep_scaling_state.worker_type == "removing":
                        raise SystemExit
                    self.process_input_queue_block = True
                    self.eep_scaling_state = None

            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # We are in a running state and so must execute a dummy pass
                # if the model didn't execute any ready requests.
                self.execute_dummy_batch()

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                # Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0

        raise SystemExit

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # Optimization - only perform finish-sync all-reduce every 32 steps.
        self.step_counter += 1
        if self.step_counter % 32 != 0:
            return True

        return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        from copy import deepcopy

        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        new_parallel_config = deepcopy(self.vllm_config.parallel_config)
        old_dp_size = new_parallel_config.data_parallel_size
        new_parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            new_parallel_config.data_parallel_rank = (
                reconfig_request.new_data_parallel_rank
            )
        new_parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        new_parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        new_parallel_config._data_parallel_master_port_list = (
            reconfig_request.new_data_parallel_master_port_list
        )
        new_parallel_config._coord_store_port = reconfig_request.coord_store_port

        is_scale_down = reconfig_request.new_data_parallel_size < old_dp_size
        is_shutdown = (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        )

        self.eep_scaling_state = ElasticEPScalingState(
            model_executor=self.model_executor,
            engine_core=self,
            vllm_config=self.vllm_config,
            new_parallel_config=new_parallel_config,
            worker_type="removing" if is_shutdown else "existing",
            scale_type="scale_down" if is_scale_down else "scale_up",
            reconfig_request=reconfig_request,
        )
        self.process_input_queue_block = False
        logger.info(
            "[Elastic EP] Received reconfiguration request and starting scaling up/down"
        )

    def _eep_send_engine_core_notification(
        self,
        notification_type: EEPNotificationType,
        vllm_config: VllmConfig | None = None,
    ):
        """
        Send notifications to EngineCoreClient, which can then forward
        the notifications to other engine core processes. It is used for:
        1) In scale up: new core engines to notify existing core engines
           that they are ready;
        2) In scale down: removing core engines to notify EngineCoreClient
           so EngineCoreClient can release their ray placement groups;
        3) Both scale up/down: to notify EngineCoreClient that existing
           core engines have already switched to the new parallel setup.
        """
        if vllm_config is None:
            dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        else:
            dp_rank = vllm_config.parallel_config.data_parallel_rank
        notification_data = (notification_type.value, dp_rank)
        outputs = EngineCoreOutputs(
            utility_output=UtilityOutput(
                call_id=EEP_NOTIFICATION_CALL_ID,
                result=UtilityResult(notification_data),
            )
        )
        outputs.engine_index = self.engine_index

        if hasattr(self, "output_thread") and self.output_thread.is_alive():
            self.output_queue.put_nowait((0, outputs))
        else:
            encoder = MsgpackEncoder()
            with (
                zmq.Context() as ctx,
                make_zmq_socket(
                    ctx, self.addresses.outputs[0], zmq.PUSH, linger=4000
                ) as socket,
            ):
                socket.send_multipart(encoder.encode(outputs))

    def eep_handle_engine_core_notification(
        self, notification_type: str | EEPNotificationType
    ):
        """
        Handle notification received from EngineCoreClient
        (forwarded from new core engines).
        """
        assert self.eep_scaling_state is not None
        if isinstance(notification_type, str):
            notification_type = EEPNotificationType(notification_type)
        self.eep_scaling_state.handle_notification(notification_type)

    def _eep_scale_up_before_kv_init(self):
        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        self.eep_scaling_state = ElasticEPScalingState(
            model_executor=self.model_executor,
            engine_core=self,
            vllm_config=self.vllm_config,
            new_parallel_config=self.vllm_config.parallel_config,
            worker_type="new",
            scale_type="scale_up",
            reconfig_request=None,
        )
        self.eep_scaling_state.run_pre_kv_init_states()
        self.process_input_queue_block = False


class EngineCoreActorMixin:
    """
    Ray actor for running EngineCore in a data parallel context
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        addresses: EngineZmqAddresses,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        # Initialize tracer for distributed tracing if configured.
        maybe_init_worker_tracer(
            instrumenting_module_name="vllm.engine_core",
            process_kind="engine_core",
            process_name=f"DPEngineCoreActor_DP{dp_rank}",
        )

        self.addresses = addresses
        vllm_config.parallel_config.data_parallel_index = dp_rank
        vllm_config.parallel_config.data_parallel_rank_local = local_dp_rank

        # Set CUDA_VISIBLE_DEVICES as early as possible in actor life cycle
        # NOTE: in MP we set CUDA_VISIBLE_DEVICES at process creation time,
        # and this cannot be done in the same way for Ray because:
        # 1) Ray manages life cycle of all ray workers (including
        # DPEngineCoreActor)
        # 2) Ray sets CUDA_VISIBLE_DEVICES based on num_gpus configuration
        # To bypass 2, we need to also set
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES, but vLLM workers created
        # thereafter would have CUDA_VISIBLE_DEVICES set, which is sticky:
        # https://github.com/ray-project/ray/blob/e752fc319ddedd9779a0989b6d3613909bad75c9/python/ray/_private/worker.py#L456 # noqa: E501
        # This is problematic because when the vLLM worker (a Ray actor)
        # executes a task, it indexes into the sticky CUDA_VISIBLE_DEVICES
        # rather than directly using the GPU ID, potentially resulting in
        # index out of bounds error. See:
        # https://github.com/ray-project/ray/pull/40461/files#diff-31e8159767361e4bc259b6d9883d9c0d5e5db780fcea4a52ead4ee3ee4a59a78R1860 # noqa: E501
        # and get_accelerator_ids_for_accelerator_resource() in worker.py
        # of ray.
        self._set_visible_devices(vllm_config, local_dp_rank)

    def _set_visible_devices(self, vllm_config: VllmConfig, local_dp_rank: int):
        from vllm.platforms import current_platform

        if current_platform.is_xpu():
            pass
        else:
            device_control_env_var = current_platform.device_control_env_var
            self._set_cuda_visible_devices(
                vllm_config, local_dp_rank, device_control_env_var
            )

    def _set_cuda_visible_devices(
        self, vllm_config: VllmConfig, local_dp_rank: int, device_control_env_var: str
    ):
        world_size = vllm_config.parallel_config.world_size
        # Set CUDA_VISIBLE_DEVICES or equivalent.
        try:
            value = get_device_indices(
                device_control_env_var, local_dp_rank, world_size
            )
            os.environ[device_control_env_var] = value
        except IndexError as e:
            raise Exception(
                f"Error setting {device_control_env_var}: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f'base value: "{os.getenv(device_control_env_var)}"'
            ) from e

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ):
        """
        For Ray, we don't need to actually perform handshake.
        All addresses information is known before the actor creation.
        Therefore, we simply yield these addresses.
        """
        yield self.addresses

    def wait_for_init(self):
        """
        Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        pass

    def run(self):
        """
        Run the engine core busy loop.
        """
        try:
            self.run_busy_loop()  # type: ignore[attr-defined]
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.exception("EngineCore encountered a fatal error.")
            raise
        finally:
            self.shutdown()  # type: ignore[attr-defined]


class DPMoEEngineCoreActor(EngineCoreActorMixin, DPEngineCoreProc):
    """Used for MoE model data parallel cases."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        vllm_config.parallel_config.data_parallel_rank = dp_rank

        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        DPEngineCoreProc.__init__(
            self, vllm_config, local_client, "", executor_class, log_stats
        )


class EngineCoreActor(EngineCoreActorMixin, EngineCoreProc):
    """Used for non-MoE and/or non-DP cases."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.data_parallel_size_local = 1
        vllm_config.parallel_config.data_parallel_rank = 0

        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        EngineCoreProc.__init__(
            self,
            vllm_config,
            local_client,
            "",
            executor_class,
            log_stats,
            engine_index=dp_rank,
        )
