# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Callable, Mapping
from copy import copy
from typing import Any

import torch.nn as nn
from typing_extensions import TypeVar

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.distributed.parallel_state import get_dp_group
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import EngineInput, PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.renderers import renderer_from_config
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.tracing import init_tracer
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest, PauseMode
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.executor import Executor
from vllm.v1.metrics.loggers import StatLoggerFactory, StatLoggerManager
from vllm.v1.metrics.reader import Metric, get_metrics_snapshot
from vllm.v1.metrics.stats import IterationStats
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)


class LLMEngine:
    def __init__(
        self,
        vllm_config: VllmConfig,                        # 核心配置总包
        executor_class: type[Executor],                 # 硬件执行器类（包工头，如 GPUExecutor）
        log_stats: bool,                                # 是否记录并打印吞吐性能日志
        aggregate_engine_logging: bool = False,         # 是否聚合多个引擎的日志
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None, # 自定义监控打点工具
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY, # 多模态处理器注册表（处理图文等）
        use_cached_outputs: bool = False,
        multiprocess_mode: bool = False,                # 是否开启多进程模式（引擎核心与 API 层分离）
    ) -> None:
        # ---------------------------------------------------------
        # 1. 保存配置字典，方便全局随时调用
        # ---------------------------------------------------------
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.observability_config = vllm_config.observability_config

        # ---------------------------------------------------------
        # 2. 初始化链路追踪 (OpenTelemetry Tracing)
        # ---------------------------------------------------------
        # 如果用户配置了 OTLP Endpoint（比如接入了 Jaeger 或 Zipkin 等可观测性平台），
        # 这里会初始化分布式追踪。这在微服务架构下排查大模型 API 的请求延迟卡在哪里非常有用。
        tracing_endpoint = self.observability_config.otlp_traces_endpoint
        if tracing_endpoint is not None:
            init_tracer("vllm.llm_engine", tracing_endpoint)

        self.log_stats = log_stats
        # ---------------------------------------------------------
        # 3. 数据并行 (Data Parallel, DP) 通信组初始化
        # ---------------------------------------------------------
        parallel_config = vllm_config.parallel_config
        executor_backend = parallel_config.distributed_executor_backend

        # 检查是否是由外部启动器（如 torchrun）启动了多个数据并行副本
        self.external_launcher_dp = (
            parallel_config.data_parallel_size > 1
            and executor_backend == "external_launcher"
        )
        
        # 核心逻辑：如果在非多进程模式下，且使用了数据并行（而不是外部启动器），
        # 必须在初始化底层的引擎核心 (engine_core) 之前，先建立好 DP 通信组（用于进程间同步状态）。
        if (
            not multiprocess_mode
            and parallel_config.data_parallel_size > 1
            and not self.external_launcher_dp
        ):
            self.dp_group = parallel_config.stateless_init_dp_group()
        else:
            self.dp_group = None
            
        self.should_execute_dummy_batch = False

        # ---------------------------------------------------------
        # 4. 搭建流水线第一步：渲染器与 I/O 处理器
        # ---------------------------------------------------------
        # renderer 负责解析对话模板 (Chat Template) 并在多模态场景下处理图像/音频特征。
        self.renderer = renderer = renderer_from_config(self.vllm_config)
        # io_processor 负责拦截和处理特殊类型的输入输出。
        self.io_processor = get_io_processor(
            self.vllm_config,
            self.renderer,
            self.model_config.io_processor_plugin,
        )

        # ---------------------------------------------------------
        # 5. 搭建流水线第二步：输入数据转换器 (InputProcessor)
        # ---------------------------------------------------------
        # 负责把用户的原始输入 (文本字符串/图片, EngineInput)，
        # 切分成 Token，转化为底层 C++ / CUDA 引擎能懂的请求格式 (EngineCoreRequest)。
        self.input_processor = InputProcessor(self.vllm_config, renderer)

        # ---------------------------------------------------------
        # 6. 搭建流水线第三步：输出数据转换器 (OutputProcessor)
        # ---------------------------------------------------------
        # 当底层 GPU 算出一个 Token ID 后，把它送给 tokenizer 解码成人类可读的文字 (RequestOutput)，
        # 并负责控制流式输出 (Streaming) 的节奏。
        self.output_processor = OutputProcessor(
            renderer.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
            tracing_enabled=tracing_endpoint is not None,
        )

        # ---------------------------------------------------------
        # 7. 唤醒真正干活的核心模块：EngineCoreClient
        # ---------------------------------------------------------
        # 这是整个工厂的心脏！它负责接收 EngineCoreRequests，调度 GPU 计算，返回 EngineCoreOutputs。
        # 使用 make_client 模式意味着它可以是一个本地对象，也可以是一个跨进程的 RPC 客户端。
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )

        # ---------------------------------------------------------
        # 8. 监控与度量大屏：StatLoggerManager
        # ---------------------------------------------------------
        # 如果开启了日志统计，启动一个管理者专门负责收集吞吐量 (Tokens/s)、并发数、显存占用等指标。
        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                custom_stat_loggers=stat_loggers,
                enable_default_loggers=log_stats,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            # 打印一条 "引擎已成功初始化" 的基准日志
            self.logger_manager.log_engine_initialized()

        # ---------------------------------------------------------
        # 9. 向后兼容处理 (V0 架构兼容)
        # ---------------------------------------------------------
        if not multiprocess_mode:
            # 很多老版本的用户代码会直接访问 engine.model_executor。
            # 为了不让他们报错，这里手动建立一个引用，把底层的执行器暴露出来。
            self.model_executor = self.engine_core.engine_core.model_executor  # type: ignore

        # ---------------------------------------------------------
        # 10. 外部启动器的 DP 组补救措施
        # ---------------------------------------------------------
        if self.external_launcher_dp:
            # 如果是外部拉起的分布式训练/推理环境，复用已经存在的 CPU 通信组
            self.dp_group = get_dp_group().cpu_group

        # ---------------------------------------------------------
        # 11. 清理战场
        # ---------------------------------------------------------
        # 在引擎初始化期间，为了测试显存分配或多模态插件，可能会生成一些 Dummy (假) 数据。
        # 这里主动清理缓存，释放宝贵的显存/内存空间，准备迎接真实用户的请求。
        self.reset_mm_cache()

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        disable_log_stats: bool = False,
    ) -> "LLMEngine":
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            log_stats=(not disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: "EngineArgs",
        usage_context: "UsageContext" = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list["StatLoggerFactory"] | None = None,
        enable_multiprocessing: bool = False,
    ) -> "LLMEngine":
        """
        工厂方法：从 EngineArgs 参数总包创建一个完整的 LLMEngine (大语言模型引擎) 实例。
        
        参数:
            engine_args: 包含了所有初始化参数的标准化对象（我们在上一步刚刚组装好的那个）。
            usage_context: 引擎的使用上下文（指明是谁在调用它，比如是离线跑脚本，还是在线 API 服务器）。
            stat_loggers: 外部注入的统计日志记录器列表（用于自定义监控和打点）。
            enable_multiprocessing: 是否开启多进程模式运行引擎的主循环。
        """

        # ---------------------------------------------------------
        # 1. 拆包与配置降级：将扁平的参数转化为层次化的系统配置
        # ---------------------------------------------------------
        # engine_args 里面有几十个参数。这一步会将它们分类、校验，并转化为 vLLM 内部真正使用的
        # 层次化配置对象 (VllmConfig)，这通常包含了：
        # ModelConfig(模型结构), CacheConfig(显存管理), ParallelConfig(分布式策略), SchedulerConfig(调度器) 等。
        vllm_config = engine_args.create_engine_config(usage_context)
        
        # ---------------------------------------------------------
        # 2. 硬件路由：动态寻找最合适的“执行器”
        # ---------------------------------------------------------
        # 这是非常关键的多态设计！vLLM 支持单卡、多卡(Ray/Torchrun)、甚至是非 N 卡硬件（如 TPU、AMD、Neuron）。
        # get_class 会根据 vllm_config 里面的配置（比如 tensor_parallel_size 数量，设备类型等），
        # 智能地返回一个具体的 Executor 类，比如 GPUExecutor, RayGPUExecutor, 或者 TPUExecutor。
        executor_class = Executor.get_class(vllm_config)

        # ---------------------------------------------------------
        # 3. 环境变量覆盖（特征开关 Feature Flag）
        # ---------------------------------------------------------
        # 检查操作系统环境变量。系统级变量通常具有最高优先级。
        # 如果开发者在终端里设置了 `export VLLM_ENABLE_V1_MULTIPROCESSING=1`，
        # 则强制开启多进程模式（这通常用于 vLLM 新老架构的平滑过渡或特定调试场景）。
        if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.debug("Enabling multiprocessing for LLMEngine.")
            enable_multiprocessing = True

        # ---------------------------------------------------------
        # 4. 终极调用：实例化 LLMEngine
        # ---------------------------------------------------------
        # cls() 等同于调用 LLMEngine 的原生 __init__ 构造函数。
        # 把刚才准备好的硬核组件全部塞进去，真正唤醒引擎。
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,                 # 引擎以后干活，就指挥这个 executor 去控制显卡
            log_stats=not engine_args.disable_log_stats,   # 是否打印性能吞吐日志
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=enable_multiprocessing,
        )

    def get_num_unfinished_requests(self) -> int:
        return self.output_processor.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        has_unfinished = self.output_processor.has_unfinished_requests()
        if self.dp_group is None:
            return has_unfinished or self.engine_core.dp_engines_running()
        return self.has_unfinished_requests_dp(has_unfinished)

    def has_unfinished_requests_dp(self, has_unfinished: bool) -> bool:
        aggregated_has_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, has_unfinished
        )
        if not has_unfinished and aggregated_has_unfinished:
            self.should_execute_dummy_batch = True
        return aggregated_has_unfinished

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        if not hasattr(self, "_supported_tasks"):
            # Cache the result
            self._supported_tasks = self.engine_core.get_supported_tasks()

        return self._supported_tasks

    def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""

        request_ids = self.output_processor.abort_requests(request_ids, internal)
        self.engine_core.abort_requests(request_ids)

    def add_request(
        self,
        request_id: str,                           # 请求的唯一 ID
        prompt: "EngineCoreRequest | PromptType | EngineInput", # 原始提示（文本、Token 或预处理对象）
        params: "SamplingParams | PoolingParams",  # 采样或池化参数
        arrival_time: float | None = None,         # 请求到达时间（用于性能统计）
        lora_request: "LoRARequest | None" = None, # LoRA 适配器请求
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,    # 分布式追踪头
        priority: int = 0,                         # 优先级
        prompt_text: str | None = None,            # 原始提示文本
    ) -> str:
        # ---------------------------------------------------------
        # 1. 基础校验
        # ---------------------------------------------------------
        if not isinstance(request_id, str):
            raise TypeError(f"request_id must be a string, got {type(request_id)}")

        # ---------------------------------------------------------
        # 2. 将原始输入加工为标准 Request 对象
        # ---------------------------------------------------------
        # 检查输入是否已经是底层格式（旧版兼容逻辑）
        if isinstance(prompt, EngineCoreRequest):
            logger.warning_once("直接传入 EngineCoreRequest 已被弃用...")
            request = prompt
            # 校验 ID 是否一致
            if request_id != request.request_id:
                logger.warning_once("传入的 request_id 与对象内部 ID 不符，以对象内部为准")
        else:
            # 【核心动作】：调用 input_processor 将文本/图片等转化为 EngineCoreRequest
            # 这里会完成分词 (Tokenization) 和多模态数据的预处理
            request = self.input_processor.process_inputs(
                request_id,
                prompt,
                params,
                supported_tasks=self.get_supported_tasks(),
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
            )
            # 提取提示词文本用于后续输出显示
            prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)

        # 确保 Request 对象绑定了最终的 ID
        self.input_processor.assign_request_id(request)
        req_id = request.request_id
        
        # 重新获取 params（可能在 process_inputs 中被更新或克隆过）
        params = request.params

        # ---------------------------------------------------------
        # 3. 处理“一对多”生成 (Fan-out)
        # ---------------------------------------------------------
        # n 代表对于同一个 Prompt，我们要生成多少个独立的完成结果（Completions）
        n = params.n if isinstance(params, SamplingParams) else 1

        # 场景 A：常规的一对一生成 (n=1)
        if n == 1:
            # 在输出处理器中登记该请求，准备接收结果
            self.output_processor.add_request(request, prompt_text, None, 0)
            # 【关键】：将请求正式提交给底层 EngineCore (GPU 调度核心)
            self.engine_core.add_request(request)
            return req_id

        # 场景 B：一对多生成 (n > 1)
        # 比如用户要求：对这一个问题给我生成 5 个不同的回答（用于对比或选优）
        parent_req = ParentRequest(request) # 创建一个父请求容器
        for idx in range(n):
            # 获取子请求的 ID 和参数
            request_id, child_params = parent_req.get_child_info(idx)
            
            # 克隆请求对象，为每个子请求分配独立的身份，但共享大部分数据
            child_request = request if idx == n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = child_params

            # 在输出处理器中登记子请求，并关联父请求
            self.output_processor.add_request(
                child_request, prompt_text, parent_req, idx
            )
            # 将每个子请求分别发给 EngineCore
            self.engine_core.add_request(child_request)

        return req_id

    def step(self) -> list[RequestOutput | PoolingRequestOutput]:
        if self.should_execute_dummy_batch:
            self.should_execute_dummy_batch = False
            self.engine_core.execute_dummy_batch()
            return []

        # 1) Get EngineCoreOutput from the EngineCore.
        with record_function_or_nullcontext("llm_engine step: get_output"):
            outputs = self.engine_core.get_output()

        # 2) Process EngineCoreOutputs.
        with record_function_or_nullcontext("llm_engine step: process_outputs"):
            iteration_stats = IterationStats() if self.log_stats else None
            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,
                engine_core_timestamp=outputs.timestamp,
                iteration_stats=iteration_stats,
            )
            self.output_processor.update_scheduler_stats(outputs.scheduler_stats)

        # 3) Abort any reqs that finished due to stop strings.
        with record_function_or_nullcontext("llm_engine step: abort_requests"):
            self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        # 4) Record stats
        with record_function_or_nullcontext("llm_engine step: record_stats"):
            if (
                self.logger_manager is not None
                and outputs.scheduler_stats is not None
                and len(outputs.outputs) > 0
            ):
                self.logger_manager.record(
                    scheduler_stats=outputs.scheduler_stats,
                    iteration_stats=iteration_stats,
                    mm_cache_stats=self.renderer.stat_mm_cache(),
                )
                self.do_log_stats_with_interval()

        return processed_outputs.request_outputs

    def start_profile(self, profile_prefix: str | None = None):
        self.engine_core.profile(True, profile_prefix)

    def stop_profile(self):
        self.engine_core.profile(False)

    def reset_mm_cache(self):
        self.renderer.clear_mm_cache()
        self.engine_core.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.engine_core.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings computed with old weights are not reused.
        """
        self.engine_core.reset_encoder_cache()

    def sleep(self, level: int = 1, mode: PauseMode = "abort"):
        self.engine_core.sleep(level, mode)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(1, level)

    def wake_up(self, tags: list[str] | None = None):
        self.engine_core.wake_up(tags)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    def get_metrics(self) -> list[Metric]:
        assert self.log_stats, "Stat logging disabled"
        return get_metrics_snapshot()

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        return self.renderer.get_tokenizer()

    def do_log_stats(self) -> None:
        """Log stats if logging is enabled."""
        if self.logger_manager:
            self.logger_manager.log()

    def do_log_stats_with_interval(self) -> None:
        """Log stats when the time interval has passed."""
        now = time.time()
        if not hasattr(self, "_last_log_time"):
            self._last_log_time = now
        if now - self._last_log_time >= envs.VLLM_LOG_STATS_INTERVAL:
            self.do_log_stats()
            self._last_log_time = now

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        return self.engine_core.pin_lora(lora_id)

    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        return self.collective_rpc("apply_model", args=(func,))

    def __del__(self):
        dp_group = getattr(self, "dp_group", None)
        if dp_group is not None and not self.external_launcher_dp:
            stateless_destroy_torch_distributed_process_group(dp_group)
