# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import torch.nn as nn
from pydantic import ValidationError
from tqdm.auto import tqdm
from typing_extensions import TypeVar, overload

from vllm.beam_search import (
    BeamSearchInstance,
    BeamSearchOutput,
    BeamSearchSequence,
    create_sort_beams_key_function,
)
from vllm.config import (
    AttentionConfig,
    CompilationConfig,
    PoolerConfig,
    ProfilerConfig,
    StructuredOutputsConfig,
    is_init_field,
)
from vllm.config.compilation import CompilationMode
from vllm.config.model import (
    ConvertOption,
    HfOverrides,
    ModelDType,
    RunnerOption,
    TokenizerMode,
)
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateConfig,
    ChatTemplateContentFormatOption,
    load_chat_template,
)
from vllm.entrypoints.pooling.io_processor_factories import init_pooling_io_processors
from vllm.entrypoints.pooling.scoring.io_processor import (
    ScoringIOProcessor,
)
from vllm.entrypoints.pooling.scoring.typing import ScoreInput
from vllm.entrypoints.pooling.typing import OfflineInputsContext, OfflineOutputsContext
from vllm.entrypoints.utils import log_non_default_args
from vllm.inputs import (
    DataPrompt,
    EngineInput,
    PromptType,
    TextPrompt,
    TokensPrompt,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.outputs import (
    ClassificationRequestOutput,
    EmbeddingRequestOutput,
    PoolingRequestOutput,
    RequestOutput,
    ScoringRequestOutput,
)
from vllm.platforms import current_platform
from vllm.pooling_params import PoolingParams
from vllm.renderers import ChatParams, merge_kwargs
from vllm.renderers.inputs.preprocess import (
    conversation_to_seq,
    parse_model_prompt,
    prompt_to_seq,
)
from vllm.sampling_params import BeamSearchParams, RequestOutputKind, SamplingParams
from vllm.tasks import PoolingTask
from vllm.tokenizers import TokenizerLike
from vllm.usage.usage_lib import UsageContext
from vllm.utils.counter import Counter
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.utils.tqdm_utils import maybe_tqdm
from vllm.v1.engine import PauseMode
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.sample.logits_processor import LogitsProcessor

if TYPE_CHECKING:
    from vllm.v1.metrics.reader import Metric

logger = init_logger(__name__)

_O = TypeVar(
    "_O",
    bound=RequestOutput | PoolingRequestOutput,
    default=RequestOutput | PoolingRequestOutput,
)
_P = TypeVar("_P", bound=SamplingParams | PoolingParams | None)
_R = TypeVar("_R", default=Any)


class LLM:
    """一个用于根据给定提示（prompts）和采样参数生成文本的大语言模型（LLM）。

    该类包含一个分词器（tokenizer）、一个语言模型（可能分布在多个 GPU 上），
    以及为中间状态（即 KV 缓存）分配的 GPU 内存空间。给定一批提示和采样参数，
    该类利用智能批处理机制和高效的内存管理，从模型中生成文本。

    参数：
        model: HuggingFace Transformers 模型的名称或路径。
        tokenizer: HuggingFace Transformers 分词器的名称或路径。
        tokenizer_mode: 分词器模式。"auto" 表示在可用时使用快速分词器，
            "slow" 表示始终使用慢速分词器。
        skip_tokenizer_init: 若为 True，则跳过分词器和反分词器的初始化。
            此时要求输入中提供有效的 prompt_token_ids，并且 prompt 为 None。
        trust_remote_code: 在下载模型和分词器时是否信任远程代码（例如来自 HuggingFace 的代码）。
        allowed_local_media_path: 允许 API 请求从服务器文件系统中指定的目录
            读取本地图像或视频。这存在安全风险，仅应在受信任的环境中启用。
        allowed_media_domains: 若设置此参数，则仅允许属于该域名的媒体 URL
            用于多模态输入。
        tensor_parallel_size: 用于张量并行分布式执行的 GPU 数量。
        dtype: 模型权重和激活值的数据类型。目前支持 `float32`、`float16` 和 `bfloat16`。
            若设为 `auto`，则使用 Transformers 模型配置中的 `dtype` 属性。
            但是，如果配置中的 `dtype` 是 `float32`，则会改用 `float16`。
        quantization: 用于量化模型权重的方法。目前支持 "awq"、"gptq" 和 "fp8"（实验性）。
            如果为 None，首先检查模型配置文件中的 `quantization_config` 属性。
            如果该属性也为 None，则假定模型权重未量化，并使用 `dtype` 确定权重的数据类型。
        revision: 要使用的具体模型版本，可以是分支名、标签名或提交 ID。
        tokenizer_revision: 要使用的具体分词器版本，可以是分支名、标签名或提交 ID。
        chat_template: 要应用的聊天模板。
        seed: 用于初始化采样随机数生成器的种子。
        gpu_memory_utilization: 保留用于模型权重、激活值和 KV 缓存的 GPU 内存比例（0 到 1 之间）。
            较高的值会增大 KV 缓存大小，从而提高模型吞吐量。
            但如果该值过高，可能会导致内存不足（OOM）错误。
        kv_cache_memory_bytes: 每个 GPU 上 KV 缓存的大小（以字节为单位）。
            默认为 None，此时 vLLM 会根据 gpu_memory_utilization 自动推断 KV 缓存大小。
            但用户也可能希望手动指定 KV 缓存内存大小。
            与使用 gpu_memory_utilization 相比，kv_cache_memory_bytes 提供了更精细的内存控制。
            注意：当 kv_cache_memory_bytes 不为 None 时，将忽略 gpu_memory_utilization。
        cpu_offload_gb: 用于卸载模型权重的 CPU 内存大小（单位：GiB）。
            这实际上增加了可用于保存模型权重的 GPU 内存空间，
            但代价是每次前向传播都需要进行 CPU-GPU 数据传输。
        offload_group_size: 预取卸载（Prefetch offloading）：每 N 层划分为一组。
            卸载每组中最后 `offload_num_in_group` 层。默认为 0（禁用）。
        offload_num_in_group: 预取卸载：每组中要卸载的层数。默认为 1。
        offload_prefetch_step: 预取卸载：提前预取的层数。
            值越大可隐藏更多延迟，但会占用更多 GPU 内存。默认为 1。
        offload_params: 预取卸载：要选择性卸载的参数名称片段集合。
            只有名称包含这些片段之一的参数才会被卸载（例如，
            {"gate_up_proj", "down_proj"} 表示 MLP 权重，
            或 {"w13_weight", "w2_weight"} 表示 MoE 专家权重）。
            如果为 None 或空，则卸载所有参数。
        enforce_eager: 是否强制启用 eager 执行模式。
            如果为 True，则禁用 CUDA Graph，始终以 eager 模式执行模型。
            如果为 False，则混合使用 CUDA Graph 和 eager 执行。
        enable_return_routed_experts: 是否返回路由专家信息。
        disable_custom_all_reduce: 参见 [ParallelConfig][vllm.config.ParallelConfig]。
        hf_token: 用于远程文件 HTTP Bearer 授权的 token。
            如果设为 `True`，将使用运行 `hf auth login` 时生成的 token
            （存储在 `~/.cache/huggingface/token` 中）。
        hf_overrides: 如果是字典，则包含要传递给 HuggingFace 配置的参数；
            如果是可调用对象，则用于更新 HuggingFace 配置。
        mm_processor_kwargs: 传递给模型多模态处理器（如图像处理器）的参数，
            用于覆盖通过 `AutoProcessor.from_pretrained` 获取的多模态处理器配置。
            可用的覆盖项取决于所运行的模型。
            例如，对于 Phi-3-Vision：`{"num_crops": 4}`。
        pooler_config: 为池化模型初始化非默认的池化配置，
            例如 `PoolerConfig(seq_pooling_type="MEAN", use_activation=False)`。
        compilation_config: 可以是整数或字典。
            如果是整数，则作为编译优化的模式；
            如果是字典，则可指定完整的编译配置。
        attention_config: 注意力机制的配置。可以是字典或 AttentionConfig 实例。
            如果是字典，将被转换为 AttentionConfig。
            允许指定注意力后端及其他注意力相关设置。
        **kwargs: 传递给 [`EngineArgs`][vllm.EngineArgs] 的参数。

    注意：
        该类旨在用于离线推理。对于在线服务，请改用
        [AsyncLLMEngine][vllm.AsyncLLMEngine] 类。
    """

    def __init__(
        self,
        model: str,
        *,
        runner: RunnerOption = "auto",
        convert: ConvertOption = "auto",
        tokenizer: str | None = None,
        tokenizer_mode: TokenizerMode | str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        allowed_media_domains: list[str] | None = None,
        tensor_parallel_size: int = 1,
        dtype: ModelDType = "auto",
        quantization: QuantizationMethods | None = None,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        chat_template: Path | str | None = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        cpu_offload_gb: float = 0,
        offload_group_size: int = 0,
        offload_num_in_group: int = 1,
        offload_prefetch_step: int = 1,
        offload_params: set[str] | None = None,
        enforce_eager: bool = False,
        enable_return_routed_experts: bool = False,
        disable_custom_all_reduce: bool = False,
        hf_token: bool | str | None = None,
        hf_overrides: HfOverrides | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        pooler_config: PoolerConfig | None = None,
        structured_outputs_config: dict[str, Any]
        | StructuredOutputsConfig
        | None = None,
        profiler_config: dict[str, Any] | ProfilerConfig | None = None,
        attention_config: dict[str, Any] | AttentionConfig | None = None,
        kv_cache_memory_bytes: int | None = None,
        compilation_config: int | dict[str, Any] | CompilationConfig | None = None,
        logits_processors: list[str | type[LogitsProcessor]] | None = None,
        **kwargs: Any,
    ) -> None:
        """LLM constructor."""

        from typing import Any
from pathlib import Path
import cloudpickle
from pydantic import ValidationError
import logging

logger = logging.getLogger(__name__)

# 注意：为了代码能通过静态类型检查，我保留了原有的类型注解，
# 实际运行环境中需要存在对应的类型定义（如 RunnerOption, ConvertOption 等）。

class LLM:
    def __init__(
        self,
        model: str,
        *,
        runner: "RunnerOption" = "auto",
        convert: "ConvertOption" = "auto",
        tokenizer: str | None = None,
        tokenizer_mode: "TokenizerMode" | str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        allowed_media_domains: list[str] | None = None,
        tensor_parallel_size: int = 1,
        dtype: "ModelDType" = "auto",
        quantization: "QuantizationMethods" | None = None,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        chat_template: Path | str | None = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        cpu_offload_gb: float = 0,
        offload_group_size: int = 0,
        offload_num_in_group: int = 1,
        offload_prefetch_step: int = 1,
        offload_params: set[str] | None = None,
        enforce_eager: bool = False,
        enable_return_routed_experts: bool = False,
        disable_custom_all_reduce: bool = False,
        hf_token: bool | str | None = None,
        hf_overrides: "HfOverrides" | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        pooler_config: "PoolerConfig" | None = None,
        structured_outputs_config: dict[str, Any]
        | "StructuredOutputsConfig"
        | None = None,
        profiler_config: dict[str, Any] | "ProfilerConfig" | None = None,
        attention_config: dict[str, Any] | "AttentionConfig" | None = None,
        kv_cache_memory_bytes: int | None = None,
        compilation_config: int | dict[str, Any] | "CompilationConfig" | None = None,
        logits_processors: list[str | type["LogitsProcessor"]] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        LLM (大语言模型) 推理引擎的构造函数。
        
        参数 (Args):
            model: Hugging Face 模型名称或本地模型路径。
            runner: 运行器选项，用于指定后端的执行模式，默认为 "auto"。
            convert: 权重转换选项，默认为 "auto"。
            tokenizer: 分词器的名称或路径。如果为 None，则默认使用与 `model` 相同的路径。
            tokenizer_mode: 分词器的加载模式（例如 "slow", "fast" 或 "auto"）。
            skip_tokenizer_init: 如果为 True，则跳过分词器的初始化（适用于纯视觉或纯特征提取任务）。
            trust_remote_code: 是否允许在本地机器上执行来自 Hugging Face Hub 的自定义模型代码。开启需注意安全风险。
            allowed_local_media_path: 多模态模型允许访问的本地多媒体文件路径（白名单机制）。
            allowed_media_domains: 多模态模型允许从中下载多媒体文件的域名列表。
            tensor_parallel_size: 张量并行的数量（通常对应使用的 GPU 数量）。
            dtype: 模型权重的数据类型（如 "float16", "bfloat16", "auto"）。"auto" 会根据模型配置自动推断。
            quantization: 使用的量化方法（如 "awq", "gptq", "squeezellm" 等）。如果为 None，则不进行量化。
            revision: Hugging Face 模型的版本（branch 名字、tag 或者 commit hash）。
            tokenizer_revision: Hugging Face 分词器的版本。
            chat_template: 用于对话格式化的自定义 jinja 聊天模板的路径或字符串。
            seed: 随机种子，用于确保生成过程的可重复性。
            gpu_memory_utilization: 分配给模型和 KV Cache 的 GPU 显存比例（0 到 1 之间）。默认为 0.9 (90%)。
            cpu_offload_gb: 卸载到 CPU 的模型参数/KV Cache 的容量限制（单位：GB）。
            offload_group_size: 参数卸载相关的组大小配置。
            offload_num_in_group: 卸载组内的数量配置。
            offload_prefetch_step: 卸载时的预取步长，用于优化 I/O 延迟。
            offload_params: 具体指定需要卸载的参数名称集合。
            enforce_eager: 如果为 True，则强制使用 PyTorch Eager 模式执行（禁用 CUDA Graphs，便于调试或处理动态形状）。
            enable_return_routed_experts: 是否允许返回 MoE（混合专家）模型中被激活的路由专家信息。
            disable_custom_all_reduce: 如果为 True，则禁用自定义的 all-reduce 算子，回退到默认的 NCCL 操作。
            hf_token: 访问私有或受限 Hugging Face 模型库所需的认证 Token。
            hf_overrides: 传递给 Hugging Face `transformers` 模型配置的覆盖参数。
            mm_processor_kwargs: 传递给多模态处理器（Multimodal Processor）的额外关键字参数。
            pooler_config: 特征池化层（Pooler）的配置选项（常用于 Embedding 模型）。
            structured_outputs_config: 结构化输出（如 JSON schema 约束生成）的配置参数。
            profiler_config: 性能分析器（Profiler）的配置，用于分析推理性能瓶颈。
            attention_config: 注意力机制后端的特定配置（例如 FlashAttention 的配置）。
            kv_cache_memory_bytes: 强制指定分配给 KV Cache 的内存大小（字节）。如果不指定，将根据 gpu_memory_utilization 自动计算。
            compilation_config: 模型编译（如 `torch.compile`）的配置选项，用于加速推理。
            logits_processors: 用于修改或惩罚生成词汇概率分布的自定义逻辑处理器列表。
            **kwargs: 其他未显式定义的额外参数。
        """

        # ---------------------------------------------------------
        # 1. 向后兼容处理：处理已废弃的 'swap_space' 参数
        # 遗弃是因为该参数改名为了cpu_offload
        # ---------------------------------------------------------
        if "swap_space" in kwargs:
            # 移除已废弃的参数，避免后续传递引发错误
            kwargs.pop("swap_space")
            import warnings

            # 触发弃用警告，提醒开发者更新代码。stacklevel=2 表示警告指向调用者所在的代码行
            warnings.warn(
                "The 'swap_space' parameter is deprecated and ignored. "
                "It will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        # ---------------------------------------------------------
        # 2. 默认行为设置：日志统计
        # ---------------------------------------------------------
        # 如果用户没有显式配置 disable_log_stats，则默认将其设置为 True (禁用统计日志打印)
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        # ---------------------------------------------------------
        # 3. 分布式执行 / Worker 处理逻辑
        # 如果自己定义了一个类，这段函数保证它能被正确传递
        # ---------------------------------------------------------
        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            # 检查 worker_cls 是否为一个类类型 (class/type)
            # 如果是类类型，因为在多进程 (multiprocessing/ray) 环境下直接传递类可能导致 pickling (序列化) 问题，
            # 所以这里使用 cloudpickle 将其序列化成字节流安全地传递。
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        # ---------------------------------------------------------
        # 4. KV Cache 传输 / 离线推断的高级配置处理
        # 检查kv传输的参数配置
        # ---------------------------------------------------------
        if "kv_transfer_config" in kwargs and isinstance(
            kwargs["kv_transfer_config"], dict
        ):
            # 将字典形式的配置实例化为数据类/Pydantic模型对象
            from vllm.config.kv_transfer import KVTransferConfig

            raw_config_dict = kwargs["kv_transfer_config"]
            try:
                # 尝试通过字典解包实例化对象
                kwargs["kv_transfer_config"] = KVTransferConfig(**raw_config_dict)
            except ValidationError as e:
                # 捕获 Pydantic 抛出的校验错误（例如字典内缺失必填字段，或类型不符）
                logger.error(
                    "Failed to convert 'kv_transfer_config' dict to "
                    "KVTransferConfig object. Dict: %s. Error: %s",
                    raw_config_dict,
                    e,
                )
                # 包装并重新抛出一个 ValueError，包含清晰的错误上下文，中断初始化
                raise ValueError(f"Invalid 'kv_transfer_config' provided: {e}") from e

        # ---------------------------------------------------------
        # 5. 变量初始化兜底
        # ---------------------------------------------------------
        # 确保 hf_overrides 始终为一个字典对象，避免后续访问字典方法时抛出 NoneType 错误
        if hf_overrides is None:
            hf_overrides = {}
            

        # ---------------------------------------------------------
        # 6. 配置对象实例化辅助函数
        # ---------------------------------------------------------
        def _make_config(value: Any, cls: type[_R]) -> _R:
            """
            将字典 (dict)、None 或现有的实例转换为目标配置类 (cls) 的实例。
            这是一个内部闭包函数，用于统一处理各种复杂的子配置项。
            """
            if value is None:
                # 如果用户没传，返回该配置类的默认实例
                return cls()
            if isinstance(value, dict):
                # 如果用户传的是字典，则使用字典解包来实例化。
                # is_init_field 配合过滤：只保留目标类 (cls) 真正需要的参数，
                # 丢弃字典里多余/无效的键，防止抛出意外参数的错误。
                return cls(**{k: v for k, v in value.items() if is_init_field(cls, k)})  # type: ignore[arg-type]
            
            # 如果已经是对应的类实例，直接返回
            return value

        # ---------------------------------------------------------
        # 7. 各种高级特性的配置解析
        # ---------------------------------------------------------
        # 处理模型编译配置 (Torch Compile 等机制)
        if isinstance(compilation_config, int):
            # 如果传的是整数，通常代表预设的编译优化等级 (mode)
            compilation_config_instance = CompilationConfig(
                mode=CompilationMode(compilation_config)
            )
        else:
            # 否则通过常规字典或实例解析
            compilation_config_instance = _make_config(
                compilation_config, CompilationConfig
            )

        # 处理结构化输出 (如强制生成 JSON 格式)、性能分析器、注意力机制的特定配置
        structured_outputs_instance = _make_config(
            structured_outputs_config, StructuredOutputsConfig
        )
        #性能记录器
        profiler_config_instance = _make_config(profiler_config, ProfilerConfig)
        #注意力机制配置
        attention_config_instance = _make_config(attention_config, AttentionConfig)

        # ---------------------------------------------------------
        # 8. 分布式 / 数据并行 (Data Parallel) 安全校验
        # ---------------------------------------------------------
        # 获取用户设置的数据并行大小（默认为 1）
        _dp_size = int(kwargs.get("data_parallel_size", 1))
        # 获取分布式执行后端的类型
        _distributed_executor_backend = kwargs.get("distributed_executor_backend")
        
        # 危险操作拦截：如果试图在单进程模式下使用多个数据并行副本...
        if (
            _dp_size > 1
            and not _distributed_executor_backend == "external_launcher"  # 且没有使用外部启动器 (如 torchrun, ray)
            and not current_platform.is_tpu()                             # 且当前硬件不是 TPU
        ):
            # 抛出致命错误。因为在纯单进程环境 (普通 python 脚本) 里强行跑数据并行会导致进程死锁 (hang)。
            raise ValueError(
                f"LLM(data_parallel_size={_dp_size}) is not supported for single-"
                "process usage and may hang. Please use "
                "the explicit multi-process data-parallel example at "
                "'examples/offline_inference/data_parallel.py'."
            )

        # ---------------------------------------------------------
        # 9. 组装终极参数包 (EngineArgs)
        # ---------------------------------------------------------
        # 将用户传入的所有散装参数，以及刚刚处理好的实例化配置对象，
        # 全部打包进一个标准化的数据类 EngineArgs 中。
        # 这样做是为了隔离 API 层 (LLM类) 和 底层执行引擎层 (LLMEngine类) 的接口。
        engine_args = EngineArgs(
            model=model,
            runner=runner,
            convert=convert,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            allowed_local_media_path=allowed_local_media_path,
            allowed_media_domains=allowed_media_domains,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            cpu_offload_gb=cpu_offload_gb,
            offload_group_size=offload_group_size,
            offload_num_in_group=offload_num_in_group,
            offload_prefetch_step=offload_prefetch_step,
            offload_params=offload_params or set(),
            enforce_eager=enforce_eager,
            enable_return_routed_experts=enable_return_routed_experts,
            disable_custom_all_reduce=disable_custom_all_reduce,
            hf_token=hf_token,
            hf_overrides=hf_overrides,
            mm_processor_kwargs=mm_processor_kwargs,
            pooler_config=pooler_config,
            structured_outputs_config=structured_outputs_instance,
            profiler_config=profiler_config_instance,
            attention_config=attention_config_instance,
            compilation_config=compilation_config_instance,
            logits_processors=logits_processors,
            **kwargs,
        )

        # 记录所有非默认配置的日志，方便用户排查性能问题或行为异常
        log_non_default_args(engine_args)

        # ---------------------------------------------------------
        # 10. 核心动作：实例化底层大模型引擎
        # ---------------------------------------------------------
        # 这里是真正发生显存分配、模型权重加载、进程启动的地方。
        # usage_context 告诉底层引擎：我是通过顶层的 `LLM` 类（通常是离线批处理模式）来调用的你。
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS
        )
        
        # 将底层引擎解析好的模型配置保存到顶层属性，方便快速访问
        self.model_config = self.llm_engine.model_config
        self.engine_class = type(self.llm_engine)

        # ---------------------------------------------------------
        # 11. 初始化各种辅助工具、计数器和处理器
        # ---------------------------------------------------------
        # 用于为发给引擎的每个请求生成唯一的 ID
        self.request_counter = Counter()
        self.default_sampling_params: dict[str, Any] | None = None

        # 获取该模型支持的任务类型（比如：是生成文本 Generation，还是提取特征 Embedding/Pooling）
        supported_tasks = self.llm_engine.get_supported_tasks()
        self.supported_tasks = supported_tasks
        
        # 检查是否包含池化（Pooling/Embedding）任务
        self.pooling_task = self.model_config.get_pooling_task(supported_tasks)
        if self.pooling_task is not None:
            logger.info("Supported pooling task: %s", self.pooling_task)

        # 绑定渲染器（用于多模态图像/文本交错排版）、对话模板加载、输入/输出处理器
        self.runner_type = self.model_config.runner_type
        self.renderer = self.llm_engine.renderer
        
        # 加载用户提供的或者模型默认的 Jinja 对话模板 (Chat Template)
        self.chat_template = load_chat_template(chat_template)
        
        self.io_processor = self.llm_engine.io_processor
        self.input_processor = self.llm_engine.input_processor
        self.chat_template_config = ChatTemplateConfig(chat_template=self.chat_template)
        
        # 如果支持 Pooling 任务，初始化对应的 I/O 处理器
        self.pooling_io_processors = init_pooling_io_processors(
            supported_tasks=supported_tasks,
            model_config=self.model_config,
            renderer=self.renderer,
            chat_template_config=self.chat_template_config,
        )
        
        # 缓存自身的 __repr__ 字符串表示。
        # 因为在分布式环境下，获取引擎状态可能需要进行高昂的 RPC 通信，缓存可以提升打印对象时的性能。
        self._cached_repr: str | None = None

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "LLM":
        """Create an LLM instance from EngineArgs."""
        return cls(**vars(engine_args))

    def get_tokenizer(self) -> TokenizerLike:
        return self.llm_engine.get_tokenizer()

    def get_world_size(self, include_dp: bool = True) -> int:
        """Get the world size from the parallel config.

        Args:
            include_dp: If True (default), returns the world size including
                data parallelism (TP * PP * DP). If False, returns the world
                size without data parallelism (TP * PP).

        Returns:
            The world size (tensor_parallel_size * pipeline_parallel_size),
            optionally multiplied by data_parallel_size if include_dp is True.
        """
        parallel_config = self.llm_engine.vllm_config.parallel_config
        if include_dp:
            return parallel_config.world_size_across_dp
        return parallel_config.world_size

    def reset_mm_cache(self) -> None:
        self.renderer.clear_mm_cache()
        self.llm_engine.reset_mm_cache()

    def get_default_sampling_params(self) -> SamplingParams:
        if self.default_sampling_params is None:
            self.default_sampling_params = self.model_config.get_diff_sampling_param()
        if self.default_sampling_params:
            return SamplingParams.from_optional(**self.default_sampling_params)
        return SamplingParams()

    def generate(
        self,
        prompts: "PromptType" | Sequence["PromptType"],
        sampling_params: "SamplingParams" | Sequence["SamplingParams"] | None = None,
        *,
        use_tqdm: bool | Callable[..., "tqdm"] = True,
        lora_request: Sequence["LoRARequest"] | "LoRARequest" | None = None,
        priority: list[int] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list["RequestOutput"]:
        """
        为输入的提示（prompts）生成补全文本。

        该方法会自动对给定的提示进行批处理，并考虑内存限制。
        为了获得最佳性能，请将所有提示放入一个列表中，然后一次性传入此方法。

        参数 (Args)：
            prompts: 输入给大模型的提示词。支持单条字符串，也支持字符串列表（进行批量推理）。
            sampling_params: 控制文本生成的采样参数（如 temperature, top_p, max_tokens 等）。
                - 如果为 None：使用模型默认参数。
                - 如果是单个对象：该参数应用于列表中所有的提示词。
                - 如果是列表：必须与 prompts 长度一致，实现“一对一”的个性化采样配置。
            use_tqdm: 是否显示进度条。支持传入自定义的 tqdm 构造函数。
            lora_request: 如果使用了 LoRA 微调插件，通过此参数指定要加载的 LoRA 适配器请求。
            priority: 请求优先级。在任务极多时，优先级高的请求会排在前面。需要配合特定的调度策略。
            tokenization_kwargs: 传递给分词器 (tokenizer.encode) 的额外参数，用于覆盖默认行为。

        返回 (Returns)：
            List[RequestOutput]: 生成结果列表，顺序与输入的 prompts 顺序严格对应。
        """

        # ---------------------------------------------------------
        # 1. 任务类型校验：确保“专业对口”
        # ---------------------------------------------------------
        # 获取当前模型的运行器类型 (runner_type)。
        # vLLM 支持多种模式，比如 "generate" (生成文本) 或 "pooling" (提取向量)。
        runner_type = self.model_config.runner_type
        
        # 如果模型被配置为“非生成类”任务（例如它是一个专门提取特征的 Embedding 模型），
        # 却被调用了 .generate() 方法，则直接报错。
        if runner_type != "generate":
            raise ValueError(
                "LLM.generate() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model."
            )

        # ---------------------------------------------------------
        # 2. 默认参数补全
        # ---------------------------------------------------------
        # 如果用户没有传入任何采样参数 (sampling_params)，
        # 则去获取系统初始化时设定的默认参数。
        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        # ---------------------------------------------------------
        # 3. 核心转发：进入真正的执行流程
        # ---------------------------------------------------------
        # LLM.generate 实际上是一个“壳”，它所有的重活儿都交给了内部私有方法 `_run_completion`。
        # _run_completion 会负责：
        #   a. 把文字通过 InputProcessor 转成 ID
        #   b. 将请求发给刚才我们聊过的 EngineCore
        #   c. 收集结果并返回给用户
        return self._run_completion(
            prompts=prompts,
            params=sampling_params,
            output_type=RequestOutput,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            priority=priority,
        )

    def enqueue(
        self,
        prompts: PromptType | Sequence[PromptType],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[str]:
        """Enqueue prompts for generation without waiting for completion.

        This method adds requests to the engine queue but does not start
        processing them. Use wait_for_completion() to process the queued
        requests and get results.

        Args:
            prompts: The prompts to the LLM. See generate() for details.
            sampling_params: The sampling parameters for text generation.
            lora_request: LoRA request to use for generation, if any.
            priority: The priority of the requests, if any.
            use_tqdm: If True, shows a tqdm progress bar while adding requests.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of request IDs for the enqueued requests.
        """
        runner_type = self.model_config.runner_type
        if runner_type != "generate":
            raise ValueError("LLM.enqueue() is only supported for generative models.")

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        return self._add_completion_requests(
            prompts=prompts,
            params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            priority=priority,
            tokenization_kwargs=tokenization_kwargs,
        )

    @overload
    def wait_for_completion(
        self,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[RequestOutput | PoolingRequestOutput]: ...

    @overload
    def wait_for_completion(
        self,
        output_type: type[_O] | tuple[type[_O], ...],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[_O]: ...

    def wait_for_completion(
        self,
        output_type: type[Any] | tuple[type[Any], ...] | None = None,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[Any]:
        """Wait for all enqueued requests to complete and return results.

        This method processes all requests currently in the engine queue
        and returns their outputs. Use after enqueue() to get results.

        Args:
            output_type: The expected output type, defaults to RequestOutput.
            use_tqdm: If True, shows a tqdm progress bar.

        Returns:
            A list of output objects for all completed requests.
        """
        if output_type is None:
            output_type = (RequestOutput, PoolingRequestOutput)

        return self._run_engine(output_type, use_tqdm=use_tqdm)

    def _resolve_mm_lora(
        self,
        prompt: EngineInput,
        lora_request: LoRARequest | None,
    ) -> LoRARequest | None:
        if prompt["type"] != "multimodal":
            return lora_request

        lora_config = self.llm_engine.vllm_config.lora_config
        default_mm_loras = None if lora_config is None else lora_config.default_mm_loras
        if not default_mm_loras:
            return lora_request

        prompt_modalities = prompt["mm_placeholders"].keys()
        intersection = set(prompt_modalities).intersection(default_mm_loras.keys())
        if not intersection:
            return lora_request

        if len(intersection) > 1:
            # TODO: Would be nice to be able to have multiple loras per prompt
            logger.warning(
                "Multiple modality specific loras were registered and would be "
                "used by a single prompt consuming several modalities; "
                "currently we only support one lora per request; as such, "
                "lora(s) registered with modalities: %s will be skipped",
                intersection,
            )
            return lora_request

        # Build the LoRA request; the ID of the default mm lora is the
        # index of the modality name sorted alphabetically + 1.
        modality_name = intersection.pop()
        modality_lora_path = default_mm_loras[modality_name]
        modality_lora_id = sorted(default_mm_loras).index(modality_name) + 1

        # If we have a collision, warn if there is a collision,
        # but always send the explicitly provided request.
        if lora_request:
            if lora_request.lora_int_id != modality_lora_id:
                logger.warning(
                    "A modality with a registered lora and a lora_request "
                    "with a different ID were provided; falling back to the "
                    "lora_request as we only apply one LoRARequest per prompt"
                )
            return lora_request

        return LoRARequest(
            modality_name,
            modality_lora_id,
            modality_lora_path,
        )

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        """
        Execute an RPC call on all workers.

        Args:
            method: Name of the worker method to execute, or a callable that
                is serialized and sent to all workers to execute.

                If the method is a callable, it should accept an additional
                `self` argument, in addition to the arguments passed in `args`
                and `kwargs`. The `self` argument will be the worker object.
            timeout: Maximum time in seconds to wait for execution. Raises a
                [`TimeoutError`][] on timeout. `None` means wait indefinitely.
            args: Positional arguments to pass to the worker method.
            kwargs: Keyword arguments to pass to the worker method.

        Returns:
            A list containing the results from each worker.

        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """

        return self.llm_engine.collective_rpc(method, timeout, args, kwargs)

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        """
        Run a function directly on the model inside each worker,
        returning the result for each of them.

        !!! warning
            To reduce the overhead of data transfer, avoid returning large
            arrays or tensors from this method. If you must return them,
            make sure you move them to CPU first to avoid taking up additional
            VRAM!
        """
        return self.llm_engine.apply_model(func)

    def beam_search(
        self,
        prompts: list[TokensPrompt | TextPrompt],
        params: BeamSearchParams,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        use_tqdm: bool = False,
        concurrency_limit: int | None = None,
    ) -> list[BeamSearchOutput]:
        """
        Generate sequences using beam search.

        Args:
            prompts: A list of prompts. Each prompt can be a string or a list
                of token IDs.
            params: The beam search parameters.
            lora_request: LoRA request to use for generation, if any.
            use_tqdm: Whether to use tqdm to display the progress bar.
            concurrency_limit: The maximum number of concurrent requests.
                If None, the number of concurrent requests is unlimited.
        """
        # TODO: how does beam search work together with length penalty,
        # frequency, penalty, and stopping criteria, etc.?
        beam_width = params.beam_width
        max_tokens = params.max_tokens
        temperature = params.temperature
        ignore_eos = params.ignore_eos
        length_penalty = params.length_penalty

        tokenizer = self.renderer.get_tokenizer()
        eos_token_id = tokenizer.eos_token_id
        sort_beams_key = create_sort_beams_key_function(eos_token_id, length_penalty)

        engine_inputs = self._preprocess_cmpl(prompts)
        lora_requests = self._lora_request_to_seq(lora_request, len(engine_inputs))

        if use_tqdm and concurrency_limit is not None:
            logger.warning(
                "Progress bar is not supported when using concurrency_limit. "
                "Disabling progress bar."
            )
            use_tqdm = False

        if concurrency_limit is None:
            concurrency_limit = len(engine_inputs)

        # generate 2 * beam_width candidates at each step
        # following the huggingface transformers implementation
        # at https://github.com/huggingface/transformers/blob/e15687fffe5c9d20598a19aeab721ae0a7580f8a/src/transformers/generation/beam_search.py#L534 # noqa
        sampling_params = SamplingParams(
            logprobs=2 * beam_width,
            max_tokens=1,
            temperature=temperature,
            skip_clone=True,  # Internal beam search, safe to skip clone
        )
        instances: list[BeamSearchInstance] = []

        for lora_req, prompt in zip(lora_requests, engine_inputs):
            if prompt["type"] == "embeds":
                raise NotImplementedError(
                    "Embedding prompt not supported for beam search"
                )

            instances.append(
                BeamSearchInstance(
                    prompt,
                    lora_request=lora_req,
                    logprobs=None,
                ),
            )

        for prompt_start in range(0, len(instances), concurrency_limit):
            instances_batch = instances[prompt_start : prompt_start + concurrency_limit]

            token_iter = range(max_tokens)
            if use_tqdm:
                token_iter = tqdm(
                    token_iter, desc="Beam search", unit="token", unit_scale=False
                )
                logger.warning(
                    "The progress bar shows the upper bound on token steps and "
                    "may finish early due to stopping conditions. It does not "
                    "reflect instance-level progress."
                )
            for _ in token_iter:
                all_beams: list[BeamSearchSequence] = list(
                    sum((instance.beams for instance in instances_batch), [])
                )
                pos = [0] + list(
                    itertools.accumulate(
                        len(instance.beams) for instance in instances_batch
                    )
                )
                instance_start_and_end: list[tuple[int, int]] = list(
                    zip(pos[:-1], pos[1:])
                )

                if len(all_beams) == 0:
                    break

                # only runs for one step
                # we don't need to use tqdm here
                output = self._render_and_run_requests(
                    prompts=(beam.get_prompt() for beam in all_beams),
                    params=self._params_to_seq(sampling_params, len(all_beams)),
                    output_type=RequestOutput,
                    lora_requests=[beam.lora_request for beam in all_beams],
                    use_tqdm=False,
                )

                for (start, end), instance in zip(
                    instance_start_and_end, instances_batch
                ):
                    instance_new_beams = []
                    for i in range(start, end):
                        current_beam = all_beams[i]
                        result = output[i]

                        if result.outputs[0].logprobs is not None:
                            # if `result.outputs[0].logprobs` is None, it means
                            # the sequence is completed because of the
                            # max-model-len or abortion. we don't need to add
                            # it to the new beams.
                            logprobs = result.outputs[0].logprobs[0]
                            for token_id, logprob_obj in logprobs.items():
                                new_beam = BeamSearchSequence(
                                    current_beam.orig_prompt,
                                    tokens=current_beam.tokens + [token_id],
                                    logprobs=current_beam.logprobs + [logprobs],
                                    lora_request=current_beam.lora_request,
                                    cum_logprob=current_beam.cum_logprob
                                    + logprob_obj.logprob,
                                )

                                if token_id == eos_token_id and not ignore_eos:
                                    instance.completed.append(new_beam)
                                else:
                                    instance_new_beams.append(new_beam)
                    sorted_beams = sorted(
                        instance_new_beams, key=sort_beams_key, reverse=True
                    )
                    instance.beams = sorted_beams[:beam_width]

        outputs = []
        for instance in instances:
            instance.completed.extend(instance.beams)
            sorted_completed = sorted(
                instance.completed, key=sort_beams_key, reverse=True
            )
            best_beams = sorted_completed[:beam_width]

            for beam in best_beams:
                beam.text = tokenizer.decode(beam.tokens)

            outputs.append(BeamSearchOutput(sequences=best_beams))

        return outputs

    def _preprocess_cmpl(
        self,
        prompts: Sequence[PromptType],
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> Sequence[EngineInput]:
        """
        Convert prompt inputs from LLM APIs (other than [LLM.chat][]) into
        a format that can be passed to `_add_request`.

        Refer to [LLM.generate][] for a complete description of the arguments.

        Returns:
            A list of `EngineInput` objects ready to be passed into LLMEngine.
        """
        renderer = self.renderer
        model_config = self.model_config

        parsed_prompts = [
            parse_model_prompt(model_config, prompt) for prompt in prompts
        ]
        tok_params = renderer.default_cmpl_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )

        return renderer.render_cmpl(parsed_prompts, tok_params)

    def _preprocess_cmpl_one(
        self,
        prompt: PromptType,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> EngineInput:
        (engine_input,) = self._preprocess_cmpl([prompt], tokenization_kwargs)
        return engine_input

    def _preprocess_chat(
        self,
        conversations: Sequence[list[ChatCompletionMessageParam]],
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        chat_template_kwargs: dict[str, Any] | None = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> Sequence[EngineInput]:
        """
        Convert a list of conversations into prompts so that they can then
        be used as input for other LLM APIs.

        Refer to [LLM.chat][] for a complete description of the arguments.

        Returns:
            A list of `EngineInput` objects ready to be passed into LLMEngine.
        """
        renderer = self.renderer

        chat_params = ChatParams(
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            chat_template_kwargs=merge_kwargs(
                chat_template_kwargs,
                dict(
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    tools=tools,
                    tokenize=is_mistral_tokenizer(renderer.tokenizer),
                ),
            ),
        )
        tok_params = renderer.default_chat_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )

        _, engine_inputs = renderer.render_chat(
            conversations,
            chat_params,
            tok_params,
            prompt_extras={"mm_processor_kwargs": mm_processor_kwargs},
        )

        return engine_inputs

    def _preprocess_chat_one(
        self,
        conversation: list[ChatCompletionMessageParam],
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        chat_template_kwargs: dict[str, Any] | None = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> EngineInput:
        (engine_input,) = self._preprocess_chat(
            [conversation],
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
            tokenization_kwargs=tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        return engine_input

    def chat(
        self,
        messages: list[ChatCompletionMessageParam]
        | Sequence[list[ChatCompletionMessageParam]],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[RequestOutput]:
        """
        Generate responses for a chat conversation.

        The chat conversation is converted into a text prompt using the
        tokenizer and calls the [generate][vllm.LLM.generate] method to generate
        the responses.

        Multi-modal inputs can be passed in the same way you would pass them
        to the OpenAI API.

        Args:
            messages: A sequence of conversations or a single conversation.

                - Each conversation is represented as a list of messages.
                - Each message is a dictionary with 'role' and 'content' keys.

            sampling_params: The sampling parameters for text generation.
                If None, we use the default sampling parameters. When it
                is a single value, it is applied to every prompt. When it
                is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The template to use for structuring the chat.
                If not provided, the model's default chat template will be used.
            chat_template_content_format: The format to render message content.

                - "string" will render the content as a string.
                  Example: `"Who are you?"`
                - "openai" will render the content as a list of dictionaries,
                  similar to OpenAI schema.
                  Example: `[{"type": "text", "text": "Who are you?"}]`

            add_generation_prompt: If True, adds a generation template
                to each message.
            continue_final_message: If True, continues the final message in
                the conversation instead of starting a new one. Cannot be
                `True` if `add_generation_prompt` is also `True`.
            chat_template_kwargs: Additional kwargs to pass to the chat
                template.
            tokenization_kwargs: Overrides for `tokenizer.encode`.
            mm_processor_kwargs: Overrides for `processor.__call__`.

        Returns:
            A list of `RequestOutput` objects containing the generated
            responses in the same order as the input messages.
        """
        model_config = self.model_config
        runner_type = model_config.runner_type
        if runner_type != "generate":
            raise ValueError(
                "LLM.chat() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model."
            )

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        return self._run_chat(
            messages=messages,
            params=sampling_params,
            output_type=RequestOutput,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
            tokenization_kwargs=tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )

    def encode(
        self,
        prompts: PromptType | Sequence[PromptType] | DataPrompt,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        pooling_task: PoolingTask | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[PoolingRequestOutput]:
        """Apply pooling to the hidden states corresponding to the input
        prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            pooling_task: Override the pooling task to use.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `PoolingRequestOutput` objects containing the
            pooled hidden states in the same order as the input prompts.
        """

        self._verify_pooling_task(pooling_task)

        if isinstance(prompts, dict) and "data" in prompts:
            if self.io_processor is None:
                raise ValueError(
                    "No IOProcessor plugin installed. Please refer "
                    "to the documentation and to the "
                    "'prithvi_geospatial_mae_io_processor' "
                    "offline inference example for more details."
                )

            # Validate the request data is valid for the loaded plugin
            prompt_data = prompts.get("data")
            if prompt_data is None:
                raise ValueError(
                    "The 'data' field of the prompt is expected to contain "
                    "the prompt data and it cannot be None. "
                    "Refer to the documentation of the IOProcessor "
                    "in use for more details."
                )
            validated_prompt = self.io_processor.parse_data(prompt_data)

            # obtain the actual model prompts from the pre-processor
            prompts = self.io_processor.pre_process(prompt=validated_prompt)
            prompts_seq = prompt_to_seq(prompts)

            params_seq: Sequence[PoolingParams] = [
                self.io_processor.merge_pooling_params(param)
                for param in self._params_to_seq(
                    pooling_params,
                    len(prompts_seq),
                )
            ]
            for p in params_seq:
                if p.task is None:
                    p.task = "plugin"

            outputs = self._run_completion(
                prompts=prompts_seq,
                params=params_seq,
                output_type=PoolingRequestOutput,
                use_tqdm=use_tqdm,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
            )

            # get the post-processed model outputs
            assert self.io_processor is not None
            processed_outputs = self.io_processor.post_process(outputs)

            return [
                PoolingRequestOutput[Any](
                    request_id="",
                    outputs=processed_outputs,
                    num_cached_tokens=getattr(
                        processed_outputs, "num_cached_tokens", 0
                    ),
                    prompt_token_ids=[],
                    finished=True,
                )
            ]
        else:
            if pooling_params is None:
                # Use default pooling params.
                pooling_params = PoolingParams()

            prompts_seq = prompt_to_seq(prompts)
            params_seq = self._params_to_seq(pooling_params, len(prompts_seq))

            for param in params_seq:
                if param.task is None:
                    param.task = pooling_task
                elif param.task != pooling_task:
                    msg = (
                        f"You cannot overwrite {param.task=!r} with {pooling_task=!r}!"
                    )
                    raise ValueError(msg)

            if pooling_task in self.pooling_io_processors:
                io_processor = self.pooling_io_processors[pooling_task]
                processor_inputs = io_processor.pre_process_offline(
                    ctx=OfflineInputsContext(
                        prompts=prompts_seq, tokenization_kwargs=tokenization_kwargs
                    )
                )
                seq_lora_requests = self._lora_request_to_seq(
                    lora_request, len(prompts_seq)
                )
                seq_priority = self._priority_to_seq(None, len(prompts))

                self._render_and_add_requests(
                    prompts=processor_inputs,
                    params=params_seq,
                    lora_requests=seq_lora_requests,
                    priorities=seq_priority,
                )

                outputs = self._run_engine(
                    use_tqdm=use_tqdm, output_type=PoolingRequestOutput
                )
                outputs = io_processor.post_process_offline(
                    ctx=OfflineOutputsContext(outputs=outputs)
                )
            else:
                outputs = self._run_completion(
                    prompts=prompts_seq,
                    params=params_seq,
                    output_type=PoolingRequestOutput,
                    use_tqdm=use_tqdm,
                    lora_request=lora_request,
                    tokenization_kwargs=tokenization_kwargs,
                )
        return outputs

    def _verify_pooling_task(self, pooling_task: PoolingTask | None):
        if self.runner_type != "pooling":
            raise ValueError(
                "LLM.encode() is only supported for pooling models. "
                "Try passing `--runner pooling` to use the model as a "
                "pooling model."
            )

        if pooling_task is None:
            raise ValueError(
                "pooling_task required for `LLM.encode`\n"
                "Please use one of the more specific methods or set the "
                "pooling_task when using `LLM.encode`:\n"
                "  - For embeddings, use `LLM.embed(...)` "
                'or `pooling_task="embed"`.\n'
                "  - For classification logits, use `LLM.classify(...)` "
                'or `pooling_task="classify"`.\n'
                "  - For similarity scores, use `LLM.score(...)`.\n"
                "  - For rewards, use `LLM.reward(...)` "
                'or `pooling_task="token_classify"`\n'
                "  - For token classification, "
                'use `pooling_task="token_classify"`\n'
                '  - For multi-vector retrieval, use `pooling_task="token_embed"`'
            )

        if (
            pooling_task in ("embed", "token_embed")
            and pooling_task not in self.supported_tasks
        ):
            raise ValueError(
                "Embedding API is not supported by this model. "
                "Try converting the model using `--convert embed`."
            )

        if (
            pooling_task in ("classify", "token_classify")
            and pooling_task not in self.supported_tasks
        ):
            raise ValueError(
                "Classification API is not supported by this model. "
                "Try converting the model using `--convert classify`."
            )

        # plugin task uses io_processor.parse_request to verify inputs
        if pooling_task != "plugin" and pooling_task != self.pooling_task:
            if pooling_task not in self.supported_tasks:
                raise ValueError(
                    f"Unsupported task: {pooling_task!r} "
                    f"Supported tasks: {self.supported_tasks}"
                )
            else:
                logger.warning_once(
                    "Pooling multitask support is deprecated and will "
                    "be removed in v0.20. When the default pooling task is "
                    "not what you want, you need to manually specify it "
                    'via PoolerConfig(task="%s"). ',
                    pooling_task,
                )

    def embed(
        self,
        prompts: PromptType | Sequence[PromptType],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[EmbeddingRequestOutput]:
        """
        Generate an embedding vector for each prompt.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `EmbeddingRequestOutput` objects containing the
            embedding vectors in the same order as the input prompts.
        """

        items = self.encode(
            prompts,
            use_tqdm=use_tqdm,
            pooling_params=pooling_params,
            lora_request=lora_request,
            pooling_task="embed",
            tokenization_kwargs=tokenization_kwargs,
        )

        return [EmbeddingRequestOutput.from_base(item) for item in items]

    def classify(
        self,
        prompts: PromptType | Sequence[PromptType],
        *,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[ClassificationRequestOutput]:
        """
        Generate class logits for each prompt.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `ClassificationRequestOutput` objects containing the
            embedding vectors in the same order as the input prompts.
        """

        items = self.encode(
            prompts,
            use_tqdm=use_tqdm,
            pooling_params=pooling_params,
            lora_request=lora_request,
            pooling_task="classify",
            tokenization_kwargs=tokenization_kwargs,
        )

        return [ClassificationRequestOutput.from_base(item) for item in items]

    def reward(
        self,
        prompts: PromptType | Sequence[PromptType],
        /,
        *,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[PoolingRequestOutput]:
        """
        Generate rewards for each prompt.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `PoolingRequestOutput` objects containing the
            pooled hidden states in the same order as the input prompts.
        """
        return self.encode(
            prompts,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            pooling_params=pooling_params,
            pooling_task="token_classify",
            tokenization_kwargs=tokenization_kwargs,
        )

    def score(
        self,
        data_1: ScoreInput | list[ScoreInput],
        data_2: ScoreInput | list[ScoreInput],
        /,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        pooling_params: PoolingParams | None = None,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        chat_template: str | None = None,
    ) -> list[ScoringRequestOutput]:
        """Generate similarity scores for all pairs `<text,text_pair>` or
          `<multi-modal data, multi-modal data pair>`.

        The inputs can be `1 -> 1`, `1 -> N` or `N -> N`.
        In the `1 - N` case the `data_1` input will be replicated `N`
        times to pair with the `data_2` inputs.
        The input pairs are used to build a list of prompts for the
        cross encoder model. This class automatically batches the prompts,
        considering the memory constraint. For the best performance, put all
        of your inputs into a single list and pass it to this method.

        Supports both text and multi-modal data (images, etc.) when used with
        appropriate multi-modal models. For multi-modal inputs, ensure the
        prompt structure matches the model's expected input format.

        Args:
            data_1: Can be a single prompt, a list of prompts or
                `ScoreMultiModalParam`, which can contain either text or
                multi-modal data. When a list, it must have the same length as
                the `data_2` list.
            data_2: The data to pair with the query to form the input to
                the LLM. Can be text or multi-modal data. See [PromptType]
                [vllm.inputs.PromptType] for more details about the format of
                each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The chat template to use for the scoring. If None, we
                use the model's default chat template.
            tokenization_kwargs: Overrides for `tokenizer.encode`.
        Returns:
            A list of `ScoringRequestOutput` objects containing the
            generated scores in the same order as the input prompts.
        """

        if self.runner_type != "pooling":
            raise ValueError(
                "LLM.score() is only supported for pooling models. "
                "Try passing `--runner pooling` to use the model as a "
                "pooling model."
            )

        score_type = self.model_config.score_type
        if (
            score_type == "cross-encoder"
            and getattr(self.model_config.hf_config, "num_labels", 0) != 1
        ):
            raise ValueError("Scoring API is only enabled for num_labels == 1.")

        if score_type is None or score_type not in self.pooling_io_processors:
            raise ValueError("This model does not support the Scoring API.")

        io_processor = self.pooling_io_processors[score_type]
        assert isinstance(io_processor, ScoringIOProcessor)

        pooling_task = io_processor.pooling_task
        scoring_data = io_processor.valid_inputs(data_1, data_2)
        n_queries = len(scoring_data.data_1)

        ctx = OfflineInputsContext(
            prompts=scoring_data,
            pooling_params=pooling_params,
            tokenization_kwargs=tokenization_kwargs,
            chat_template=chat_template,
            n_queries=n_queries,
        )

        processor_inputs = io_processor.pre_process_offline(ctx)

        seq_lora_requests = self._lora_request_to_seq(
            lora_request, len(processor_inputs)
        )

        if ctx.pooling_params is None:
            ctx.pooling_params = PoolingParams()
        params_seq = self._params_to_seq(ctx.pooling_params, len(processor_inputs))

        for param in params_seq:
            if param.task is None:
                param.task = pooling_task
            elif param.task != pooling_task:
                msg = f"You cannot overwrite {param.task=!r} with {pooling_task=!r}!"
                raise ValueError(msg)

        seq_priority = self._priority_to_seq(None, len(processor_inputs))

        self._render_and_add_requests(
            prompts=processor_inputs,
            params=params_seq,
            lora_requests=seq_lora_requests,
            priorities=seq_priority,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm, output_type=PoolingRequestOutput)
        outputs = io_processor.post_process_offline(
            ctx=OfflineOutputsContext(outputs=outputs, n_queries=n_queries),
        )

        return [ScoringRequestOutput.from_base(item) for item in outputs]

    def start_profile(self, profile_prefix: str | None = None) -> None:
        """Start profiling with optional custom trace prefix.

        Args:
            profile_prefix: Optional prefix for the trace file names. If provided,
                           trace files will be named as "<prefix>_dp<X>_pp<Y>_tp<Z>".
                           If not provided, default naming will be used.
        """
        self.llm_engine.start_profile(profile_prefix)

    def stop_profile(self) -> None:
        self.llm_engine.stop_profile()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.llm_engine.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def sleep(self, level: int = 1, mode: PauseMode = "abort"):
        """
        Put the engine to sleep. The engine should not process any requests.
        The caller should guarantee that no requests are being processed
        during the sleep period, before `wake_up` is called.

        Args:
            level: The sleep level.
                - Level 0: Pause scheduling but continue accepting requests.
                           Requests are queued but not processed.
                - Level 1: Offload model weights to CPU, discard KV cache.
                           The content of kv cache is forgotten. Good for
                           sleeping and waking up the engine to run the same
                           model again. Please make sure there's enough CPU
                           memory to store the model weights.
                - Level 2: Discard all GPU memory (weights + KV cache).
                           Good for sleeping and waking up the engine to run
                           a different model or update the model, where
                           previous model weights are not needed. It reduces
                           CPU memory pressure.
            mode: How to handle any existing requests, can be "abort", "wait",
                or "keep".
        """
        self.llm_engine.sleep(level=level, mode=mode)

    def wake_up(self, tags: list[str] | None = None):
        """
        Wake up the engine from sleep mode. See the [sleep][vllm.LLM.sleep]
        method for more details.

        Args:
            tags: An optional list of tags to reallocate the engine memory
                for specific memory allocations. Values must be in
                `("weights", "kv_cache", "scheduling")`. If None, all memory
                is reallocated. wake_up should be called with all tags
                (or None) before the engine is used again.
                Use tags=["scheduling"] to resume from level 0 sleep.
        """
        self.llm_engine.wake_up(tags)

    def get_metrics(self) -> list["Metric"]:
        """Return a snapshot of aggregated metrics from Prometheus.

        Returns:
            A `MetricSnapshot` instance capturing the current state
            of all aggregated metrics from Prometheus.

        Note:
            This method is only available with the V1 LLM engine.
        """
        return self.llm_engine.get_metrics()

    def _params_to_seq(
        self,
        params: _P | Sequence[_P],
        num_requests: int,
    ) -> Sequence[_P]:
        if isinstance(params, Sequence):
            if len(params) != num_requests:
                raise ValueError(
                    f"The lengths of prompts ({params}) "
                    f"and params ({len(params)}) must be the same."
                )

            return params

        return [params] * num_requests

    def _lora_request_to_seq(
        self,
        lora_request: LoRARequest | None | Sequence[LoRARequest | None],
        num_requests: int,
    ) -> Sequence[LoRARequest | None]:
        if isinstance(lora_request, Sequence):
            if len(lora_request) != num_requests:
                raise ValueError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and lora_request ({len(lora_request)}) must be the same."
                )

            return lora_request

        return [lora_request] * num_requests

    def _priority_to_seq(
        self,
        priority: list[int] | None,
        num_requests: int,
    ) -> Sequence[int]:
        if priority is not None:
            if len(priority) != num_requests:
                raise ValueError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and priority ({len(priority)}) must be the same."
                )

            return priority

        return [0] * num_requests

    def _add_completion_requests(
        self,
        prompts: "PromptType" | Sequence["PromptType"],
        params: "SamplingParams"
        | "PoolingParams"
        | Sequence["SamplingParams" | "PoolingParams"],
        *,
        use_tqdm: bool | Callable[..., "tqdm"] = True,
        lora_request: Sequence["LoRARequest"] | "LoRARequest" | None = None,
        priority: list[int] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        内部方法：将原始输入转化为标准序列，预处理提示词，并将其正式注册到引擎中。
        """

        # ---------------------------------------------------------
        # 1. 输入标准化 (Normalization)
        # ---------------------------------------------------------
        # 将输入强制转换为序列格式。
        # 即使你只传了一个字符串 "Hello"，prompt_to_seq 也会把它变成 ["Hello"]。
        seq_prompts = prompt_to_seq(prompts)
        
        # 将采样参数、LoRA 请求和优先级也转为序列，并确保它们的长度与提示词数量对齐。
        # 例如：如果你传了 10 个提示词但只传了 1 组参数，这里会自动将该参数复制 10 份。
        seq_params = self._params_to_seq(params, len(seq_prompts))
        seq_lora_requests = self._lora_request_to_seq(lora_request, len(seq_prompts))
        seq_priority = self._priority_to_seq(priority, len(seq_prompts))

        # ---------------------------------------------------------
        # 2. 预处理与渲染 (Rendering)
        # ---------------------------------------------------------
        # 这里使用了生成器表达式，对每一个 prompt 进行迭代处理。
        # maybe_tqdm 是一个工具，如果 use_tqdm=True，它就会在终端打印出渲染进度条。
        
        processed_prompts = (
            # _preprocess_cmpl_one 是关键：
            # 它负责把原始提示词（可能是文本、字典或图片）转换成 vLLM 内部通用的输入格式。
            # 如果需要分词（Tokenization），也会在这里进行初步处理。
            self._preprocess_cmpl_one(prompt, tokenization_kwargs)
            for prompt in maybe_tqdm(
                seq_prompts,
                use_tqdm=use_tqdm,
                desc="Rendering prompts", # 进度条显示的描述文字
            )
        )

        # ---------------------------------------------------------
        # 3. 提交给下一级处理器
        # ---------------------------------------------------------
        # 将预处理完的所有数据（提示词流、对齐后的参数、LoRA、优先级）
        # 传给 _render_and_add_requests 方法。
        # 该方法会为每个请求分配 Request ID，并真正塞进 EngineCore 的待处理队列中。
        return self._render_and_add_requests(
            prompts=processed_prompts,
            params=seq_params,
            lora_requests=seq_lora_requests,
            priorities=seq_priority,
        )
    
    def _run_completion(
        self,
        prompts: "PromptType" | Sequence["PromptType"],
        params: "SamplingParams" 
        | "PoolingParams" 
        | Sequence["SamplingParams" | "PoolingParams"],
        output_type: type["_O"],  # 期望的输出类型（如 RequestOutput 或 PoolingOutput）
        *,
        use_tqdm: bool | Callable[..., "tqdm"] = True,
        lora_request: Sequence["LoRARequest"] | "LoRARequest" | None = None,
        priority: list[int] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ):
        """
        内部核心执行方法：负责将输入请求送入引擎并循环直到生成结束。
        """

        # ---------------------------------------------------------
        # 第一步：把“订单”加入待办队列 (_add_completion_requests)
        # ---------------------------------------------------------
        # 这个方法干了三件脏活：
        # 1. 格式化：把各种奇奇怪怪的输入格式统一化。
        # 2. Tokenize：调用分词器把文字变成数字 ID。
        # 3. 注册：为每个提示词生成唯一 RequestID，并塞进底层引擎的待处理池里。
        self._add_completion_requests(
            prompts=prompts,
            params=params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            priority=priority,
            tokenization_kwargs=tokenization_kwargs,
        )

        # ---------------------------------------------------------
        # 第二步：正式运转引擎 (_run_engine)
        # ---------------------------------------------------------
        # 这是真正的“计算循环”发生的地方。
        # 只要引擎里还有没跑完的请求，这个方法就会不停地：
        #   a. 命令 GPU 跑一次前向传播 (Step)。
        #   b. 拿到这一轮出来的 Token。
        #   c. 检查是不是有人已经生成完了（比如遇到了停止词 <|endoftext|>）。
        #   d. 最终把所有结果打包成 output_type 指定的列表返回。
        return self._run_engine(use_tqdm=use_tqdm, output_type=output_type)

    def _run_chat(
        self,
        messages: list[ChatCompletionMessageParam]
        | Sequence[list[ChatCompletionMessageParam]],
        params: SamplingParams
        | PoolingParams
        | Sequence[SamplingParams | PoolingParams],
        output_type: type[_O],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ):
        seq_convs = conversation_to_seq(messages)
        seq_params = self._params_to_seq(params, len(seq_convs))
        seq_lora_requests = self._lora_request_to_seq(lora_request, len(seq_convs))

        return self._render_and_run_requests(
            prompts=(
                self._preprocess_chat_one(
                    conversation,
                    chat_template=chat_template,
                    chat_template_content_format=chat_template_content_format,
                    chat_template_kwargs=chat_template_kwargs,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    tools=tools,
                    tokenization_kwargs=tokenization_kwargs,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
                for conversation in maybe_tqdm(
                    seq_convs,
                    use_tqdm=use_tqdm,
                    desc="Rendering conversations",
                )
            ),
            params=seq_params,
            output_type=output_type,
            lora_requests=seq_lora_requests,
            use_tqdm=use_tqdm,
        )

    def _render_and_run_requests(
        self,
        prompts: Iterable[EngineInput],
        params: Sequence[SamplingParams | PoolingParams],
        output_type: type[_O],
        *,
        lora_requests: Sequence[LoRARequest | None] | None = None,
        priorities: Sequence[int] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ):
        if isinstance(prompts, (list, tuple)):
            logger.warning_once(
                "Rendering all prompts before adding them to the engine "
                "is less efficient than performing both on the same prompt "
                "before processing the next prompt. You should instead pass "
                "a generator that renders one prompt per iteration, as that allows "
                "engine execution to begin for the first prompt while processing "
                "the next prompt."
            )

        self._render_and_add_requests(
            prompts=prompts,
            params=params,
            lora_requests=lora_requests,
            priorities=priorities,
        )

        return self._run_engine(output_type, use_tqdm=use_tqdm)

    def _render_and_add_requests(
            self,
            prompts: Iterable["EngineInput"],               # 经过预处理的可迭代输入流
            params: Sequence["SamplingParams" | "PoolingParams"], # 对齐后的采样/池化参数序列
            *,
            lora_requests: Sequence["LoRARequest | None"] | None = None, # 选填：LoRA 适配器列表
            priorities: Sequence[int] | None = None,        # 选填：每个请求的优先级
        ) -> list[str]:
            # 用于记录成功提交到引擎的所有请求 ID，方便出错时统一销毁
            added_request_ids: list[str] = []

            try:
                # 1. 开始循环处理每一个请求
                for i, prompt in enumerate(prompts):
                    # 2. 调用内部方法 _add_request 真正将单个请求塞进引擎
                    # 这里会发生：分配 Request ID、将请求转化为 EngineCoreRequest
                    request_id = self._add_request(
                        prompt,
                        params[i],  # 对应索引的参数
                        # -----------------------------------------------------
                        # 3. 多模态 LoRA 解析 (_resolve_mm_lora)
                        # -----------------------------------------------------
                        # 如果是多模态模型（如图文），某些 LoRA 适配器可能需要特殊处理。
                        # 这里会根据 prompt 内容和用户传入的 LoRA 请求，解析出最终适用的 LoRA 配置。
                        lora_request=self._resolve_mm_lora(
                            prompt,
                            None if lora_requests is None else lora_requests[i],
                        ),
                        # 如果没有指定优先级，默认给 0 (普通)
                        priority=0 if priorities is None else priorities[i],
                    )
                    
                    # 将生成的 ID 存入临时列表
                    added_request_ids.append(request_id)

            except Exception as e:
                # ---------------------------------------------------------
                # 4. 原子性保证：失败回滚 (Fail-safe)
                # ---------------------------------------------------------
                # 这是一个非常硬核的设计。
                # 假设你一次性提交 100 个任务，前 50 个成功了，但第 51 个因为显存瞬间爆了或其他原因挂了。
                # 此时，前 50 个任务已经在 GPU 进程里开始跑了。
                # 为了不浪费资源和保持状态一致，如果中间报错，要把已经提交成功的任务全部“撤回 (Abort)”。
                if added_request_ids:
                    # 告诉底层引擎：这一批任务作废，停止它们的计算并释放内存
                    self.llm_engine.abort_request(added_request_ids, internal=True)
                
                # 继续向上抛出异常，让用户知道提交失败了
                raise e

            # 返回所有成功注册的任务 ID 列表
            return added_request_ids

    def _add_request(
        self,
        prompt: "EngineInput",                # 已经预处理好的输入（可能是 Token ID 或 文本+图片）
        params: "SamplingParams | PoolingParams", # 生成参数（温度、Top-P 等）
        lora_request: "LoRARequest | None" = None, # 选填：特定的 LoRA 权重
        priority: int = 0,                    # 优先级（数字越小通常优先级越高）
    ) -> str:
        """
        内部方法：为单个请求分配 ID，进行最后的参数微调，并将其正式提交给底层引擎。
        """

        # ---------------------------------------------------------
        # 1. 离线模式优化：只关心最终结果
        # ---------------------------------------------------------
        # 如果是生成任务（SamplingParams），这里会将输出类型强制设为 FINAL_ONLY。
        # 为什么？
        # 在离线批处理（Offline Inference）中，我们通常不需要像 ChatGPT 聊天那样“一个字一个字”地流式显示。
        # 设置为 FINAL_ONLY 可以让底层引擎减少不必要的中间数据传输（IPC 往返），从而提升批量处理的吞吐量。
        if isinstance(params, SamplingParams):
            # 告诉引擎：别给我发中间过程，等整个句子写完了再一起给我。
            params.output_kind = RequestOutputKind.FINAL_ONLY

        # ---------------------------------------------------------
        # 2. 唯一身份标识生成 (ID Allocation)
        # ---------------------------------------------------------
        # 调用类属性中维护的 self.request_counter。
        # next() 会让计数器自增 1，并转换成字符串。
        # 这样确保了在同一个 LLM 实例中，每个请求都有一个唯一的“准考证号”。
        request_id = str(next(self.request_counter))

        # ---------------------------------------------------------
        # 3. 跨越边界：提交给底层核心引擎 (The Hand-off)
        # ---------------------------------------------------------
        # 这里正式调用了我们在初始化时创建的 self.llm_engine（即 LLMEngine 实例）。
        # 一旦调用了这个 add_request，这个任务就正式进入了底层的：
        #   - 调度器队列 (Scheduler Queue)
        #   - 显存分配流程 (Block Manager)
        # 此时，主线程只需要拿着这个 request_id，之后就可以去索要结果了。
        return self.llm_engine.add_request(
            request_id,
            prompt,
            params,
            lora_request=lora_request,
            priority=priority,
        )

    def _run_engine(
        self,
        output_type: type[_O] | tuple[type[_O], ...],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[_O]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            pbar = tqdm_func(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, output: {0:.2f} toks/s"),
            )

        # Run the engine.
        outputs: list[_O] = []
        total_in_toks = 0
        total_out_toks = 0
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                assert isinstance(output, output_type)
                if output.finished:
                    outputs.append(output)  # type: ignore[arg-type]
                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            # Calculate tokens only for RequestOutput
                            n = len(output.outputs)
                            assert output.prompt_token_ids is not None
                            total_in_toks += len(output.prompt_token_ids) * n
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]
                            total_out_toks += sum(
                                len(stp.token_ids) for stp in output.outputs
                            )
                            out_spd = total_out_toks / pbar.format_dict["elapsed"]
                            pbar.postfix = (
                                f"est. speed input: {in_spd:.2f} toks/s, "
                                f"output: {out_spd:.2f} toks/s"
                            )
                            pbar.update(n)
                        else:
                            pbar.update(1)
                        if pbar.n == num_requests:
                            pbar.refresh()

        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))

    def init_weight_transfer_engine(
        self, request: WeightTransferInitRequest | dict
    ) -> None:
        """
        Initialize weight transfer for RL training.

        Args:
            request: Weight transfer initialization request with backend-specific info
        """
        init_info_dict = (
            request["init_info"] if isinstance(request, dict) else request.init_info
        )

        self.llm_engine.collective_rpc(
            "init_weight_transfer_engine", kwargs={"init_info": init_info_dict}
        )

    def update_weights(self, request: WeightTransferUpdateRequest | dict) -> None:
        """
        Update the weights of the model.

        Args:
            request: Weight update request with backend-specific update info
        """
        update_info_dict = (
            request["update_info"] if isinstance(request, dict) else request.update_info
        )

        self.llm_engine.collective_rpc(
            "update_weights", kwargs={"update_info": update_info_dict}
        )

    def __repr__(self) -> str:
        """Return a transformers-style hierarchical view of the model."""
        # Cache the result to avoid repeated collective_rpc calls
        if self._cached_repr is None:
            results = self.llm_engine.collective_rpc("get_model_inspection")
            # In distributed settings, we get results from all workers
            # Just return the first one (they should all be the same)
            if results:
                self._cached_repr = results[0]
            else:
                self._cached_repr = f"LLM(model={self.model_config.model!r})"
        return self._cached_repr
