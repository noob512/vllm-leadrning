# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for selecting and loading models."""

import inspect
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from typing_extensions import assert_never

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.model_loader.reload import (
    record_metadata_for_reloading,
    set_torchao_reload_attrs,
)
from vllm.model_executor.models.interfaces import SupportsQuant
from vllm.tracing import instrument
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


@instrument(span_name="Initialize model") # 分布式追踪：记录这整个函数执行（也就是模型初始化阶段）所花费的时间
def initialize_model(
    vllm_config: VllmConfig, # vLLM 全局配置大对象，包含模型配置、缓存配置、并行配置等
    *,
    prefix: str = "", # 模块名称前缀，通常在处理多模态或复杂嵌套模型时，用来区分不同子模块的参数名
    model_class: type[nn.Module] | None = None, # 目标模型的 PyTorch 类（比如 LlamaForCausalLM）。如果不传，会去 config 里猜
    model_config: ModelConfig | None = None, # 专门关于模型结构（层数、维度等）的配置对象
) -> nn.Module:
    """Initialize a model with the given configurations.
    使用给定的配置来实例化模型的“骨架”（注意：这里通常只是建立网络结构，还没有加载权重数据）
    """
    
    # --- 1. 配置预处理与模型架构推断 ---
    
    # 如果没传具体的模型配置，就从全局的 vllm_config 里拿默认的
    if model_config is None:
        model_config = vllm_config.model_config
        
    # 如果没传该用哪个 PyTorch 类来建模型，就去查字典。
    # get_model_architecture 会读取 config.json 里的 "architectures" 字段（如 ["LlamaForCausalLM"]），
    # 然后在 vLLM 自己注册的模型库里找到对应的实际 Python 类。
    if model_class is None:
        model_class, _ = get_model_architecture(model_config)

    # --- 2. 量化配置预处理 ---
    # 如果开启了量化（比如 AWQ, GPTQ 等），提前设置一下。
    # 这步很重要，因为量化模型的某些层（比如 Linear）会被替换成专门的量化算子类（比如 QLinear）。
    if vllm_config.quant_config is not None:
        configure_quant_config(vllm_config.quant_config, model_class)

    # --- 3. 探查类的构造函数（区分新老 API） ---
    
    # 使用 Python 内置的 inspect 模块，获取刚刚找到的那个 model_class 类的 __init__ 方法长什么样
    signatures = inspect.signature(model_class.__init__)
    # 提取它需要接收的所有参数名字（是个列表）
    all_params = [param.name for param in signatures.parameters.values()]
    
    # --- 4. 新版 vLLM 模型类的初始化路径 ---
    
    # vLLM 的最新规范是：所有模型类都应该统一只接收 vllm_config 和 prefix 两个核心参数。
    if "vllm_config" in all_params and "prefix" in all_params:
        # new-style model class
        # 建立一个上下文，把当前的 vllm_config 设为全局线程可见，方便底层算子随时取用。
        with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
            # 实例化大模型（也就是搭起那些空壳参数层）
            model = model_class(vllm_config=vllm_config, prefix=prefix)
            # 记录一些元数据，主要是为了以后在不重启进程的情况下，支持动态重载模型（比如热切换不同的 LoRA 或权重）。
            record_metadata_for_reloading(model)
            # 成功返回搭好的骨架
            return model

    # --- 5. 兼容老版（第三方）模型类的回退路径 ---
    
    # 如果执行到了这里，说明刚才那个 if 没进去，也就是找到的类，它的 __init__ 参数不符合新规范。
    # 这通常是因为用户自己写了一个老版本的模型实现，并且强行注册进了 vLLM。
    
    # 打印黄色的弃用警告，提醒开发者赶紧改代码，符合最新的设计规范。
    msg = (
        "vLLM model class should accept `vllm_config` and `prefix` as "
        "input arguments. Possibly you have an old-style model class"
        " registered from out of tree and it is used for new vLLM version. "
        "Check https://docs.vllm.ai/en/latest/design/arch_overview.html "
        "for the design and update the model class accordingly."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)

    logger.warning(
        "Trying to guess the arguments for old-style model class %s",
        model_class,
    )
    
    # try to be compatible with old-style model class
    # 既然是老类，我们就来“猜”它需要什么参数。准备一个空字典。
    kwargs: dict[str, Any] = {}
    
    # 老版本的类，通常是要啥传啥。我们看看它的 __init__ 签名里有哪些名字，有的就强行塞进去。
    if "prefix" in all_params:
        kwargs["prefix"] = prefix
    if "config" in all_params: # 这里注意，老代码可能直接要 HuggingFace 的 config 对象
        kwargs["config"] = model_config.hf_config
    if "cache_config" in all_params:
        kwargs["cache_config"] = vllm_config.cache_config
    if "quant_config" in all_params:
        kwargs["quant_config"] = vllm_config.quant_config
    if "lora_config" in all_params:
        kwargs["lora_config"] = vllm_config.lora_config
    if "scheduler_config" in all_params:
        kwargs["scheduler_config"] = vllm_config.scheduler_config
        
    # 参数猜完并打包好之后，依然开启上下文
    with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
        # 把字典拆包(**kwargs)，扔给这个老版本类去初始化
        model = model_class(**kwargs)
        record_metadata_for_reloading(model)

    return model


def process_weights_after_loading(
    model: nn.Module, model_config: ModelConfig, target_device: torch.device
) -> None:
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            # When quant methods need to process weights after loading
            # (for repacking, quantizing, etc), they expect parameters
            # to be on the global target device. This scope is for the
            # case where cpu offloading is used, where we will move the
            # parameters onto device for processing and back off after.
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)

    # Initialize post-load attention weights for both Attention and MLA.
    # NOTE: Happens after other modules so we can easily decompress weights.
    for _, module in model.named_modules():
        if isinstance(module, (Attention, MLAAttention)) and hasattr(
            module, "process_weights_after_loading"
        ):
            # TODO(lucas): see if there is a way to unify the signatures
            # of process_weights_after_loading
            with device_loading_context(module, target_device):
                module.process_weights_after_loading(model_config.dtype)

    # Needed for torchao model reloading via model.reload_weights
    # @kylesayrs @jerryzh168 this can be removed if callers move to `reload_weights`
    if model_config.quantization == "torchao":
        set_torchao_reload_attrs(model, model_config)


@contextmanager
def device_loading_context(module: torch.nn.Module, target_device: torch.device):
    if target_device.type == "cpu":
        # If target is CPU, no need to move anything
        yield module
        return

    original_device_states: dict[str, torch.device] = {}
    uva_offloaded_parameters: list[str] = []

    # Store original device states and move parameters to GPU if they're on CPU
    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            original_device_states[name] = p.device
            p.data = p.data.to(target_device)
        if getattr(p, "_vllm_is_uva_offloaded", False):
            uva_offloaded_parameters.append(name)
        # Parameters already on target device are not touched

    try:
        yield module

    finally:
        use_pin_memory = (
            is_pin_memory_available()
            and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY
        )
        # Restore parameters to their original devices, ignoring new parameters
        for name, p in module.named_parameters():
            if name in original_device_states:
                original_device: torch.device = original_device_states[name]
                p.data = p.data.to(original_device)

            # parameter is UVA offloaded, but was replaced with a new device tensor
            # re-offload it to CPU using UVA
            if name in uva_offloaded_parameters and not getattr(
                p, "_vllm_is_uva_offloaded", False
            ):
                cpu_data = p.data.to(device="cpu")
                if use_pin_memory:
                    cpu_data = cpu_data.pin_memory()
                p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                p._vllm_is_uva_offloaded = True


_MODEL_ARCH_BY_HASH = dict[int, tuple[type[nn.Module], str]]()
"""Caches the outputs of `_get_model_architecture`."""


def _get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    from vllm.model_executor.models.adapters import as_embedding_model, as_seq_cls_model

    architectures = getattr(model_config.hf_config, "architectures", [])

    model_cls, arch = model_config.registry.resolve_model_cls(
        architectures,
        model_config=model_config,
    )

    if arch == model_config._get_transformers_backend_cls():
        assert model_config.model_impl != "vllm"
        if model_config.model_impl == "auto":
            logger.warning_once(
                "%s has no vLLM implementation, falling back to Transformers "
                "implementation. Some features may not be supported and "
                "performance may not be optimal.",
                arch,
            )

    convert_type = model_config.convert_type
    if convert_type == "none":
        pass
    elif convert_type == "embed":
        logger.debug_once("Converting to embedding model.")
        model_cls = as_embedding_model(model_cls)
    elif convert_type == "classify":
        logger.debug_once("Converting to sequence classification model.")
        model_cls = as_seq_cls_model(model_cls)
    else:
        assert_never(convert_type)

    return model_cls, arch


def get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    """
    获取模型架构类（如 LlamaForCausalLM）和架构名称。
    这是一个带缓存的“前端”代理函数，利用哈希值来加速查询。
    """
    
    # --- 第一步：生成“指纹”（哈希值） ---
    # 把能够唯一决定一个模型架构的所有关键属性打包在一起，生成一个独一无二的 Hash 键值。
    key = hash(
        (
            model_config.model,               # 模型的路径或名字 (例如: "meta-llama/Llama-2-7b")
            model_config.convert_type,        # 权重转换类型（是否需要特定的格式转换）
            model_config.runner_type,         # 运行器类型（是生成模型、池化模型还是多模态模型等）
            model_config.trust_remote_code,   # 安全标识（是否允许执行 HuggingFace 上的自定义 Python 代码）
            model_config.model_impl,          # 内部实现标记（vLLM 可能有同一模型的多种底层实现）
            
            # 从 config.json 中提取的架构列表（如 ["LlamaForCausalLM"]）。
            # 【关键细节】：因为 Python 中的列表（List）是可变的，无法被 hash()，
            # 所以必须强制转换成不可变的元组（Tuple）才能作为字典的 Key。
            tuple(getattr(model_config.hf_config, "architectures", [])),
        )
    )
    
    # --- 第二步：查缓存（拦截重复劳动） ---
    # _MODEL_ARCH_BY_HASH 是一个全局字典。
    # 如果这个“指纹”之前已经来过，直接从字典里拿出结果返回，瞬间完成。
    if key in _MODEL_ARCH_BY_HASH:
        return _MODEL_ARCH_BY_HASH[key]

    # --- 第三步：干苦力活（缓存未命中） ---
    # 如果是第一次见到这个“指纹”，就去调用那个带下划线的真正的底层函数。
    # _get_model_architecture 内部包含非常复杂的逻辑：它要去读取文件、解析 JSON、
    # 遍历 vLLM 庞大的已注册模型注册表（Registry），寻找匹配的类。这个过程相对耗时。
    model_arch = _get_model_architecture(model_config)
    
    # --- 第四步：记录结果并返回 ---
    # 拿到结果后，把它存到全局字典里。下次同样的配置再来要架构类，就不用再执行第三步了。
    _MODEL_ARCH_BY_HASH[key] = model_arch
    return model_arch


def get_model_cls(model_config: ModelConfig) -> type[nn.Module]:
    return get_model_architecture(model_config)[0]


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]


@dataclass
class ParamMapping:
    """
    A class to handle parameter mapping for model weight loading.
    It creates a bidirectional mapping between packed parameters and their
    constituent parts.
    """

    packed_mapping: dict[str, list[str]]
    inverse_packed_mapping: dict[str, tuple[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        for packed_name, sub_params in self.packed_mapping.items():
            # Skip self-contained cases (e.g., {"W_pack": ["W_pack"]})
            if len(sub_params) == 1 and sub_params[0] == packed_name:
                continue
            for index, param_name in enumerate(sub_params):
                self.inverse_packed_mapping[param_name] = (
                    packed_name,
                    index,
                )

    def get_sub_modules(self, module_name: str) -> tuple[str, list[str]] | None:
        for key, value in self.packed_mapping.items():
            if module_name.endswith(key):
                return key, value
        return None


def configure_quant_config(
    quant_config: QuantizationConfig, model_class: type[nn.Module]
):
    """
    Pass packed_modules_mapping by reference to quant_config so that
    quant_config can properly match fused modules

    Note that model attributes are passed by reference to quant_config,
    enabling them to be updated by model_class.__new__ (ex. chatglm, qwen)

    Once the `SupportsQuant` mixin has been added to all models, this
    function can be removed
    """
    if not issubclass(model_class, SupportsQuant):
        hf_to_vllm_mapper = getattr(model_class, "hf_to_vllm_mapper", None)
        packed_mapping = getattr(model_class, "packed_modules_mapping", None)

        # pass mappings by reference to quant_config
        if hf_to_vllm_mapper is not None:
            quant_config.apply_vllm_mapper(hf_to_vllm_mapper)
        if packed_mapping is not None:
            quant_config.packed_modules_mapping = packed_mapping
