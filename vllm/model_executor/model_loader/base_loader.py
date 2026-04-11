# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.reload import finalize_layerwise_processing
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.platforms import current_platform
from vllm.tracing import instrument
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """
        [抽象方法] 将权重加载到模型中。
        子类（如 DefaultModelLoader）必须实现此方法。
        
        该 API 允许“原地”（inplace）加载，即直接修改已经初始化好的 nn.Module 对象，
        将磁盘上的权重填入模型的张量中。
        """
        raise NotImplementedError

    @instrument(span_name="Load model") # 分布式追踪：记录整个模型加载过程的耗时
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        """根据给定的配置加载模型并返回。"""
        
        # --- 1. 设备环境准备 ---
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        
        # 确定加载权重的设备。如果 load_config 没指定，默认使用训练/推理的主设备。
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)

        # --- 2. 模型初始化（创建空壳） ---
        # set_default_torch_dtype: 确保在创建模型时，未显式指定类型的张量使用模型预设精度（如 BF16/FP16）
        with set_default_torch_dtype(model_config.dtype):
            # 进入目标设备上下文（如 cuda:0），确保模型张量直接在该设备上分配
            with target_device:
                # 按照模型架构（Llama, Qwen等）实例化 PyTorch nn.Module。
                # 此时模型参数通常是随机初始化的，或者是 Meta Tensor（元张量）。
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )

            # 调试日志：检查模型结构（如每一层的定义是否符合预期）
            log_model_inspection(model)

            # --- 3. 权重注入（核心动作） ---
            logger.debug("Loading weights on %s ...", load_device)
            
            # 【关键点】调用子类实现的 load_weights。
            # 虽然这段代码在 BaseModelLoader 中，但 self 指向的是具体的子类实例。
            # DefaultModelLoader 会在这里接管，执行多线程读取磁盘文件并填充到 model 中。
            self.load_weights(model, model_config)

            # --- 4. 显存监控（针对 GPU） ---
            # 记录加载完权重后的显存峰值。
            # 这对于 vLLM 极为重要，因为后续分配 KV Cache 显存池时需要扣除这部分已占用的空间。
            if current_platform.is_cuda():
                # 获取当前设备分配的最大显存
                peak_memory = torch.accelerator.max_memory_allocated()
                logger.debug_once(
                    "Peak GPU memory after loading weights: %s GiB",
                    format_gib(peak_memory),
                    scope="local",
                )

            # --- 5. 权重后处理与量化对齐 ---
            # 如果模型启用了“在线量化”（Online Quantization），
            # 权重加载后通常需要进行最后一层处理（如计算缩放因子或转换格式）。
            if _has_online_quant(model):
                # 逐层完成量化计算的收尾工作
                finalize_layerwise_processing(model, model_config)

            # 通用的后处理逻辑（如针对特定算子的张量重新排列/Layout 转换）
            process_weights_after_loading(model, model_config, target_device)

        # --- 6. 切换至推理模式 ---
        # 调用 eval() 关闭 Dropout 和 BatchNorm，确保推理行为的一致性
        return model.eval()


def log_model_inspection(model: nn.Module) -> None:
    """Log model structure if VLLM_LOG_MODEL_INSPECTION=1."""
    if not envs.VLLM_LOG_MODEL_INSPECTION:
        return

    from vllm.model_inspection import format_model_inspection

    logger.info("vLLM model structure:\n%s", format_model_inspection(model))


def _has_online_quant(model: nn.Module):
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if getattr(quant_method, "uses_meta_device", False):
            return True

    return False
