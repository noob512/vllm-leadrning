# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/offloader.py
"""Base classes for model parameter offloading."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import TYPE_CHECKING

import torch.nn as nn

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import OffloadConfig

logger = init_logger(__name__)


"""
class relation:

BaseOffloader (ABC)
  * implemented by: UVAOffloader
  * implemented by: PrefetchOffloader
    * uses: _ModuleOffloader
        * uses: _BaseParamOffloader (ABC)
            * implemented by: _CpuParamOffloader
"""


class BaseOffloader(ABC):
    """Base class for model parameter offloading strategies.

    Offloaders control how model parameters are stored and loaded during
    inference. Different strategies trade memory for compute/transfer time.
    """

    @abstractmethod
    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Wrap modules with offloading logic.

        Args:
            modules_generator: Generator yielding modules to potentially offload.

        Returns:
            List of modules, potentially with offloading hooks installed.
        """
        pass

    def post_init(self):
        """Called after model construction completes.

        Offloaders can use this to:
        - Finalize parameter storage
        - Start initial prefetching
        - Allocate shared resources
        """
        return

    def sync_prev_onload(self) -> None:  # noqa: B027
        """Sync previous onload operations. Override in subclasses."""
        pass

    def join_after_forward(self) -> None:  # noqa: B027
        """Join streams after forward. Override in subclasses."""
        pass

    def _wait_for_layer(self, layer_idx: int) -> None:  # noqa: B027
        """Wait for layer prefetch. Override in subclasses."""
        pass

    def _start_prefetch(self, layer_idx: int) -> None:  # noqa: B027
        """Start layer prefetch. Override in subclasses."""
        pass


class NoopOffloader(BaseOffloader):
    """No-op offloader that returns modules as-is without any offloading."""

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Return modules unchanged."""
        return list(modules_generator)


# Global singleton offloader instance (defaults to no-op).
_instance: BaseOffloader = NoopOffloader()


def get_offloader() -> BaseOffloader:
    """Get the global offloader instance."""
    return _instance


def set_offloader(instance: BaseOffloader) -> None:
    """Set the global offloader instance."""
    global _instance
    _instance = instance
    if isinstance(instance, NoopOffloader):
        logger.debug_once(
            "Offloader set to NoopOffloader (no offloading).", scope="local"
        )
    else:
        logger.info_once("Offloader set to %s", type(instance).__name__, scope="local")


def create_offloader(offload_config: "OffloadConfig") -> BaseOffloader:
    """
    根据传入的卸载配置 (offload_config) 创建并返回具体的卸载器 (Offloader) 实例。

    【路由逻辑】
    如果用户明确指定了 `offload_backend` (例如强制设为 "uva" 或 "prefetch")，则直接使用指定的后端。
    如果设为默认的 ``"auto"``，则按照以下优先级自动选择：
    1. 预取优先：如果配置了 `offload_group_size > 0`，选择 prefetch (预取) 后端。
    2. UVA 兜底：如果配置了 `cpu_offload_gb > 0`，选择 uva (统一虚拟寻址) 后端。
    3. 都不配：什么都不做，返回 NoopOffloader。
    """
    
    # 局部导入，避免循环引用 (Circular import)，因为这些卸载器内部可能也依赖了一些基础模块
    from vllm.model_executor.offloader.prefetch import PrefetchOffloader
    from vllm.model_executor.offloader.uva import UVAOffloader

    # ==========================================
    # 1. 解析配置对象
    # ==========================================
    backend = offload_config.offload_backend  # 获取用户指定的后端类型 ("auto", "uva", "prefetch" 等)
    uva = offload_config.uva                  # 获取 UVA 相关的配置子项
    prefetch = offload_config.prefetch        # 获取 Prefetch (预取) 相关的配置子项

    # ==========================================
    # 2. "auto" 模式下的自动推断 (优先级仲裁)
    # ==========================================
    if backend == "auto":
        # 优先级 1：Prefetch 预取机制
        # 如果用户设置了预取组大小，说明用户想用激进的流水线预取策略
        if prefetch.offload_group_size > 0:
            backend = "prefetch"
            
        # 优先级 2：UVA 静态映射机制
        # 如果没开预取，但给了 CPU 卸载内存配额，就走我们之前分析过的 UVA 路径
        elif uva.cpu_offload_gb > 0:
            backend = "uva"
            
        # 优先级 3：不卸载
        # 如果都没配置，说明显存够用，返回一个“空转”的卸载器 (No operation)
        else:
            return NoopOffloader()

    # ==========================================
    # 3. 实例化具体的卸载器 (Factory 模式生成)
    # ==========================================
    
    # 分支 A：创建预取卸载器 (复杂流水线机制)
    if backend == "prefetch":
        return PrefetchOffloader(
            group_size=prefetch.offload_group_size,         # 每次预取包含几层网络
            num_in_group=prefetch.offload_num_in_group,     # 组内参数量控制
            prefetch_step=prefetch.offload_prefetch_step,   # 提前多少步触发预取
            offload_params=prefetch.offload_params,         # 指定需要卸载的参数类型
            mode="cpu",                                     # 目前预取的目标设备是 CPU
        )
        
    # 分支 B：创建 UVA 卸载器 (静态 Pinned Memory 机制)
    elif backend == "uva":
        return UVAOffloader(
            # 将 GB 转换为底层分配器需要的 Bytes (字节)。
            # 注意：1024**3 是严谨的 Gibibyte (GiB) 到 Byte 的换算
            cpu_offload_max_bytes=int(uva.cpu_offload_gb * 1024**3), 
            cpu_offload_params=uva.cpu_offload_params,      # 指定允许被 UVA 卸载的参数正则匹配规则
        )
        
    # 分支 C：兜底方案 (比如用户传了一个不支持的 backend 字符串)
    else:
        return NoopOffloader()
