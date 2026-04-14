# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define KV connector functionality mixin for model runners.
"""

import copy
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)
from vllm.v1.worker.utils import AttentionGroup

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


# Defined as a kv connector functionality mixin for ModelRunner (GPU, TPU)
class KVConnectorModelRunnerMixin:
    @staticmethod
    def kv_connector_no_forward(
        scheduler_output: "SchedulerOutput", vllm_config: VllmConfig
    ) -> ModelRunnerOutput:
        # KV send/recv even if no work to do.
        with (
            set_forward_context(None, vllm_config),
            KVConnectorModelRunnerMixin._get_kv_connector_output(
                scheduler_output, wait_for_save=False
            ) as kv_connector_output,
        ):
            pass

        if kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    @staticmethod
    def maybe_get_kv_connector_output(
        scheduler_output: "SchedulerOutput",
        defer_finalize: bool = False,
    ) -> AbstractContextManager[KVConnectorOutput | None]:
        return (
            KVConnectorModelRunnerMixin._get_kv_connector_output(
                scheduler_output, defer_finalize=defer_finalize
            )
            if has_kv_transfer_group()
            else nullcontext()
        )

    @staticmethod
    def finalize_kv_connector() -> None:
        """Finalize the KV connector: wait_for_save and clear metadata.

        Call after draft model forward when defer_finalize=True was used.
        """
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            kv_connector.wait_for_save()
            kv_connector.clear_connector_metadata()

    # This context manager must be used within an active forward context.
    # It encapsulates the entire KV connector lifecycle within execute_model
    @staticmethod
    @contextmanager
    def _get_kv_connector_output(
        scheduler_output: "SchedulerOutput",
        wait_for_save: bool = True,
        defer_finalize: bool = False,
    ) -> Generator[KVConnectorOutput, None, None]:
        output = KVConnectorOutput()

        # Update KVConnector with the KVConnector metadata forward().
        kv_connector = get_kv_transfer_group()
        assert isinstance(kv_connector, KVConnectorBase)
        assert scheduler_output.kv_connector_metadata is not None
        kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

        # Background KV cache transfers happen here.
        # These transfers are designed to be async and the requests
        # involved may be disjoint from the running requests.
        # Do this here to save a collective_rpc.
        kv_connector.start_load_kv(get_forward_context())
        try:
            yield output
        finally:
            if wait_for_save and not defer_finalize:
                kv_connector.wait_for_save()

            output.finished_sending, output.finished_recving = (
                kv_connector.get_finished(scheduler_output.finished_req_ids)
            )
            output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()

            output.kv_connector_stats = kv_connector.get_kv_connector_stats()
            output.kv_cache_events = kv_connector.get_kv_connector_kv_cache_events()
            output.kv_connector_worker_meta = kv_connector.build_connector_worker_meta()

            if not defer_finalize:
                kv_connector.clear_connector_metadata()

    @staticmethod
    def use_uniform_kv_cache(
        attn_groups: list[list[AttentionGroup]],
        cache_dtype: CacheDType,
    ) -> bool:
        """
        Determines whether a uniform KV layout should be used.
        A uniform layout means all layers KV caches will share the same
        underlying tensor, where for a given block number, the respective
        KV data for all layers will be contiguous.
        This will allow efficient KV transfer of per-block KV data for all
        layers at once.
        Note this layout will only be applied given 3 conditions:
        1. The KV Cache config contains just a single group where all layers
            have the same page size.
        2. A KV connector is configured, and the KV connector instance prefers
            to use this layout (prefer_cross_layer_blocks() returns True)
        2. The flash attention backend supports this layout
            (get_kv_cache_stride_order(True) includes a placement for a
            num_layers dimension)

        Note that the actual placement of the num_layers dimensions
        in the unified layers tensors will be determined by the attention
        backend.
        Thus, the layers KV data may still not be contiguous per block
        if the attention backend does not support it.

        Args:
            attn_groups: The list of attention groups for this model
            cache_dtype: The KV cache dtype
        Returns:
            True if we should use a uniform KV cache layout.
        """

        if not has_kv_transfer_group():
            return False
        if not get_kv_transfer_group().prefer_cross_layer_blocks:
            return False

        if len(attn_groups) != 1 or len(attn_groups[0]) != 1:
            return False

        attn_group = attn_groups[0][0]
        kv_cache_spec = attn_group.kv_cache_spec
        if not isinstance(kv_cache_spec, AttentionSpec):
            return False

        attn_backend = attn_group.backend
        kv_cache_shape = attn_backend.get_kv_cache_shape(
            1234,
            kv_cache_spec.block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
            cache_dtype_str=cache_dtype,
        )

        try:
            kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                include_num_layers_dimension=True
            )
        except (AttributeError, NotImplementedError):
            return False

        # check that attention backend includes a layers dimension
        if len(kv_cache_stride_order) != len(kv_cache_shape) + 1:
            return False

        # stride_order[0] == 0 means num_layers stays first in physical
        # layout (identity permutation), so cross-layer is unsupported.
        return kv_cache_stride_order[0] != 0

    @staticmethod
    def allocate_uniform_kv_caches(
        kv_cache_config: KVCacheConfig,
        attn_groups: list[list[AttentionGroup]],
        cache_dtype: CacheDType,
        device: torch.device,
        kernel_block_sizes: list[int],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, type[AttentionBackend]]:
        """
        【VIP 快速通道】：专为所有层布局相同的模型（如 Llama, Qwen）设计。
        核心动作：跳过一层层申请显存的繁琐步骤，直接向 GPU 申请一块物理连续的“通天巨无霸”张量。
        """
        # 【步骤 1：获取模板与防呆校验】
        # 既然大家长得都一样，直接抓取第 0 组的第 0 个算子作为全网的规格模板。
        attn_group = attn_groups[0][0]
        kv_cache_spec = attn_group.kv_cache_spec
        assert isinstance(kv_cache_spec, AttentionSpec)

        # 再次严格校验：是不是所有层要求的显存大小真的完全一样？
        tensor_sizes = set(
            kv_cache_tensor.size for kv_cache_tensor in kv_cache_config.kv_cache_tensors
        )
        assert len(tensor_sizes) == 1  # 如果不是唯一的，直接报错阻断
        tensor_size = tensor_sizes.pop() # 获取单层需要的字节数

        # 【步骤 2：计算全网总显存需求】
        page_size = kv_cache_spec.page_size_bytes
        assert tensor_size % page_size == 0
        num_blocks = tensor_size // page_size # 单层包含的逻辑 Block 数量
        num_layers = len(kv_cache_config.kv_cache_tensors) # 模型总层数（如 32 层）
        
        # 💡 算出总建筑面积 = 单层面积 * 总层数
        total_size = tensor_size * num_layers 

        # 【步骤 3：逻辑 Block 向物理 Kernel Block 转换】
        assert len(kernel_block_sizes) == 1
        kernel_block_size = kernel_block_sizes[0]
        # 把调度器的大 Block 切成底层算子需要的小 Block（如 256 切成 4 个 64）
        num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size
        kernel_num_blocks = num_blocks * num_blocks_per_kv_block

        # 【步骤 4：向底层算子询问“最佳张量形状”】
        attn_backend = attn_group.backend
        # 不同的 CUDA 算子（FlashAttention/XFormers）对显存排布有不同的偏好。
        # 这里向底层索要它最喜欢的 4D 形状，比如：[num_blocks, num_heads, head_size, block_size]
        kv_cache_shape = attn_backend.get_kv_cache_shape(
            kernel_num_blocks,
            kernel_block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
            cache_dtype_str=cache_dtype,
        )

        # 把“层数（num_layers）”作为最高维度加进去，变成 5D 形状。
        # 此时形状类似：[32层, 1000块, 32头, 128维度, 64块大小]
        kv_cache_shape = (num_layers,) + kv_cache_shape

        # 【步骤 5：核心黑科技！内存交织优化 (Stride Order)】
        try:
            # 询问底层算子：你希望内存的物理先后顺序怎么排？
            kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                include_num_layers_dimension=True
            )
            assert len(kv_cache_stride_order) == len(kv_cache_shape)
        except (AttributeError, NotImplementedError):
            # 如果算子没要求，就按标准顺序 (0, 1, 2, 3, 4)
            kv_cache_stride_order = tuple(range(len(kv_cache_shape)))

        # 根据算子要求的物理顺序，把原本的形状“洗牌（打乱维度顺序）”
        kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)

        logger.info("Allocating a cross layer KV cache of shape %s", kv_cache_shape)

        # 【步骤 6：真正向显卡要内存（暴力开辟一维连续显存）】
        # allocate one contiguous buffer for all layers
        # 极其粗暴且优雅：先要一块纯粹的 1 维连续字节数组 (torch.int8)，
        # 然后强行转换数据类型 (view dtype)，最后强行折叠成洗牌后的多维形状 (view shape)。
        cross_layers_kv_cache = (
            torch.zeros(total_size, dtype=torch.int8, device=device)
            .view(kv_cache_spec.dtype)
            .view(kv_cache_shape)
        )

        # 【步骤 7：恢复人类/上层的逻辑视角】
        # Maintain original KV shape view.
        # 虽然物理上内存已经被洗牌交织了，但上层的 Python 代码如果不按顺序读写会疯掉的。
        # 所以算出逆向序列 (inv_order)，用 permute() 生成一个“逻辑上正常”的视图。
        inv_order = [
            kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
        ]
        # permute 不会移动任何物理显存！它只是修改了张量内部的步长（Stride）元数据。
        permuted_kv_cache = cross_layers_kv_cache.permute(*inv_order)

        # 【步骤 8：切蛋糕与交房】
        kv_caches = {}
        for i, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):
            # 沿着第 0 维（num_layers）切片。
            # 比如 tensor = permuted_kv_cache[0] 就是第 0 层专属的几十 GB 显存指针。
            tensor = permuted_kv_cache[i]
            
            # 把切好的片，挂载到具体的层名字上（如 layer_0）
            for layer_name in kv_cache_tensor.shared_by:
                kv_caches[layer_name] = tensor

        return kv_caches, cross_layers_kv_cache, attn_backend
