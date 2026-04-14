# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UVA-based CPU offloading using Unified Virtual Addressing."""

from collections.abc import Generator

import torch
import torch.nn as nn
from torch.func import functional_call

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader
from vllm.utils.mem_utils import format_gib
from vllm.utils.platform_utils import is_pin_memory_available, is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


class UVAOffloader(BaseOffloader):
    """
    使用统一虚拟寻址 (UVA, Unified Virtual Addressing) 实现零拷贝 (zero-copy) 访问的卸载器。

    核心原理：此卸载器会将模型参数移动到“固定/锁页 CPU 内存” (pinned CPU memory) 中，
    并利用 UVA 技术为这些内存创建 CUDA 视图。这使得 GPU 能够通过 PCIe 总线直接寻址和读取 
    CPU 内存中的数据，而无需先执行显式的 Host-to-Device 内存拷贝。
    
    性能权衡：虽然节省了显存并免去了显式的传输开销，但访问速度受限于 PCIe 的物理带宽，
    通常显著慢于直接访问 GPU 本地显存 (HBM)。

    降级机制：如果用户通过环境变量禁用了 UVA 功能，该卸载器将回退到基于 `functional_call` 
    的方法，在需要计算时才按需将参数传输到 GPU。

    参数 (Args):
        cpu_offload_max_bytes (int): 允许卸载到 CPU 内存的参数最大字节数（预算上限）。
        cpu_offload_params (set[str] | None): 可选择性卸载的参数名称片段集合（白名单）。
            如果为空，则表示在不超过 `cpu_offload_max_bytes` 预算的前提下，所有参数都可以被卸载。
    """

    def __init__(
        self,
        cpu_offload_max_bytes: int,
        cpu_offload_params: set[str] | None = None,
    ):
        # 设定允许占用的最大 CPU 内存字节数，作为资源上限
        self.cpu_offload_max_bytes = cpu_offload_max_bytes
        
        # 初始化当前已卸载的字节数为 0，后续在实际分配和移动参数时会累加此值进行额度检查
        self.cpu_offload_bytes = 0
        
        # 存储允许卸载的参数名称集合。如果传入 None，则初始化为空集合 (set())
        self.cpu_offload_params = cpu_offload_params or set()

        # 决定是否启用锁页内存 (Pinned Memory)。
        # 锁页内存在物理内存中是连续且不可被操作系统 swap 到磁盘上的，它是实现高带宽 PCIe 传输和 UVA 的前置条件。
        # 启用条件：当前硬件/PyTorch环境支持锁页内存，且用户未通过环境变量强制禁用。
        self.pin_memory = (
            is_pin_memory_available()
            and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY
        )
        
        # 决定是否实际启用 UVA (统一虚拟寻址) 卸载功能。
        # 启用条件：当前系统/CUDA环境支持 UVA，且用户未通过环境变量强制禁用。
        # 如果最终为 False，系统在执行计算时将不得不退回到传统的显式内存拷贝模式。
        self.uva_offloading = (
            is_uva_available() and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA
        )

    def wrap_modules(
        self,
        # 参数 modules_generator: 这是一个生成器 (Generator)。
        # 注意：它里面装的不是造好的模型层，而是“制造模型层的图纸和流水线”。
        # 只有在下面的 for 循环向它要数据时，它才会去真刀真枪地实例化一个 nn.Module。
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """
        使用 UVA (统一虚拟寻址) 卸载技术对模块（模型层）进行包装/拦截。
        """
        
        # --- 核心拦截与卸载逻辑 ---
        # 这是一个列表推导式，但由于其数据源是生成器，实际上发生的是一个完美的流水线作业：
        # 1. for module in modules_generator: 触发生成器造出【一个】真实的模型层（比如第8层）。
        # 2. self._maybe_offload_to_cpu(module): 立刻把这个新鲜出炉的层送进卸载判定逻辑。
        #    在这个内部函数中，系统会根据前面设定的预算 (cpu_offload_max_bytes) 决定要不要掏空这个层的参数。
        #    如果掏空了，参数会被塞进锁页内存，原位置会留下一个 UVA 的指针。
        # 3. 处理完的层被存入 modules 这个真正的列表中。
        modules = [self._maybe_offload_to_cpu(module) for module in modules_generator]
        
        # --- 成果汇报 ---
        # self.cpu_offload_bytes 是我们在 __init__ 里初始化的那个计数器。
        # 在经历了上面的循环后，_maybe_offload_to_cpu 函数会在内部不断累加被成功卸载到 CPU 的字节数。
        # 如果最终累加值大于 0，说明我们确实验证并执行了卸载动作。
        if self.cpu_offload_bytes > 0:
            # 打印一条日志，告诉用户一共卸载了多少显存（通常转换为 GiB 格式展示）
            logger.info(
                "Total CPU offloaded parameters: %s",
                format_gib(self.cpu_offload_bytes),
            )
            
        # 返回最终组装好的列表（这些层里，该被卸载的张量已经被替换为指向 CPU 的指针了）
        return modules

    def _maybe_offload_to_cpu(self, module: nn.Module) -> nn.Module:
        """
        核心执行逻辑：如果预算允许，利用 UVA 将模型模块的参数卸载到 CPU 内存。
        """
        
        # --- 1. 快速拦截与提前放行 (Early Exits) ---
        # 尝试获取当前层的第一块参数。如果没有参数（比如这是一个纯逻辑层如 Dropout），直接放行。
        if (params := next(module.parameters(), None)) is None:
            return module

        # 获取参数当前所在的设备（通常此时在显存 GPU 上，或在 Meta 虚拟设备上）
        device = params.device

        # 如果参数已经在 CPU 上了，说明不需要处理，直接放行
        if device == torch.device("cpu"):
            return module

        # 【查预算】：如果之前卸载的总字节数已经达到了你设定的预算上限，停止卸载，直接放行
        if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
            return module

        # --- 2. 逐个参数执行卸载 ---
        # 标记当前层是否真的有参数被卸载了
        offloaded_parameters = False
        
        # 遍历这个层里的每一个参数（比如一个 Linear 层会有 weight 和 bias）
        for name, p in module.named_parameters():
            
            # 【精确查账】：采用“参数级”卸载策略。
            # 如果在卸载当前层一半参数的时候，预算突然花光了，就立刻打断 (break)。
            # 这意味着同一个模型层里，可能发生“权重(weight)被卸载到了 CPU，但偏置(bias)留在了 GPU”的情况。
            if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
                break

            # 【白名单检查】：如果用户指定了只卸载某些特定的参数（比如只卸载专家层的权重）
            if self.cpu_offload_params:
                # 检查当前参数名是否在白名单中。
                # 前后加点 `.` 是一种非常聪明的字符串匹配技巧，确保只匹配完整的层级名。
                # 例如 "experts.w2_weight" 能匹配上 "mlp.experts.w2_weight"，
                # 但不会错误地匹配到 "mlp.experts.w2_weight_scale"（因为加点后变成了 ".w2_weight_scale."）
                should_offload = any(
                    f".{param}." in f".{name}." for param in self.cpu_offload_params
                )
                if not should_offload:
                    continue # 不在白名单里，跳过这个参数

            # --- 3. 物理大搬家 (Data Transfer) ---
            # 真正把数据转移到 CPU（这里通常是一个异步的或阻塞的拷贝动作）
            cpu_data = p.data.to(device="cpu")
            
            # 如果启用了锁页内存（为了加速 PCIe 传输和支持 UVA）
            if self.pin_memory:
                cpu_data = cpu_data.pin_memory()

            # --- 4. 偷天换日：替换原始参数 ---
            if not self.uva_offloading:
                # 如果不支持 UVA（降级模式），就只能老老实实把参数变成一个纯 CPU Tensor
                p.data = cpu_data
            else:
                # 【终极魔法 UVA】：如果支持 UVA！
                # 这一步不会把数据拷回 GPU，而是调用底层 C++ 接口，根据刚才的 CPU 锁页内存，
                # 生成一个“披着 GPU Tensor 外衣”的视图。
                # GPU 访问它时，硬件会自动通过 PCIe 跨总线去主板内存里拿数据。
                p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                # 打上烙印，告诉系统这个参数虽然看起来在 GPU 上，但实际上在 CPU 里
                p._vllm_is_uva_offloaded = True

            # 【记账】：把当前参数占用的真实字节数（元素个数 * 单个元素字节数）累加进总账单
            self.cpu_offload_bytes += p.data.numel() * p.data.element_size()
            offloaded_parameters = True

        # --- 5. 降级模式的补救措施 (Fallback Hook) ---
        # 如果有参数被卸载了，但系统不支持 UVA，该怎么办？
        # 模型在做前向传播 (forward) 时，如果发现参数在 CPU，输入数据在 GPU，直接就崩溃报错了。
        if offloaded_parameters and not self.uva_offloading:
            # 备份原始的 forward 函数
            original_forward = module.forward

            # 现场写一个新的、拦截式的 forward 函数
            def forward(*args, **kwargs):
                # 临时把 forward 换回原版，防止递归死循环
                module.forward = original_forward
                
                # 动态抢救：把属于这个层的所有的参数，按需（临时）拷贝回 GPU (device)
                # non_blocking=True 尽可能让拷贝和计算异步重叠，减少卡顿
                device_state = {
                    k: v.to(device, non_blocking=True)
                    for k, v in module.state_dict().items()
                }

                # 使用 PyTorch 的 functional_call 神技！
                # 这个函数允许你“用一套外来的参数（刚刚拷到GPU上的 device_state），去运行当前这个层的逻辑”
                output = functional_call(
                    module,
                    device_state,
                    args=args,
                    kwargs=kwargs,
                    tie_weights=False,
                )
                
                # 算完之后，把拦截器装回去，等待下一次 forward
                module.forward = forward
                return output

            # 狸猫换太子，把当前层正常的 forward 换成我们写的这个带“动态拷贝”功能的拦截器
            module.forward = forward

        # 改造完成，返回这个“被掏空”或“被劫持”的模型层
        return module
