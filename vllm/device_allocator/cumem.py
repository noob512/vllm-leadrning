# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# cumem-based pytorch pluggable allocator to implement sleep mode.
# other approaches tried but failed:
# - cuda-python package binding
# - custom libcuda driver ctypes wrapper
# both of them failed because of cuda context mismatch.
# not sure why, they are created from a different context.
# the only successful approach is to call cuda driver API in C.
import dataclasses
import gc
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.system_utils import find_loaded_library

logger = init_logger(__name__)


cumem_available = False
libcudart: Any = None
try:
    from vllm.cumem_allocator import (
        init_module,
        python_create_and_map,
        python_unmap_and_release,
    )
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    lib_name = find_loaded_library("cumem_allocator")
    libcudart = CudaRTLibrary()
    cumem_available = True
except ModuleNotFoundError:
    # only cuda and rocm platforms support cumem allocator
    init_module = None
    python_create_and_map = None
    python_unmap_and_release = None
    lib_name = None

# py_device, py_alignedSize, py_d_mem, py_p_memHandle
HandleType = tuple[int, int, int, int]


@dataclasses.dataclass
class AllocationData:
    handle: HandleType
    tag: str
    cpu_backup_tensor: torch.Tensor | None = None


def create_and_map(allocation_handle: HandleType) -> None:
    python_create_and_map(*allocation_handle)


def unmap_and_release(allocation_handle: HandleType) -> None:
    python_unmap_and_release(*allocation_handle)


def get_pluggable_allocator(
    python_malloc_fn: Callable[[HandleType], None],
    python_free_func: Callable[[int], HandleType],
) -> torch.cuda.memory.CUDAPluggableAllocator:
    init_module(python_malloc_fn, python_free_func)
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        lib_name, "my_malloc", "my_free"
    )
    return new_alloc




@contextmanager
def use_memory_pool_with_allocator(
    # 参数 1: Python 端的内存分配回调函数。
    # 当 PyTorch 底层需要显存时，会调用这个函数。
    python_malloc_fn: Callable[[HandleType], None], 
    
    # 参数 2: Python 端的内存释放回调函数。
    # 当 PyTorch 底层释放显存时，会把物理地址 (int) 传给这个函数。
    python_free_func: Callable[[int], HandleType],
) -> Iterator[
    # 返回值注解：这是一个生成器，产出一个包含【内存池对象】和【分配器对象】的元组。
    tuple[torch.cuda.memory.MemPool, torch.cuda.memory.CUDAPluggableAllocator]
]:
    """
    创建一个上下文环境。在这个环境内，PyTorch 将使用我们自定义的回调函数来管理 GPU 显存。
    """
    
    # ==========================================
    # 1. 组装“可插拔分配器” (Pluggable Allocator)
    # ==========================================
    # get_pluggable_allocator 通常是一个绑定了 C++ 扩展的方法。
    # 它将 Python 的分配/释放函数打包，生成一个 PyTorch 底层能够识别的“自定义分配器 (new_alloc)”。
    # 这相当于给 PyTorch 换了一个按我们规则办事的新“显存管家”。
    new_alloc = get_pluggable_allocator(python_malloc_fn, python_free_func)
    
    # ==========================================
    # 2. 建立专属内存池 (MemPool)
    # ==========================================
    # 使用刚刚创建的分配器 (new_alloc._allocator，即底层的 C++ 分配器指针)
    # 来初始化一个全新的 PyTorch 内存池 (MemPool)。
    # 以后放进这个池子里的张量，都会通过我们的 python_malloc_fn 和 python_free_func 来管理。
    mem_pool = torch.cuda.memory.MemPool(new_alloc._allocator)
    
    # ==========================================
    # 3. 激活并移交控制权
    # ==========================================
    # 调用 PyTorch 原生的上下文管理器 `torch.cuda.memory.use_mem_pool`，
    # 强制将当前线程接下来的所有显存分配操作，都路由到刚刚建好的 mem_pool 中。
    with torch.cuda.memory.use_mem_pool(mem_pool):
        
        # 将创建好的 内存池 (mem_pool) 和 分配器 (new_alloc) 打包产出，交还给外层调用者。
        # 注意：这里为什么要把它们 yield 出去？
        # 正如在 CuMemAllocator 代码中看到的，外层需要把这两个对象保存到字典中 (强引用)，
        # 防止它们在 with 块执行期间被 Python 垃圾回收机制 (GC) 意外销毁，从而引发 C++ 底层崩溃。
        yield mem_pool, new_alloc

    # 当退出 with 块时，PyTorch 的 use_mem_pool 会自动恢复默认的显存分配机制。


class CuMemAllocator:
    """
    一个用于管理 CUDA 张量 (tensors) 内存池的单例类 (Singleton)。
    当分配器 (allocator) 进入休眠状态 (sleep) 时，该内存池中的内存可以被卸载 (offload) 或直接丢弃 (discard)。

    【核心机制】
    1. 上下文管理 `use_memory_pool(tag)`: 
       在特定的上下文环境内创建的所有张量，都会被分配到这个自定义的内存池中，
       并且会被打上与上下文中传入的 `tag` (标签) 相同的标记。以方便后续按组别进行管理。
       
    2. 休眠操作 `sleep`: 
       调用此方法时，所有带有指定标签的张量会被卸载 (offload) 到 CPU 内存中（作为备份），
       而其余的张量将被直接丢弃。这样可以极大地节省 GPU 显存。
       
    3. 唤醒操作 `wake_up`: 
       调用此方法时，之前被卸载到 CPU 内存中的张量会被重新加载回 GPU 显存中，
       而那些之前被丢弃的张量将只保留一个空壳（无实际物理显存占用）。

    【为什么设计为单例模式 (Singleton)？】
    这是由于 PyTorch 底层机制和 C 扩展的限制决定的。
    当已分配的张量触发 Python 的垃圾回收 (Garbage Collection) 时，PyTorch 会调用一个释放回调函数 (free callback)，
    进而触发 Python 层面的 `python_free_callback` 方法。
    底层的 C 扩展代码中使用了一个**全局变量**来存储该回调函数的指针。如果我们创建了多个 `CuMemAllocator` 实例，
    这个全局变量就会被不断覆盖，导致释放回调机制混乱，最终引发内存泄漏或程序崩溃。
    因此，全局只能存在一个实例来统管回调机制。
    """

    # 存储全局唯一的类实例。初始化为 None。
    instance: "CuMemAllocator | None" = None
    
    # 默认分配标签。如果在使用上下文管理器时没有传入特定标签，将使用此默认值。
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "CuMemAllocator":
        """
        获取 CuMemAllocator 单例实例的方法。
        
        提示：
        由于本类是单例模式，外部调用者不应该直接调用构造函数 `CuMemAllocator()` 实例化对象，
        而是应该全局统一调用 `CuMemAllocator.get_instance()`。
        
        Returns:
            CuMemAllocator: 返回全局唯一的内存分配器实例。
            
        Raises:
            AssertionError: 如果底层的 cumem 模块/环境不可用，会在此处阻断并报错。
        """
        # 前置检查：确保底层的 C 扩展内存管理器已经加载且可用
        assert cumem_available, "cumem allocator is not available"
        
        # 懒汉式 (Lazy initialization) 单例实现：
        # 如果当前 instance 还是空的，说明是第一次调用，此时再真正创建对象
        if CuMemAllocator.instance is None:
            CuMemAllocator.instance = CuMemAllocator()
            
        return CuMemAllocator.instance

    def __init__(self):
        """
        初始化 CuMemAllocator 实例。
        这个方法在且仅在 get_instance() 第一次被调用时执行。
        """
        
        # ==========================================
        # 1. 环境变量与兼容性检查 (Compatibility Check)
        # ==========================================
        # 获取 PyTorch CUDA 内存分配器的环境变量配置
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        # 检查是否开启了 expandable_segments (可扩展段) 特性。
        # 为什么报错？ PyTorch 的 expandable_segments 特性用于减少显存碎片，
        # 但它目前与自定义的底层内存池机制 (cumem) 存在冲突。如果强行一起使用会导致程序崩溃或行为异常。
        # 这里是一个保护性防御，如果用户开了这个特性，直接报错阻断并给出 GitHub issue 链接。
        assert "expandable_segments:True" not in conf, (
            "Expandable segments are not compatible with memory pool. "
            "Please track https://github.com/pytorch/pytorch/issues/147851 "
            "for the latest updates."
        )

        # ==========================================
        # 2. 核心状态与数据结构初始化 (State Initialization)
        # ==========================================
        # 字典：用于记录 内存指针(int) 到 分配元数据(AllocationData) 的映射关系。
        # 当 PyTorch 申请显存时，我们返回一个内存地址（指针）。我们需要记住这个地址归属哪个标签、具体信息是什么，
        # 这样在触发 free (释放) 时，我们才知道应该处理哪块内存。
        self.pointer_to_data: dict[int, AllocationData] = {}
        
        # 字符串：记录当前正在使用的分配标签 (tag)。
        # 当进入 use_memory_pool("my_tag") 上下文时，这个值会被修改为 "my_tag"。
        self.current_tag: str = CuMemAllocator.default_tag
        
        # 字典：用于存储不同标签对应的真实底层内存分配器和内存池对象。
        # 键是 tag 字符串，值是底层的 C++ 分配器/池实例。实现按 tag 隔离显存。
        self.allocator_and_pools: dict[str, Any] = {}
        
        # ==========================================
        # 3. 强引用保护机制 (防止垃圾回收导致的 C++ 崩溃)
        # ==========================================
        # 这是一个非常经典的 Python 绑定 C++ 的避坑操作。
        # `self._python_malloc_callback` 在 Python 中实际上是在每次调用时动态生成的一个“绑定方法 (bound-method)”对象。
        # 如果我们直接把这个动态生成的对象传给底层的 C 扩展代码，Python 的垃圾回收器 (GC) 会认为 Python 层已经没有任何变量指向它了，从而将它回收销毁。
        # 随后，当 C 代码尝试回调这个已经被销毁的函数时，就会引发灾难性的段错误 (Segmentation fault/Coredump)。
        # 
        # 解决方案：在这里将它们赋值给实例属性 (self.python_x_callback)，
        # 创建一个“强引用 (strong reference)”。只要当前的 CuMemAllocator 实例存活，
        # 这两个回调函数的 Python 对象就不会被销毁，从而保证 C++ 底层能安全地调用它们。
        self.python_malloc_callback = self._python_malloc_callback
        self.python_free_callback = self._python_free_callback

    def _python_malloc_callback(self, allocation_handle: HandleType) -> None:
        """
        内部的回调方法。
        当内存池中发生新的内存分配时，底层的 C++ 代码会调用此方法，
        用于在 Python 端同步存储这块内存的元数据（记录这块内存归谁管、有多大等）。
        
        :param allocation_handle: 这是一个来自底层 C++ 的元组 (Tuple)。
            通常包含分配的详细信息。根据代码上下文推断：
            - allocation_handle[1]: 分配的内存字节数 (size)
            - allocation_handle[2]: 分配的内存物理地址指针 (address / pointer)
        """
        
        # 1. 提取内存物理地址
        # 从底层的句柄元组中提取出第 3 个元素（索引为 2），这代表张量在 GPU 上的真实物理起始地址。
        py_d_mem = allocation_handle[2]
        
        # 2. 登记造册 (核心逻辑)
        # 将这块新分配的内存地址记录到管家的全局字典 (pointer_to_data) 中。
        # AllocationData 记录了两个关键信息：
        #   - allocation_handle: 底层完整的操作句柄（之后释放内存时需要用到）。
        #   - self.current_tag: 这块内存属于哪个标签（比如 "default" 或 "lora_weights"）。
        # 这样一来，只要拿着内存地址 (py_d_mem)，我们随时能查出它属于哪个车间 (tag)。
        self.pointer_to_data[py_d_mem] = AllocationData(
            allocation_handle, self.current_tag
        )
        
        # 3. 打印调试日志
        # 记录一条 Debug 级别的日志，方便开发者追踪显存的分配情况。
        # 日志内容包含了：分配的字节数 (allocation_handle[1])、所属标签 (self.current_tag) 以及物理地址 (py_d_mem)。
        logger.debug(
            "Allocated %s bytes for %s with address %s from cumem allocator",
            allocation_handle[1],
            self.current_tag,
            py_d_mem,
        )
        
        return

    def _python_free_callback(self, ptr: int) -> HandleType:
        """
        内部的回调方法。
        当内存池中的某块内存被释放时，底层的 C++ 代码会调用此方法。
        它的主要作用是根据内存地址查账、清理相关的 CPU 备份，并把底层的操作句柄还给 C++。
        
        :param ptr: 底层 C++ 传来的显存物理起始地址 (一个整数)。
        :return HandleType: 返回当初分配时底层生成的完整句柄 (凭证)，让底层去执行真正的物理释放。
        """
        
        # 1. 查账并注销 (核心逻辑)
        # 拿着底层传来的物理地址 (ptr)，去管家的全局账本 (pointer_to_data) 里查找。
        # .pop(ptr) 会把这条记录提取出来的同时，从字典中删除它（相当于在账本上划掉）。
        # 提取出来的 data 是一个 AllocationData 对象，包含了当初记下的 tag 和 handle。
        data = self.pointer_to_data.pop(ptr)
        
        # 2. 清理 CPU 备份 (释放系统内存)
        # 这是该内存管家“卸载 (Offload)”功能的核心体现！
        # 如果之前这个张量为了省显存，被休眠 (sleep) 备份到了普通的 CPU 内存中，
        # 那么现在既然整个张量都要被彻底销毁了，存在 CPU 里的备份也得一并清理。
        # 将其设置为 None 后，Python 的垃圾回收器 (GC) 就会自动把那块 CPU 内存回收掉。
        if data.cpu_backup_tensor is not None:
            data.cpu_backup_tensor = None
            
        # 3. 打印调试日志
        # 记录释放行为：释放了多少字节 (data.handle[1])，属于哪个标签 (data.tag)，地址是什么 (ptr)。
        logger.debug(
            "Freed %s bytes for %s with address %s from cumem allocator",
            data.handle[1],
            data.tag,
            ptr,
        )
        
        # 4. 归还底层凭证
        # 底层 C++ 只给了我们一个地址，但 C++ 自己要真正解除物理显存映射时，需要完整的凭证 (handle)。
        # 所以我们把查账查出来的完整 handle 返回给底层，让它去完成最终的物理级清理。
        return data.handle

    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
        """
        Put the allocator in sleep mode.
        All data in the memory allocation with the specified tag will be
        offloaded to CPU memory, and others will be discarded.

        :param offload_tags: The tags of the memory allocation that will be
            offloaded. The rest of the memory allocation will be discarded.
        """
        if offload_tags is None:
            # by default, allocated tensors are offloaded
            # when the allocator sleeps
            offload_tags = (CuMemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)

        assert isinstance(offload_tags, tuple)

        total_bytes = 0
        backup_bytes = 0

        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            total_bytes += handle[1]
            if data.tag in offload_tags:
                backup_bytes += handle[1]
                size_in_bytes = handle[1]
                cpu_backup_tensor = torch.empty(
                    size_in_bytes,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=is_pin_memory_available(),
                )
                cpu_ptr = cpu_backup_tensor.data_ptr()
                libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)
                data.cpu_backup_tensor = cpu_backup_tensor
            unmap_and_release(handle)

        logger.info(
            "CuMemAllocator: sleep freed %.2f GiB memory in total, of which "
            "%.2f GiB is backed up in CPU and the rest %.2f GiB is discarded "
            "directly.",
            total_bytes / 1024**3,
            backup_bytes / 1024**3,
            (total_bytes - backup_bytes) / 1024**3,
        )

        gc.collect()
        torch.cuda.empty_cache()

    def wake_up(self, tags: list[str] | None = None) -> None:
        """
        Wake up the allocator from sleep mode.
        All data that is previously offloaded will be loaded back to GPU
        memory, and the rest of the data will have empty memory.

        :param tags: The tags of the memory allocation that will be loaded
            back to GPU memory. If None, all memory allocation will be loaded
            back to GPU memory.
        """
        for ptr, data in self.pointer_to_data.items():
            if tags is None or data.tag in tags:
                handle = data.handle
                create_and_map(handle)
                if data.cpu_backup_tensor is not None:
                    cpu_backup_tensor = data.cpu_backup_tensor
                    if cpu_backup_tensor is not None:
                        size_in_bytes = (
                            cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
                        )
                        cpu_ptr = cpu_backup_tensor.data_ptr()
                        libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)
                        data.cpu_backup_tensor = None

    @contextmanager
    def use_memory_pool(self, tag: str | None = None):
        """
        一个用于启用自定义内存池的上下文管理器 (Context Manager)。
        在这个 `with` 代码块内部创建的所有显存分配，都会被自动放入这个指定的内存池中，
        并且被打上你传入的 `tag`（标签）。

        :param tag: 内存分配的标签。如果传入 None，则默认使用 `CuMemAllocator.default_tag` (即 "default")。
        """
        
        # ==========================================
        # 1. 准备阶段：处理标签与状态切换
        # ==========================================
        if tag is None:
            tag = CuMemAllocator.default_tag

        assert isinstance(tag, str), "标签必须是字符串类型"

        # 保存进入上下文之前的旧标签。
        # 为什么要保存？因为上下文是可以嵌套的，退出当前上下文时，必须恢复到之前的状态。
        old_tag = self.current_tag
        
        # 将当前类的活跃标签切换为用户指定的 tag
        self.current_tag = tag

        # ==========================================
        # 2. 核心阶段：挂载底层 C++ 内存池
        # ==========================================
        # 调用底层的 C++ 上下文管理器 `use_memory_pool_with_allocator`。
        # 这里把之前做过“强引用”保护的两个回调函数传给 C++。
        with use_memory_pool_with_allocator(
            self.python_malloc_callback, self.python_free_callback
        ) as data:
            
            # 【避坑 PyTorch Bug #146431】
            # 在 PyTorch 2.6 中存在一个垃圾回收 (GC) 相关的 Bug。
            # 如果不把底层返回的分配器/内存池对象 (data) 强行保存在字典里，
            # 它可能会在上下文执行期间被 Python 意外回收，导致程序崩溃。
            # 因此，这里强行给它续命，存放到 self.allocator_and_pools 字典中。
            self.allocator_and_pools[tag] = data
            
            # yield 是上下文管理器的分水岭。
            # 程序运行到这里，会暂停，并去执行用户写在 `with allocator.use_memory_pool("my_tag"):` 里面的业务代码。
            # 等用户的代码执行完毕，再继续执行 yield 后面的清理逻辑。
            yield

            # ==========================================
            # 3. 退出阶段：手动显存清理 (Workaround)
            # ==========================================
            # 【避坑 PyTorch Bug #145168】
            # 当使用自定义（可插拔）分配器时，直接调用原生的 `torch.cuda.empty_cache()` 会报错。
            # 这会导致一个严重问题：如果代码中（比如在线量化场景）临时分配了一大块显存，用完就扔了，
            # 这些显存虽然被标记为“未使用”，但物理上没有还给系统，造成显存泄漏。
            
            # 解决方案：开发者写了一段手动清空缓存的代码。
            # 第一步：获取当前底层内存池的内存快照 (snapshot)，看看所有内存块的状态。
            allocations = data[0].snapshot()
            
            # 第二步：遍历所有内存块，找出那些逻辑上已释放（allocated_size == 0）但物理上还没释放的块。
            for allocation in allocations:
                if allocation["allocated_size"] == 0:
                    # 获取底层的操作句柄 (handle)
                    handle = self._python_free_callback(allocation["address"])
                    # 手动将该内存块解除映射，并彻底释放回 GPU 系统物理显存中
                    unmap_and_release(handle)
        
        # ==========================================
        # 4. 恢复状态
        # ==========================================
        # 退出上下文，将管家的当前活跃标签恢复为之前保存的旧标签
        self.current_tag = old_tag

    def get_current_usage(self) -> int:
        """
        Get the total number of bytes allocated in the memory pool.
        """
        sum_bytes: int = 0
        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            sum_bytes += handle[1]
        return sum_bytes
