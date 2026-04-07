# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import os
import threading
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing import Process, connection
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import msgspec
import zmq

from vllm import envs
from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.ray.ray_env import get_env_vars_to_copy
from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.utils.system_utils import get_mp_context
from vllm.v1.engine.coordinator import DPCoordinator
from vllm.v1.executor import Executor
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

STARTUP_POLL_PERIOD_MS = 10000


class CoreEngineState(Enum):
    NEW = auto()
    CONNECTED = auto()
    READY = auto()


class CoreEngine:
    """One per data parallel rank, used to track state during handshaking."""

    def __init__(self, index: int = 0, local: bool = True):
        self.local = local
        self.identity = index.to_bytes(2, "little")

        self.state = CoreEngineState.NEW


@dataclass
class EngineZmqAddresses:
    # ZMQ input socket addresses for each front-end client (requests)
    inputs: list[str]
    # ZMQ output socket addresses for each front-end client (responses)
    outputs: list[str]
    # ZMQ input socket address of DP coordinator if applicable
    coordinator_input: str | None = None
    # ZMQ output socket address of DP coordinator if applicable
    coordinator_output: str | None = None
    # ZMQ socket for front-end to connect to DP coordinator.
    # Not used by engine, just relayed to front-end in handshake response.
    # Only required for external DP LB case.
    frontend_stats_publish_address: str | None = None


@dataclass
class EngineHandshakeMetadata:
    """Metadata sent to each engine process during startup handshake,
    including addresses of the front-end ZMQ queues that they should
    connect to.
    """

    addresses: EngineZmqAddresses
    parallel_config: dict[str, int | str | list[int]]


class CoreEngineProcManager:
    """
    工具类：负责 AsyncLLM 和 LLMEngine 所使用的后台进程的
    创建 (Creation)、就绪检查 (Readiness) 和 停机 (Shutdown)。
    """

    def __init__(
        self,
        local_engine_count: int,       # 本机要启动的引擎总数（通常等于本机 GPU 数）
        start_index: int,             # 全局起始 Rank 索引
        local_start_index: int,       # 本机起始 Rank 索引
        vllm_config: "VllmConfig",    # 全局配置对象
        local_client: bool,           # 是否为本地客户端模式
        handshake_address: str,       # 握手地址（用于子进程向主进程报到）
        executor_class: type["Executor"], # 执行器类（如分布式执行器）
        log_stats: bool,              # 是否记录统计信息
        client_handshake_address: str | None = None, # 客户端侧的握手地址（选填）
        tensor_queue: "Queue" | None = None,         # 多模态张量共享队列（IPC）
    ):
        # 1. 获取多进程上下文（在 vLLM 中通常显式指定为 'spawn' 模式）
        context = get_mp_context()
        
        # 2. 准备所有子进程共享的通用参数（相当于给所有厨师发的通用食谱）
        common_kwargs = {
            "vllm_config": vllm_config,
            "local_client": local_client,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
            "tensor_queue": tensor_queue,
        }

        if client_handshake_address:
            common_kwargs["client_handshake_address"] = client_handshake_address

        # 判断是否处于数据并行 (DP) 模式
        is_dp = vllm_config.parallel_config.data_parallel_size > 1

        # 延迟导入，防止循环依赖
        from vllm.v1.engine.core import EngineCoreProc

        self.processes: list[BaseProcess] = [] # 存放创建好的进程对象
        local_dp_ranks = []
        
        # ---------------------------------------------------------
        # 3. 循环创建进程对象（配置阶段，尚未真正启动）
        # ---------------------------------------------------------
        for index in range(local_engine_count):
            local_index = local_start_index + index
            global_index = start_index + index

            local_dp_ranks.append(local_index)
            
            # 使用 context.Process 创建进程
            self.processes.append(
                context.Process(
                    # 进程的入口函数：EngineCoreProc.run_engine_core
                    target=EngineCoreProc.run_engine_core,
                    # 进程名：如 EngineCore_DP0
                    name=f"EngineCore_DP{global_index}" if is_dp else "EngineCore",
                    # 将通用参数与当前进程特有的 Rank 索引合并
                    kwargs=common_kwargs 
                    | {"dp_rank": global_index, "local_dp_rank": local_index},
                )
            )

        # 4. 注册“终结器” (weakref.finalize)
        # 这是一个保险，确保当管理器对象被销毁时，会自动调用 shutdown 函数关闭所有子进程
        self._finalizer = weakref.finalize(self, shutdown, self.processes)
        self.manager_stopped = threading.Event()
        self.failed_proc_name: str | None = None

        # ---------------------------------------------------------
        # 5. 正式启动进程 (Execution Stage)
        # ---------------------------------------------------------
        try:
            for proc, local_dp_rank in zip(self.processes, local_dp_ranks):
                # 针对不同平台和配置处理设备可见性（如 CUDA_VISIBLE_DEVICES）
                # 如果是非 CUDA 平台或者使用了 Ray，需要手动设置环境变量来隔离 GPU
                if is_dp and (
                    not current_platform.is_cuda_alike()
                    or vllm_config.parallel_config.use_ray
                ):
                    # set_device_control_env_var 确保进程只能看到它该用的那块显卡
                    with set_device_control_env_var(vllm_config, local_dp_rank):
                        proc.start() # 【真正的 OS 进程诞生点】
                else:
                    # 对于标准 CUDA 环境，隔离通常在子进程内部通过 torch.accelerator 处理
                    proc.start()
        finally:
            # 6. 异常回滚
            # 如果在启动过程中发现某些进程已经意外结束（启动失败），则关掉所有已启动的进程。
            if self.finished_procs():
                self.shutdown()

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown engine core processes with configurable timeout."""
        self.manager_stopped.set()
        if self._finalizer.detach() is not None:
            shutdown(self.processes, timeout=timeout)

    def monitor_engine_liveness(self) -> None:
        """Monitor engine core process liveness."""

        sentinel_to_proc = {proc.sentinel: proc for proc in self.processes}
        sentinels = set(sentinel_to_proc.keys())

        while sentinels and not self.manager_stopped.is_set():
            died_sentinels = connection.wait(sentinels, timeout=1)

            for sentinel in died_sentinels:
                proc = sentinel_to_proc.pop(cast(int, sentinel))
                exitcode = proc.exitcode
                if exitcode != 0 and not self.manager_stopped.is_set():
                    self.failed_proc_name = proc.name
            if died_sentinels:
                # Any engine exit currently triggers a shutdown. Future
                # work (e.g., Elastic and fault-tolerant EP) will add finer-grained
                # handling for different exit scenarios.
                break

        self.shutdown()

    def sentinels(self) -> list:
        return [proc.sentinel for proc in self.processes]

    def finished_procs(self) -> dict[str, int]:
        """Returns dict of proc name -> exit code for any finished procs."""
        return {
            proc.name: proc.exitcode
            for proc in self.processes
            if proc.exitcode is not None
        }


class SignalCallback:
    """Safely trigger a callback from signal handler context via a dedicated thread."""

    def __init__(self, callback: Callable[[], None]):
        self._callback = callback
        self._event = threading.Event()
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="signal-callback",
        )
        self._thread.start()

    def _run(self):
        self._event.wait()
        if not self._stopped:
            self._callback()

    def trigger(self):
        self._event.set()

    def stop(self):
        self._stopped = True
        self._event.set()


@contextlib.contextmanager
def set_device_control_env_var(
    vllm_config: VllmConfig, local_dp_rank: int
) -> Iterator[None]:
    """
    Temporarily set CUDA_VISIBLE_DEVICES or equivalent
    for engine subprocess.
    """
    world_size = vllm_config.parallel_config.world_size
    local_world_size = vllm_config.parallel_config.local_world_size
    evar = current_platform.device_control_env_var

    value = get_device_indices(evar, local_dp_rank, world_size, local_world_size)
    with patch.dict(os.environ, values=((evar, value),)):
        yield


def get_device_indices(
    device_control_env_var: str,
    local_dp_rank: int,
    world_size: int,
    local_world_size: int | None = None,
):
    """
    Returns a comma-separated string of device indices for the specified
    data parallel rank.

    For example, if world_size=2 and local_dp_rank=1, and there are 4 devices,
    this will select devices 2 and 3 for local_dp_rank=1.
    """
    if local_world_size is None:
        local_world_size = world_size
    try:
        value = ",".join(
            str(current_platform.device_id_to_physical_device_id(i))
            for i in range(
                local_dp_rank * world_size,
                local_dp_rank * world_size + local_world_size,
            )
        )
    except IndexError as e:
        raise Exception(
            f"Error setting {device_control_env_var}: "
            f"local range: [{local_dp_rank * world_size}, "
            f"{(local_dp_rank + 1) * world_size}) "
            "base value: "
            f'"{os.getenv(device_control_env_var)}"'
        ) from e
    return value


class CoreEngineActorManager:
    """
    Utility class to handle creation, readiness, and shutdown
    of core engine Ray actors used by the AsyncLLM and LLMEngine.

    Different from CoreEngineProcManager, this class manages
    core engines for both local and remote nodes.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        placement_groups: list["PlacementGroup"] | None = None,
        local_dp_ranks: list[int] | None = None,
    ):
        import copy

        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        from vllm.v1.engine.core import DPMoEEngineCoreActor, EngineCoreActor

        dp_size = vllm_config.parallel_config.data_parallel_size
        actor_class = (
            DPMoEEngineCoreActor
            if dp_size > 1 and vllm_config.model_config.is_moe
            else EngineCoreActor
        )

        self.local_engine_actors: list[ray.ActorHandle] = []
        self.remote_engine_actors: list[ray.ActorHandle] = []

        env_vars_list = get_env_vars_to_copy(destination=actor_class.__name__)
        self.env_vars_dict = {
            name: os.environ[name] for name in env_vars_list if name in os.environ
        }
        runtime_env = RuntimeEnv(env_vars=self.env_vars_dict)

        self.addresses = addresses
        self.executor_class = executor_class
        self.log_stats = log_stats
        local_engine_count = vllm_config.parallel_config.data_parallel_size_local
        world_size = vllm_config.parallel_config.world_size
        self.manager_stopped = threading.Event()
        self.failed_proc_name: str | None = None

        if ray.is_initialized():
            logger.info("Ray is already initialized. Skipping Ray initialization.")
        else:
            ray.init()

        parallel_config = vllm_config.parallel_config
        if parallel_config.enable_elastic_ep:
            from vllm.distributed.utils import create_tcp_store

            ip = parallel_config.data_parallel_master_ip
            store = create_tcp_store(
                ip,
                0,
                is_master=True,
                world_size=-1,
                wait_for_workers=False,
            )
            parallel_config._coord_store_port = store.port
            self._coord_store = store

        if placement_groups is not None:
            assert local_dp_ranks is not None, (
                "local_dp_ranks must be provided if placement_groups is provided"
            )
            assert len(placement_groups) == len(local_dp_ranks), (
                "placement_groups and local_dp_ranks must have the same length"
            )
            logger.info("Using provided placement groups")
            # TODO(rui): validate passed-in placement groups
            self.created_placement_groups = []
        else:
            placement_groups, local_dp_ranks = (
                CoreEngineActorManager.create_dp_placement_groups(vllm_config)
            )
            self.created_placement_groups = placement_groups
        assert len(placement_groups) == dp_size, (
            "Number of placement groups must match data parallel size"
        )

        self.placement_group_is_local = []
        refs = []
        for index, local_index, pg in zip(
            range(dp_size), local_dp_ranks, placement_groups
        ):
            dp_vllm_config = copy.deepcopy(vllm_config)
            dp_vllm_config.parallel_config.placement_group = pg
            local_client = index < local_engine_count

            if dp_size > 1 and dp_vllm_config.kv_transfer_config is not None:
                # modify the engine_id and append the local_dp_rank to it to ensure
                # that the kv_transfer_config is unique for each DP rank.
                dp_vllm_config.kv_transfer_config.engine_id = (
                    f"{dp_vllm_config.kv_transfer_config.engine_id}_dp{local_index}"
                )

            # Ray XPU known issue: dpctl initializes the GPU runtime early, so
            # setting device env vars in Ray actor's initialization method
            # will not affect device selection. See:
            # https://github.com/ray-project/ray/blob/master/python/ray/_private/accelerators/intel_gpu.py#L56 # noqa: E501
            if current_platform.is_xpu():
                device_evar = current_platform.device_control_env_var
                device_indices = get_device_indices(
                    device_evar, local_index, world_size
                )
                actor_env_vars = self.env_vars_dict.copy()
                actor_env_vars[device_evar] = device_indices
                runtime_env = RuntimeEnv(env_vars=actor_env_vars)

            actor = (
                ray.remote(actor_class)
                .options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=world_size,
                    ),
                    runtime_env=runtime_env,
                )
                .remote(
                    vllm_config=dp_vllm_config,
                    executor_class=executor_class,
                    log_stats=log_stats,
                    local_client=local_client,
                    addresses=addresses,
                    dp_rank=index,
                    local_dp_rank=local_index,
                )
            )
            if local_client:
                self.local_engine_actors.append(actor)
            else:
                self.remote_engine_actors.append(actor)
            self.placement_group_is_local.append(local_client)
            refs.append(actor.wait_for_init.remote())

        ray.get(refs)
        self.run_refs = []
        self.actor_run_ref_dict = dict()
        for actor in self.local_engine_actors + self.remote_engine_actors:
            ref = actor.run.remote()
            self.run_refs.append(ref)
            self.actor_run_ref_dict[actor] = ref

    @staticmethod
    def create_dp_placement_groups(
        vllm_config: VllmConfig,
    ) -> tuple[list["PlacementGroup"], list[int]]:
        """
        Create placement groups for data parallel.
        """

        import ray
        from ray._private.state import available_resources_per_node

        logger.info("Creating placement groups for data parallel")
        dp_master_ip = vllm_config.parallel_config.data_parallel_master_ip
        dp_size = vllm_config.parallel_config.data_parallel_size
        dp_size_local = vllm_config.parallel_config.data_parallel_size_local

        available_resources = available_resources_per_node()
        world_size = vllm_config.parallel_config.world_size
        placement_groups: list[PlacementGroup] = []
        local_dp_ranks: list[int] = []

        dp_master_ip_key = f"node:{dp_master_ip}"
        nodes = sorted(
            available_resources.values(), key=lambda x: dp_master_ip_key not in x
        )
        assert len(nodes) > 0, "No nodes with resources found in Ray cluster."
        assert dp_master_ip_key in nodes[0], (
            f"The DP master node (ip: {dp_master_ip}) is missing or dead"
        )
        device_str = current_platform.ray_device_key
        n_node_devices: list[int] = [
            int(node_resources[device_str])
            for node_resources in nodes
            if device_str in node_resources
        ]
        assert n_node_devices, f"No {device_str} found in Ray cluster."
        max_device_per_node = max(n_node_devices)

        pack_strategy = envs.VLLM_RAY_DP_PACK_STRATEGY
        _supported_pack_strategies = ("strict", "fill", "span")
        if pack_strategy not in _supported_pack_strategies:
            raise ValueError(
                f"{envs.VLLM_RAY_DP_PACK_STRATEGY} is not supported. "
                "Make sure to set `VLLM_RAY_DP_PACK_STRATEGY` "
                f"to one of {_supported_pack_strategies}"
            )

        all2all_backend = vllm_config.parallel_config.all2all_backend
        if pack_strategy == "fill" and (
            all2all_backend == "deepep_high_throughput"
            or all2all_backend == "deepep_low_latency"
        ):
            raise ValueError(
                "DeepEP kernels require EP ranks [0,7] (same for [8,15], ...) "
                "to be on the same node, but VLLM_RAY_DP_PACK_STRATEGY=fill "
                "does not guarantee that. "
                "Please use VLLM_RAY_DP_PACK_STRATEGY=strict instead."
            )

        if pack_strategy in ("strict", "fill"):
            placement_strategy = "STRICT_PACK"
        else:
            placement_strategy = "PACK"
            assert world_size > max_device_per_node, (
                f"World size {world_size} is smaller than the "
                "maximum number of devices per node "
                f"{max_device_per_node}. Make sure to set "
                "`VLLM_RAY_DP_PACK_STRATEGY` to `strict` or `fill`"
            )

            # if we need multiple nodes per dp group, we require for now that
            # available nodes are homogeneous
            assert set(n_node_devices) == {max_device_per_node}, (
                f"Nodes are not homogeneous, {nodes}"
            )
            assert world_size % max_device_per_node == 0, (
                f"For multi-node data parallel groups, world_size ({world_size}) must "
                f"be a multiple of number of devices per node ({max_device_per_node})."
            )
            assert len(n_node_devices) * max_device_per_node >= world_size * dp_size, (
                f"Not enough total available nodes ({len(n_node_devices)}) "
                f"and devices per node ({max_device_per_node}) "
                f"to satisfy required world size {world_size} and data parallel size "
                f"{dp_size}"
            )
            assert dp_size_local == 1, (
                f"data-parallel-size-local {dp_size_local} should be set as the "
                "default (1) for VLLM_RAY_DP_PACK_STRATEGY=span. "
                "The actual data-parallel-size-local will be auto determined."
            )

        # bundles collected for a single DP rank from multiple nodes,
        # for "span" pack strategy
        collected_bundles = []
        for node_resources in nodes:
            node_ip_keys = [
                key
                for key in node_resources
                if key != "node:__internal_head__" and key.startswith("node:")
            ]
            assert len(node_ip_keys) == 1, (
                f"Zero or multiple node IP keys found in node resources: {node_ip_keys}"
            )
            node_ip_key = node_ip_keys[0]
            node_ip = node_ip_key.split(":")[1]

            n_device_on_node = int(node_resources.get(device_str, 0))
            if pack_strategy == "span" and n_device_on_node != 0:
                # Strictly speaking,
                # dp_size_available = n_device_on_node / world_size
                # and is a fraction, but we use 1 for easier processing
                dp_size_available = 1
            else:
                dp_size_available = n_device_on_node // world_size

            if node_ip == dp_master_ip:
                if dp_size_available < dp_size_local:
                    raise ValueError(
                        f"Not enough resources to allocate {dp_size_local} DP ranks "
                        f"on DP master node {dp_master_ip}, possible to fit "
                        f"{dp_size_available} DP ranks."
                    )
                dp_size_to_allocate = dp_size_local
            elif pack_strategy == "strict":
                if dp_size_available < dp_size_local:
                    logger.info(
                        "Skipping node %s as %s DP ranks could not fit, "
                        "possible to fit %s DP ranks",
                        node_ip,
                        dp_size_local,
                        dp_size_available,
                    )
                    continue
                dp_size_to_allocate = dp_size_local
            else:
                # for "pack_strategy" in "fill" and "span"
                # we always take everything that's available
                dp_size_to_allocate = dp_size_available

            for i in range(dp_size_to_allocate):
                device_bundle = [{device_str: 1.0, "node:" + node_ip: 0.001}]
                if pack_strategy == "span":
                    collected_bundles += device_bundle * n_device_on_node
                    assert len(collected_bundles) <= world_size, (
                        "collected_bundles should be <= world_size, "
                        f"but got {len(collected_bundles)=} and {world_size=}"
                    )

                    # we only create a placement group if we collected enough devices
                    if len(collected_bundles) < world_size:
                        continue

                    bundles = collected_bundles + [{"CPU": 1.0}]
                    collected_bundles = []
                else:
                    bundles = device_bundle * world_size + [{"CPU": 1.0}]

                pg = ray.util.placement_group(
                    name=f"dp_rank_{len(placement_groups)}",
                    strategy=placement_strategy,
                    bundles=bundles,
                )
                placement_groups.append(pg)
                local_dp_ranks.append(i)
                if len(placement_groups) == dp_size:
                    break

        if len(placement_groups) < dp_size:
            raise ValueError(
                f"Not enough resources to allocate {dp_size} "
                "placement groups, only created "
                f"{len(placement_groups)} placement groups. "
                "Available resources: "
                f"{available_resources}"
            )
        assert len(placement_groups) == dp_size, (
            f"Created {len(placement_groups)} DP placement groups, expected {dp_size}"
        )
        assert len(local_dp_ranks) == dp_size, (
            f"local_dp_ranks length {len(local_dp_ranks)} does not match "
            f"expected {dp_size}"
        )
        return placement_groups, local_dp_ranks

    @staticmethod
    def add_dp_placement_groups(
        old_vllm_config: VllmConfig, new_data_parallel_size: int
    ) -> tuple[list["PlacementGroup"], list[int]]:
        """
        Add placement groups for new data parallel size.
        """
        import ray
        from ray._private.state import (
            available_resources_per_node,
            total_resources_per_node,
        )
        from ray.util.state import list_nodes

        old_dp_size = old_vllm_config.parallel_config.data_parallel_size
        num_pg_to_create = new_data_parallel_size - old_dp_size

        if num_pg_to_create <= 0:
            return [], []

        dp_master_ip = old_vllm_config.parallel_config.data_parallel_master_ip
        world_size = old_vllm_config.parallel_config.world_size

        nodes = list_nodes()
        nodes = sorted(nodes, key=lambda node: node.node_ip != dp_master_ip)
        assert nodes[0].node_ip == dp_master_ip, "The first node must be the head node"
        assert len(nodes) == 1 or nodes[1].node_ip != dp_master_ip, (
            "There can only be one head node"
        )

        available_resources = available_resources_per_node()
        total_resources = total_resources_per_node()

        placement_groups = []
        local_dp_ranks = []
        num_pg_created = 0

        device_str = current_platform.ray_device_key
        for node in nodes:
            if num_pg_created >= num_pg_to_create:
                break

            node_ip = node.node_ip
            node_id = node.node_id
            if device_str not in available_resources[node_id]:
                continue
            available_gpus = int(available_resources[node_id][device_str])

            # Get total GPUs on this node from the node's resources
            # Ray stores node resources with node ID as key
            total_gpus = int(total_resources[node_id][device_str])

            # Calculate used GPUs and used engines on this node
            used_gpus = max(0, total_gpus - available_gpus)
            used_engines_on_node = used_gpus // world_size

            # Calculate how many new engines this node can accommodate
            available_engine_count = available_gpus // world_size

            # Create placement groups for new engines on this node
            for i in range(available_engine_count):
                if num_pg_created >= num_pg_to_create:
                    break

                rank = old_dp_size + num_pg_created

                # Create bundles with node constraint for master node
                if node_ip == dp_master_ip:
                    bundles = [
                        {device_str: 1.0, "node:" + dp_master_ip: 0.001}
                    ] * world_size + [{"CPU": 1.0}]
                else:
                    bundles = [{device_str: 1.0}] * world_size + [{"CPU": 1.0}]

                pg = ray.util.placement_group(
                    name=f"dp_rank_{rank}",
                    strategy="STRICT_PACK",
                    bundles=bundles,
                )
                placement_groups.append(pg)

                # Local rank starts from the number of engines already used
                # on this node
                local_rank = used_engines_on_node + i
                local_dp_ranks.append(local_rank)
                num_pg_created += 1

        return placement_groups, local_dp_ranks

    def scale_up_elastic_ep(
        self, cur_vllm_config: VllmConfig, new_data_parallel_size: int
    ) -> None:
        import copy

        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        from vllm.v1.engine.core import DPMoEEngineCoreActor, EngineCoreActor

        actor_class = (
            DPMoEEngineCoreActor
            if cur_vllm_config.model_config.is_moe
            else EngineCoreActor
        )

        cur_data_parallel_size = len(self.local_engine_actors) + len(
            self.remote_engine_actors
        )

        assert new_data_parallel_size > cur_data_parallel_size, (
            f"New data parallel size {new_data_parallel_size} must be greater "
            f"than current data parallel size {cur_data_parallel_size} "
            "for scale up"
        )

        placement_groups, local_dp_ranks = self.add_dp_placement_groups(
            cur_vllm_config, new_data_parallel_size
        )

        world_size = cur_vllm_config.parallel_config.world_size
        dp_master_ip = cur_vllm_config.parallel_config.data_parallel_master_ip
        new_local_engines = 0

        runtime_env = RuntimeEnv(
            env_vars=self.env_vars_dict | {"VLLM_ELASTIC_EP_SCALE_UP_LAUNCH": "1"}
        )
        for i, (pg, local_rank) in enumerate(zip(placement_groups, local_dp_ranks)):
            rank = cur_data_parallel_size + i
            dp_vllm_config = copy.deepcopy(cur_vllm_config)
            dp_vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
            dp_vllm_config.parallel_config.placement_group = pg

            # Check if this placement group is on the head node
            local_client = any(
                bundle.get("node:" + dp_master_ip, 0) > 0 for bundle in pg.bundle_specs
            )

            if local_client:
                new_local_engines += 1
                # Update data_parallel_size_local
                dp_vllm_config.parallel_config.data_parallel_size_local = (
                    cur_vllm_config.parallel_config.data_parallel_size_local
                    + new_local_engines
                )

            actor = (
                ray.remote(actor_class)
                .options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=world_size,
                    ),
                    runtime_env=runtime_env,
                )
                .remote(
                    vllm_config=dp_vllm_config,
                    executor_class=self.executor_class,
                    log_stats=self.log_stats,
                    local_client=local_client,
                    addresses=self.addresses,
                    dp_rank=rank,
                    local_dp_rank=local_rank,
                )
            )

            if local_client:
                self.local_engine_actors.append(actor)
            else:
                self.remote_engine_actors.append(actor)
            self.created_placement_groups.append(pg)
            self.placement_group_is_local.append(local_client)

        ray.get(
            [
                actor.wait_for_init.remote()
                for actor in (
                    self.local_engine_actors[-new_local_engines:]
                    if new_local_engines > 0
                    else []
                )
                + self.remote_engine_actors[
                    -(len(placement_groups) - new_local_engines) :
                ]
            ]
        )

        actors = (
            self.local_engine_actors[-new_local_engines:]
            if new_local_engines > 0
            else []
        ) + self.remote_engine_actors[-(len(placement_groups) - new_local_engines) :]

        for actor in actors:
            ref = actor.run.remote()
            self.run_refs.append(ref)
            self.actor_run_ref_dict[actor] = ref

        cur_vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        # Update old_vllm_config with new data_parallel_size_local if any new
        # local engines were added
        if new_local_engines > 0:
            cur_vllm_config.parallel_config.data_parallel_size_local += (
                new_local_engines
            )

    def scale_down_elastic_ep(
        self, cur_data_parallel_size: int, new_data_parallel_size: int
    ) -> None:
        import ray

        assert cur_data_parallel_size > new_data_parallel_size, (
            f"cur_data_parallel_size {cur_data_parallel_size} must be greater "
            f"than new_data_parallel_size {new_data_parallel_size} "
            "for scale down"
        )
        for _ in range(cur_data_parallel_size - new_data_parallel_size):
            pg = self.created_placement_groups.pop()
            is_local = self.placement_group_is_local.pop()
            if is_local:
                self.local_engine_actors.pop()
            else:
                self.remote_engine_actors.pop()
            ray.util.remove_placement_group(pg)

    def remove_run_refs_for_scale_down(self, removed_dp_size: int) -> None:
        if removed_dp_size <= 0:
            return
        flags = self.placement_group_is_local[-removed_dp_size:]
        li = len(self.local_engine_actors) - 1
        ri = len(self.remote_engine_actors) - 1
        for is_local in reversed(flags):
            if is_local:
                actor = self.local_engine_actors[li]
                li -= 1
            else:
                actor = self.remote_engine_actors[ri]
                ri -= 1
            ref = self.actor_run_ref_dict.pop(actor)
            self.run_refs.remove(ref)

    def get_run_refs(self):
        return self.run_refs

    def monitor_engine_liveness(self) -> None:
        import ray

        while not self.manager_stopped.is_set():
            actor_run_refs = list(self.get_run_refs())
            if not actor_run_refs:
                logger.info(
                    "There are no actors to monitor currently. "
                    "The monitoring function is about to terminate."
                )
                break
            actor_done_refs, _ = ray.wait(actor_run_refs, timeout=5)
            unexpected_failure = False
            for actor_ref in actor_done_refs:
                if self.manager_stopped.is_set():
                    break
                if actor_ref not in self.get_run_refs():
                    # The run refs may have been updated by elastic scale-down.
                    continue
                try:
                    ray.get(actor_ref)
                except ray.exceptions.RayActorError:
                    self.failed_proc_name = f"Actor {actor_ref}"
                    unexpected_failure = True

            if unexpected_failure:
                break

        self.shutdown()

    def shutdown(self, timeout: float | None = None) -> None:
        import ray

        self.manager_stopped.set()
        for actor in self.local_engine_actors + self.remote_engine_actors:
            ray.kill(actor)
        for pg in self.created_placement_groups:
            ray.util.remove_placement_group(pg)


def get_engine_zmq_addresses(
    vllm_config: VllmConfig,
    num_api_servers: int = 1,
) -> EngineZmqAddresses:
    """Allocate ZMQ addresses for engine-client communication."""
    parallel_config = vllm_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local
    local_start_index = parallel_config.data_parallel_rank_local
    dp_size = parallel_config.data_parallel_size
    host = parallel_config.data_parallel_master_ip
    local_engines_only = parallel_config.local_engines_only

    # In offline mode there is an LLM instance per DP rank and
    # one core engine per LLM, see
    # examples/offline_inference/data_parallel.py.
    offline_mode = local_start_index is not None

    # client_local_only = True for cases where this front-end
    # sends requests only to colocated engines.
    client_local_only = (
        offline_mode or local_engines_only or (local_engine_count == dp_size)
    )
    # NOTE(yongji): handling scaling from intra-node to inter-node
    if parallel_config.enable_elastic_ep:
        client_local_only = False

    return EngineZmqAddresses(
        inputs=[
            get_engine_client_zmq_addr(client_local_only, host)
            for _ in range(num_api_servers)
        ],
        outputs=[
            get_engine_client_zmq_addr(client_local_only, host)
            for _ in range(num_api_servers)
        ],
    )


@contextlib.contextmanager
def launch_core_engines(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    addresses: EngineZmqAddresses,
    num_api_servers: int = 1,
) -> Iterator[
    tuple[
        CoreEngineProcManager | CoreEngineActorManager | None,
        DPCoordinator | None,
        EngineZmqAddresses,
        Queue | None,
    ]
]:
    """
    启动推理引擎进程和数据并行（DP）协调器进程。
    作为一个上下文管理器，它负责这些进程的生命周期管理（启动与清理）。
    """

    # ---------------------------------------------------------
    # 1. 提取并行配置参数
    # ---------------------------------------------------------
    parallel_config = vllm_config.parallel_config
    dp_size = parallel_config.data_parallel_size            # 总的数据并行组数
    local_engine_count = parallel_config.data_parallel_size_local # 本机需要启动的引擎数
    local_start_index = parallel_config.data_parallel_rank_local # 本机引擎的起始 Rank
    dp_rank = parallel_config.data_parallel_rank            # 当前前端进程的全局 Rank
    host = parallel_config.data_parallel_master_ip          # 主节点 IP
    local_engines_only = parallel_config.local_engines_only # 是否仅管理本机引擎

    # 如果起始索引不为空，通常意味着是离线推理模式（Offline Mode）
    offline_mode = local_start_index is not None

    # ---------------------------------------------------------
    # 2. 多模态张量共享通道 (Tensor IPC)
    # ---------------------------------------------------------
    # 对于多模态（如图像），如果配置了 torch_shm，通过 multiprocessing.Queue 
    # 在 API Server 和 Engine 之间建立“零拷贝”绿色通道，直接传内存地址。
    tensor_queue: Queue | None = None
    multimodal_config = vllm_config.model_config.multimodal_config
    if multimodal_config is not None and multimodal_config.mm_tensor_ipc == "torch_shm":
        # 注意：目前该数据流仅支持 DP=1（单副本）
        tensor_queue = get_mp_context().Queue()

    # ---------------------------------------------------------
    # 3. 启动 DP 协调器 (The Coordinator)
    # ---------------------------------------------------------
    # 协调器就像是“交通警察”，负责：
    #   - 收集各引擎队列状态并发布，实现负载均衡（LB）
    #   - 如果是 MoE 模型，还要协调不同波次的专家计算同步
    run_coordinator = (
        vllm_config.needs_dp_coordinator and not offline_mode and dp_rank == 0
    )

    if run_coordinator:
        coordinator = DPCoordinator(
            parallel_config,
            enable_wave_coordination=vllm_config.model_config.is_moe,
        )
        # 获取协调器对应的 ZMQ 地址并同步到全局地址对象中
        addresses.coordinator_input, addresses.coordinator_output = (
            coordinator.get_engine_socket_addresses()
        )
        addresses.frontend_stats_publish_address = (
            coordinator.get_stats_publish_address()
        )
        logger.info("已启动 DP 协调器进程 (PID: %d)", coordinator.proc.pid)
    else:
        coordinator = None

    # ---------------------------------------------------------
    # 4. Ray 后端特殊处理
    # ---------------------------------------------------------
    # 如果配置使用 Ray 来管理分布式进程，则走 ActorManager 路线
    if parallel_config.data_parallel_backend == "ray":
        logger.info("正在启动基于 Ray 的数据并行后端")
        engine_actor_manager = CoreEngineActorManager(
            vllm_config=vllm_config,
            addresses=addresses,
            executor_class=executor_class,
            log_stats=log_stats,
        )
        yield engine_actor_manager, coordinator, addresses, tensor_queue
        return

    # ---------------------------------------------------------
    # 5. 确定“握手”名单 (Handshake Strategy)
    # ---------------------------------------------------------
    # 我们启动了子进程，主进程必须等子进程“报到”。这里决定要等哪些 Rank 报到。
    if offline_mode:
        engines_to_handshake = [CoreEngine(index=dp_rank, local=True)]
    elif dp_rank == 0:
        # Rank 0 作为主控，通常需要确认所有副本（本地+远程）的状态
        engines_to_handshake = [
            CoreEngine(index=i, local=(i < local_engine_count)) for i in range(dp_size)
        ]
    else:
        # 非 0 节点只关注它负责启动的本地引擎
        engines_to_handshake = [
            CoreEngine(index=i, local=True)
            for i in range(dp_rank, dp_rank + local_engine_count)
        ]

    # 决定握手地址是否仅限本地（IPC）还是需要网络通信（TCP）
    handshake_local_only = offline_mode or local_engine_count == dp_size
    if parallel_config.enable_elastic_ep: # 弹性扩展模式下必须支持远程连接
        handshake_local_only = False

    handshake_address = get_engine_client_zmq_addr(
        handshake_local_only, host, parallel_config.data_parallel_rpc_port
    )

    # ---------------------------------------------------------
    # 6. 正式开工：启动 Engine 进程
    # ---------------------------------------------------------
    # 创建一个临时 ZMQ ROUTER 套接字，专门用来“点名”确认子进程启动情况
    with zmq_socket_ctx(
        handshake_address if not (local_engines_only and dp_rank > 0) else get_open_zmq_ipc_path(), 
        zmq.ROUTER, bind=True
    ) as handshake_socket:
        
        if local_engine_count:
            # 【分身术】：启动本地的 EngineCore 进程管理
            local_engine_manager = CoreEngineProcManager(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                handshake_address=handshake_address,
                # ... 传入各种复杂的启动参数 ...
                tensor_queue=tensor_queue,
            )
        else:
            local_engine_manager = None

        # ---------------------------------------------------------
        # 7. 交还控制权给调用者
        # ---------------------------------------------------------
        # 此时进程已经 start() 了，但模型权重可能还在加载中
        yield local_engine_manager, coordinator, addresses, tensor_queue

        # ---------------------------------------------------------
        # 8. 最后的守望：确认启动完毕 (Handshake Wait)
        # ---------------------------------------------------------
        # 当外部代码完成初始化后（退出 with 块前），
        # 阻塞在这里，直到所有子进程发回“我好了”的消息。
        # 这里会发生：加载权重、分配显存、PagedAttention 初始化等。
        wait_for_engine_startup(
            handshake_socket,
            addresses,
            engines_to_handshake,
            parallel_config,
            dp_size > 1 and vllm_config.model_config.is_moe,
            vllm_config.cache_config,
            local_engine_manager,
            coordinator.proc if coordinator else None,
        )


def wait_for_engine_startup(
    handshake_socket: zmq.Socket,
    addresses: EngineZmqAddresses,
    core_engines: list[CoreEngine],
    parallel_config: ParallelConfig,
    coordinated_dp: bool,
    cache_config: CacheConfig,
    proc_manager: CoreEngineProcManager | None,
    coord_process: Process | None,
):
    # Wait for engine core process(es) to send ready messages.
    local_count = parallel_config.data_parallel_size_local
    remote_count = len(core_engines) - local_count
    # [local, remote] counts
    conn_pending, start_pending = [local_count, remote_count], [0, 0]
    poller = zmq.Poller()
    poller.register(handshake_socket, zmq.POLLIN)

    remote_should_be_headless = (
        not parallel_config.data_parallel_hybrid_lb
        and not parallel_config.data_parallel_external_lb
    )

    if proc_manager is not None:
        for sentinel in proc_manager.sentinels():
            poller.register(sentinel, zmq.POLLIN)
    if coord_process is not None:
        poller.register(coord_process.sentinel, zmq.POLLIN)
    while any(conn_pending) or any(start_pending):
        events = poller.poll(STARTUP_POLL_PERIOD_MS)
        if not events:
            if any(conn_pending):
                logger.debug(
                    "Waiting for %d local, %d remote core engine proc(s) to connect.",
                    *conn_pending,
                )
            if any(start_pending):
                logger.debug(
                    "Waiting for %d local, %d remote core engine proc(s) to start.",
                    *start_pending,
                )
            continue
        if len(events) > 1 or events[0][0] != handshake_socket:
            # One of the local core processes exited.
            finished = proc_manager.finished_procs() if proc_manager else {}
            if coord_process is not None and coord_process.exitcode is not None:
                finished[coord_process.name] = coord_process.exitcode
            raise RuntimeError(
                "Engine core initialization failed. "
                "See root cause above. "
                f"Failed core proc(s): {finished}"
            )

        # Receive HELLO and READY messages from the input socket.
        eng_identity, ready_msg_bytes = handshake_socket.recv_multipart()
        eng_index = int.from_bytes(eng_identity, "little")
        engine = next((e for e in core_engines if e.identity == eng_identity), None)
        if engine is None:
            raise RuntimeError(
                f"Message from engine with unexpected data parallel rank: {eng_index}"
            )
        msg = msgspec.msgpack.decode(ready_msg_bytes)
        status, local, headless = msg["status"], msg["local"], msg["headless"]
        if local != engine.local:
            raise RuntimeError(
                f"{status} message from "
                f"{'local' if local else 'remote'} "
                f"engine {eng_index}, expected it to be "
                f"{'local' if engine.local else 'remote'}"
            )

        # Remote engines must be headless iff we aren't in hybrid dp lb mode.
        if not local and headless != remote_should_be_headless:
            if headless:
                raise RuntimeError(
                    f"Remote engine {eng_index} must not use "
                    f"--headless in external or hybrid dp lb "
                    f"mode"
                )
            else:
                raise RuntimeError(
                    f"Remote engine {eng_index} must use "
                    f"--headless unless in external or hybrid "
                    f"dp lb mode"
                )

        if status == "HELLO" and engine.state == CoreEngineState.NEW:
            # Send init message with DP config info.
            init_message = msgspec.msgpack.encode(
                EngineHandshakeMetadata(
                    addresses=addresses,
                    parallel_config={
                        k: getattr(parallel_config, k)
                        for k in (
                            "data_parallel_master_ip",
                            "data_parallel_master_port",
                            "_data_parallel_master_port_list",
                            "data_parallel_size",
                        )
                    }
                    if coordinated_dp
                    else {},
                )
            )
            handshake_socket.send_multipart((eng_identity, init_message), copy=False)
            conn_pending[0 if local else 1] -= 1
            start_pending[0 if local else 1] += 1
            engine.state = CoreEngineState.CONNECTED
        elif status == "READY" and engine.state == CoreEngineState.CONNECTED:
            # Setup KV cache config with initialization state from
            # engine core process. Sum values from all engines in DP case.
            num_gpu_blocks = cache_config.num_gpu_blocks or 0
            num_gpu_blocks += msg["num_gpu_blocks"]
            cache_config.num_gpu_blocks = num_gpu_blocks

            # In external DP LB mode, the coordinator address that the
            # front-end procs connect to is obtained from rank 0 via
            # one of the engine handshakes, and passed to the local
            # front-end process in the response from the other.
            if addresses.frontend_stats_publish_address is None:
                addresses.frontend_stats_publish_address = msg.get("dp_stats_address")

            # Validate config hash consistency across DP workers for MoE models.
            if coordinated_dp:
                worker_config_hash = msg.get("parallel_config_hash")
                expected_hash = parallel_config.compute_hash()
                if worker_config_hash != expected_hash:
                    raise RuntimeError(
                        f"Configuration mismatch detected for engine "
                        f"{eng_index}. All DP workers must have identical "
                        f"configurations for parameters that affect collective "
                        f"communication (e.g., enable_eplb, "
                        f"eplb_config.log_balancedness). "
                        f"Worker hash: {worker_config_hash}, "
                        f"Expected hash: {expected_hash}. "
                        f"Please ensure all workers are started with the same "
                        f"command-line arguments."
                    )

            start_pending[0 if local else 1] -= 1
            engine.state = CoreEngineState.READY
        else:
            raise RuntimeError(
                f"Unexpected {status} message for "
                f"{'local' if local else 'remote'} engine "
                f"{eng_index} in {engine.state} state."
            )

        logger.debug(
            "%s from %s core engine process %s.",
            status,
            "local" if local else "remote",
            eng_index,
        )
