# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import ipaddress
import os
import socket
import sys
import warnings
from collections.abc import (
    Iterator,
    Sequence,
)
from typing import Any
from uuid import uuid4

import psutil
import zmq
import zmq.asyncio
from urllib3.util import parse_url

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def close_sockets(sockets: Sequence[zmq.Socket | zmq.asyncio.Socket]):
    for sock in sockets:
        if sock is not None:
            sock.close(linger=0)


def get_ip() -> str:
    host_ip = envs.VLLM_HOST_IP
    if "HOST_IP" in os.environ and "VLLM_HOST_IP" not in os.environ:
        logger.warning(
            "The environment variable HOST_IP is deprecated and ignored, as"
            " it is often used by Docker and other software to"
            " interact with the container's network stack. Please "
            "use VLLM_HOST_IP instead to set the IP address for vLLM processes"
            " to communicate with each other."
        )
    if host_ip:
        return host_ip

    # IP is not set, try to get it from the network interface

    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
            return s.getsockname()[0]
    except Exception:
        pass

    # try ipv6
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as s:
            # Google's public DNS server, see
            # https://developers.google.com/speed/public-dns/docs/using#addresses
            s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
            return s.getsockname()[0]
    except Exception:
        pass

    warnings.warn(
        "Failed to get the IP address, using 0.0.0.0 by default."
        "The value can be set by the environment variable"
        " VLLM_HOST_IP or HOST_IP.",
        stacklevel=2,
    )
    return "0.0.0.0"


def test_loopback_bind(address: str, family: int) -> bool:
    try:
        s = socket.socket(family, socket.SOCK_DGRAM)
        s.bind((address, 0))  # Port 0 = auto assign
        s.close()
        return True
    except OSError:
        return False


def get_loopback_ip() -> str:
    loopback_ip = envs.VLLM_LOOPBACK_IP
    if loopback_ip:
        return loopback_ip

    # VLLM_LOOPBACK_IP is not set, try to get it based on network interface

    if test_loopback_bind("127.0.0.1", socket.AF_INET):
        return "127.0.0.1"
    elif test_loopback_bind("::1", socket.AF_INET6):
        return "::1"
    else:
        raise RuntimeError(
            "Neither 127.0.0.1 nor ::1 are bound to a local interface. "
            "Set the VLLM_LOOPBACK_IP environment variable explicitly."
        )


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def split_host_port(host_port: str) -> tuple[str, int]:
    # ipv6
    if host_port.startswith("["):
        host, port = host_port.rsplit("]", 1)
        host = host[1:]
        port = port.split(":")[1]
        return host, int(port)
    else:
        host, port = host_port.split(":")
        return host, int(port)


def join_host_port(host: str, port: int) -> str:
    if is_valid_ipv6_address(host):
        return f"[{host}]:{port}"
    else:
        return f"{host}:{port}"


def get_distributed_init_method(ip: str, port: int) -> str:
    return get_tcp_uri(ip, port)


def get_tcp_uri(ip: str, port: int) -> str:
    if is_valid_ipv6_address(ip):
        return f"tcp://[{ip}]:{port}"
    else:
        return f"tcp://{ip}:{port}"


def get_open_zmq_ipc_path() -> str:
    base_rpc_path = envs.VLLM_RPC_BASE_PATH
    return f"ipc://{base_rpc_path}/{uuid4()}"


def get_open_zmq_inproc_path() -> str:
    return f"inproc://{uuid4()}"


def get_open_port() -> int:
    """
    Get an open port for the vLLM process to listen on.
    An edge case to handle, is when we run data parallel,
    we need to avoid ports that are potentially used by
    the data parallel master process.
    Right now we reserve 10 ports for the data parallel master
    process. Currently it uses 2 ports.
    """
    if "VLLM_DP_MASTER_PORT" in os.environ:
        dp_master_port = envs.VLLM_DP_MASTER_PORT
        reserved_port_range = range(dp_master_port, dp_master_port + 10)
        while True:
            candidate_port = _get_open_port()
            if candidate_port not in reserved_port_range:
                return candidate_port
    return _get_open_port()


def get_open_ports_list(count: int = 5) -> list[int]:
    """Get a list of unique open ports.

    When VLLM_PORT is set, scans upward from that port, advancing
    the start position after each find so every port is unique.
    """
    ports_set = set[int]()
    if envs.VLLM_PORT is not None:
        next_port = envs.VLLM_PORT
        for _ in range(count):
            port = _get_open_port(start_port=next_port, max_attempts=1000)
            ports_set.add(port)
            next_port = port + 1
        return list(ports_set)
    else:
        while len(ports_set) < count:
            ports_set.add(get_open_port())

    return list(ports_set)


def _get_open_port(
    start_port: int | None = None,
    max_attempts: int | None = None,
) -> int:
    start_port = start_port if start_port is not None else envs.VLLM_PORT
    port = start_port
    if port is not None:
        attempts = 0
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                port += 1  # Increment port number if already in use
                logger.info("Port %d is already in use, trying port %d", port - 1, port)
            attempts += 1
            if max_attempts is not None and attempts >= max_attempts:
                raise RuntimeError(
                    f"Could not find open port after {max_attempts} "
                    f"attempts starting from port {start_port}"
                )
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def find_process_using_port(port: int) -> psutil.Process | None:
    # TODO: We can not check for running processes with network
    # port on macOS. Therefore, we can not have a full graceful shutdown
    # of vLLM. For now, let's not look for processes in this case.
    # Ref: https://www.florianreinhard.de/accessdenied-in-psutil/
    if sys.platform.startswith("darwin"):
        return None

    our_pid = os.getpid()
    for conn in psutil.net_connections():
        if conn.laddr.port == port and (conn.pid is not None and conn.pid != our_pid):
            try:
                return psutil.Process(conn.pid)
            except psutil.NoSuchProcess:
                return None
    return None


def split_zmq_path(path: str) -> tuple[str, str, str]:
    """Split a zmq path into its parts."""
    parsed = parse_url(path)
    if not parsed.scheme:
        raise ValueError(f"Invalid zmq path: {path}")

    scheme = parsed.scheme
    host = parsed.hostname or ""
    port = "" if parsed.port is None else str(parsed.port)
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]  # Remove brackets for IPv6 address

    if scheme == "tcp" and not all((host, port)):
        # The host and port fields are required for tcp
        raise ValueError(f"Invalid zmq path: {path}")

    if scheme != "tcp" and port:
        # port only makes sense with tcp
        raise ValueError(f"Invalid zmq path: {path}")

    return scheme, host, port


def make_zmq_path(scheme: str, host: str, port: int | None = None) -> str:
    """Make a ZMQ path from its parts.

    Args:
        scheme: The ZMQ transport scheme (e.g. tcp, ipc, inproc).
        host: The host - can be an IPv4 address, IPv6 address, or hostname.
        port: Optional port number, only used for TCP sockets.

    Returns:
        A properly formatted ZMQ path string.
    """
    if port is None:
        return f"{scheme}://{host}"
    if is_valid_ipv6_address(host):
        return f"{scheme}://[{host}]:{port}"
    return f"{scheme}://{host}:{port}"


# 该函数适配自 SGLang 项目，专门为大模型推理（SRT）优化了 ZMQ 的传输性能
def make_zmq_socket(
    ctx: zmq.asyncio.Context | zmq.Context,  # ZMQ 上下文（支持异步或同步）
    path: str,                              # 套接字连接路径（如 ipc://... 或 tcp://...）
    socket_type: Any,                       # ZMQ 套接字类型（如 PUSH, PULL, ROUTER 等）
    bind: bool | None = None,               # 是执行 bind (服务端) 还是 connect (客户端)
    identity: bytes | None = None,          # 套接字身份标识（用于 ROUTER/DEALER 路由）
    linger: int | None = None,              # 关闭时的残留时间（毫秒）
    router_handover: bool = False,          # 是否允许 ROUTER 身份接管
) -> zmq.Socket | zmq.asyncio.Socket:
    """创建一个具有正确 绑定(bind)/连接(connect) 语义并经过性能优化的 ZMQ 套接字。"""

    # 获取系统虚拟内存状态，用于后续动态调整缓冲区大小
    mem = psutil.virtual_memory()
    socket = ctx.socket(socket_type)

    # ---------------------------------------------------------
    # 1. 动态缓冲区优化 (Throughput Optimization)
    # ---------------------------------------------------------
    # 目的：在大内存机器上通过增大 TCP/IPC 缓冲区来提升大数据块（如 Tensor）的传输吞吐量。
    total_mem = mem.total / 1024**3      # 总内存 (GB)
    available_mem = mem.available / 1024**3 # 可用内存 (GB)
    
    # 性能策略：
    # - 如果系统内存充足 (>32GB 总量且 >16GB 可用)：
    #   设置一个巨大的 0.5GB (512MB) 缓冲区。这对于传输多模态大图片或 KV Cache 至关重要。
    # - 如果内存较小：
    #   使用系统默认值 (-1)，防止因分配过多缓冲区导致 OOM。
    buf_size = int(0.5 * 1024**3) if total_mem > 32 and available_mem > 16 else -1

    # ---------------------------------------------------------
    # 2. 自动判定 Bind/Connect 语义
    # ---------------------------------------------------------
    # 如果用户没指定 bind，则根据 ZMQ 的标准惯例自动判定：
    # PUSH, SUB, XSUB 通常作为下游客户端 (connect)
    if bind is None:
        bind = socket_type not in (zmq.PUSH, zmq.SUB, zmq.XSUB)

    # ---------------------------------------------------------
    # 3. 高水位标志 (HWM) 与 缓冲区设置
    # ---------------------------------------------------------
    # RCVHWM/SNDHWM = 0 表示不限制内存队列长度，防止因队列满而丢弃推理请求。
    # RCVBUF/SNDBUF 设置为上面计算出的 buf_size。
    
    # 接收端优化
    if socket_type in (zmq.PULL, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    # 发送端优化
    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    # ---------------------------------------------------------
    # 4. 路由接管机制 (ROUTER Handover)
    # ---------------------------------------------------------
    # 场景：当一个 GPU 进程崩溃后立即重启，它会使用相同的 Identity 重新连接。
    # 开启此选项后，ROUTER 会允许新连接直接替换掉旧的“死连接”，实现无缝恢复。
    if socket_type == zmq.ROUTER and router_handover:
        socket.setsockopt(zmq.ROUTER_HANDOVER, 1)

    # ---------------------------------------------------------
    # 5. 其他常规设置
    # ---------------------------------------------------------
    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)

    if linger is not None:
        socket.setsockopt(zmq.LINGER, linger)

    # XPUB 模式下开启详细日志，用于监控订阅状态
    if socket_type == zmq.XPUB:
        socket.setsockopt(zmq.XPUB_VERBOSE, True)

    # ---------------------------------------------------------
    # 6. IPv6 自动检测与适配
    # ---------------------------------------------------------
    # 解析路径，如果发现是基于 TCP 的 IPv6 地址，则显式开启套接字的 IPv6 支持。
    scheme, host, _ = split_zmq_path(path)
    if scheme == "tcp" and is_valid_ipv6_address(host):
        socket.setsockopt(zmq.IPV6, 1)

    # ---------------------------------------------------------
    # 7. 最终连接
    # ---------------------------------------------------------
    if bind:
        socket.bind(path)
    else:
        socket.connect(path)

    return socket


@contextlib.contextmanager
def zmq_socket_ctx(
    path: str,
    socket_type: Any,
    bind: bool | None = None,
    linger: int = 0,
    identity: bytes | None = None,
    router_handover: bool = False,
) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    ctx = zmq.Context()  # type: ignore[attr-defined]
    try:
        yield make_zmq_socket(
            ctx,
            path,
            socket_type,
            bind=bind,
            identity=identity,
            router_handover=router_handover,
        )
    except KeyboardInterrupt:
        logger.debug("Got Keyboard Interrupt.")

    finally:
        ctx.destroy(linger=linger)
