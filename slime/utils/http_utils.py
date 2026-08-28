import asyncio
import ipaddress
import json
import logging
import os
import random
import socket
import weakref

import httpx

logger = logging.getLogger(__name__)

SLIME_HOST_IP_ENV = "SLIME_HOST_IP"


def find_available_port(base_port: int):
    port = base_port + random.randint(100, 1000)
    while True:
        if is_port_available(port):
            return port
        if port < 60000:
            port += 42
        else:
            port -= 43


def is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except OSError:
            return False
        except OverflowError:
            return False


def get_host_info():
    hostname = socket.gethostname()

    if env_overwrite_local_ip := os.getenv(SLIME_HOST_IP_ENV, None):
        return hostname, env_overwrite_local_ip

    def _is_loopback(ip):
        return ip.startswith("127.") or ip == "::1"

    def _resolve_ip(family, test_target_ip):
        """
        Attempt to get the local LAN IP for the specific family (IPv4/IPv6).
        Strategy: UDP Probe (Preferred) -> Hostname Resolution (Fallback) -> None
        """

        # Strategy 1: UDP Connect Probe (Most accurate, relies on routing table)
        # Useful when the machine has a default gateway or internet access.
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as s:
                # The IP doesn't need to be reachable, but the routing table must exist.
                s.connect((test_target_ip, 80))
                ip = s.getsockname()[0]
                if not _is_loopback(ip):
                    return ip
        except Exception:
            pass  # Route unreachable or network error, move to next strategy.

        # Strategy 2: Hostname Resolution (Fallback for offline clusters)
        # Useful for offline environments where UDP connect fails but /etc/hosts is configured.
        try:
            # getaddrinfo allows specifying the family (AF_INET or AF_INET6)
            # Result format: [(family, type, proto, canonname, sockaddr), ...]
            infos = socket.getaddrinfo(hostname, None, family=family, type=socket.SOCK_STREAM)

            for info in infos:
                ip = info[4][0]  # The first element of sockaddr is the IP
                # Must filter out loopback addresses to avoid "127.0.0.1" issues
                if not _is_loopback(ip):
                    return ip
        except Exception:
            pass

        return None

    prefer_ipv6 = os.getenv("SLIME_PREFER_IPV6", "0").lower() in ("1", "true", "yes", "on")
    local_ip = None
    final_fallback = "127.0.0.1"

    if prefer_ipv6:
        # [Strict Mode] IPv6 Only
        # 1. Try UDP V6 Probe
        # 2. Try Hostname Resolution (V6)
        # If failed, fallback to V6 loopback. Never mix with V4.
        local_ip = _resolve_ip(socket.AF_INET6, "2001:4860:4860::8888")
        final_fallback = "::1"
    else:
        # [Strict Mode] IPv4 Only (Default)
        # 1. Try UDP V4 Probe
        # 2. Try Hostname Resolution (V4)
        # If failed, fallback to V4 loopback. Never mix with V6.
        local_ip = _resolve_ip(socket.AF_INET, "8.8.8.8")
        final_fallback = "127.0.0.1"

    return hostname, local_ip or final_fallback


def _wrap_ipv6(host):
    """Wrap IPv6 address in [] if needed."""
    try:
        ipaddress.IPv6Address(host.strip("[]"))
        return f"[{host.strip('[]')}]"
    except ipaddress.AddressValueError:
        return host


_PROXY_ENV_VARS = ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")


def scrub_proxy_env() -> None:
    """Remove HTTP(S) proxy variables from the current process environment.

    The sglang router only talks to rollout engines inside the cluster, but its
    Rust HTTP client (reqwest) honors proxy env vars by default and NO_PROXY
    rarely covers every worker IP. A cluster egress proxy (e.g. clash) kills
    tunneled connections that stay silent for ~60s, which aborts every long
    non-streaming /generate in flight: engines log client disconnects, the
    router circuit breaker opens on the resulting failures, and all new
    requests get 503 no_available_workers.

    Set SLIME_SCRUB_PROXY=0 for deployments that intentionally proxy
    router-to-worker traffic.
    """
    if os.environ.get("SLIME_SCRUB_PROXY", "1") == "0":
        return
    removed = [key for key in _PROXY_ENV_VARS if os.environ.pop(key, None) is not None]
    if removed:
        logger.debug(f"Scrubbed proxy env vars for router process: {removed}")


def run_router(args):
    try:
        # Must happen before launch_router builds the reqwest client.
        scrub_proxy_env()
        from sglang_router.launch_router import launch_router

        router = launch_router(args)
        if router is None:
            return 1
        return 0
    except Exception as e:
        logger.info(e)
        return 1


_http_client: httpx.AsyncClient | None = None
_http_clients_by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = (
    weakref.WeakKeyDictionary()
)
_client_concurrency: int = 0

# Optional Ray-based distributed POST dispatch
_distributed_post_enabled: bool = False
_post_actors: list[object] = []
_post_actor_idx: int = 0


def _make_http_client(max_connections: int | None = None) -> httpx.AsyncClient:
    concurrency = max(1, int(max_connections or _client_concurrency or 1))
    return httpx.AsyncClient(
        limits=httpx.Limits(max_connections=concurrency),
        timeout=httpx.Timeout(None),
        trust_env=False,  # internal SGLang comm only — never route through system proxy
    )


def get_http_client(max_connections: int | None = None) -> httpx.AsyncClient:
    """Return an AsyncClient scoped to the current asyncio event loop."""
    global _http_client

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        if _http_client is None or _http_client.is_closed:
            _http_client = _make_http_client(max_connections)
        return _http_client

    client = _http_clients_by_loop.get(loop)
    if client is None or client.is_closed:
        client = _make_http_client(max_connections)
        _http_clients_by_loop[loop] = client

    _http_client = client
    return client


def _next_actor():
    global _post_actor_idx
    if not _post_actors:
        return None
    actor = _post_actors[_post_actor_idx % len(_post_actors)]
    _post_actor_idx = (_post_actor_idx + 1) % len(_post_actors)
    return actor


async def _post(client, url, payload, max_retries=60, headers=None):
    retry_count = 0
    while retry_count < max_retries:
        response = None
        try:
            response = await client.post(url, json=payload or {}, headers=headers)
            response.raise_for_status()
            content = await response.aread()
            try:
                output = json.loads(content)
            except json.JSONDecodeError:
                output = content.decode() if isinstance(content, bytes) else content
        except Exception as e:
            retry_count += 1

            if isinstance(e, httpx.HTTPStatusError):
                response_text = e.response.text
            else:
                response_text = None

            logger.info(
                f"Error: {e}, retrying... (attempt {retry_count}/{max_retries}, url={url}, response={response_text})"
            )
            if retry_count >= max_retries:
                logger.info(f"Max retries ({max_retries}) reached, failing... (url={url})")
                raise e
            await asyncio.sleep(1)
            continue
        finally:
            if response is not None:
                await response.aclose()
        break

    return output


def get_rollout_num_engines(args) -> int:
    """Return the number of rollout HTTP engines behind the router."""
    if (num_engines := getattr(args, "rollout_num_engines", None)) is not None:
        return int(num_engines)

    rollout_num_gpus = getattr(args, "rollout_num_gpus", None) or 0
    rollout_num_gpus_per_engine = getattr(args, "rollout_num_gpus_per_engine", None) or 1
    if rollout_num_gpus <= 0:
        return 0
    return max(1, rollout_num_gpus // rollout_num_gpus_per_engine)


def get_sglang_client_concurrency(args) -> int:
    """Return client concurrency capped by each engine's running-request limit."""
    num_engines = get_rollout_num_engines(args)
    per_engine_concurrency = args.sglang_server_concurrency
    max_running_requests = getattr(args, "sglang_max_running_requests", None)
    if max_running_requests is not None:
        per_engine_concurrency = min(per_engine_concurrency, int(max_running_requests))
    return max(1, int(per_engine_concurrency)) * num_engines


def init_http_client(args):
    """Initialize HTTP client and optionally enable distributed POST via Ray."""
    global _http_client, _client_concurrency, _distributed_post_enabled
    if get_rollout_num_engines(args) <= 0:
        return

    _client_concurrency = get_sglang_client_concurrency(args)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        _http_client = None
    else:
        get_http_client(_client_concurrency)

    # Optionally initialize distributed POST via Ray without changing interfaces
    if args.use_distributed_post:
        _init_ray_distributed_post(args)
        _distributed_post_enabled = True


def _init_ray_distributed_post(args):
    """Initialize one or more Ray async actors per node for HTTP POST.

    Uses NodeAffinitySchedulingStrategy to place actors on alive Ray nodes.
    Creates ``args.num_gpus_per_node`` actors per node.
    """
    global _post_actors
    if _post_actors:
        return  # Already initialized

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from slime.ray.utils import add_default_ray_env_vars

    # Discover alive nodes
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    if not nodes:
        raise RuntimeError("No alive Ray nodes to place HTTP POST actors.")

    # Define the async actor
    @ray.remote
    class _HttpPosterActor:
        def __init__(self, concurrency: int):
            self._concurrency = max(1, int(concurrency))

        async def do_post(self, url, payload, max_retries=60, headers=None):
            return await _post(get_http_client(self._concurrency), url, payload, max_retries, headers=headers)

    # Create actors per node
    created = []
    total_actors = max(1, len(nodes) * args.num_gpus_per_node)
    per_actor_conc = max(1, (_client_concurrency + total_actors - 1) // total_actors)

    for node in nodes:
        node_id = node["NodeID"]
        scheduling = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        for _ in range(args.num_gpus_per_node):
            actor = _HttpPosterActor.options(
                name=None,
                lifetime="detached",
                runtime_env={"env_vars": add_default_ray_env_vars()},
                scheduling_strategy=scheduling,
                max_concurrency=per_actor_conc,
                # Use tiny CPU to schedule
                num_cpus=0.001,
            ).remote(per_actor_conc)
            created.append(actor)

    _post_actors = created


async def post(url, payload, max_retries=60, headers=None):
    # If distributed mode is enabled and actors exist, dispatch via Ray.
    if _distributed_post_enabled and _post_actors:
        try:
            actor = _next_actor()
            if actor is not None:
                # Await the Ray ObjectRef directly. The previous
                # `asyncio.to_thread(ray.get, obj_ref)` blocked an OS thread
                # from the default ThreadPoolExecutor (capped at
                # `min(32, cpu+4)`), which becomes a hard upper bound on the
                # number of in-flight POSTs that can be waited on in parallel
                # and produces large tail latencies under high concurrency.
                obj_ref = actor.do_post.remote(url, payload, max_retries, headers=headers)
                return await obj_ref
        except Exception as e:
            logger.info(f"[http_utils] Distributed POST failed, falling back to local: {e} (url={url})")
            # fall through to local

    return await _post(get_http_client(), url, payload, max_retries, headers=headers)


async def get(url):
    response = await get_http_client().get(url)
    response.raise_for_status()
    content = await response.aread()
    output = json.loads(content)
    return output
