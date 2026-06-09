#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Host:
    ssh_target: str
    node_addr: str


def strip_user(host: str) -> str:
    host = host.split("@", 1)[-1]
    return host.removeprefix("[").removesuffix("]")


def resolve_names(name: str) -> set[str]:
    names = {name}
    try:
        infos = socket.getaddrinfo(name, None)
    except socket.gaierror:
        return names
    for info in infos:
        names.add(info[4][0])
    return names


def local_names() -> set[str]:
    names = {"localhost", "127.0.0.1", "::1"}
    for getter in (socket.gethostname, socket.getfqdn):
        try:
            value = getter()
        except OSError:
            continue
        if value:
            names.add(value)
            names.update(resolve_names(value))
    try:
        output = subprocess.check_output(["hostname", "-I"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        output = ""
    names.update(part for part in output.split() if part)
    return names


def parse_hostfile(path: Path) -> list[Host]:
    hosts: list[Host] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        ssh_target = fields[0]
        node_addr = ""
        for token in fields[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key in {"addr", "node_addr", "node-ip-address", "ip"}:
                node_addr = value
        hosts.append(Host(ssh_target=ssh_target, node_addr=node_addr or strip_user(ssh_target)))

    if not hosts:
        raise SystemExit(f"error: hostfile has no usable nodes: {path}")
    return hosts


def is_local(host: Host, local: set[str]) -> bool:
    candidates = set()
    candidates.update(resolve_names(strip_user(host.ssh_target)))
    candidates.update(resolve_names(host.node_addr))
    return bool(candidates & local)


def detect_node_index(hosts: list[Host]) -> int:
    explicit = os.environ.get("MULTI_NODE_NODE_INDEX")
    if explicit is not None:
        try:
            idx = int(explicit)
        except ValueError:
            raise SystemExit(f"error: MULTI_NODE_NODE_INDEX={explicit} is not an integer")
        if not 0 <= idx < len(hosts):
            raise SystemExit(f"error: MULTI_NODE_NODE_INDEX={explicit} is invalid for {len(hosts)} host(s)")
        return idx

    local = local_names()
    for idx, host in enumerate(hosts):
        if is_local(host, local):
            return idx
    raise SystemExit("error: current node is not listed in hostfile; run from a listed host or set MULTI_NODE_NODE_INDEX")


def configure_cluster_shape(hosts: list[Host]) -> None:
    expected = len(hosts)
    actor_num_nodes = os.environ.get("ACTOR_NUM_NODES")
    if actor_num_nodes and actor_num_nodes != str(expected):
        raise SystemExit(f"error: ACTOR_NUM_NODES={actor_num_nodes} but hostfile has {expected} node(s)")
    os.environ["ACTOR_NUM_NODES"] = str(expected)


def ip_addr_interfaces() -> list[tuple[str, str]]:
    try:
        output = subprocess.check_output(["ip", "-o", "-4", "addr", "show"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []

    interfaces: list[tuple[str, str]] = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[2] != "inet":
            continue
        interfaces.append((fields[3].split("/", 1)[0], fields[1]))
    return interfaces


def detect_iface_for_ip(ip_addr: str) -> str:
    for local_ip, ifname in ip_addr_interfaces():
        if local_ip == ip_addr:
            return ifname
    return ""


def detect_default_10_iface() -> str:
    for local_ip, ifname in ip_addr_interfaces():
        if local_ip.startswith("10."):
            return ifname
    return ""


def configure_slime_network_env(node_addr: str) -> None:
    local_ip = os.environ.get("SLIME_HOST_IP") or node_addr
    if not local_ip:
        try:
            output = subprocess.check_output(["hostname", "-I"], text=True, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            output = ""
        local_ip = output.split()[0] if output.split() else ""

    if local_ip:
        os.environ["SLIME_HOST_IP"] = local_ip

    socket_ifname = os.environ.get("SOCKET_IFNAME") or detect_iface_for_ip(node_addr)
    if not socket_ifname and local_ip:
        socket_ifname = detect_iface_for_ip(local_ip)
    if not socket_ifname:
        socket_ifname = detect_default_10_iface()

    if socket_ifname:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", socket_ifname)
        os.environ.setdefault("TP_SOCKET_IFNAME", socket_ifname)
        os.environ.setdefault("NCCL_SOCKET_IFNAME", socket_ifname)
        os.environ.setdefault("NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME", socket_ifname)

    print(
        "multi_node_train: slime network "
        f"SLIME_HOST_IP={os.environ.get('SLIME_HOST_IP', '')} "
        f"GLOO_SOCKET_IFNAME={os.environ.get('GLOO_SOCKET_IFNAME', '')} "
        f"NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME', '')} "
        f"NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME={os.environ.get('NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME', '')}"
    )


def quote_env(name: str, value: str) -> str:
    return shlex.quote(f"{name}={value}")


def scp_opts_from_ssh_opts(ssh_opts: list[str]) -> list[str]:
    scp_opts: list[str] = []
    idx = 0
    while idx < len(ssh_opts):
        opt = ssh_opts[idx]
        if opt == "-p":
            scp_opts.append("-P")
            if idx + 1 < len(ssh_opts):
                scp_opts.append(ssh_opts[idx + 1])
                idx += 2
            else:
                idx += 1
            continue
        if opt.startswith("-p") and opt != "-p":
            scp_opts.append("-P" + opt[2:])
            idx += 1
            continue
        scp_opts.append(opt)
        idx += 1
    return scp_opts


def scp_opts_from_env(ssh_opts: list[str]) -> list[str]:
    explicit = os.environ.get("MULTI_NODE_SCP_OPTS")
    if explicit is not None:
        return shlex.split(explicit)
    return scp_opts_from_ssh_opts(ssh_opts)


def env_flag_enabled(name: str, default: str = "1") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}


def kernelgym_health_command(repo_root: Path) -> list[str]:
    return [
        "python3",
        str(repo_root / "scripts/check_kernelgym_health.py"),
        "--url",
        os.environ.get("KERNELGYM_URL", "http://127.0.0.1:20391"),
        "--timeout",
        os.environ.get("KERNELGYM_HEALTH_TIMEOUT", "5"),
        "--attempts",
        os.environ.get("KERNELGYM_HEALTH_ATTEMPTS", "3"),
        "--interval",
        os.environ.get("KERNELGYM_HEALTH_INTERVAL", "2"),
    ]


def check_kernelgym_health(repo_root: Path, label: str) -> None:
    if not env_flag_enabled("MULTI_NODE_KERNELGYM_HEALTH_CHECK", "1"):
        print(f"multi_node_train: skipping KernelGym health check for {label}")
        return

    command = kernelgym_health_command(repo_root)
    print(f"multi_node_train: checking KernelGym health for {label}: {' '.join(shlex.quote(arg) for arg in command)}")
    subprocess.run(command, check=True)


def forwarded_env(base: dict[str, str]) -> list[str]:
    optional_names = [
        "RAY_PORT",
        "RAY_DASHBOARD_PORT",
        "RAY_DASHBOARD_AGENT_GRPC_PORT",
        "RAY_DASHBOARD_AGENT_LISTEN_PORT",
        "RAY_RUNTIME_ENV_AGENT_PORT",
        "RAY_METRICS_EXPORT_PORT",
        "RAY_NUM_CPUS",
        "RAY_TMPDIR",
        "RAY_JOB_NO_FOLLOW",
        "SLIME_REQUIRE_HOST_HEALTH",
        "SLIME_RAY_KILL_PYTHON_ON_START",
        "MULTI_NODE_KERNELGYM_HEALTH_CHECK",
        "KERNELGYM_URL",
        "KERNELGYM_HEALTH_TIMEOUT",
        "KERNELGYM_HEALTH_ATTEMPTS",
        "KERNELGYM_HEALTH_INTERVAL",
    ]
    env = [quote_env(name, value) for name, value in base.items()]
    env.extend(quote_env(name, os.environ[name]) for name in optional_names if name in os.environ)
    return env


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_sha256(host: Host, path: Path, ssh_opts: list[str]) -> str:
    command = f"test -f {shlex.quote(str(path))} && sha256sum {shlex.quote(str(path))} | awk '{{print $1}}'"
    result = subprocess.run(
        ["ssh", *ssh_opts, host.ssh_target, command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""


def sync_file_to_worker(host: Host, local_path: Path, remote_path: Path, ssh_opts: list[str]) -> None:
    local_digest = sha256_file(local_path)
    remote_digest = remote_sha256(host, remote_path, ssh_opts)
    if remote_digest == local_digest:
        return

    print(f"Syncing {local_path} -> {host.ssh_target}:{remote_path}")
    subprocess.run(
        ["ssh", *ssh_opts, host.ssh_target, f"mkdir -p {shlex.quote(str(remote_path.parent))}"],
        check=True,
    )
    subprocess.run(["scp", *scp_opts_from_env(ssh_opts), str(local_path), f"{host.ssh_target}:{remote_path}"], check=True)


def sync_worker_inputs(host: Host, script: Path, train_script: Path, hostfile: Path, repo_root: Path, ssh_opts: list[str]) -> None:
    for path in (
        script,
        hostfile,
        train_script,
        repo_root / "scripts/check_kernelgym_health.py",
        repo_root / "scripts/ray/start_cluster.sh",
    ):
        sync_file_to_worker(host, path, path, ssh_opts)


def launch_worker(
    idx: int,
    hosts: list[Host],
    script: Path,
    train_script: Path,
    train_args: list[str],
    hostfile: Path,
    repo_root: Path,
) -> None:
    host = hosts[idx]
    log_file = Path(f"/tmp/slime_multi_node_train_worker_{idx}.log")
    base_env = {
        "MULTI_NODE_NODE_INDEX": str(idx),
        "HOSTFILE": str(hostfile),
        "MASTER_ADDR": os.environ["MASTER_ADDR"],
        "NODE_ADDR": host.node_addr,
        "ACTOR_NUM_NODES": os.environ["ACTOR_NUM_NODES"],
    }
    command = " ".join(
        [
            "cd",
            shlex.quote(str(repo_root)),
            "&&",
            "env",
            *forwarded_env(base_env),
            "python3",
            shlex.quote(str(script)),
            shlex.quote(str(train_script)),
            *(shlex.quote(arg) for arg in train_args),
            ">",
            shlex.quote(str(log_file)),
            "2>&1",
            "<",
            "/dev/null",
        ]
    )
    ssh_opts = shlex.split(os.environ.get("MULTI_NODE_SSH_OPTS", ""))
    timeout_s = int(os.environ.get("MULTI_NODE_WORKER_START_TIMEOUT_S", "180"))
    sync_worker_inputs(host, script, train_script, hostfile, repo_root, ssh_opts)
    print(f"Starting worker {idx}/{os.environ['ACTOR_NUM_NODES']}: {host.ssh_target} ({host.node_addr}); log: {log_file}")
    subprocess.run(["ssh", "-n", *ssh_opts, host.ssh_target, command], check=True, timeout=timeout_s)


def ray_job_address() -> str:
    dashboard_port = os.environ.get("RAY_DASHBOARD_PORT", "8265")
    return f"http://127.0.0.1:{dashboard_port}"


def start_ray_cluster(role: str, node_addr: str, repo_root: Path) -> None:
    check_kernelgym_health(repo_root, f"{role} {node_addr}")
    env = os.environ.copy()
    env["RAY_ROLE"] = role
    env["NODE_ADDR"] = node_addr
    env.setdefault("SLIME_RAY_KILL_PYTHON_ON_START", "0")
    command = f"cd {shlex.quote(str(repo_root))} && source {shlex.quote(str(repo_root / 'scripts/ray/start_cluster.sh'))}"
    subprocess.run(["bash", "-lc", command], env=env, check=True)


def start_worker_cluster(repo_root: Path) -> int:
    start_ray_cluster("worker", os.environ["NODE_ADDR"], repo_root)
    print(f"Ray worker on {os.environ.get('NODE_ADDR', '?')} joined head {os.environ['MASTER_ADDR']}.")
    return 0


def wait_for_head_ray() -> None:
    timeout_s = int(os.environ.get("MULTI_NODE_HEAD_WAIT_TIMEOUT_S", "300"))
    address = ray_job_address()
    deadline = time.monotonic() + timeout_s
    print(f"Waiting for Ray head job server at {address} before launching workers...")
    while time.monotonic() < deadline:
        if subprocess.run(["ray", "job", "list", f"--address={address}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            print(f"Ray head is ready: {address}")
            return
        time.sleep(2)
    raise TimeoutError(f"Ray head job server did not become ready within {timeout_s}s: {address}")


def alive_ray_nodes(address: str) -> int:
    result = subprocess.run(["ray", "list", "nodes", f"--address={address}", "--format", "json"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        return 0
    try:
        nodes = json.loads(result.stdout)
    except json.JSONDecodeError:
        return 0
    return sum(1 for node in nodes if node.get("state") == "ALIVE")


def wait_for_ray_nodes(expected: int) -> None:
    address = ray_job_address()
    timeout_s = int(os.environ.get("MULTI_NODE_NODE_WAIT_TIMEOUT_S", "300"))
    deadline = time.monotonic() + timeout_s
    print(f"Waiting for {expected} Ray node(s) to register...")
    last_alive = 0
    while time.monotonic() < deadline:
        last_alive = alive_ray_nodes(address)
        print(f"  alive nodes: {last_alive}/{expected}")
        if last_alive >= expected:
            return
        time.sleep(5)
    raise TimeoutError(f"only {last_alive}/{expected} nodes joined; aborting. Check worker ray start + network.")


def run_head_with_workers(hosts: list[Host], script: Path, train_script: Path, train_args: list[str], hostfile: Path, repo_root: Path) -> int:
    start_ray_cluster("head", os.environ["NODE_ADDR"], repo_root)
    wait_for_head_ray()
    for idx in range(1, len(hosts)):
        launch_worker(idx, hosts, script, train_script, train_args, hostfile, repo_root)
    wait_for_ray_nodes(len(hosts))

    env = os.environ.copy()
    env["SLIME_SKIP_RAY_START"] = "1"
    env["RAY_JOB_ADDRESS"] = ray_job_address()
    return subprocess.run(["bash", str(train_script), *train_args], env=env).returncode


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Launch a slime Ray training script across a hostfile.",
        epilog=(
            "Hostfile: one node per line; first non-comment line is head. "
            "First token is ssh target; optional tokens include addr=IP."
        ),
    )
    parser.add_argument("--hostfile", "-H", default=os.environ.get("HOSTFILE", "hostfile"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("train_script")
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    script = Path(__file__).resolve()
    repo_root = script.parent
    hostfile = Path(args.hostfile).expanduser().resolve()
    train_script = Path(args.train_script).expanduser().resolve()
    if not hostfile.is_file():
        raise SystemExit(f"error: hostfile not found: {args.hostfile}\n       create ./hostfile or pass --hostfile PATH")
    if not train_script.is_file():
        raise SystemExit(f"error: training script not found: {args.train_script}")

    hosts = parse_hostfile(hostfile)
    configure_cluster_shape(hosts)
    node_idx = detect_node_index(hosts)

    role = "head" if node_idx == 0 else "worker"
    os.environ["RAY_ROLE"] = role
    os.environ.setdefault("MASTER_ADDR", hosts[0].node_addr)
    os.environ.setdefault("NODE_ADDR", hosts[node_idx].node_addr)
    configure_slime_network_env(os.environ["NODE_ADDR"])

    print(
        f"multi_node_train: role={role} node_index={node_idx} node_addr={os.environ['NODE_ADDR']} "
        f"master={os.environ['MASTER_ADDR']} nodes={os.environ['ACTOR_NUM_NODES']}"
    )
    print(f"multi_node_train: hostfile={hostfile}")
    print(f"multi_node_train: train_script={train_script}")

    if args.dry_run:
        for idx, host in enumerate(hosts):
            head_or_worker = "head" if idx == 0 else "worker"
            print(f"  [{idx}] ssh={host.ssh_target} node_addr={host.node_addr} {head_or_worker}")
        return 0

    if role == "worker":
        return start_worker_cluster(repo_root)

    if role == "head" and len(hosts) > 1:
        return run_head_with_workers(hosts, script, train_script, args.train_args, hostfile, repo_root)

    return subprocess.run(["bash", str(train_script), *args.train_args]).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (RuntimeError, TimeoutError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
