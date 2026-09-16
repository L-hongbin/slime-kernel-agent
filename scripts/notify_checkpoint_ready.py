#!/usr/bin/env python3
"""Read-only checkpoint monitor; queue a Codex message, never stop/train/evaluate.

Run on the current Codex host. SSH probes read only the named checkpoint.
Delivery uses the supported ``codex queue`` CLI, not private database writes.
An event ID lets the receiving agent deduplicate an ambiguous delivery retry.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import logging
import os
import re
import shlex
import sqlite3
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LOG = logging.getLogger("checkpoint-ready")
REMOTE_PROBE = r"""
import hashlib,json,pickle,sys,urllib.request
from pathlib import Path
a=json.loads(sys.argv[1]); root=Path(a['checkpoint_root']); iteration=a['iteration']
p=root/f'iter_{iteration:07d}'; marker=root/'latest_checkpointed_iteration.txt'
tracker=int(marker.read_text().strip()) if marker.is_file() else None
out={'tracker':tracker,'exists':p.is_dir(),'shards':{},'global_files':{}}
if p.is_dir():
 out['shards']={f.name:f.stat().st_size for f in p.glob('*.distcp') if f.is_file()}
 out['global_files']={n:(p/n).stat().st_size for n in ['.metadata','common.pt','metadata.json'] if (p/n).is_file()}
if tracker is not None and tracker>=iteration and len(out['global_files'])==3 and all(out['global_files'].values()):
 import torch
 torch.set_num_threads(1)
 metadata=pickle.loads((p/'.metadata').read_bytes())
 common=torch.load(p/'common.pt',map_location='cpu',weights_only=False)
 required={}
 for entry in metadata.storage_data.values():
  name=str(entry.relative_path)
  if Path(name).name!=name or entry.offset<0 or entry.length<=0:raise ValueError('Invalid DCP storage extent')
  required[name]=max(required.get(name,0),entry.offset+entry.length)
 out['metadata']={'iteration':common['iteration'],'required_bytes':required,
  'storage_entries':len(metadata.storage_data),'sha256':hashlib.sha256((p/'.metadata').read_bytes()).hexdigest()}
if a.get('job_id'):
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 job=json.load(opener.open('http://127.0.0.1:8268/api/jobs/'+a['job_id'],timeout=8))
 out['job']={k:job.get(k) for k in ['submission_id','status','start_time','end_time']}
 out['runtime_package']=job.get('runtime_env',{}).get('working_dir')
 out['formal_config_sha256']=hashlib.sha256(Path(a['formal_config']).read_bytes()).hexdigest()
print(json.dumps(out))
"""


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def event_id(config, kind="ready"):
    identity = {k: config[k] for k in ("thread_id", "job_id", "checkpoint_root", "iteration")}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return f"checkpoint-{kind}-{digest}"


def validate_config(config):
    uuid.UUID(config["thread_id"])
    root = Path(config["checkpoint_root"])
    if not root.is_absolute() or root.name != "checkpoints" or "experiments" not in root.parts or ".." in root.parts:
        raise ValueError("Expected an explicit experiment checkpoint root")
    if type(config["iteration"]) is not int or config["iteration"] < 0:
        raise ValueError("Expected a nonnegative checkpoint iteration")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", config["job_id"]):
        raise ValueError("Invalid job ID")
    if len(config["hosts"]) < 2 or len({h["label"] for h in config["hosts"]}) != len(config["hosts"]):
        raise ValueError("Expected distinct actor hosts")
    if sum(bool(h.get("job_head")) for h in config["hosts"]) != 1:
        raise ValueError("Expected exactly one job head")
    shard_names = []
    for host in config["hosts"]:
        if not re.fullmatch(r"[A-Za-z0-9_.@:-]+", host["target"]):
            raise ValueError("Invalid SSH target")
        names = host["expected_shards"]
        if not names or not all(re.fullmatch(r"__\d+_\d+\.distcp", n) for n in names):
            raise ValueError("Expected explicit DCP shard names")
        shard_names.extend(names)
    if len(set(shard_names)) != len(shard_names):
        raise ValueError("Node-local shard names must not overlap")
    if not 10 <= config.get("poll_seconds", 60) <= 600:
        raise ValueError("Poll interval must be between 10 and 600 seconds")
    return config


def inspect_host(host, config):
    payload = {k: config[k] for k in ("checkpoint_root", "iteration")}
    if host.get("job_head"):
        payload.update({k: config[k] for k in ("job_id", "formal_config")})
    remote_command = shlex.join(["python3", "-c", REMOTE_PROBE, json.dumps(payload)])
    result = subprocess.run(
        [
            "ssh",
            "-p",
            str(config.get("ssh_port", 23538)),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=10",
            "-o",
            "ServerAliveCountMax=2",
            host["target"],
            remote_command,
        ],
        text=True,
        capture_output=True,
        timeout=50,
        check=True,
    )
    return json.loads(result.stdout)


def inspect(config):
    with ThreadPoolExecutor(max_workers=len(config["hosts"])) as pool:
        futures = {host["label"]: pool.submit(inspect_host, host, config) for host in config["hosts"]}
        hosts = {label: future.result() for label, future in futures.items()}
    return {"checked_at_utc": utc_now(), "hosts": hosts}


def assess(config, snapshot):
    """Require publication, the exact two-node layout and every metadata extent."""
    reasons = []
    union = {}
    metadata = []
    for host in config["hosts"]:
        label = host["label"]
        observed = snapshot["hosts"].get(label, {})
        shards = observed.get("shards", {})
        if not observed.get("exists"):
            reasons.append(f"{label}: target absent")
        if set(shards) != set(host["expected_shards"]) or any(size <= 0 for size in shards.values()):
            reasons.append(f"{label}: incomplete shard layout")
        if set(union) & set(shards):
            reasons.append("overlapping shard names")
        union.update(shards)
        if observed.get("metadata"):
            metadata.append(observed)
        if host.get("job_head"):
            if observed.get("job", {}).get("submission_id") != config["job_id"]:
                reasons.append("job identity mismatch")
            if observed.get("runtime_package") != config["runtime_package"]:
                reasons.append("runtime package mismatch")
            if observed.get("formal_config_sha256") != config["formal_config_sha256"]:
                reasons.append("formal configuration changed")
    valid_metadata = [
        m
        for m in metadata
        if m.get("tracker", -1) >= config["iteration"]
        and m["metadata"].get("iteration") == config["iteration"]
        and all(m.get("global_files", {}).get(n, 0) > 0 for n in (".metadata", "common.pt", "metadata.json"))
    ]
    if not valid_metadata:
        reasons.append("global save completion not verified")
    else:
        for observed in valid_metadata:
            required = observed["metadata"].get("required_bytes", {})
            if not required or set(required) != set(union):
                reasons.append("metadata shard references differ from two-node union")
            for name, end in required.items():
                if end <= 0 or union.get(name, 0) < end:
                    reasons.append(f"incomplete storage extent: {name}")
    return {"ready": not reasons, "reasons": reasons}


def ready_message(config, evidence_path):
    step = config["iteration"] + 1
    return (
        f"[checkpoint-ready event_id={event_id(config)}]\n"
        f"外部只读监控确认当前实验 step{step}（iter_{config['iteration']:07d}）已完成全局保存，"
        "两机分片及 metadata 文件范围通过检查，并连续两次确认。\n"
        f"Ray job: {config['job_id']}\nCheckpoint root: {config['checkpoint_root']}\n"
        f"检查证据: {evidence_path}\n"
        "用户已明确要求：监控只发 Queue message，由当前主 Agent 判断和执行停止训练、启动测试。"
        "请重新核查现场与 checkpoint 完整性，确认事件尚未处理，再停止这个指定训练作业，"
        "保留并汇集目标 checkpoint，校验转换和四节点代码/配置/权重同步后，"
        "沿用此前 baseline 对照协议启动 KernelBench L1–L3 测试（每题 8 条、三轮 24K/32K/40K）。"
        "若有异常先诊断，禁止重复提交评测或影响其它实验。"
        "外部监控没有停止任何训练、没有修改 checkpoint、没有启动测试。"
    )


def queued_event_id(config, marker):
    """Read-only reconciliation of a pending CLI delivery; never mutate Codex DB."""
    database = config.get("queue_database")
    if not database or not Path(database).is_file():
        return None
    try:
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=5) as connection:
            row = connection.execute(
                "SELECT id FROM queued_items WHERE thread_id=? AND instr(payload_json,?)>0 LIMIT 1",
                (config["thread_id"], marker),
            ).fetchone()
    except sqlite3.Error:
        LOG.warning("Optional queue reconciliation unavailable; using CLI acknowledgement")
        return None
    return row[0] if row else None


def send_queue(config, message, marker):
    # Normal retries deduplicate pending events. If a crash happens after delivery
    # and consumption but before our receipt is saved, the agent deduplicates marker.
    existing = queued_event_id(config, marker)
    if existing:
        return {"queue_id": existing, "reconciled_pending": True, "at_utc": utc_now()}
    result = subprocess.run(
        [config["codex_binary"], "queue", "--thread", config["thread_id"], "--message", message],
        text=True,
        capture_output=True,
        timeout=45,
        check=True,
    )
    match = re.search(r"Queued message ([a-zA-Z0-9-]+) for thread ([a-zA-Z0-9-]+)", result.stdout)
    if not match or match[2] != config["thread_id"]:
        raise RuntimeError("No matching Codex queue acknowledgement")
    return {"queue_id": match[1], "at_utc": utc_now()}


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_monitor(config, state_path, *, probe=inspect, sender=send_queue, sleep=time.sleep, max_polls=None):
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    identity = event_id(config)
    if state and (state.get("event_id") != identity or state.get("config_sha256") != config["config_sha256"]):
        raise ValueError("State belongs to a different monitoring configuration")
    if state.get("delivery"):
        LOG.info("Event already queued: %s", state["delivery"]["queue_id"])
        return state
    state.update(
        event_id=identity,
        config_sha256=config["config_sha256"],
        pid=os.getpid(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        remote_probe_sha256=hashlib.sha256(REMOTE_PROBE.encode()).hexdigest(),
        started_at_utc=utc_now(),
        authority="read checkpoints and queue messages only",
    )
    ready_polls = 0
    errors = 0
    previous_phase = None
    polls = 0
    while max_polls is None or polls < max_polls:
        polls += 1
        delivery_attempt = False
        try:
            snapshot = probe(config)
            result = assess(config, snapshot)
            state.update(snapshot=snapshot, assessment=result, last_check_utc=utc_now(), last_error=None)
            ready_polls = ready_polls + 1 if result["ready"] else 0
            state.update(ready_confirmations=ready_polls, phase="confirming" if ready_polls else "waiting")
            atomic_json(state_path, state)
            if ready_polls >= 2:
                delivery_attempt = True
                receipt = sender(config, ready_message(config, state_path), identity)
                state.update(phase="queued", delivery=receipt, notified_at_utc=utc_now())
                atomic_json(state_path, state)
                LOG.info("Queued checkpoint-ready event: %s", receipt["queue_id"])
                return state
            jobs = [h["job"] for h in snapshot["hosts"].values() if "job" in h]
            if not result["ready"] and any(j.get("status") in ("FAILED", "STOPPED", "SUCCEEDED") for j in jobs):
                raise RuntimeError("Training is terminal but target checkpoint readiness is unverified")
            published = any(
                isinstance(h.get("tracker"), int) and h["tracker"] >= config["iteration"]
                for h in snapshot["hosts"].values()
            )
            identity_error = any(
                reason in result["reasons"]
                for reason in ("job identity mismatch", "runtime package mismatch", "formal configuration changed")
            )
            if not result["ready"] and (published or identity_error):
                raise RuntimeError(
                    "Published target or experiment identity failed verification: " + "; ".join(result["reasons"])
                )
            errors = 0
        except Exception as error:
            ready_polls = 1 if delivery_attempt else 0
            errors += 1
            # Do not print subprocess argv/environment; it is unnecessary evidence.
            state.update(
                phase="probe_error",
                last_check_utc=utc_now(),
                last_error=type(error).__name__ + ": " + str(error)[:180],
                consecutive_errors=errors,
            )
            atomic_json(state_path, state)
            if errors >= 5 and not state.get("problem_delivery"):
                marker = event_id(config, "problem")
                message = (
                    f"[checkpoint-monitor problem event_id={marker}] "
                    f"step{config['iteration']+1} 监控连续检查失败，请主 Agent 核查。"
                    f"证据：{state_path}。这不是 checkpoint-ready 通知，不能据此停止训练或启动测试。"
                )
                try:
                    state["problem_delivery"] = sender(config, message, marker)
                    atomic_json(state_path, state)
                except Exception:
                    LOG.warning("Problem message queue unavailable; will retry")
        if state["phase"] != previous_phase:
            LOG.info("phase=%s ready_confirmations=%s", state["phase"], ready_polls)
            previous_phase = state["phase"]
        if max_polls is None or polls < max_polls:
            sleep(config.get("poll_seconds", 60))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Read-only check: never enqueue or write state")
    mode.add_argument("--watch", action="store_true", help="Poll then queue; no training/service mutations")
    parser.add_argument("--state-dir", type=Path)
    args = parser.parse_args()
    config = validate_config(json.loads(args.config.read_text()))
    config["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    if args.once:
        snapshot = inspect(config)
        print(
            json.dumps(
                {
                    "config_sha256": config["config_sha256"],
                    "snapshot": snapshot,
                    "assessment": assess(config, snapshot),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.state_dir is None:
        parser.error("--watch requires --state-dir")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with (args.state_dir / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_monitor(config, (args.state_dir / "state.json").resolve())


if __name__ == "__main__":
    main()
