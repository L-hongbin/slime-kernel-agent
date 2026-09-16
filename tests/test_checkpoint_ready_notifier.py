from __future__ import annotations

import copy
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import notify_checkpoint_ready as notifier

NUM_GPUS = 0


@pytest.fixture
def config():
    return {
        "thread_id": "01a05c40-a7b4-7612-96d6-555be8518042",
        "job_id": "test-training",
        "iteration": 99,
        "checkpoint_root": "/data/experiments/test/checkpoints",
        "runtime_package": "gcs://test.zip",
        "formal_config_sha256": "config-hash",
        "config_sha256": "monitor-config-hash",
        "codex_binary": "/bin/codex",
        "poll_seconds": 60,
        "hosts": [
            {"label": "head", "target": "root@head", "job_head": True, "expected_shards": ["__1_0.distcp"]},
            {"label": "actor", "target": "root@actor", "expected_shards": ["__0_0.distcp"]},
        ],
    }


@pytest.fixture
def ready(config):
    return {
        "hosts": {
            "head": {
                "exists": True,
                "tracker": None,
                "shards": {"__1_0.distcp": 100},
                "global_files": {},
                "job": {"submission_id": config["job_id"], "status": "RUNNING"},
                "runtime_package": config["runtime_package"],
                "formal_config_sha256": "config-hash",
            },
            "actor": {
                "exists": True,
                "tracker": 99,
                "shards": {"__0_0.distcp": 120},
                "global_files": {".metadata": 10, "common.pt": 10, "metadata.json": 10},
                "metadata": {"iteration": 99, "required_bytes": {"__0_0.distcp": 120, "__1_0.distcp": 100}},
            },
        }
    }


def test_complete_split_checkpoint_only_coordinator_has_metadata(config, ready):
    assert notifier.validate_config(config) == config
    assert notifier.assess(config, ready) == {"ready": True, "reasons": []}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["hosts"]["actor"].update(tracker=79),
        lambda r: r["hosts"]["actor"]["metadata"].update(iteration=79),
        lambda r: r["hosts"]["actor"]["global_files"].update({".metadata": 0}),
        lambda r: r["hosts"]["actor"].pop("metadata"),
        lambda r: r["hosts"]["head"]["shards"].update({"__1_0.distcp": 99}),
        lambda r: r["hosts"]["head"].update(shards={}),
        lambda r: r["hosts"]["head"].update(exists=False),
        lambda r: r["hosts"]["actor"]["metadata"].update(required_bytes={}),
        lambda r: r["hosts"]["actor"]["metadata"]["required_bytes"].update({"__2_0.distcp": 40}),
        lambda r: r["hosts"]["head"].update(formal_config_sha256="changed"),
        lambda r: r["hosts"]["head"].update(runtime_package="gcs://wrong.zip"),
        lambda r: r["hosts"]["head"]["job"].update(submission_id="another-training"),
        lambda r: r["hosts"].pop("head"),
    ],
)
def test_incomplete_or_wrong_lineage_is_not_ready(config, ready, mutation):
    mutation(ready)
    assert not notifier.assess(config, ready)["ready"]


def test_newer_published_tracker_does_not_change_exact_target(config, ready):
    ready["hosts"]["actor"]["tracker"] = 119
    assert notifier.assess(config, ready)["ready"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(thread_id="wrong"),
        lambda c: c.update(checkpoint_root="/"),
        lambda c: c.update(iteration=-1),
        lambda c: c.update(poll_seconds=0),
        lambda c: c["hosts"][0].update(target="root@head; echo unsafe"),
        lambda c: c["hosts"][1].update(expected_shards=["__1_0.distcp"]),
        lambda c: c["hosts"][1].update(label="head"),
        lambda c: c["hosts"][0].update(job_head=False),
    ],
)
def test_configuration_identity_validation(config, mutation):
    mutation(config)
    with pytest.raises(ValueError):
        notifier.validate_config(config)


def test_wait_then_two_ready_checks_enqueue_once_and_restart_noop(config, ready, tmp_path):
    waiting = copy.deepcopy(ready)
    waiting["hosts"]["actor"].pop("metadata")
    snapshots = iter([waiting, ready, ready])
    sent = []

    def sender(c, message, marker):
        sent.append((message, marker))
        return {"queue_id": "message-id"}

    state_path = tmp_path / "state.json"
    state = notifier.run_monitor(
        config, state_path, probe=lambda c: next(snapshots), sender=sender, sleep=lambda s: None, max_polls=3
    )
    assert state["phase"] == "queued"
    assert len(sent) == 1
    assert "step100（iter_0000099）" in sent[0][0]
    assert "外部监控没有停止任何训练" in sent[0][0]
    notifier.run_monitor(config, state_path, probe=lambda c: pytest.fail("must not poll after receipt"), sender=sender)
    assert len(sent) == 1


def test_ready_observation_must_be_consecutive(config, ready, tmp_path):
    waiting = copy.deepcopy(ready)
    waiting["hosts"]["head"].update(shards={})
    snapshots = iter([ready, waiting, ready])
    state = notifier.run_monitor(
        config,
        tmp_path / "state.json",
        probe=lambda c: next(snapshots),
        sender=lambda *a: pytest.fail("premature queue"),
        sleep=lambda s: None,
        max_polls=3,
    )
    assert state["phase"] == "confirming"


def test_probe_failure_alert_is_not_a_ready_event(config, tmp_path):
    sent = []

    def failing(c):
        raise TimeoutError("SSH temporarily unavailable")

    state = notifier.run_monitor(
        config,
        tmp_path / "state.json",
        probe=failing,
        sender=lambda c, m, e: sent.append(m) or {"queue_id": "alert"},
        sleep=lambda s: None,
        max_polls=6,
    )
    assert len(sent) == 1
    assert "这不是 checkpoint-ready 通知" in sent[0]
    assert "delivery" not in state


def test_terminal_training_before_checkpoint_alerts_without_stop(config, ready, tmp_path):
    ready["hosts"]["head"]["job"]["status"] = "FAILED"
    ready["hosts"]["actor"].pop("metadata")
    sent = []
    notifier.run_monitor(
        config,
        tmp_path / "state.json",
        probe=lambda c: ready,
        sender=lambda c, m, e: sent.append(m) or {"queue_id": "alert"},
        sleep=lambda s: None,
        max_polls=5,
    )
    assert len(sent) == 1 and "problem event_id=" in sent[0]


def test_published_incomplete_checkpoint_alerts_without_ready(config, ready, tmp_path):
    ready["hosts"]["head"]["shards"]["__1_0.distcp"] = 1
    sent = []
    notifier.run_monitor(
        config,
        tmp_path / "state.json",
        probe=lambda c: ready,
        sender=lambda c, m, e: sent.append(m) or {"queue_id": "alert"},
        sleep=lambda s: None,
        max_polls=5,
    )
    assert len(sent) == 1 and "problem event_id=" in sent[0]


def test_queue_retry_and_receipt(config, ready, tmp_path):
    calls = []

    def sender(c, m, e):
        calls.append(e)
        if len(calls) == 1:
            raise TimeoutError("queue unavailable")
        return {"queue_id": "event"}

    state = notifier.run_monitor(
        config, tmp_path / "state.json", probe=lambda c: ready, sender=sender, sleep=lambda s: None, max_polls=4
    )
    assert state["delivery"]["queue_id"] == "event"
    assert len(set(calls)) == 1


def test_cannot_reuse_another_config_state(config, tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"event_id": "another"}))
    with pytest.raises(ValueError, match="different"):
        notifier.run_monitor(config, state)


def test_queue_cli_targets_only_existing_thread(config, monkeypatch):
    seen = []

    def fake_run(argv, **kwargs):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, f"Queued message abc-123 for thread {config['thread_id']}.\n", "")

    monkeypatch.setattr(notifier.subprocess, "run", fake_run)
    result = notifier.send_queue(config, "event text", "marker")
    assert result["queue_id"] == "abc-123"
    assert seen == [["/bin/codex", "queue", "--thread", config["thread_id"], "--message", "event text"]]


def test_pending_event_deduplicates_without_cli_write(config, tmp_path, monkeypatch):
    path = tmp_path / "queue.sqlite"
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE queued_items(id TEXT, thread_id TEXT, payload_json TEXT)")
        c.execute("INSERT INTO queued_items VALUES(?,?,?)", ("existing-id", config["thread_id"], "message marker-123"))
    config["queue_database"] = str(path)
    monkeypatch.setattr(notifier.subprocess, "run", lambda *a, **k: pytest.fail("must not queue twice"))
    assert notifier.send_queue(config, "text", "marker-123")["queue_id"] == "existing-id"


def test_once_mode_never_queues_or_writes_monitor_state(config, ready, tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    state_dir = tmp_path / "not-created"
    monkeypatch.setattr(sys, "argv", ["notifier", "--config", str(path), "--once", "--state-dir", str(state_dir)])
    monkeypatch.setattr(notifier, "inspect", lambda c: ready)
    monkeypatch.setattr(notifier, "send_queue", lambda *a: pytest.fail("must not enqueue"))
    notifier.main()
    assert json.loads(capsys.readouterr().out)["assessment"]["ready"]
    assert not state_dir.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
