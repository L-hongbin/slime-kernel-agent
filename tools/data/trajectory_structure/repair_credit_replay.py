"""Execute explicitly planned repair counterfactuals against an existing KernelGym.

One task at a time; no deployment, model generation, reward changes or automatic
payload retry. A plan names saved audit cases and exact scoped source edits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def build_variant(case, variant):
    turn = next(t for t in case["structure"]["turns"] if t["turn_idx"] == variant["base_turn"])
    sections = {name: entry["source"] for name, entry in turn["sections"].items()}
    if set(sections) != {"CUDA_KERNELS", "APPLY_BINDINGS", "MODEL_NEW"}:
        raise ValueError("counterfactual needs the complete submitted program")
    for edit in variant.get("edits", []):
        section = edit["section"]
        if not edit["old"] or sections[section].count(edit["old"]) != edit.get("expected_occurrences", 1):
            raise ValueError(f"source edit is missing/ambiguous: {section} {edit['old'][:80]}")
        sections[section] = sections[section].replace(edit["old"], edit["new"])
    code = "\n\n".join(
        f"### {name}\n```{lang}\n{sections[name]}\n```"
        for name, lang in [("CUDA_KERNELS", "cpp"), ("APPLY_BINDINGS", "cpp"), ("MODEL_NEW", "python")]
    )
    return code, sections


def compact(result):
    state = result.get("result") or result
    state = state.get("env_state") or state
    return {k: state.get(k) for k in ["status", "compiled", "correctness", "decoy_kernel", "error", "error_message"]}


def planned_precheck(plan, name, code):
    records = plan.get("client_precheck_results")
    if records is None:
        return None
    record = records[name]
    if record["candidate_sha256"] != sha(code) or not isinstance(record["passed"], bool):
        raise ValueError("client precheck does not match this candidate")
    if not record["passed"] and not isinstance(record.get("state"), dict):
        raise ValueError("failed client precheck has no saved feedback")
    return record


def task_evaluation(plan, task, case=None):
    result = dict(plan["evaluation"])
    recorded = (case or {}).get("evaluation_overrides", {})
    result.update(recorded)
    overrides = task.get("evaluation_overrides", {})
    if set(overrides) - {"precision", "entry_point"}:
        raise ValueError("only explicit task precision and entry point overrides are supported")
    if any(k in recorded and recorded[k] != v for k, v in overrides.items()):
        raise ValueError("task overrides differ from recorded training protocol")
    result.update(overrides)
    if result.get("precision") not in {"fp32", "fp16", "bf16"}:
        raise ValueError("invalid per-task precision")
    return result


def run(args):
    plan = json.loads(args.plan.read_text())
    for source in plan.get("client_precheck_source", {}).values():
        if sha(Path(source["path"]).read_text()) != source["sha256"]:
            raise ValueError("archived client precheck source has changed")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, value=None):
        data = json.dumps(value).encode() if value is not None else None
        req = urllib.request.Request(
            args.url.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"}
        )
        with opener.open(req, timeout=60) as response:
            return json.load(response)

    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    package = {
        "plan": plan,
        "plan_sha256": sha(args.plan.read_text()),
        "runner_sha256": sha(Path(__file__).read_text()),
        "server_url": args.url,
        "submitted": args.submit,
        "requests": [],
    }
    if args.submit:
        health = request("/health")
        if health.get("status") != "healthy":
            raise RuntimeError("KernelGym is not healthy; no requests submitted")
        save("server_health_before.json", health)
        schema = request("/openapi.json")
        save("server_api_schema.json", schema)
    receipts = []
    for task in plan["tasks"]:
        case_path = (args.plan.parent / task["case_path"]).resolve()
        case = json.loads(case_path.read_text())
        assert case["reference_code"], case_path
        code, sections = build_variant(case, task)
        precheck = planned_precheck(plan, task["name"], code)
        task_id = "repairdiff_" + task["name"] + "_" + uuid.uuid4().hex[:8]
        payload = {
            **task_evaluation(plan, task, case),
            "task_id": task_id,
            "reference_code": case["reference_code"],
            "kernel_code": code,
            "uuid": "repairdiff_" + sha(case["reference_code"])[:24],
        }
        case_sha = sha(case_path.read_text())
        rec = {
            "name": task["name"],
            "task_id": task_id,
            "case_path": str(case_path),
            "case_sha256": case_sha,
            "candidate_sha256": sha(code),
            "reference_sha256": sha(case["reference_code"]),
            "interpretation": task.get("interpretation"),
            "base_turn": task["base_turn"],
        }
        package["requests"].append(rec)
        save("request_" + task["name"] + ".json", payload)
        save("manifest.json", package)
        if precheck is not None:
            save("client_precheck_" + task["name"] + ".json", precheck)
        if not args.submit:
            continue
        if precheck is not None and not precheck["passed"]:
            result = {"env_state": precheck["state"], "origin": "archived_client_precheck"}
            save("result_" + task["name"] + ".json", result)
            rec = {
                **rec,
                "submitted_to_server": False,
                "result": compact(result),
                "task_status": {"status": "client_rejected"},
            }
            receipts.append(rec)
            save("receipts.json", receipts)
            print(json.dumps({"name": task["name"], "status": "client_rejected"}), flush=True)
            if task.get("require_correct"):
                raise RuntimeError("Saved correct control failed archived client precheck")
            continue
        started = time.monotonic()
        # A transport failure after submission is intentionally NOT retried:
        # keep the persisted task ID for manual reconciliation.
        submitted = request("/evaluate", payload)
        save("submission_" + task["name"] + ".json", submitted)
        assert submitted.get("task_id", task_id) == task_id, submitted
        while True:
            status = request("/status/" + task_id)
            if status.get("status") in {"completed", "failed", "timeout", "cancelled"}:
                break
            if time.monotonic() - started > args.wait_timeout:
                save("unresolved_" + task["name"] + ".json", status)
                raise TimeoutError(f"Task outcome unknown; reconcile {task_id}, do not resubmit")
            time.sleep(1)
        result = request("/results/" + task_id)
        save("result_" + task["name"] + ".json", result)
        rec = {
            **rec,
            "submitted_to_server": True,
            "elapsed_seconds": time.monotonic() - started,
            "task_status": status,
            "result": compact(result),
        }
        receipts.append(rec)
        save("receipts.json", receipts)
        print(
            json.dumps({k: v for k, v in rec.items() if k not in {"task_status", "case_path"}}, ensure_ascii=False),
            flush=True,
        )
        if task.get("require_correct") and (rec["result"]["correctness"] is not True or rec["result"]["decoy_kernel"]):
            raise RuntimeError("Saved correct control did not reproduce; stop causal interpretation: " + task_id)
    package["completed"] = True
    save("manifest.json", package)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:20211")
    p.add_argument("--wait-timeout", type=float, default=900)
    p.add_argument("--submit", action="store_true")
    run(p.parse_args())
