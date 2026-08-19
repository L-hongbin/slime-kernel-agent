#!/usr/bin/env python3
"""Driver for the V4 LoRA reload-crash repro (see lora_reload_repro.sh).

Waits for the sglang engine, then runs the exact slime weight-sync loop against
the HTTP endpoints: for each "RL step" it (unload previous) -> load a fresh
random adapter -> generate with lora_path. The first load+generate is expected to
pass; the crash reproduces on the first generate AFTER a reload. Reports the
iteration and step where the engine dies.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import requests
import torch


def _read_config(hf_ckpt: str) -> dict:
    with open(f"{hf_ckpt}/config.json") as f:
        return json.load(f)


def build_fake_adapter(
    cfg: dict,
    rank: int,
    b_scale: float = 0.0,
    *,
    shared_expert: bool = False,
) -> tuple[dict, dict]:
    """PEFT state-dict + config matching the V4 served leaf modules/shapes.

    Names mirror slime's converter output (base_model.model.layers.N.attn.*).
    Compressor kv/gate are kept separate (wkv/wgate); sglang fuses them to
    wkv_gate at load. dtype bf16 to match the LoRA buffer dtype.

    ``b_scale=0`` (default) makes lora_B all-zeros → the served model is
    EXACTLY the base model (delta = B@A = 0), so the LoRA kernels still run
    (rank>0) but produce no perturbation. This isolates a reload/cuda-graph
    kernel bug from value-induced garbage (random B perturbs attention →
    garbage MoE routing → DeepEP dispatch OOB, a first-load crash unrelated to
    the reload bug). Use a small b_scale (e.g. 1e-3) for a near-identity
    trained-like adapter.
    """
    hidden = cfg["hidden_size"]
    head_dim = cfg["head_dim"]
    n_heads = cfg["num_attention_heads"]
    q_lora_rank = cfg["q_lora_rank"]
    o_lora_rank = cfg["o_lora_rank"]
    o_groups = cfg["o_groups"]
    compress_ratios = cfg["compress_ratios"]
    n_layers = cfg["num_hidden_layers"]

    def AB(in_dim, out_dim):
        a = (torch.randn(rank, in_dim) * 0.02).to(torch.bfloat16)
        if b_scale == 0.0:
            b = torch.zeros(out_dim, rank, dtype=torch.bfloat16)
        else:
            b = (torch.randn(out_dim, rank) * b_scale).to(torch.bfloat16)
        return a, b

    sd: dict[str, torch.Tensor] = {}
    for L in range(n_layers):
        pre = f"base_model.model.layers.{L}.attn"
        # attention projections (present on every decoder layer)
        for leaf, (in_d, out_d) in {
            "wq_a": (hidden, q_lora_rank),
            "wq_b": (q_lora_rank, n_heads * head_dim),
            "wkv": (hidden, head_dim),
            "wo_b": (o_groups * o_lora_rank, hidden),
        }.items():
            a, b = AB(in_d, out_d)
            sd[f"{pre}.{leaf}.lora_A.weight"] = a
            sd[f"{pre}.{leaf}.lora_B.weight"] = b
        # compressor kv/gate only where a compressor exists (ratio != 0)
        ratio = compress_ratios[L] if L < len(compress_ratios) else 0
        if ratio != 0:
            coff = 2 if ratio == 4 else 1
            comp_out = coff * head_dim
            for leaf in ("wkv", "wgate"):
                a, b = AB(hidden, comp_out)
                sd[f"{pre}.compressor.{leaf}.lora_A.weight"] = a
                sd[f"{pre}.compressor.{leaf}.lora_B.weight"] = b

    target_modules = ["wq_a", "wq_b", "wkv", "wo_b", "wgate"]
    if shared_expert:
        # Shared-expert adapters, NATIVE leaf names exactly as slime's exporter
        # emits them (w1=gate, w3=up, w2=down): sglang renames + stacks w1/w3
        # into the tp1-replicated gate_up_proj and serves w2 as down_proj.
        moe_inter = cfg["moe_intermediate_size"] * cfg.get("n_shared_experts", 1)
        for L in range(n_layers):
            spre = f"base_model.model.layers.{L}.ffn.shared_experts"
            for leaf, (in_d, out_d) in {
                "w1": (hidden, moe_inter),
                "w3": (hidden, moe_inter),
                "w2": (moe_inter, hidden),
            }.items():
                a, b = AB(in_d, out_d)
                sd[f"{spre}.{leaf}.lora_A.weight"] = a
                sd[f"{spre}.{leaf}.lora_B.weight"] = b
        target_modules += ["w1", "w2", "w3"]

    config_dict = {
        "peft_type": "lora",
        "r": rank,
        "lora_alpha": 2 * rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": target_modules,
    }
    return sd, config_dict


def wait_healthy(base: str, timeout_s: int = 1800) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(f"{base}/health_generate", timeout=10)
            if r.status_code == 200:
                print(f"[driver] engine healthy after {time.time()-t0:.0f}s", flush=True)
                return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError("engine did not become healthy")


_ADAPTER_ROOT = "/dev/shm/slime_lora_repro"


def load_adapter(base: str, name: str, tensors: dict, config_dict: dict) -> tuple[requests.Response, str]:
    """Disk-path transport (matches slime's sglang_engine.load_lora_adapter_from_tensors):
    write the PEFT adapter to a local tmpfs dir and load it BY PATH via
    /load_lora_adapter, so every co-located DP worker reads the SAME persistent file.
    The old /load_lora_adapter_from_tensors IPC path serialized ONE handle that all N
    workers deserialized, and torch's file_system refcount removed the /dev/shm file
    when the first workers finished — before a slow rank opened it ("unable to open
    shared memory object ... No such file"), crashing the server. Returns (resp, dir);
    the caller frees the dir on unload/reuse."""
    import json
    import tempfile

    from safetensors.torch import save_file

    os.makedirs(_ADAPTER_ROOT, exist_ok=True)
    adapter_dir = tempfile.mkdtemp(prefix=f"{name}_", dir=_ADAPTER_ROOT)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in tensors.items()},
        os.path.join(adapter_dir, "adapter_model.safetensors"),
    )
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
        json.dump(config_dict, f)
    resp = requests.post(
        f"{base}/load_lora_adapter",
        json={"lora_name": name, "lora_path": adapter_dir},
        timeout=300,
    )
    return resp, adapter_dir


def _rmtree(adapter_dir: str | None) -> None:
    import shutil

    if adapter_dir:
        shutil.rmtree(adapter_dir, ignore_errors=True)


def unload_adapter(base: str, name: str) -> requests.Response:
    return requests.post(f"{base}/unload_lora_adapter", json={"lora_name": name}, timeout=120)


def _dump_cycle_state(step: int) -> None:
    """Per-cycle growth probe (the team-lead's instrumentation ask): tmpfs usage,
    leftover torch_* IPC files (should stay ~0 on the disk path), our adapter-dir
    count (should stay bounded ~2), and GPU0 memory."""
    import glob
    import shutil
    import subprocess

    torch_ipc = len(glob.glob("/dev/shm/torch_*"))
    adapter_dirs = len(glob.glob(os.path.join(_ADAPTER_ROOT, "*")))
    shm = shutil.disk_usage("/dev/shm")
    try:
        gpu = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            .stdout.strip()
            .splitlines()[0]
        )
    except Exception:
        gpu = "?"
    print(
        f"[driver] step {step} STATE: /dev/shm used={shm.used/2**30:.2f}G "
        f"torch_ipc_files={torch_ipc} adapter_dirs={adapter_dirs} gpu0_mem_used_MiB={gpu}",
        flush=True,
    )


def generate(base: str, name: str, n: int = 8, max_new: int = 32) -> requests.Response:
    # Distinct prompts -> a real decode batch (exercise the cuda-graph LoRA
    # decode path, not a single radix-collapsed sequence).
    prompts = [f"Question {i}: describe the number {i} in one sentence." for i in range(n)]
    payload = {
        "text": prompts,
        "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0},
        "lora_path": name,
    }
    return requests.post(f"{base}/generate", json=payload, timeout=600)


def sustained_generate(base: str, name: str, rounds: int, max_new: int) -> None:
    """Mirror the REAL rollout after a swap: sustained decode with the new adapter
    while the OLD adapter's slot has been freed. Fires many rounds with VARIED batch
    sizes (hit different captured cuda-graph decode variants), LONG outputs (thousands
    of graph replays), and SHARED prefixes across rounds (radix cache repopulation
    under the new adapter). This is the path the 12-cycle load->gen->unload repro
    never exercised (it generated only before the unload). Raises on any failure so
    the caller reports the crash."""
    import concurrent.futures as cf

    # A pool of shared long prefixes so later rounds get prefix cache hits.
    prefixes = [
        "In a detailed technical explanation, walk through step by step how "
        f"topic number {p} works, covering background, mechanism, and examples. " * 3
        for p in range(8)
    ]
    for r in range(rounds):
        bs = (1, 2, 4, 8, 16, 32)[r % 6]  # sweep batch sizes -> different graphs
        prompts = [prefixes[(r + i) % len(prefixes)] + f" Iteration {r}-{i}." for i in range(bs)]
        payload = {
            "text": prompts,
            "sampling_params": {"max_new_tokens": max_new, "temperature": 0.7, "top_p": 0.95},
            "lora_path": name,
        }
        # a few concurrent requests too, to vary the running-batch composition
        with cf.ThreadPoolExecutor(max_workers=3) as ex:
            futs = [ex.submit(requests.post, f"{base}/generate", json=payload, timeout=600) for _ in range(2)]
            for f in futs:
                resp = f.result()
                assert resp.status_code == 200, f"sustained round {r} bs={bs}: {resp.status_code} {resp.text[:200]}"
        if r % 10 == 0:
            print(f"[driver] sustained round {r}/{rounds} bs={bs} OK (adapter {name})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=31000)
    ap.add_argument("--hf-ckpt", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument(
        "--num-layers",
        type=int,
        default=0,
        help="Truncated-model override: build adapter weights for only the "
        "first N layers (must match the server's num_hidden_layers override). "
        "0 = use config.json's num_hidden_layers.",
    )
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--b-scale", type=float, default=0.0)
    ap.add_argument("--shared-expert", action="store_true")
    ap.add_argument(
        "--mode",
        choices=("same-slot", "alternating"),
        default="same-slot",
        help=(
            "same-slot: reproduce the crash — unload the constant-named adapter, "
            "then reload the SAME name into the same mem-pool slot (needs "
            "--max-loras-per-batch 1). alternating: the QeRL-style fix — LOAD a "
            "NEW alternating name (slime_lora_0/1) while the OLD one is still "
            "resident, generate, THEN unload the OLD name (needs "
            "--max-loras-per-batch >= 2)."
        ),
    )
    ap.add_argument(
        "--sustain-rounds",
        type=int,
        default=0,
        help="alternating mode: after each swap's unload, run this many rounds of "
        "sustained varied-batch/long/prefix-reuse generation with the NEW adapter "
        "(mirrors the real rollout that decodes for minutes after the old adapter's "
        "slot is freed). 0 = off (original short-generate-before-unload behaviour).",
    )
    ap.add_argument("--sustain-max-new", type=int, default=256)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    cfg = _read_config(args.hf_ckpt)
    if args.num_layers:
        cfg["num_hidden_layers"] = args.num_layers

    wait_healthy(base)

    if args.mode == "alternating":
        return _run_alternating(base, cfg, args)

    name = "slime_lora"
    prev_loaded = False
    prev_dir = None
    for i in range(args.iters):
        print(f"\n[driver] ===== RL step {i} =====", flush=True)
        try:
            if prev_loaded:
                # slime unloads the previous constant-named adapter, then reloads.
                r = unload_adapter(base, name)
                print(f"[driver] step {i} unload -> {r.status_code}", flush=True)
                assert r.status_code == 200, r.text
                _rmtree(prev_dir)

            tensors, config_dict = build_fake_adapter(
                cfg,
                args.rank,
                b_scale=args.b_scale,
                shared_expert=args.shared_expert,
            )
            r, prev_dir = load_adapter(base, name, tensors, config_dict)
            print(f"[driver] step {i} load -> {r.status_code}", flush=True)
            assert r.status_code == 200, r.text
            prev_loaded = True

            # THE forward-after-(re)load. Crash reproduces here for i>=1.
            r = generate(base, name)
            print(f"[driver] step {i} generate -> {r.status_code}", flush=True)
            assert r.status_code == 200, r.text
            out = r.json()
            txt = out[0]["text"] if isinstance(out, list) else out.get("text", "")
            print(f"[driver] step {i} OK, sample completion: {txt[:60]!r}", flush=True)

            # liveness probe: an illegal access surfaces here even if /generate
            # returned before the async fault propagated.
            wait_healthy(base, timeout_s=60)
            print(f"[driver] step {i} engine still healthy", flush=True)
        except Exception as e:
            print(f"[driver] step {i} FAILED: {type(e).__name__}: {e}", flush=True)
            print(
                f"[driver] ===> reload crash reproduced at RL step {i} " f"({'FIRST-LOAD' if i == 0 else 'RELOAD'})",
                flush=True,
            )
            return 1

    print(f"\n[driver] ALL {args.iters} load/reload cycles PASSED — crash NOT reproduced.", flush=True)
    return 0


def _run_alternating(base: str, cfg: dict, args) -> int:
    """The QeRL-style fix: LOAD the new alternating name BEFORE unloading the old
    one (double buffer). Mirrors slime's ``plan_lora_swap`` exactly. Each step's
    /generate uses the just-loaded name; the old adapter is unloaded only after the
    forward succeeds. Passes where same-slot crashes."""

    def alt_name(version: int) -> str:
        return f"slime_lora_{version % 2}"

    prev_name = None
    dirs: dict[str, str] = {}  # name -> on-disk dir (freed on unload / reuse)
    for i in range(args.iters):
        version = i + 1  # mirror the updater's weight_version (1-based)
        new_name = alt_name(version)
        print(f"\n[driver] ===== RL step {i} (alternating, new_name={new_name}) =====", flush=True)
        try:
            # Reusing an alternating name: drop its stale dir (from 2 cycles ago).
            _rmtree(dirs.pop(new_name, None))
            # LOAD new first (old still resident + cuda-graph-referenced).
            tensors, config_dict = build_fake_adapter(
                cfg,
                args.rank,
                b_scale=args.b_scale,
                shared_expert=args.shared_expert,
            )
            r, dirs[new_name] = load_adapter(base, new_name, tensors, config_dict)
            print(f"[driver] step {i} load {new_name} -> {r.status_code}", flush=True)
            assert r.status_code == 200, r.text

            # THE forward-after-load, now against the freshly-loaded NEW slot.
            r = generate(base, new_name)
            print(f"[driver] step {i} generate {new_name} -> {r.status_code}", flush=True)
            assert r.status_code == 200, r.text
            out = r.json()
            txt = out[0]["text"] if isinstance(out, list) else out.get("text", "")
            print(f"[driver] step {i} OK, sample completion: {txt[:60]!r}", flush=True)

            import os as _os

            _sustain_first = _os.environ.get("SUSTAIN_BEFORE_UNLOAD", "0") == "1"
            if _sustain_first and getattr(args, "sustain_rounds", 0) > 0:
                # DISCRIMINATOR: sustained decode with BOTH adapters resident
                # (before the unload). Passing here while post-unload crashes
                # pins the trigger on the unload/freed-slot serving state.
                print(
                    f"[driver] step {i} sustained-generate {args.sustain_rounds} rounds "
                    f"(PRE-unload, adapter {new_name})",
                    flush=True,
                )
                sustained_generate(base, new_name, args.sustain_rounds, args.sustain_max_new)

            # UNLOAD old only after the new one is live and served.
            if prev_name is not None and prev_name != new_name:
                r = unload_adapter(base, prev_name)
                print(f"[driver] step {i} unload {prev_name} -> {r.status_code}", flush=True)
                assert r.status_code == 200, r.text
                _rmtree(dirs.pop(prev_name, None))
            prev_name = new_name

            # Mirror the real rollout: SUSTAINED decode with the new adapter AFTER
            # the old adapter's slot was freed (the path smoke13 crashed on ~17 min
            # in; the 12-cycle short-generate-before-unload repro never hit it).
            if not _sustain_first and getattr(args, "sustain_rounds", 0) > 0:
                print(
                    f"[driver] step {i} sustained-generate {args.sustain_rounds} rounds "
                    f"(post-unload, adapter {new_name})",
                    flush=True,
                )
                sustained_generate(base, new_name, args.sustain_rounds, args.sustain_max_new)

            wait_healthy(base, timeout_s=60)
            _dump_cycle_state(i)
            print(f"[driver] step {i} engine still healthy", flush=True)
        except Exception as e:
            print(f"[driver] step {i} FAILED: {type(e).__name__}: {e}", flush=True)
            print(f"[driver] ===> alternating swap crashed at RL step {i}", flush=True)
            return 1

    print(f"\n[driver] ALL {args.iters} alternating swap cycles PASSED.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
