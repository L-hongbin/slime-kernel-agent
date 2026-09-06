#!/usr/bin/env python3
"""Install/check the audited Qwen GDN + MTP fixes in an isolated SGLang copy.

Upstream: sgl-project/sglang#35985 (FA3 context headroom), #35821 (empty
Mamba radix entries and verification tracking). The latter is adapted to the
pinned 0.5.16 allocator's ``free`` API. Unknown source layouts fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

REPLACEMENTS = {
    "srt/managers/scheduler_components/weight_updater.py": (
        """            success, message = self.tp_worker.update_weights_from_distributed(recv_req)
""",
        """            # Receive once on the target's NCCL group, then dispatch native
            # Qwen MTP tensors to the draft loader instead of dropping them.
            target = self.tp_worker.model_runner.model
            draft_runner = _get_draft_model_runner(self.draft_worker)
            original_load = target.load_weights
            def load_target_and_mtp(weights):
                weights = list(weights)
                mtp_weights = [(n, w) for n, w in weights if n.startswith("mtp.")]
                if not mtp_weights or self.draft_worker is None:
                    return original_load(weights)
                if draft_runner is None or type(draft_runner.model).__name__ != "Qwen3_5ForCausalLMMTP":
                    raise RuntimeError("Native MTP weight update requires a Qwen3_5ForCausalLMMTP draft runner")
                result = original_load([(n, w) for n, w in weights if not n.startswith("mtp.")])
                draft_runner.model.load_weights(mtp_weights)
                logger.info("Updated %d native MTP tensors from actor", len(mtp_weights))
                return result
            target.load_weights = load_target_and_mtp
            try:
                success, message = self.tp_worker.update_weights_from_distributed(recv_req)
            finally:
                target.load_weights = original_load
""",
    ),
    "srt/layers/attention/flashattention_backend.py": (
        "        self.speculative_step_id = speculative_step_id\n",
        """        # Spec draft_extend/verify metadata includes draft tokens past the
        # logical context wall; CUDA-graph page tables need the same headroom.
        if self.speculative_num_draft_tokens:
            self.max_context_len = int(model_runner.model_config.context_len) + int(
                self.speculative_num_draft_tokens
            )
            self.max_num_pages = (
                self.max_context_len + self.page_size - 1
            ) // self.page_size

        self.speculative_step_id = speculative_step_id
""",
    ),
    "srt/mem_cache/mamba_radix_cache.py": (
        """            if cache_len is None:
                cache_len = 0
""",
        """            if cache_len is None or cache_len == 0:
                # An empty radix entry must never become a stale Mamba COW
                # source. Only release the KV tail owned by this request.
                self.token_to_kv_pool_allocator.free(
                    kv_indices[req.cache_protected_len :]
                )
                self.req_to_token_pool.free_mamba_cache(req)
                self.dec_lock_ref(req.last_node)
                return
""",
    ),
    "srt/speculative/spec_utils.py": (
        """            to_track_ith = torch.clamp(
                tracking_point - seq_lens_pre_verify - 1, min=0
            ).to(torch.int64)
""",
        """            to_track_ith = torch.clamp(
                torch.minimum(tracking_point - seq_lens_pre_verify, accept_lens - 1),
                min=0,
            ).to(torch.int64)
""",
    ),
}


def patch_source(source: str, relative_path: str, *, check_only: bool = False) -> str:
    old, new = REPLACEMENTS[relative_path]
    if source.count(new) == 1 and source.replace(new, "").count(old) == 0:
        return source
    if source.count(new) != 0 or source.count(old) != 1:
        raise RuntimeError(f"Unknown or partially patched SGLang layout: {relative_path}")
    if check_only:
        raise RuntimeError(f"Qwen MTP runtime patch missing: {relative_path}")
    return source.replace(old, new)


def patch_package(package: Path, *, check_only: bool = False) -> dict:
    changes = {}
    for relative_path in REPLACEMENTS:
        path = package / relative_path
        before = path.read_text()
        after = patch_source(before, relative_path, check_only=check_only)
        changes[path] = (before, after)
    manifest = {}
    for path, (before, after) in changes.items():
        if before != after:
            path.write_text(after)
        manifest[str(path.relative_to(package))] = {
            "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
        }
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument(
        "--overlay-dir", type=Path, help="Copy installed SGLang here before patching; add this directory to PYTHONPATH"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.package_root and args.overlay_dir:
        parser.error("Choose --package-root or --overlay-dir")
    package = args.package_root
    if package is None:
        package = Path(importlib.util.find_spec("sglang").origin).parent
    if args.overlay_dir:
        dest = args.overlay_dir / "sglang"
        if not dest.exists():
            if args.check:
                parser.error("Overlay does not exist")
            shutil.copytree(package, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        package = dest
    manifest = patch_package(package, check_only=args.check)
    print(json.dumps({"package": str(package), "patches": manifest}, indent=2))


if __name__ == "__main__":
    main()
