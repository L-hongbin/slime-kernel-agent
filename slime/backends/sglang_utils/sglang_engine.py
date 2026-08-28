import dataclasses
import ipaddress
import logging
import multiprocessing
import os
import time

import requests
import sglang_router
from packaging.version import parse
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from urllib3.exceptions import NewConnectionError

from slime.backends.sglang_utils.external import get_server_info
from slime.ray.ray_actor import RayActor
from slime.utils import accelerator
from slime.utils.http_utils import get_host_info

logger = logging.getLogger(__name__)
DEFAULT_ROUTER_REGISTRATION_TIMEOUT_SECS = 300.0


def get_base_gpu_id(args, rank):
    num_gpus = min(args.num_gpus_per_node, args.rollout_num_gpus_per_engine)
    if args.colocate:
        start_index = (rank * num_gpus) % args.num_gpus_per_node
    else:
        num_actor_gpus = 0 if args.debug_rollout_only else args.actor_num_gpus_per_node * args.actor_num_nodes
        start_index = (num_actor_gpus + rank * num_gpus) % args.num_gpus_per_node
    return start_index


def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
    # Expandable segments help the colocated training actor tolerate repeated
    # cache releases, but SGLang's allocator/sleep path does not support them.
    # The rollout Ray actor inherits the job environment, so remove the option
    # before spawning every SGLang server and its children.
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.environ.pop("PYTORCH_ALLOC_CONF", None)

    if getattr(server_args, "encoder_only", False):
        from sglang.srt.disaggregation.encode_server import launch_server_process as sglang_launch_server_process

        return sglang_launch_server_process(
            server_args,
            start_method="spawn",
            wait_for_server=True,
        )

    from sglang.srt.entrypoints.http_server import launch_server

    multiprocessing.set_start_method("spawn", force=True)
    server_args.host = server_args.host.strip("[]")
    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    p.start()

    if getattr(server_args, "node_rank", 0) != 0:
        return p

    _wait_server_healthy(
        base_url=server_args.url(),
        api_key=server_args.api_key,
        is_process_alive=lambda: p.is_alive(),
    )

    return p


def _wait_server_healthy(base_url, api_key, is_process_alive):
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }

    with requests.Session() as session:
        while True:
            try:
                response = session.get(f"{base_url}/health_generate", headers=headers)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)


class SGLangEngine(RayActor):
    def __init__(
        self,
        args,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int | None = None,
        sglang_overrides: dict | None = None,
        num_gpus_per_engine: int | None = None,
    ):
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.sglang_overrides = sglang_overrides or {}
        self.num_gpus_per_engine = num_gpus_per_engine
        # Name of the LoRA adapter this engine currently serves (alternating sync
        # path). Set on a successful ``load_lora_adapter_from_tensors``; the rollout
        # queries it each step so ``/generate`` ``lora_path`` tracks the live name.
        self._active_lora_name: str | None = None
        # lora_name -> on-disk temp adapter dir (disk-path transport). Kept alive
        # while the adapter is loaded (so sglang's implicit reload can re-read it),
        # deleted when the adapter is unloaded or its name is reused. See
        # ``load_lora_adapter_from_tensors``.
        self._lora_dirs: dict[str, str] = {}

    def init(
        self,
        dist_init_addr,
        port,
        nccl_port,
        host=None,
        disaggregation_bootstrap_port=None,
        router_ip=None,
        router_port=None,
    ):
        self.router_ip = router_ip if router_ip is not None else self.args.sglang_router_ip
        self.router_port = router_port if router_port is not None else self.args.sglang_router_port

        host = host or get_host_info()[1]

        def _format_v6_uri(addr):
            if not addr or addr.startswith("["):
                return addr
            try:
                if ipaddress.ip_address(addr).version == 6:
                    return f"[{addr}]"
            except ValueError:
                pass
            return addr

        host = _format_v6_uri(host)
        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        dist_init_addr = f"{_format_v6_uri(ip_part)}:{port_part}"

        server_args_dict, external_engine_need_check_fields = _compute_server_args(
            self.args,
            self.rank,
            dist_init_addr,
            nccl_port,
            host,
            port,
            self.worker_type,
            disaggregation_bootstrap_port,
            base_gpu_id=self.base_gpu_id,
            sglang_overrides=self.sglang_overrides,
            num_gpus_per_engine=self.num_gpus_per_engine,
        )

        self.node_rank = server_args_dict["node_rank"]
        self.server_host = server_args_dict["host"]  # with [] if ipv6
        self.server_port = server_args_dict["port"]
        logger.warning(
            "SGLangEngine launch rank=%s worker_type=%s base_gpu_id=%s "
            "tp_size=%s CUDA_VISIBLE_DEVICES=%s host=%s port=%s",
            self.rank,
            self.worker_type,
            server_args_dict.get("base_gpu_id"),
            server_args_dict.get("tp_size"),
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            self.server_host,
            self.server_port,
        )

        if self.args.rollout_external:
            self._init_external(server_args_dict, external_engine_need_check_fields=external_engine_need_check_fields)
        else:
            self._init_normal(server_args_dict)

    def _init_external(self, expect_server_args, external_engine_need_check_fields):
        logger.info(f"Use external SGLang engine (rank={self.rank}, expect_server_args={expect_server_args})")

        def _sanity_check_server_args(actual_server_args, expect_server_args):
            for name in external_engine_need_check_fields:
                expect_value = expect_server_args.get(name)
                actual_value = actual_server_args.get(name)
                assert (
                    actual_value == expect_value
                ), f"{name=} {expect_value=} {actual_value=} {expect_server_args=} {actual_server_args=}"

        actual_server_args = get_server_info(f"http://{self.server_host}:{self.server_port}")
        _sanity_check_server_args(actual_server_args, expect_server_args)
        self._register_to_router(expect_server_args)

    def _init_normal(self, server_args_dict):
        logger.info(f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}")
        self.process = launch_server_process(ServerArgs(**server_args_dict))
        self._register_to_router(server_args_dict)

    def _register_to_router(self, server_args_dict):
        if self.worker_type == "encoder":
            return

        if self.node_rank == 0 and self.router_ip and self.router_port:
            worker_url = f"http://{self.server_host}:{self.server_port}"
            registration_timeout = float(
                os.environ.get("SGLANG_ROUTER_REGISTRATION_TIMEOUT_SECS", DEFAULT_ROUTER_REGISTRATION_TIMEOUT_SECS)
            )
            if parse(sglang_router.__version__) <= parse("0.2.1"):
                registration_url = f"http://{self.router_ip}:{self.router_port}/add_worker?url={worker_url}"
                logger.info(
                    "Register SGLang worker with router: url=%s worker_type=%s timeout=%.1fs",
                    registration_url,
                    self.worker_type,
                    registration_timeout,
                )
                assert self.worker_type == "regular", "pd disaggregation is not supported in old router."
                response = requests.post(registration_url, timeout=registration_timeout)
            else:
                payload = {
                    "url": worker_url,
                    "worker_type": self.worker_type,
                }
                if self.worker_type == "prefill":
                    bootstrap_port = server_args_dict.get("disaggregation_bootstrap_port")
                    if bootstrap_port is None:
                        raise RuntimeError(
                            f"Prefill worker {worker_url} does not have disaggregation_bootstrap_port; "
                            "cannot register it to the PD router."
                        )
                    payload["bootstrap_port"] = bootstrap_port
                registration_url = f"http://{self.router_ip}:{self.router_port}/workers"
                logger.info(
                    "Register SGLang worker with router: url=%s payload=%s timeout=%.1fs",
                    registration_url,
                    payload,
                    registration_timeout,
                )
                response = requests.post(
                    registration_url,
                    json=payload,
                    timeout=registration_timeout,
                )
            response.raise_for_status()
            logger.info(
                "Registered SGLang worker with router: worker_url=%s worker_type=%s status=%s",
                worker_url,
                self.worker_type,
                response.status_code,
            )

    def _make_request(self, endpoint: str, payload: dict | None = None, timeout: float = 600.0):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        if self.node_rank != 0:
            return

        url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
        # Never wait forever on an engine control endpoint: /unload_lora_adapter
        # blocks in sglang's lora_registry.wait_for_unload until the adapter's
        # request refs drain, which cannot happen while generation is paused —
        # rank0 hung ~50 min here and collapsed the weight-sync gloo barrier
        # (formal r7f/r7g, 2026-07-10; codex-diagnosed).
        response = requests.post(url, json=payload or {}, timeout=timeout)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        return response.json()

    def health_generate(self, timeout: float = 5.0) -> bool:
        """Run /health_generate on the underlying SGLang HTTP server.

        Args:
            timeout: Timeout for the health request in seconds.

        Returns:
            True if the server responds with HTTP 200.

        Raises:
            requests.RequestException: If the request fails for any reason, including timeout.
        """
        if self.node_rank != 0:
            return True

        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/health_generate",
            timeout=timeout,
        )
        response.raise_for_status()
        return True

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """
        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_tensor",
            payload,
        )

    def _lora_engine_is_single_node(self) -> bool:
        """True when this rollout engine occupies a single node (all its LoRA
        workers are co-located with this node-0 actor)."""
        gpus_per_engine = self.num_gpus_per_engine or self.args.rollout_num_gpus_per_engine
        return max(1, gpus_per_engine // self.args.num_gpus_per_node) == 1

    def _write_lora_adapter_dir(self, lora_name: str, tensors: dict, config_dict: dict) -> str:
        """Materialize a PEFT adapter (``adapter_model.safetensors`` +
        ``adapter_config.json``) into a fresh temp dir on local tmpfs and return it.

        sglang's disk loader reads ``target_modules``/``r``/``lora_alpha`` from the
        config and globs ``*.safetensors``; both the disk and from-tensors paths run
        the same ``_normalize_weights`` (incl. the V4 ``wkv_gate`` fusion), so the
        served result is identical.
        """
        import json
        import os
        import tempfile

        from safetensors.torch import save_file

        root = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
        base = os.path.join(root, "slime_lora_adapters") if root else tempfile.gettempdir()
        os.makedirs(base, exist_ok=True)
        adapter_dir = tempfile.mkdtemp(prefix=f"{lora_name}_", dir=base)
        # safetensors needs contiguous, non-aliased CPU tensors.
        state = {name: t.detach().to("cpu").contiguous() for name, t in tensors.items()}
        save_file(state, os.path.join(adapter_dir, "adapter_model.safetensors"))
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
            json.dump(config_dict, f)
        return adapter_dir

    def _discard_lora_dir(self, lora_name: str) -> None:
        import shutil

        adapter_dir = self._lora_dirs.pop(lora_name, None)
        if adapter_dir is not None:
            shutil.rmtree(adapter_dir, ignore_errors=True)

    def load_lora_adapter_from_tensors(
        self,
        lora_name: str,
        tensors: dict,
        config_dict: dict,
        weight_version: str | None = None,
        load_format: str | None = None,
    ):
        """Hot-load a LoRA adapter (base + adapter served) via a DISK-PATH transport.

        ``tensors`` is a PEFT-named ``{name: cpu_tensor}`` state dict (produced by
        the LoRA-adapter weight-sync path). We do NOT ship it as one
        ``MultiprocessingSerializer`` IPC handle: sglang broadcasts that single
        serialized string to every co-located DP worker, and torch's ``file_system``
        sharing reference-counts the backing ``/dev/shm`` file across the N
        independent deserializers — the first workers to finish drop the refcount to
        zero and remove the file before a slower rank opens it, crashing that
        scheduler with ``unable to open shared memory object … No such file`` (which
        cascades over gloo and kills the server; caught on the 8-GPU alternating
        smoke at the 6th load). Instead we materialize the adapter to a temp dir on
        local tmpfs and load it by path: every worker reads the SAME persistent file
        we own (no refcount race). ``/load_lora_adapter`` is synchronous over all DP
        replies, so by the time it returns every worker has read the weights into CPU
        tensors; the dir is retained (not deleted here) so sglang's implicit reload
        from ``lora_ref_cache`` can re-read it, and is freed on unload / name reuse.

        Also records ``weight_version`` so the ``--ci-test`` version-equality check
        passes even though a LoRA load does not touch the base weights.
        """
        if self.node_rank != 0:
            return

        # Local tmpfs is only visible to workers on THIS node. Multi-node engines
        # place workers on other nodes that cannot read a node-0 path — fail loudly
        # rather than silently serve a half-loaded adapter (multi-node needs a
        # shared-FS materialization, not yet implemented).
        if not self._lora_engine_is_single_node():
            raise RuntimeError(
                "LoRA disk-path adapter transport requires single-node rollout engines "
                f"(this engine spans {max(1, (self.num_gpus_per_engine or self.args.rollout_num_gpus_per_engine) // self.args.num_gpus_per_node)} nodes); "
                "multi-node LoRA serving needs a shared-filesystem adapter path (TODO)."
            )

        # Reusing an alternating name: its previous dir (from 2 syncs ago, long
        # since fully read) is safe to drop now.
        self._discard_lora_dir(lora_name)
        adapter_dir = self._write_lora_adapter_dir(lora_name, tensors, config_dict)
        try:
            result = self._make_request(
                "load_lora_adapter",
                {"lora_name": lora_name, "lora_path": adapter_dir},
            )
        except Exception:
            # The request raising (connection error / HTTP error) would leak the
            # freshly written adapter dir on tmpfs (codex review 2026-07-09).
            import shutil

            shutil.rmtree(adapter_dir, ignore_errors=True)
            raise
        # sglang can return HTTP 200 with success=False; only record the live name /
        # bump the version / retain the dir on an actual success (a premature bump
        # would let --ci-test pass — or the rollout route lora_path — while the
        # engine serves the old/missing adapter).
        load_ok = result is None or result.get("success", True)
        if load_ok:
            self._lora_dirs[lora_name] = adapter_dir
            self._active_lora_name = lora_name
            if weight_version is not None:
                self._make_request("update_weight_version", {"new_version": str(weight_version)})
        else:
            import shutil

            shutil.rmtree(adapter_dir, ignore_errors=True)
        return result

    def unload_lora_adapter(self, lora_name: str, timeout: float = 120.0):
        """Unload a previously loaded LoRA adapter by name (paired with the
        alternating load in the per-iteration adapter sync).

        Bounded + best-effort: sglang waits for the adapter's request refs to
        drain before unloading, so this MUST be called with generation resumed
        (the updater defers it until after continue_generation) and may still
        time out if refs leaked — the caller retries at the next swap."""
        if self.node_rank != 0:
            return
        try:
            result = self._make_request("unload_lora_adapter", {"lora_name": lora_name}, timeout=timeout)
        except requests.exceptions.Timeout:
            logger.warning(
                "unload_lora_adapter(%s) timed out after %ss; adapter left resident "
                "(will be retried at the next swap)",
                lora_name,
                timeout,
            )
            return None
        # The adapter is now unregistered; free its on-disk temp dir (no implicit
        # reload can reference it anymore).
        self._discard_lora_dir(lora_name)
        # If we just unloaded the adapter the rollout believes is active, clear the
        # cached name so a stale name is never reported by get_active_lora_name.
        # (In the normal alternating swap we unload the OLD name while _active is
        # the just-loaded NEW name, so this is a defensive no-op there.)
        if lora_name == self._active_lora_name:
            self._active_lora_name = None
        return result

    def get_active_lora_name(self) -> str | None:
        """Name of the LoRA adapter this engine currently serves, or ``None`` before
        any adapter has been loaded. Queried by the rollout each step so ``/generate``
        ``lora_path`` tracks the live (alternating) adapter name."""
        return self._active_lora_name

    def flush_cache(self):
        """Flush the cache of the server."""
        if self.node_rank != 0:
            return
        # flush cache will not return status_code 200 when there are pending requests
        for _ in range(60):
            try:
                response = requests.get(f"http://{self.server_host}:{self.server_port}/flush_cache")
                if response.status_code == 200:
                    break
                logger.info(f"Error flushing cache: HTTP {response.status_code} {response.text!r}")
                time.sleep(1)
            except NewConnectionError as e:
                raise e
            except Exception as e:
                logger.info(f"Error flushing cache: {e}")
                time.sleep(1)
                continue
        else:
            raise TimeoutError("Timeout while flushing cache.")

    def get_url(self):
        if self.node_rank != 0:
            return None
        return f"http://{self.server_host}:{self.server_port}"

    def shutdown(self):
        # Free any retained LoRA adapter temp dirs (disk-path transport).
        import shutil

        for adapter_dir in self._lora_dirs.values():
            shutil.rmtree(adapter_dir, ignore_errors=True)
        self._lora_dirs.clear()

        if self.args.rollout_external:
            return

        logger.info(f"Shutdown engine {self.server_host}:{self.server_port}...")
        if self.worker_type != "encoder" and self.node_rank == 0:
            worker_url = f"http://{self.server_host}:{self.server_port}"
            response = None
            try:
                all_workers = requests.get(f"http://{self.router_ip}:{self.router_port}/workers").json()["workers"]
                for worker in all_workers:
                    if worker["url"] == worker_url:
                        worker_id = worker["id"]
                        response = requests.delete(f"http://{self.router_ip}:{self.router_port}/workers/{worker_id}")
                        break
                else:
                    logger.warning(f"Worker {worker_url} not found in router during shutdown.")
            except Exception as e:
                logger.warning(f"Failed to fetch workers list or remove worker: {e}")

            if response is not None:
                response.raise_for_status()
        kill_process_tree(self.process.pid)

    def get_weight_version(self):
        if self.node_rank != 0:
            return
        url = f"http://{self.server_host}:{self.server_port}/get_weight_version"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()["weight_version"]

    def release_memory_occupation(self):
        self.flush_cache()
        return self._make_request("release_memory_occupation")

    def resume_memory_occupation(self, tags: list[str] = None):
        """
        Available tags for multi-stage resume: weights, kv_cache
        """
        return self._make_request(
            "resume_memory_occupation",
            {"tags": tags},
        )

    def check_weights(self, action: str):
        return self._make_request("weights_checker", {"action": action})

    def pull_weights(self, target_version: int):
        """Have the engine sync every host it spans to target_version: each host pulls the
        published weights (a full checkpoint copied as-is, or deltas verified per-tensor and
        applied onto the local checkpoint) into its local checkpoint dir. The engine reloads
        it afterwards via update_weights_from_disk."""
        return self._make_request(
            "pull_weights",
            {
                "local_checkpoint_dir": self.args.update_weight_local_checkpoint_dir,
                "source_dir": self.args.update_weight_disk_dir,
                "target_version": target_version,
            },
        )

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str | None = None,
        weight_version: str | None = None,
    ):
        """Reload weights from the checkpoint at *model_path* without restarting the engine."""
        payload: dict = {"model_path": model_path}
        if load_format is not None:
            payload["load_format"] = load_format
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request("update_weights_from_disk", payload)

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def destroy_weights_update_group(self, group_name):
        try:
            return self._make_request(
                "destroy_weights_update_group",
                {
                    "group_name": group_name,
                },
            )
        except requests.exceptions.RequestException:
            # catch the case there the engine is just created and does not have the group.
            pass

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        flush_cache=False,
        weight_version: str | None = None,
        load_format: str | None = None,
    ):
        payload = {
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        if load_format is not None:
            payload["load_format"] = load_format
        return self._make_request(
            "update_weights_from_distributed",
            payload,
        )

    def pause_generation(self):
        if self.node_rank != 0:
            return
        response = requests.post(f"http://{self.server_host}:{self.server_port}/pause_generation", json={})
        response.raise_for_status()
        return response

    def continue_generation(self):
        if self.node_rank != 0:
            return
        response = requests.post(f"http://{self.server_host}:{self.server_port}/continue_generation", json={})
        response.raise_for_status()
        return response

    def post_process_weights(
        self,
        restore_weights_before_load: bool = False,
        post_process_quantization: bool = False,
    ):
        """
        Run post-load weight processing on the SGLang server.

        This is used for restore-before-load and post-load quantization hooks.
        """

        return self._make_request(
            "post_process_weights",
            {
                "restore_weights_before_load": restore_weights_before_load,
                "post_process_quantization": post_process_quantization,
            },
        )

    def start_profile(
        self,
        # The output directory
        output_dir: str | None = None,
        # If set, it profile as many as this number of steps.
        # If it is set, profiling is automatically stopped after this step, and
        # the caller doesn't need to run stop_profile.
        start_step: int | None = None,
        num_steps: int | None = None,
        activities: list[str] | None = None,
        profile_by_stage: bool = False,
        with_stack: bool | None = None,
        record_shapes: bool | None = None,
    ):
        if self.node_rank != 0:
            return
        response = requests.post(
            f"http://{self.server_host}:{self.server_port}/start_profile",
            json={
                "output_dir": output_dir,
                "start_step": start_step,
                "num_steps": num_steps,
                "activities": activities,
                "profile_by_stage": profile_by_stage,
                "with_stack": with_stack,
                "record_shapes": record_shapes,
            },
        )
        response.raise_for_status()
        return response

    def stop_profile(self):
        if self.node_rank != 0:
            return
        response = requests.post(f"http://{self.server_host}:{self.server_port}/stop_profile", json={})
        response.raise_for_status()
        return response

    def simulate_crash(self):
        if self.args.rollout_external or not getattr(self, "process", None):
            logger.info(
                "simulate_crash called but no local engine process exists (rollout_external=%s); skip kill",
                self.args.rollout_external,
            )
            return

        logger.info(f"Simulating crash on engine {self.server_host}:{self.server_port}...")
        self.shutdown()


def _compute_server_args(
    args,
    rank,
    dist_init_addr,
    nccl_port,
    host,
    port,
    worker_type: str = "regular",
    disaggregation_bootstrap_port: int | None = None,
    base_gpu_id: int | None = None,
    sglang_overrides: dict | None = None,
    num_gpus_per_engine: int | None = None,
):
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    normalized_overrides = {key.replace("-", "_"): value for key, value in (sglang_overrides or {}).items()}
    pp_size = int(normalized_overrides.get("pp_size", args.sglang_pp_size))
    tp_size = int(normalized_overrides.get("tp_size", _gpus_per_engine // pp_size))
    nnodes = max(1, _gpus_per_engine // args.num_gpus_per_node)
    node_rank = rank % nnodes
    base = base_gpu_id if base_gpu_id is not None else get_base_gpu_id(args, rank)
    base = accelerator.resolve_visible_device_id(base)
    kwargs = {
        # rollout may serve a different ckpt than the trainer (DSpark draft
        # stages); weight-sync/tokenizer keep using args.hf_checkpoint.
        "model_path": getattr(args, "rollout_model_path", None) or args.hf_checkpoint,
        "trust_remote_code": True,
        "random_seed": args.seed + rank * args.num_gpus_per_node,
        # memory
        "enable_memory_saver": args.offload_rollout,
        # distributed
        "host": host,
        "port": port,
        "nccl_port": nccl_port,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "dist_init_addr": dist_init_addr,
        "gpu_id_step": 1,
        "base_gpu_id": base,
        # parallel
        "tp_size": tp_size,
        "dp_size": args.sglang_dp_size,
        "pp_size": pp_size,
        "ep_size": args.sglang_ep_size,
        # always skip warmup to prevent warmup timeout.
        "skip_server_warmup": True,
        # always enable draft weights cpu backup so that we run training without mtp weights.
        "enable_draft_weights_cpu_backup": True,
        # Always enable Prometheus metrics so the router /engine_metrics endpoint
        # is available for external scraping.
        "enable_metrics": True,
    }

    if worker_type == "prefill":
        kwargs["disaggregation_mode"] = "prefill"
        kwargs["load_balance_method"] = "follow_bootstrap_room"
        assert (
            disaggregation_bootstrap_port is not None
        ), "disaggregation_bootstrap_port must be set for prefill worker"
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == "decode":
        kwargs["disaggregation_mode"] = "decode"
        kwargs["prefill_round_robin_balance"] = True
    elif worker_type == "encoder":
        kwargs["encoder_only"] = True

    if args.use_rollout_routing_replay:
        kwargs["enable_return_routed_experts"] = True
    if args.fp16:
        kwargs["dtype"] = "float16"
    external_engine_need_check_fields = [k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS]

    server_arg_fields = dataclasses.fields(ServerArgs)
    server_arg_field_names = {attr.name for attr in server_arg_fields}
    unused_keys = set(kwargs.keys())
    for attr in server_arg_fields:
        if worker_type == "decode" and attr.name == "enable_hierarchical_cache":
            continue
        if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
            kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")
        unused_keys.discard(attr.name)

    # Per-server-group overrides from --sglang-config YAML.
    # Applied after base args so they take highest priority.
    if sglang_overrides:
        for key, value in sglang_overrides.items():
            normalized_key = key.replace("-", "_")
            if normalized_key != key:
                logger.warning(
                    f"sglang_overrides key '{key}' normalized to '{normalized_key}' (rank={rank}). "
                    "Please use underscore style in YAML overrides."
                )
            if normalized_key in kwargs:
                logger.info(
                    f"sglang_overrides: overriding {normalized_key}={kwargs[normalized_key]} -> {value} (rank={rank})"
                )
            kwargs[normalized_key] = value
            if normalized_key in server_arg_field_names:
                unused_keys.discard(normalized_key)
            else:
                unused_keys.add(normalized_key)

    if (
        "cuda_graph_backend_prefill" in server_arg_field_names
        and kwargs.get("enable_memory_saver")
        and kwargs.get("cuda_graph_backend_prefill") is None
    ):
        # Breakable is SGLang's default prefill backend on CUDA, but it is incompatible with memory saver mode.
        kwargs["cuda_graph_backend_prefill"] = "disabled"

    # for compatibility with old args
    if len(unused_keys) > 0:
        logger.info(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
        for key in unused_keys:
            kwargs.pop(key)

    return kwargs, external_engine_need_check_fields


_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS = [
    "model_path",
    "trust_remote_code",
    "random_seed",
    "host",
    "port",
    "nccl_port",
    "nnodes",
    "node_rank",
    "dist_init_addr",
    "gpu_id_step",
    "base_gpu_id",
    "tp_size",
    "dp_size",
    "pp_size",
    "ep_size",
    "skip_server_warmup",
    "enable_draft_weights_cpu_backup",
    "enable_metrics",
    "mem_fraction_static",
]
