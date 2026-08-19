import ast
import re
from pathlib import Path
from types import SimpleNamespace


def test_train_actor_uses_full_gpu_when_not_colocated():
    from slime.ray.placement_group import _num_gpus_per_train_actor

    assert _num_gpus_per_train_actor(SimpleNamespace(colocate=False)) == 1


def test_train_actor_keeps_fractional_gpu_for_colocation():
    from slime.ray.placement_group import _num_gpus_per_train_actor

    assert _num_gpus_per_train_actor(SimpleNamespace(colocate=True)) == 0.4


def test_role_placement_bundles_carry_custom_resource():
    from slime.ray.placement_group import _make_gpu_bundles

    assert _make_gpu_bundles(2, resource_name="slime_actor") == [
        {"GPU": 1, "CPU": 1, "slime_actor": 1},
        {"GPU": 1, "CPU": 1, "slime_actor": 1},
    ]


def test_role_placement_resources_create_separate_actor_and_rollout_pgs(monkeypatch):
    from slime.ray import placement_group as placement_group_module

    calls = []

    def fake_create(num_gpus, *, role="", resource_name=None):
        calls.append((num_gpus, role, resource_name))
        return f"{role}_pg", [f"{role}_bundle_{i}" for i in range(num_gpus)], list(range(num_gpus))

    monkeypatch.setattr(placement_group_module, "_create_placement_group", fake_create)

    args = SimpleNamespace(
        debug_train_only=False,
        debug_rollout_only=False,
        colocate=False,
        actor_num_nodes=2,
        actor_num_gpus_per_node=8,
        rollout_num_gpus=4,
        actor_placement_resource="slime_actor",
        rollout_placement_resource="slime_rollout",
        use_critic=False,
    )

    pgs = placement_group_module.create_placement_groups(args)

    assert calls == [
        (16, "actor", "slime_actor"),
        (4, "rollout", "slime_rollout"),
    ]
    assert pgs["actor"][0] == "actor_pg"
    assert pgs["rollout"][0] == "rollout_pg"
    assert pgs["critic"] is None


def test_rollout_actor_disables_sglang_tp_memory_imbalance_guard():
    source = Path(__file__).resolve().parents[1] / "slime" / "ray" / "rollout.py"
    tree = ast.parse(source.read_text())
    defaults = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SGLANG_ENGINE_ENV_DEFAULTS" for target in node.targets
        ):
            defaults = ast.literal_eval(node.value)
            break

    assert defaults is not None
    assert defaults["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] == "false"
    assert "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK" not in defaults
    assert "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK" not in defaults


def test_dsv4_rollout_smoke_uses_fp8_low_latency_recipe_by_default():
    source = (Path(__file__).resolve().parents[1] / "scripts" / "dsv4" / "rollout_smoke.sh").read_text()

    assert "ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}" in source
    assert "GPUS_PER_ENGINE=${GPUS_PER_ENGINE:-4}" in source
    assert "RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}" in source
    assert "LOG=${LOG:-${REPO}/local_artifacts/deepseek-v4/r2_logs/r4_node62_rollout_smoke_${RUN_ID}.log}" in source
    assert "r4_node62_rollout_smoke_attempt1.log" not in source
    assert 'gsub(/[^0-9.]/, "", $2)' in source
    assert "$2 + 0 > max" in source
    assert "SGLANG_DP_SIZE=${SGLANG_DP_SIZE:-4}" in source
    assert "USE_SGLANG_DEEPEP=${USE_SGLANG_DEEPEP:-0}" in source
    assert "SGLANG_DSV4_FP4_EXPERTS=${SGLANG_DSV4_FP4_EXPERTS:-0}" in source
    assert "SGLANG_ROUTER_REGISTRATION_TIMEOUT_SECS=${SGLANG_ROUTER_REGISTRATION_TIMEOUT_SECS:-120}" in source
    assert '"SGLANG_ROUTER_REGISTRATION_TIMEOUT_SECS": os.environ["SGLANG_ROUTER_REGISTRATION_TIMEOUT_SECS"]' in source
    assert "ray_start_args=(" in source
    assert 'ray start "${ray_start_args[@]}"' in source
    assert 'GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" ray start \\' not in source
    assert "python3 scripts/dsv4/verify_rollout_dump.py" in source
    assert 'python3 - "${DEBUG_DIR}/rollout_0.pt" <<' not in source
    assert "--sglang-data-parallel-size" in source
    assert "--sglang-enable-dp-attention" in source
    assert "--sglang-moe-a2a-backend deepep" in source
    assert "--sglang-deepep-config" in source
    assert 'if [[ "${USE_SGLANG_DEEPEP}" == "1" ]]' in source
    assert "--sglang-moe-runner-backend marlin" not in source


def test_train_smoke_gpu_idle_check_strips_nvidia_smi_units():
    source = (Path(__file__).resolve().parents[1] / "scripts" / "dsv4" / "train_smoke.sh").read_text()

    assert 'gsub(/[^0-9.]/, \\"\\", \\$2)' in source
    assert "\\$2 + 0 > max" in source
    assert "RAY_JOB_STATUS_MAX_FAILURES=${RAY_JOB_STATUS_MAX_FAILURES:-5}" in source
    assert "status_failures=$((status_failures + 1))" in source
    assert "status polling failed" in source


def test_r6_full_loop_smoke_starts_from_rollout_zero_and_cleans_debug_dumps():
    source = (Path(__file__).resolve().parents[1] / "scripts" / "dsv4" / "_dsv4_launch_core.sh").read_text()

    assert "START_ROLLOUT_ID=${START_ROLLOUT_ID:-0}" in source
    assert "MOE_ROUTER_TOPK" not in source
    assert "--moe-router-topk" not in source
    assert '--start-rollout-id "${START_ROLLOUT_ID}"' in source
    assert 'rm -f "${DEBUG_DIR}"/rollout_*.pt "${DEBUG_DIR}"/train_*.pt' in source
    assert 'python3 scripts/dsv4/verify_rollout_dump.py "${DEBUG_DIR}/rollout_0.pt"' in source


def test_r6_full_loop_smoke_keeps_runtime_cache_off_root_disk():
    source = (Path(__file__).resolve().parents[1] / "scripts" / "dsv4" / "_dsv4_launch_core.sh").read_text()

    assert "readonly RUNTIME_CACHE_ROOT=/dev/shm/v4r6_full_loop_cache" in source
    assert "export TMPDIR=${TMPDIR:-${RUNTIME_CACHE_ROOT}/tmp}" in source
    assert "export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${RUNTIME_CACHE_ROOT}/xdg}" in source
    assert "export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/triton}" in source
    assert "export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/torchinductor}" in source
    assert "export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-${RUNTIME_CACHE_ROOT}/cuda}" in source
    # The cache dirs reach the remote ray/engine/actor envs via the cluster
    # lib's unified env transport core list.
    lib = (Path(__file__).resolve().parents[1] / "scripts" / "dsv4" / "_dsv4_cluster_lib.sh").read_text()
    core = re.search(r"local -a core_keys=\((.*?)\n  \)", lib, flags=re.S)
    assert core, "cluster lib must declare the env-transport core_keys list"
    for key in ["TMPDIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"]:
        assert re.search(rf"^\s*{key}\s*$", core.group(1), flags=re.M), (
            f"{key} missing from the env transport core list: remote caches " "would fall back to the root disk"
        )
    assert "TRAIN_ENV_VARS_JSON=$(dsv4_build_train_env_vars_json)" in source
    # worker-side: start_ray_worker (lib) pre-creates the cache dirs on the
    # remote node and injects the transported env prefix before ray start.
    assert (
        "mkdir -p ${temp_dir} ${TMPDIR} ${XDG_CACHE_HOME} ${TRITON_CACHE_DIR} ${TORCHINDUCTOR_CACHE_DIR} ${CUDA_CACHE_PATH}"
        in lib
    )
    assert "${env_prefix}ray start" in lib


def test_sglang_engine_registers_router_with_timeout(monkeypatch):
    from slime.backends.sglang_utils import sglang_engine

    class DummyResponse:
        status_code = 200

        def raise_for_status(self):
            return None

    calls = []

    monkeypatch.setattr(sglang_engine, "ServerArgs", lambda **kwargs: kwargs)
    monkeypatch.setattr(sglang_engine, "launch_server_process", lambda _server_args: object())
    monkeypatch.setattr(sglang_engine.sglang_router, "__version__", "0.3.2")
    monkeypatch.setenv("SGLANG_ROUTER_REGISTRATION_TIMEOUT_SECS", "17")

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return DummyResponse()

    monkeypatch.setattr(sglang_engine.requests, "post", fake_post)

    engine = sglang_engine.SGLangEngine.__new__(sglang_engine.SGLangEngine)
    engine.worker_type = "regular"
    engine.node_rank = 0
    engine.router_ip = "10.0.0.1"
    engine.router_port = 3881
    engine.server_host = "10.0.0.2"
    engine.server_port = 15000

    engine._init_normal({"host": "10.0.0.2", "port": 15000})

    assert calls == [
        (
            "http://10.0.0.1:3881/workers",
            {
                "json": {"url": "http://10.0.0.2:15000", "worker_type": "regular"},
                "timeout": 17.0,
            },
        )
    ]
