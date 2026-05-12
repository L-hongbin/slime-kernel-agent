#!/usr/bin/env bash

set -u -o pipefail

NODE_ADDR=${NODE_ADDR:-${MASTER_ADDR:-}}
if [[ -z "${NODE_ADDR}" ]]; then
    if command -v hostname >/dev/null 2>&1; then
        NODE_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
fi
NODE_ADDR=${NODE_ADDR:-127.0.0.1}

RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
RAY_DASHBOARD_AGENT_GRPC_PORT=${RAY_DASHBOARD_AGENT_GRPC_PORT:-52366}
RAY_DASHBOARD_AGENT_LISTEN_PORT=${RAY_DASHBOARD_AGENT_LISTEN_PORT:-52365}
RAY_RUNTIME_ENV_AGENT_PORT=${RAY_RUNTIME_ENV_AGENT_PORT:-52367}
RAY_METRICS_EXPORT_PORT=${RAY_METRICS_EXPORT_PORT:-20000}
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-10000000000}
RAY_AGENT_REGISTER_TIMEOUT_MS=${RAY_AGENT_REGISTER_TIMEOUT_MS:-120000}
RAY_START_TIMEOUT_S=${RAY_START_TIMEOUT_S:-180}
RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray}
LOG_FILE=${LOG_FILE:-/tmp/ray_startup_issue_check_$(date +%Y%m%d_%H%M%S).log}
KEEP_RAY_RUNNING=${KEEP_RAY_RUNNING:-0}
CLEAN_RAY_TMP=${CLEAN_RAY_TMP:-1}

RAY_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}"
SYSTEM_CONFIG=${RAY_SYSTEM_CONFIG:-"{\"agent_register_timeout_ms\":${RAY_AGENT_REGISTER_TIMEOUT_MS}}"}

exec > >(tee -a "${LOG_FILE}") 2>&1

cleanup() {
    local rc=$?
    if [[ "${KEEP_RAY_RUNNING}" != "1" ]]; then
        ray stop --force >/dev/null 2>&1 || true
    fi
    exit "${rc}"
}
trap cleanup EXIT

section() {
    echo
    echo "===== $* ====="
}

dump_ray_logs() {
    local log_dir="${RAY_TMPDIR}/session_latest/logs"
    local file

    section "Ray log tails"
    echo "log_dir=${log_dir}"
    if [[ ! -d "${log_dir}" ]]; then
        echo "Ray log dir does not exist"
        return 0
    fi

    for file in \
        "${log_dir}/raylet.err" \
        "${log_dir}/raylet.out" \
        "${log_dir}/dashboard_agent.err" \
        "${log_dir}/dashboard_agent.log" \
        "${log_dir}/gcs_server.err" \
        "${log_dir}/gcs_server.out" \
        "${log_dir}/dashboard.err" \
        "${log_dir}/dashboard.log"; do
        if [[ -f "${file}" ]]; then
            echo "----- ${file} -----"
            tail -n 80 "${file}" || true
        fi
    done
}

print_result() {
    local name=$1
    local status=$2
    local detail=$3
    printf 'RESULT %-28s %-8s %s\n' "${name}" "${status}" "${detail}"
}

section "Environment"
echo "LOG_FILE=${LOG_FILE}"
echo "NODE_ADDR=${NODE_ADDR}"
echo "RAY_ADDRESS=${RAY_ADDRESS}"
echo "RAY_TMPDIR=${RAY_TMPDIR}"
echo "RAY_START_TIMEOUT_S=${RAY_START_TIMEOUT_S}"
echo "KEEP_RAY_RUNNING=${KEEP_RAY_RUNNING}"
echo "CLEAN_RAY_TMP=${CLEAN_RAY_TMP}"
hostname || true
hostname -I || true
ray --version || true
python3 - <<'PY' || true
import sys
print(sys.executable)
try:
    import ray
    print("ray_python_version=" + ray.__version__)
except Exception as exc:
    print(type(exc).__name__, exc)
PY

section "Cleanup before test"
ray stop --force || true
pkill -9 ray || true
pkill -9 sglang || true
if [[ "${CLEAN_RAY_TMP}" == "1" && -d "${RAY_TMPDIR}" ]]; then
    find "${RAY_TMPDIR}" -mindepth 1 -maxdepth 1 \
        \( -name 'session_*' -o -name 'session_latest' \) \
        -exec rm -rf {} + || true
fi

export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-120}
export RAY_agent_register_timeout_ms=${RAY_agent_register_timeout_ms:-${RAY_AGENT_REGISTER_TIMEOUT_MS}}
export no_proxy="localhost,127.0.0.1,${NODE_ADDR}${no_proxy:+,${no_proxy}}"
export NO_PROXY="localhost,127.0.0.1,${NODE_ADDR}${NO_PROXY:+,${NO_PROXY}}"

section "Issue 1: raylet dashboard-agent port-file timeout"
ray_start_cmd=(
    ray start --head
    --node-ip-address "${NODE_ADDR}"
    --port "${RAY_PORT}"
    --num-gpus "${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ' || echo 0)}"
    --object-store-memory "${RAY_OBJECT_STORE_MEMORY}"
    --disable-usage-stats
    --dashboard-host=0.0.0.0
    --dashboard-port "${RAY_DASHBOARD_PORT}"
    --dashboard-agent-grpc-port "${RAY_DASHBOARD_AGENT_GRPC_PORT}"
    --dashboard-agent-listen-port "${RAY_DASHBOARD_AGENT_LISTEN_PORT}"
    --runtime-env-agent-port "${RAY_RUNTIME_ENV_AGENT_PORT}"
    --metrics-export-port "${RAY_METRICS_EXPORT_PORT}"
    --system-config "${SYSTEM_CONFIG}"
)

set -x
if command -v timeout >/dev/null 2>&1; then
    timeout "${RAY_START_TIMEOUT_S}s" "${ray_start_cmd[@]}"
else
    "${ray_start_cmd[@]}"
fi
START_RC=$?
set +x

if [[ "${START_RC}" -eq 0 ]]; then
    print_result "issue_1_ray_start" "PASS" "ray start succeeded"
else
    print_result "issue_1_ray_start" "FAIL" "ray start failed with rc=${START_RC}"
    print_result "issue_2_job_agent" "SKIP" "ray did not start, so job agent cannot be tested"
    dump_ray_logs
    exit "${START_RC}"
fi

dump_ray_logs

section "Issue 2: Ray job agent availability"
set -x
ray job list --address="${RAY_ADDRESS}"
JOB_LIST_RC=$?
ray job submit --address="${RAY_ADDRESS}" -- echo ray_job_submit_ok
JOB_SUBMIT_RC=$?
set +x

if [[ "${JOB_LIST_RC}" -eq 0 ]]; then
    print_result "issue_2_job_list" "PASS" "ray job list succeeded"
else
    print_result "issue_2_job_list" "FAIL" "ray job list failed with rc=${JOB_LIST_RC}"
fi

if [[ "${JOB_SUBMIT_RC}" -eq 0 ]]; then
    print_result "issue_2_job_submit" "PASS" "ray job submit succeeded"
else
    print_result "issue_2_job_submit" "FAIL" "ray job submit failed with rc=${JOB_SUBMIT_RC}"
fi

section "Summary"
echo "Log saved to ${LOG_FILE}"

if [[ "${JOB_LIST_RC}" -ne 0 || "${JOB_SUBMIT_RC}" -ne 0 ]]; then
    dump_ray_logs
    exit 1
fi
