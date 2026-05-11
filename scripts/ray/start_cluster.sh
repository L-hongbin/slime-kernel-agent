#!/usr/bin/env bash

set -euo pipefail

dump_ray_logs() {
    local ray_log_dir="${RAY_TMPDIR:-/tmp/ray}/session_latest/logs"
    local log_file
    local tail_lines

    echo "ray start failed; dumping Ray logs from ${ray_log_dir}"
    if [[ ! -d "${ray_log_dir}" ]]; then
        echo "Ray log directory does not exist: ${ray_log_dir}"
        return
    fi

    for log_file in \
        "${ray_log_dir}/raylet.err" \
        "${ray_log_dir}/raylet.out" \
        "${ray_log_dir}/dashboard_agent.err" \
        "${ray_log_dir}/dashboard_agent.log" \
        "${ray_log_dir}/gcs_server.err" \
        "${ray_log_dir}/dashboard.err" \
        "${ray_log_dir}/gcs_server.out" \
        "${ray_log_dir}/dashboard.log"; do
        if [[ -f "${log_file}" ]]; then
            tail_lines=80
            if [[ "${log_file}" == *gcs_server.out ]]; then
                tail_lines=30
            fi
            echo "===== ${log_file} ====="
            tail -n "${tail_lines}" "${log_file}" || true
        fi
    done
}

stop_ray_processes() {
    pkill -9 sglang || true
    sleep 3
    ray stop --force || true
    pkill -9 ray || true
    pkill -9 python || true
    sleep 3
    pkill -9 ray || true
    pkill -9 python || true
}

detect_num_gpus() {
    local detected_gpus

    if command -v nvidia-smi >/dev/null 2>&1; then
        detected_gpus=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    else
        detected_gpus=0
    fi

    NUM_GPUS=${NUM_GPUS:-${detected_gpus}}
    if [[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -le 0 ]]; then
        NUM_GPUS=8
    fi
    export NUM_GPUS
    echo "NUM_GPUS: ${NUM_GPUS}"
}

detect_nvlink() {
    local nvlink_count

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvlink_count=$(nvidia-smi topo -m 2>/dev/null | { grep -o 'NV[0-9][0-9]*' || true; } | wc -l)
        nvlink_count=${nvlink_count//[[:space:]]/}
    else
        nvlink_count=0
    fi

    if [[ "${nvlink_count}" -gt 0 ]]; then
        HAS_NVLINK=1
    else
        HAS_NVLINK=0
    fi
    export HAS_NVLINK
    echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${nvlink_count} NVLink references)"
}

start_ray_cluster() {
    RAY_ROLE=${RAY_ROLE:-head}
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    NODE_ADDR=${NODE_ADDR:-${MASTER_ADDR}}
    RAY_PORT=${RAY_PORT:-6379}
    RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
    RAY_DASHBOARD_AGENT_GRPC_PORT=${RAY_DASHBOARD_AGENT_GRPC_PORT:-52366}
    RAY_DASHBOARD_AGENT_LISTEN_PORT=${RAY_DASHBOARD_AGENT_LISTEN_PORT:-52365}
    # Keep this outside Ray's default worker port range, 10002-19999.
    RAY_METRICS_EXPORT_PORT=${RAY_METRICS_EXPORT_PORT:-20000}
    RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-10000000000}
    export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-120}

    export MASTER_ADDR NODE_ADDR RAY_PORT RAY_DASHBOARD_PORT

    local no_proxy_extra="localhost,127.0.0.1,${MASTER_ADDR},${NODE_ADDR}"
    export no_proxy="${no_proxy_extra}${no_proxy:+,${no_proxy}}"
    export NO_PROXY="${no_proxy_extra}${NO_PROXY:+,${NO_PROXY}}"

    stop_ray_processes
    detect_num_gpus
    detect_nvlink

    if [[ "${RAY_ROLE}" == "head" ]]; then
        if ! ray start --head \
            --node-ip-address "${MASTER_ADDR}" \
            --port "${RAY_PORT}" \
            --num-gpus "${NUM_GPUS}" \
            --object-store-memory "${RAY_OBJECT_STORE_MEMORY}" \
            --disable-usage-stats \
            --dashboard-host=0.0.0.0 \
            --dashboard-port "${RAY_DASHBOARD_PORT}" \
            --dashboard-agent-grpc-port "${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
            --dashboard-agent-listen-port "${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
            --metrics-export-port "${RAY_METRICS_EXPORT_PORT}"; then
            dump_ray_logs
            return 1
        fi

        RAY_JOB_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}"
    elif [[ "${RAY_ROLE}" == "worker" ]]; then
        if ! ray start \
            --address "${MASTER_ADDR}:${RAY_PORT}" \
            --node-ip-address "${NODE_ADDR}" \
            --num-gpus "${NUM_GPUS}" \
            --disable-usage-stats; then
            dump_ray_logs
            return 1
        fi

        RAY_JOB_ADDRESS=""
    else
        echo "error: RAY_ROLE must be 'head' or 'worker', got '${RAY_ROLE}'" >&2
        return 2
    fi

    export RAY_JOB_ADDRESS
}

start_ray_cluster
