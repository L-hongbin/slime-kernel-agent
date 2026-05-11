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

clean_ray_session_dirs() {
    local ray_tmpdir="${RAY_TMPDIR:-/tmp/ray}"

    [[ "${RAY_CLEAN_TMP_SESSION_DIRS:-1}" == "1" ]] || return
    [[ -d "${ray_tmpdir}" ]] || return

    find "${ray_tmpdir}" -mindepth 1 -maxdepth 1 \
        \( -name 'session_*' -o -name 'session_latest' \) \
        -exec rm -rf {} + || true
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

preseed_dashboard_agent_port_files() {
    set +x

    local ray_tmpdir="${RAY_TMPDIR:-/tmp/ray}"
    local timeout_s=${RAY_PORT_FILE_PRESEED_TIMEOUT_S:-30}
    local deadline=$((SECONDS + timeout_s))
    local started_at
    local session_link
    local session_dir
    local raylet_out
    local raylet_mtime
    local node_id

    # Some Ray builds fatal after ~15s if dashboard agent port files are not
    # written yet, even though the dashboard agent is still starting.
    started_at=$(date +%s)
    session_link="${ray_tmpdir}/session_latest"

    while ((SECONDS < deadline)); do
        session_dir=$(readlink -f "${session_link}" 2>/dev/null || true)
        if [[ -n "${session_dir}" ]]; then
            raylet_out="${session_dir}/logs/raylet.out"
            if [[ -f "${raylet_out}" ]]; then
                raylet_mtime=$(stat -c %Y "${raylet_out}" 2>/dev/null || echo 0)
                if [[ "${raylet_mtime}" -ge "${started_at}" ]]; then
                    node_id=$(sed -n 's/.*Setting node ID node_id=\([0-9a-f]*\).*/\1/p' "${raylet_out}" | tail -n 1)
                    if [[ -n "${node_id}" ]]; then
                        printf '%s' "${RAY_DASHBOARD_AGENT_GRPC_PORT}" >"${session_dir}/metrics_agent_port_${node_id}"
                        printf '%s' "${RAY_METRICS_EXPORT_PORT}" >"${session_dir}/metrics_export_port_${node_id}"
                        printf '%s' "${RAY_DASHBOARD_AGENT_LISTEN_PORT}" >"${session_dir}/dashboard_agent_listen_port_${node_id}"
                        echo "Preseeded Ray dashboard port files for node ${node_id} in ${session_dir}"
                        return 0
                    fi
                fi
            fi
        fi
        sleep 0.2
    done

    echo "Did not preseed Ray dashboard port files within ${timeout_s}s"
    return 0
}

wait_for_ray_job_server() {
    local timeout_s=${RAY_JOB_SERVER_WAIT_TIME_S:-120}
    local deadline=$((SECONDS + timeout_s))

    while ((SECONDS < deadline)); do
        if ray job list --address="${RAY_JOB_ADDRESS}" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done

    echo "error: Ray job server did not become ready within ${timeout_s}s: ${RAY_JOB_ADDRESS}" >&2
    dump_ray_logs
    return 1
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
    RAY_RUNTIME_ENV_AGENT_PORT=${RAY_RUNTIME_ENV_AGENT_PORT:-52367}
    RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-10000000000}
    export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-120}
    export RAY_agent_register_timeout_ms=${RAY_agent_register_timeout_ms:-120000}

    export MASTER_ADDR NODE_ADDR RAY_PORT RAY_DASHBOARD_PORT

    local no_proxy_extra="localhost,127.0.0.1,${MASTER_ADDR},${NODE_ADDR}"
    export no_proxy="${no_proxy_extra}${no_proxy:+,${no_proxy}}"
    export NO_PROXY="${no_proxy_extra}${NO_PROXY:+,${NO_PROXY}}"

    stop_ray_processes
    clean_ray_session_dirs
    detect_num_gpus
    detect_nvlink

    local port_preseed_pid=""
    if [[ "${RAY_PRESEED_DASHBOARD_PORT_FILES:-0}" == "1" ]]; then
        preseed_dashboard_agent_port_files &
        port_preseed_pid=$!
    fi

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
            --runtime-env-agent-port "${RAY_RUNTIME_ENV_AGENT_PORT}" \
            --metrics-export-port "${RAY_METRICS_EXPORT_PORT}"; then
            [[ -z "${port_preseed_pid}" ]] || kill "${port_preseed_pid}" 2>/dev/null || true
            dump_ray_logs
            return 1
        fi
        [[ -z "${port_preseed_pid}" ]] || wait "${port_preseed_pid}" || true

        RAY_JOB_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}"
        wait_for_ray_job_server
    elif [[ "${RAY_ROLE}" == "worker" ]]; then
        if ! ray start \
            --address "${MASTER_ADDR}:${RAY_PORT}" \
            --node-ip-address "${NODE_ADDR}" \
            --num-gpus "${NUM_GPUS}" \
            --dashboard-agent-grpc-port "${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
            --dashboard-agent-listen-port "${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
            --runtime-env-agent-port "${RAY_RUNTIME_ENV_AGENT_PORT}" \
            --metrics-export-port "${RAY_METRICS_EXPORT_PORT}" \
            --disable-usage-stats; then
            [[ -z "${port_preseed_pid}" ]] || kill "${port_preseed_pid}" 2>/dev/null || true
            dump_ray_logs
            return 1
        fi
        [[ -z "${port_preseed_pid}" ]] || wait "${port_preseed_pid}" || true

        RAY_JOB_ADDRESS=""
    else
        echo "error: RAY_ROLE must be 'head' or 'worker', got '${RAY_ROLE}'" >&2
        return 2
    fi

    export RAY_JOB_ADDRESS
}

start_ray_cluster
