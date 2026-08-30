#!/usr/bin/env bash
# Preflight host resource checks before launching a multi-node training job.

set -euo pipefail

label="$(hostname 2>/dev/null || echo unknown)"
expected_gpus=0
min_cpus=64
check_cpu_health=1
check_gpu_occupancy=1
gpu_memory_used_max_mib="${GPU_MEMORY_USED_MAX_MIB:-1024}"
load_max_ratio="0.7"
idle_min_percent=50
cpu_window=1
check_kernelgym_health=1
kernelgym_url="${KERNELGYM_URL:-http://127.0.0.1:20211}"
kernelgym_health_script=""
kernelgym_health_attempts="${KERNELGYM_HEALTH_ATTEMPTS:-3}"
kernelgym_health_interval="${KERNELGYM_HEALTH_INTERVAL:-2}"
kernelgym_health_backoff="${KERNELGYM_HEALTH_BACKOFF:-1}"
kernelgym_health_max_interval="${KERNELGYM_HEALTH_MAX_INTERVAL:-60}"
python_bin="${PYTHON_BIN:-python3}"

usage() {
    cat <<'EOF'
Usage: check_host_resources.sh [options]

Options:
  --label LABEL                Human-readable host label for logs.
  --expected-gpus N            Minimum GPUs expected on this host.
  --min-cpus N                 Minimum online CPUs expected on this host.
  --load-max-ratio R           Fail if 1m loadavg is greater than R * nproc.
  --idle-min-percent N         Fail if sampled CPU idle percent is below N.
  --cpu-window SECONDS         Sampling window for CPU idle percent.
  --kernelgym-url URL          KernelGym base URL for /health preflight.
  --kernelgym-health-script P  Path to scripts/check_kernelgym_health.py.
  --kernelgym-health-attempts N  Number of KernelGym attempts; 0 retries forever.
  --kernelgym-health-interval S  Initial seconds between attempts.
  --kernelgym-health-backoff F   Retry interval multiplier (>=1).
  --kernelgym-health-max-interval S  Maximum seconds between attempts.
  --python-bin PATH            Python interpreter used for KernelGym health.
  --skip-cpu-health            Only check CPU count, not current load/idle.
  --gpu-memory-used-max-mib N  Fail occupancy check if any GPU exceeds N MiB.
  --skip-gpu-occupancy         Check GPU count, but allow existing GPU processes.
  --skip-kernelgym-health      Skip KernelGym /health preflight.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --label)
            label="$2"
            shift 2
            ;;
        --expected-gpus)
            expected_gpus="$2"
            shift 2
            ;;
        --min-cpus)
            min_cpus="$2"
            shift 2
            ;;
        --load-max-ratio)
            load_max_ratio="$2"
            shift 2
            ;;
        --idle-min-percent)
            idle_min_percent="$2"
            shift 2
            ;;
        --cpu-window)
            cpu_window="$2"
            shift 2
            ;;
        --kernelgym-url)
            kernelgym_url="$2"
            shift 2
            ;;
        --kernelgym-health-script)
            kernelgym_health_script="$2"
            shift 2
            ;;
        --kernelgym-health-attempts)
            kernelgym_health_attempts="$2"
            shift 2
            ;;
        --kernelgym-health-interval)
            kernelgym_health_interval="$2"
            shift 2
            ;;
        --kernelgym-health-backoff)
            kernelgym_health_backoff="$2"
            shift 2
            ;;
        --kernelgym-health-max-interval)
            kernelgym_health_max_interval="$2"
            shift 2
            ;;
        --python-bin)
            python_bin="$2"
            shift 2
            ;;
        --skip-cpu-health)
            check_cpu_health=0
            shift
            ;;
        --gpu-memory-used-max-mib)
            gpu_memory_used_max_mib="$2"
            shift 2
            ;;
        --skip-gpu-occupancy)
            check_gpu_occupancy=0
            shift
            ;;
        --skip-kernelgym-health)
            check_kernelgym_health=0
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

status=0

if ! [[ "${gpu_memory_used_max_mib}" =~ ^[0-9]+$ ]]; then
    echo "error: --gpu-memory-used-max-mib must be a non-negative integer" >&2
    exit 2
fi

fail() {
    echo "resource-check: FAILED: $*" >&2
    status=1
}

online_cpus() {
    nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 0
}

read_cpu_totals() {
    local cpu user nice system idle iowait irq softirq steal guest guest_nice
    read -r cpu user nice system idle iowait irq softirq steal guest guest_nice < /proc/stat
    echo "$((user + nice + system + idle + iowait + irq + softirq + steal)) $((idle + iowait))"
}

check_cpu() {
    local ncpu load1 load5 load15 rest running ours_r load_limit totals1 totals2 total1 idle1 total2 idle2 dt didle idlepct

    ncpu="$(online_cpus)"
    echo "resource-check: ${label}: cpu online=${ncpu}, min_required=${min_cpus}"
    if [[ "${ncpu}" -lt "${min_cpus}" ]]; then
        fail "${label}: online CPUs ${ncpu} < required ${min_cpus}"
    fi

    if [[ "${check_cpu_health}" != "1" ]]; then
        echo "resource-check: ${label}: CPU health load/idle check skipped"
        return
    fi

    read -r load1 load5 load15 rest < /proc/loadavg
    running="$(awk '/^procs_running/{print $2}' /proc/stat)"
    totals1="$(read_cpu_totals)"
    sleep "${cpu_window}"
    totals2="$(read_cpu_totals)"
    read -r total1 idle1 <<<"${totals1}"
    read -r total2 idle2 <<<"${totals2}"
    dt=$((total2 - total1))
    didle=$((idle2 - idle1))
    idlepct="$(awk -v di="${didle}" -v dt="${dt}" 'BEGIN{printf "%.0f", (dt > 0) ? 100 * di / dt : 100}')"
    ours_r="$(ps -eL -o stat= 2>/dev/null | awk '$1 ~ /^R/ {n++} END{print n + 0}')"
    load_limit="$(awk -v n="${ncpu}" -v ratio="${load_max_ratio}" 'BEGIN{printf "%.1f", n * ratio}')"

    echo "resource-check: ${label}: loadavg(1m)=${load1}, load_limit=${load_limit}, procs_running=${running}, cpu_idle=${idlepct}%, container_R=${ours_r}"

    if awk -v load="${load1}" -v limit="${load_limit}" 'BEGIN{exit !(load > limit)}'; then
        fail "${label}: loadavg(1m) ${load1} > ${load_limit}"
    fi
    if [[ "${idlepct}" -lt "${idle_min_percent}" ]]; then
        fail "${label}: CPU idle ${idlepct}% < ${idle_min_percent}%"
    fi
    if [[ "${running:-0}" -gt "$((ours_r + ncpu / 4))" ]] 2>/dev/null && [[ "${running:-0}" -gt 64 ]]; then
        echo "resource-check: ${label}: warning: host procs_running=${running} is much larger than container-visible R=${ours_r}; load may be outside this container"
    fi
}

check_gpu() {
    local detected summary occupancy memory_used high_memory

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        fail "${label}: nvidia-smi not found"
        return
    fi

    detected="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    echo "resource-check: ${label}: gpu detected=${detected}, expected=${expected_gpus}"
    if [[ "${detected}" -lt "${expected_gpus}" ]]; then
        fail "${label}: detected GPUs ${detected} < expected ${expected_gpus}"
    fi

    summary="$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "${summary}" ]]; then
        echo "resource-check: ${label}: GPU summary:"
        echo "${summary}" | sed 's/^/  /'
    fi

    if [[ "${check_gpu_occupancy}" != "1" ]]; then
        echo "resource-check: ${label}: GPU occupancy check skipped"
        return
    fi

    if ! memory_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>&1)"; then
        fail "${label}: GPU memory usage query failed: ${memory_used}"
        return
    fi
    high_memory="$(echo "${memory_used}" | awk -v limit="${gpu_memory_used_max_mib}" '$1 + 0 > limit {print NR - 1 ", " $1 " MiB"}')"
    if [[ -n "${high_memory}" ]]; then
        fail "${label}: GPU memory usage exceeds ${gpu_memory_used_max_mib} MiB; compute processes may be hidden by a PID namespace:"
        echo "${high_memory}" | sed 's/^/  gpu /' >&2
    fi

    if ! occupancy="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>&1)"; then
        fail "${label}: GPU occupancy query failed: ${occupancy}"
        return
    fi

    occupancy="$(echo "${occupancy}" | sed '/^[[:space:]]*$/d')"
    if [[ -n "${occupancy}" ]]; then
        fail "${label}: GPU compute processes are already running:"
        echo "${occupancy}" | sed 's/^/  /' >&2
        return
    fi

    echo "resource-check: ${label}: GPU compute-process list is empty"
}

resolve_kernelgym_health_script() {
    local source_path script_dir

    if [[ -n "${kernelgym_health_script}" ]]; then
        echo "${kernelgym_health_script}"
        return
    fi

    source_path="${BASH_SOURCE[0]:-}"
    if [[ -n "${source_path}" && -f "${source_path}" ]]; then
        script_dir="$(cd -- "$(dirname -- "${source_path}")" >/dev/null 2>&1 && pwd)"
        if [[ -f "${script_dir}/check_kernelgym_health.py" ]]; then
            echo "${script_dir}/check_kernelgym_health.py"
            return
        fi
    fi

    echo "scripts/check_kernelgym_health.py"
}

check_kernelgym() {
    local health_script

    if [[ "${check_kernelgym_health}" != "1" ]]; then
        echo "resource-check: ${label}: KernelGym health check skipped"
        return
    fi

    health_script="$(resolve_kernelgym_health_script)"
    if [[ ! -f "${health_script}" ]]; then
        fail "${label}: KernelGym health script not found: ${health_script}"
        return
    fi

    echo "resource-check: ${label}: checking KernelGym health at ${kernelgym_url}"
    if ! "${python_bin}" "${health_script}" \
        --url "${kernelgym_url}" \
        --attempts "${kernelgym_health_attempts}" \
        --interval "${kernelgym_health_interval}" \
        --backoff-factor "${kernelgym_health_backoff}" \
        --max-interval "${kernelgym_health_max_interval}"; then
        fail "${label}: KernelGym health check failed"
    fi
}

echo "resource-check: checking ${label}"
check_cpu
check_gpu

if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
fi

check_kernelgym

if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
fi

echo "resource-check: ${label}: OK"
