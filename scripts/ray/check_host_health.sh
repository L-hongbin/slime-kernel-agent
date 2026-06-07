#!/bin/bash
# Pre-launch host health check — detect CPU saturation / runaway / possible malware BEFORE a run.
#
# Why: 2026-06-06 incident — a crypto-miner saturated all CPU cores; it lived in a different PID
# namespace, so it was INVISIBLE to this container's `ps`/`top` (we saw loadavg ~600 but only ~19
# locally-runnable threads and couldn't find the source). It inflated training launch ~6x
# (~4min -> ~24min). KEY: /proc/loadavg and /proc/stat are HOST-GLOBAL, so they DO see such load
# even when `ps` (PID-namespaced) does not. This script uses those host-global signals.
#
# Usage:  bash scripts/ray/check_host_health.sh
# Exit 0 = healthy; exit 1 = abnormal (loaded). start_cluster.sh BLOCKS launch on exit 1 by
#   default; set SLIME_REQUIRE_HOST_HEALTH=0 to downgrade to warn-only.
set -u

ncpu=$(nproc)
read -r load1 load5 load15 _ < /proc/loadavg
running=$(awk '/^procs_running/{print $2}' /proc/stat)

# CPU idle% over a 1s window (/proc/stat cpu line: user nice system idle iowait irq softirq steal ...)
read -r _ u1 n1 s1 idle1 _ < /proc/stat
sleep 1
read -r _ u2 n2 s2 idle2 _ < /proc/stat
tot=$(( (u2+n2+s2+idle2) - (u1+n1+s1+idle1) ))
didle=$(( idle2 - idle1 ))
idlepct=$(awk -v di="$didle" -v dt="$tot" 'BEGIN{printf "%.0f", (dt>0)?100*di/dt:100}')

# container-visible runnable threads (R state) — if host procs_running >> this, load is OUTSIDE us
ours_R=$(ps -eL -o stat= 2>/dev/null | grep -c '^R' || echo 0)

load_warn=$(awk -v n="$ncpu" 'BEGIN{printf "%.0f", n*0.7}')

printf 'host-health: nproc=%s loadavg(1m)=%s procs_running(host)=%s cpu_idle=%s%% container_R=%s\n' \
  "$ncpu" "$load1" "$running" "$idlepct" "$ours_R"

status=0
warns=()
if awk -v l="$load1" -v t="$load_warn" 'BEGIN{exit !(l>t)}'; then
  warns+=("loadavg ${load1} > ${load_warn} (0.7*nproc) — host CPU heavily loaded"); status=1
fi
if [ "${idlepct:-100}" -lt 50 ]; then
  warns+=("CPU idle ${idlepct}% (<50%) — something is burning CPU (runaway/malware? esp. if no training job is running)"); status=1
fi
# namespace tell-tale: host runnable far exceeds what we can see locally => load is from outside this container
if [ "${running:-0}" -gt "$(( ours_R + ncpu/4 ))" ] 2>/dev/null && [ "${running:-0}" -gt 64 ]; then
  warns+=("host procs_running=${running} >> container-visible R=${ours_R} → load is largely OUTSIDE this container (other tenants or host-level malware) — you cannot see/fix it from in here")
fi

if [ "$status" -eq 0 ]; then
  echo "host-health: OK — host looks idle/healthy, safe to launch."
else
  for w in "${warns[@]}"; do echo "host-health: ⚠️  $w"; done
  echo "host-health: FAILED — a saturated host inflates startup ~6x and skews all profiling (see handoffs/launch_speedup/). Investigate before launching."
fi
exit "$status"
