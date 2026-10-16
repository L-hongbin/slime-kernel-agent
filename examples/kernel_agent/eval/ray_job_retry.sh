#!/usr/bin/env bash
# Source this file and call ray_job_submit_with_retry with ray job submit arguments.
# Only retry dashboard/Job Agent startup races, never an already submitted job.
ray_job_submit_with_retry() (
    local max_attempts="${RAY_SUBMIT_MAX_ATTEMPTS:-12}"
    local retry_interval="${RAY_SUBMIT_RETRY_INTERVAL:-5}"
    local submit_tmpdir attempt status
    local -a pipeline_status

    if [[ ! "${max_attempts}" =~ ^[1-9][0-9]*$ ]]; then
        echo "RAY_SUBMIT_MAX_ATTEMPTS must be a positive integer" >&2
        return 2
    fi
    if [[ ! "${retry_interval}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "RAY_SUBMIT_RETRY_INTERVAL must be a non-negative number of seconds" >&2
        return 2
    fi

    submit_tmpdir=$(mktemp -d) || return 1
    trap 'rm -rf -- "${submit_tmpdir}"' EXIT
    set -o pipefail
    for ((attempt = 1; attempt <= max_attempts; attempt++)); do
        # Keep logs visible while remembering whether Ray accepted the job.
        if ray job submit "$@" 2>"${submit_tmpdir}/stderr" | tee "${submit_tmpdir}/stdout"; then
            cat "${submit_tmpdir}/stderr" >&2
            return 0
        else
            pipeline_status=("${PIPESTATUS[@]}")
            status=${pipeline_status[0]}
            if (( status == 0 )); then
                status=${pipeline_status[1]}
            fi
        fi
        cat "${submit_tmpdir}/stderr" >&2
        if (( attempt == max_attempts )) ||
            grep -Eq "Job .* submitted successfully" "${submit_tmpdir}/stdout" "${submit_tmpdir}/stderr" ||
            ! grep -Eq 'No available agent to submit job|Connection refused|Failed to establish a new connection' "${submit_tmpdir}/stderr"; then
            echo "Ray job submit failed (status ${status}); not retrying." >&2
            return "${status}"
        fi
        echo "Ray Job Agent not ready; retrying after attempt ${attempt}/${max_attempts} in ${retry_interval}s" >&2
        sleep "${retry_interval}" || return $?
    done
)
