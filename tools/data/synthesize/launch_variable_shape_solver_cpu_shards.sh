#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export SHAPE_SOLVER_MODULE=tools.data.synthesize.solve_variable_shape_delta
export SHAPE_EXPECTED_SOLVER_CONTRACT=shape_variable_multislot_solver_v5
exec "${script_dir}/launch_shape_solver_cpu_shards.sh" "$@"
