#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export SHAPE_SOLVER_MODULE=tools.data.synthesize.solve_variable_shape_delta
export SHAPE_SOLVER_SUPPORTS_GROUP_SCOPE=1
export SHAPE_GROUP_SCOPE=${SHAPE_GROUP_SCOPE:-generic}
export SHAPE_EXPECTED_GROUP_SCOPE=${SHAPE_GROUP_SCOPE}
export SHAPE_EXPECTED_SOLVER_SOURCE_SHA256
SHAPE_EXPECTED_SOLVER_SOURCE_SHA256=$(sha256sum \
  "${script_dir}/solve_variable_shape_delta.py" | awk '{print $1}')
export SHAPE_EXPECTED_MULTIDIM_SOURCE_SHA256
SHAPE_EXPECTED_MULTIDIM_SOURCE_SHA256=$(sha256sum \
  "${script_dir}/solve_multidim_shape_coverage.py" | awk '{print $1}')
export SHAPE_EXPECTED_SHAPE_HELPER_SOURCE_SHA256
SHAPE_EXPECTED_SHAPE_HELPER_SOURCE_SHA256=$(sha256sum \
  "${script_dir}/solve_shape_coverage.py" | awk '{print $1}')
case ${SHAPE_GROUP_SCOPE} in
  generic)
    export SHAPE_EXPECTED_SOLVER_CONTRACT=shape_variable_multislot_solver_v5
    export SHAPE_EXPECTED_SOLVER_GENERATOR=same_factory_product_variable_2_to_5_soft_p2_50_v3
    ;;
  nonleading_no_explicit_batch)
    export SHAPE_EXPECTED_SOLVER_CONTRACT=shape_variable_multislot_solver_v6
    export SHAPE_EXPECTED_SOLVER_GENERATOR=same_factory_product_variable_2_to_5_nonleading_no_explicit_batch_soft_p2_50_v1
    ;;
  balanced_nonleading_no_explicit_batch)
    export SHAPE_EXPECTED_SOLVER_CONTRACT=shape_variable_multislot_solver_v7
    export SHAPE_EXPECTED_SOLVER_GENERATOR=same_factory_product_variable_2_to_5_balanced_nonleading_soft_p2_50_v1
    ;;
  *)
    echo "unknown SHAPE_GROUP_SCOPE: ${SHAPE_GROUP_SCOPE}" >&2
    exit 2
    ;;
esac
exec "${script_dir}/launch_shape_solver_cpu_shards.sh" "$@"
