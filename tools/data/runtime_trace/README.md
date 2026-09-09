# Runtime component candidates

This workflow independently captures NVBit SASS instruction events, recovers a
conservative subset of dependencies, and backtracks **candidate** neighborhoods
to a caller-selected best answer. It does not use source similarity, task-specific
signatures, or the state-matching implementation. It does not certify component
retention or change training rewards. The report and measured limits are in
[the component-tracking handoff](../../../handoffs/paper/runtime_component_tracking.md).

## Build and capture

Install `requirements.txt` for CPU matching. Download the public NVBit 1.8 x86_64
release to `local_artifacts/component_tracking/nvbit_release_x86_64`; its binary
core remains an external dependency. CUDA 12.9 / H20 is the tested configuration.

```bash
make -C tools/data/runtime_trace
python -m tools.data.runtime_trace.check_runtime_trace
```

The native library and build products go to
`local_artifacts/component_tracking/build/`. Run only trusted workloads in an
isolated process. Check GPU occupancy and verify synchronized source/binary hashes
before node-local execution. Use a fresh trace prefix and a wall-time limit:

```bash
timeout 180 env CUDA_VISIBLE_DEVICES=0 \
  PATH=/usr/local/cuda/bin:$PATH \
  LD_PRELOAD="$PWD/local_artifacts/component_tracking/build/runtime_trace.so" \
  RUNTIME_TRACE_OUTPUT="$PWD/local_artifacts/component_tracking/new_capture" \
  RUNTIME_TRACE_START_ENABLED=0 RUNTIME_TRACE_CTAS=1 \
  RUNTIME_TRACE_LAUNCHES=4 RUNTIME_TRACE_CAPACITY=500000 \
  python -m tools.data.runtime_trace.reference_replay /path/to/reference.json \
    --tracer "$PWD/local_artifacts/component_tracking/build/runtime_trace.so" \
    --output local_artifacts/component_tracking/new_result
```

`reference_replay` consumes `{id, reference_code, source_sha256}` and runs the
original shapes/dtypes with initialization seed 42 and input seed 17, default
training mode and `no_grad`. `replay.py` instead consumes a saved TVM-FFI
submission and requires a KernelGym checkout whose `source_manifest.json` matches
all sources. It uses the normal compiler/precheck, evaluation mode and the
existing TF32 diagnostic policy. Neither runner starts an evaluator service.

With `RUNTIME_TRACE_START_ENABLED=0`, synchronize, call
`ctypes.CDLL(None).runtime_trace_set_enabled(1)`, execute the measured forward,
synchronize, then disable it. This API affects only capture; it is not a model
feedback channel. Clear `LD_PRELOAD` from subprocess environments after startup
so compilers/children do not inherit the capture destination.

Native defaults capture one CTA from at most four kernels. Set CTAS and LAUNCHES
to `-1` for all, subject to the event buffer limit. A kernel-name filter can select
a diagnostic region. Sampling, skipped launches and dropped events are explicit
coverage gaps; they are not full-program evidence. The tool serializes launches,
rejects active CUDA Graphs and multiple active contexts/host launch threads, and
is unsuitable for scored latency. Unhandled APIs are recorded as unknown.

The JSONL stores static SASS/operands, launch/configuration events and completion
counts. Each selected launch has a 40-byte-per-event binary file (address,
constant bits, instruction, CTA, thread, evaluated guard and active mask). A
missing completion/end or a count mismatch cannot be treated as successful full
capture. Subword/unaligned constant-bank operands are not read by the 32-bit
NVBit constant helper; their value remains unknown.

## Graphs and backtracking

```bash
python -m tools.data.runtime_trace.extract \
  --trace local_artifacts/component_tracking/new_capture \
  --allocations local_artifacts/component_tracking/new_result/allocations.json \
  --context /path/to/context.json \
  --output local_artifacts/component_tracking/new_graph.json
python -m tools.data.runtime_trace.lineage \
  --snapshots /path/to/snapshots.json \
  --output local_artifacts/component_tracking/new_candidates
```

A comparison context contains `task`, `model`, `input_signature`,
`environment_signature`, `decision_turn` and `horizon`; incompatible task/model/
input/environment values cannot produce candidates. Snapshots are a JSON list of
`{turn, correct, score, graph}`, with graph paths relative to that JSON file. The
score must be supplied explicitly. Earliest maximum wins; a missing winning graph
never silently selects an inferior version. All-failed trajectories produce no
candidate labels.

The graph separates comparable attributes from evidence such as function names,
PCs, raw pointers and SASS locators. Buffer byte regions use declared storage
identities. Registers use supported explicit operand contracts, including vector
widths and uniform carry predicates. Memory edges require byte overlap plus
same-thread, same-stream or verified full-CTA barrier order. Log arrival order
never establishes cross-thread dependence. Output values have distinct nodes,
including aliases of an input storage. Unsupported instructions, opaque CBANK
roles, missing allocations and incomplete observations remain unknown.

Repeated instruction execution is encoded losslessly as per-thread
predicate/mask runs and adjacent-thread ranges. Memory access sequences are
fingerprinted per instruction/thread, preserving offsets and iteration order.
The resulting compressed graph is **not** an isomorphism proof about the fully
expanded dynamic dependence graph.

`match.py` uses independently written candidate indexing plus bounded NetworkX
VF2 checks of rooted attributed multigraph neighborhoods. Unknown alternatives,
ambiguous repeated fragments and inconsistent partial mappings are retained as
uncertainty. Candidate records explicitly set `certifies_retention=false`.
`lineage.py` records earliest *observed candidate* correspondence and whether a
center lies on an observed output-dependency path. `eligible_for_credit` is false;
ABI/prologue similarity, local candidates and node counts are not reward labels.

## Evidence boundaries

The retained artifacts distinguish early joint-prototype results from final
independent analysis. The final collector, dependency recovery, matcher and
backtracker live entirely in this directory; no peer implementation is imported.
The same target binaries/input sources may be used to cross-check independently
collected facts. This agreement does not certify graph semantics or state merges.

CPU checks live beside this offline workflow, outside slime CI. Real CUDA
calibrations include rename/helper extraction, changed scalar, transpose errors,
shared-memory GEMM, split computation, overlapping writes, races and predication.
The report records exact output controls, failed instrumentation, fixes, code
provenance, costs and unresolved scalability/retention gaps.
