# Open-source CUDA-kernel data survey and ingestion handoff

**Survey cutoff:** 2026-07-26; cleanup status updated 2026-07-27
**Repository:** `slime-v4flash-lora`
**Scope:** open-source data that may improve CUDA-kernel generation, including
executable PyTorch reference tasks, generated CUDA/Triton solutions, reasoning
traces, real deployment workloads, and evaluation-only benchmarks.

Downloading every remote corpus, changing a training launcher, running a new
model-training experiment, and claiming target-GPU correctness are out of
scope. The 2026-07-27 update runs the existing v4 CPU cleanup on KernelBook and
records the explicit decision not to train on MultiKernelBench. Remote data was
downloaded only when it already had a defensible path to the current project
format; otherwise this handoff records the blocker and required adapter or
split.

## Executive decision

The survey now has two fully v4 CPU-cleaned, separately versioned augmentation
candidates. Nothing in this survey was concatenated into the main 45,958-row
v4 training parquet. MultiKernelBench is explicitly excluded from training.

| Decision class | Source | Concrete result | Current decision |
| --- | --- | ---: | --- |
| Best prepared augmentation | [CUDA-Agent-Ops-6K](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K) | 6,000 source -> 6,000 converted -> **3,979 v4-kept** | Keep separate until target-GPU eager/compile validation, attribution, mixture weighting, and benchmark-split review are approved. V4 did not run validation-set decontamination, so this is not a contamination claim. |
| Excluded benchmark-derived source | [MultiKernelBench](https://github.com/wzzll123/MultiKernelBench) | 401 references -> 300 NVIDIA/PyTorch converted -> 212 old-contract survivors; most overlap KernelBench | **Do not include in training.** Preserve the source and historical artifacts only for provenance and analysis. |
| Largest v4-cleaned candidate | [KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook) | 18,162 source -> 18,153 converted -> **12,066 v4-kept** | Keep separate. CPU semantics passed; Researcher Reciprocity approval, repository-grouped splitting, complete KernelBench L1/L2/L3 decontamination, and target-GPU eager/compile validation remain required. |
| Adapter queue | FastKernels | 407 downloaded baselines -> 0 direct conversions | High-value production-aligned source, but it needs workload capture because it has no top-level `get_inputs()` contract. |
| Auxiliary code/SFT | THUNLP TritonBench | 4,024 crawl + 4,133 synth rows downloaded -> 0 RL conversions | Retain for code SFT or retrieval; it is candidate Triton code rather than executable PyTorch reference tasks. |

The main conclusion is that row count is not the limiting factor. The scarce
asset is an independently diverse task with all of: an executable oracle,
non-fixed inputs, a reproducible correctness harness, usable licensing, and a
split that does not destroy a benchmark.

## What has actually been downloaded and processed

All downloaded revisions and checksums are recorded in
`Data/external/SOURCES.json`. Raw sources are under
`Data/external/sources/`; conversions, provenance, audits, rejections, and
summaries are under `Data/external/converted/`.

| Source | Pinned revision | Downloaded | Converted to current DrKernel Arrow/prompt schema | Cleanup level | Usable row claim |
| --- | --- | ---: | ---: | --- | ---: |
| CUDA-Agent-Ops-6K | `44a734c78c947bfcba5189cbfd13f57a6d29a698` | 6,000 | 6,000 | Full v4 CPU semantic audit | 3,979 separate candidates |
| MultiKernelBench | `460a972c9be7ce18035984321d38b910e27df95f` | 401 | 300; 101 NPU-only references excluded | Earlier CPU contract retained only as history | 0 training candidates |
| KernelBook permissive | `b76504d85f7f14ef4b1fad81f136f638f2ce625b` | 18,162 | 18,153; nine ambiguous conversions rejected | Full v4 CPU semantic audit | 12,066 separate candidates |
| FastKernels | `34d7cd6c8de2573edb19eb78aa00302c01b94fef` | 407 baseline modules | 0 | Inventory only | 0 |
| THUNLP TritonBench | `603e28a5050e8c268f6883a69709d477a272d49a` | 8,157 training-support rows | 0 by design | Source retained | 0 RL references |

The shared copy at
`/ms/FM/lihongbin/dataset/CUDA_RL/cuda_rl/prompt_tvm_v2/external_augmentation_20260724/`
contains the converted candidates, audits, licenses, and TritonBench support
files. It remains separate from every training launcher.

### What “converted” means

`python -m tools.data.cleaning.external` only supports sources with an
executable PyTorch reference, an input constructor, and an entry-point class.
For supported sources it:

1. reuses the production parquet's exact Arrow schema and CUDA/TVM-FFI prompt
   prefix;
2. normalizes the effective entry-point class to `Model` without changing task
   behavior;
3. assigns a stable source-derived UUID; and
4. writes a row-level provenance sidecar with source identity and revision.

It intentionally does not wrap arbitrary CUDA/Triton text in a fake
`Model/get_inputs` shell. Such a wrapper would make the schema look compatible
without creating a correctness oracle or valid input distribution.

## Completed cleanup evidence

### CUDA-Agent-Ops-6K v4

The official dataset describes 6,000 synthesized, executable operator tasks and
an upstream pipeline that checks eager/compile execution, stochasticity,
degenerate outputs, runtime range, and KernelBench similarity. Our result does
not trust that claim blindly: all rows were converted and independently passed
through the v4 CPU reference-semantic contract. See the [official dataset
card](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K) and
[CUDA-Agent paper](https://arxiv.org/abs/2602.24286).

| V4 disposition or diagnostic | Rows |
| --- | ---: |
| Source and converted | 6,000 |
| Kept as a separate candidate | **3,979 (66.317%)** |
| Rejected | 2,021 |
| Quarantined subset | 30 |
| Runtime failure, primary | 1,216 |
| Within-source semantic duplicate, primary | 437 |
| Unused initialized state, primary | 113 |
| Exact semantic duplicate against main training, primary | 57 |
| Fixed `get_inputs()` values | 6; all rejected |
| Train/eval output difference | 295; diagnostic only |
| Train/eval state difference | 309; diagnostic only |

The six fixed-input cases were exhaustively read and were true fixed contracts. This handoff is the retained record for those external-source v4 results; the superseded standalone main-DrKernel v4 handoff has been removed.

### MultiKernelBench

The downloaded repository is MIT-licensed and its references span activation,
attention, convolution, matrix multiplication, indexing, broadcasting,
normalization, loss, math, pooling, optimization, and reduction. Of 401 source
files, 101 are NPU-only and were not misrepresented as CUDA tasks. The 300
converted NVIDIA/PyTorch rows produced:

| Earlier-contract primary result | Rows |
| --- | ---: |
| Kept | **212** |
| Timeout/runtime failure | 61 |
| Different natural input did not change output | 15 |
| Non-finite output | 7 |
| Same-input instability | 2 |
| Unused forward argument | 2 |
| Random forward | 1 |

This result predates v4's 20-call fixed-input test, effect-aware dead-code
classification, normalized `ops`, and train/eval diagnostics. Deeper comparison
also found that 164/212 survivors exactly match a KernelBench full AST, 168/212
match a KernelBench `Model` AST, and 171/212 have strict token Jaccard above
0.8. Per the 2026-07-27 project decision, MultiKernelBench is not a training
source; no v4 rerun or mixture work is planned.

### KernelBook

[KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook) contains 18,162
PyTorch/TorchInductor-Triton pairs from real repositories. That is the largest
source of independent-looking code structure found in this survey. Conversion
retained 18,153 rows. On 2026-07-27 all converted rows were rebuilt from the raw
conversion under the same v4 CPU semantic contract used for the main corpus and
CUDA-Agent. Exact semantic deduplication used the 71,996-row main source and the
3,979-row CUDA-Agent v4 survivor set; MultiKernelBench was not an ingestion
dependency.

| V4 primary disposition | Rows |
| --- | ---: |
| Kept as a separate candidate | **12,066 (66.468%)** |
| Rejected or quarantined | 6,087 |
| Static rejection | 5,063 |
| Exact semantic duplicate | 4,211 |
| Unused forward argument | 362 |
| Random forward | 179 |
| Unused initialized module/state | 177 |
| Other static rejection | 134 |
| Runtime/import failure | 472 |
| Input sensitivity inconclusive; quarantined | 186 |
| Synthetic sensitivity only; quarantined | 155 |
| Same-input output instability | 85 |
| Different natural/synthetic input did not change output | 52 |
| Non-finite output | 48 |
| Fixed `get_inputs()` values | 26 |
| Quarantined rows, including three stateful/RNG dead effects | 344 |

The job executed 13,090 acceptance tasks and 1,385 separate train/eval mode
diagnostics with 112 workers, one Torch thread per worker, a 60-second per-task
timeout, and 8,192-row batches. It completed in 2,973.9 seconds. Every one of
the 18,153 source rows has one audit decision; the output UUIDs equal the
12,066 `keep=true` decisions; quarantine UUIDs equal the 344 quarantined
decisions; the raw, kept, and quarantine Arrow schemas match; kept semantic
hashes are unique; and every kept row has runtime verdict `passed`.

Train/eval behavior remains diagnostic, as specified by v4. Among retained
rows, 813 have different train/eval outputs, 802 are stochastic only in train
mode, five change module state across modes, and 18 have inconclusive mode
evidence. Seventeen inconclusive cases use weight normalization that cannot be
deep-copied by the current mode harness; one mode diagnostic timed out while
its independent acceptance task passed.

A deterministic, category-stratified manual review read 49 concrete references.
Fixed-input, non-finite, input-insensitive, and same-input-unstable examples
were genuine. `floor(rand[0,1))` was a representative synthetic-only case: all
natural draws return zero, while a scaled probe changes the output. The review
also exposed conservative boundaries: dead computation includes harmless
discarded `x.size()` reads, and a random-noise branch can be masked by a
zero-initialized weight. The v4 artifact intentionally preserves those strict
rejections. Of the 472 runtime failures, 385 depend on an unbundled
`_paritybench_helpers._mock_config`; these rows are non-self-contained now but
could be recovered by a future provenance-preserving converter repair.

The 12,066 survivors come from 6,346 repositories; the largest repository has
37 rows (0.31%), so no single repository dominates the candidate. This does
not replace a repository-grouped train/validation split. The dataset-level
[June 9 Researcher Reciprocity License](https://huggingface.co/datasets/GPUMODE/KernelBook/blob/main/LICENSE)
also governs training use in addition to per-row source licenses. The clean v4
parquet is therefore a CPU-audited candidate, not permission to train.

## Newly verified sources not yet imported

### Reference-task and workload diversity

| Source | Verified content | Diversity value | Decision and conversion path |
| --- | --- | --- | --- |
| [FastKernels](https://github.com/Snowflake-AI-Research/fastkernels) | 407 local baseline modules: L1 116, L2 159, L3 78, L4 54; Apache-2.0 | Production-style model workloads and higher abstraction levels | Already pinned. Build a workload-capture adapter that records constructor values, args/kwargs, dtype/shape constraints, and oracle output; then run v4 and target-GPU checks. |
| [Meta TritonBench](https://github.com/meta-pytorch/tritonbench) | Maintained PyTorch custom operators with example inputs and production dependencies; BSD-3-Clause | FlashAttention, FBGEMM, Liger, CUTLASS, and other realistic operator families | Adapter seed, not an immediate parquet. Preserve an operator-level holdout and pin all submodules before capture. |
| [FlashInfer Trace](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace) | At revision `da915083...`: 190 definition files, 111 workload files, 393 solution files, and 196 trace files; Apache-2.0 | Modern inference shapes, quantization, attention, MoE, correctness, performance, and environment traces | Evaluation-first. Hold out definitions before any use; adapt only disjoint definitions or candidate traces and revalidate on the target stack. File counts are a dated inventory, not row counts. |
| [GPUMODE categorized Triton data](https://huggingface.co/datasets/GPUMODE/categorized_triton_data_permissive) | 864 GitHub Triton snippets with repository, path, commit, category, URL, and permissive per-row license; MIT wrapper | Real open-source kernel idioms and useful code/provenance diversity | High-value code SFT/RAG source. Run syntax/import checks, exact and fuzzy code deduplication, preserve per-row notices, and do not label it as RL reference data because it lacks inputs and PyTorch oracles. |

### Candidate kernels, reasoning, and optimization evidence

These sources may improve CUDA code style, iterative optimization, a verifier,
or within-problem ranking. They do not add one independent RL task per row.

| Source | Scale and evidence | Decision |
| --- | --- | --- |
| [GPUMODE KernelBot data](https://huggingface.co/datasets/GPUMODE/kernelbot-data) | 483,453 competition submissions, about 6.11 GB, with successful/deduplicated subsets and scores across AMD, NVIDIA NVFP4, PMPP, Trimul, Helion, and linear algebra | Valuable for successful/failed contrast pairs, user optimization trajectories, and within-problem ranking. It covers relatively few repeated problem identities; require Researcher Reciprocity approval, problem-level split, and correctness/timing replay. |
| [Makora Triton GPU latency](https://huggingface.co/datasets/makora-ai/triton-gpu-latency) | 544,028 train + 57,024 test candidate programs, roughly one third failed, mostly repeated KernelBench references | Use within problem for candidate SFT, failure classification, ranking, or latency modeling. Do not treat the 601,052 programs as 601,052 task references; hardware, timing units, warmup, and harness are undocumented. |
| [ConCuR](https://huggingface.co/datasets/lkongam/ConCuR), [paper](https://arxiv.org/abs/2510.07356) | Paper reports 4,892 curated PyTorch/reasoning/CUDA examples derived from KernelBook tasks | Promising reasoning SFT, but not downloaded: the repository has no dataset card or declared license, and it adds candidate solutions rather than independent reference diversity. |
| [CUDA Engineer Archive](https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive) | 30,615 KernelBench-derived candidate kernels; CC-BY-4.0 | Offline-RL/candidate-analysis evidence only. Do not import its KernelBench task references as training diversity. |
| [DrKernel cold-start 8K](https://huggingface.co/datasets/hkust-nlp/drkernel-coldstart-8k) | 8,920 trajectories; MIT; same upstream task family as the current corpus | Auxiliary trajectory SFT only; it does not provide independent task coverage. |
| [THUNLP TritonBench](https://github.com/thunlp/TritonBench) | 4,024 crawled + 4,133 synthetic training-support rows; Apache-2.0; already downloaded | Candidate-code SFT or retrieval after provenance, syntax, and dedup checks. No direct DrKernel RL conversion. |
| [KernelBook Triton reasoning traces](https://huggingface.co/datasets/ppbhatt500/kernelbook-triton-reasoning-traces) | 170 rows; Apache-2.0; its card reports about 15% incorrect kernels | Small diagnostic reasoning set only; exclude incorrect kernels and any benchmark-derived task references. |
| [CudaPerf paper](https://arxiv.org/abs/2607.20908) | Reports 2,903 C-to-CUDA and 1,013 PyTorch-to-CUDA programs, multiple valid candidates, and structural/performance labels | Potentially very valuable, but paper-only as of the survey cutoff: no official downloadable dataset or license was located. Monitor. |
| [DICE/CuKe paper](https://arxiv.org/abs/2602.11715) | Describes an augmented CUDA SFT dataset | Monitor. No independent official data release or license was located. |

### Evaluation-only holdouts

The following datasets are high quality precisely because they should remain
unseen. Downloading them is unnecessary for the current training build and
training on them would either violate the license or forfeit the benchmark.

| Source | Value | Hard boundary |
| --- | --- | --- |
| [NVIDIA SOL-ExecBench](https://huggingface.co/datasets/nvidia/SOL-ExecBench) | 235 real-model workloads spanning text, vision, speech, forward/backward, FP32/BF16/FP16/FP8/NVFP4, with symbolic axes and concrete workloads | NVIDIA's [Evaluation Dataset License](https://huggingface.co/datasets/nvidia/SOL-ExecBench/blob/63699402f003496acc3af4eb534a5304a8ac1ea9/LICENSE) limits use to internal evaluation/benchmarking and explicitly excludes training. Never ingest. |
| [NVIDIA ComputeEval](https://huggingface.co/datasets/nvidia/compute-eval) | 566 tasks in the current datapack covering kernels, runtime APIs, memory, parallel algorithms, and GPU libraries; human-reviewed | Same evaluation-only license; no training or redistribution. Reserve all versions. |
| [ParallelKernelBench Problems](https://huggingface.co/datasets/togethercomputer/ParallelKernelBench_Problems) | 87 deterministic multi-GPU tasks, normally 8xH100/BF16/five trials, including collectives, FSDP, tensor/expert/context parallelism, distributed FFT, GNN, and rendering workloads | Apache-2.0 permits reuse, but training would destroy a rare multi-GPU benchmark. Reserve all 87 unless the project explicitly accepts that cost. |
| [KernelBenchX](https://huggingface.co/datasets/BonnieWang/KernelBenchX) | 176 deterministic Triton-generation tasks across 15 categories plus 110 before/after repair records; Apache-2.0 | Hold out all 176 task references. The repair corpus is usable only if its task identities are excluded from the evaluation split. |
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | Maintained core Torch-to-CUDA/Triton benchmark and existing project evaluation target | Keep all selected validation/evaluation tasks out of training. V4 validation decontamination was explicitly disabled, so no clean artifact in this handoff carries a complete contamination guarantee. |

## Diversity assessment

The sources cover different axes and should not be collapsed into a single row
count.

| Diversity axis | Best available sources | Current gap |
| --- | --- | --- |
| Composed PyTorch operator tasks | CUDA-Agent-Ops-6K; current main corpus | Mostly synthetic operator composition; target-GPU difficulty still unmeasured. |
| Real-repository PyTorch structure | KernelBook | CPU v4 is complete; license approval, repository grouping, benchmark decontamination, and target-GPU validation remain outstanding. Compiler-generated Triton pairs may dominate a naive mixture. |
| Production model/operator structure | FastKernels; Meta TritonBench | Requires workload capture and dependency pinning. |
| Real inference shapes and low-precision workloads | FlashInfer Trace; SOL-ExecBench | FlashInfer needs definition-level split/adaptation; SOL is evaluation-only. |
| Multi-GPU collectives and parallel training | ParallelKernelBench | Only 87 tasks and should remain held out; no clean training analogue found. |
| Human/competition optimization trajectories | KernelBot | Many submissions but few repeated problem identities; license and hardware replay required. |
| Real open-source Triton idioms | GPUMODE categorized Triton; TritonBench crawl | Missing executable PyTorch oracle/input contract; suitable for auxiliary code training, not RL task rows. |
| Correct/incorrect candidate contrast | KernelBot; Makora; KernelBenchX repair corpus | Correctness and timing must be replayed; benchmark identities must be isolated. |
| C-to-CUDA transformation | CudaPerf paper | No public downloadable release found. |

For the already-clean CUDA-Agent candidate, the 3,979 survivors contain 394
normalized unique operator names and 3,338 unique operator signatures. The
source-family distribution is 3,369 `torch#2`, 302 `torch#3`, 110 `torch#1`,
92 `torch#4`, 64 `transformers`, and 42 `torch#5`. This is good composition
diversity, but it is not the same as real-model, distributed, backward,
low-precision, or production-shape diversity.

## Recommended ingestion plan

| Priority | Action | Acceptance gate | Why this order |
| --- | --- | --- | --- |
| P0 | Validate the 3,979 CUDA-Agent survivors on the target GPU in eager and `torch.compile`, then run a capped mixture ablation | Reproducible hardware/software manifest; compile and eager correctness; runtime distribution; attribution; explicit split decision; measurable training benefit | It is the smaller v4 candidate, has permissive CC-BY-4.0 terms, and is the cheapest controlled augmentation experiment. |
| Exclude | Do not ingest MultiKernelBench | Keep its UUIDs/source identities out of every training mixture | The user excluded it and most old survivors overlap KernelBench. |
| P0 | Obtain approval for KernelBook's dataset-level and row-level licenses, split the 12,066 v4 survivors by repository, decontaminate against complete KernelBench L1/L2/L3, and validate on the target GPU | Legal approval; no repository crosses splits; no selected benchmark near-duplicate; eager/compile correctness; mode-inconclusive rows handled explicitly; mixture cap | It is now the largest v4 CPU-audited source and the best real-repository diversity candidate. |
| P1 | Build one reusable workload-capture adapter for FastKernels and Meta TritonBench | Captured constructor, args/kwargs, dtype/shape constraints, deterministic replay, PyTorch oracle, stable provenance | Produces higher-value real workloads instead of padding the dataset with more synthetic operator combinations. |
| P1 | Import the 864 permissive Triton snippets as a separate code-SFT/RAG corpus | Syntax/import success, exact/fuzzy dedup, preserved source URL/commit/license, no benchmark files | Low storage cost and real code diversity, but deliberately separate from RL references. |
| P2 | Build candidate-ranking corpora from problem-grouped KernelBot/Makora rows | Problem-level split, correctness replay, target-hardware retiming, no cross-problem latency comparison | Useful for optimization behavior and verifiers, not task-count augmentation. |
| Hold | SOL-ExecBench, ComputeEval, ParallelKernelBench, KernelBenchX tasks, selected KernelBench tasks | Never included in train or retrieval indexes used during generation | Protects license compliance and credible evaluation. |
| Monitor | CudaPerf and CuKe releases; ConCuR license/card | Official artifact, immutable revision, usable license, provenance, and executable validation | Their papers/data appear valuable, but the present releases are incomplete for responsible ingestion. |

## Why the remaining public data was not downloaded blindly

1. Evaluation leakage is irreversible for a benchmark once its reference tasks
   enter pretraining, SFT, RL, or generation-time retrieval.
2. A candidate kernel plus a claimed latency is not an executable reference
   task. It requires problem grouping, a trusted oracle, correctness replay,
   hardware-aware retiming, and failure handling.
3. An open download endpoint does not imply training permission. ConCuR has no
   declared license; KernelBook/KernelBot have Researcher Reciprocity terms;
   NVIDIA evaluation datasets explicitly prohibit training.
4. Bulk repeated candidates can overwhelm a mixture without adding task
   diversity. KernelBot and Makora are valuable, but should be sampled within
   problem rather than weighted by raw row count.
5. Adapting production workloads is slower than copying text, but it is the
   path most likely to add missing backward, low-precision, dynamic-shape, and
   real-model coverage.

## Artifact integrity and reproduction

| Artifact | SHA-256 or status |
| --- | --- |
| `Data/external/sources/CUDA-Agent-Ops-6K/data.parquet` | `d4404fc181cad72081f8c1cba720cc1b511fb34dee924079e898bf1e83f6257c` |
| `Data/external/converted/cuda_agent_ops_6k.converted.parquet` | `d3e0e7cdc2a658ca9cca2c2adbf21f599a780adaccbc88c55130ccd52ab98066` |
| `Data/external/converted/cuda_agent_ops_6k.converted.provenance.jsonl` | `9d0099ea230db9bde3fb6273f2cba1ee96c7a911e5ffb41c7dbfbf5673795e43` |
| `Data/external/converted/cuda_agent_ops_6k.clean.v4.parquet` | `58f421c60b44ce873a47dec8bdabed5cbc9691f7467889a01c3b774fae9b853f` |
| `Data/external/converted/cuda_agent_ops_6k.quarantine.v4.parquet` | `4b860fa0d1b170ccd70988f6309467e44bae4d52e2ed038775bff1c96c6369c4` |
| `Data/external/converted/cuda_agent_ops_6k.clean.v4.audit.jsonl` | `0fe5690e93f6c2db77639ab018d14bbedc8bf4980d5f524879dc6806ae230ff1` |
| `Data/external/converted/multikernelbench.cuda_tasks.raw.parquet` | `95b4dbfe992263c10bfe3f60a9035673035fe28491c6766f6ba905695fbb5f08` |
| `Data/external/converted/multikernelbench.cuda_tasks.clean.parquet` | `fd8e9bfa62a4436a05ed06109e947d79b53f157d6ff0fb78bb4b3eab16e547cd` |
| `Data/external/converted/multikernelbench.cuda_tasks.raw.provenance.jsonl` | `68b20de095f0985377c67c3effafb9aac873fc3859f487962c12ba899822516f` |
| `Data/external/sources/KernelBook/dataset_permissive.parquet` | `64af2baa3c9a835dac85a5d0c772cc698610bccd5c63c44a56ebe0afaf5d9ed2` |
| `Data/external/converted/kernelbook.cuda_tasks.raw.parquet` | `67c2cd156bdd2e6320a84def15b4b18e1c05515eaa2d59b3c0df440774a9ea11` |
| `Data/external/converted/kernelbook.cuda_tasks.static_clean.parquet` | `b70d5aa1bdf3d2f7ad2cacd9d84c168ced1dc3d32f17bc903048ba6adc0b9a77` |
| `Data/external/converted/kernelbook.cuda_tasks.raw.provenance.jsonl` | `32b7852fddd8459aecdcc3f57722e79becb41441a0db6c69e0d1ffd1fcd73e11` |
| `Data/external/converted/kernelbook.cuda_tasks.clean.v4.parquet` | `71badf2b7e0f978cda138003898ddeed5ad408f7779725a2026d266e793a5aab` |
| `Data/external/converted/kernelbook.cuda_tasks.quarantine.v4.parquet` | `007a34f06d9fd00e2317f66988989d7962d2dd14729c3f74bb495520c0b5fe5a` |
| `Data/external/converted/kernelbook.cuda_tasks.clean.v4.audit.jsonl` | `83daaa1f942618f9fcb43e725bc0abf3d28a4cb64e942ecfb8bebb69fb755fb0` |
| `Data/external/converted/kernelbook.cuda_tasks.clean.v4.summary.json` | `216d45c7bc5b8e43be43a7680cd6ecd48d47999b67ad0e36f454e9b266cc385d` |

Recompute local hashes before publishing or moving an artifact:

```bash
sha256sum \
  Data/external/converted/cuda_agent_ops_6k.clean.v4.parquet \
  Data/external/converted/kernelbook.cuda_tasks.clean.v4.parquet \
  Data/external/converted/kernelbook.cuda_tasks.clean.v4.audit.jsonl
```

The supported conversion entry point is:

```bash
python -m tools.data.cleaning.external --help
```

The external v4 artifacts and hashes above remain authoritative for CUDA-Agent and KernelBook. External conversion and cleanup outputs must continue to be named and versioned separately; never overwrite the main clean parquet or silently append rows.

The command that produced the historical v4 hashes used the removed
incremental interface and remains recoverable from Git history. To apply the
current, stricter policy without overwriting those artifacts, run:

```bash
python -m tools.data.cleaning.pipeline clean \
  Data/external/converted/kernelbook.cuda_tasks.raw.parquet \
  Data/external/converted/kernelbook.cuda_tasks.clean.current.parquet \
  --dedup-against Data/prompt_tvm_v2/drkernel_rl_thinking.parquet \
  --dedup-against Data/external/converted/cuda_agent_ops_6k.clean.v4.parquet \
  --workers 4 --timeout 60 --overwrite
```

The current `--dedup-against` applies exact, token-Jaccard, and AST structural
checks together, so this command is intentionally not a reproduction of the
historical exact-only result.

## Evidence method

Local claims were recomputed from conversion summaries, cleanup summaries,
parquet counts, audit files, and direct SHA-256 calculation. Upstream facts were
checked against official GitHub repositories, Hugging Face dataset repositories,
license files, and primary papers on 2026-07-26; public repository heads were
recorded in `Data/external/SOURCES.json`. Representative source and reference
examples were manually read from CUDA-Agent, MultiKernelBench, KernelBook,
categorized Triton data, SOL-ExecBench, ParallelKernelBench, KernelBenchX, and
FlashInfer Trace. The 2026-07-27 KernelBook update additionally read 49
category-stratified kept, rejected, quarantined, and mode-diagnostic references.
Remote-only sources were not executed locally, so their cards' correctness or
performance claims remain upstream claims rather than our verification.

## Known limitations

- This is a survey and ingestion-status handoff, not a claim that any external
  mixture improves the trained model. That requires controlled ablations.
- CUDA-Agent is CPU-semantically clean under v4, not yet target-GPU certified.
- KernelBook is CPU-semantically clean under v4, but license approval,
  repository grouping, full benchmark decontamination, and target-GPU checks
  remain open. Eighteen retained rows have inconclusive train/eval diagnostics.
- MultiKernelBench is explicitly excluded from training; its old artifacts are
  retained only for reproducibility.
- Per the explicit v4 scope decision, validation-set contamination checks were
  not run for the current external v4 artifact.
- Remote-only inventory facts are pinned to the listed revisions or the
  2026-07-26 cutoff. Moving upstream datasets must be repinned before use.
- License observations are operational screening, not legal advice.

## Bottom line

The next training-data experiment is a **separate, capped blend of the 3,979
CUDA-Agent survivors after target-GPU validation**. The 12,066 KernelBook v4
survivors are the next and larger candidate only after license approval,
repository-grouped splitting, complete KernelBench decontamination, and target-
GPU validation. MultiKernelBench must not enter training. The strongest truly
new coverage—production workloads, multi-GPU behavior, low precision, and real
optimization trajectories—comes from FastKernels/Meta TritonBench adapters,
FlashInfer Trace, ParallelKernelBench, and KernelBot, but each belongs to a
different training or evaluation lane and should not be flattened into the
current RL-reference parquet.
