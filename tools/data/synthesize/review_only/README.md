# Shape + dtype final additive review protocol

This directory contains read-only review tooling for the final shape → dtype additive artifact.  Layout is explicitly deferred and is not included in the additive data, H20 post-selection gate, deterministic sample, or Kimi packet.  The tooling does not run a model, invoke Kimi, alter selected data, or alter the frozen synthesis pipeline

After the four dtype H20 shards have been copied back, audit their artifact lineage, source bindings, result payloads, and three-trial numerical evidence before selection:

```bash
python -m tools.data.synthesize.review_only.audit_dtype_runtime \
  "$RUN/dtype" --repo-root "$PWD" --require-complete \
  --output "$RUN/dtype/review_runtime_independent/audit.json"
```

Run only after `finalize_shape_dtype_expansion.py` has written `final_summary.json` and both H20 post-selection directories have been synchronized back to the artifact host.  `$RUN` must carry freezes produced from the same source revision; the output directory must not exist

```bash
python -m tools.data.synthesize.review_only.audit_csp_dag_final \
  --run-root "$RUN" \
  --base "$BASE" \
  --output-dir "$RUN/review_final_independent" \
  --prepare-kimi
```

The audit checks the exact runtime/finalization freezes, bound base-quality and dtype-runtime independent audits, selected passed subsequences, serial parent lineage, additive row conservation, global UUID/reference/AST collisions, and four-shard H20 post-selection coverage.  It does not rerun the validators whose source-bound results it consumes

Only `passed` records may carry H20/trial semantic payloads.  `unsupported` is a source-bound semantic rejection: it must use a nonempty `UnsupportedCase:` reason and exact identity/self-binding fields, and it must not claim GPU/trial/tolerance proof.  The result reports the dtype unsupported-reason histogram; failed, timeout, worker-protocol, and unknown statuses remain P1

The deterministic sample covers the ten tracked low-level families, code and operator-count buckets, all three lanes, safe scatter, SDPA, loss, and short/long code.  Fake class-count coverage is not a stratum.  A missing stratum is P1; cross-root near-neighbor evidence comes only from the base near-dedup audit, not from shape/dtype siblings

`--prepare-kimi` only writes the fixed 28-row packet, assignments, prompt hashes, and expected raw-output locations.  It intentionally has no model invocation path.  Start Kimi only after the final artifact-ready notification; save each response verbatim at the assigned `raw_output_path` and record its SHA-256 in `kimi_output_hashes.jsonl`

After all 28 responses have been saved, run the separate verify-only gate.  It checks the immutable 28-row assignment set, every prompt and raw-output path/SHA-256 binding, and every response before emitting a source-bound summary.  Each response must contain only one unfenced UTF-8 JSON object, with exactly `assignment_id`, `uuid`, `verdict`, `severity`, `reasons`, `duplicate_concern`, and `semantic_concern`; the first two fields must echo the assignment.  `reasons` is a nonempty string or a nonempty list of nonempty strings, and both concern fields are JSON booleans.  `PASS` requires `severity="none"` and both concerns false; `CONDITIONAL` requires `P2`; `FAIL` requires `P0` or `P1`

```bash
python -m tools.data.synthesize.review_only.verify_kimi_k3_packet \
  --packet-dir "$RUN/review_final_independent/kimi_k3_packet" \
  --review-summary "$RUN/review_final_independent/summary.json" \
  --output-dir "$RUN/review_kimi_final_verify"
```

The verify-only summary passes only when the packet is exactly the packet bound by the passing independent-review summary.  A `FAIL`, `P0`, or `P1` blocks the final gate.  `CONDITIONAL/P2` remains a reported non-blocking review concern

Near-dedup reporting has two scopes.  The source-bound base `near_dedup/retained.summary.json` remains the quantitative diversity result for independently synthesized roots.  The additive audit reports child normalized-AST collision groups separately: same root plus distinct shape/dtype lanes are intentional perturbation siblings and remain visible; any exact reference, UUID, or normalized-AST collision across roots is P1.  Do not rerun an aggregate child-level near-dedup filter or quote its apparent rate as generative diversity, because additive siblings are deliberately near one another
