# Shape semantic audit sample

结论和人工复核见 `SUMMARY.md`。

- Population: 63,979 runtime-valid children
- Sample: 264 unique children
- Lane kinds: {'model_shape': 224, 'static_solver': 40}
- Operator-family occurrences: {'activation': 79, 'attention_recurrent': 24, 'conv': 31, 'fft_sparse': 6, 'indexing_scatter': 27, 'loss_distance': 33, 'matmul_linear': 75, 'normalization': 25, 'other_only': 41, 'pooling': 20, 'reduction': 79, 'shape_layout': 97, 'sort_select': 11}
- Selection: deterministic heuristic-extreme anchors plus round-robin coverage of lane, source, variant, slot/factory count and operator family; this is not an equal-probability sample.
- This is a semantic audit, not a new runtime validation or training approval.
