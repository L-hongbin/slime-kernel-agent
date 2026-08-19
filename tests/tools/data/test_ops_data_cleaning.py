from __future__ import annotations

import json
import subprocess
import sys
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import torch

NUM_GPUS = 0

_REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from tools.data.cleaning.ast_similarity import (
    _maximum_assignment_sum,
    _significant_trees,
    ast_structure_similarity,
    read_ast_similarity_baselines,
)
from tools.data.cleaning.audit_summary import main as audit_summary_main
from tools.data.cleaning.complexity import (
    KernelBenchKNNClassifier,
    KernelBenchTaxonomyClassifier,
    deterministic_level1_selection,
    extract_complexity_features,
    extract_operator_signature,
    feature_dict,
    level1_cap_candidate_bases,
)
from tools.data.cleaning.external import _accepted_jsonl_rows, normalize_entry_point
from tools.data.cleaning.pipeline import (
    CURRENT_POLICY,
    CleanupConfig,
    _model_ops_metadata,
    _read_semantic_baselines,
    _runtime_failure_category,
    _write_filtered_parquet,
    infer_effective_entry_point,
    parse_args,
    run_cleanup,
)
from tools.data.cleaning.runtime_validation import (
    audit_train_eval_modes_detailed,
    validate_ops_text,
    validate_ops_text_detailed,
)
from tools.data.cleaning.shards import main as shards_main
from tools.data.cleaning.similarity import best_token_jaccard_match, python_token_set, token_jaccard
from tools.data.cleaning.static_analysis import analyze_reference, canonicalize_reference
from tools.data.cleaning.subsets import align_records_by_effective_uuid, select_record

_CLEANER_COMMAND = (sys.executable, "-m", "tools.data.cleaning.pipeline")

_HEADER = """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
"""


class ComplexityStratificationTest(unittest.TestCase):
    def test_level1_cap_includes_single_op_behind_identity_despite_level3_knn(self) -> None:
        code = (
            _HEADER
            + """
        self.identity = nn.Identity()

    def forward(self, x):
        x = self.identity(x)
        return torch.amax(x, dim=1)
"""
        )
        features = extract_complexity_features(code)
        signature = extract_operator_signature(code)
        prediction = mock.Mock(level="level3", structural_level="level3")

        self.assertEqual(signature, ("nn.Identity", "torch.amax"))
        self.assertEqual(
            level1_cap_candidate_bases(features, signature, prediction),
            ("single_compute_after_explicit_noop_removal",),
        )

    def test_features_ignore_get_inputs_and_measure_model_composition(self) -> None:
        single = (
            _HEADER
            + """
    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    a = torch.randn(8)
    b = torch.rand(8)
    return [a + b]
"""
        )
        fused = (
            _HEADER
            + """
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        x = self.linear(x)
        return torch.relu(x) + x

def get_inputs():
    return [torch.randn(8)]
"""
        )
        single_features = feature_dict(extract_complexity_features(single))
        fused_features = feature_dict(extract_complexity_features(fused))
        self.assertEqual(single_features["forward_call_count"], 1)
        self.assertEqual(single_features["init_nn_constructor_count"], 0)
        self.assertGreater(fused_features["forward_call_count"], single_features["forward_call_count"])
        self.assertEqual(fused_features["init_nn_constructor_count"], 1)
        self.assertEqual(extract_operator_signature(single), ("torch.relu",))
        self.assertEqual(
            extract_operator_signature(fused),
            ("nn.Linear", "operator.add", "torch.relu"),
        )

    def test_knn_prediction_and_leave_one_out_are_deterministic(self) -> None:
        features = [
            (1, 1, 0, 0, 2, 0, 0, 1, 5, 20),
            (1, 1, 1, 1, 2, 0, 0, 1, 5, 22),
            (3, 3, 2, 2, 6, 0, 1, 1, 7, 35),
            (4, 4, 2, 3, 7, 0, 1, 1, 8, 38),
            (8, 7, 4, 6, 12, 1, 2, 3, 12, 90),
            (10, 9, 6, 8, 15, 1, 3, 4, 14, 120),
        ]
        labels = ["level1", "level1", "level2", "level2", "level3", "level3"]
        classifier = KernelBenchKNNClassifier(features, labels, [f"r{i}" for i in range(6)], neighbors=1)
        self.assertEqual(classifier.predict(features[0]).level, "level1")
        self.assertEqual(classifier.predict(features[-1]).level, "level3")
        report = classifier.leave_one_out_report()
        self.assertEqual(report["rows"], 6)
        self.assertEqual(sum(sum(row.values()) for row in report["confusion"].values()), 6)

    def test_taxonomy_gate_does_not_call_arbitrary_short_fusion_level1(self) -> None:
        features = [
            (1, 1, 0, 0, 2, 0, 0, 1, 5, 20),
            (2, 2, 0, 0, 2, 0, 0, 1, 6, 22),
            (3, 3, 1, 2, 5, 0, 1, 1, 7, 35),
            (8, 7, 4, 6, 12, 1, 2, 3, 12, 90),
        ]
        labels = ["level1", "level1", "level2", "level3"]
        signatures = [
            ("torch.relu",),
            ("operator.sub", "torch.clamp"),
            ("nn.Linear", "operator.add", "torch.relu"),
            tuple(f"op{i}" for i in range(8)),
        ]
        classifier = KernelBenchTaxonomyClassifier(
            features, signatures, labels, [f"r{i}" for i in range(4)], neighbors=1
        )
        single = classifier.predict(features[0], ("torch.tan",))
        known_composite = classifier.predict(features[1], signatures[1])
        arbitrary_fusion = classifier.predict(features[1], ("torch.tan", "torch.std"))
        self.assertEqual((single.level, single.basis), ("level1", "single_compute_signature"))
        self.assertEqual((known_composite.level, known_composite.basis), ("level1", "known_level1_signature"))
        self.assertEqual((arbitrary_fusion.level, arbitrary_fusion.basis), ("level2", "structural_knn_non_level1"))

    def test_operator_signature_counts_inline_modules_and_comprehension_bodies(self) -> None:
        code = (
            _HEADER
            + """
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        x = nn.SiLU()(x)
        return torch.stack([self.linear(xi) for xi in x])
"""
        )
        self.assertEqual(
            extract_operator_signature(code),
            ("nn.Linear", "nn.SiLU", "torch.stack"),
        )

    def test_level1_cap_preserves_non_level1_and_source_order_decisions(self) -> None:
        records = [
            (0, "a", "level1"),
            (1, "b", "level1"),
            (2, "c", "level1"),
            (3, "d", "level1"),
            (4, "e", "level2"),
            (5, "f", "level3"),
        ]
        selected, report = deterministic_level1_selection(records, max_fraction=0.30, seed=7)
        self.assertTrue({4, 5}.issubset(selected))
        self.assertEqual(report["output_level1_rows"], 0)
        self.assertLessEqual(report["output_level1_fraction"], 0.30)
        self.assertTrue(report["cap_applied"])


class StaticAnalysisTest(unittest.TestCase):
    def test_normalizes_ops_from_effective_model_not_stale_metadata(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.weight = nn.Parameter(torch.randn(2, 2))

    def forward(self, x):
        x = torch.softmax(x, dim=1)
        x = self.pool(x)
        return x @ self.weight

def get_inputs():
    return [torch.randn(1, 2, 4, 4)]
"""
        )
        self.assertEqual(
            json.loads(_model_ops_metadata(code, "Model")),
            ["nn.MaxPool2d", "torch.softmax"],
        )

    def test_kwargs_is_a_valid_dynamic_forward_input(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, **kwargs):
        return kwargs["x"] * 2

def get_inputs():
    return {"x": torch.ones(2)}
"""
        )
        result = analyze_reference(code, "Model")
        self.assertNotIn("forward_has_no_inputs", result.fatal_reasons)
        self.assertEqual(validate_ops_text(code, gpu_fallback=False), "fixed_input_values")

    def test_varargs_are_tracked_as_input_dependencies(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, *args):
        return torch.stack(args)

def get_inputs():
    return [torch.ones(2), torch.zeros(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertEqual(result.fatal_reasons, ())
        self.assertIn("forward_output_dependency:dependent", result.flags)

    def test_normalizes_external_entry_point_and_explicit_super(self) -> None:
        code = """import torch
class ExternalOp(torch.nn.Module):
    def __init__(self):
        super(ExternalOp, self).__init__()
    def forward(self, x):
        return x + 1
def get_inputs():
    return [torch.ones(2)]
def get_init_inputs():
    return []
"""
        normalized = normalize_entry_point(code, "ExternalOp")
        self.assertIn("class Model", normalized)
        self.assertIn("super(Model, self)", normalized)
        self.assertEqual(analyze_reference(normalized, "Model").fatal_reasons, ())

    def test_converts_internal_accepted_jsonl_with_incomplete_provenance(self) -> None:
        item = {
            "attempt": 17,
            "code": _HEADER
            + """
    def forward(self, x):
        return x + 1

def get_inputs():
    return [torch.randn(2)]

def get_init_inputs():
    return []
""",
            "data_source": "llm_cuda_agent#2",
            "filter": {"reason": "accepted"},
            "ops": '["torch.add"]',
            "repair_round": 1,
            "requested_ops": ["torch.add"],
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "accepted.jsonl"
            path.write_text(json.dumps(item) + "\n", encoding="utf-8")
            rows, provenance, summary = _accepted_jsonl_rows(path, "prefix\n", "delivery-1")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["data_source"], "cuda_llm_internal_accepted_ops")
        self.assertEqual(rows[0]["extra_info"]["level"], "llm_cuda_agent#2")
        self.assertTrue(rows[0]["extra_info"]["uuid"].startswith("accepted_ops_"))
        self.assertEqual(provenance[0]["source_requested_ops"], ["torch.add"])
        self.assertEqual(provenance[0]["source_filter"], {"reason": "accepted"})
        self.assertEqual(summary["source_rows"], 1)
        self.assertIn("license absent", summary["provenance_status"])

    def test_finds_dead_calls_before_identical_terminal_branch_returns(self) -> None:
        code = (
            _HEADER
            + """
        self.other = nn.Linear(2, 2)

    def forward(self, x):
        out = x + 1
        target = self.other(x)
        loss = torch.nn.functional.mse_loss(out, target)
        if torch.compiler.is_compiling():
            return out
        else:
            return out

def get_inputs():
    return [torch.randn(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("dead_forward_computation", result.fatal_reasons)
        self.assertTrue(any("mse_loss" in flag for flag in result.flags))

    def test_rejects_unseeded_fractional_max_pool_module(self) -> None:
        code = (
            _HEADER
            + """
        self.pool = nn.FractionalMaxPool2d(2, output_ratio=(0.5, 0.5))

    def forward(self, x):
        return self.pool(x)

def get_inputs():
    return [torch.randn(2, 3, 8, 8)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("random_forward", result.fatal_reasons)
        self.assertIn("random_forward_call:self.pool", result.flags)

    def test_allows_fractional_max_pool_with_explicit_random_samples(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x, random_samples):
        return torch.nn.functional.fractional_max_pool2d(
            x, 2, output_ratio=(0.5, 0.5), _random_samples=random_samples
        )

def get_inputs():
    return [torch.randn(2, 3, 8, 8), torch.rand(2, 3, 2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertNotIn("random_forward", result.fatal_reasons)

    def test_rejects_execution_mode_dependent_forward(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        if torch.compiler.is_compiling():
            return torch.relu(x)
        return torch.sigmoid(x)

def get_inputs():
    return [torch.randn(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("execution_mode_dependent_forward", result.fatal_reasons)
        self.assertIn(
            "execution_mode_dependent_forward:torch.compiler.is_compiling",
            result.flags,
        )

    def test_rejects_filesystem_serialization_in_reference(self) -> None:
        code = """import tempfile
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.value = torch.load(path)
    def forward(self, x):
        return x + self.value

def get_init_inputs():
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        torch.save(torch.ones(2), handle.name)
        return [handle.name]

def get_inputs():
    return [torch.randn(2)]
"""
        result = analyze_reference(code, "Model")
        self.assertIn("unsafe_reference_code", result.fatal_reasons)
        self.assertTrue(any(flag.startswith("unsafe_call:tempfile.NamedTemporaryFile") for flag in result.flags))
        self.assertTrue(any(flag.startswith("unsafe_filesystem_serialization:torch.load") for flag in result.flags))

    def test_allows_in_memory_torch_serialization(self) -> None:
        code = """import io
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.buffer = io.BytesIO()
        torch.save(torch.ones(2), self.buffer)
        self.buffer.seek(0)
        self.value = torch.load(self.buffer, weights_only=True)
    def forward(self, x):
        buffer = io.BytesIO()
        torch.save(x, buffer)
        buffer.seek(0)
        return torch.load(buffer, weights_only=True) + self.value

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(2)]
"""
        result = analyze_reference(code, "Model")
        self.assertNotIn("unsafe_reference_code", result.fatal_reasons)

    def test_rejects_unused_input(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x, ignored):
        return x + 1

def get_inputs():
    return [torch.ones(2), torch.ones(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("unused_forward_argument", result.fatal_reasons)
        self.assertIn("unused_forward_arg:ignored", result.flags)

    def test_rejects_random_forward(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return x + torch.randn_like(x)

def get_inputs():
    return [torch.ones(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("random_forward", result.fatal_reasons)

    def test_distribution_statistics_are_not_random_forward(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        distribution = torch.distributions.Bernoulli(probs=torch.sigmoid(x))
        return x * distribution.mean

def get_inputs():
    return [torch.randn(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertNotIn("random_forward", result.fatal_reasons)

    def test_distribution_sampling_is_random_forward(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        distribution = torch.distributions.Bernoulli(probs=torch.sigmoid(x))
        return x * distribution.sample()

def get_inputs():
    return [torch.randn(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("random_forward", result.fatal_reasons)
        self.assertIn("random_forward_call:distribution.sample", result.flags)

    def test_explicitly_seeded_generator_is_deterministic(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        generator = torch.Generator(device=x.device)
        generator.manual_seed(0)
        return x + torch.rand(x.shape, generator=generator, device=x.device)

def get_inputs():
    return [torch.randn(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertNotIn("random_forward", result.fatal_reasons)
        self.assertIn("deterministic_seeded_forward_rng:torch.rand", result.flags)

    def test_explicitly_seeded_functional_rng_is_deterministic(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        torch.manual_seed(0)
        x = F.dropout(x, training=True)
        return F.gumbel_softmax(x)

def get_inputs():
    return [torch.randn(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertNotIn("random_forward", result.fatal_reasons)
        self.assertIn("deterministic_seeded_forward_rng:F.dropout", result.flags)
        self.assertIn("deterministic_seeded_forward_rng:F.gumbel_softmax", result.flags)

    def test_conditional_unknown_seed_does_not_hide_randomness(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed
    def forward(self, x):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        return torch.multinomial(x, 1)

def get_init_inputs():
    return [None]

def get_inputs():
    return [torch.rand(4)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("random_forward", result.fatal_reasons)

    def test_conditional_known_seed_is_deterministic(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed
    def forward(self, x):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        return torch.multinomial(x, 1)

def get_init_inputs():
    return [42]

def get_inputs():
    return [torch.rand(4)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertNotIn("random_forward", result.fatal_reasons)
        self.assertIn("deterministic_seeded_forward_rng:torch.multinomial", result.flags)

    def test_tracks_local_inplace_output_dependency(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x, index):
        output = torch.zeros_like(x)
        output.scatter_(0, index, x)
        return output

def get_inputs():
    return [torch.randn(2), torch.tensor([1, 0])]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertNotIn("forward_output_independent", result.fatal_reasons)
        self.assertIn("forward_output_dependency:dependent", result.flags)

    def test_rejects_forward_tensor_serialization_even_in_memory(self) -> None:
        code = """import io
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        buffer = io.BytesIO()
        torch.save(x, buffer)
        return x + 1

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(2)]
"""
        result = analyze_reference(code, "Model")
        self.assertIn("non_kernel_forward_serialization", result.fatal_reasons)

    def test_rejects_unconditional_functional_dropout(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return F.dropout(x)

def get_inputs():
    return [torch.ones(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("random_forward", result.fatal_reasons)
        self.assertIn("functional_dropout:F.dropout:p=0.5", result.flags)

    def test_rejects_aliased_unconditional_functional_dropout(self) -> None:
        code = """import torch
import torch.nn as nn
from torch.nn.functional import dropout as stochastic_mask

class Model(nn.Module):
    def forward(self, x):
        return stochastic_mask(x, p=0.25, training=True)

def get_inputs():
    return [torch.ones(2)]
"""
        result = analyze_reference(code, "Model")
        self.assertIn("random_forward", result.fatal_reasons)
        self.assertIn("random_forward_call:stochastic_mask", result.flags)

    def test_functional_dropout_respects_training_false(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return F.dropout(x, p=0.5, training=False)

def get_inputs():
    return [torch.ones(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertNotIn("random_forward", result.fatal_reasons)
        self.assertIn("functional_dropout:F.dropout:disabled", result.flags)

    def test_functional_dropout_resolves_zero_constructor_probability(self) -> None:
        code = """import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        return F.dropout2d(x, p=self.probability)

def get_inputs():
    return [torch.ones(2, 2)]

def get_init_inputs():
    probability = 0.0
    return [probability]
"""
        result = analyze_reference(code, "Model")
        self.assertNotIn("random_forward", result.fatal_reasons)
        self.assertIn("functional_dropout:F.dropout2d:degenerate_p=0.0", result.flags)

    def test_functional_dropout_with_model_mode_is_only_a_mode_candidate(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return F.dropout(x, p=0.5, training=self.training)

def get_inputs():
    return [torch.ones(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertNotIn("random_forward", result.fatal_reasons)
        self.assertIn("functional_dropout:F.dropout:dynamic_training", result.flags)
        self.assertIn("train_eval_static_candidate:explicit_self_training", result.flags)

    def test_rejects_zero_input_forward(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self):
        return torch.ones(2)

def get_inputs():
    return []
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("forward_has_no_inputs", result.fatal_reasons)

    def test_rejects_placeholder_identity_forward(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        # Placeholder: the real implementation was never supplied.
        return x

def get_inputs():
    return [torch.ones(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("identity_forward", result.fatal_reasons)
        self.assertIn("identity_forward_arg:x", result.flags)

    def test_rejects_dead_forward_call_overwritten_before_return(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        mask = torch.ge(x, 0)
        x = torch.gather(x, 0, torch.zeros_like(x, dtype=torch.long))
        x = torch.logical_not(mask)
        return x

def get_inputs():
    return [torch.ones(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("dead_forward_computation", result.fatal_reasons)
        self.assertTrue(
            any(flag.endswith("calls=torch.gather,torch.zeros_like") for flag in result.flags),
            result.flags,
        )

    def test_keeps_all_live_forward_assignments(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        mask = torch.ge(x, 0)
        gathered = torch.gather(x, 0, torch.zeros_like(x, dtype=torch.long))
        return torch.where(mask, gathered, x)

def get_inputs():
    return [torch.ones(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertNotIn("dead_forward_computation", result.fatal_reasons)

    def test_dead_binding_of_in_place_result_is_not_rejected(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        ignored_alias = x.add_(1)
        return x

def get_inputs():
    return [torch.ones(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertNotIn("dead_forward_computation", result.fatal_reasons)

    def test_dead_batch_norm_is_classified_as_stateful_not_pure(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x, replacement):
        x = self.bn(x)
        x = replacement + 1
        return x

def get_inputs():
    return [torch.randn(8, 4), torch.randn(8, 4)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("dead_forward_stateful_effect", result.fatal_reasons)
        self.assertNotIn("dead_forward_computation", result.fatal_reasons)

    def test_dead_dropout_is_classified_as_rng_effect(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, replacement):
        x = self.dropout(x)
        x = replacement + 1
        return x

def get_inputs():
    return [torch.randn(8, 4), torch.randn(8, 4)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("dead_forward_rng_effect", result.fatal_reasons)
        self.assertNotIn("dead_forward_computation", result.fatal_reasons)

    def test_dead_default_attention_is_pure_when_dropout_is_zero(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(8, 2)

    def forward(self, query, key, value, replacement):
        output, _ = self.attention(query, key, value)
        output = replacement + 1
        return output

def get_inputs():
    return [torch.randn(3, 2, 8), torch.randn(3, 2, 8), torch.randn(3, 2, 8), torch.randn(3, 2, 8)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("dead_forward_computation", result.fatal_reasons)
        self.assertNotIn("dead_forward_unknown_effect", result.fatal_reasons)

    def test_dead_attention_with_dynamic_dropout_has_unknown_effect(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(8, 2, dropout=dropout)

    def forward(self, query, key, value, replacement):
        output, _ = self.attention(query, key, value)
        output = replacement + 1
        return output

def get_inputs():
    return [torch.randn(3, 2, 8), torch.randn(3, 2, 8), torch.randn(3, 2, 8), torch.randn(3, 2, 8)]
def get_init_inputs():
    return [0.5]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("dead_forward_unknown_effect", result.fatal_reasons)
        self.assertNotIn("dead_forward_computation", result.fatal_reasons)

    def test_rejects_initialized_module_unreachable_from_forward(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self):
        super().__init__()
        self.used = nn.ReLU()
        self.abandoned = nn.TripletMarginLoss()

    def forward(self, x):
        return self.used(x)

def get_inputs():
    return [torch.randn(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("unused_forward_module", result.fatal_reasons)
        self.assertIn("unused_forward_module:abandoned:nn.TripletMarginLoss", result.flags)
        self.assertIn("unused_forward_loss_module:abandoned:nn.TripletMarginLoss", result.flags)

    def test_marks_train_eval_static_candidates(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return self.dropout(x) if self.training else x

def get_inputs():
    return [torch.randn(8, 4)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("train_eval_static_candidate:dropout", result.flags)
        self.assertIn("train_eval_static_candidate:explicit_self_training", result.flags)

    def test_module_used_by_reachable_helper_is_not_rejected(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def helper(self, x):
        return self.relu(x)

    def forward(self, x):
        return self.helper(x)

def get_inputs():
    return [torch.randn(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertNotIn("unused_forward_module", result.fatal_reasons)

    def test_modules_composed_into_used_container_are_not_rejected(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.sequence = nn.Sequential(self.relu, self.sigmoid)

    def forward(self, x):
        return self.sequence(x)

def get_inputs():
    return [torch.randn(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertNotIn("unused_forward_module", result.fatal_reasons)

    def test_dynamic_module_traversal_is_left_unclassified(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        for child in self.children():
            x = child(x)
        return x

def get_inputs():
    return [torch.randn(2)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertNotIn("unused_forward_module", result.fatal_reasons)

    def test_rejects_global_read_instead_of_same_named_init_argument(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        return torch.sum(x, dim=dim)

dim = 1

def get_inputs():
    return [torch.randn(2, 2)]

def get_init_inputs():
    return [dim]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("forward_uses_global_instead_of_init_argument", result.fatal_reasons)
        self.assertIn("forward_uses_global_instead_of_init_argument:dim", result.flags)

    def test_same_named_forward_argument_is_not_misclassified_as_global(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self, dim):
        super().__init__()

    def forward(self, x, dim):
        return torch.sum(x, dim=dim)

def get_inputs():
    return [torch.randn(2, 2), 1]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertNotIn("forward_uses_global_instead_of_init_argument", result.fatal_reasons)

    def test_rejects_random_scalar_model_initialization(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.sum(dim=self.dim)

def get_inputs():
    return [torch.ones(2, 2)]

def get_init_inputs():
    return [torch.randint(0, 2, (1,)).item()]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("random_init_scalar", result.fatal_reasons)
        self.assertIn("random_init_scalar_call:torch.randint", result.flags)

    def test_rejects_duplicate_contract_definitions(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return x + 1

    def forward(self, x):
        return x * 2

def get_inputs():
    return [torch.ones(2)]

def get_inputs():
    return [torch.ones(3)]

def get_init_inputs():
    return []
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("duplicate_forward_definition", result.fatal_reasons)
        self.assertIn("duplicate_get_inputs_definition", result.fatal_reasons)

    def test_canonicalizes_last_contract_definitions_before_analysis(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return x + 1

    def forward(self, x):
        return x * 2

def get_inputs():
    return [torch.ones(2)]

def get_inputs():
    return [torch.ones(3)]

def get_init_inputs():
    return []
"""
        )
        canonical, flags = canonicalize_reference(code, "Model")
        self.assertIn("canonicalized_duplicate_forward_definition", flags)
        self.assertIn("canonicalized_duplicate_get_inputs_definition", flags)
        self.assertNotIn("duplicate_forward_definition", analyze_reference(canonical, "Model").fatal_reasons)
        self.assertEqual(canonical.count("def forward"), 1)
        self.assertEqual(canonical.count("def get_inputs"), 1)
        self.assertIn("return x * 2", canonical)
        self.assertIn("torch.ones(3)", canonical)

    def test_materializes_supported_random_init_scalars(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self, dim, scale):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        return x.sum(dim=self.dim) * self.scale

def get_inputs():
    return [torch.ones(2, 2)]

def get_init_inputs():
    return [torch.randint(0, 2, (1,)).item(), torch.randn(1).item()]
"""
        )
        canonical, flags = canonicalize_reference(code, "Model")
        self.assertIn("canonicalized_random_init_scalar:torch.randint", flags)
        self.assertIn("canonicalized_random_init_scalar:torch.randn", flags)
        self.assertNotIn("random_init_scalar", analyze_reference(canonical, "Model").fatal_reasons)
        self.assertIn("return [0, 1.0]", canonical)

    def test_rejects_proven_input_independent_output(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        x = torch.linspace(0, 1, 4)
        return x

def get_inputs():
    return [torch.ones(4)]
"""
        )
        result = analyze_reference(code, "Model")
        self.assertIn("forward_output_independent", result.fatal_reasons)

    def test_repairs_helper_entry_point_metadata(self) -> None:
        code = """import torch
class Mish(torch.nn.Module):
    def forward(self, x):
        return x
class Model(torch.nn.Module):
    def forward(self, x):
        return x + 1
def get_inputs():
    return [torch.ones(1)]
"""
        row = {"prompt": [{"role": "user", "content": f"Return class ModelNew.\n{code}"}]}
        effective, repairs = infer_effective_entry_point(code, row, "Mish")
        self.assertEqual(effective, "Model")
        self.assertEqual(repairs, {"entry_point": "Model", "module_name": "Model"})


class DynamicValidationTest(unittest.TestCase):
    def test_parameter_container_output_is_compared_by_contents(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self):
        super().__init__()
        self.parameters_out = nn.ParameterDict({
            "weight": nn.Parameter(torch.randn(4)),
        })
        self.parameters_list_out = nn.ParameterList([
            nn.Parameter(torch.randn(4)),
        ])

    def forward(self, x):
        return x * 2, self.parameters_out, self.parameters_list_out

def get_inputs():
    return [torch.randn(4)]

def get_init_inputs():
    return []
"""
        )
        result = validate_ops_text_detailed(code, validation_seeds=1)
        self.assertEqual(result["verdict"], "passed")

    def test_aligns_cleaned_subsets_by_repaired_effective_uuid(self) -> None:
        records = [
            {"row_index": 0, "uuid": "source-a", "repairs": {"uuid": "clean-a"}},
            {"row_index": 1, "uuid": "source-b", "repairs": {}},
        ]
        aligned = align_records_by_effective_uuid(records, ["source-b", "clean-a"])
        self.assertEqual([record["row_index"] for record in aligned], [1, 0])
        with self.assertRaisesRegex(ValueError, "absent from audit"):
            align_records_by_effective_uuid(records, ["unknown"])
        with self.assertRaisesRegex(ValueError, "duplicate UUIDs"):
            align_records_by_effective_uuid(records, ["clean-a", "clean-a"])

    def test_audit_subset_selector_separates_timeout_and_mode_flags(self) -> None:
        timeout = {
            "keep": False,
            "runtime_verdict": "Failed",
            "runtime_detail": "validation seed 2 failed: timeout_after_60s",
            "mode_flags": [],
        }
        timeout_args = Namespace(
            keep_status="rejected",
            runtime_verdict=["Failed"],
            failure_category=["timeout"],
            require_any_mode_flag=[],
            exclude_mode_flag=[],
        )
        self.assertTrue(select_record(timeout, timeout_args))

        mode_variant = {
            "keep": True,
            "runtime_verdict": "passed",
            "runtime_detail": "passed",
            "mode_flags": ["train_eval_output_diff", "train_only_stochastic"],
        }
        stable_args = Namespace(
            keep_status="kept",
            runtime_verdict=[],
            failure_category=[],
            require_any_mode_flag=[],
            exclude_mode_flag=["train_eval_output_diff", "train_only_stochastic"],
        )
        self.assertFalse(select_record(mode_variant, stable_args))

    def test_quarantines_shape_only_forward_argument(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x, shape_source):
        return x.view(shape_source.shape)

def get_inputs():
    return [torch.randn(4), torch.randn(2, 2)]

def get_init_inputs():
    return []
"""
        )
        result = validate_ops_text_detailed(
            code,
            require_each_forward_input_sensitive=True,
        )
        self.assertEqual(result["verdict"], "forward_argument_sensitivity_inconclusive")
        self.assertIn("positional:1", result["detail"])

    def test_each_keyword_forward_argument_can_be_proven_sensitive(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, *, x, y):
        return x + y

def get_inputs():
    return {"x": torch.randn(8), "y": torch.randn(8)}

def get_init_inputs():
    return []
"""
        )
        result = validate_ops_text_detailed(
            code,
            validation_seeds=2,
            require_each_forward_input_sensitive=True,
        )
        self.assertEqual(result["verdict"], "passed")

    def test_mode_audit_rebuilds_weight_norm_model_when_deepcopy_fails(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.utils.weight_norm(nn.Linear(4, 4))

    def forward(self, x):
        return self.linear(x)

def get_inputs():
    return [torch.randn(2, 4)]

def get_init_inputs():
    return []
"""
        )
        result = audit_train_eval_modes_detailed(code)
        self.assertEqual(result["verdict"], "mode_audit_complete")
        self.assertEqual(result["mode_flags"], ["train_eval_same"])

    def test_mode_audit_handles_uninitialized_lazy_module_state(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self):
        super().__init__()
        self.linear = nn.LazyLinear(4)

    def forward(self, x):
        return self.linear(x)

def get_inputs():
    return [torch.randn(2, 3)]

def get_init_inputs():
    return []
"""
        )
        result = audit_train_eval_modes_detailed(code)
        self.assertEqual(result["verdict"], "mode_audit_complete")
        self.assertEqual(result["mode_flags"], ["train_eval_same"])

    def test_multi_seed_failure_category_preserves_underlying_exception(self) -> None:
        self.assertEqual(
            _runtime_failure_category("validation seed 1 failed: NameError: missing_name"),
            "NameError",
        )
        self.assertEqual(
            _runtime_failure_category("validation seed 2 failed: timeout_after_60s"),
            "timeout",
        )

    def test_identity_passes(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return x * 2

def get_inputs():
    return [torch.randn(4)]
"""
        )
        self.assertEqual(validate_ops_text(code, gpu_fallback=False), "passed")

    def test_constant_output_is_rejected(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return torch.ones_like(x)

def get_inputs():
    return [torch.arange(4, dtype=torch.float32)]
"""
        )
        self.assertEqual(
            validate_ops_text(code, gpu_fallback=False),
            "fixed_input_values",
        )

    def test_random_output_is_rejected(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return torch.rand_like(x)

def get_inputs():
    return [torch.zeros(32)]
"""
        )
        self.assertEqual(validate_ops_text(code, gpu_fallback=False), "no_same_output")

    def test_three_same_input_calls_catch_delayed_state_instability(self) -> None:
        code = (
            _HEADER
            + """
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x if self.calls < 3 else x + 1

def get_inputs():
    return [torch.zeros(4)]
"""
        )
        self.assertEqual(validate_ops_text(code, gpu_fallback=False), "no_same_output")

    def test_large_probe_crosses_locally_constant_argsort_region(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return torch.argsort(x)

def get_inputs():
    return [torch.tensor([1.0, 2.0, 3.0])]
"""
        )
        self.assertEqual(validate_ops_text(code, gpu_fallback=False), "fixed_input_values")

    def test_range_preserving_probe_keeps_positive_domain_valid(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return torch.rsqrt(x)

def get_inputs():
    return [torch.tensor([1.0, 2.0, 3.0])]
"""
        )
        self.assertEqual(validate_ops_text(code, gpu_fallback=False), "fixed_input_values")

    def test_boundary_probe_exercises_nonfinite_predicates(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return torch.isnan(x), torch.isinf(x), torch.isneginf(x)

def get_inputs():
    return [torch.ones(4)]
"""
        )
        self.assertEqual(validate_ops_text(code, gpu_fallback=False), "fixed_input_values")

    def test_nonfinite_output_is_rejected(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return torch.log(-torch.abs(x))

def get_inputs():
    return [torch.ones(2)]
"""
        )
        self.assertEqual(validate_ops_text(code, gpu_fallback=False), "non_finite_output")

    def test_nonfinite_input_is_allowed_when_the_model_handles_it(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return torch.nan_to_num(x)

def get_inputs():
    return [torch.tensor([float('nan'), 1.0])]
"""
        )
        self.assertEqual(validate_ops_text(code, gpu_fallback=False), "fixed_input_values")

    def test_explicit_cuda_placement_is_normalized_for_cpu_validation(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return x.to("cuda") * 2

def get_inputs():
    return [torch.arange(4, dtype=torch.float32, device="cuda")]
"""
        )
        self.assertEqual(validate_ops_text(code, gpu_fallback=False), "fixed_input_values")

    def test_variable_natural_inputs_with_saturated_outputs_remain_synthetic_only(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return torch.count_nonzero(x)

def get_inputs():
    return [torch.randn(128)]
"""
        )
        result = validate_ops_text_detailed(code)
        self.assertEqual(result["verdict"], "synthetic_sensitivity_only")
        self.assertIn("all_zero", result["detail"])

    def test_fixed_nan_inputs_compare_equal(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return torch.nan_to_num(x)

def get_inputs():
    return [torch.tensor([float("nan"), 1.0])]
"""
        )
        result = validate_ops_text_detailed(code)
        self.assertEqual(result["verdict"], "fixed_input_values")
        self.assertIn("20 consecutive calls", result["detail"])

    def test_fixed_python_scalar_nan_inputs_compare_equal(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return torch.nan_to_num(torch.as_tensor(x))

def get_inputs():
    return [float("nan")]
"""
        )
        result = validate_ops_text_detailed(code)
        self.assertEqual(result["verdict"], "fixed_input_values")
        self.assertIn("20 consecutive calls", result["detail"])

    def test_late_discrete_input_change_avoids_fixed_input_false_positive(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return x * 2

input_calls = 0
def get_inputs():
    global input_calls
    input_calls += 1
    return [torch.tensor([input_calls >= 14])]
"""
        )
        result = validate_ops_text_detailed(code)
        self.assertEqual(result["verdict"], "passed")
        self.assertIn("confirmation call 14", result["detail"])

    def test_train_eval_audit_marks_output_rng_and_state_differences(self) -> None:
        dropout_code = (
            _HEADER
            + """
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return self.dropout(x)

def get_inputs():
    return [torch.randn(32)]
"""
        )
        dropout = validate_ops_text_detailed(dropout_code, audit_train_eval=True)
        self.assertIn("train_eval_output_diff", dropout["mode_flags"])
        self.assertIn("train_only_stochastic", dropout["mode_flags"])

        batch_norm_code = (
            _HEADER
            + """
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x):
        return self.bn(x)

def get_inputs():
    return [torch.randn(8, 4)]
"""
        )
        batch_norm = validate_ops_text_detailed(batch_norm_code, audit_train_eval=True)
        self.assertIn("train_eval_output_diff", batch_norm["mode_flags"])
        self.assertIn("train_eval_state_diff", batch_norm["mode_flags"])

    def test_train_eval_audit_does_not_change_natural_input_rng_sequence(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return self.dropout(x.float())

def get_inputs():
    return [torch.randint(0, 2, (1,))]
"""
        )
        baseline = validate_ops_text_detailed(code, seed=5, audit_train_eval=False)
        audited = validate_ops_text_detailed(code, seed=5, audit_train_eval=True)
        self.assertEqual(audited["verdict"], baseline["verdict"])
        self.assertEqual(audited["detail"], baseline["detail"])
        self.assertTrue(audited["mode_flags"])

    def test_cpu_validation_does_not_probe_cuda_driver_when_seeding(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return x * 2

def get_inputs():
    return [torch.randn(4)]
"""
        )
        with mock.patch.object(
            torch.cuda,
            "is_available",
            side_effect=AssertionError("CPU validation touched the CUDA driver"),
        ):
            result = validate_ops_text_detailed(code, device="cpu", audit_train_eval=True)
        self.assertEqual(result["verdict"], "passed")
        self.assertEqual(result["mode_flags"], ["train_eval_same"])

    def test_uses_five_natural_input_trials_before_synthetic_probes(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return x * 2

input_calls = 0
def get_inputs():
    global input_calls
    input_calls += 1
    value = 2.0 if input_calls >= 6 else 1.0
    return [torch.full((4,), value)]
"""
        )
        result = validate_ops_text_detailed(code, fresh_input_trials=5)
        self.assertEqual(result["verdict"], "passed")
        self.assertIn("fresh trial 5", result["detail"])

    def test_low_fraction_natural_output_change_gets_distinct_verdict(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return x > 0

input_calls = 0
def get_inputs():
    global input_calls
    input_calls += 1
    x = torch.zeros(10000)
    x[input_calls % x.numel()] = 1
    return [x]
"""
        )
        result = validate_ops_text_detailed(
            code,
            fresh_input_trials=5,
            min_natural_output_change_fraction=1e-3,
        )
        self.assertEqual(result["verdict"], "natural_output_low_activity")
        self.assertIn("max_leaf_change_fraction=0.0002", result["detail"])

    def test_multiple_validation_seeds_must_all_pass(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return x * 2

def get_inputs():
    return [torch.randn(32)]
"""
        )
        result = validate_ops_text_detailed(code, validation_seeds=3)
        self.assertEqual(result["verdict"], "passed")
        self.assertIn("passed 3 validation seed(s)", result["detail"])

    def test_unstable_input_shape_is_rejected(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return x * 2

input_call_count = 0
def get_inputs():
    global input_call_count
    input_call_count += 1
    return [torch.ones(input_call_count)]
"""
        )
        self.assertEqual(validate_ops_text(code, gpu_fallback=False), "unstable_input_structure")


class SimilarityTest(unittest.TestCase):
    def test_ast_similarity_matches_renamed_structure(self) -> None:
        left = (
            _HEADER
            + """
    def forward(self, x):
        value = torch.relu(x)
        return value + 2
"""
        )
        right = left.replace("x", "data").replace("relu", "sigmoid").replace("2", "9")
        left_trees = _significant_trees(left, "Model")
        right_trees = _significant_trees(right, "Model")
        score = ast_structure_similarity(
            left_trees,
            right_trees,
            left_total_weight=sum(tree.node_count * tree.weight for tree in left_trees),
            right_total_weight=sum(tree.node_count * tree.weight for tree in right_trees),
        )
        self.assertEqual(score, 1.0)

    def test_maximum_assignment_is_not_greedy(self) -> None:
        self.assertEqual(_maximum_assignment_sum([[10, 9], [9, 0]]), 18)

    def test_semantic_baseline_defaults_missing_entry_point_to_model(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return x + 1
"""
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "validation.parquet"
            pq.write_table(
                pa.Table.from_pylist([{"reward_model": {"ground_truth": code}, "extra_info": {"name": "task"}}]),
                path,
            )
            keys = _read_semantic_baselines(
                [path],
                code_key="reward_model.ground_truth",
                entry_point_key="extra_info.entry_point",
            )
            structural = read_ast_similarity_baselines(
                [path],
                code_key="reward_model.ground_truth",
                entry_point_key="extra_info.entry_point",
            )
        semantic_hash = analyze_reference(code, "Model").semantic_hash
        self.assertIn(("Model", semantic_hash), keys)
        self.assertEqual(len(structural), 1)
        self.assertEqual(structural[0].entry_point, "Model")

    def test_semantic_baseline_repairs_helper_entry_point_before_hashing(self) -> None:
        code = """import torch
class Mish(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))
class Model(torch.nn.Module):
    def forward(self, x):
        return Mish()(x)
def get_inputs():
    return [torch.randn(8)]
"""
        row = {
            "prompt": [{"role": "user", "content": f"Return class ModelNew.\n{code}"}],
            "reward_model": {"ground_truth": code},
            "extra_info": {"entry_point": "Mish"},
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.parquet"
            pq.write_table(pa.Table.from_pylist([row]), path)
            keys = _read_semantic_baselines(
                [path],
                code_key="reward_model.ground_truth",
                entry_point_key="extra_info.entry_point",
            )
        semantic_hash = analyze_reference(code, "Model").semantic_hash
        self.assertIn(("Model", semantic_hash), keys)

    def test_python_token_jaccard_ignores_comments_and_layout(self) -> None:
        left = "x = torch.relu(x)  # comment\nreturn x\n"
        right = "x=torch.relu(x)\n\nreturn x\n"
        self.assertEqual(token_jaccard(python_token_set(left), python_token_set(right)), 1.0)

    def test_best_match_uses_strict_threshold(self) -> None:
        from tools.data.cleaning.similarity import TokenJaccardBaseline

        tokens = python_token_set("x = x + 1\n")
        baseline = TokenJaccardBaseline(0, 4, tokens)
        self.assertIsNone(best_token_jaccard_match("x = x + 1\n", [baseline], threshold=1.0))
        match = best_token_jaccard_match("x = x + 1\n", [baseline], threshold=0.8)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual((match.dataset_index, match.row_index), (0, 4))


class ParquetRewriteTest(unittest.TestCase):
    def test_current_cli_resolves_one_strict_policy(self) -> None:
        config = parse_args(["clean", "input.parquet", "output.parquet"])
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.policy, CURRENT_POLICY)
        self.assertEqual(config.policy.validation_seeds, 3)
        self.assertEqual(config.policy.fresh_input_trials, 20)
        self.assertEqual(config.policy.fixed_input_repeats, 40)
        self.assertTrue(config.policy.audit_train_eval_all)
        self.assertTrue(config.policy.require_each_forward_input_sensitive)

    def test_cleanup_refuses_to_overwrite_its_input(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            pq.write_table(pa.table({"value": [1]}), source)
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                run_cleanup(
                    CleanupConfig(
                        command="static",
                        input=source,
                        output=source,
                        overwrite=True,
                    )
                )

    def test_nested_metadata_repairs_preserve_schema(self) -> None:
        schema = pa.schema(
            [
                pa.field("value", pa.int64()),
                pa.field(
                    "extra_info",
                    pa.struct(
                        [
                            pa.field("entry_point", pa.string()),
                            pa.field("module_name", pa.string()),
                            pa.field("uuid", pa.string()),
                        ]
                    ),
                ),
            ]
        )
        table = pa.Table.from_pylist(
            [
                {
                    "value": 1,
                    "extra_info": {"entry_point": "Mish", "module_name": "Mish", "uuid": "duplicate"},
                }
            ],
            schema=schema,
        )
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            output = Path(directory) / "output.parquet"
            pq.write_table(table, source)
            repairs = {0: {"entry_point": "Model", "module_name": "Model", "uuid": "clean_hash"}}
            _write_filtered_parquet(source, output, [True], repairs, batch_size=1, compression="zstd")
            rewritten = pq.read_table(output)
        self.assertEqual(rewritten.schema, schema)
        self.assertEqual(
            rewritten.to_pylist()[0]["extra_info"],
            {"entry_point": "Model", "module_name": "Model", "uuid": "clean_hash"},
        )

    def test_cli_repairs_a_surviving_duplicate_uuid(self) -> None:
        def row(expression: str) -> dict[str, object]:
            code = (
                _HEADER
                + f"""
    def forward(self, x):
        return {expression}

def get_inputs():
    return [torch.ones(2)]
"""
            )
            return {
                "prompt": [{"content": f"Return ModelNew.\n{code}", "role": "user"}],
                "ability": "kernel_optimization",
                "reward_model": {"ground_truth": code, "style": "rule"},
                "extra_info": {
                    "entry_point": "Model",
                    "module_name": "Model",
                    "ops": "[]",
                    "uuid": "duplicate",
                },
            }

        table = pa.Table.from_pylist([row("x * 2"), row("x + 1")])
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            output = Path(directory) / "output.parquet"
            pq.write_table(table, source)
            subprocess.run(
                [
                    *_CLEANER_COMMAND,
                    "static",
                    str(source),
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            rows = pq.read_table(output).to_pylist()
            summary = json.loads(output.with_name("output.summary.json").read_text())
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({item["extra_info"]["uuid"] for item in rows}), 2)
        self.assertEqual(summary["repair_counts"], {"uuid": 1})

    def test_cli_quarantines_a_static_effect_reason(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x, replacement):
        x = self.bn(x)
        return replacement + 1

def get_inputs():
    return [torch.randn(8, 4), torch.randn(8, 4)]
"""
        )
        row = {
            "prompt": [{"content": f"Return ModelNew.\n{code}", "role": "user"}],
            "ability": "kernel_optimization",
            "reward_model": {"ground_truth": code, "style": "rule"},
            "extra_info": {"entry_point": "Model", "module_name": "Model", "ops": "[]", "uuid": "state"},
        }
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            output = Path(directory) / "output.parquet"
            quarantine = Path(directory) / "output.quarantine.parquet"
            pq.write_table(pa.Table.from_pylist([row]), source)
            subprocess.run(
                [
                    *_CLEANER_COMMAND,
                    "static",
                    str(source),
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(output.with_name("output.summary.json").read_text())
            quarantine_rows = pq.ParquetFile(quarantine).metadata.num_rows
        self.assertEqual(summary["output_rows"], 0)
        self.assertEqual(summary["quarantined_rows"], 1)
        self.assertEqual(quarantine_rows, 1)

    def test_cli_reuses_a_matching_runtime_audit(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return x * 2

def get_inputs():
    return [torch.ones(2)]
"""
        )
        row = {
            "prompt": [{"content": f"Return ModelNew.\n{code}", "role": "user"}],
            "ability": "kernel_optimization",
            "reward_model": {"ground_truth": code, "style": "rule"},
            "extra_info": {
                "entry_point": "Model",
                "module_name": "Model",
                "ops": "[]",
                "uuid": "unique",
            },
        }
        semantic_hash = analyze_reference(code, "Model").semantic_hash
        audit_record = {
            "row_index": 0,
            "uuid": "unique",
            "entry_point": "Model",
            "semantic_hash": semantic_hash,
            "runtime_verdict": "passed",
            "runtime_detail": "",
        }
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            output = Path(directory) / "output.parquet"
            audit = Path(directory) / "runtime.audit.jsonl"
            pq.write_table(pa.Table.from_pylist([row]), source)
            audit.write_text(json.dumps(audit_record) + "\n", encoding="utf-8")
            subprocess.run(
                [
                    *_CLEANER_COMMAND,
                    "recover",
                    str(source),
                    str(output),
                    "--prior-audit",
                    str(audit),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(output.with_name("output.summary.json").read_text())
        self.assertEqual(summary["output_rows"], 1)
        self.assertEqual(summary["runtime_verdict_counts"], {"passed": 1})
        self.assertEqual(summary["configuration"]["runtime_reused_from"], str(audit.resolve()))

    def test_train_eval_timeout_cannot_replace_reused_acceptance_verdict(self) -> None:
        code = """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x):
        return self.bn(x)

def get_inputs():
    return [torch.randn(8, 4)]

def get_init_inputs():
    return []
"""
        row = {
            "prompt": [{"content": f"Return ModelNew.\n{code}", "role": "user"}],
            "ability": "kernel_optimization",
            "reward_model": {"ground_truth": code, "style": "rule"},
            "extra_info": {
                "entry_point": "Model",
                "module_name": "Model",
                "ops": "[]",
                "uuid": "mode-timeout",
            },
        }
        audit_record = {
            "row_index": 0,
            "uuid": "mode-timeout",
            "entry_point": "Model",
            "semantic_hash": analyze_reference(code, "Model").semantic_hash,
            "runtime_verdict": "passed",
            "runtime_detail": "prior acceptance passed",
            "mode_flags": [],
        }
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            output = Path(directory) / "output.parquet"
            prior_audit = Path(directory) / "runtime.audit.jsonl"
            pq.write_table(pa.Table.from_pylist([row]), source)
            prior_audit.write_text(json.dumps(audit_record) + "\n", encoding="utf-8")
            subprocess.run(
                [
                    *_CLEANER_COMMAND,
                    "recover",
                    str(source),
                    str(output),
                    "--prior-audit",
                    str(prior_audit),
                    "--timeout",
                    "0.001",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(output.with_name("output.summary.json").read_text())
            decision = json.loads(output.with_name("output.audit.jsonl").read_text())
        self.assertEqual(summary["output_rows"], 1)
        self.assertEqual(decision["runtime_verdict"], "passed")
        self.assertIn("train_eval_audit_failed", decision["mode_flags"])
        self.assertIn("train_eval_inconclusive", decision["mode_flags"])

    def test_cli_selectively_reruns_a_prior_sensitivity_failure(self) -> None:
        def make_row(code: str, uuid: str) -> dict[str, object]:
            return {
                "prompt": [{"content": f"Return ModelNew.\n{code}", "role": "user"}],
                "ability": "kernel_optimization",
                "reward_model": {"ground_truth": code, "style": "rule"},
                "extra_info": {"entry_point": "Model", "module_name": "Model", "ops": "[]", "uuid": uuid},
            }

        argsort_code = (
            _HEADER
            + """
    def forward(self, x):
        return torch.argsort(x)

def get_inputs():
    return [torch.tensor([1.0, 2.0, 3.0])]
"""
        )
        linear_code = (
            _HEADER
            + """
    def forward(self, x):
        return x * 2

def get_inputs():
    return [torch.ones(3)]
"""
        )
        rows = [make_row(argsort_code, "argsort"), make_row(linear_code, "linear")]
        audit_records = []
        for index, (code, uuid, verdict) in enumerate(
            [(argsort_code, "argsort", "different_input_not_changed"), (linear_code, "linear", "passed")]
        ):
            audit_records.append(
                {
                    "row_index": index,
                    "uuid": uuid,
                    "entry_point": "Model",
                    "semantic_hash": analyze_reference(code, "Model").semantic_hash,
                    "runtime_verdict": verdict,
                    "runtime_detail": "",
                    "flags": [],
                }
            )
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            output = Path(directory) / "output.parquet"
            audit = Path(directory) / "v1.audit.jsonl"
            indices = Path(directory) / "indices.txt"
            pq.write_table(pa.Table.from_pylist(rows), source)
            audit.write_text("".join(json.dumps(record) + "\n" for record in audit_records), encoding="utf-8")
            indices.write_text("0\n", encoding="utf-8")
            with mock.patch("builtins.print"):
                run_cleanup(
                    CleanupConfig(
                        command="recover",
                        input=source,
                        output=output,
                        workers=1,
                        timeout=60,
                        prior_audit=audit,
                        rerun_indices_path=indices,
                        device="cpu",
                    )
                )
            records = [json.loads(line) for line in output.with_name("output.audit.jsonl").read_text().splitlines()]
        self.assertFalse(records[0]["keep"])
        self.assertTrue(records[1]["keep"])
        self.assertEqual(records[0]["runtime_source"], "rerun_v5")
        self.assertEqual(records[0]["runtime_verdict"], "fixed_input_values")
        self.assertEqual(records[1]["runtime_source"], "reused_v1")

    def test_runtime_recovery_reruns_all_mode_audit(self) -> None:
        code = (
            _HEADER
            + """
    def __init__(self):
        super().__init__()
        self.parameters_out = nn.ParameterDict({
            "weight": nn.Parameter(torch.randn(4)),
        })

    def forward(self, x):
        return x * 2, self.parameters_out

def get_inputs():
    return [torch.randn(4)]

def get_init_inputs():
    return []
"""
        )
        row = {
            "prompt": [{"content": f"Return ModelNew.\n{code}", "role": "user"}],
            "ability": "kernel_optimization",
            "reward_model": {"ground_truth": code, "style": "rule"},
            "extra_info": {
                "entry_point": "Model",
                "module_name": "Model",
                "ops": "[]",
                "uuid": "container-output",
            },
        }
        prior = {
            "row_index": 0,
            "uuid": "container-output",
            "entry_point": "Model",
            "semantic_hash": analyze_reference(code, "Model").semantic_hash,
            "runtime_verdict": "no_same_output",
            "runtime_detail": "old object-identity comparison",
            "keep": False,
            "flags": [],
            "mode_flags": ["train_eval_output_diff"],
        }
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            output = Path(directory) / "output.parquet"
            audit = Path(directory) / "v1.audit.jsonl"
            indices = Path(directory) / "indices.txt"
            pq.write_table(pa.Table.from_pylist([row]), source)
            audit.write_text(json.dumps(prior) + "\n", encoding="utf-8")
            indices.write_text("0\n", encoding="utf-8")
            policy = replace(
                CURRENT_POLICY,
                validation_seeds=1,
                fresh_input_trials=2,
                fixed_input_repeats=3,
            )
            with mock.patch("builtins.print"):
                run_cleanup(
                    CleanupConfig(
                        command="recover",
                        input=source,
                        output=output,
                        workers=1,
                        timeout=60,
                        prior_audit=audit,
                        rerun_indices_path=indices,
                        device="cpu",
                        policy=policy,
                    )
                )
            record = json.loads(output.with_name("output.audit.jsonl").read_text())
            summary = json.loads(output.with_name("output.summary.json").read_text())
        self.assertTrue(record["keep"])
        self.assertEqual(record["runtime_verdict"], "passed")
        self.assertEqual(record["mode_flags"], ["train_eval_same"])
        self.assertEqual(summary["configuration"]["train_eval_rows_executed"], 1)

    def test_cli_deduplicates_against_reference_parquet(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return x * 2

def get_inputs():
    return [torch.ones(2)]
"""
        )
        row = {
            "prompt": [{"content": f"Return ModelNew.\n{code}", "role": "user"}],
            "ability": "kernel_optimization",
            "reward_model": {"ground_truth": code, "style": "rule"},
            "extra_info": {
                "entry_point": "Model",
                "module_name": "Model",
                "ops": "[]",
                "uuid": "one",
            },
        }
        with TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.parquet"
            source = Path(directory) / "source.parquet"
            output = Path(directory) / "output.parquet"
            pq.write_table(pa.Table.from_pylist([row]), baseline)
            row["extra_info"]["uuid"] = "two"
            pq.write_table(pa.Table.from_pylist([row]), source)
            subprocess.run(
                [
                    *_CLEANER_COMMAND,
                    "static",
                    str(source),
                    str(output),
                    "--dedup-against",
                    str(baseline),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(output.with_name("output.summary.json").read_text())
        self.assertEqual(summary["output_rows"], 0)
        self.assertEqual(summary["reason_counts_nonexclusive"], {"semantic_duplicate_against": 1})

    def test_current_policy_denies_manually_confirmed_uuid(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return x * 2

def get_inputs():
    return [torch.randn(2)]
"""
        )
        row = {
            "prompt": [{"content": f"Return ModelNew.\n{code}", "role": "user"}],
            "ability": "kernel_optimization",
            "reward_model": {"ground_truth": code, "style": "rule"},
            "extra_info": {
                "entry_point": "Model",
                "level": "0",
                "module_name": "Model",
                "ops": "[]",
                "uuid": "cuda_llm_258008",
            },
        }
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            output = Path(directory) / "output.parquet"
            pq.write_table(pa.Table.from_pylist([row]), source)
            subprocess.run(
                [
                    *_CLEANER_COMMAND,
                    "static",
                    str(source),
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            record = json.loads(output.with_name("output.audit.jsonl").read_text())
            summary = json.loads(output.with_name("output.summary.json").read_text())
        self.assertFalse(record["keep"])
        self.assertIn("explicit_uuid_denylist", record["reasons"])
        self.assertIn("explicit_uuid_denylist:cuda_llm_258008", record["flags"])
        self.assertEqual(summary["configuration"]["denied_uuids"], sorted(CURRENT_POLICY.denied_uuids))

    def test_cli_quarantines_a_reused_runtime_verdict(self) -> None:
        code = (
            _HEADER
            + """
    def forward(self, x):
        return torch.count_nonzero(x)

def get_inputs():
    return [torch.randn(8)]
"""
        )
        row = {
            "prompt": [{"content": f"Return ModelNew.\n{code}", "role": "user"}],
            "ability": "kernel_optimization",
            "reward_model": {"ground_truth": code, "style": "rule"},
            "extra_info": {
                "entry_point": "Model",
                "module_name": "Model",
                "ops": "[]",
                "uuid": "inconclusive",
            },
        }
        prior = {
            "row_index": 0,
            "uuid": "inconclusive",
            "entry_point": "Model",
            "semantic_hash": analyze_reference(code, "Model").semantic_hash,
            "runtime_verdict": "input_sensitivity_inconclusive",
            "runtime_detail": "all probes matched",
            "flags": ["forward_output_dependency:dependent"],
            "mode_flags": ["train_eval_same"],
            "keep": True,
        }
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            output = Path(directory) / "clean.parquet"
            quarantine = Path(directory) / "clean.quarantine.parquet"
            audit = Path(directory) / "v2.audit.jsonl"
            pq.write_table(pa.Table.from_pylist([row]), source)
            audit.write_text(json.dumps(prior) + "\n", encoding="utf-8")
            subprocess.run(
                [
                    *_CLEANER_COMMAND,
                    "recover",
                    str(source),
                    str(output),
                    "--prior-audit",
                    str(audit),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            record = json.loads(output.with_name("clean.audit.jsonl").read_text())
            summary = json.loads(output.with_name("clean.summary.json").read_text())
            quarantined = pq.read_table(quarantine).to_pylist()
        self.assertFalse(record["keep"])
        self.assertTrue(record["quarantined"])
        self.assertEqual(
            record["reasons"],
            ["quarantined_runtime_verdict:input_sensitivity_inconclusive"],
        )
        self.assertEqual(summary["contract_version"], CURRENT_POLICY.contract_version)
        self.assertEqual(summary["configuration"]["policy"]["validation_seeds"], 3)
        self.assertEqual(summary["quarantined_rows"], 1)
        self.assertEqual(len(quarantined), 1)

    def test_cli_rejects_token_jaccard_near_duplicate(self) -> None:
        baseline_code = (
            _HEADER
            + """
    def forward(self, x):
        x = torch.relu(x)
        return x * 2

def get_inputs():
    return [torch.ones(8)]
"""
        )
        candidate_code = baseline_code.replace("torch.ones(8)", "torch.ones(16)")

        def row(code: str, uuid: str) -> dict[str, object]:
            return {
                "prompt": [{"content": f"Return ModelNew.\n{code}", "role": "user"}],
                "ability": "kernel_optimization",
                "reward_model": {"ground_truth": code, "style": "rule"},
                "extra_info": {
                    "entry_point": "Model",
                    "module_name": "Model",
                    "ops": "[]",
                    "uuid": uuid,
                },
            }

        with TemporaryDirectory() as directory:
            baseline = Path(directory) / "validation.parquet"
            source = Path(directory) / "source.parquet"
            output = Path(directory) / "output.parquet"
            pq.write_table(pa.Table.from_pylist([row(baseline_code, "baseline")]), baseline)
            pq.write_table(pa.Table.from_pylist([row(candidate_code, "candidate")]), source)
            subprocess.run(
                [
                    *_CLEANER_COMMAND,
                    "static",
                    str(source),
                    str(output),
                    "--dedup-against",
                    str(baseline),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            record = json.loads(output.with_name("output.audit.jsonl").read_text())
        self.assertIn("token_jaccard_duplicate_against", record["reasons"])
        self.assertTrue(any(flag.startswith("token_jaccard_duplicate_against:0:0:") for flag in record["flags"]))

    def test_cli_rejects_ast_structural_near_duplicate(self) -> None:
        baseline_code = (
            _HEADER
            + """
        self.activation = nn.ReLU()

    def forward(self, x):
        value = self.activation(x)
        return value + 2

def get_inputs():
    return [torch.ones(8)]
"""
        )
        candidate_code = (
            baseline_code.replace("activation", "gate")
            .replace("ReLU", "Sigmoid")
            .replace("value", "result")
            .replace(" + 2", " + 9")
        )

        def row(code: str, uuid: str) -> dict[str, object]:
            return {
                "prompt": [{"content": f"Return ModelNew.\n{code}", "role": "user"}],
                "ability": "kernel_optimization",
                "reward_model": {"ground_truth": code, "style": "rule"},
                "extra_info": {
                    "entry_point": "Model",
                    "module_name": "Model",
                    "ops": "[]",
                    "uuid": uuid,
                },
            }

        with TemporaryDirectory() as directory:
            baseline = Path(directory) / "validation.parquet"
            source = Path(directory) / "source.parquet"
            output = Path(directory) / "output.parquet"
            pq.write_table(pa.Table.from_pylist([row(baseline_code, "baseline")]), baseline)
            pq.write_table(pa.Table.from_pylist([row(candidate_code, "candidate")]), source)
            subprocess.run(
                [
                    *_CLEANER_COMMAND,
                    "static",
                    str(source),
                    str(output),
                    "--dedup-against",
                    str(baseline),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            record = json.loads(output.with_name("output.audit.jsonl").read_text())
        self.assertIn("ast_similarity_duplicate_against", record["reasons"])
        self.assertTrue(any(flag.startswith("ast_similarity_duplicate_against:0:0:") for flag in record["flags"]))


class AuditSummaryTest(unittest.TestCase):
    def test_aggregates_complete_audit_and_verifies_clean_rows(self) -> None:
        records = [
            {
                "row_index": 0,
                "keep": True,
                "quarantined": False,
                "reasons": [],
                "flags": ["train_eval_same"],
                "mode_flags": ["train_eval_same"],
                "repairs": {"uuid": "fixed"},
                "runtime_verdict": "passed",
                "runtime_detail": "",
                "runtime_source": "fresh_v5",
            },
            {
                "row_index": 1,
                "keep": False,
                "quarantined": True,
                "reasons": ["quarantined_runtime_verdict:synthetic_sensitivity_only"],
                "flags": ["runtime_verdict_quarantined"],
                "mode_flags": [],
                "repairs": {},
                "runtime_verdict": "synthetic_sensitivity_only",
                "runtime_detail": "synthetic probe affine changed output",
                "runtime_source": "fresh_v5",
            },
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "audit.jsonl"
            audit.write_text("".join(json.dumps(row) + "\n" for row in records))
            input_path = root / "input.parquet"
            clean_path = root / "clean.parquet"
            summary_path = root / "summary.json"
            pq.write_table(pa.table({"x": [1, 2]}), input_path)
            pq.write_table(pa.table({"x": [1]}), clean_path)
            with mock.patch("builtins.print"):
                audit_summary_main(
                    [
                        "--audit-jsonl",
                        str(audit),
                        "--output-summary",
                        str(summary_path),
                        "--input-parquet",
                        str(input_path),
                        "--clean-parquet",
                        str(clean_path),
                    ]
                )
            summary = json.loads(summary_path.read_text())
        self.assertEqual(summary["input_rows_considered"], 2)
        self.assertEqual(summary["output_rows"], 1)
        self.assertEqual(summary["runtime_sensitivity_probe_counts"], {"affine": 1})
        self.assertEqual(summary["train_eval_counts_nonexclusive"], {"train_eval_same": 1})
        self.assertEqual(summary["attempted_repair_counts"], {"uuid": 1})
        self.assertEqual(summary["repair_counts"], {"uuid": 1})


class ShardMergeTest(unittest.TestCase):
    def test_splits_parquet_into_balanced_contiguous_shards(self) -> None:
        rows = [{"extra_info": {"uuid": str(index)}} for index in range(7)]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.parquet"
            output_dir = root / "shards"
            manifest_path = root / "manifest.json"
            pq.write_table(pa.Table.from_pylist(rows), source)
            with mock.patch("builtins.print"):
                shards_main(
                    [
                        "split",
                        "--input",
                        str(source),
                        "--output-dir",
                        str(output_dir),
                        "--shards",
                        "3",
                        "--prefix",
                        "part",
                        "--manifest",
                        str(manifest_path),
                    ]
                )
            result = json.loads(manifest_path.read_text())
            shard_rows = [
                pq.read_table(item["path"])["extra_info"].combine_chunks().field("uuid").to_pylist()
                for item in result["shards"]
            ]
            manifest_record = json.loads(manifest_path.read_text())

        self.assertEqual([item["rows"] for item in result["shards"]], [3, 2, 2])
        self.assertEqual(shard_rows, [["0", "1", "2"], ["3", "4"], ["5", "6"]])
        self.assertEqual(manifest_record["input_rows"], 7)

    def test_overlays_runtime_by_repaired_effective_uuid(self) -> None:
        base_records = [
            {
                "row_index": 0,
                "uuid": "duplicate",
                "entry_point": "Model",
                "semantic_hash": "a",
                "repairs": {},
                "keep": True,
                "flags": ["static-a"],
                "reasons": [],
                "runtime_verdict": "skipped_by_option",
                "runtime_detail": "",
            },
            {
                "row_index": 1,
                "uuid": "duplicate",
                "entry_point": "Model",
                "semantic_hash": "b",
                "repairs": {"uuid": "repaired"},
                "keep": True,
                "flags": ["static-b"],
                "reasons": [],
                "runtime_verdict": "skipped_by_option",
                "runtime_detail": "",
            },
        ]
        runtime_records = [
            {
                "row_index": 0,
                "uuid": "duplicate",
                "entry_point": "Model",
                "semantic_hash": "a",
                "keep": True,
                "flags": ["train_eval_same"],
                "reasons": [],
                "quarantined": False,
                "runtime_verdict": "passed",
                "runtime_detail": "ok-a",
                "runtime_source": "fresh_v5",
                "mode_flags": ["train_eval_same"],
                "mode_detail": "",
            },
            {
                "row_index": 0,
                "uuid": "repaired",
                "entry_point": "Model",
                "semantic_hash": "b",
                "keep": False,
                "flags": ["runtime_verdict_quarantined"],
                "reasons": ["quarantined_runtime_verdict:synthetic_sensitivity_only"],
                "quarantined": True,
                "runtime_verdict": "synthetic_sensitivity_only",
                "runtime_detail": "probe",
                "runtime_source": "fresh_v5",
                "mode_flags": [],
                "mode_detail": "",
            },
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.jsonl"
            runtime_one = root / "one.jsonl"
            runtime_two = root / "two.jsonl"
            output = root / "output.jsonl"
            manifest = root / "manifest.json"
            base.write_text("".join(json.dumps(row) + "\n" for row in base_records))
            runtime_one.write_text(json.dumps(runtime_records[0]) + "\n")
            runtime_two.write_text(json.dumps(runtime_records[1]) + "\n")
            with mock.patch("builtins.print"):
                shards_main(
                    [
                        "overlay",
                        "--base-audit",
                        str(base),
                        "--runtime-audit",
                        str(runtime_one),
                        "--runtime-audit",
                        str(runtime_two),
                        "--output-audit",
                        str(output),
                        "--manifest",
                        str(manifest),
                    ]
                )
            result = json.loads(manifest.read_text())
            overlaid = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual([record["runtime_detail"] for record in overlaid], ["ok-a", "probe"])
        self.assertEqual(
            [record["flags"] for record in overlaid], [["train_eval_same"], ["runtime_verdict_quarantined"]]
        )
        self.assertEqual(
            [record["reasons"] for record in overlaid],
            [[], ["quarantined_runtime_verdict:synthetic_sensitivity_only"]],
        )
        self.assertEqual([record["uuid"] for record in overlaid], ["duplicate", "duplicate"])
        self.assertEqual(overlaid[1]["repairs"]["uuid"], "repaired")
        self.assertEqual(result["runtime_rows_overlaid"], 2)

    def test_merges_parquets_and_audits_in_source_uuid_order(self) -> None:
        def row(uuid: str) -> dict[str, object]:
            return {
                "extra_info": {"uuid": uuid},
                "reward_model": {"ground_truth": f"code-{uuid}"},
            }

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.parquet"
            shard_one = root / "one.parquet"
            shard_two = root / "two.parquet"
            audit_one = root / "one.audit.jsonl"
            audit_two = root / "two.audit.jsonl"
            output = root / "merged.parquet"
            output_audit = root / "merged.audit.jsonl"
            manifest = root / "manifest.json"
            pq.write_table(pa.Table.from_pylist([row("a"), row("b"), row("c")]), source)
            pq.write_table(pa.Table.from_pylist([row("c"), row("a")]), shard_one)
            pq.write_table(pa.Table.from_pylist([row("b")]), shard_two)
            audit_one.write_text(
                "".join(
                    json.dumps({"row_index": index, "uuid": uuid}) + "\n" for index, uuid in enumerate(("c", "a"))
                ),
                encoding="utf-8",
            )
            audit_two.write_text(
                json.dumps({"row_index": 0, "uuid": "b"}) + "\n",
                encoding="utf-8",
            )
            with mock.patch("builtins.print"):
                shards_main(
                    [
                        "merge",
                        "--input",
                        str(shard_one),
                        "--input",
                        str(shard_two),
                        "--audit-jsonl",
                        str(audit_one),
                        "--audit-jsonl",
                        str(audit_two),
                        "--source-order",
                        str(source),
                        "--output-input",
                        str(output),
                        "--output-audit",
                        str(output_audit),
                        "--manifest",
                        str(manifest),
                    ]
                )
            result = json.loads(manifest.read_text())
            output_uuids = pq.read_table(output, columns=["extra_info.uuid"])["uuid"].to_pylist()
            audit_records = [json.loads(line) for line in output_audit.read_text().splitlines()]
            manifest_record = json.loads(manifest.read_text())

        self.assertEqual(output_uuids, ["a", "b", "c"])
        self.assertEqual([record["uuid"] for record in audit_records], output_uuids)
        self.assertEqual([record["row_index"] for record in audit_records], [0, 1, 2])
        self.assertEqual(result["output_rows"], 3)
        self.assertEqual(manifest_record["order"], result["order"])


if __name__ == "__main__":
    unittest.main()
