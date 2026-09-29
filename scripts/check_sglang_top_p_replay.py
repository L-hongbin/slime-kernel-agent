"""Validate SGLang top-p replay support, optionally against a live server.

SGLang 0.5.15 uses the repository's legacy ``return_top_p_token_ids``
extension. SGLang 0.5.20 uses its native ``return_sampling_mask`` protocol plus
a small patch that adds EAGLE/NEXTN support and unbounded top-p admission.
"""

import argparse
import inspect
import json
import math
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import ProxyHandler, Request, build_opener

import torch
from sglang.srt.layers import sampler
from sglang.srt.sampling.sampling_params import TOP_K_ALL

try:
    from sglang.srt.layers import logprob_processor
except ImportError:
    from sglang.srt.layers.utils import logprob as logprob_processor


def _is_native_sampling_mask_runtime() -> bool:
    return hasattr(logprob_processor, "_build_spec_sampling_mask")


def _check_native_sampling_mask(check_sort_reuse: bool) -> list[dict]:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput, SamplingMaskStatus
    from sglang.srt.managers.scheduler_components.batch_result_processor import SchedulerBatchResultProcessor

    build_spec_sampling_mask = logprob_processor._build_spec_sampling_mask
    if check_sort_reuse:
        source = inspect.getsource(build_spec_sampling_mask)
        assert source.count(".sort(") == 1, "speculative replay must sort once"
        sampler_source = inspect.getsource(sampler.Sampler._build_sampling_mask_output)
        assert "realized_max" in sampler_source, "ordinary replay must use dynamic packed width"

    checks = []
    for top_p in (0.01, 0.95, 0.999):
        logits = torch.randn(8, 23) * 4
        logprobs = torch.log_softmax(logits, dim=-1)
        sampling = SimpleNamespace(
            is_all_greedy=False,
            temperatures=torch.tensor([[1.0], [1.3]]),
            top_ks=torch.tensor([TOP_K_ALL, TOP_K_ALL]),
            top_ps=torch.tensor([top_p, top_p]),
            min_ps=torch.zeros(2),
            need_top_p_sampling=True,
            need_min_p_sampling=False,
            sampling_mask_batch_indices=torch.tensor([0, 1]),
        )
        batch = SimpleNamespace(sampling_info=sampling)
        predict = logits.argmin(-1)
        accept_lens = torch.tensor([4, 2])
        output, _, valid, selected_logprobs = build_spec_sampling_mask(
            batch,
            logprobs,
            predict,
            accept_lens,
            4,
            max_tokens=23,
        )
        assert all(int(status) == SamplingMaskStatus.OK for status in output.statuses)
        probs = logprobs.exp()
        for row in range(8):
            is_valid = row % 4 < int(accept_lens[row // 4])
            length = int(output.lengths[row])
            if not is_valid:
                assert length == 0
                continue
            ids = output.token_ids[row, :length]
            sorted_probs, indices = probs[row].sort(descending=True)
            expected_ids = indices[(sorted_probs.cumsum(0) - sorted_probs) <= top_p]
            expected = set(expected_ids.tolist()) | {int(predict[row])}
            assert set(ids.tolist()) == expected
            keep = torch.zeros(23, dtype=torch.bool)
            keep[list(expected)] = True
            expected_logprob = logits[row, predict[row]] - logits[row, keep].logsumexp(0)
            torch.testing.assert_close(selected_logprobs[row], expected_logprob)

        logits_output = LogitsProcessorOutput(next_token_logits=None, sampling_mask_output=output)
        reqs = [
            SimpleNamespace(return_sampling_mask=True),
            SimpleNamespace(return_sampling_mask=True),
        ]
        SchedulerBatchResultProcessor.materialize_sampling_mask_output(
            reqs,
            logits_output,
            # The 0.5.20 scheduler normalization returns Python lists here.
            accept_lens=accept_lens.tolist(),
            stride=4,
        )
        assert [len(value) for value in logits_output.next_token_sampling_mask_idx] == [
            4,
            2,
        ]
        assert [len(value) for value in logits_output.next_token_sampling_logprobs] == [
            4,
            2,
        ]
        checks.append({"top_p": top_p, "spec_accept_lengths": [4, 2], "dense_reference": "passed"})

    if check_sort_reuse:
        checks.append(
            {
                "native_sampling_mask": True,
                "dynamic_packed_width": True,
                "spec_sort_count": 1,
            }
        )
    return checks


def _check_legacy_top_p(check_sort_reuse: bool) -> list[dict]:
    compute_spec_v2_logprobs = logprob_processor.compute_spec_v2_logprobs
    renorm_logprob_over_top_p = logprob_processor.renorm_logprob_over_top_p
    checks = []
    for top_p in (0.01, 0.95, 0.999):
        logits = torch.randn(8, 23) * 4
        sampling = SimpleNamespace(
            is_all_greedy=False,
            temperatures=torch.tensor([[1.0], [1.3]]),
            top_ks=torch.tensor([TOP_K_ALL, TOP_K_ALL]),
            top_ps=torch.tensor([top_p, top_p]),
            min_ps=torch.zeros(2),
            need_top_p_sampling=True,
            need_return_top_p_token_ids=True,
            return_top_p_token_ids=torch.tensor([True, True]),
        )
        batch = SimpleNamespace(seq_lens=[10, 20], sampling_info=sampling, top_logprobs_nums=[], token_ids_logprobs=[])
        output = SimpleNamespace(next_token_logits=logits, next_token_top_p_token_ids=None)
        # Tail tokens exercise the same force-keep rule as the training loss.
        predict = logits.argmin(-1)
        accept_index = torch.arange(8).reshape(2, 4)
        accept_lens = torch.tensor([4, 2])
        with patch.object(
            logprob_processor, "_top_p_keep_mask_sorted", wraps=logprob_processor._top_p_keep_mask_sorted
        ) as sort:
            compute_spec_v2_logprobs(batch, output, predict, accept_index, accept_lens, 3)
            if check_sort_reuse:
                assert sort.call_count == 1, f"MTP replay sorted {sort.call_count} times"
        scaled = logits / sampling.temperatures.repeat_interleave(4, dim=0)
        probs = scaled.softmax(-1)
        for row in range(8):
            valid = row % 4 < accept_lens[row // 4]
            ids = output.next_token_top_p_token_ids[row]
            if not valid:
                assert ids is None
                continue
            sorted_probs, indices = probs[row].sort(descending=True)
            expected_ids = indices[(sorted_probs.cumsum(0) - sorted_probs) <= top_p]
            assert set(ids.tolist()) == set(expected_ids.tolist())
            keep = torch.zeros(23, dtype=torch.bool)
            keep[expected_ids] = True
            keep[predict[row]] = True
            expected_logprob = scaled[row, predict[row]] - scaled[row, keep].logsumexp(0)
            torch.testing.assert_close(output.next_token_logprobs.flatten()[row], expected_logprob)
        assert torch.isfinite(output.next_token_logprobs).all()
        # A mixed batch must leave unrequested rows' logprobs unchanged.
        mixed = renorm_logprob_over_top_p(
            probs[:2],
            sampling.top_ks,
            sampling.top_ps,
            sampling.min_ps,
            True,
            False,
            torch.tensor([True, False]),
            predict[:2],
        )
        torch.testing.assert_close(mixed[1], probs[1].log())
        checks.append({"top_p": top_p, "spec_accept_lengths": [4, 2], "dense_reference": "passed"})

    if check_sort_reuse:
        # Ordinary sampling computes the support before selecting the token,
        # then reuses it for logprob normalization after the token is known.
        # Include ties, mixed requests, top-k/min-p intersections and tail tokens.
        for top_k, top_p, min_p in ((TOP_K_ALL, 0.95, 0.0), (3, 0.8, 0.2), (2, 1.0, 0.0)):
            probs = torch.tensor([[0.4, 0.2, 0.2, 0.15, 0.05], [0.1, 0.1, 0.3, 0.3, 0.2]])
            sampling = SimpleNamespace(
                need_return_top_p_token_ids=True,
                return_top_p_token_ids=torch.tensor([True, False]),
                top_ks=torch.tensor([top_k, top_k]),
                top_ps=torch.tensor([top_p, top_p]),
                min_ps=torch.tensor([min_p, min_p]),
                need_top_p_sampling=top_p != 1.0,
                need_min_p_sampling=min_p != 0.0,
            )
            output = SimpleNamespace(next_token_top_p_token_ids=None)
            original_sort = logprob_processor._top_p_keep_mask_sorted
            with (
                patch.object(sampler, "_top_p_keep_mask_sorted", wraps=original_sort) as sampler_sort,
                patch.object(logprob_processor, "_top_p_keep_mask_sorted", wraps=original_sort) as logprob_sort,
            ):
                support = sampler.Sampler._attach_top_p_token_ids_to_output(None, output, probs, sampling, False)
                before = tuple(t.clone() for t in support)
                kwargs = dict(
                    probs=probs,
                    top_ks=sampling.top_ks,
                    top_ps=sampling.top_ps,
                    min_ps=sampling.min_ps,
                    need_top_p_sampling=sampling.need_top_p_sampling,
                    need_min_p_sampling=sampling.need_min_p_sampling,
                    request_mask=sampling.return_top_p_token_ids,
                    force_keep_token_ids=torch.tensor([4, 0]),
                )
                actual = renorm_logprob_over_top_p(**kwargs, precomputed_support=support)
                assert sampler_sort.call_count + logprob_sort.call_count == 1
            expected = renorm_logprob_over_top_p(**kwargs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for saved, current in zip(before, support, strict=True):
                torch.testing.assert_close(saved, current, rtol=0, atol=0)
            assert output.next_token_top_p_token_ids[1] is None
            assert torch.isfinite(actual[0, 4])
        checks.append({"ordinary_and_mtp_sort_count": 1, "mixed_filters_and_force_keep": "passed"})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Optional live SGLang server URL, without /generate.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--check-sort-reuse",
        action="store_true",
        help="Require one full sort per replay step.",
    )
    args = parser.parse_args()
    torch.set_default_device(args.device)
    torch.manual_seed(17)

    native = _is_native_sampling_mask_runtime()
    checks = (
        _check_native_sampling_mask(args.check_sort_reuse) if native else _check_legacy_top_p(args.check_sort_reuse)
    )

    if args.url:
        from slime.utils.types import _extract_rollout_top_p_token_data

        payload = {
            "text": "Write a CUDA kernel that adds two arrays. Explain your approach.",
            "return_logprob": True,
            "sampling_params": {
                "max_new_tokens": 128,
                "ignore_eos": True,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": -1,
            },
        }
        if native:
            payload["return_sampling_mask"] = True
        else:
            payload["sampling_params"]["custom_params"] = {"return_top_p_token_ids": True}
        request = Request(
            args.url.rstrip("/") + "/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with build_opener(ProxyHandler({})).open(request, timeout=180) as response:
            metadata = json.load(response)["meta_info"]
        count = metadata["completion_tokens"]
        ids, offsets = _extract_rollout_top_p_token_data(metadata, expected_num_tokens=count)
        assert count == 128 and len(offsets) == count + 1
        assert all(offsets[i + 1] > offsets[i] for i in range(count))
        assert len(metadata["output_token_logprobs"]) == count
        if native:
            assert len(metadata["output_token_sampling_logprobs"]) == count
            assert all(math.isfinite(value) for value in metadata["output_token_sampling_logprobs"])
        else:
            assert all(math.isfinite(item[0]) for item in metadata["output_token_logprobs"])
        checks.append({"http_tokens": count, "kept_ids": len(ids), "finite_logprobs": True})
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
