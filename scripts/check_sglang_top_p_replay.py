"""Manually validate the installed SGLang top-p replay patch, optionally over HTTP.

Run with the intended SGLang overlay first in PYTHONPATH. This is a runtime
preflight, not a CI test: it requires the patched SGLang installation and, for
--url, a live speculative-decoding server.
"""

import argparse
import json
import math
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import ProxyHandler, Request, build_opener

import torch
from sglang.srt.layers import logprob_processor, sampler
from sglang.srt.layers.logprob_processor import compute_spec_v2_logprobs, renorm_logprob_over_top_p
from sglang.srt.sampling.sampling_params import TOP_K_ALL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Optional live SGLang server URL, without /generate.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--check-sort-reuse", action="store_true", help="Require one full sort per replay step.")
    args = parser.parse_args()
    torch.set_default_device(args.device)
    torch.manual_seed(17)
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
            if args.check_sort_reuse:
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

    if args.check_sort_reuse:
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
                "custom_params": {"return_top_p_token_ids": True},
            },
        }
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
        assert all(math.isfinite(item[0]) for item in metadata["output_token_logprobs"])
        checks.append({"http_tokens": count, "kept_ids": len(ids), "finite_logprobs": True})
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
