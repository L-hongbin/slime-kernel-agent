#!/usr/bin/env python3
"""Verify the Qwen3 thinking-template prefix-cache breakage and the preserve_thinking fix.

See handoffs/in_progress/HANDOFF_CACHE_DRKERNEL_HYBRID.md.
Run on a box with transformers + the model tokenizer (e.g. 192.168.16.22):
    python3 preserve_thinking_prefix_match.py

It faithfully simulates one multi-turn boundary:
  turn-1 prompt (add_generation_prompt injects '<think>\n')
  -> model generates  reasoning + '</think>' + answer + '<|im_end|>'
  -> harness strips stop markers, appends as assistant message, appends tool user msg
  -> turn-2 prompt re-rendered through chat template, re-encoded
and reports the radix longest-common-prefix of (turn2 ids) vs (turn1 ids + generated ids)
for preserve_thinking in {False, True}.
"""
from transformers import AutoTokenizer

MODEL = "/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def enc(s):
    return tok.encode(s, add_special_tokens=False)


def lcp(a, b):
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def strip_markers(t):
    for m in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        t = t.replace(m, "")
    return t.strip()


problem = "Optimize torch.relu(x)*2.0 with a custom CUDA kernel. Return a complete extension."
reasoning = (
    "Let me think. The op is elementwise: relu then scale by 2. I can fuse both into a single "
    "kernel to avoid an extra global-memory round trip. Grid-stride loop, 256 threads/block, "
    "bounds check for the tail. Validate float first, then consider float4 vectorization."
)
answer = (
    "Here's the fused kernel:\n\n```cuda\n__global__ void rs(const float* x, float* y, int n){\n"
    "  int i=blockIdx.x*blockDim.x+threadIdx.x;\n  if(i<n){float v=x[i]; y[i]=(v>0.f?v:0.f)*2.f;}\n}\n```\n"
    "Launch blocks=(n+255)/256."
)
tool = "Compilation failed: error: identifier 'blockIdx' undefined in host code. nvcc exit 1. (truncated)"

p1 = tok.apply_chat_template([{"role": "user", "content": problem}], tokenize=False, add_generation_prompt=True)
ids1 = enc(p1)
gen_text = reasoning + "\n</think>\n\n" + answer + "<|im_end|>"
G = enc(gen_text)
asst_full = strip_markers(tok.decode(G))
msgs2 = [
    {"role": "user", "content": problem},
    {"role": "assistant", "content": asst_full},
    {"role": "user", "content": tool},
]
tree = ids1 + G

print(f"BPE round-trip identity (encode(decode(G))==G): {enc(tok.decode(G)) == G}")
for pt in (False, True):
    p2 = tok.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True, preserve_thinking=pt)
    ids2 = enc(p2)
    hit = lcp(ids2, tree)
    print(
        f"preserve_thinking={pt}: turn2_prompt={len(ids2)}  radixLCP={hit}/{len(tree)}  "
        f"generated_reused={max(0, hit - len(ids1))}/{len(G)}  reasoning_in_prompt={'Let me think' in p2}"
    )
