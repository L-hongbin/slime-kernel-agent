#!/usr/bin/env python3
"""Patch sglang so the DeepSeek-V4 C4 lightning-indexer runs DENSE when
V4_DENSE_ATTENTION=<index_topk> is set (>0), matching the train forward which
drops the indexer (attends all compressed-KV positions).

Two read paths of config.index_topk must honor the override:
  1. get_dsa_index_topk(config)  -> prefill dense-threshold + deepseek_v2 path
  2. C4Indexer.__init__          -> decode top-k (indexer.py: self.index_topk = config.index_topk)

Setting index_topk >= max_context/compress_ratio makes topk_len=min((pos+1)//CR, TOP_K)
select ALL compressed positions -> dense. Idempotent; env-gated (off by default).
Run on every node that serves sglang. Usage: python patch_sglang_dense_attn.py
"""

SGL = "/sgl-workspace/sglang/python/sglang"

MC = f"{SGL}/srt/configs/model_config.py"
IDX = f"{SGL}/srt/layers/attention/dsv4/indexer.py"

MARK = "V4_DENSE_ATTENTION"  # idempotency marker


def patch_model_config():
    t = open(MC).read()
    if MARK in t:
        return "model_config: already patched"
    old = "def get_dsa_index_topk(config: PretrainedConfig) -> int:\n    assert is_deepseek_dsa(config)\n    return config.index_topk"
    new = (
        "def get_dsa_index_topk(config: PretrainedConfig) -> int:\n"
        "    assert is_deepseek_dsa(config)\n"
        "    import os as _os  # V4_DENSE_ATTENTION: force dense C4 attention to match train\n"
        "    _d = _os.environ.get('V4_DENSE_ATTENTION')\n"
        "    if _d and int(_d) > 0:\n"
        "        return int(_d)\n"
        "    return config.index_topk"
    )
    assert old in t, "get_dsa_index_topk anchor not found"
    open(MC, "w").write(t.replace(old, new, 1))
    return "model_config: patched"


def patch_indexer():
    t = open(IDX).read()
    if MARK in t:
        return "indexer: already patched"
    old = "        self.index_topk = config.index_topk"
    new = (
        "        import os as _os  # V4_DENSE_ATTENTION: force dense C4 attention (decode top-k)\n"
        "        _d = _os.environ.get('V4_DENSE_ATTENTION')\n"
        "        self.index_topk = int(_d) if (_d and int(_d) > 0) else config.index_topk"
    )
    assert old in t, "indexer index_topk anchor not found"
    open(IDX, "w").write(t.replace(old, new, 1))
    return "indexer: patched"


if __name__ == "__main__":
    print(patch_model_config())
    print(patch_indexer())
    import py_compile

    py_compile.compile(MC, doraise=True)
    py_compile.compile(IDX, doraise=True)
    print("compile OK")
