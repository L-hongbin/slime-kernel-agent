"""DeepSeek-V4-Flash mcore-port scaffold (slime-local).

M0: a self-contained V4 model module that wires in the three validated tilelang
kernels (A1 attention / B2 compression / B1 mHC).  It does NOT reuse mcore's
``TransformerBlock`` / ``get_gpt_decoder_block_spec`` (those carry a single
``[S,B,H]`` stream + BDA residuals and don't pass ``input_ids``); see
``decoder.py`` for why.  Full HF<->mcore state-dict mapping / bridge is M5.
"""

from .attention import V4Attention
from .compressor import V4CSACompressor, V4HCACompressor
from .decoder import V4DecoderLayer, V4HyperConnection, V4Model

# M-impl: the mcore LanguageModule (TP=PP=EP=1) + slime provider. Imported lazily by
# name to avoid importing megatron.core at package import time (the M0/M1 torch path
# does not need it); see mcore_model.py / model_provider.py.

__all__ = [
    "V4Model",
    "V4DecoderLayer",
    "V4HyperConnection",
    "V4Attention",
    "V4CSACompressor",
    "V4HCACompressor",
]
