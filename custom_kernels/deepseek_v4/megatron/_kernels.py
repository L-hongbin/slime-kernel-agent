"""Kernel-import shim for the V4-Flash mcore scaffold (M0).

The three validated tilelang kernels live in sibling directories
(``../attention``, ``../compression``, ``../mhc``) and were written to be run
*from inside their own directory* (e.g. compression's ``kernel.py`` does
``from reference import csa_build_overlap``).  Rather than refactor the kernel
packages (out of scope for M0 / they're under active optimisation), this module
loads each ``kernel.py`` by file path with its own directory on ``sys.path`` so
its sibling-relative imports resolve, and re-exports the public entry points:

    v4flash_attention   (A1)  -- attention/kernel.py
    hca_compress        (B2)  -- compression/kernel.py
    csa_compress        (B2)  -- compression/kernel.py
    hyper_connection            (B1, hybrid, any dtype)  -- mhc/kernel.py
    hyper_connection_sglang     (B1, sglang fwd, bf16)   -- mhc/kernel.py

Import order matters: compression's ``kernel.py`` imports ``reference`` from its
own dir, so we register that dir on ``sys.path`` (and load ``reference`` first)
before exec'ing the kernel module.
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_KROOT = os.path.dirname(_HERE)  # custom_kernels/deepseek_v4/
_ATTN_DIR = os.path.join(_KROOT, "attention")
_COMP_DIR = os.path.join(_KROOT, "compression")
_MHC_DIR = os.path.join(_KROOT, "mhc")


def _load(name: str, path: str, extra_syspath: list[str]):
    for p in extra_syspath:
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# compression's kernel.py does `from reference import csa_build_overlap`; make
# sure *its* reference.py (not attention's) is the one that resolves.
_comp_ref = _load("v4_comp_reference", os.path.join(_COMP_DIR, "reference.py"), [_COMP_DIR])
sys.modules["reference"] = _comp_ref  # satisfy the kernel's bare `import reference`

_attn = _load("v4_a1_kernel", os.path.join(_ATTN_DIR, "kernel.py"), [_ATTN_DIR])
_comp = _load("v4_b2_kernel", os.path.join(_COMP_DIR, "kernel.py"), [_COMP_DIR])
_mhc = _load("v4_b1_kernel", os.path.join(_MHC_DIR, "kernel.py"), [_MHC_DIR])

v4flash_attention = _attn.v4flash_attention
hca_compress = _comp.hca_compress
csa_compress = _comp.csa_compress
hyper_connection = _mhc.hyper_connection
hyper_connection_sglang = _mhc.hyper_connection_sglang

__all__ = [
    "v4flash_attention",
    "hca_compress",
    "csa_compress",
    "hyper_connection",
    "hyper_connection_sglang",
]
