"""DrKernel dataset cleanup implementation and command-line entry points.

Modules deliberately stay lazy so importing static analysis does not load
Torch or PyArrow and ``python -m`` entry points do not execute a module twice.
"""
