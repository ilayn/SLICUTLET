"""
SLICUTLET: Python bindings for SLICOT C translations
====================================================

A Python interface to selected SLICOT routines translated to modern C.

Refer to individual function docstrings for usage details.
"""

from slicutlet.pyslicutlet import (
    py_ab01nd as ab01nd,
    py_ab04md as ab04md,
    py_ab05md as ab05md,
    py_ab05nd as ab05nd,
    py_ab07nd as ab07nd,
    py_mb01pd as mb01pd,
    py_mb01qd as mb01qd,
    py_mb03oy as mb03oy,
    py_mc01td as mc01td,
)

__version__ = "0.0.1"
__all__ = [
    "ab01nd",
    "ab04md",
    "ab05md",
    "ab05nd",
    "ab07nd",
    "mb01pd",
    "mb01qd",
    "mb03oy",
    "mc01td"
]
