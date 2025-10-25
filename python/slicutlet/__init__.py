"""
SLICUTLET: Python bindings for SLICOT C translations
====================================================

A Python interface to selected SLICOT routines translated to modern C.

Refer to individual function docstrings for usage details.
"""

from slicutlet.pyslicutlet import (
    py_mc01td as mc01td,
    py_ab04md as ab04md,
    py_ab05md as ab05md,
    py_ab07nd as ab07nd,
    py_mb03oy as mb03oy
)

__version__ = "0.0.1"
__all__ = ["mc01td", "ab04md", "ab05md", "ab07nd", "mb03oy"]
