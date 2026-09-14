"""Read pickles written on another machine or another NumPy.

The interim artifacts travel: written on the machine that ran a stage,
read on whatever runs the next one. Two things break when they do. A
``WindowsPath`` inside a pickle cannot be instantiated on Linux, or a
``PosixPath`` on Windows, even though the path itself is never used. And a
pickle written under NumPy 2 refers to ``numpy._core``, which NumPy 1 does
not have.

Both are fixed the same way: teach the unpickler where to look, for the
duration of the read only.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import os
import pathlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd


__all__ = ["read_pickle_compat"]

_NUMPY_ALIAS_MAP = {
    "numpy._core": "numpy.core",
    "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
    "numpy._core.multiarray": "numpy.core.multiarray",
    "numpy._core.numeric": "numpy.core.numeric",
    "numpy._core.numerictypes": "numpy.core.numerictypes",
    "numpy._core.overrides": "numpy.core.overrides",
    "numpy._core.umath": "numpy.core.umath",
}


def install_numpy_pickle_aliases() -> None:
    """Expose NumPy 2-style internal module paths for NumPy 1.x unpickling."""
    for alias, target in _NUMPY_ALIAS_MAP.items():
        if alias in sys.modules:
            continue
        try:
            sys.modules[alias] = importlib.import_module(target)
        except ModuleNotFoundError:
            continue


@contextmanager
def portable_pathlib_classes():
    """Temporarily map foreign-platform pathlib classes while unpickling."""
    original_posix_path = pathlib.PosixPath
    original_windows_path = pathlib.WindowsPath
    try:
        if os.name == "nt":
            pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc]
        else:
            pathlib.WindowsPath = pathlib.PosixPath  # type: ignore[misc]
        yield
    finally:
        pathlib.PosixPath = original_posix_path  # type: ignore[misc]
        pathlib.WindowsPath = original_windows_path  # type: ignore[misc]


def read_pickle_compat(path: str | Path, **kwargs: Any):
    """Read pickles across NumPy versions and operating-system path classes."""
    install_numpy_pickle_aliases()
    with portable_pathlib_classes():
        try:
            return pd.read_pickle(path, **kwargs)
        except ModuleNotFoundError as exc:
            if "numpy._core" not in str(exc):
                raise
            install_numpy_pickle_aliases()
            return pd.read_pickle(path, **kwargs)
