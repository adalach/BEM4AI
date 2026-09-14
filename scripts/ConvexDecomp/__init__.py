"""Convex decomposition of OSM building footprints.

Derived from ConvexDecomp (https://github.com/adalach/ConvexDecomp), reduced to
what the BEM4AI pipeline runs and extended where it needed more: alignment of a
neighbour's edges to a reference building, and a decomposition runner that
labels every zone with the part of the building it came from. See README.md.

The modules read from `helpers.utils` rather than carrying their own copies of
the same geometry helpers, so the notebooks directory has to be importable. It
already is inside a notebook, where Jupyter's working directory is
`notebooks/`; the two lines below make the package work from anywhere else -
pytest, a script, a plain interpreter at the repository root.
"""

import sys
from pathlib import Path

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2] / "notebooks"
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))
