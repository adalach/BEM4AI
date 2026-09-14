"""Repository paths, and the one import that makes `helpers` and `scripts` reachable.

Every notebook starts with a single line::

    from helpers.paths import PROJECT_ROOT, DATA_DIR, INTERIM_DIR

It also provides :func:`display_path`, which every notebook should use when it
prints a path, so that printed output is identical on every machine.

`helpers` resolves without any setup because Jupyter runs a notebook with the
notebook's own directory (`notebooks/`) as the working directory, whether the
server was started there or at the repository root.

Importing this module then puts the repository root on `sys.path`, so
`scripts.<module>` is importable from a notebook as well. The root itself is
derived from this file's location rather than from the working directory, so it
is correct no matter how the kernel was started.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Repository root: this file is <root>/notebooks/helpers/paths.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"

# Created on import, so that a notebook's first write does not have to check.
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

for _import_root in (PROJECT_ROOT, NOTEBOOKS_DIR):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))


def display_path(path: Path | str) -> str:
    """Return ``path`` as it should be printed: relative to the repository root.

    Notebook output then reads the same on every machine, instead of embedding
    whoever ran it last. A path outside the repository is returned in full, so that
    printing one can never raise; nothing shorter would identify it anyway.

    Use this for printing only. A path being *stored* in a table should use
    ``path.relative_to(PROJECT_ROOT).as_posix()`` directly, so that a path from
    outside the repository raises instead of being silently written as absolute.
    """
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:  # outside the repository
        return str(path)


__all__ = [
    "PROJECT_ROOT",
    "NOTEBOOKS_DIR",
    "SCRIPTS_DIR",
    "DATA_DIR",
    "INTERIM_DIR",
    "display_path",
]
