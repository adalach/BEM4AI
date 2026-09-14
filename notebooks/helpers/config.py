"""Load validated JSON configuration from the repository config directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import PROJECT_ROOT


def load_json_config(name: str) -> dict[str, Any]:
    """Load one top-level JSON object from ``config/`` with useful errors."""
    if Path(name).name != name or not name.endswith(".json"):
        raise ValueError(f"Configuration name must be a JSON filename: {name!r}")

    path = PROJECT_ROOT / "config" / name
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Configuration root must be a JSON object: {path}")
    return value
