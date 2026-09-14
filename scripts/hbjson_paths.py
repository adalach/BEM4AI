"""Resolve HBJSON index paths for the converter and BEM review server."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _row_value(row: object, name: str) -> object:
    getter = getattr(row, "get", None)
    if callable(getter):
        return getter(name)
    return getattr(row, name, None)


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _candidate_path(value: object, hbjson_dir: Path) -> Path | None:
    text = _clean_text(value)
    if text is None:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = hbjson_dir / path
    return path.resolve()


def resolve_hbjson_path(row: object, hbjson_dir: str | Path) -> Path | None:
    """Return the first existing indexed path, or the standard sample path."""
    hbjson_dir = Path(hbjson_dir).expanduser().resolve()
    candidates = [
        _candidate_path(_row_value(row, "file_path"), hbjson_dir),
        _candidate_path(_row_value(row, "bhjsons"), hbjson_dir),
    ]

    sample_id = _clean_text(_row_value(row, "sample_id"))
    sample_path = (hbjson_dir / f"{sample_id}.hbjson").resolve() if sample_id else None
    candidates.append(sample_path)

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is not None and candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)

    for candidate in unique_candidates:
        if candidate.is_file():
            return candidate
    return sample_path or (unique_candidates[0] if unique_candidates else None)
