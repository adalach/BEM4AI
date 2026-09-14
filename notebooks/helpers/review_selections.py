"""Apply exclusions saved by the footprint, building, and BEM review apps."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .pickle_compat import read_pickle_compat


SAMPLE_ID_COLUMN = "sample_id"
REMOVED_BUILDINGS_KEY = "buildings_review_removed"
PRESERVED_BUILDING_TABLES = {"buildings_m_bad", REMOVED_BUILDINGS_KEY}

__all__ = [
    "apply_bem_review_selections",
    "apply_building_review_selections",
    "apply_footprint_review_selections",
]


def _load_selection_ids(csv_path: str | Path) -> list[str]:
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        return []

    selections = pd.read_csv(path, dtype={SAMPLE_ID_COLUMN: "string"})
    if SAMPLE_ID_COLUMN not in selections.columns:
        raise KeyError(f"{path} must contain a '{SAMPLE_ID_COLUMN}' column.")

    ordered_ids: list[str] = []
    seen: set[str] = set()
    for value in selections[SAMPLE_ID_COLUMN].dropna():
        sample_id = str(value).strip()
        if sample_id and sample_id not in seen:
            ordered_ids.append(sample_id)
            seen.add(sample_id)
    return ordered_ids


def _require_sample_table(value: object, source: Path | str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame) or SAMPLE_ID_COLUMN not in value.columns:
        raise KeyError(f"{source} must contain a table with a '{SAMPLE_ID_COLUMN}' column.")
    return value


def _sample_ids(value: object) -> set[str]:
    if not isinstance(value, pd.DataFrame) or SAMPLE_ID_COLUMN not in value.columns:
        return set()
    return set(value[SAMPLE_ID_COLUMN].dropna().astype(str))


def _partition_selected(
    frame: pd.DataFrame,
    selected_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = frame[SAMPLE_ID_COLUMN].astype("string").isin(selected_ids)
    return frame.loc[~selected].copy(), frame.loc[selected].copy()


def _restrict_to_ids(frame: pd.DataFrame, active_ids: set[str]) -> pd.DataFrame:
    active = frame[SAMPLE_ID_COLUMN].astype("string").isin(active_ids)
    return frame.loc[active].copy()


def _merge_removed_rows(existing: object, removed: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in (existing, removed) if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not frames:
        return removed.iloc[0:0].copy()

    merged = pd.concat(frames, ignore_index=True)
    if SAMPLE_ID_COLUMN in merged.columns:
        merged = merged.drop_duplicates(subset=[SAMPLE_ID_COLUMN], keep="first")
    return merged.reset_index(drop=True)


def _atomic_pickle(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.review.tmp")
    try:
        pd.to_pickle(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.review.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_table(path: Path) -> pd.DataFrame:
    value = (
        pd.read_csv(path, dtype={SAMPLE_ID_COLUMN: "string"})
        if path.suffix.lower() == ".csv"
        else read_pickle_compat(path)
    )
    return _require_sample_table(value, path)


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        _atomic_csv(frame, path)
    else:
        _atomic_pickle(frame, path)


def _available_archive_path(path: Path) -> Path:
    if not path.exists():
        return path

    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem}.review-{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def apply_footprint_review_selections(
    buildings_path: str | Path,
    neighbors_path: str | Path,
    csv_path: str | Path,
    removed_path: str | Path,
) -> dict[str, object]:
    """Apply footprint-selector exclusions to the Stage 3 geometry tables."""
    buildings_path = Path(buildings_path).expanduser().resolve()
    neighbors_path = Path(neighbors_path).expanduser().resolve()
    removed_path = Path(removed_path).expanduser().resolve()

    missing = [str(path) for path in (buildings_path, neighbors_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing footprint-review artifacts: " + ", ".join(missing))

    buildings = _require_sample_table(read_pickle_compat(buildings_path), buildings_path)
    neighbors = _require_sample_table(read_pickle_compat(neighbors_path), neighbors_path)
    existing_removed = read_pickle_compat(removed_path) if removed_path.is_file() else None

    selected_ids = _load_selection_ids(csv_path)
    selected_set = set(selected_ids)
    current_ids = _sample_ids(buildings)
    known_ids = current_ids | _sample_ids(existing_removed)
    if not selected_ids:
        return {
            "selection_csv": str(Path(csv_path).expanduser().resolve()),
            "selected_ids": [],
            "unknown_ids": [],
            "removed_now": [],
            "active_buildings": len(buildings),
            "removed_neighbor_rows": 0,
        }

    active_buildings, removed_buildings = _partition_selected(buildings, selected_set)
    active_neighbors, removed_neighbors = _partition_selected(neighbors, selected_set)
    removed_now = _sample_ids(removed_buildings)

    if removed_now:
        _atomic_pickle(active_buildings, buildings_path)
        _atomic_pickle(active_neighbors, neighbors_path)
        _atomic_pickle(_merge_removed_rows(existing_removed, removed_buildings), removed_path)

    return {
        "selection_csv": str(Path(csv_path).expanduser().resolve()),
        "selected_ids": selected_ids,
        "unknown_ids": [sample_id for sample_id in selected_ids if sample_id not in known_ids],
        "removed_now": sorted(removed_now),
        "active_buildings": len(active_buildings),
        "removed_neighbor_rows": len(removed_neighbors),
    }


def apply_building_review_selections(
    transformed_paths: Iterable[str | Path],
    csv_path: str | Path,
) -> dict[str, object]:
    """Apply building-selector exclusions to transformed Stage 4 payloads."""
    paths = [Path(path).expanduser().resolve() for path in transformed_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing building-review payloads: " + ", ".join(missing))

    selected_ids = _load_selection_ids(csv_path)
    selected_set = set(selected_ids)
    payloads: dict[Path, dict[str, object]] = {}
    known_ids: set[str] = set()

    for path in paths:
        payload = read_pickle_compat(path)
        if not isinstance(payload, dict):
            raise TypeError(f"{path} does not contain a transformed payload dictionary.")
        buildings = _require_sample_table(payload.get("buildings_m"), f"{path}:buildings_m")
        payloads[path] = payload
        known_ids.update(_sample_ids(buildings))
        known_ids.update(_sample_ids(payload.get(REMOVED_BUILDINGS_KEY)))

    if not selected_ids:
        return {
            "selection_csv": str(Path(csv_path).expanduser().resolve()),
            "selected_ids": [],
            "unknown_ids": [],
            "payloads": {
                path.name: {
                    "removed_now": [],
                    "active_buildings": len(payload["buildings_m"]),
                    "removed_table_rows": {},
                }
                for path, payload in payloads.items()
            },
        }

    reports: dict[str, dict[str, object]] = {}
    for path, payload in payloads.items():
        # Already validated in the loop above, where known_ids was collected.
        active_buildings, removed_buildings = _partition_selected(payload["buildings_m"], selected_set)
        active_ids = _sample_ids(active_buildings)
        removed_now = _sample_ids(removed_buildings)

        removed_table_rows: dict[str, int] = {}
        updated = dict(payload)
        for key, value in payload.items():
            if (
                isinstance(value, pd.DataFrame)
                and SAMPLE_ID_COLUMN in value.columns
                and key not in PRESERVED_BUILDING_TABLES
            ):
                filtered = _restrict_to_ids(value, active_ids)
                removed_table_rows[key] = len(value) - len(filtered)
                updated[key] = filtered

        updated[REMOVED_BUILDINGS_KEY] = _merge_removed_rows(
            payload.get(REMOVED_BUILDINGS_KEY),
            removed_buildings,
        )

        if removed_now:
            _atomic_pickle(updated, path)

        reports[path.name] = {
            "removed_now": sorted(removed_now),
            "active_buildings": len(active_buildings),
            "removed_table_rows": removed_table_rows,
        }

    return {
        "selection_csv": str(Path(csv_path).expanduser().resolve()),
        "selected_ids": selected_ids,
        "unknown_ids": [sample_id for sample_id in selected_ids if sample_id not in known_ids],
        "payloads": reports,
    }


def apply_bem_review_selections(
    csv_path: str | Path,
    table_paths: Sequence[str | Path],
    artifact_dirs: Sequence[str | Path],
    removed_tables_path: str | Path,
    rejected_artifact_root: str | Path,
) -> dict[str, object]:
    """Apply BEM-selector exclusions to model indexes and exported model files."""
    selected_ids = _load_selection_ids(csv_path)
    selected_set = set(selected_ids)
    removed_tables_path = Path(removed_tables_path).expanduser().resolve()
    rejected_artifact_root = Path(rejected_artifact_root).expanduser().resolve()

    if not selected_ids:
        return {
            "selection_csv": str(Path(csv_path).expanduser().resolve()),
            "selected_ids": [],
            "removed_model_ids": [],
            "unknown_ids": [],
            "tables": {},
            "moved_files": [],
        }

    previous_removed: dict[str, pd.DataFrame] = {}
    if removed_tables_path.is_file():
        value = read_pickle_compat(removed_tables_path)
        if not isinstance(value, dict):
            raise TypeError(f"{removed_tables_path} must contain a dictionary of removed tables.")
        previous_removed = value

    known_ids: set[str] = set()
    reports: dict[str, dict[str, int]] = {}
    updated_removed = dict(previous_removed)
    removed_tables_changed = False
    removed_model_ids: set[str] = set()

    for raw_path in table_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            continue

        frame = _load_table(path)
        known_ids.update(_sample_ids(frame))
        known_ids.update(_sample_ids(previous_removed.get(path.name)))
        active, removed = _partition_selected(frame, selected_set)
        if not removed.empty:
            removed_model_ids.update(_sample_ids(removed))
            updated_removed[path.name] = _merge_removed_rows(previous_removed.get(path.name), removed)
            _write_table(active, path)
            removed_tables_changed = True
        reports[path.name] = {"removed_now": len(removed), "active_rows": len(active)}

    moved_files: list[str] = []
    for raw_directory in artifact_dirs:
        directory = Path(raw_directory).expanduser().resolve()
        if not directory.is_dir():
            continue

        selected_files = sorted(
            path for path in directory.iterdir() if path.is_file() and path.stem in selected_set
        )
        for source in selected_files:
            known_ids.add(source.stem)
            removed_model_ids.add(source.stem)
            destination = rejected_artifact_root / directory.name / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination = _available_archive_path(destination)
            shutil.move(str(source), str(destination))
            moved_files.append(str(destination))

    if removed_tables_changed:
        _atomic_pickle(updated_removed, removed_tables_path)

    return {
        "selection_csv": str(Path(csv_path).expanduser().resolve()),
        "selected_ids": selected_ids,
        "removed_model_ids": sorted(removed_model_ids),
        "unknown_ids": [sample_id for sample_id in selected_ids if sample_id not in known_ids],
        "tables": reports,
        "moved_files": moved_files,
    }
