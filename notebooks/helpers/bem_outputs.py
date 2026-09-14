"""Preview, export, validate, and index the Honeybee models created in Stage 5."""

from __future__ import annotations

import json
import runpy
from collections.abc import Mapping, Sequence
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from honeybee.cli.validate import validate_model
from honeybee.model import Model

from .utils import shorten

__all__ = ["export_validate_and_index", "load_plotters", "preview_models"]


def load_plotters(project_root: Path):
    """Load the static and interactive renderers kept under `scripts/`."""
    directory = project_root / "scripts" / "jupyter-honeybee-plot"
    paths = {
        "static": directory / "plot_honeybee_models.py",
        "interactive": directory / "plot_honeybee_model.py",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Honeybee plotting helpers: {missing}")
    static = runpy.run_path(str(paths["static"]))["plot_honeybee_models"]
    interactive = runpy.run_path(str(paths["interactive"]))["plot_honeybee_model"]
    return static, interactive


def preview_models(
    plotter,
    models: Mapping[str, Model],
    buildings: pd.DataFrame,
    sample_ids: Sequence[str],
    *,
    title: str,
) -> None:
    """Show selected models in one compact horizontal row."""
    selected_ids = [str(sample_id) for sample_id in sample_ids if str(sample_id) in models]
    if not selected_ids:
        raise ValueError("No Honeybee models are available for preview.")

    index = buildings.assign(sample_id=buildings["sample_id"].astype(str)).set_index(
        "sample_id"
    )
    titles = []
    for sample_id in selected_ids:
        row = index.loc[sample_id]
        zones_per_level = sum("_L0" in room.identifier for room in models[sample_id].rooms)
        titles.append(
            f"{sample_id} | L={int(row['levels_est'])}, "
            f"h={float(row['level_height_m']):.1f} m | Z={zones_per_level}"
        )
    plotter(
        models=[models[sample_id] for sample_id in selected_ids],
        model_titles=titles,
        main_title=title,
        grid=(len(selected_ids), 1),
        view="iso",
        camera_zoom=1.25,
        background_color="white",
        surface_opacity=1.0,
        show_wireframe=True,
        wireframe_line_width=0.6,
        wireframe_show_vertices=False,
        base_model_height=380,
        show=True,
    )


def _honeybee_validation(path: Path) -> tuple[bool, str]:
    """Run Honeybee's structured validator and return a short failure message."""
    try:
        report = json.loads(
            validate_model(str(path), extension="All", json=True, output_file=None)
        )
    except Exception as error:
        return False, shorten(str(error))
    if report.get("valid"):
        return True, ""
    details = {"fatal_error": report.get("fatal_error"), "errors": report.get("errors", [])}
    return False, shorten(json.dumps(details, ensure_ascii=False))


def _inspect_export(path: Path, maximum_zones: int, minimum_windows: int) -> dict:
    """Validate one HBJSON and apply the dataset's Zone and window limits."""
    reasons = []
    honeybee_valid, message = _honeybee_validation(path)
    if not honeybee_valid:
        reasons.append("honeybee_validation_failed")

    zones = pd.NA
    windows = pd.NA
    try:
        model = Model.from_hbjson(str(path))
        zones = len(model.rooms)
        windows = len(model.apertures)
        if zones < 1:
            reasons.append("no_zones")
        if zones > maximum_zones:
            reasons.append(f"zones_above_{maximum_zones}")
        if windows < minimum_windows:
            reasons.append("no_windows")
    except Exception as error:
        reasons.append("hbjson_parse_failed")
        message = shorten(str(error))
    return {
        "sample_id": path.stem,
        "n_zones": zones,
        "n_windows": windows,
        "removal_reasons": list(dict.fromkeys(reasons)),
        "validation_message": message,
    }


def _write_model_indexes(
    buildings: gpd.GeoDataFrame,
    valid_ids: set[str],
    _output_dir: Path,
    interim_dir: Path,
    has_neighbor_shades: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the full model table and the compact Stage 7 handoff index."""
    full = buildings[buildings["sample_id"].astype(str).isin(valid_ids)].copy()
    full["sample_id"] = full["sample_id"].astype(str)
    full = full.drop(
        columns=["building_id", "source_sample_id", "osm_meta_json"], errors="ignore"
    )
    full["has_neighbor_shades"] = bool(has_neighbor_shades)
    full["model_id"] = full["sample_id"]
    full["bhjsons"] = full["sample_id"].map(lambda sample_id: f"{sample_id}.hbjson")
    full["file_path"] = full["bhjsons"]
    compact = full[
        ["sample_id", "model_id", "city_key", "has_neighbor_shades", "bhjsons", "file_path"]
    ].drop_duplicates("sample_id").reset_index(drop=True)
    full = full.reset_index(drop=True)

    full.to_pickle(interim_dir / "hbjsons_df_full.pkl")
    compact.to_pickle(interim_dir / "hbjsons_df.pkl")
    full.to_csv(interim_dir / "buildings_bem.csv", index=False)
    return full, compact


def export_validate_and_index(
    models: dict[str, Model],
    buildings: gpd.GeoDataFrame,
    output_dir: Path,
    interim_dir: Path,
    *,
    has_neighbor_shades: bool,
    minimum_windows: int = 1,
    maximum_zones: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Export HBJSONs, remove unusable models, and write the Stage 5 indexes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_ids = set(models)
    for path in output_dir.glob("*.hbjson"):
        if path.stem not in model_ids:
            path.unlink()
    for sample_id, model in models.items():
        model.to_hbjson(str(output_dir / f"{sample_id}.hbjson"))

    checks = [
        _inspect_export(output_dir / f"{sample_id}.hbjson", maximum_zones, minimum_windows)
        for sample_id in tqdm(
            models,
            total=len(models),
            desc="Validating HBJSON models",
            unit="model",
        )
    ]
    checks_df = pd.DataFrame(checks)
    rejected = checks_df[checks_df["removal_reasons"].map(bool)].copy()
    valid_ids = set(checks_df.loc[~checks_df["removal_reasons"].map(bool), "sample_id"])

    for sample_id in set(models).difference(valid_ids):
        models.pop(sample_id, None)
        path = output_dir / f"{sample_id}.hbjson"
        if path.is_file():
            path.unlink()

    full, compact = _write_model_indexes(
        buildings, valid_ids, output_dir, interim_dir, has_neighbor_shades
    )
    return full, compact, rejected.reset_index(drop=True)
