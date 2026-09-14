#!/usr/bin/env python3
"""Convert Stage 5 HBJSON models into validated IDF and EPJSON files.

Stage 7 calls this module after weather assignment. It also imports the shared
simulation-parameter and EnergyPlus output-request functions when running the
models. See ``README_convert_hbjson_exports_to_idf_epjson.md`` for the complete
workflow and output contract.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import importlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from honeybee.model import Model
from honeybee_energy.run import run_osw, to_openstudio_sim_folder
from honeybee_energy.simulation.output import SimulationOutput
from honeybee_energy.simulation.parameter import (
    SimulationControl,
    SimulationParameter,
    SizingParameter,
)
from tqdm import tqdm

if __package__:
    from .hbjson_paths import resolve_hbjson_path
else:
    from hbjson_paths import resolve_hbjson_path

try:
    from ladybug.config import folders as lb_folders
except Exception:  # pragma: no cover - optional dependency path helper
    lb_folders = None


# EnergyPlus writes a variable to SQL only when the IDF requests it. These
# variables provide the annual and hourly values used by the result and graph
# extraction stages.
DATASET_COMMON_OUTPUT_VARIABLES = (
    "Facility All Zones Ventilation At Target Voz Time",
    "Facility Any Zone Ventilation Above Target Voz Time",
    "Facility Any Zone Ventilation Below Target Voz Time",
    "Facility Any Zone Ventilation When Unoccupied Time",
    "Site Diffuse Solar Radiation Rate per Area",
    "Site Direct Solar Radiation Rate per Area",
    "Site Outdoor Air Drybulb Temperature",
    "Site Outdoor Air Relative Humidity",
    "Surface Inside Face Conduction Heat Transfer Rate",
    "Surface Inside Face Temperature",
    "Surface Outside Face Conduction Heat Transfer Rate",
    "Surface Outside Face Incident Beam Solar Radiation Rate per Area",
    "Surface Outside Face Incident Ground Diffuse Solar Radiation Rate per Area",
    "Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area",
    "Surface Outside Face Incident Solar Radiation Rate per Area",
    "Surface Outside Face Temperature",
    "Surface Window Heat Gain Rate",
    "Surface Window Heat Loss Rate",
    "Surface Window Transmitted Solar Radiation Rate",
    "Zone Air System Sensible Cooling Rate",
    "Zone Air System Sensible Heating Rate",
    "Zone Air Temperature",
    "Zone Cooling Setpoint Not Met Time",
    "Zone Cooling Setpoint Not Met While Occupied Time",
    "Zone Heating Setpoint Not Met Time",
    "Zone Heating Setpoint Not Met While Occupied Time",
    "Zone Ideal Loads Zone Total Cooling Rate",
    "Zone Ideal Loads Zone Total Heating Rate",
    "Zone Mean Air Temperature",
    "Zone Operative Temperature",
    "Zone Total Internal Total Heating Rate",
    "Zone Windows Total Heat Gain Rate",
    "Zone Windows Total Heat Loss Rate",
)
DATASET_HOURLY_ONLY_OUTPUT_VARIABLES = (
    "Enclosure Exterior Windows Total Transmitted Beam Solar Radiation Rate",
    "Enclosure Interior Windows Total Transmitted Beam Solar Radiation Rate",
)
REPLACED_OUTPUT_OBJECT_TYPES = frozenset(
    {
        "OutputControl:Table:Style",
        "Output:Table:SummaryReports",
        "Output:VariableDictionary",
        "OutputControl:ReportingTolerances",
        "Output:SQLite",
        "Output:Variable",
    }
)


@dataclass(frozen=True)
class ExportConfig:
    """Filesystem locations used by one conversion run."""

    hbjson_dir: Path
    hbjson_index: Path
    idf_dir: Path
    epjson_dir: Path
    temp_dir: Path
    summary_csv: Path


@dataclass(frozen=True)
class SiteLocation:
    """EnergyPlus site fields read from the assigned EPW file."""

    name: str
    latitude: float
    longitude: float
    time_zone: float
    elevation: float


@dataclass(frozen=True)
class EnergyPlusStructure:
    """Object counts used to verify the canonical zone-based HVAC structure."""

    version: str | None
    space_count: int
    space_list_count: int
    design_spec_oa_spacelist_count: int
    hvac_template_ideal_loads_count: int
    zone_hvac_ideal_loads_count: int
    zone_hvac_equipment_connections_count: int
    space_hvac_equipment_connections_count: int
    zone_count: int

    @property
    def has_zone_based_hvac(self) -> bool:
        return (
            self.hvac_template_ideal_loads_count == 0
            and self.zone_hvac_ideal_loads_count > 0
            and self.zone_hvac_equipment_connections_count > 0
            and self.space_hvac_equipment_connections_count == 0
        )

    @property
    def has_expanded_space_hvac(self) -> bool:
        # Backward-compatible alias for older summaries and helper scripts.
        return self.has_zone_based_hvac

    def failure_message(self, label: str) -> str:
        issues = []
        if self.hvac_template_ideal_loads_count > 0:
            issues.append("contains HVACTemplate ideal-load objects")
        if self.zone_hvac_ideal_loads_count <= 0:
            issues.append("missing ZoneHVAC:IdealLoadsAirSystem objects")
        if self.zone_hvac_equipment_connections_count <= 0:
            issues.append("missing ZoneHVAC:EquipmentConnections objects")
        if self.space_hvac_equipment_connections_count > 0:
            issues.append("contains SpaceHVAC:EquipmentConnections objects")
        return f"{label} is not in canonical zone-HVAC form: {', '.join(issues)}."


STRUCTURE_SUMMARY_FIELDS = (
    "version",
    "space_count",
    "space_list_count",
    "design_spec_oa_spacelist_count",
    "hvac_template_ideal_loads_count",
    "zone_hvac_ideal_loads_count",
    "zone_hvac_equipment_connections_count",
    "space_hvac_equipment_connections_count",
    "zone_count",
    "has_zone_based_hvac",
    "has_expanded_space_hvac",
)
CONVERSION_RESULT_COLUMNS = (
    "sample_id",
    "city_key",
    "hbjson_path",
    "epw_path",
    "ddy_path",
    "idf_path",
    "epjson_path",
    "status",
    "idf_valid",
    "epjson_valid",
    *(f"idf_{field}" for field in STRUCTURE_SUMMARY_FIELDS),
    *(f"epjson_{field}" for field in STRUCTURE_SUMMARY_FIELDS),
    "elapsed_s",
    "message",
)


def _new_conversion_result(
    *,
    sample_id: str,
    city_key: str,
    hbjson_path: Path,
    epw_path: Path,
    ddy_path: Path,
    idf_path: Path,
    epjson_path: Path,
) -> dict[str, object]:
    """Create one complete summary row before conversion status is known."""
    result: dict[str, object] = dict.fromkeys(CONVERSION_RESULT_COLUMNS)
    result.update(
        {
            "sample_id": sample_id,
            "city_key": city_key,
            "hbjson_path": str(hbjson_path),
            "epw_path": str(epw_path),
            "ddy_path": str(ddy_path),
            "idf_path": str(idf_path),
            "epjson_path": str(epjson_path),
            "status": "unknown",
            "idf_valid": False,
            "epjson_valid": False,
            "message": "",
        }
    )
    return result


# Project paths and input tables

def infer_project_root() -> Path:
    """Return the repository containing this script."""
    return Path(__file__).resolve().parents[1]


def build_export_config(project_root: Path) -> ExportConfig:
    """Build the standard Stage 5 input and Stage 7 output locations."""
    interim_dir = project_root / "data" / "interim"
    output_dir = project_root / "output"
    return ExportConfig(
        hbjson_dir=output_dir / "hbjsons",
        hbjson_index=interim_dir / "hbjsons_df.pkl",
        idf_dir=output_dir / "idf",
        epjson_dir=output_dir / "epjson",
        temp_dir=output_dir / "idf_epjson_conversion_temp",
        summary_csv=interim_dir / "hbjson_to_idf_epjson.csv",
    )


def resolve_energyplus_executable(explicit_path: str | Path | None = None) -> Path:
    """Find EnergyPlus from an argument, environment, PATH, or common installs."""
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    configured_path = os.environ.get("ENERGYPLUS_EXE")
    if configured_path:
        candidates.append(Path(configured_path).expanduser())

    which_energyplus = shutil.which("energyplus")
    if which_energyplus:
        candidates.append(Path(which_energyplus))

    if lb_folders is not None:
        for attr in ("energyplus_path", "energyplus_exe_path"):
            value = getattr(lb_folders, attr, None)
            if value:
                candidates.append(Path(value))

    applications_dir = Path("/Applications")
    if applications_dir.is_dir():
        candidates.extend(
            sorted(
                applications_dir.glob("OpenStudio-*/EnergyPlus/energyplus"),
                reverse=True,
            )
        )
        candidates.extend(
            sorted(
                applications_dir.glob("EnergyPlus-*/energyplus"),
                reverse=True,
            )
        )

    candidates.extend(
        [
            Path("/usr/local/bin/energyplus"),
            Path("/opt/homebrew/bin/energyplus"),
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve the EnergyPlus executable. "
        "Download EnergyPlus from https://energyplus.net/downloads and then add it to PATH, "
        "set ENERGYPLUS_EXE, or pass --energyplus-exe."
    )


def resolve_openstudio_executable() -> Path:
    """Require the OpenStudio CLI used for HBJSON translation."""
    which_openstudio = shutil.which("openstudio")
    if which_openstudio:
        return Path(which_openstudio).resolve()
    raise FileNotFoundError(
        "Could not resolve the OpenStudio CLI executable on PATH. "
        "The measures-only HBJSON -> IDF conversion requires OpenStudio."
    )


def _install_numpy_pickle_compat_aliases() -> None:
    """Allow pickles written with NumPy 2.x internals to load under NumPy 1.x."""
    try:
        importlib.import_module("numpy._core.numeric")
        return
    except ModuleNotFoundError:
        pass

    aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.umath": "numpy.core.umath",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
    }
    for alias_name, target_name in aliases.items():
        try:
            sys.modules.setdefault(alias_name, importlib.import_module(target_name))
        except ModuleNotFoundError:
            continue


def read_pickle_compat(path: Path) -> object:
    """Read a project pickle across supported NumPy layouts."""
    _install_numpy_pickle_compat_aliases()
    return pd.read_pickle(path)


def load_weather_assets(project_root: Path) -> pd.DataFrame:
    """Load and validate the Stage 6 city-to-weather assignment table."""
    path = project_root / "data" / "interim" / "city_weather_assets.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Stage 6 weather assignment table: {path}"
        )

    weather_assets_df = read_pickle_compat(path)
    if not isinstance(weather_assets_df, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame in {path}")

    required = {"city_key", "epw_file", "ddy_file"}
    missing = sorted(required.difference(weather_assets_df.columns))
    if missing:
        raise KeyError(f"Weather assignment table is missing columns: {missing}")

    return weather_assets_df.copy()


def load_hbjson_index(config: ExportConfig) -> pd.DataFrame:
    """Load the Stage 5 model index or reconstruct it from exported HBJSONs."""
    if config.hbjson_index.exists():
        obj = read_pickle_compat(config.hbjson_index)
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"Expected a DataFrame in {config.hbjson_index}")
        return obj.copy()

    hbjson_paths = sorted(config.hbjson_dir.glob("*.hbjson"))
    if not hbjson_paths:
        return pd.DataFrame(columns=["sample_id", "city_key", "bhjsons", "file_path"])

    rows = []
    for hbjson_path in hbjson_paths:
        sample_id = hbjson_path.stem
        parts = sample_id.split("_")
        city_key = "_".join(parts[:-1]) if len(parts) > 1 else sample_id
        rows.append(
            {
                "sample_id": sample_id,
                "city_key": city_key,
                "bhjsons": hbjson_path.name,
                "file_path": hbjson_path.resolve(),
            }
        )
    return pd.DataFrame(rows)


def build_conversion_table(
    project_root: Path,
    config: ExportConfig,
    weather_assets_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join each model to its HBJSON, EPW, and DDY paths through ``city_key``."""
    hbjson_index = load_hbjson_index(config)

    if hbjson_index.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "city_key",
                "hbjson_path",
                "epw_path",
                "ddy_path",
            ]
        )

    if "sample_id" not in hbjson_index.columns:
        if "model_id" in hbjson_index.columns:
            hbjson_index["sample_id"] = hbjson_index["model_id"].astype(str)
        else:
            raise KeyError("HBJSON index is missing 'sample_id'")

    hbjson_index["sample_id"] = hbjson_index["sample_id"].astype(str)

    if "city_key" not in hbjson_index.columns:
        hbjson_index["city_key"] = hbjson_index["sample_id"].map(
            lambda value: "_".join(value.split("_")[:-1]) if "_" in value else value
        )

    if "bhjsons" not in hbjson_index.columns:
        hbjson_index["bhjsons"] = hbjson_index["sample_id"].map(lambda value: f"{value}.hbjson")

    hbjson_index["hbjson_path"] = hbjson_index.apply(
        lambda row: resolve_hbjson_path(row, config.hbjson_dir), axis=1
    )

    merged = hbjson_index.merge(
        weather_assets_df[["city_key", "epw_file", "ddy_file"]].drop_duplicates("city_key"),
        on="city_key",
        how="left",
    )

    epw_dir = project_root / "output" / "weather" / "epw"
    ddy_dir = project_root / "output" / "weather" / "ddy"

    merged["epw_path"] = merged["epw_file"].astype(str).map(
        lambda value: (epw_dir / value).resolve()
    )
    merged["ddy_path"] = merged["ddy_file"].astype(str).map(
        lambda value: (ddy_dir / value).resolve()
    )
    return merged[
        ["sample_id", "city_key", "hbjson_path", "epw_path", "ddy_path"]
    ].copy()


# Simulation settings and dataset outputs

def build_simulation_parameter(ddy_path: Path) -> SimulationParameter:
    """Build annual and sizing-period settings from the assigned DDY file."""
    sim_output = SimulationOutput(
        reporting_frequency="Hourly",
        include_sqlite=True,
        include_html=False,
    )

    sim_par = SimulationParameter(
        output=sim_output,
        simulation_control=SimulationControl(
            do_zone_sizing=True,
            do_system_sizing=True,
            do_plant_sizing=True,
            run_for_sizing_periods=True,
            run_for_run_periods=True,
        ),
        sizing_parameter=SizingParameter(
            heating_factor=1.25,
            cooling_factor=1.15,
        ),
    )
    sim_par.sizing_parameter.add_from_ddy_996_004(str(ddy_path))
    return sim_par


def load_site_location_from_epw(epw_path: Path) -> SiteLocation:
    """Read the EnergyPlus site location from the first EPW record."""
    with epw_path.open("r", encoding="utf-8", errors="ignore") as handle:
        location_line = handle.readline().strip()

    parts = [part.strip() for part in location_line.split(",")]
    if len(parts) < 10 or parts[0].upper() != "LOCATION":
        raise ValueError(f"Could not parse LOCATION header from EPW: {epw_path}")

    return SiteLocation(
        name=parts[1] or "Site 1",
        latitude=float(parts[6]),
        longitude=float(parts[7]),
        time_zone=float(parts[8]),
        elevation=float(parts[9]),
    )


def inject_site_location_into_idf(idf_path: Path, site_location: SiteLocation) -> None:
    """Replace the translated IDF location with the assigned EPW location."""
    text = idf_path.read_text(encoding="utf-8")
    replacement = (
        "Site:Location,\n"
        f"  {site_location.name},\n"
        f"  {site_location.latitude},\n"
        f"  {site_location.longitude},\n"
        f"  {site_location.time_zone},\n"
        f"  {site_location.elevation};"
    )
    updated, n_subs = re.subn(
        r"Site:Location,\s*.*?;",
        replacement,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if n_subs != 1:
        raise ValueError(f"Could not find a Site:Location object to patch in {idf_path}")
    idf_path.write_text(updated, encoding="utf-8")


def split_idf_objects(text: str) -> list[str]:
    """Split IDF text at object terminators while retaining formatting and comments."""
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines(keepends=True):
        current.append(line)
        before_comment = line.split("!", 1)[0]
        if ";" in before_comment:
            blocks.append("".join(current))
            current = []
    if current:
        blocks.append("".join(current))
    return blocks


def _idf_object_type(block: str) -> str | None:
    """Read the object type from one IDF block."""
    for raw_line in block.splitlines():
        line = raw_line.split("!", 1)[0].strip()
        if not line or "," not in line:
            continue
        return line.split(",", 1)[0].strip()
    return None


def _render_dataset_output_block() -> str:
    """Render the SQL, summary, tolerance, and variable request objects."""
    parts = [
        "OutputControl:Table:Style,\n"
        "  Comma,                                !- Column Separator\n"
        "  None;                                 !- Unit Conversion\n\n",
        "Output:Table:SummaryReports,\n"
        "  AllSummary;                           !- Report 1 Name\n\n",
        "Output:VariableDictionary,\n"
        "  IDF,                                  !- Key Field\n"
        "  Unsorted;                             !- Sort Option\n\n",
        "OutputControl:ReportingTolerances,\n"
        "  1.11,                                 !- Heating Setpoint Not Met Tolerance {deltaC}\n"
        "  1.11;                                 "
        "!- Cooling Setpoint Not Met Tolerance {deltaC}\n\n",
        "Output:SQLite,\n"
        "  SimpleAndTabular;                     !- Option Type\n\n",
    ]

    for variable_name in DATASET_COMMON_OUTPUT_VARIABLES:
        for reporting_frequency in ("Annual", "Hourly"):
            parts.append(
                "Output:Variable,\n"
                "  *,                                    !- Key Value\n"
                f"  {variable_name},\n"
                f"  {reporting_frequency};\n\n"
            )

    for variable_name in DATASET_HOURLY_ONLY_OUTPUT_VARIABLES:
        parts.append(
            "Output:Variable,\n"
            "  *,                                    !- Key Value\n"
            f"  {variable_name},\n"
            "  Hourly;\n\n"
        )

    return "".join(parts)


def apply_dataset_output_requests_to_idf(idf_path: Path) -> None:
    """Replace reporting objects with the idempotent BEM4AI output contract."""
    text = idf_path.read_text(encoding="utf-8")
    blocks = split_idf_objects(text)

    filtered_blocks = [
        block.lstrip("\r\n")
        for block in blocks
        if _idf_object_type(block) not in REPLACED_OUTPUT_OBJECT_TYPES
    ]

    insert_at = 0
    for idx, block in enumerate(filtered_blocks):
        if _idf_object_type(block) == "Version":
            insert_at = idx + 1
            break

    filtered_blocks.insert(insert_at, _render_dataset_output_block())
    idf_path.write_text("".join(filtered_blocks), encoding="utf-8")


# IDF and EPJSON structural validation

def _idf_object_counts(text: str) -> Counter[str]:
    """Count declared IDF objects without matching names used as field values."""
    stripped = re.sub(r"!.*$", "", text, flags=re.MULTILINE)
    counts: Counter[str] = Counter()
    for chunk in stripped.split(";"):
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if not lines:
            continue
        first_line = lines[0]
        object_type, sep, _ = first_line.partition(",")
        if not sep:
            continue
        object_type = object_type.strip()
        if object_type:
            counts[object_type] += 1
    return counts


def inspect_idf_structure(idf_path: Path) -> EnergyPlusStructure:
    """Inspect the HVAC attachment and optional space metadata in an IDF."""
    text = idf_path.read_text(encoding="utf-8", errors="ignore")
    counts = _idf_object_counts(text)
    version_match = re.search(
        r"(?ims)^\s*Version\s*,\s*([^;!,]+)",
        text,
    )
    return EnergyPlusStructure(
        version=version_match.group(1).strip() if version_match else None,
        space_count=counts["Space"],
        space_list_count=counts["SpaceList"],
        design_spec_oa_spacelist_count=counts["DesignSpecification:OutdoorAir:SpaceList"],
        hvac_template_ideal_loads_count=counts["HVACTemplate:Zone:IdealLoadsAirSystem"],
        zone_hvac_ideal_loads_count=counts["ZoneHVAC:IdealLoadsAirSystem"],
        zone_hvac_equipment_connections_count=counts["ZoneHVAC:EquipmentConnections"],
        space_hvac_equipment_connections_count=counts["SpaceHVAC:EquipmentConnections"],
        zone_count=counts["Zone"],
    )


def inspect_epjson_structure(epjson_path: Path) -> EnergyPlusStructure:
    """Inspect the same structural contract in an EPJSON file."""
    with epjson_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    version = None
    version_obj = data.get("Version")
    if isinstance(version_obj, dict):
        for fields in version_obj.values():
            if isinstance(fields, dict):
                version = fields.get("version_identifier") or fields.get("Version Identifier")
                if version:
                    version = str(version)
                    break

    def count(object_type: str) -> int:
        value = data.get(object_type)
        return len(value) if isinstance(value, dict) else 0

    return EnergyPlusStructure(
        version=version,
        space_count=count("Space"),
        space_list_count=count("SpaceList"),
        design_spec_oa_spacelist_count=count("DesignSpecification:OutdoorAir:SpaceList"),
        hvac_template_ideal_loads_count=count("HVACTemplate:Zone:IdealLoadsAirSystem"),
        zone_hvac_ideal_loads_count=count("ZoneHVAC:IdealLoadsAirSystem"),
        zone_hvac_equipment_connections_count=count("ZoneHVAC:EquipmentConnections"),
        space_hvac_equipment_connections_count=count("SpaceHVAC:EquipmentConnections"),
        zone_count=count("Zone"),
    )


def add_structure_fields(
    result: dict[str, object],
    structure: EnergyPlusStructure,
    *,
    prefix: str,
) -> None:
    """Add one format's structural validation fields to a summary row."""
    result[f"{prefix}_version"] = structure.version
    result[f"{prefix}_space_count"] = structure.space_count
    result[f"{prefix}_space_list_count"] = structure.space_list_count
    result[f"{prefix}_design_spec_oa_spacelist_count"] = (
        structure.design_spec_oa_spacelist_count
    )
    result[f"{prefix}_hvac_template_ideal_loads_count"] = (
        structure.hvac_template_ideal_loads_count
    )
    result[f"{prefix}_zone_hvac_ideal_loads_count"] = (
        structure.zone_hvac_ideal_loads_count
    )
    result[f"{prefix}_zone_hvac_equipment_connections_count"] = (
        structure.zone_hvac_equipment_connections_count
    )
    result[f"{prefix}_space_hvac_equipment_connections_count"] = (
        structure.space_hvac_equipment_connections_count
    )
    result[f"{prefix}_zone_count"] = structure.zone_count
    result[f"{prefix}_has_zone_based_hvac"] = structure.has_zone_based_hvac
    result[f"{prefix}_has_expanded_space_hvac"] = structure.has_expanded_space_hvac


def cleanup_energyplus_artifacts(directory: Path) -> None:
    """Remove side files produced by EnergyPlus ``--convert-only``."""
    for pattern in ("eplusout.*", "*.audit", "*.bnd", "*.csv", "*.mdd", "*.mtd", "*.rdd"):
        for path in directory.glob(pattern):
            if path.is_file():
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


@lru_cache(maxsize=None)
def read_energyplus_version(energyplus_exe: Path) -> str:
    """Read the version reported by an EnergyPlus executable."""
    completed = subprocess.run(
        [str(energyplus_exe), "--version"],
        text=True,
        capture_output=True,
        check=False,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    match = re.search(r"\bVersion\s+(\d+(?:\.\d+)+)", output, flags=re.IGNORECASE)
    if completed.returncode != 0 or match is None:
        raise RuntimeError(
            f"Could not determine the EnergyPlus version from {energyplus_exe}. "
            f"Reported output: {output.strip() or '(none)'}"
        )
    return match.group(1)


def _major_minor(version: str) -> tuple[int, int] | None:
    """Return the major and minor numbers used for IDF compatibility."""
    match = re.match(r"\s*(\d+)\.(\d+)", version)
    return (int(match.group(1)), int(match.group(2))) if match else None


def model_format_version(input_path: Path) -> str | None:
    """Read the declared EnergyPlus version from an IDF or EPJSON file."""
    if input_path.suffix.lower() == ".idf":
        return inspect_idf_structure(input_path).version
    if input_path.suffix.lower() == ".epjson":
        return inspect_epjson_structure(input_path).version
    return None


def energyplus_version_mismatch(input_path: Path, energyplus_exe: Path) -> str | None:
    """Explain an input/executable version mismatch, or return ``None``."""
    model_version = model_format_version(input_path)
    executable_version = read_energyplus_version(energyplus_exe)
    if model_version is None:
        return None
    if _major_minor(model_version) == _major_minor(executable_version):
        return None
    return (
        f"EnergyPlus version mismatch: {input_path.name} declares {model_version}, "
        f"but {energyplus_exe} reports {executable_version}. Install the pinned "
        "requirements and use the matching EnergyPlus executable."
    )


def run_convert_only(
    input_path: Path,
    energyplus_exe: Path,
    *,
    cleanup_generated_output: bool,
) -> tuple[bool, Path | None, str]:
    """Validate one format and return the opposite format generated by EnergyPlus."""
    generated_suffix = ".epjson" if input_path.suffix.lower() == ".idf" else ".idf"
    generated_path = input_path.with_suffix(generated_suffix)

    mismatch = energyplus_version_mismatch(input_path, energyplus_exe)
    if mismatch:
        return False, None, mismatch

    completed = subprocess.run(
        [str(energyplus_exe), "--convert-only", input_path.name],
        cwd=input_path.parent,
        text=True,
        capture_output=True,
        check=False,
    )

    cleanup_energyplus_artifacts(input_path.parent)

    message = completed.stderr.strip() or completed.stdout.strip()
    success = completed.returncode == 0
    existing_generated_path = generated_path if generated_path.exists() else None

    if cleanup_generated_output and existing_generated_path is not None:
        existing_generated_path.unlink()
        existing_generated_path = None

    return success, existing_generated_path, message


# Per-model conversion

def convert_one_model(
    row: pd.Series,
    config: ExportConfig,
    energyplus_exe: Path,
    *,
    overwrite: bool,
    keep_temp: bool,
    validate_existing: bool,
    require_zone_hvac: bool = True,
) -> dict[str, object]:
    """Reuse, repair, or generate one validated IDF/EPJSON pair."""
    sample_id = str(row["sample_id"])
    city_key = str(row["city_key"])
    hbjson_path = Path(row["hbjson_path"])
    epw_path = Path(row["epw_path"])
    ddy_path = Path(row["ddy_path"])

    idf_path = config.idf_dir / f"{sample_id}.idf"
    epjson_path = config.epjson_dir / f"{sample_id}.epjson"

    result = _new_conversion_result(
        sample_id=sample_id,
        city_key=city_key,
        hbjson_path=hbjson_path,
        epw_path=epw_path,
        ddy_path=ddy_path,
        idf_path=idf_path,
        epjson_path=epjson_path,
    )

    if not hbjson_path.is_file():
        result["status"] = "missing_hbjson"
        result["message"] = f"HBJSON not found: {hbjson_path}"
        return result
    if not epw_path.is_file():
        result["status"] = "missing_epw"
        result["message"] = f"EPW not found: {epw_path}"
        return result
    if not ddy_path.is_file():
        result["status"] = "missing_ddy"
        result["message"] = f"DDY not found: {ddy_path}"
        return result

    idf_path.parent.mkdir(parents=True, exist_ok=True)
    epjson_path.parent.mkdir(parents=True, exist_ok=True)
    config.temp_dir.mkdir(parents=True, exist_ok=True)

    # Reuse a complete existing pair after structural checks.
    if idf_path.exists() and epjson_path.exists() and not overwrite:
        idf_structure = inspect_idf_structure(idf_path)
        epjson_structure = inspect_epjson_structure(epjson_path)
        add_structure_fields(result, idf_structure, prefix="idf")
        add_structure_fields(result, epjson_structure, prefix="epjson")
        existing_has_zone_hvac = (
            idf_structure.has_zone_based_hvac
            and epjson_structure.has_zone_based_hvac
        )
        if require_zone_hvac and not existing_has_zone_hvac:
            result["message"] = (
                idf_structure.failure_message("Existing IDF")
                if not idf_structure.has_zone_based_hvac
                else epjson_structure.failure_message("Existing EPJSON")
            )
        elif not validate_existing:
            result["status"] = "skipped_existing"
            result["message"] = "Skipped existing IDF/EPJSON outputs."
            return result
        else:
            idf_ok, _, idf_msg = run_convert_only(
                idf_path, energyplus_exe, cleanup_generated_output=False
            )
            generated_epjson = idf_path.with_suffix(".epjson")
            if generated_epjson.exists():
                if generated_epjson.resolve() != epjson_path.resolve():
                    if epjson_path.exists():
                        epjson_path.unlink()
                    shutil.move(str(generated_epjson), str(epjson_path))
                else:
                    generated_epjson = None

            epjson_ok, generated_idf, epjson_msg = run_convert_only(
                epjson_path, energyplus_exe, cleanup_generated_output=True
            )
            if generated_idf and generated_idf.exists():
                generated_idf.unlink()

            result["idf_valid"] = idf_ok
            result["epjson_valid"] = epjson_ok
            result["status"] = "validated_existing" if idf_ok and epjson_ok else "invalid_existing"
            result["message"] = "; ".join(
                part for part in [idf_msg, epjson_msg] if part
            )
            return result

    # Recover a missing EPJSON directly from its existing IDF.
    if idf_path.exists() and not epjson_path.exists() and not overwrite:
        apply_dataset_output_requests_to_idf(idf_path)
        idf_structure = inspect_idf_structure(idf_path)
        add_structure_fields(result, idf_structure, prefix="idf")
        if require_zone_hvac and not idf_structure.has_zone_based_hvac:
            result["message"] = idf_structure.failure_message("Existing IDF")
        else:
            idf_ok, generated_epjson_path, idf_msg = run_convert_only(
                idf_path,
                energyplus_exe,
                cleanup_generated_output=False,
            )
            result["idf_valid"] = idf_ok

            if not idf_ok:
                result["status"] = "idf_validation_failed"
                result["message"] = idf_msg
                return result

            if generated_epjson_path is None or not generated_epjson_path.exists():
                result["status"] = "epjson_not_generated"
                result["message"] = (
                    "EnergyPlus convert-only succeeded but did not create an EPJSON file."
                )
                return result

            shutil.move(str(generated_epjson_path), str(epjson_path))
            epjson_structure = inspect_epjson_structure(epjson_path)
            add_structure_fields(result, epjson_structure, prefix="epjson")
            if require_zone_hvac and not epjson_structure.has_zone_based_hvac:
                result["status"] = "epjson_structure_failed"
                result["message"] = epjson_structure.failure_message("Generated EPJSON")
                return result
            result["epjson_valid"] = True
            result["status"] = "restored_missing_epjson"
            result["message"] = idf_msg
            return result

    # Recover a missing IDF, then regenerate EPJSON from the normalized IDF.
    if epjson_path.exists() and not idf_path.exists() and not overwrite:
        epjson_structure = inspect_epjson_structure(epjson_path)
        add_structure_fields(result, epjson_structure, prefix="epjson")
        if require_zone_hvac and not epjson_structure.has_zone_based_hvac:
            result["message"] = epjson_structure.failure_message("Existing EPJSON")
        else:
            epjson_ok, generated_idf_path, epjson_msg = run_convert_only(
                epjson_path,
                energyplus_exe,
                cleanup_generated_output=False,
            )
            result["epjson_valid"] = epjson_ok

            if not epjson_ok:
                result["status"] = "epjson_validation_failed"
                result["message"] = epjson_msg
                return result

            if generated_idf_path is None or not generated_idf_path.exists():
                result["status"] = "idf_not_generated"
                result["message"] = (
                    "EnergyPlus convert-only succeeded but did not create an IDF file."
                )
                return result

            shutil.move(str(generated_idf_path), str(idf_path))
            apply_dataset_output_requests_to_idf(idf_path)
            idf_structure = inspect_idf_structure(idf_path)
            add_structure_fields(result, idf_structure, prefix="idf")
            if require_zone_hvac and not idf_structure.has_zone_based_hvac:
                result["status"] = "idf_structure_failed"
                result["message"] = idf_structure.failure_message("Generated IDF")
                return result
            _, regenerated_epjson_path, regenerated_epjson_msg = run_convert_only(
                idf_path,
                energyplus_exe,
                cleanup_generated_output=False,
            )
            if regenerated_epjson_path is not None and regenerated_epjson_path.exists():
                if epjson_path.exists():
                    epjson_path.unlink()
                shutil.move(str(regenerated_epjson_path), str(epjson_path))
                epjson_structure = inspect_epjson_structure(epjson_path)
                add_structure_fields(result, epjson_structure, prefix="epjson")
            result["idf_valid"] = True
            result["status"] = "restored_missing_idf"
            result["message"] = "; ".join(
                part for part in [epjson_msg, regenerated_epjson_msg] if part
            )
            return result

    # No reusable pair remains, so translate the HBJSON through OpenStudio.
    if idf_path.exists():
        idf_path.unlink()
    if epjson_path.exists():
        epjson_path.unlink()

    model_temp_dir = config.temp_dir / sample_id
    if model_temp_dir.exists():
        shutil.rmtree(model_temp_dir)
    model_temp_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    try:
        model = Model.from_hbjson(str(hbjson_path))
        sim_par = build_simulation_parameter(ddy_path)
        site_location = load_site_location_from_epw(epw_path)

        _, osw_path, generated_idf = to_openstudio_sim_folder(
            model,
            str(model_temp_dir),
            epw_file=str(epw_path),
            sim_par=sim_par,
            enforce_rooms=True,
        )

        if generated_idf is None:
            if osw_path is None:
                raise RuntimeError(
                    f"OpenStudio translation did not produce an IDF or OSW for {sample_id}."
                )
            _, generated_idf = run_osw(
                str(osw_path),
                silent=True,
                measures_only=True,
            )

        generated_idf_path = (
            Path(generated_idf)
            if generated_idf
            else model_temp_dir / "run" / "in.idf"
        )
        if not generated_idf_path.is_file():
            raise FileNotFoundError(
                f"Generated IDF not found for {sample_id}: {generated_idf_path}"
            )

        if idf_path.exists():
            idf_path.unlink()
        shutil.copy2(generated_idf_path, idf_path)
        inject_site_location_into_idf(idf_path, site_location)
        apply_dataset_output_requests_to_idf(idf_path)
        idf_structure = inspect_idf_structure(idf_path)
        add_structure_fields(result, idf_structure, prefix="idf")
        if require_zone_hvac and not idf_structure.has_zone_based_hvac:
            result["status"] = "idf_structure_failed"
            result["message"] = idf_structure.failure_message("Generated IDF")
            return result

        idf_ok, generated_epjson_path, idf_msg = run_convert_only(
            idf_path,
            energyplus_exe,
            cleanup_generated_output=False,
        )

        if not idf_ok:
            result["status"] = "idf_validation_failed"
            result["message"] = idf_msg
            return result

        if generated_epjson_path is None or not generated_epjson_path.exists():
            result["status"] = "epjson_not_generated"
            result["message"] = (
                "EnergyPlus convert-only succeeded but did not create an EPJSON file."
            )
            result["idf_valid"] = True
            return result

        if epjson_path.exists():
            epjson_path.unlink()
        shutil.move(str(generated_epjson_path), str(epjson_path))
        epjson_structure = inspect_epjson_structure(epjson_path)
        add_structure_fields(result, epjson_structure, prefix="epjson")
        if require_zone_hvac and not epjson_structure.has_zone_based_hvac:
            result["status"] = "epjson_structure_failed"
            result["message"] = epjson_structure.failure_message("Generated EPJSON")
            return result

        epjson_ok, generated_validation_idf, epjson_msg = run_convert_only(
            epjson_path,
            energyplus_exe,
            cleanup_generated_output=True,
        )
        if generated_validation_idf and generated_validation_idf.exists():
            generated_validation_idf.unlink()

        result["idf_valid"] = True
        result["epjson_valid"] = epjson_ok
        result["status"] = "converted" if epjson_ok else "epjson_validation_failed"
        result["message"] = "; ".join(part for part in [idf_msg, epjson_msg] if part)
        return result
    except Exception as exc:  # pragma: no cover - pipeline/runtime failure path
        result["status"] = "conversion_failed"
        result["message"] = str(exc)
        return result
    finally:
        result["elapsed_s"] = round(time.perf_counter() - start, 3)
        if not keep_temp and model_temp_dir.exists():
            shutil.rmtree(model_temp_dir, ignore_errors=True)


def _convert_one_model_worker(
    row_dict: dict[str, object],
    config: ExportConfig,
    energyplus_exe: Path,
    *,
    overwrite: bool,
    keep_temp: bool,
    validate_existing: bool,
    require_zone_hvac: bool,
) -> dict[str, object]:
    """Rebuild a Series inside a worker process before converting one model."""
    row = pd.Series(row_dict)
    return convert_one_model(
        row,
        config,
        energyplus_exe,
        overwrite=overwrite,
        keep_temp=keep_temp,
        validate_existing=validate_existing,
        require_zone_hvac=require_zone_hvac,
    )


# Batch orchestration and command-line interface

def convert_export(
    project_root: Path,
    config: ExportConfig,
    weather_assets_df: pd.DataFrame,
    energyplus_exe: Path,
    *,
    limit: int | None,
    overwrite: bool,
    keep_temp: bool,
    validate_existing: bool,
    require_zone_hvac: bool = True,
    workers: int = 1,
) -> pd.DataFrame:
    """Convert the selected model table and write the per-model summary CSV."""
    conversion_table = build_conversion_table(project_root, config, weather_assets_df)
    if conversion_table.empty:
        empty_df = pd.DataFrame(columns=CONVERSION_RESULT_COLUMNS)
        empty_df.to_csv(config.summary_csv, index=False)
        return empty_df

    if limit is not None:
        conversion_table = conversion_table.head(limit).copy()

    config.idf_dir.mkdir(parents=True, exist_ok=True)
    config.epjson_dir.mkdir(parents=True, exist_ok=True)
    config.temp_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Processing %s model(s)", len(conversion_table))

    rows = []
    if workers <= 1:
        for _, row in tqdm(
            conversion_table.iterrows(),
            total=len(conversion_table),
            desc="HBJSON -> IDF/EPJSON",
        ):
            rows.append(
                convert_one_model(
                    row,
                    config,
                    energyplus_exe,
                    overwrite=overwrite,
                    keep_temp=keep_temp,
                    validate_existing=validate_existing,
                    require_zone_hvac=require_zone_hvac,
                )
            )
    else:
        row_dicts = conversion_table.to_dict("records")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _convert_one_model_worker,
                    row_dict,
                    config,
                    energyplus_exe,
                    overwrite=overwrite,
                    keep_temp=keep_temp,
                    validate_existing=validate_existing,
                    require_zone_hvac=require_zone_hvac,
                )
                for row_dict in row_dicts
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="HBJSON -> IDF/EPJSON",
            ):
                rows.append(future.result())

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(config.summary_csv, index=False)

    if not keep_temp and config.temp_dir.exists() and not any(config.temp_dir.iterdir()):
        config.temp_dir.rmdir()

    return summary_df


def convert_exports(
    *,
    project_root: Path | None = None,
    limit: int | None = None,
    overwrite: bool = False,
    keep_temp: bool = False,
    validate_existing: bool = False,
    require_zone_hvac: bool = True,
    energyplus_exe: str | Path | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Load the standard project inputs and run the Stage 7 conversion batch."""
    project_root = infer_project_root() if project_root is None else Path(project_root).resolve()
    config = build_export_config(project_root)
    resolve_openstudio_executable()
    energyplus_path = resolve_energyplus_executable(energyplus_exe)
    weather_assets_df = load_weather_assets(project_root)
    return convert_export(
        project_root,
        config,
        weather_assets_df,
        energyplus_path,
        limit=limit,
        overwrite=overwrite,
        keep_temp=keep_temp,
        validate_existing=validate_existing,
        require_zone_hvac=require_zone_hvac,
        workers=workers,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for a standalone conversion run."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert final HBJSON exports to IDF and EPJSON using the current "
            "project weather files."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=infer_project_root(),
        help="Project root. Defaults to the repository containing this script.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of models to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate IDF/EPJSON files even if both already exist.",
    )
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help=(
            "When overwrite is disabled, validate complete existing IDF/EPJSON "
            "pairs instead of skipping them."
        ),
    )
    parser.add_argument(
        "--allow-template-idf",
        action="store_true",
        help=(
            "Allow template-style EnergyPlus outputs. By default the converter requires "
            "zone-level ZoneHVAC ideal-load objects and ZoneHVAC equipment "
            "connections, while Space/SpaceList remain optional metadata."
        ),
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help=(
            "Keep the intermediate OpenStudio workflow folders under "
            "output/idf_epjson_conversion_temp."
        ),
    )
    parser.add_argument(
        "--energyplus-exe",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to the EnergyPlus executable used for "
            "--convert-only validation."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes to use.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run conversion from the command line and print status counts."""
    args = parse_args(argv)
    summary_df = convert_exports(
        project_root=args.project_root,
        limit=args.limit,
        overwrite=args.overwrite,
        keep_temp=args.keep_temp,
        validate_existing=args.validate_existing,
        require_zone_hvac=not args.allow_template_idf,
        energyplus_exe=args.energyplus_exe,
        workers=args.workers,
    )

    if summary_df.empty:
        print("No HBJSON files were available for conversion.")
        return 0

    status_counts = (
        summary_df.groupby("status", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("status")
    )
    print(status_counts.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
