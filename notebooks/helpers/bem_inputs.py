"""Load and prepare the tabular inputs consumed by BEM4AI Stage 5."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .pickle_compat import read_pickle_compat
from .utils import is_missing

__all__ = [
    "estimate_floors",
    "filter_bem_inputs",
    "load_archetype_bundle",
    "load_transformed_artifacts",
    "prepare_bem_inputs",
]

TRANSFORMED_KEYS = ("buildings_m", "neighbors_m", "shades_gdf")
WINDOW_KEYS = ("final_windows_gdf", "final_windows", "windows_final", "window_walls_gdf")
RUNTIME_ID_COLUMNS = ("building_id", "source_sample_id")
REQUIRED_ELEMENT_COLUMNS = (
    "wall_hb_construction",
    "wall_hb_material",
    "roof_hb_construction",
    "roof_hb_material",
    "floor_hb_construction",
    "floor_hb_material",
    "window_hb_construction",
    "window_hb_material",
)


def load_transformed_artifacts(path: Path) -> dict[str, object]:
    """Load the Stage 4 bundle and remove identifiers not used after Stage 4."""
    payload = read_pickle_compat(path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected {path.name} to contain a mapping payload.")

    artifacts = {key: payload.get(key) for key in TRANSFORMED_KEYS}
    artifacts["final_windows_gdf"] = next(
        (payload[key] for key in WINDOW_KEYS if key in payload),
        None,
    )
    if artifacts["buildings_m"] is None:
        raise KeyError(f"{path.name} is missing required artifact 'buildings_m'.")

    for name, value in artifacts.items():
        if isinstance(value, (pd.DataFrame, gpd.GeoDataFrame)):
            artifacts[name] = value.copy().drop(columns=RUNTIME_ID_COLUMNS, errors="ignore")

    if not isinstance(artifacts["buildings_m"], gpd.GeoDataFrame):
        raise TypeError("The 'buildings_m' artifact must be a GeoDataFrame.")
    return artifacts


def load_archetype_bundle(path: Path) -> dict[str, pd.DataFrame]:
    """Load the five building-stock tables produced by Stage 2."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing archetype bundle: {path}. Run BEM4AI_2_Archetypes.ipynb first."
        )
    bundle = read_pickle_compat(path)
    if not isinstance(bundle, Mapping):
        raise TypeError(f"Expected {path.name} to contain a mapping of DataFrames.")

    required = {"cities", "building_types", "age_classes", "elements", "materials"}
    missing = sorted(required.difference(bundle))
    if missing:
        raise KeyError(f"{path.name} is missing tables: {missing}")
    invalid = [name for name in required if not isinstance(bundle[name], pd.DataFrame)]
    if invalid:
        raise TypeError(f"Tables in {path.name} are not DataFrames: {invalid}")
    return {name: bundle[name].copy() for name in required}


def _attach_city_ids(buildings: gpd.GeoDataFrame, stock_cities: pd.DataFrame) -> gpd.GeoDataFrame:
    """Map the Stage 1 city key to the building-stock city and country IDs."""
    if "city_key" not in buildings.columns:
        raise KeyError("buildings_m must contain 'city_key'.")
    if stock_cities["city_key"].duplicated().any():
        raise ValueError("The archetype cities table contains duplicate city_key values.")

    result = buildings.copy()
    lookup = stock_cities.set_index("city_key")
    result["city_id"] = result["city_key"].map(lookup["id"]).astype("Int64")
    result["country_id"] = result["city_key"].map(lookup["country_id"]).astype("Int64")
    return result


def _attach_age_mapping(
    buildings: gpd.GeoDataFrame,
    age_classes: pd.DataFrame,
    age_config: Mapping,
) -> gpd.GeoDataFrame:
    """Translate the GHSL construction-age code to the building-stock age class."""
    result = buildings.copy()
    age_label_to_id = dict(zip(age_classes["label"], age_classes["id"]))
    age_maps = {
        field: {int(code): values.get(field) for code, values in age_config.items()}
        for field in ("epoch", "typical_year", "age_class_label", "hb_vintage_file")
    }
    result["builtage_code"] = pd.to_numeric(result["builtage_val"], errors="coerce").astype("Int64")
    result["epoch_txt"] = result["builtage_code"].map(age_maps["epoch"])
    result["typical_year"] = result["builtage_code"].map(age_maps["typical_year"])
    result["age_class_label"] = result["builtage_code"].map(age_maps["age_class_label"])
    result["hb_vintage_file"] = result["builtage_code"].map(age_maps["hb_vintage_file"])
    result["age_class_id"] = result["age_class_label"].map(age_label_to_id).astype("Int64")
    return result


def _attach_climate_mapping(
    buildings: gpd.GeoDataFrame,
    cities: pd.DataFrame,
    config: Mapping,
) -> gpd.GeoDataFrame:
    """Translate the city Koppen-Geiger code to a Honeybee construction-set ID."""
    required = {"city_key", "CL_KOP_CUR_2025"}
    if not required.issubset(cities.columns):
        raise KeyError("cities_df must contain 'city_key' and 'CL_KOP_CUR_2025'.")

    climate_rows = cities[["city_key", "CL_KOP_CUR_2025"]].dropna(subset=["city_key"])
    conflicting = climate_rows.groupby("city_key")["CL_KOP_CUR_2025"].nunique(dropna=False)
    if conflicting.gt(1).any():
        raise ValueError("cities_df contains conflicting climate codes for the same city_key.")

    result = buildings.copy()
    climate_lookup = climate_rows.drop_duplicates("city_key").set_index("city_key")["CL_KOP_CUR_2025"]
    result["koppen_code"] = result["city_key"].map(climate_lookup).astype("Int64")
    numeric_to_kg = {int(code): value for code, value in config["kg_numeric_to_code"].items()}
    kg_to_ashrae = config["kg_to_ashrae"]
    result["kg_code"] = result["koppen_code"].map(numeric_to_kg)
    result["ashrae_zone"] = result["kg_code"].map(
        {code: values.get("zone") for code, values in kg_to_ashrae.items()}
    )
    result["ashrae_zone_desc"] = result["kg_code"].map(
        {code: values.get("desc") for code, values in kg_to_ashrae.items()}
    )
    result["ashrae_zone_num"] = result["ashrae_zone"].astype("string").str.extract(
        r"(\d+)", expand=False
    )
    result["hb_vintage"] = result["hb_vintage_file"].astype("string").str.replace(
        "_data.json", "", regex=False
    )
    result["hb_construction_set_id"] = [
        f"{vintage}::ClimateZone{zone}::Mass"
        if pd.notna(vintage) and pd.notna(zone)
        else None
        for vintage, zone in zip(result["hb_vintage"], result["ashrae_zone_num"])
    ]
    return result


def _attach_element_materials(
    buildings: gpd.GeoDataFrame,
    elements: pd.DataFrame,
    materials: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Join wall, roof, floor, and window definitions by archetype IDs."""
    wanted_codes = {"WALL", "ROOF", "FLOOR", "WINDOW"}
    selected = elements[elements["code"].isin(wanted_codes)]
    element_ids = dict(zip(selected["code"], selected["id"]))
    if set(element_ids) != wanted_codes:
        raise KeyError("The archetype elements table must define WALL, ROOF, FLOOR, and WINDOW.")

    required = {
        "country_id", "building_type_id", "age_class_id", "element_id", "u_value",
        "honeybee_construction", "honeybee_material",
    }
    missing = sorted(required.difference(materials.columns))
    if missing:
        raise KeyError(f"The archetype materials table is missing columns: {missing}")

    result = buildings.copy()
    merge_keys = ["country_id", "building_type_id", "age_class_id"]
    for element_code, element_id in element_ids.items():
        suffix = element_code.lower()
        element_materials = materials.loc[
            materials["element_id"].eq(element_id),
            merge_keys + ["honeybee_construction", "honeybee_material", "u_value"],
        ].copy()
        duplicates = element_materials.duplicated(merge_keys, keep=False)
        if duplicates.any():
            examples = element_materials.loc[duplicates, merge_keys].head(3)
            raise ValueError(
                f"Duplicate {element_code} rows for the same archetype keys:\n"
                + examples.to_string(index=False)
            )
        element_materials = element_materials.rename(
            columns={
                "honeybee_construction": f"{suffix}_hb_construction",
                "honeybee_material": f"{suffix}_hb_material",
                "u_value": f"{suffix}_u_value",
            }
        )
        result = result.merge(
            element_materials,
            on=merge_keys,
            how="left",
            validate="many_to_one",
        )
    return result


def prepare_bem_inputs(
    buildings: gpd.GeoDataFrame,
    cities: pd.DataFrame,
    archetypes: Mapping[str, pd.DataFrame],
    config: Mapping,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Attach every building-stock and climate field needed for model creation."""
    result = _attach_city_ids(buildings, archetypes["cities"])
    result = _attach_age_mapping(result, archetypes["age_classes"], config["ghsl_builtage"])
    result = _attach_climate_mapping(result, cities, config)
    result = _attach_element_materials(result, archetypes["elements"], archetypes["materials"])
    return result, archetypes["building_types"].copy()


def filter_bem_inputs(buildings: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Separate buildings that have every input required to construct a BEM."""
    result = buildings.copy()

    def exclusion_reasons(row: pd.Series) -> list[str]:
        found = []
        if is_missing(row.get("city_id")) or is_missing(row.get("country_id")):
            found.append("missing_city_or_country_mapping")
        builtage = row.get("builtage_code")
        if pd.notna(builtage) and int(builtage) == 0:
            found.append("builtage_0_unmapped")
        height = pd.to_numeric(pd.Series([row.get("builth_val")]), errors="coerce").iloc[0]
        if pd.isna(height) or float(height) <= 0:
            found.append("missing_or_nonpositive_building_height")
        if is_missing(row.get("ashrae_zone_num")):
            found.append("missing_climate_zone")
        if is_missing(row.get("hb_construction_set_id")):
            found.append("missing_hb_construction_set")
        for column in REQUIRED_ELEMENT_COLUMNS:
            if is_missing(row.get(column)):
                found.append(f"missing_{column}")
        return list(dict.fromkeys(found))

    result["bem_exclusion_reasons"] = result.apply(exclusion_reasons, axis=1)
    excluded = result[result["bem_exclusion_reasons"].map(bool)].copy()
    ready = result[~result["bem_exclusion_reasons"].map(bool)].copy()
    return ready, excluded


def estimate_floors(buildings: gpd.GeoDataFrame, config: Mapping) -> gpd.GeoDataFrame:
    """Estimate floor counts while keeping floor heights within configured bounds."""
    result = buildings.copy()
    bounds = config.get("floor_height_bounds", {}).get("default", {})
    minimum = float(bounds.get("min", 2.6))
    maximum = float(bounds.get("max", 3.4))
    midpoint = float(bounds.get("mid", (minimum + maximum) / 2.0))
    if not 0 < minimum <= midpoint <= maximum:
        raise ValueError("The default floor-height bounds in hb_mappings.json are invalid.")

    height = pd.to_numeric(result["builth_val"], errors="coerce")
    if height.isna().any() or height.le(0).any():
        raise ValueError("BEM inputs contain a missing or non-positive builth_val.")
    minimum_floors = np.ceil(height / maximum)
    maximum_floors = np.floor(height / minimum)
    midpoint_floors = np.rint(height / midpoint)

    # A very short building can have no integer floor count inside the bounds.
    infeasible = minimum_floors > maximum_floors
    closest = np.maximum(1, midpoint_floors)
    minimum_floors = minimum_floors.where(~infeasible, closest)
    maximum_floors = maximum_floors.where(~infeasible, closest)
    levels = midpoint_floors.clip(lower=minimum_floors, upper=maximum_floors)
    result["levels_est"] = levels.clip(lower=1).astype(int)
    result["level_height_m"] = (height / result["levels_est"]).round(1).clip(2.0, 5.0)
    return result
