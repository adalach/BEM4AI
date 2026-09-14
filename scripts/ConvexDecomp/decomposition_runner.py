"""
Prepare the polygons each building is decomposed from, and run the adaptive
convex decomposer over them.

Adapted from ``convexdecomp.osm.diagnostics`` in
https://github.com/adalach/ConvexDecomp. Upstream passes bare polygons around,
so a finished zone cannot say which part of the building it came from. Here
every polygon travels as a dict carrying a ``source_kind`` (``perimeter``,
``interior``, ``whole_building`` or ``whole_building_fallback``) and a finer
``seed_kind``, and the runner splits its output into ``zones_perim``,
``zones_core`` and the combined ``zones``.

The same runner serves both passes: the main decomposition over perimeter
shells and interiors, and the whole-building retry for buildings whose zones
come out as slivers. Because ``whole_building_fallback`` is neither
``perimeter`` nor ``interior``, the retry's core and perimeter lists come out
empty on their own.
"""

from __future__ import annotations

from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from helpers.utils import count_exterior_vertices

from .osm_convex_decomposer import decompose_polygon_with_stats, extract_polygons
from .osm_perimeter_subdivider import subdivide_perimeter_gdf

__all__ = [
    "build_parts_geometry",
    "collect_decomposition_inputs",
    "decompose_inputs",
    "group_seed_parts",
    "min_part_area",
    "prepare_decomposition_inputs",
    "run_building_decomposition",
    "sum_time_list",
    "whole_building_inputs",
]


# The keys every result dict carries, so a building that produced nothing still
# has the same columns as one that decomposed.
EMPTY_RESULT: dict[str, Any] = {
    "zones": [],
    "zones_core": [],
    "zones_perim": [],
    "convex_decomp_success": False,
    "convex_variants_total": 0,
    "convex_parts_counts": [],
    "polygon_fully_convex_flags": [],
    "building_fully_convex": False,
    "building_decomposition_failed": True,
    "n_successful_polygons": 0,
    "building_has_small_zone": False,
    "convex_parts_geom": None,
    "polygon_times_seconds": [],
    "polygon_fallbacks": [],
    "polygon_search_terminated_by": [],
    "polygon_search_depths_used": [],
    "polygon_search_widths_used": [],
    "polygon_search_attempt_counts": [],
    "n_failed_polygons": 0,
    "n_small_zone_polygons": 0,
    "n_retried_polygons": 0,
    "total_search_attempts": 0,
    "max_search_depth_used": 0,
    "max_search_width_used": 0,
    "time_total_seconds": 0.0,
}


def build_parts_geometry(parts: list[Polygon]) -> BaseGeometry | None:
    """One geometry from a list of convex parts, or None if there are none."""
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return MultiPolygon(parts)


def sum_time_list(values: Iterable[float | None] | None) -> float:
    """Total of a per-polygon timing list, ignoring None and unparseable entries."""
    if not isinstance(values, (list, tuple)):
        return 0.0
    clean_values = []
    for value in values:
        if value is None:
            continue
        try:
            clean_values.append(float(value))
        except (TypeError, ValueError):
            continue
    return float(np.nansum(clean_values)) if clean_values else 0.0


def min_part_area(parts: list[Polygon]) -> float:
    """Area of the smallest part, or NaN when the list holds nothing usable."""
    areas = []
    for part in parts or []:
        if part is None or getattr(part, "is_empty", True):
            continue
        try:
            areas.append(float(part.area))
        except Exception:
            continue
    return float(min(areas)) if areas else float("nan")


def group_seed_parts(parts_gdf: gpd.GeoDataFrame, *, id_col: str = "sample_id") -> dict[str, list[Polygon]]:
    """Seed polygons per sample_id, dropping empty geometries."""
    grouped: dict[str, list[Polygon]] = {}
    if parts_gdf is None or parts_gdf.empty:
        return grouped
    for sample_id, group in parts_gdf.groupby(id_col):
        grouped[str(sample_id)] = [
            geom
            for geom in group.geometry
            if geom is not None and not getattr(geom, "is_empty", True)
        ]
    return grouped


def collect_decomposition_inputs(
    row: pd.Series,
    *,
    use_precheck: bool,
    trapezoid_seed_map: dict[str, list[Polygon]] | None = None,
    corner_seed_map: dict[str, list[Polygon]] | None = None,
    remainder_seed_map: dict[str, list[Polygon]] | None = None,
) -> list[dict]:
    """The labelled polygons one building is decomposed from.

    A building with a perimeter is decomposed shell by shell: the trapezoid,
    corner and remainder seeds when the pre-check produced them, otherwise the
    raw perimeter parts, plus the interior. A building without a perimeter is
    decomposed whole.

    Whichever route is taken, the polygons returned cover the whole footprint.
    The perimeter is what decides that: the interior is only ever the leftover
    inside it, so a route that produced no perimeter cannot be used even when it
    did produce an interior, and falls through to the next one.
    """
    sample_id = str(row["sample_id"])
    interior = [
        {"geometry": geom, "source_kind": "interior", "seed_kind": "interior"}
        for geom in extract_polygons(row.get("interior_geom"))
    ]

    if bool(row.get("perimeter_defined", False)):
        if use_precheck:
            seeds: list[dict] = []
            for geom in (trapezoid_seed_map or {}).get(sample_id, []):
                seeds.append({"geometry": geom, "source_kind": "perimeter", "seed_kind": "trapezoid"})
            for geom in (corner_seed_map or {}).get(sample_id, []):
                seeds.append({"geometry": geom, "source_kind": "perimeter", "seed_kind": "corner"})
            for geom in (remainder_seed_map or {}).get(sample_id, []):
                seeds.append({"geometry": geom, "source_kind": "perimeter", "seed_kind": "remainder"})
            if seeds:
                return seeds + interior

        # The pre-check produced no seed, so use the perimeter shells as they are.
        parts = [
            {"geometry": geom, "source_kind": "perimeter", "seed_kind": "perimeter_part"}
            for geom in extract_polygons(row.get("perimeter_parts"))
        ]
        if parts:
            return parts + interior

    return [
        {"geometry": geom, "source_kind": "whole_building", "seed_kind": "whole_building"}
        for geom in extract_polygons(row.get("geometry"))
    ]


def whole_building_inputs(row: pd.Series) -> list[dict]:
    """The retry input for a building whose shell-by-shell zoning came out badly."""
    return [
        {"geometry": geom, "source_kind": "whole_building_fallback", "seed_kind": "whole_building"}
        for geom in extract_polygons(row.get("geom_local"))
    ]


def prepare_decomposition_inputs(
    buildings_gdf: gpd.GeoDataFrame,
    *,
    use_precheck: bool,
    precheck_cfg: Any,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Attach `decomp_inputs` to every building, with the seed frames alongside.

    Returns the annotated buildings plus the trapezoid, corner and remainder
    seed frames, which stay around as decomposition diagnostics.
    """
    work = gpd.GeoDataFrame(buildings_gdf.copy(), geometry="geometry", crs=buildings_gdf.crs)

    if use_precheck:
        work, trapezoid_seed_gdf, corner_seed_gdf, remainder_seed_gdf = subdivide_perimeter_gdf(
            work,
            id_col="sample_id",
            cfg=precheck_cfg,
        )
        trapezoid_seed_map = group_seed_parts(trapezoid_seed_gdf)
        corner_seed_map = group_seed_parts(corner_seed_gdf)
        remainder_seed_map = group_seed_parts(remainder_seed_gdf)
    else:
        empty_geometry = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=work.crs)
        trapezoid_seed_gdf = empty_geometry.copy()
        corner_seed_gdf = empty_geometry.copy()
        remainder_seed_gdf = empty_geometry.copy()
        trapezoid_seed_map = {}
        corner_seed_map = {}
        remainder_seed_map = {}

    work["decomp_inputs"] = work.apply(
        collect_decomposition_inputs,
        axis=1,
        use_precheck=use_precheck,
        trapezoid_seed_map=trapezoid_seed_map,
        corner_seed_map=corner_seed_map,
        remainder_seed_map=remainder_seed_map,
    )
    work["decomp_polygons"] = work["decomp_inputs"].apply(lambda items: [item["geometry"] for item in items])
    work["n_polygons"] = work["decomp_polygons"].apply(len)
    work["n_vertices"] = work["decomp_polygons"].apply(count_exterior_vertices)
    return work, trapezoid_seed_gdf, corner_seed_gdf, remainder_seed_gdf


def decompose_inputs(
    inputs: list[dict],
    cfg: Any,
    *,
    sample_id: str,
    building_geometry: BaseGeometry | None = None,
    zone_id_prefix: str = "poly",
    index: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Decompose one building's polygons and collect its diagnostics.

    Returns the building-level result and one row per input polygon. Zones are
    kept only when every polygon of the building reached a fully convex result,
    so a building is never half-decomposed.
    """
    zones_all: list[Polygon] = []
    zones_core: list[Polygon] = []
    zones_perim: list[Polygon] = []
    convex_parts_counts = []
    polygon_flags = []
    polygon_times = []
    polygon_fallbacks = []
    polygon_stop_reasons = []
    polygon_search_depths = []
    polygon_search_widths = []
    polygon_search_attempt_counts = []
    convex_variants_total = 0
    building_has_small_zone = False
    n_failed_polygons = 0
    n_small_zone_polygons = 0
    polygon_rows: list[dict[str, Any]] = []

    for polygon_idx, item in enumerate(inputs, start=1):
        polygon = item["geometry"]
        source_kind = item.get("source_kind", "unknown")
        seed_kind = item.get("seed_kind", source_kind)

        stats = decompose_polygon_with_stats(polygon, cfg)
        best_variant = [
            part
            for part in stats["best_variant"]
            if part is not None and not getattr(part, "is_empty", True)
        ]

        convex_variants_total += int(stats["n_variants"])
        convex_parts_counts.append(int(stats["n_parts"]))
        polygon_flags.append(bool(stats["fully_convex_best"]))
        polygon_times.append(stats.get("time_seconds"))
        polygon_fallbacks.append(stats.get("fallback_used"))
        polygon_stop_reasons.append(stats.get("search_terminated_by"))
        polygon_search_depths.append(stats.get("search_depth_used"))
        polygon_search_widths.append(stats.get("search_width_used"))
        polygon_search_attempt_counts.append(int(stats.get("search_attempt_count", 0) or 0))
        building_has_small_zone = building_has_small_zone or bool(stats["has_small_zone"])
        n_failed_polygons += int(not bool(stats["fully_convex_best"]))
        n_small_zone_polygons += int(bool(stats["has_small_zone"]))

        zones_all.extend(best_variant)
        if source_kind == "interior":
            zones_core.extend(best_variant)
        elif source_kind == "perimeter":
            zones_perim.extend(best_variant)

        polygon_row = {
            "sample_id": sample_id,
            "polygon_idx": polygon_idx,
            "zone_id": f"{sample_id}_{zone_id_prefix}_{polygon_idx:02d}",
            "source_kind": source_kind,
            "seed_kind": seed_kind,
            "fully_convex_best": bool(stats["fully_convex_best"]),
            "has_small_zone": bool(stats["has_small_zone"]),
            "n_variants": int(stats["n_variants"]),
            "n_parts": int(stats["n_parts"]),
            "search_terminated_by": stats.get("search_terminated_by"),
            "n_failed_terminal_states": int(stats.get("n_failed_terminal_states", 0) or 0),
            "n_depth_limit_dead_ends": int(stats.get("n_depth_limit_dead_ends", 0) or 0),
            "n_no_reflex_dead_ends": int(stats.get("n_no_reflex_dead_ends", 0) or 0),
            "n_min_area_dead_ends": int(stats.get("n_min_area_dead_ends", 0) or 0),
            "n_states_seen": int(stats.get("n_states_seen", 0) or 0),
            "search_depth_used": stats.get("search_depth_used"),
            "search_width_used": stats.get("search_width_used"),
            "search_attempt_count": int(stats.get("search_attempt_count", 0) or 0),
            "search_attempt_history": list(stats.get("search_attempt_history") or []),
            "fallback_used": stats.get("fallback_used"),
            "polygon_time_seconds": stats.get("time_seconds"),
            "best_min_part_area_m2": min_part_area(best_variant),
            "geometry": polygon,
            "building_geometry": building_geometry,
            "best_variant_geom": build_parts_geometry(best_variant),
        }
        if index is not None:
            polygon_row = {"index": index, **polygon_row}
        polygon_rows.append(polygon_row)

    building_fully_convex = bool(polygon_flags) and all(polygon_flags)
    used_depths = [int(value) for value in polygon_search_depths if value is not None]
    used_widths = [int(value) for value in polygon_search_widths if value is not None]

    result = {
        "zones": zones_all if building_fully_convex else [],
        "zones_core": zones_core if building_fully_convex else [],
        "zones_perim": zones_perim if building_fully_convex else [],
        "convex_decomp_success": building_fully_convex,
        "convex_variants_total": convex_variants_total,
        "convex_parts_counts": convex_parts_counts,
        "polygon_fully_convex_flags": polygon_flags,
        "building_fully_convex": building_fully_convex,
        "building_decomposition_failed": n_failed_polygons > 0,
        "n_successful_polygons": int(sum(polygon_flags)),
        "building_has_small_zone": building_has_small_zone,
        "convex_parts_geom": build_parts_geometry(zones_all) if building_fully_convex else None,
        "polygon_times_seconds": polygon_times,
        "polygon_fallbacks": polygon_fallbacks,
        "polygon_search_terminated_by": polygon_stop_reasons,
        "polygon_search_depths_used": polygon_search_depths,
        "polygon_search_widths_used": polygon_search_widths,
        "polygon_search_attempt_counts": polygon_search_attempt_counts,
        "n_failed_polygons": n_failed_polygons,
        "n_small_zone_polygons": n_small_zone_polygons,
        "n_retried_polygons": int(sum(1 for attempts in polygon_search_attempt_counts if attempts > 1)),
        "total_search_attempts": int(sum(polygon_search_attempt_counts)),
        "max_search_depth_used": max(used_depths, default=0),
        "max_search_width_used": max(used_widths, default=0),
        "time_total_seconds": sum_time_list(polygon_times),
    }
    return result, polygon_rows


def run_building_decomposition(
    buildings_convex_gdf: gpd.GeoDataFrame,
    cfg: Any,
    *,
    progress: bool = True,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Decompose every building in the frame and join the diagnostics back on.

    Returns the buildings with the result columns attached, and one frame of
    per-polygon diagnostics.
    """
    rows = buildings_convex_gdf.iterrows()
    if progress:
        from tqdm import tqdm

        rows = tqdm(rows, total=len(buildings_convex_gdf), desc="Convex decomposition")

    results = []
    polygon_rows: list[dict[str, Any]] = []
    for idx, row in rows:
        inputs = row.get("decomp_inputs") or []
        if not inputs:
            results.append({"index": idx, **EMPTY_RESULT})
            continue

        result, rows_for_building = decompose_inputs(
            inputs,
            cfg,
            sample_id=str(row["sample_id"]),
            building_geometry=row["geometry"],
            index=idx,
        )
        results.append({"index": idx, **result})
        polygon_rows.extend(rows_for_building)

    out = buildings_convex_gdf.join(pd.DataFrame(results).set_index("index"))

    if polygon_rows:
        polygon_results_gdf = gpd.GeoDataFrame(polygon_rows, geometry="geometry", crs=out.crs)
    else:
        polygon_results_gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=out.crs)

    return out, polygon_results_gdf
