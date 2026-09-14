"""Judge the convex zones a building was cut into, and repair the bad ones.

The convex decomposer is asked for convex pieces, and it delivers them, but
convex is not the same as usable. A zone one metre wide and twenty long is
perfectly convex and useless as a thermal zone: EnergyPlus will happily solve
it, and the answer will be dominated by a surface area that no real room has.

So the zones are scored on the shape of the worst one, and a building still
carrying a hard failure gets one retry: throw away the perimeter-and-core split
and decompose the whole footprint in one go. The retry is kept only if it
actually scores better, so a building can never be made worse by trying.

Import from a notebook as::

    from helpers.zone_quality import apply_whole_building_fallback

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.ConvexDecomp.decomposition_runner import decompose_inputs, whole_building_inputs
from .utils import aspect_ratio, compactness, min_rotated_rect_dims, polygon_parts

__all__ = [
    "ZoneQualityLimits",
    "annotate_zone_quality",
    "apply_whole_building_fallback",
    "compute_zone_quality_metrics",
    "zone_quality_score",
]


@dataclass(frozen=True)
class ZoneQualityLimits:
    """Where a zone stops being merely awkward and becomes unusable.

    The ``hard`` thresholds are a verdict: a building still carrying one after
    the retry cannot be modelled and leaves the dataset. The ``awkward`` ones
    are not a verdict at all - they only count how many zones of a building are
    unpleasant, which is what separates two zonings that are otherwise tied
    when the retry is scored against what it would replace.
    """

    hard_width_m: float = 1.0
    hard_aspect_ratio: float = 10.0
    hard_compactness: float = 0.15
    awkward_width_m: float = 1.2
    awkward_aspect_ratio: float = 8.0
    awkward_compactness: float = 0.20
    #: Above this a building is over-segmented, which is one of the two reasons
    #: its zoning is worth retrying as a whole footprint.
    oversegmented_zone_count: int = 12


def compute_zone_quality_metrics(zones, limits: ZoneQualityLimits) -> dict[str, object]:
    """Shape metrics for one building's zones, plus the two failure flags.

    A zone fails hard when it is narrow *and* either elongated or non-compact.
    Narrow alone is fine: a small square zone is narrow and perfectly usable.
    """
    metrics: dict[str, object] = {
        "zone_count": 0,
        "zone_area_min_m2": float("nan"),
        "zone_width_min_m": float("nan"),
        "zone_aspect_max": float("nan"),
        "zone_compactness_min": float("nan"),
        "zone_n_hard_fail_shapes": 0,
        "zone_n_awkward_shapes": 0,
        "zone_shape_hard_fail": False,
    }
    if not isinstance(zones, list) or not zones:
        return metrics

    polys = []
    for zone in zones:
        parts = polygon_parts(zone)
        if parts:
            polys.append(max(parts, key=lambda poly: poly.area))
    if not polys:
        return metrics

    areas, widths, aspects, compactness_values = [], [], [], []
    hard_fail_count = awkward_count = 0

    for poly in polys:
        area = float(poly.area)
        width, _ = min_rotated_rect_dims(poly)
        aspect = aspect_ratio(poly)
        compact = compactness(poly)

        areas.append(area)
        widths.append(width)
        aspects.append(aspect)
        compactness_values.append(compact)

        hard_fail = (
            np.isfinite(width)
            and width < limits.hard_width_m
            and (aspect > limits.hard_aspect_ratio or compact < limits.hard_compactness)
        )
        awkward = (
            hard_fail
            or (np.isfinite(width) and width < limits.awkward_width_m)
            or aspect > limits.awkward_aspect_ratio
            or compact < limits.awkward_compactness
        )
        hard_fail_count += int(hard_fail)
        awkward_count += int(awkward)

    metrics.update(
        {
            "zone_count": len(polys),
            "zone_area_min_m2": float(np.nanmin(areas)),
            "zone_width_min_m": float(np.nanmin(widths)),
            "zone_aspect_max": float(np.nanmax(aspects)),
            "zone_compactness_min": float(np.nanmin(compactness_values)),
            "zone_n_hard_fail_shapes": hard_fail_count,
            "zone_n_awkward_shapes": awkward_count,
        }
    )
    metrics["zone_shape_hard_fail"] = bool(hard_fail_count > 0)
    return metrics


def zone_quality_score(metrics: dict[str, object]) -> tuple[float, ...]:
    """A sortable score for one zoning, higher being better.

    Compared as a tuple, so fewer unusable zones always beats fewer awkward
    ones, fewer awkward ones beats fewer zones, and the shape measures only
    settle what is left.
    """
    width = metrics.get("zone_width_min_m", float("nan"))
    compact = metrics.get("zone_compactness_min", float("nan"))
    aspect = metrics.get("zone_aspect_max", float("nan"))
    area = metrics.get("zone_area_min_m2", float("nan"))

    return (
        -float(metrics.get("zone_n_hard_fail_shapes", 0) or 0),
        -float(metrics.get("zone_n_awkward_shapes", 0) or 0),
        -float(metrics.get("zone_count", 0) or 0),
        float(width) if np.isfinite(width) else float("-inf"),
        float(compact) if np.isfinite(compact) else float("-inf"),
        -(float(aspect) if np.isfinite(aspect) else float("inf")),
        float(area) if np.isfinite(area) else float("-inf"),
    )


def annotate_zone_quality(buildings_m: gpd.GeoDataFrame, limits: ZoneQualityLimits) -> None:
    """Write the `zone_*` metric columns onto every building, in place."""
    metrics_df = buildings_m["zones"].apply(compute_zone_quality_metrics, limits=limits).apply(pd.Series)
    for column in metrics_df.columns:
        buildings_m[column] = metrics_df[column].values


def apply_whole_building_fallback(
    buildings_m: gpd.GeoDataFrame,
    polygon_results_gdf: gpd.GeoDataFrame,
    *,
    decomp_cfg: Any,
    quality_limits: ZoneQualityLimits,
    enabled: bool = True,
) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    """Re-decompose hard-failing buildings whole, and keep the retry if it is better.

    Only two kinds of building are worth retrying: one whose perimeter was never
    split off a core, and one that came out heavily over-segmented. In both the
    shell-by-shell split is what produced the slivers, so removing it is the
    repair. Anything else fails for reasons a retry would reproduce.

    Updates ``buildings_m`` in place and returns the refreshed polygon
    diagnostics plus a summary of what happened.
    """
    annotate_zone_quality(buildings_m, quality_limits)

    buildings_m["zone_shape_fallback_attempted"] = False
    buildings_m["zone_shape_fallback_used"] = False
    buildings_m["zone_shape_fallback_reason"] = pd.Series([None] * len(buildings_m), dtype="object")

    hard_fail_before = int(buildings_m["zone_shape_hard_fail"].fillna(False).sum())
    fallback_mask = (
        enabled
        & buildings_m["zone_shape_hard_fail"].fillna(False)
        & (
            buildings_m["perimeter_only"].fillna(False)
            | buildings_m["zone_count"].fillna(0).gt(quality_limits.oversegmented_zone_count)
        )
    )

    fallback_polygon_rows: list[dict] = []
    fallback_applied: list[str] = []

    for idx, row in tqdm(
        buildings_m.loc[fallback_mask].iterrows(),
        total=int(fallback_mask.sum()),
        desc="Whole-building fallback",
    ):
        buildings_m.at[idx, "zone_shape_fallback_attempted"] = True
        inputs = whole_building_inputs(row)
        if not inputs:
            continue

        result, rows_for_building = decompose_inputs(
            inputs,
            decomp_cfg,
            sample_id=str(row["sample_id"]),
            building_geometry=row.get("geom_local"),
            zone_id_prefix="fallback_poly",
        )
        if not result.get("building_fully_convex") or not result.get("zones"):
            continue

        zones = result["zones"]
        current = compute_zone_quality_metrics(row.get("zones"), quality_limits)
        proposed = compute_zone_quality_metrics(zones, quality_limits)
        if zone_quality_score(proposed) <= zone_quality_score(current):
            continue

        for key, value in result.items():
            if key in buildings_m.columns:
                buildings_m.at[idx, key] = value
        buildings_m.at[idx, "zone_shape_fallback_used"] = True
        buildings_m.at[idx, "zone_shape_fallback_reason"] = "whole_building_no_perimeter"
        fallback_applied.append(str(row["sample_id"]))
        fallback_polygon_rows.extend(rows_for_building)

    annotate_zone_quality(buildings_m, quality_limits)

    if fallback_polygon_rows:
        fallback_ids = {row["sample_id"] for row in fallback_polygon_rows}
        fallback_gdf = gpd.GeoDataFrame(fallback_polygon_rows, geometry="geometry", crs=buildings_m.crs)
        if polygon_results_gdf is None or polygon_results_gdf.empty:
            polygon_results_gdf = fallback_gdf
        else:
            kept = polygon_results_gdf[
                ~polygon_results_gdf["sample_id"].astype(str).isin(fallback_ids)
            ].copy()
            polygon_results_gdf = gpd.GeoDataFrame(
                pd.concat([kept, fallback_gdf], ignore_index=True),
                geometry="geometry",
                crs=buildings_m.crs,
            )

    summary = {
        "hard_fail_before": hard_fail_before,
        "n_candidates": int(fallback_mask.sum()),
        "applied": fallback_applied,
        "hard_fail_after": int(buildings_m["zone_shape_hard_fail"].fillna(False).sum()),
    }
    return polygon_results_gdf, summary
