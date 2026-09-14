"""
Trapezoid-first subdivision of perimeter zones using exact outer-to-inner
vertex correspondences from the perimeter builder.

The current rule is intentionally strict:
- take consecutive exact vertex matches on the nominal inner offset,
- create four-point trapezoids from those exact anchor pairs,
- add collapsed-corner triangles when one inner vertex corresponds to two
  outer vertices,
- keep the leftover perimeter residual as-is (possibly disjoint).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from .osm_perimeter_builder import (
    clean_and_validate_geometry,
    normalize_to_polygons,
    union_polygonal,
)


@dataclass(frozen=True)
class PerimeterSubdivisionConfig:
    """What the pre-check will carve out of a perimeter band before the search.

    A perimeter band around a rectangular building is four trapezoids and
    nothing else, and finding that by search is wasted effort. The pre-check
    cuts the obvious pieces first and hands the search only what is left.
    """

    #: A trapezoid smaller than this is left in the remainder instead, where
    #: the search can merge it into a neighbour rather than making it a zone.
    min_trapezoid_area_m2: float = 4.0
    #: Simplification applied to the pieces. Zero keeps them exactly as cut.
    simplify_tolerance_m: float = 0.0


def _rows_to_gdf(rows: list[dict[str, Any]], *, crs) -> gpd.GeoDataFrame:
    if rows:
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)


def _pairs_with_polygon_idx(offset_vertex_pairs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for pair in offset_vertex_pairs or []:
        if "polygon_idx" not in pair:
            pairs.append({**pair, "polygon_idx": 0})
        else:
            pairs.append(dict(pair))
    return pairs


def _segment_pairs_with_polygon_idx(offset_segment_pairs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for pair in offset_segment_pairs or []:
        if "polygon_idx" not in pair:
            pairs.append({**pair, "polygon_idx": 0})
        else:
            pairs.append(dict(pair))
    return pairs


def _outer_polygon_map(geom: BaseGeometry | None) -> dict[int, Polygon]:
    return {polygon_idx: polygon for polygon_idx, polygon in enumerate(normalize_to_polygons(geom))}


def _nominal_component_map(row: dict[str, Any] | Any) -> dict[tuple[int, int], Polygon]:
    component_map: dict[tuple[int, int], Polygon] = {}
    for part_record in row.get("nominal_interior_parts") or []:
        geometry = part_record.get("geometry")
        if geometry is None or getattr(geometry, "is_empty", True):
            continue
        component_map[(int(part_record.get("polygon_idx", 0)), int(part_record.get("component_idx", 0)))] = geometry
    if component_map:
        return component_map

    for component_idx, polygon in enumerate(normalize_to_polygons(row.get("nominal_interior_geom"))):
        component_map[(0, component_idx)] = polygon
    return component_map


def _parent_ring_coords(
    polygon_map: dict[int, Polygon],
    *,
    polygon_idx: int,
    ring_kind: str,
    ring_idx: int,
) -> list[tuple[float, float]]:
    polygon = polygon_map.get(polygon_idx)
    if polygon is None:
        return []
    if ring_kind == "hole":
        if ring_idx < 0 or ring_idx >= len(polygon.interiors):
            return []
        return list(polygon.interiors[ring_idx].coords)[:-1]
    return list(polygon.exterior.coords)[:-1]


def _inner_ring_coords(
    component_map: dict[tuple[int, int], Polygon],
    *,
    polygon_idx: int,
    component_idx: int,
    inner_ring_idx: int,
) -> list[tuple[float, float]]:
    polygon = component_map.get((polygon_idx, component_idx))
    if polygon is None:
        return []
    if inner_ring_idx >= 0:
        if inner_ring_idx >= len(polygon.interiors):
            return []
        return list(polygon.interiors[inner_ring_idx].coords)[:-1]
    return list(polygon.exterior.coords)[:-1]


def _build_trapezoid_geometry(
    pair_a: dict[str, Any],
    pair_b: dict[str, Any],
    available_geom: BaseGeometry | None,
    cfg: PerimeterSubdivisionConfig,
) -> list[Polygon]:
    candidate = Polygon(
        [
            tuple(pair_a["outer_point"]),
            tuple(pair_b["outer_point"]),
            tuple(pair_b["inner_point"]),
            tuple(pair_a["inner_point"]),
        ]
    )
    candidate = _safe_polygonal_intersection(candidate, available_geom, cfg=cfg)
    return [
        polygon
        for polygon in normalize_to_polygons(candidate)
        if polygon.area >= cfg.min_trapezoid_area_m2
    ]


def _build_corner_geometry(
    outer_point_a: tuple[float, float],
    outer_point_b: tuple[float, float],
    inner_point: tuple[float, float],
    available_geom: BaseGeometry | None,
    cfg: PerimeterSubdivisionConfig,
) -> list[Polygon]:
    candidate = Polygon([outer_point_a, outer_point_b, inner_point])
    candidate = _safe_polygonal_intersection(candidate, available_geom, cfg=cfg)
    return [
        polygon
        for polygon in normalize_to_polygons(candidate)
        if polygon.area >= cfg.min_trapezoid_area_m2
    ]


def _piecewise_difference(
    left: BaseGeometry | None,
    right: BaseGeometry | None,
    *,
    cfg: PerimeterSubdivisionConfig,
) -> BaseGeometry | None:
    left_parts = normalize_to_polygons(left)
    if not left_parts:
        return None

    right_parts = normalize_to_polygons(right)
    if not right_parts:
        return union_polygonal(left_parts)

    current_parts = left_parts
    for cutter in right_parts:
        next_parts: list[Polygon] = []
        for part in current_parts:
            try:
                diff = part.difference(cutter)
            except Exception:
                part_clean = clean_and_validate_geometry(part, simplify_tolerance_m=cfg.simplify_tolerance_m)
                cutter_clean = clean_and_validate_geometry(cutter, simplify_tolerance_m=cfg.simplify_tolerance_m)
                if part_clean is None:
                    continue
                if cutter_clean is None:
                    next_parts.extend(normalize_to_polygons(part_clean))
                    continue
                try:
                    diff = part_clean.difference(cutter_clean)
                except Exception:
                    diff = part_clean
            next_parts.extend(normalize_to_polygons(diff))
        current_parts = next_parts
        if not current_parts:
            return None

    return union_polygonal(current_parts)


def _safe_polygonal_intersection(
    left: BaseGeometry | None,
    right: BaseGeometry | None,
    *,
    cfg: PerimeterSubdivisionConfig,
) -> BaseGeometry | None:
    left_clean = clean_and_validate_geometry(left, simplify_tolerance_m=cfg.simplify_tolerance_m)
    if left_clean is None or getattr(left_clean, "is_empty", True):
        return None

    if right is None or getattr(right, "is_empty", True):
        return left_clean

    right_clean = clean_and_validate_geometry(right, simplify_tolerance_m=cfg.simplify_tolerance_m)
    if right_clean is None or getattr(right_clean, "is_empty", True):
        return None

    try:
        return clean_and_validate_geometry(
            left_clean.intersection(right_clean),
            simplify_tolerance_m=cfg.simplify_tolerance_m,
        )
    except Exception:
        pieces: list[Polygon] = []
        for left_part in normalize_to_polygons(left_clean):
            for right_part in normalize_to_polygons(right_clean):
                try:
                    part = left_part.intersection(right_part)
                except Exception:
                    continue
                pieces.extend(normalize_to_polygons(part))
        return union_polygonal(pieces)


def _safe_polygonal_difference(
    left: BaseGeometry | None,
    right: BaseGeometry | None,
    *,
    cfg: PerimeterSubdivisionConfig,
) -> BaseGeometry | None:
    left_clean = clean_and_validate_geometry(left, simplify_tolerance_m=cfg.simplify_tolerance_m)
    if left_clean is None or getattr(left_clean, "is_empty", True):
        return None

    if right is None or getattr(right, "is_empty", True):
        return left_clean

    right_clean = clean_and_validate_geometry(right, simplify_tolerance_m=cfg.simplify_tolerance_m)
    if right_clean is None or getattr(right_clean, "is_empty", True):
        return left_clean

    try:
        return clean_and_validate_geometry(
            left_clean.difference(right_clean),
            simplify_tolerance_m=cfg.simplify_tolerance_m,
        )
    except Exception:
        return clean_and_validate_geometry(
            _piecewise_difference(left_clean, right_clean, cfg=cfg),
            simplify_tolerance_m=cfg.simplify_tolerance_m,
        )


def subdivide_perimeter_row(
    row: dict[str, Any] | Any,
    cfg: PerimeterSubdivisionConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or PerimeterSubdivisionConfig()

    perimeter_geom = clean_and_validate_geometry(
        row.get("perimeter_geom"),
        simplify_tolerance_m=cfg.simplify_tolerance_m,
    )
    if perimeter_geom is None:
        return {
            "trapezoid_parts": [],
            "corner_parts": [],
            "remainder_geom": None,
            "n_trapezoids": 0,
            "n_corner_parts": 0,
            "n_remainder_parts": 0,
            "subdivision_reason": "no_perimeter_geom",
        }

    component_reasons = list(row.get("component_reasons") or [])
    retained_interior = row.get("interior_geom")
    nominal_interior = row.get("nominal_interior_geom")
    if (
        (retained_interior is None or getattr(retained_interior, "is_empty", True))
        and nominal_interior is not None
        and not getattr(nominal_interior, "is_empty", True)
        and component_reasons
        and all(reason == "interior_too_small" for reason in component_reasons)
    ):
        return {
            "trapezoid_parts": [],
            "corner_parts": [],
            "remainder_geom": perimeter_geom,
            "n_trapezoids": 0,
            "n_corner_parts": 0,
            "n_remainder_parts": len(normalize_to_polygons(perimeter_geom)),
            "subdivision_reason": "discarded_small_interior_kept_as_single_zone",
        }

    offset_vertex_pairs = _pairs_with_polygon_idx(row.get("offset_vertex_pairs"))
    offset_segment_pairs = _segment_pairs_with_polygon_idx(row.get("offset_segment_pairs"))
    if not offset_vertex_pairs and not offset_segment_pairs:
        return {
            "trapezoid_parts": [],
            "corner_parts": [],
            "remainder_geom": perimeter_geom,
            "n_trapezoids": 0,
            "n_corner_parts": 0,
            "n_remainder_parts": len(normalize_to_polygons(perimeter_geom)),
            "subdivision_reason": "insufficient_exact_vertex_matches",
        }

    outer_polygon_map = _outer_polygon_map(row.get("geometry"))
    nominal_component_map = _nominal_component_map(row)
    trapezoid_parts: list[Polygon] = []
    trapezoid_records: list[dict[str, Any]] = []
    corner_parts: list[Polygon] = []
    corner_records: list[dict[str, Any]] = []

    vertex_grouped: dict[tuple[int, int, str, int, int], list[dict[str, Any]]] = {}
    for pair in offset_vertex_pairs:
        key = (
            int(pair["polygon_idx"]),
            int(pair["component_idx"]),
            str(pair.get("parent_ring_kind", "exterior")),
            int(pair.get("parent_ring_idx", -1)),
            int(pair.get("inner_ring_idx", -1)),
        )
        vertex_grouped.setdefault(key, []).append(pair)
    segment_grouped: dict[tuple[int, int, str, int, int], list[dict[str, Any]]] = {}
    for pair in offset_segment_pairs:
        key = (
            int(pair["polygon_idx"]),
            int(pair["component_idx"]),
            str(pair.get("parent_ring_kind", "exterior")),
            int(pair.get("parent_ring_idx", -1)),
            int(pair.get("inner_ring_idx", -1)),
        )
        segment_grouped.setdefault(key, []).append(pair)

    available_geom = perimeter_geom

    for key, pairs in vertex_grouped.items():
        polygon_idx, component_idx, parent_ring_kind, parent_ring_idx, inner_ring_idx = key
        n_outer_vertices = int(pairs[0].get("n_outer_vertices", 0))
        n_inner_vertices = int(pairs[0].get("n_inner_vertices", 0))
        if n_outer_vertices <= 0 or n_inner_vertices <= 0 or len(pairs) < 2:
            continue

        ordered_pairs = sorted(pairs, key=lambda pair: int(pair["inner_vertex_idx"]))
        n_pairs = len(ordered_pairs)

        for idx in range(n_pairs):
            pair_a = ordered_pairs[idx]
            pair_b = ordered_pairs[(idx + 1) % n_pairs]

            outer_step = int(pair_a.get("orientation_step", 1))
            expected_outer_next = (int(pair_a["outer_vertex_idx"]) + outer_step) % n_outer_vertices
            expected_inner_next = (int(pair_a["inner_vertex_idx"]) + 1) % n_inner_vertices
            if int(pair_b["outer_vertex_idx"]) != expected_outer_next:
                continue
            if int(pair_b["inner_vertex_idx"]) != expected_inner_next:
                continue

            trapezoids = _build_trapezoid_geometry(pair_a, pair_b, available_geom, cfg)
            if not trapezoids:
                continue

            for trapezoid in trapezoids:
                trapezoid_parts.append(trapezoid)
                trapezoid_records.append(
                    {
                        "polygon_idx": polygon_idx,
                        "component_idx": component_idx,
                        "parent_ring_kind": parent_ring_kind,
                        "parent_ring_idx": parent_ring_idx,
                        "inner_ring_idx": inner_ring_idx,
                        "outer_start_vertex_idx": int(pair_a["outer_vertex_idx"]),
                        "outer_end_vertex_idx": int(pair_b["outer_vertex_idx"]),
                        "inner_start_vertex_idx": int(pair_a["inner_vertex_idx"]),
                        "inner_end_vertex_idx": int(pair_b["inner_vertex_idx"]),
                        "construction_mode": "exact_vertex",
                        "geometry": trapezoid,
                    }
                )
            available_geom = (
                _safe_polygonal_difference(
                    available_geom,
                    union_polygonal(trapezoids),
                    cfg=cfg,
                )
                if available_geom is not None
                else None
            )

    for key, pairs in segment_grouped.items():
        polygon_idx, component_idx, parent_ring_kind, parent_ring_idx, inner_ring_idx = key
        outer_coords = _parent_ring_coords(
            outer_polygon_map,
            polygon_idx=polygon_idx,
            ring_kind=parent_ring_kind,
            ring_idx=parent_ring_idx,
        )
        inner_coords = _inner_ring_coords(
            nominal_component_map,
            polygon_idx=polygon_idx,
            component_idx=component_idx,
            inner_ring_idx=inner_ring_idx,
        )
        if not outer_coords or not inner_coords:
            continue

        n_outer_vertices = int(pairs[0].get("n_outer_vertices", len(outer_coords)))
        n_inner_vertices = int(pairs[0].get("n_inner_vertices", len(inner_coords)))
        segment_map = {int(pair["inner_segment_idx"]): int(pair["outer_segment_idx"]) for pair in pairs}

        for inner_vertex_idx in range(n_inner_vertices):
            prev_inner_segment_idx = (inner_vertex_idx - 1) % n_inner_vertices
            next_inner_segment_idx = inner_vertex_idx
            outer_prev_segment_idx = segment_map.get(prev_inner_segment_idx)
            outer_next_segment_idx = segment_map.get(next_inner_segment_idx)
            if outer_prev_segment_idx is None or outer_next_segment_idx is None:
                continue

            gap = (outer_next_segment_idx - outer_prev_segment_idx) % n_outer_vertices
            if gap != 2:
                continue

            outer_vertex_idx_a = (outer_prev_segment_idx + 1) % n_outer_vertices
            outer_vertex_idx_b = (outer_prev_segment_idx + 2) % n_outer_vertices
            inner_point = tuple(inner_coords[inner_vertex_idx])
            outer_point_a = tuple(outer_coords[outer_vertex_idx_a])
            outer_point_b = tuple(outer_coords[outer_vertex_idx_b])
            corner_geoms = _build_corner_geometry(
                outer_point_a,
                outer_point_b,
                inner_point,
                available_geom,
                cfg,
            )
            if not corner_geoms:
                continue

            for corner_geom in corner_geoms:
                corner_parts.append(corner_geom)
                corner_records.append(
                    {
                        "polygon_idx": polygon_idx,
                        "component_idx": component_idx,
                        "parent_ring_kind": parent_ring_kind,
                        "parent_ring_idx": parent_ring_idx,
                        "inner_ring_idx": inner_ring_idx,
                        "outer_start_vertex_idx": int(outer_vertex_idx_a),
                        "outer_end_vertex_idx": int(outer_vertex_idx_b),
                        "inner_vertex_idx": int(inner_vertex_idx),
                        "construction_mode": "collapsed_corner",
                        "geometry": corner_geom,
                    }
                )

            if available_geom is not None:
                available_geom = _safe_polygonal_difference(
                    available_geom,
                    union_polygonal(corner_geoms),
                    cfg=cfg,
                )

    all_matched_parts = trapezoid_parts + corner_parts
    matched_union = union_polygonal(all_matched_parts)
    remainder_geom = (
        _safe_polygonal_difference(perimeter_geom, matched_union, cfg=cfg)
        if matched_union is not None
        else perimeter_geom
    )
    remainder_geom = clean_and_validate_geometry(remainder_geom, simplify_tolerance_m=cfg.simplify_tolerance_m)

    return {
        "trapezoid_parts": trapezoid_parts,
        "trapezoid_records": trapezoid_records,
        "corner_parts": corner_parts,
        "corner_records": corner_records,
        "remainder_geom": remainder_geom,
        "n_trapezoids": len(trapezoid_parts),
        "n_corner_parts": len(corner_parts),
        "n_remainder_parts": len(normalize_to_polygons(remainder_geom)),
        "subdivision_reason": (
            "matched_trapezoids_and_corners"
            if trapezoid_parts and corner_parts
            else "matched_trapezoids"
            if trapezoid_parts
            else "collapsed_corner_parts"
            if corner_parts
            else "insufficient_consecutive_matches"
        ),
    }


def subdivide_perimeter_gdf(
    gdf: gpd.GeoDataFrame,
    *,
    id_col: str = "sample_id",
    cfg: PerimeterSubdivisionConfig | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    cfg = cfg or PerimeterSubdivisionConfig()

    result_gdf = gdf.copy()
    trapezoid_rows: list[dict[str, Any]] = []
    corner_rows: list[dict[str, Any]] = []
    remainder_rows: list[dict[str, Any]] = []
    subdivision_reasons: list[str] = []
    subdivision_errors: list[str | None] = []
    trapezoid_counts: list[int] = []
    corner_counts: list[int] = []
    remainder_counts: list[int] = []
    remainder_geoms: list[BaseGeometry | None] = []

    for _, row in result_gdf.iterrows():
        try:
            result = subdivide_perimeter_row(row, cfg=cfg)
            subdivision_error = None
        except Exception as exc:
            perimeter_geom = clean_and_validate_geometry(
                row.get("perimeter_geom"),
                simplify_tolerance_m=cfg.simplify_tolerance_m,
            )
            result = {
                "trapezoid_parts": [],
                "trapezoid_records": [],
                "corner_parts": [],
                "corner_records": [],
                "remainder_geom": perimeter_geom,
                "n_trapezoids": 0,
                "n_corner_parts": 0,
                "n_remainder_parts": len(normalize_to_polygons(perimeter_geom)),
                "subdivision_reason": "subdivision_failed_fallback_to_unsplit_perimeter",
            }
            subdivision_error = f"{type(exc).__name__}: {exc}"
        subdivision_reasons.append(result["subdivision_reason"])
        subdivision_errors.append(subdivision_error)
        trapezoid_counts.append(result["n_trapezoids"])
        corner_counts.append(result["n_corner_parts"])
        remainder_counts.append(result["n_remainder_parts"])
        remainder_geoms.append(result["remainder_geom"])

        sample_id = row[id_col]
        for trapezoid_idx, trapezoid_record in enumerate(result.get("trapezoid_records", []), start=1):
            trapezoid_rows.append(
                {
                    id_col: sample_id,
                    "trapezoid_idx": trapezoid_idx,
                    "zone_id": f"{sample_id}_trapezoid_{trapezoid_idx:02d}",
                    "subdivision_reason": result["subdivision_reason"],
                    "polygon_idx": trapezoid_record["polygon_idx"],
                    "component_idx": trapezoid_record["component_idx"],
                    "parent_ring_kind": trapezoid_record["parent_ring_kind"],
                    "parent_ring_idx": trapezoid_record["parent_ring_idx"],
                    "inner_ring_idx": trapezoid_record["inner_ring_idx"],
                    "outer_start_vertex_idx": trapezoid_record["outer_start_vertex_idx"],
                    "outer_end_vertex_idx": trapezoid_record["outer_end_vertex_idx"],
                    "inner_start_vertex_idx": trapezoid_record["inner_start_vertex_idx"],
                    "inner_end_vertex_idx": trapezoid_record["inner_end_vertex_idx"],
                    "construction_mode": trapezoid_record.get("construction_mode", "exact_vertex"),
                    "geometry": trapezoid_record["geometry"],
                }
            )

        for corner_idx, corner_record in enumerate(result.get("corner_records", []), start=1):
            corner_rows.append(
                {
                    id_col: sample_id,
                    "corner_idx": corner_idx,
                    "zone_id": f"{sample_id}_corner_{corner_idx:02d}",
                    "subdivision_reason": result["subdivision_reason"],
                    "polygon_idx": corner_record["polygon_idx"],
                    "component_idx": corner_record["component_idx"],
                    "parent_ring_kind": corner_record["parent_ring_kind"],
                    "parent_ring_idx": corner_record["parent_ring_idx"],
                    "inner_ring_idx": corner_record["inner_ring_idx"],
                    "outer_start_vertex_idx": corner_record["outer_start_vertex_idx"],
                    "outer_end_vertex_idx": corner_record["outer_end_vertex_idx"],
                    "inner_vertex_idx": corner_record["inner_vertex_idx"],
                    "construction_mode": corner_record.get("construction_mode", "collapsed_corner"),
                    "geometry": corner_record["geometry"],
                }
            )

        for remainder_idx, polygon in enumerate(normalize_to_polygons(result["remainder_geom"]), start=1):
            remainder_rows.append(
                {
                    id_col: sample_id,
                    "remainder_idx": remainder_idx,
                    "zone_id": f"{sample_id}_remainder_{remainder_idx:02d}",
                    "subdivision_reason": result["subdivision_reason"],
                    "geometry": polygon,
                }
            )

    result_gdf["trapezoid_subdivision_reason"] = subdivision_reasons
    result_gdf["trapezoid_subdivision_error"] = subdivision_errors
    result_gdf["n_trapezoids"] = trapezoid_counts
    result_gdf["n_corner_parts"] = corner_counts
    result_gdf["n_remainder_parts"] = remainder_counts
    result_gdf["trapezoid_remainder_geom"] = remainder_geoms

    trapezoids_gdf = _rows_to_gdf(trapezoid_rows, crs=gdf.crs)
    corners_gdf = _rows_to_gdf(corner_rows, crs=gdf.crs)
    remainder_gdf = _rows_to_gdf(remainder_rows, crs=gdf.crs)
    return result_gdf, trapezoids_gdf, corners_gdf, remainder_gdf
