"""
Perimeter/core zoning for irregular OSM building footprints.

The module implements a construction-first pipeline:
- clean the input polygon or multipolygon,
- build a nominal interior by inward offset,
- keep only interior components that pass simple geometric checks,
- define the perimeter as the residual of the original footprint.

This keeps the notebook focused on orchestration and plotting while the
geometry logic lives in reusable Python code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from helpers.utils import make_geometry_valid, polygon_parts, union_polygons

from .hole_splitter import holeless_polygons


MITRE_JOIN_STYLE = 2


@dataclass(frozen=True)
class PerimeterConfig:
    """How deep the perimeter band is, and when a footprint gets a core at all.

    Offsetting a footprint inward by ``perimeter_depth_m`` leaves the daylit
    band outside the offset and whatever survives inside as the core. Most of
    the remaining knobs decide whether that survivor is a room or an artifact
    of the offset, because a thin or ragged leftover is worse than no core.

    All lengths are metres and all areas square metres, in the local metric
    frame the caller works in.
    """

    #: Depth of the daylit band measured in from the facade.
    perimeter_depth_m: float = 4.5

    #: Smallest a finished zone may be. Anything below is absorbed back.
    min_zone_area_m2: float = 4.0
    #: Smallest a core may be before it is absorbed into the perimeter.
    min_interior_area_m2: float = 4.0
    #: A footprint smaller than this gets no perimeter and is decomposed whole.
    min_input_area_m2: float = 2.0

    #: Shortest edge a finished zone may have. Only enforced where a caller
    #: asks for it; see ``enforce_input_edge_threshold``.
    min_zone_edge_m: float = 2.0
    #: Whether to reject an input footprint whose shortest edge is below
    #: ``min_zone_edge_m``. Off by default: a short edge is common in real
    #: footprints and says nothing about whether the building can be zoned.
    enforce_input_edge_threshold: bool = False

    #: A core must touch the facade offset on at least this many separate
    #: stretches, each at least ``min_contact_group_length_m`` long. One
    #: contact means the "core" is really a notch in the perimeter.
    min_contact_groups: int = 2
    min_contact_group_length_m: float = 0.5
    #: How sharp a turn ends one contact stretch and starts the next.
    contact_turn_angle_deg: float = 35.0

    #: Opening applied to the core to break the necks left by the offset, so
    #: that a dumbbell shape becomes two cores or none rather than one bad one.
    interior_opening_m: float = 0.5
    #: Offset used when probing which side of a boundary the core lies on.
    interior_probe_offset_m: float = 0.5
    #: Edges shorter than this are cleaned off the core after the opening.
    interior_min_edge_cleanup_m: float = 0.1
    #: Simplification applied to the core. Zero keeps it exactly as offset.
    interior_simplify_tolerance_m: float = 0.0

    #: Whether a perimeter containing a hole is cut into parts around it.
    split_perimeter_holes: bool = True

    #: Numerical hygiene: simplification of intermediate geometry, the distance
    #: below which two coordinates are the same point, and the area difference
    #: below which two areas are equal.
    simplify_tolerance_m: float = 0.001
    eps_m: float = 0.01
    area_tol: float = 1e-6
    #: Mitre, so that an offset corner stays a corner instead of rounding off.
    join_style: int = MITRE_JOIN_STYLE

    #: How close in angle and distance an offset edge has to be to its original
    #: to count as the same wall. This correspondence is what lets the
    #: subdivider cut trapezoids between the two.
    offset_match_angle_tol_deg: float = 5.0
    offset_match_distance_tol_m: float = 0.5


@dataclass(frozen=True)
class ComponentDecision:
    keep: bool
    reason: str
    boundary_contact_groups: int
    shortest_edge_m: float
    area_m2: float


def normalize_to_polygons(geom: BaseGeometry | None) -> list[Polygon]:
    """Repair `geom`, then flatten it into real (non-degenerate) polygons."""
    if not isinstance(geom, BaseGeometry):
        return []
    return polygon_parts(make_geometry_valid(geom, drop_invalid=True))


def union_polygonal(parts: list[Polygon] | BaseGeometry | None) -> BaseGeometry | None:
    """Repair every part, then union whatever polygonal geometry survives."""
    candidates = [parts] if isinstance(parts, BaseGeometry) else list(parts or [])
    return union_polygons([p for part in candidates for p in normalize_to_polygons(part)])


def clean_and_validate_geometry(
    geom: BaseGeometry | None,
    *,
    simplify_tolerance_m: float,
) -> BaseGeometry | None:
    geom = make_geometry_valid(geom, drop_invalid=True)
    if geom is None or getattr(geom, "is_empty", True):
        return None

    if simplify_tolerance_m > 0:
        try:
            geom = geom.simplify(simplify_tolerance_m, preserve_topology=True)
        except Exception:
            pass

    geom = make_geometry_valid(geom, drop_invalid=True)
    cleaned = union_polygonal(geom)
    return cleaned


def geometry_area_m2(geom: BaseGeometry | None) -> float:
    return float(sum(poly.area for poly in normalize_to_polygons(geom)))


def polygon_count(geom: BaseGeometry | None) -> int:
    return len(normalize_to_polygons(geom))


def iter_ring_segments(coords: list[tuple[float, float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]


def segment_length(segment: tuple[tuple[float, float], tuple[float, float]]) -> float:
    (x1, y1), (x2, y2) = segment
    return float(math.hypot(x2 - x1, y2 - y1))


def segment_angle_deg(segment: tuple[tuple[float, float], tuple[float, float]]) -> float:
    (x1, y1), (x2, y2) = segment
    return float((math.degrees(math.atan2(y2 - y1, x2 - x1)) + 360.0) % 360.0)


def turn_angle_deg(angle_a: float, angle_b: float) -> float:
    diff = abs(angle_b - angle_a) % 360.0
    return float(min(diff, 360.0 - diff))


def parallel_angle_diff_deg(angle_a: float, angle_b: float) -> float:
    diff = abs(angle_b - angle_a) % 180.0
    return float(min(diff, 180.0 - diff))


def shortest_edge_length_m(geom: BaseGeometry | None) -> float:
    lengths: list[float] = []
    for poly in normalize_to_polygons(geom):
        rings = [poly.exterior, *poly.interiors]
        for ring in rings:
            coords = list(ring.coords)
            for segment in iter_ring_segments(coords):
                length = segment_length(segment)
                if length > 0:
                    lengths.append(length)
    return float(min(lengths)) if lengths else float("nan")


def inward_offset(geom: BaseGeometry | None, distance_m: float, *, join_style: int) -> BaseGeometry | None:
    if geom is None or getattr(geom, "is_empty", True):
        return None
    try:
        offset = geom.buffer(-distance_m, join_style=join_style)
    except Exception:
        return None
    return union_polygonal(offset)


def symmetric_opening(geom: BaseGeometry | None, distance_m: float, *, join_style: int) -> BaseGeometry | None:
    if geom is None or getattr(geom, "is_empty", True):
        return None
    if distance_m <= 0:
        return union_polygonal(geom)

    try:
        shrunk = geom.buffer(-distance_m, join_style=join_style)
    except Exception:
        return union_polygonal(geom)

    if getattr(shrunk, "is_empty", True):
        return union_polygonal(geom)

    try:
        reopened = shrunk.buffer(distance_m, join_style=join_style)
    except Exception:
        return union_polygonal(geom)

    cleaned = clean_and_validate_geometry(reopened, simplify_tolerance_m=0.0)
    if cleaned is None or getattr(cleaned, "is_empty", True):
        return union_polygonal(geom)
    return cleaned


def remove_tiny_ring_segments(coords: list[tuple[float, float]], min_edge_len_m: float) -> list[tuple[float, float]]:
    if min_edge_len_m <= 0:
        return coords

    points = [tuple(map(float, pt)) for pt in coords[:-1]]
    if len(points) < 3:
        return coords

    changed = True
    while changed and len(points) >= 3:
        changed = False
        n = len(points)
        for idx in range(n):
            next_idx = (idx + 1) % n
            if math.hypot(points[next_idx][0] - points[idx][0], points[next_idx][1] - points[idx][1]) < min_edge_len_m:
                del points[next_idx]
                changed = True
                break

    if len(points) < 3:
        return coords
    return [*points, points[0]]


def remove_tiny_polygon_edges(poly: Polygon, min_edge_len_m: float) -> Polygon:
    if min_edge_len_m <= 0 or poly.is_empty:
        return poly

    exterior = remove_tiny_ring_segments(list(poly.exterior.coords), min_edge_len_m)
    interiors = [remove_tiny_ring_segments(list(ring.coords), min_edge_len_m) for ring in poly.interiors]

    try:
        cleaned = Polygon(exterior, interiors)
    except Exception:
        return poly

    valid = make_geometry_valid(cleaned, drop_invalid=True)
    if isinstance(valid, Polygon) and not valid.is_empty:
        return valid
    return poly


def ring_segment_records(ring) -> list[dict[str, Any]]:
    coords = list(ring.coords)
    records: list[dict[str, Any]] = []
    for idx, segment in enumerate(iter_ring_segments(coords)):
        point_a, point_b = segment
        records.append(
            {
                "idx": idx,
                "segment": segment,
                "line": LineString(segment),
                "length_m": segment_length(segment),
                "angle_deg": segment_angle_deg(segment),
                "p1": point_a,
                "p2": point_b,
            }
        )
    return records


def match_ring_correspondence(
    parent_ring,
    inner_ring,
    *,
    component_idx: int,
    parent_ring_kind: str,
    parent_ring_idx: int,
    inner_ring_idx: int,
    cfg: PerimeterConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[LineString]]:
    parent_segments = ring_segment_records(parent_ring)
    inner_segments = ring_segment_records(inner_ring)
    n_parent_vertices = len(parent_segments)
    n_inner_vertices = len(inner_segments)
    if not parent_segments or not inner_segments:
        return [], [], []

    segment_pairs: list[dict[str, Any]] = []
    vertex_pairs: list[dict[str, Any]] = []
    match_lines: list[LineString] = []
    matched_parent_by_inner: dict[int, int] = {}

    for inner_segment in inner_segments:
        best_parent_segment: dict[str, Any] | None = None
        best_score: float | None = None

        for parent_segment in parent_segments:
            angle_diff = parallel_angle_diff_deg(inner_segment["angle_deg"], parent_segment["angle_deg"])
            if angle_diff > cfg.offset_match_angle_tol_deg:
                continue

            distance_to_parent = float(inner_segment["line"].distance(parent_segment["line"]))
            distance_error = abs(distance_to_parent - cfg.perimeter_depth_m)
            if distance_error > cfg.offset_match_distance_tol_m:
                continue

            score = angle_diff + distance_error
            if best_parent_segment is None or score < best_score:
                best_parent_segment = parent_segment
                best_score = score

        if best_parent_segment is None:
            continue

        matched_parent_by_inner[inner_segment["idx"]] = int(best_parent_segment["idx"])
        segment_pairs.append(
            {
                "component_idx": component_idx,
                "parent_ring_kind": parent_ring_kind,
                "parent_ring_idx": int(parent_ring_idx),
                "inner_ring_idx": int(inner_ring_idx),
                "inner_segment_idx": int(inner_segment["idx"]),
                "outer_segment_idx": int(best_parent_segment["idx"]),
                "n_outer_vertices": int(n_parent_vertices),
                "n_inner_vertices": int(n_inner_vertices),
                "angle_diff_deg": float(
                    parallel_angle_diff_deg(inner_segment["angle_deg"], best_parent_segment["angle_deg"])
                ),
                "distance_to_outer_m": float(inner_segment["line"].distance(best_parent_segment["line"])),
            }
        )

    for inner_vertex_idx in range(n_inner_vertices):
        prev_inner_segment = inner_segments[(inner_vertex_idx - 1) % n_inner_vertices]
        cur_inner_segment = inner_segments[inner_vertex_idx]
        parent_prev_idx = matched_parent_by_inner.get(prev_inner_segment["idx"])
        parent_cur_idx = matched_parent_by_inner.get(cur_inner_segment["idx"])
        if parent_prev_idx is None or parent_cur_idx is None:
            continue

        expected_next_idx = (parent_prev_idx + 1) % n_parent_vertices
        expected_prev_idx = (parent_prev_idx - 1) % n_parent_vertices
        if parent_cur_idx == expected_next_idx:
            outer_vertex_idx = parent_cur_idx
            parent_vertex = tuple(float(v) for v in parent_segments[parent_cur_idx]["p1"])
            orientation_step = 1
        elif parent_cur_idx == expected_prev_idx:
            outer_vertex_idx = parent_prev_idx
            parent_vertex = tuple(float(v) for v in parent_segments[parent_prev_idx]["p1"])
            orientation_step = -1
        else:
            continue

        inner_vertex = tuple(float(v) for v in cur_inner_segment["p1"])
        vertex_pairs.append(
            {
                "component_idx": component_idx,
                "parent_ring_kind": parent_ring_kind,
                "parent_ring_idx": int(parent_ring_idx),
                "inner_ring_idx": int(inner_ring_idx),
                "inner_vertex_idx": int(inner_vertex_idx),
                "outer_vertex_idx": int(outer_vertex_idx),
                "outer_point": parent_vertex,
                "inner_point": inner_vertex,
                "outer_prev_segment_idx": int(parent_prev_idx),
                "outer_next_segment_idx": int(parent_cur_idx),
                "inner_prev_segment_idx": int(prev_inner_segment["idx"]),
                "inner_next_segment_idx": int(cur_inner_segment["idx"]),
                "orientation_step": int(orientation_step),
                "n_outer_vertices": int(n_parent_vertices),
                "n_inner_vertices": int(n_inner_vertices),
            }
        )
        match_lines.append(LineString([parent_vertex, inner_vertex]))

    return segment_pairs, vertex_pairs, match_lines


def build_offset_correspondence(
    parent_polygon: BaseGeometry | None,
    nominal_interior: BaseGeometry | None,
    cfg: PerimeterConfig,
) -> dict[str, Any]:
    if not isinstance(parent_polygon, Polygon) or parent_polygon.is_empty:
        return empty_offset_correspondence(nominal_interior=nominal_interior)
    n_outer_vertices = max(len(list(parent_polygon.exterior.coords)) - 1, 0)
    if nominal_interior is None or getattr(nominal_interior, "is_empty", True):
        return empty_offset_correspondence(
            n_outer_vertices=n_outer_vertices,
            nominal_interior=nominal_interior,
        )

    segment_pairs: list[dict[str, Any]] = []
    vertex_pairs: list[dict[str, Any]] = []
    match_lines: list[LineString] = []
    matched_outer_vertex_indices: set[int] = set()
    used_parent_hole_indices: set[int] = set()

    for component_idx, inner_polygon in enumerate(normalize_to_polygons(nominal_interior)):
        outer_seg_pairs, outer_vertex_pairs, outer_lines = match_ring_correspondence(
            parent_polygon.exterior,
            inner_polygon.exterior,
            component_idx=component_idx,
            parent_ring_kind="exterior",
            parent_ring_idx=-1,
            inner_ring_idx=-1,
            cfg=cfg,
        )
        segment_pairs.extend(outer_seg_pairs)
        vertex_pairs.extend(outer_vertex_pairs)
        match_lines.extend(outer_lines)
        matched_outer_vertex_indices.update(pair["outer_vertex_idx"] for pair in outer_vertex_pairs)

        for inner_hole_idx, inner_hole_ring in enumerate(inner_polygon.interiors):
            best_parent_hole_idx: int | None = None
            best_distance: float | None = None
            inner_hole_line = LineString(list(inner_hole_ring.coords))
            for parent_hole_idx, parent_hole_ring in enumerate(parent_polygon.interiors):
                if parent_hole_idx in used_parent_hole_indices:
                    continue
                parent_hole_line = LineString(list(parent_hole_ring.coords))
                distance = float(inner_hole_line.centroid.distance(parent_hole_line.centroid))
                if best_parent_hole_idx is None or distance < best_distance:
                    best_parent_hole_idx = parent_hole_idx
                    best_distance = distance

            if best_parent_hole_idx is None:
                continue

            used_parent_hole_indices.add(best_parent_hole_idx)
            hole_seg_pairs, hole_vertex_pairs, hole_lines = match_ring_correspondence(
                parent_polygon.interiors[best_parent_hole_idx],
                inner_hole_ring,
                component_idx=component_idx,
                parent_ring_kind="hole",
                parent_ring_idx=best_parent_hole_idx,
                inner_ring_idx=inner_hole_idx,
                cfg=cfg,
            )
            segment_pairs.extend(hole_seg_pairs)
            vertex_pairs.extend(hole_vertex_pairs)
            match_lines.extend(hole_lines)

    match_lines_geom: BaseGeometry | None = None
    if match_lines:
        match_lines_geom = MultiLineString(match_lines) if len(match_lines) > 1 else match_lines[0]

    matched_outer_vertex_list = sorted(matched_outer_vertex_indices)
    unmatched_outer_vertex_list = [idx for idx in range(n_outer_vertices) if idx not in matched_outer_vertex_indices]
    nominal_interior_parts = [
        {
            "polygon_idx": 0,
            "component_idx": component_idx,
            "geometry": polygon,
        }
        for component_idx, polygon in enumerate(normalize_to_polygons(nominal_interior))
    ]
    return {
        "offset_segment_pairs": segment_pairs,
        "offset_vertex_pairs": vertex_pairs,
        "offset_match_lines_geom": match_lines_geom,
        "nominal_interior_geom": nominal_interior,
        "nominal_interior_parts": nominal_interior_parts,
        "n_offset_segment_pairs": len(segment_pairs),
        "n_offset_vertex_pairs": len(vertex_pairs),
        "matched_outer_vertex_indices": matched_outer_vertex_list,
        "unmatched_outer_vertex_indices": unmatched_outer_vertex_list,
    }


def empty_offset_correspondence(
    *,
    n_outer_vertices: int = 0,
    nominal_interior: BaseGeometry | None = None,
) -> dict[str, Any]:
    nominal_interior_parts = [
        {
            "polygon_idx": 0,
            "component_idx": component_idx,
            "geometry": polygon,
        }
        for component_idx, polygon in enumerate(normalize_to_polygons(nominal_interior))
    ]
    return {
        "offset_segment_pairs": [],
        "offset_vertex_pairs": [],
        "offset_match_lines_geom": None,
        "nominal_interior_geom": nominal_interior,
        "nominal_interior_parts": nominal_interior_parts,
        "n_offset_segment_pairs": 0,
        "n_offset_vertex_pairs": 0,
        "matched_outer_vertex_indices": [],
        "unmatched_outer_vertex_indices": list(range(n_outer_vertices)),
    }


def count_contact_groups_on_ring(
    ring,
    interior_component: Polygon,
    *,
    distance_limit_m: float,
    turn_tolerance_deg: float,
    min_group_length_m: float,
) -> int:
    coords = list(ring.coords)
    segments = iter_ring_segments(coords)
    if not segments:
        return 0

    flags: list[bool] = []
    lengths: list[float] = []
    angles: list[float] = []

    for segment in segments:
        line = LineString(segment)
        flags.append(float(line.distance(interior_component)) <= distance_limit_m)
        lengths.append(segment_length(segment))
        angles.append(segment_angle_deg(segment))

    groups: list[dict[str, Any]] = []
    for idx, flag in enumerate(flags):
        if not flag:
            continue

        if not groups:
            groups.append({"start": idx, "end": idx, "length": lengths[idx]})
            continue

        prev_idx = idx - 1
        prev_flag = flags[prev_idx]
        prev_angle = angles[prev_idx]
        cur_angle = angles[idx]
        if prev_flag and turn_angle_deg(prev_angle, cur_angle) <= turn_tolerance_deg:
            groups[-1]["end"] = idx
            groups[-1]["length"] += lengths[idx]
        else:
            groups.append({"start": idx, "end": idx, "length": lengths[idx]})

    if len(groups) > 1:
        first = groups[0]
        last = groups[-1]
        wrap_prev_idx = last["end"]
        wrap_next_idx = first["start"]
        if (
            flags[wrap_prev_idx]
            and flags[wrap_next_idx]
            and turn_angle_deg(angles[wrap_prev_idx], angles[wrap_next_idx]) <= turn_tolerance_deg
        ):
            merged = {
                "start": last["start"],
                "end": first["end"],
                "length": last["length"] + first["length"],
            }
            groups = [merged, *groups[1:-1]]

    return int(sum(group["length"] >= min_group_length_m for group in groups))


def count_boundary_contact_groups(
    interior_component: Polygon,
    parent_polygon: Polygon,
    cfg: PerimeterConfig,
) -> int:
    rings = [parent_polygon.exterior, *parent_polygon.interiors]
    count = 0
    for ring in rings:
        count += count_contact_groups_on_ring(
            ring,
            interior_component,
            distance_limit_m=cfg.perimeter_depth_m + cfg.eps_m,
            turn_tolerance_deg=cfg.contact_turn_angle_deg,
            min_group_length_m=cfg.min_contact_group_length_m,
        )
    return count


def classify_interior_component(
    interior_component: Polygon,
    parent_polygon: Polygon,
    cfg: PerimeterConfig,
) -> ComponentDecision:
    area_m2 = float(interior_component.area)
    shortest_edge_m = shortest_edge_length_m(interior_component)

    if area_m2 < cfg.min_interior_area_m2:
        return ComponentDecision(False, "interior_too_small", 0, shortest_edge_m, area_m2)

    probe_offset_m = cfg.interior_probe_offset_m
    if probe_offset_m > 0:
        probe_core = inward_offset(interior_component, probe_offset_m, join_style=cfg.join_style)
        if probe_core is None:
            return ComponentDecision(False, "interior_probe_disappeared", 0, shortest_edge_m, area_m2)

    boundary_contact_groups = count_boundary_contact_groups(interior_component, parent_polygon, cfg)
    if boundary_contact_groups < cfg.min_contact_groups:
        return ComponentDecision(
            False,
            "insufficient_boundary_contact_groups",
            boundary_contact_groups,
            shortest_edge_m,
            area_m2,
        )

    return ComponentDecision(True, "kept", boundary_contact_groups, shortest_edge_m, area_m2)


def prepare_interior_component(
    interior_component: Polygon,
    cfg: PerimeterConfig,
) -> Polygon:
    candidate = interior_component
    if cfg.interior_simplify_tolerance_m > 0:
        try:
            simplified = candidate.simplify(
                cfg.interior_simplify_tolerance_m,
                preserve_topology=True,
            )
            cleaned = clean_and_validate_geometry(simplified, simplify_tolerance_m=0.0)
            if isinstance(cleaned, Polygon) and not cleaned.is_empty:
                candidate = cleaned
        except Exception:
            pass

    candidate = remove_tiny_polygon_edges(candidate, cfg.interior_min_edge_cleanup_m)
    return candidate


def prepare_nominal_interior(
    nominal_interior: BaseGeometry | None,
    cfg: PerimeterConfig,
) -> BaseGeometry | None:
    candidate = union_polygonal(nominal_interior)
    if candidate is None or getattr(candidate, "is_empty", True):
        return None

    if cfg.interior_opening_m > 0:
        candidate = symmetric_opening(candidate, cfg.interior_opening_m, join_style=cfg.join_style)
        if candidate is None or getattr(candidate, "is_empty", True):
            return None

    prepared_parts: list[Polygon] = []
    for polygon in normalize_to_polygons(candidate):
        prepared_parts.append(prepare_interior_component(polygon, cfg))
    return union_polygonal(prepared_parts)


def final_zone_is_valid(
    geom: BaseGeometry | None,
    cfg: PerimeterConfig,
    *,
    enforce_edge_threshold: bool = True,
) -> bool:
    if geom is None or getattr(geom, "is_empty", True):
        return False
    if geometry_area_m2(geom) < cfg.min_zone_area_m2:
        return False
    if enforce_edge_threshold:
        shortest_edge_m = shortest_edge_length_m(geom)
        if np.isfinite(shortest_edge_m) and shortest_edge_m < cfg.min_zone_edge_m:
            return False
    return True


def split_perimeter_parts(perimeter_geom: BaseGeometry | None, cfg: PerimeterConfig) -> list[Polygon]:
    if perimeter_geom is None or getattr(perimeter_geom, "is_empty", True):
        return []
    if cfg.split_perimeter_holes:
        parts = holeless_polygons(
            perimeter_geom,
            area_tol=cfg.area_tol,
            max_edges_per_hole=None,
        )
        if parts:
            return [part for part in parts if part is not None and not part.is_empty and part.area > cfg.area_tol]
    return [part for part in normalize_to_polygons(perimeter_geom) if part.area > cfg.area_tol]


def process_single_polygon(
    polygon: Polygon,
    cfg: PerimeterConfig,
) -> dict[str, Any]:
    polygon = clean_and_validate_geometry(polygon, simplify_tolerance_m=cfg.simplify_tolerance_m)
    n_outer_vertices = 0
    if isinstance(polygon, Polygon) and not polygon.is_empty:
        n_outer_vertices = max(len(list(polygon.exterior.coords)) - 1, 0)
    offset_meta = empty_offset_correspondence(n_outer_vertices=n_outer_vertices)
    if polygon is None:
        return {
            "processed": False,
            "perimeter_defined": False,
            "perimeter": False,
            "has_interior": False,
            "perimeter_only": False,
            "perimeter_reason": "invalid_after_cleanup",
            "interior_geom": None,
            "perimeter_geom": None,
            "perimeter_parts": [],
            "interior_area_m2": 0.0,
            "perimeter_area_m2": 0.0,
            "n_interior_parts": 0,
            "n_perimeter_parts": 0,
            "shortest_input_edge_m": float("nan"),
            "shortest_interior_edge_m": float("nan"),
            "max_boundary_contact_groups": 0,
            "component_reasons": [],
            **offset_meta,
        }

    input_area_m2 = float(polygon.area)
    shortest_input_edge_m = shortest_edge_length_m(polygon)
    if input_area_m2 < cfg.min_input_area_m2:
        return {
            "processed": False,
            "perimeter_defined": False,
            "perimeter": False,
            "has_interior": False,
            "perimeter_only": False,
            "perimeter_reason": "too_small",
            "interior_geom": None,
            "perimeter_geom": None,
            "perimeter_parts": [],
            "interior_area_m2": 0.0,
            "perimeter_area_m2": 0.0,
            "n_interior_parts": 0,
            "n_perimeter_parts": 0,
            "shortest_input_edge_m": shortest_input_edge_m,
            "shortest_interior_edge_m": float("nan"),
            "max_boundary_contact_groups": 0,
            "component_reasons": [],
            **offset_meta,
        }

    if cfg.enforce_input_edge_threshold and np.isfinite(shortest_input_edge_m) and shortest_input_edge_m < cfg.min_zone_edge_m:
        return {
            "processed": False,
            "perimeter_defined": False,
            "perimeter": False,
            "has_interior": False,
            "perimeter_only": False,
            "perimeter_reason": "input_too_narrow",
            "interior_geom": None,
            "perimeter_geom": None,
            "perimeter_parts": [],
            "interior_area_m2": 0.0,
            "perimeter_area_m2": 0.0,
            "n_interior_parts": 0,
            "n_perimeter_parts": 0,
            "shortest_input_edge_m": shortest_input_edge_m,
            "shortest_interior_edge_m": float("nan"),
            "max_boundary_contact_groups": 0,
            "component_reasons": [],
            **offset_meta,
        }

    nominal_interior = inward_offset(polygon, cfg.perimeter_depth_m, join_style=cfg.join_style)
    nominal_interior = prepare_nominal_interior(nominal_interior, cfg)
    offset_meta = build_offset_correspondence(polygon, nominal_interior, cfg)
    if nominal_interior is None:
        perimeter_parts = split_perimeter_parts(polygon, cfg)
        return {
            "processed": True,
            "perimeter_defined": True,
            "perimeter": False,
            "has_interior": False,
            "perimeter_only": True,
            "perimeter_reason": "perimeter_only_no_nominal_interior",
            "interior_geom": None,
            "perimeter_geom": polygon,
            "perimeter_parts": perimeter_parts,
            "interior_area_m2": 0.0,
            "perimeter_area_m2": float(polygon.area),
            "n_interior_parts": 0,
            "n_perimeter_parts": len(perimeter_parts),
            "shortest_input_edge_m": shortest_input_edge_m,
            "shortest_interior_edge_m": float("nan"),
            "max_boundary_contact_groups": 0,
            "component_reasons": [],
            **offset_meta,
        }

    kept_interiors: list[Polygon] = []
    component_reasons: list[str] = []
    kept_shortest_edges: list[float] = []
    boundary_group_counts: list[int] = []

    for component in normalize_to_polygons(nominal_interior):
        decision = classify_interior_component(component, polygon, cfg)
        component_reasons.append(decision.reason)
        if decision.keep:
            kept_interiors.append(component)
            if np.isfinite(decision.shortest_edge_m):
                kept_shortest_edges.append(decision.shortest_edge_m)
            boundary_group_counts.append(decision.boundary_contact_groups)

    interior_geom = union_polygonal(kept_interiors)
    if interior_geom is not None:
        valid_interiors = [
            poly
            for poly in normalize_to_polygons(interior_geom)
            if final_zone_is_valid(poly, cfg, enforce_edge_threshold=False)
        ]
        interior_geom = union_polygonal(valid_interiors)

    if interior_geom is None:
        perimeter_geom = polygon
        perimeter_parts = split_perimeter_parts(perimeter_geom, cfg)
        return {
            "processed": True,
            "perimeter_defined": True,
            "perimeter": False,
            "has_interior": False,
            "perimeter_only": True,
            "perimeter_reason": "perimeter_only_all_interiors_absorbed",
            "interior_geom": None,
            "perimeter_geom": perimeter_geom,
            "perimeter_parts": perimeter_parts,
            "interior_area_m2": 0.0,
            "perimeter_area_m2": float(perimeter_geom.area),
            "n_interior_parts": 0,
            "n_perimeter_parts": len(perimeter_parts),
            "shortest_input_edge_m": shortest_input_edge_m,
            "shortest_interior_edge_m": float("nan"),
            "max_boundary_contact_groups": int(max(boundary_group_counts, default=0)),
            "component_reasons": component_reasons,
            **offset_meta,
        }

    perimeter_geom = union_polygonal(polygon.difference(interior_geom))
    perimeter_parts = split_perimeter_parts(perimeter_geom, cfg)

    return {
        "processed": True,
        "perimeter_defined": True,
        "perimeter": True,
        "has_interior": True,
        "perimeter_only": False,
        "perimeter_reason": "perimeter_with_interior",
        "interior_geom": interior_geom,
        "perimeter_geom": perimeter_geom,
        "perimeter_parts": perimeter_parts,
        "interior_area_m2": geometry_area_m2(interior_geom),
        "perimeter_area_m2": geometry_area_m2(perimeter_geom),
        "n_interior_parts": polygon_count(interior_geom),
        "n_perimeter_parts": len(perimeter_parts),
        "shortest_input_edge_m": shortest_input_edge_m,
        "shortest_interior_edge_m": float(min(kept_shortest_edges)) if kept_shortest_edges else float("nan"),
        "max_boundary_contact_groups": int(max(boundary_group_counts, default=0)),
        "component_reasons": component_reasons,
        **offset_meta,
    }


def process_geometry(
    geom: BaseGeometry | None,
    cfg: PerimeterConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or PerimeterConfig()
    cleaned = clean_and_validate_geometry(geom, simplify_tolerance_m=cfg.simplify_tolerance_m)
    if cleaned is None:
        return process_single_polygon(Polygon(), cfg)

    polygons = normalize_to_polygons(cleaned)
    if not polygons:
        return process_single_polygon(Polygon(), cfg)

    if len(polygons) == 1:
        return process_single_polygon(polygons[0], cfg)

    perim_parts: list[Polygon] = []
    interior_parts: list[Polygon] = []
    component_reasons: list[str] = []
    any_processed = False
    any_interior = False
    shortest_input_edges: list[float] = []
    shortest_interior_edges: list[float] = []
    boundary_group_counts: list[int] = []
    offset_segment_pairs: list[dict[str, Any]] = []
    offset_vertex_pairs: list[dict[str, Any]] = []
    offset_match_line_geoms: list[BaseGeometry] = []
    nominal_interior_parts_raw: list[Polygon] = []
    nominal_interior_parts_records: list[dict[str, Any]] = []
    for polygon_idx, polygon in enumerate(polygons):
        result = process_single_polygon(polygon, cfg)
        component_reasons.extend(result["component_reasons"])
        offset_segment_pairs.extend(
            [{**pair, "polygon_idx": polygon_idx} for pair in result["offset_segment_pairs"]]
        )
        offset_vertex_pairs.extend(
            [{**pair, "polygon_idx": polygon_idx} for pair in result["offset_vertex_pairs"]]
        )
        if result["offset_match_lines_geom"] is not None and not getattr(result["offset_match_lines_geom"], "is_empty", True):
            offset_match_line_geoms.append(result["offset_match_lines_geom"])
        nominal_interior_parts_raw.extend(normalize_to_polygons(result["nominal_interior_geom"]))
        nominal_interior_parts_records.extend(
            [
                {**part_record, "polygon_idx": polygon_idx}
                for part_record in result["nominal_interior_parts"]
            ]
        )
        if np.isfinite(result["shortest_input_edge_m"]):
            shortest_input_edges.append(float(result["shortest_input_edge_m"]))
        if np.isfinite(result["shortest_interior_edge_m"]):
            shortest_interior_edges.append(float(result["shortest_interior_edge_m"]))
        boundary_group_counts.append(int(result["max_boundary_contact_groups"]))

        if result["processed"]:
            any_processed = True
            perim_parts.extend(normalize_to_polygons(result["perimeter_geom"]))
            interior_parts.extend(normalize_to_polygons(result["interior_geom"]))
            any_interior = any_interior or bool(result["has_interior"])
        else:
            perim_parts.extend(normalize_to_polygons(polygon))

    if not any_processed:
        matched_outer_vertices = sorted(
            {(pair["polygon_idx"], pair["outer_vertex_idx"]) for pair in offset_vertex_pairs}
        )
        return {
            "processed": False,
            "perimeter_defined": False,
            "perimeter": False,
            "has_interior": False,
            "perimeter_only": False,
            "perimeter_reason": "not_suitable_for_processing",
            "interior_geom": None,
            "perimeter_geom": None,
            "perimeter_parts": [],
            "interior_area_m2": 0.0,
            "perimeter_area_m2": 0.0,
            "n_interior_parts": 0,
            "n_perimeter_parts": 0,
            "shortest_input_edge_m": float(min(shortest_input_edges)) if shortest_input_edges else float("nan"),
            "shortest_interior_edge_m": float("nan"),
            "max_boundary_contact_groups": int(max(boundary_group_counts, default=0)),
            "component_reasons": component_reasons,
            "offset_segment_pairs": offset_segment_pairs,
            "offset_vertex_pairs": offset_vertex_pairs,
            "offset_match_lines_geom": unary_union(offset_match_line_geoms) if offset_match_line_geoms else None,
            "nominal_interior_geom": union_polygonal(nominal_interior_parts_raw),
            "nominal_interior_parts": nominal_interior_parts_records,
            "n_offset_segment_pairs": len(offset_segment_pairs),
            "n_offset_vertex_pairs": len(offset_vertex_pairs),
            "matched_outer_vertex_indices": matched_outer_vertices,
            "unmatched_outer_vertex_indices": [],
        }

    interior_geom = union_polygonal(interior_parts)
    perimeter_geom = union_polygonal(perim_parts)
    perimeter_split_parts = split_perimeter_parts(perimeter_geom, cfg)
    offset_match_lines_geom = unary_union(offset_match_line_geoms) if offset_match_line_geoms else None
    matched_outer_vertices = sorted(
        {(pair["polygon_idx"], pair["outer_vertex_idx"]) for pair in offset_vertex_pairs}
    )

    return {
        "processed": True,
        "perimeter_defined": True,
        "perimeter": any_interior,
        "has_interior": any_interior,
        "perimeter_only": not any_interior,
        "perimeter_reason": "perimeter_with_interior" if any_interior else "perimeter_only_multipolygon",
        "interior_geom": interior_geom,
        "perimeter_geom": perimeter_geom,
        "perimeter_parts": perimeter_split_parts,
        "interior_area_m2": geometry_area_m2(interior_geom),
        "perimeter_area_m2": geometry_area_m2(perimeter_geom),
        "n_interior_parts": polygon_count(interior_geom),
        "n_perimeter_parts": len(perimeter_split_parts),
        "shortest_input_edge_m": float(min(shortest_input_edges)) if shortest_input_edges else float("nan"),
        "shortest_interior_edge_m": float(min(shortest_interior_edges)) if shortest_interior_edges else float("nan"),
        "max_boundary_contact_groups": int(max(boundary_group_counts, default=0)),
        "component_reasons": component_reasons,
        "offset_segment_pairs": offset_segment_pairs,
        "offset_vertex_pairs": offset_vertex_pairs,
        "offset_match_lines_geom": offset_match_lines_geom,
        "nominal_interior_geom": union_polygonal(nominal_interior_parts_raw),
        "nominal_interior_parts": nominal_interior_parts_records,
        "n_offset_segment_pairs": len(offset_segment_pairs),
        "n_offset_vertex_pairs": len(offset_vertex_pairs),
        "matched_outer_vertex_indices": matched_outer_vertices,
        "unmatched_outer_vertex_indices": [],
    }


def build_perimeter_zones_for_gdf(
    gdf: gpd.GeoDataFrame,
    *,
    geometry_col: str = "geometry",
    cfg: PerimeterConfig | None = None,
    drop_existing: bool = True,
) -> gpd.GeoDataFrame:
    cfg = cfg or PerimeterConfig()
    out = gdf.copy()

    result_columns = [
        "processed",
        "perimeter_defined",
        "perimeter",
        "has_interior",
        "perimeter_only",
        "perimeter_reason",
        "interior_geom",
        "perimeter_geom",
        "perimeter_parts",
        "interior_area_m2",
        "perimeter_area_m2",
        "n_interior_parts",
        "n_perimeter_parts",
        "shortest_input_edge_m",
        "shortest_interior_edge_m",
        "max_boundary_contact_groups",
        "component_reasons",
        "offset_segment_pairs",
        "offset_vertex_pairs",
        "offset_match_lines_geom",
        "nominal_interior_geom",
        "nominal_interior_parts",
        "n_offset_segment_pairs",
        "n_offset_vertex_pairs",
        "matched_outer_vertex_indices",
        "unmatched_outer_vertex_indices",
    ]
    if drop_existing:
        out = out.drop(columns=result_columns, errors="ignore")

    records = [process_geometry(geom, cfg=cfg) for geom in out[geometry_col]]
    result_df = pd.DataFrame(records, index=out.index)
    out = out.join(result_df)
    return out


def summarize_perimeter_results(gdf: pd.DataFrame) -> dict[str, Any]:
    perimeter_defined_series = gdf["perimeter_defined"].fillna(False).astype(bool)
    perimeter_series = gdf["perimeter"].fillna(False).astype(bool)
    has_interior_series = gdf["has_interior"].fillna(False).astype(bool)
    perimeter_only_series = gdf["perimeter_only"].fillna(False).astype(bool)
    reason_counts = gdf["perimeter_reason"].fillna("missing").value_counts().to_dict()

    return {
        "n_total": int(len(gdf)),
        "n_perimeter_defined": int(perimeter_defined_series.sum()),
        "n_no_perimeter_defined": int((~perimeter_defined_series).sum()),
        "n_with_interior": int((perimeter_series & has_interior_series).sum()),
        "n_perimeter_only": int(perimeter_only_series.sum()),
        "reason_counts": reason_counts,
    }
