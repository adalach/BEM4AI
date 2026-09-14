"""Square up a footprint's walls without reshaping the building.

An OSM footprint is traced by hand, so walls that are meant to be parallel are
a degree or two apart and walls that are meant to be one line are offset by a
few centimetres. Left alone, each of those becomes a separate surface in the
energy model.

The fix is to cluster edges that are nearly colinear by angle and offset, snap
the ring's vertices onto the cluster lines, and then clean up the near-collinear
vertices and very short segments that snapping leaves behind.

It is deliberately conservative: a vertex may only move so far, and a result
that changes the area, stretches an edge or moves the outline more than the
guards in `EdgeAlignmentConfig` allow is thrown away in favour of the original.
This is preprocessing, not redrawing.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import LinearRing, Polygon

from helpers.utils import count_exterior_vertices

__all__ = [
    "EdgeAlignmentConfig",
    "align_polygon_edges_via_line_clustering",
    "align_buildings_gdf",
    "get_line_clusters_for_polygon",
    "align_neighbors_gdf",
]


@dataclass
class EdgeAlignmentConfig:
    """Which edges count as the same wall, and how far one may be moved.

    Alignment works in two stages. Edges are first grouped into *families* by
    direction alone - all the roughly north-facing walls together - and then,
    within a family, into *lines* by direction and perpendicular offset, which
    is what actually gets snapped. The family tolerance is the looser of the
    two on purpose: a wall may be a couple of degrees off and still belong to
    the same run of the building, while only edges that are nearly the same
    line should be pulled onto one.

    The last five fields are the guards. They are what separates squaring up a
    footprint from redrawing it: any aligned result that moves a vertex too
    far, changes the area, stretches an edge or shifts the outline beyond these
    limits is discarded in favour of the original.

    All lengths are metres in the local metric frame.
    """

    #: Group edges by direction before clustering by line. With this off,
    #: clustering sees every edge at once and cross-matches opposite walls.
    use_angle_families: bool = True
    #: How far apart in direction two edges may be and share a family,
    family_angle_tol_deg: float = 3.0
    #: and the tighter pair of tolerances for sharing a line within one.
    line_angle_tol_deg: float = 2.0
    line_offset_tol_m: float = 0.2

    #: A line cluster is only strong enough to snap other edges onto if it
    #: carries this much edge length in total, or one edge at least this long.
    #: Without it a pair of stray fragments can drag a real wall off true.
    min_cluster_total_length_m: float = 2.0
    min_cluster_longest_edge_m: float = 1.0
    #: A cluster too weak by those limits is offered to a strong neighbouring
    #: cluster within these looser tolerances before being left alone.
    small_cluster_reassign_angle_tol_deg: float = 4.0
    small_cluster_reassign_offset_tol_m: float = 0.4

    #: Degenerate edges, below which an edge has no meaningful direction.
    min_edge_len_m: float = 1e-12
    #: Vertices this close to straight are dropped after snapping,
    collinear_tol_deg: float = 1.0
    #: as are segments this short.
    min_seg_len_m: float = 0.05

    #: The guards. No vertex may move further than this,
    max_vertex_snap_shift_m: float = 0.4
    #: the area must stay within this band,
    align_min_area_ratio: float = 0.97
    align_max_area_ratio: float = 1.03
    #: no edge may grow by more than this factor,
    align_max_edge_stretch: float = 2.0
    #: and the outline may not move further than this anywhere.
    align_max_hausdorff_m: float = 0.5


def _points_array(coords) -> np.ndarray:
    arr = np.asarray(coords, dtype=float)
    if len(arr) > 1 and np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]
    return arr


def _ring_edges(points: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = len(points)
    return [(points[i], points[(i + 1) % n]) for i in range(n)] if n > 1 else []


def _edge_params(p: np.ndarray, q: np.ndarray, cfg: EdgeAlignmentConfig):
    v = q - p
    length = float(np.linalg.norm(v))
    if length <= cfg.min_edge_len_m:
        direction = np.array([1.0, 0.0], dtype=float)
    else:
        direction = v / length
    normal = np.array([-direction[1], direction[0]], dtype=float)
    rho = float(np.dot(normal, p))
    alpha = math.atan2(normal[1], normal[0])
    if alpha < 0:
        alpha += math.pi
        normal = -normal
        rho = -rho
    return alpha, rho, length, normal, direction


def _angle_diff(a: float, b: float) -> float:
    diff = abs(a - b)
    return min(diff, math.pi - diff)


def _project_point_to_line(p: np.ndarray, normal: np.ndarray, rho: float) -> np.ndarray:
    return p - normal * (np.dot(normal, p) - rho)


def _intersect_lines(n1: np.ndarray, r1: float, n2: np.ndarray, r2: float):
    A = np.vstack([n1, n2])
    b = np.array([r1, r2], dtype=float)
    det = np.linalg.det(A)
    if abs(det) < 1e-12:
        return False, np.zeros(2)
    return True, np.linalg.solve(A, b)


def _cluster_edges(edges: List[Dict], cfg: EdgeAlignmentConfig) -> List[Dict]:
    angle_tol = math.radians(cfg.line_angle_tol_deg)
    clusters: List[Dict] = []

    def _rebuild_cluster(members: List[Dict]) -> Dict:
        weights = [max(float(member["length"]), cfg.min_edge_len_m) for member in members]
        total_weight = float(sum(weights))

        normal_sum = np.zeros(2, dtype=float)
        for member, weight in zip(members, weights):
            normal_sum += np.asarray(member["n"], dtype=float) * weight

        normal_norm = float(np.linalg.norm(normal_sum))
        if normal_norm <= 1e-12:
            normal = np.asarray(members[0]["n"], dtype=float)
        else:
            normal = normal_sum / normal_norm

        midpoints = [
            0.5 * (np.asarray(member["p"], dtype=float) + np.asarray(member["q"], dtype=float))
            for member in members
        ]
        rho = float(
            sum(float(np.dot(normal, midpoint)) * weight for midpoint, weight in zip(midpoints, weights))
            / total_weight
        )

        alpha = math.atan2(float(normal[1]), float(normal[0]))
        if alpha < 0:
            alpha += math.pi
            normal = -normal
            rho = -rho

        cluster = {
            "alpha": alpha,
            "rho": rho,
            "weight": total_weight,
            "members": members,
            "n": normal,
        }
        family_members = [member for member in members if member.get("family_idx") is not None]
        if family_members:
            by_family: Dict[int, float] = {}
            alpha_by_family: Dict[int, float] = {}
            for member in family_members:
                family_idx = int(member["family_idx"])
                weight = max(float(member["length"]), cfg.min_edge_len_m)
                by_family[family_idx] = by_family.get(family_idx, 0.0) + weight
                alpha_by_family[family_idx] = float(member.get("family_alpha", member["alpha"]))
            best_family_idx = max(by_family, key=by_family.get)
            cluster["family_idx"] = int(best_family_idx)
            cluster["family_alpha"] = float(alpha_by_family[best_family_idx])
        else:
            cluster["family_idx"] = None
            cluster["family_alpha"] = None
        return cluster

    for edge in edges:
        alpha, rho = edge["alpha"], edge["rho"]
        weight = max(edge["length"], cfg.min_edge_len_m)
        placed = False

        for cluster in clusters:
            if _angle_diff(alpha, cluster["alpha"]) <= angle_tol and abs(rho - cluster["rho"]) <= cfg.line_offset_tol_m:
                total_weight = cluster["weight"] + weight
                cluster["alpha"] = (cluster["alpha"] * cluster["weight"] + alpha * weight) / total_weight
                cluster["rho"] = (cluster["rho"] * cluster["weight"] + rho * weight) / total_weight
                cluster["weight"] = total_weight
                cluster["members"].append(edge)
                placed = True
                break

        if not placed:
            clusters.append({"alpha": alpha, "rho": rho, "weight": weight, "members": [edge]})

    clusters = [_rebuild_cluster(cluster["members"]) for cluster in clusters]

    def _cluster_total_length(cluster: Dict) -> float:
        return float(sum(float(member["length"]) for member in cluster["members"]))

    def _cluster_longest_edge(cluster: Dict) -> float:
        if not cluster["members"]:
            return 0.0
        return float(max(float(member["length"]) for member in cluster["members"]))

    def _is_small_cluster(cluster: Dict) -> bool:
        return (
            _cluster_total_length(cluster) < cfg.min_cluster_total_length_m
            or _cluster_longest_edge(cluster) < cfg.min_cluster_longest_edge_m
        )

    if len(clusters) <= 1:
        return clusters

    substantial_clusters = [cluster for cluster in clusters if not _is_small_cluster(cluster)]
    if not substantial_clusters:
        return clusters

    angle_reassign_tol = math.radians(cfg.small_cluster_reassign_angle_tol_deg)
    kept_clusters: List[Dict] = []

    for cluster in clusters:
        if not _is_small_cluster(cluster):
            kept_clusters.append(cluster)
            continue

        best_target = None
        best_cost = (float("inf"), float("inf"))
        for target in substantial_clusters:
            if target is cluster:
                continue
            angle_delta = _angle_diff(cluster["alpha"], target["alpha"])
            offset_delta = abs(cluster["rho"] - target["rho"])
            if angle_delta > angle_reassign_tol or offset_delta > cfg.small_cluster_reassign_offset_tol_m:
                continue
            cost = (angle_delta, offset_delta)
            if cost < best_cost:
                best_target = target
                best_cost = cost

        if best_target is None:
            kept_clusters.append(cluster)
            continue

        best_target["members"].extend(cluster["members"])

    rebuilt_clusters = []
    for cluster in kept_clusters:
        rebuilt_clusters.append(_rebuild_cluster(cluster["members"]))

    return rebuilt_clusters


def _weighted_mean_alpha(members: List[Dict], cfg: EdgeAlignmentConfig) -> float:
    weights = np.asarray([max(float(member["length"]), cfg.min_edge_len_m) for member in members], dtype=float)
    normals = np.asarray([np.asarray(member["n"], dtype=float) for member in members], dtype=float)
    mean_normal = (normals * weights[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(mean_normal))
    if norm <= 1e-12:
        return float(members[0]["alpha"])
    mean_normal /= norm
    alpha = float(math.atan2(mean_normal[1], mean_normal[0]))
    if alpha < 0:
        alpha += math.pi
    return alpha


def _cluster_angle_families(edges: List[Dict], cfg: EdgeAlignmentConfig) -> List[Dict]:
    if not edges:
        return []

    angle_tol = math.radians(cfg.family_angle_tol_deg)
    ordered = sorted(edges, key=lambda edge: float(edge["alpha"]))
    families: List[Dict] = []

    for edge in ordered:
        if not families:
            families.append({"members": [edge], "alpha": float(edge["alpha"])})
            continue

        current = families[-1]
        candidate_members = [*current["members"], edge]
        candidate_alpha = _weighted_mean_alpha(candidate_members, cfg)
        if all(_angle_diff(float(member["alpha"]), candidate_alpha) <= angle_tol for member in candidate_members):
            current["members"] = candidate_members
            current["alpha"] = candidate_alpha
        else:
            families.append({"members": [edge], "alpha": float(edge["alpha"])})

    if len(families) > 1:
        wrap_members = [*families[-1]["members"], *families[0]["members"]]
        wrap_alpha = _weighted_mean_alpha(wrap_members, cfg)
        if all(_angle_diff(float(member["alpha"]), wrap_alpha) <= angle_tol for member in wrap_members):
            merged = {"members": wrap_members, "alpha": wrap_alpha}
            families = [merged, *families[1:-1]]

    rebuilt = []
    for family_idx, family in enumerate(families):
        members = family["members"]
        alpha = _weighted_mean_alpha(members, cfg)
        rebuilt.append(
            {
                "family_idx": int(family_idx),
                "alpha": alpha,
                "members": members,
                "total_length_m": float(sum(float(member["length"]) for member in members)),
            }
        )
    return rebuilt


def _assign_edge_to_angle_family(edge: Dict, families: List[Dict], cfg: EdgeAlignmentConfig) -> Dict:
    if not families:
        return edge

    best_family = min(families, key=lambda family: _angle_diff(float(edge["alpha"]), float(family["alpha"])))
    alpha = float(best_family["alpha"])
    normal = np.array([math.cos(alpha), math.sin(alpha)], dtype=float)
    rho = float(np.dot(normal, 0.5 * (np.asarray(edge["p"], dtype=float) + np.asarray(edge["q"], dtype=float))))
    direction = np.array([-normal[1], normal[0]], dtype=float)
    adjusted = dict(edge)
    adjusted["alpha_raw"] = float(edge["alpha"])
    adjusted["family_alpha"] = alpha
    adjusted["family_idx"] = int(best_family["family_idx"])
    adjusted["alpha"] = alpha
    adjusted["n"] = normal
    adjusted["rho"] = rho
    adjusted["d"] = direction
    return adjusted


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    cosv = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return math.degrees(math.acos(cosv))


def _clean_ring(coords: np.ndarray, cfg: EdgeAlignmentConfig) -> np.ndarray:
    if len(coords) < 4:
        return coords

    points = [coords[0]]
    for p in coords[1:]:
        if np.linalg.norm(p - points[-1]) >= 1e-12:
            points.append(p)
    if np.linalg.norm(points[0] - points[-1]) > 1e-12:
        points.append(points[0])
    points = np.asarray(points)

    keep = []
    n = len(points) - 1
    for i in range(n):
        p_prev = points[(i - 1) % n]
        p_cur = points[i]
        p_next = points[(i + 1) % n]

        v1 = p_cur - p_prev
        v2 = p_next - p_cur

        if np.linalg.norm(v1) < cfg.min_seg_len_m or np.linalg.norm(v2) < cfg.min_seg_len_m:
            continue

        if abs(180.0 - _angle_between(v1, v2)) <= cfg.collinear_tol_deg:
            continue

        keep.append(p_cur)

    if len(keep) < 3:
        return points

    keep = np.asarray(keep)
    return np.vstack([keep, keep[0]])


def _snap_ring_to_clusters(points: np.ndarray, edge_models: List[Dict], cfg: EdgeAlignmentConfig) -> np.ndarray:
    n = len(points)
    if n == 0:
        return points
    if len(edge_models) != n:
        if len(edge_models) == 0:
            return np.vstack([points, points[0]])
        edge_models = (edge_models * (n // len(edge_models) + 1))[:n]

    new_points = np.empty_like(points)
    for i in range(n):
        e_prev = edge_models[(i - 1) % n]
        e_cur = edge_models[i]
        n1, r1 = e_prev["n"], e_prev["rho"]
        n2, r2 = e_cur["n"], e_cur["rho"]

        ok, intersection = _intersect_lines(n1, r1, n2, r2)
        if ok and np.isfinite(intersection).all() and np.linalg.norm(intersection - points[i]) <= cfg.max_vertex_snap_shift_m:
            new_points[i] = intersection
            continue

        projected = _project_point_to_line(points[i], n2, r2)
        if np.isfinite(projected).all() and np.linalg.norm(projected - points[i]) <= cfg.max_vertex_snap_shift_m:
            new_points[i] = projected
        else:
            new_points[i] = points[i]

    return np.vstack([new_points, new_points[0]])


def _map_edge_to_cluster(edge: Dict, clusters: List[Dict], cfg: EdgeAlignmentConfig) -> Dict:
    best = None
    best_cost = (float("inf"), float("inf"))
    angle_tol = math.radians(cfg.line_angle_tol_deg)
    alpha = edge["alpha"]
    rho = edge["rho"]

    for cluster in clusters:
        angle_delta = _angle_diff(alpha, cluster["alpha"])
        offset_delta = abs(rho - cluster["rho"])
        if angle_delta <= angle_tol and offset_delta <= cfg.line_offset_tol_m:
            cost = (angle_delta, offset_delta)
            if cost < best_cost:
                best = cluster
                best_cost = cost

    if best is None:
        best = {"n": np.asarray(edge["n"], dtype=float), "rho": float(edge["rho"])}

    return best


def _polygon_has_finite_coords(poly: Polygon) -> bool:
    if not isinstance(poly, Polygon) or poly.is_empty:
        return False
    try:
        ext = np.asarray(poly.exterior.coords, dtype=float)
    except Exception:
        return False
    if not np.isfinite(ext).all():
        return False
    for ring in poly.interiors:
        try:
            coords = np.asarray(ring.coords, dtype=float)
        except Exception:
            return False
        if not np.isfinite(coords).all():
            return False
    return True


def _polygon_max_edge_length(poly: Polygon) -> float:
    coords = np.asarray(poly.exterior.coords[:-1], dtype=float)
    if len(coords) < 2:
        return 0.0
    return float(np.linalg.norm(np.roll(coords, -1, axis=0) - coords, axis=1).max())


def _safe_hausdorff_distance(original: Polygon, adjusted: Polygon) -> float:
    if not _polygon_has_finite_coords(original) or not _polygon_has_finite_coords(adjusted):
        return float("inf")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in hausdorff_distance",
                category=RuntimeWarning,
            )
            value = float(original.hausdorff_distance(adjusted))
    except Exception:
        return float("inf")
    return value if np.isfinite(value) else float("inf")


def _alignment_shape_is_reasonable(original: Polygon, adjusted: Polygon, cfg: EdgeAlignmentConfig) -> bool:
    if not isinstance(adjusted, Polygon) or adjusted.is_empty:
        return False
    if original.area <= 1e-9 or adjusted.area <= 1e-9:
        return False

    area_ratio = adjusted.area / original.area
    if area_ratio < cfg.align_min_area_ratio or area_ratio > cfg.align_max_area_ratio:
        return False

    if _safe_hausdorff_distance(original, adjusted) > cfg.align_max_hausdorff_m:
        return False

    original_max_edge = _polygon_max_edge_length(original)
    adjusted_max_edge = _polygon_max_edge_length(adjusted)
    if original_max_edge > 1e-9 and adjusted_max_edge > original_max_edge * cfg.align_max_edge_stretch:
        return False

    return True


def _raw_edges_from_polygon(poly: Polygon, cfg: EdgeAlignmentConfig) -> List[Dict]:
    raw_edges: List[Dict] = []

    ext_points = _points_array(list(poly.exterior.coords))
    for p, q in _ring_edges(ext_points):
        alpha, rho, length, normal, direction = _edge_params(p, q, cfg)
        raw_edges.append({"p": p, "q": q, "alpha": alpha, "rho": rho, "length": length, "n": normal, "d": direction})

    for ring in poly.interiors:
        ring_points = _points_array(list(ring.coords))
        for p, q in _ring_edges(ring_points):
            alpha, rho, length, normal, direction = _edge_params(p, q, cfg)
            raw_edges.append({"p": p, "q": q, "alpha": alpha, "rho": rho, "length": length, "n": normal, "d": direction})

    return raw_edges


def _find_reference_cluster(
    raw_alpha: float,
    raw_rho: float,
    reference_clusters: List[Dict],
    cfg: EdgeAlignmentConfig,
    offset_tol_m: float,
) -> Dict | None:
    """Match a raw edge against reference clusters (per-edge, not per-cluster).

    Uses the family angle tolerance (typically 3°) rather than the tighter
    line-cluster tolerance (typically 2°), because cross-building edges carry
    more digitization noise than edges within a single polygon.
    """
    angle_tol = math.radians(cfg.family_angle_tol_deg)
    best: Dict | None = None
    best_cost = (float("inf"), float("inf"))
    for rc in reference_clusters:
        ad = _angle_diff(raw_alpha, rc["alpha"])
        od = abs(raw_rho - rc["rho"])
        if ad <= angle_tol and od <= offset_tol_m:
            cost = (ad, od)
            if cost < best_cost:
                best = rc
                best_cost = cost
    return best


def _map_ring_edges_with_reference(
    points: np.ndarray,
    angle_families: List[Dict],
    clusters: List[Dict],
    reference_clusters: List[Dict] | None,
    cfg: EdgeAlignmentConfig,
    ref_offset_tol_m: float,
) -> Tuple[List[Dict], int]:
    """Map each ring edge to its best line: reference cluster first, own cluster as fallback.

    Two-pass reference matching: first try the raw edge parameters, then try
    with the family-adjusted angle.  This catches edges where the own-polygon
    family assignment brings the angle closer to the reference direction.
    """
    mappings: List[Dict] = []
    n_ref = 0
    for p, q in _ring_edges(points):
        alpha, rho, length, normal, direction = _edge_params(p, q, cfg)

        if reference_clusters:
            ref_match = _find_reference_cluster(alpha, rho, reference_clusters, cfg, ref_offset_tol_m)
            if ref_match is not None:
                mappings.append(ref_match)
                n_ref += 1
                continue

            edge_model = {"p": p, "q": q, "alpha": alpha, "rho": rho, "length": length, "n": normal, "d": direction}
            if angle_families:
                family_model = _assign_edge_to_angle_family(edge_model, angle_families, cfg)
                ref_match = _find_reference_cluster(
                    family_model["alpha"], family_model["rho"],
                    reference_clusters, cfg, ref_offset_tol_m,
                )
                if ref_match is not None:
                    mappings.append(ref_match)
                    n_ref += 1
                    continue
                mappings.append(_map_edge_to_cluster(family_model, clusters, cfg))
            else:
                mappings.append(_map_edge_to_cluster(edge_model, clusters, cfg))
        else:
            edge_model = {"p": p, "q": q, "alpha": alpha, "rho": rho, "length": length, "n": normal, "d": direction}
            family_model = (
                _assign_edge_to_angle_family(edge_model, angle_families, cfg)
                if angle_families
                else edge_model
            )
            mappings.append(_map_edge_to_cluster(family_model, clusters, cfg))
    return mappings, n_ref


def align_polygon_edges_via_line_clustering(
    poly: Polygon,
    cfg: EdgeAlignmentConfig | None = None,
    *,
    reference_clusters: List[Dict] | None = None,
    reference_cluster_offset_tol_m: float = 0.5,
    return_diagnostics: bool = False,
) -> Polygon | tuple[Polygon, Dict]:
    if cfg is None:
        cfg = EdgeAlignmentConfig()
    if not isinstance(poly, Polygon) or poly.is_empty:
        return (poly, {"n_angle_families": 0, "angle_family_degrees": []}) if return_diagnostics else poly

    raw_edges = _raw_edges_from_polygon(poly, cfg)
    if not raw_edges:
        return (poly, {"n_angle_families": 0, "angle_family_degrees": []}) if return_diagnostics else poly

    else:
        angle_families = _cluster_angle_families(raw_edges, cfg) if cfg.use_angle_families else []

    diagnostics = {
        "n_angle_families": int(len(angle_families)) if angle_families else 0,
        "angle_family_degrees": [float(math.degrees(family["alpha"])) for family in angle_families],
    }

    edges = [
        _assign_edge_to_angle_family(edge, angle_families, cfg) if angle_families else edge
        for edge in raw_edges
    ]
    clusters = _cluster_edges(edges, cfg)

    ext_points = _points_array(list(poly.exterior.coords))
    ext_mappings, n_ref_ext = _map_ring_edges_with_reference(
        ext_points, angle_families, clusters, reference_clusters, cfg,
        reference_cluster_offset_tol_m,
    )
    snapped_ext = _snap_ring_to_clusters(ext_points, ext_mappings, cfg)
    cleaned_ext = _clean_ring(snapped_ext, cfg)
    if len(cleaned_ext) < 4:
        return (poly, diagnostics) if return_diagnostics else poly

    n_collinear_snapped = n_ref_ext

    new_interiors = []
    for ring in poly.interiors:
        ring_points = _points_array(list(ring.coords))
        if len(ring_points) < 3:
            continue
        hole_mappings, n_ref_hole = _map_ring_edges_with_reference(
            ring_points, angle_families, clusters, reference_clusters, cfg,
            reference_cluster_offset_tol_m,
        )
        snapped_hole = _snap_ring_to_clusters(ring_points, hole_mappings, cfg)
        cleaned_hole = _clean_ring(snapped_hole, cfg)
        n_collinear_snapped += n_ref_hole
        if len(cleaned_hole) < 4:
            continue
        try:
            new_interiors.append(LinearRing(cleaned_hole))
        except ValueError:
            continue

    diagnostics["n_collinear_snapped"] = n_collinear_snapped

    try:
        aligned = Polygon(cleaned_ext, holes=[list(r.coords) for r in new_interiors] if new_interiors else None)
    except Exception:
        return (poly, diagnostics) if return_diagnostics else poly

    if not aligned.is_valid:
        aligned = aligned.buffer(0)
    if not isinstance(aligned, Polygon) or aligned.is_empty:
        return (poly, diagnostics) if return_diagnostics else poly
    if not _alignment_shape_is_reasonable(poly, aligned, cfg):
        return (poly, diagnostics) if return_diagnostics else poly
    return (aligned, diagnostics) if return_diagnostics else aligned


def align_buildings_gdf(
    gdf: gpd.GeoDataFrame,
    *,
    geometry_col: str = "geometry",
    cfg: EdgeAlignmentConfig | None = None,
) -> gpd.GeoDataFrame:
    if cfg is None:
        cfg = EdgeAlignmentConfig()

    out = gdf.copy()
    before_vertices = []
    after_vertices = []
    area_deltas = []
    aligned_geoms = []
    n_angle_families = []
    angle_family_degrees = []

    for geom in out[geometry_col]:
        if isinstance(geom, Polygon) and not geom.is_empty:
            aligned, diagnostics = align_polygon_edges_via_line_clustering(geom, cfg, return_diagnostics=True)
        else:
            aligned = geom
            diagnostics = {"n_angle_families": 0, "angle_family_degrees": []}

        aligned_geoms.append(aligned)
        n_angle_families.append(int(diagnostics["n_angle_families"]))
        angle_family_degrees.append(",".join(f"{deg:.2f}" for deg in diagnostics["angle_family_degrees"]))
        before_vertices.append(count_exterior_vertices(geom) if isinstance(geom, Polygon) else 0)
        after_vertices.append(count_exterior_vertices(aligned) if isinstance(aligned, Polygon) else 0)
        try:
            area_deltas.append(float(aligned.area - geom.area))
        except Exception:
            area_deltas.append(0.0)

    out[geometry_col] = aligned_geoms
    out["n_vertices_before_align"] = before_vertices
    out["n_vertices_after_align"] = after_vertices
    out["vertex_delta_align"] = np.asarray(after_vertices, dtype=int) - np.asarray(before_vertices, dtype=int)
    out["area_delta_align_m2"] = np.asarray(area_deltas, dtype=float)
    out["n_angle_families_align"] = np.asarray(n_angle_families, dtype=int)
    out["angle_family_degrees_align"] = angle_family_degrees
    out["alignment_changed"] = (
        (out["vertex_delta_align"] != 0) | (out["area_delta_align_m2"].abs() > 1e-9)
    )
    return out


def get_line_clusters_for_polygon(
    poly: Polygon,
    cfg: EdgeAlignmentConfig | None = None,
) -> List[Dict]:
    """Extract the line clusters (angle + offset) from a polygon's edges.

    Each returned dict has keys ``alpha``, ``rho``, ``n`` (unit normal), and
    ``members``.  The list is suitable for passing as ``reference_clusters``
    to :func:`align_polygon_edges_via_line_clustering` so that a neighbour's
    edges can be snapped to the same wall lines (colinearity, not just
    parallelism).
    """
    if cfg is None:
        cfg = EdgeAlignmentConfig()
    if not isinstance(poly, Polygon) or poly.is_empty:
        return []
    raw_edges = _raw_edges_from_polygon(poly, cfg)
    if not raw_edges:
        return []
    angle_families = _cluster_angle_families(raw_edges, cfg) if cfg.use_angle_families else []
    edges = [
        _assign_edge_to_angle_family(edge, angle_families, cfg) if angle_families else edge
        for edge in raw_edges
    ]
    return _cluster_edges(edges, cfg)


def align_neighbors_gdf(
    neighbors: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    *,
    geometry_col: str = "geometry",
    building_geometry_col: str = "geom_local_preray",
    sample_id_col: str = "sample_id",
    cfg: EdgeAlignmentConfig | None = None,
    proximity_threshold_m: float = 2.0,
    collinear_offset_tol_m: float = 0.5,
) -> gpd.GeoDataFrame:
    """Align neighbour edges using the main building's wall lines.

    For each neighbour within *proximity_threshold_m* of the main building,
    individual edges that match a main-building wall in both angle and offset
    are snapped to the exact reference line (per-edge, not global rotation).
    Edges that don't face the main building keep their own alignment.

    The function produces the same diagnostic columns as
    :func:`align_buildings_gdf`, plus ``ref_aligned`` (bool).
    """
    if cfg is None:
        cfg = EdgeAlignmentConfig()

    bld_clusters: Dict[str, List[Dict]] = {}
    bld_geom: Dict[str, Polygon] = {}

    for _, row in buildings.iterrows():
        sid = str(row[sample_id_col])
        geom = row.get(building_geometry_col)
        if isinstance(geom, Polygon) and not geom.is_empty:
            bld_clusters[sid] = get_line_clusters_for_polygon(geom, cfg)
            bld_geom[sid] = geom

    out = neighbors.copy()
    before_vertices = []
    after_vertices = []
    area_deltas = []
    aligned_geoms = []
    n_angle_families = []
    angle_family_degrees = []
    ref_aligned_flags = []

    for _, row in out.iterrows():
        geom = row[geometry_col]
        sid = str(row.get(sample_id_col, ""))

        ref_cls: List[Dict] | None = None
        main_g: Polygon | None = None
        if sid in bld_clusters and sid in bld_geom:
            main_g = bld_geom[sid]
            if (
                geom is not None
                and not getattr(geom, "is_empty", True)
                and isinstance(geom, Polygon)
            ):
                try:
                    dist = geom.distance(main_g)
                except Exception:
                    dist = float("inf")
                if dist <= proximity_threshold_m:
                    ref_cls = bld_clusters[sid]

        if isinstance(geom, Polygon) and not geom.is_empty:
            aligned, diagnostics = align_polygon_edges_via_line_clustering(
                geom, cfg,
                reference_clusters=ref_cls,
                reference_cluster_offset_tol_m=collinear_offset_tol_m,
                return_diagnostics=True,
            )
        else:
            aligned = geom
            diagnostics = {"n_angle_families": 0, "angle_family_degrees": []}

        aligned_geoms.append(aligned)
        ref_aligned_flags.append(ref_cls is not None)
        n_angle_families.append(int(diagnostics["n_angle_families"]))
        angle_family_degrees.append(
            ",".join(f"{deg:.2f}" for deg in diagnostics["angle_family_degrees"])
        )
        before_vertices.append(
            count_exterior_vertices(geom) if isinstance(geom, Polygon) else 0
        )
        after_vertices.append(
            count_exterior_vertices(aligned) if isinstance(aligned, Polygon) else 0
        )
        try:
            area_deltas.append(float(aligned.area - geom.area))
        except Exception:
            area_deltas.append(0.0)

    out[geometry_col] = aligned_geoms
    out["n_vertices_before_align"] = before_vertices
    out["n_vertices_after_align"] = after_vertices
    out["vertex_delta_align"] = (
        np.asarray(after_vertices, dtype=int) - np.asarray(before_vertices, dtype=int)
    )
    out["area_delta_align_m2"] = np.asarray(area_deltas, dtype=float)
    out["n_angle_families_align"] = np.asarray(n_angle_families, dtype=int)
    out["angle_family_degrees_align"] = angle_family_degrees
    out["alignment_changed"] = (
        (out["vertex_delta_align"] != 0) | (out["area_delta_align_m2"].abs() > 1e-9)
    )
    out["ref_aligned"] = ref_aligned_flags
    return out
