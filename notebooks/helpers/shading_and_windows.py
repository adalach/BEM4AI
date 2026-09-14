"""Work out what shades each building and where its windows can go.

Both questions are about the same thing: which parts of a building's facade see
open sky and which see the wall of the building next door. The two answers are
built from opposite ends. Shading starts from the neighbours and keeps the
surfaces that face the building; windows start from the building's own walls
and keep the stretches that are far enough from anything.

The neighbours are first merged into one polygon per sample. That matters more
than it sounds: OSM maps a terrace as separate footprints that touch, and
testing rays against each one separately reports a gap wherever two of them
meet. One merged occluder has no such seams.

Import from a notebook as::

    from helpers.shading_and_windows import extract_shading_edges

"""

from __future__ import annotations

import math
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    Point,
    Polygon,
)
from shapely.ops import nearest_points, unary_union
from shapely.prepared import prep
from tqdm import tqdm

from .utils import (
    empty_gdf,
    exterior_edges,
    iter_cleaned_edges,
    keep_polygons_min_area,
    polygon_parts,
    sample_points_along_linestring,
    sample_points_on_polygon_exterior,
    trim_segment,
)

__all__ = [
    "WindowLimits",
    "build_neighbourhood_unions",
    "build_rays",
    "extract_shading_edges",
    "facing_score",
    "find_window_candidates",
    "has_front_clear",
    "keep_samples",
    "prune_blocked_edges",
    "segment_outward_unit",
    "select_final_windows",
    "visible_subsegments",
]


def keep_samples(
    buildings_m: gpd.GeoDataFrame,
    neighbors_m: gpd.GeoDataFrame,
    keep_ids: set,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, int]:
    """Restrict both frames to ``keep_ids`` and report how many samples went.

    Every stage that can reject a building calls this, so a sample never
    survives in one frame after being dropped from the other.
    """
    all_ids = set(buildings_m["sample_id"].dropna().unique())
    n_dropped = len(all_ids - set(keep_ids))
    if n_dropped == 0:
        return buildings_m, neighbors_m, 0
    return (
        buildings_m[buildings_m["sample_id"].isin(keep_ids)].copy(),
        neighbors_m[neighbors_m["sample_id"].isin(keep_ids)].copy(),
        n_dropped,
    )


# ---------------------------------------------------------------------------
# Neighbourhood
# ---------------------------------------------------------------------------


def build_neighbourhood_unions(
    buildings_m: gpd.GeoDataFrame,
    neighbors_m: gpd.GeoDataFrame,
    *,
    buffer_m: float,
    min_area_m2: float,
    geom_col: str = "geom_local",
) -> tuple[dict, dict]:
    """Merge each sample's neighbours into one occluder polygon.

    Expanding by ``buffer_m`` before the union and shrinking by the same
    afterwards closes the hairline gaps between footprints that are drawn as
    touching but do not quite meet, without growing the result.

    Returns the polygon per sample_id and a dict of counts for the notebook.
    """
    neighborhood_map: dict = {}
    groups = dict(tuple(neighbors_m.groupby("sample_id", sort=False)))

    stats = {"total_rows": 0, "with_neighbors": 0, "before_parts": 0, "after_parts": 0, "reduced": 0}

    for sid in buildings_m["sample_id"].unique():
        neigh = groups.get(sid)
        if neigh is None or neigh.empty:
            neighborhood_map[sid] = None
            continue

        stats["with_neighbors"] += 1
        stats["total_rows"] += len(neigh)

        polys = []
        for geom in neigh[geom_col].values:
            polys.extend(polygon_parts(geom))
        before = len(polys)
        stats["before_parts"] += before
        if not polys:
            neighborhood_map[sid] = None
            continue

        buffered = [p.buffer(buffer_m, join_style=2) for p in polys]
        base = unary_union(buffered if buffered else polys)
        shrunk = base.buffer(-buffer_m, join_style=2)
        cleaned = keep_polygons_min_area(shrunk, min_area_m2)
        neighborhood_map[sid] = cleaned

        after = 0 if cleaned is None else (1 if isinstance(cleaned, Polygon) else len(cleaned.geoms))
        stats["after_parts"] += after
        if before > 0 and after < before:
            stats["reduced"] += 1

    return neighborhood_map, stats


# ---------------------------------------------------------------------------
# Shading edges
# ---------------------------------------------------------------------------


def facing_score(edge: LineString, main_centroid, is_ccw: bool) -> float:
    """How squarely an edge's outward normal points at the main building.

    ``1`` means the edge faces the building head-on, ``0`` that it is edge-on,
    negative that it faces away. Which of the two normals is outward depends on
    the ring's winding, which is why ``is_ccw`` has to be passed in; a hole
    winds the other way from the shell that contains it.
    """
    (x1, y1), (x2, y2) = np.asarray(edge.coords, dtype=float)
    ex, ey = x2 - x1, y2 - y1
    edge_len = float(np.hypot(ex, ey))
    if edge_len == 0.0:
        return 0.0
    nx, ny = (ey / edge_len, -ex / edge_len) if is_ccw else (-ey / edge_len, ex / edge_len)
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    vx, vy = float(main_centroid.x) - mx, float(main_centroid.y) - my
    vlen = float(np.hypot(vx, vy))
    if vlen == 0.0:
        return 0.0
    vx, vy = vx / vlen, vy / vlen
    return float(nx * vx + ny * vy)


def extract_shading_edges(
    buildings_m: gpd.GeoDataFrame,
    *,
    max_gap_m: float,
    geom_col: str = "geom_local",
    progress: bool = True,
) -> gpd.GeoDataFrame:
    """The neighbourhood edges that could shade each building.

    An edge is a candidate if it faces the building, or if it is within
    ``max_gap_m`` of it. The distance rule catches the wall a building shares
    with its neighbour, which is edge-on and therefore scores zero.

    Both the shell and the holes of every union part are walked, since a
    building inside a courtyard is shaded by the courtyard's inner wall.
    """
    records = []
    groups = buildings_m.groupby("sample_id", sort=False)
    if progress:
        groups = tqdm(groups, total=buildings_m["sample_id"].nunique(), desc="Extract neighborhood edges")

    for sid, brow in groups:
        main = brow[geom_col].iloc[0]
        if main is None or getattr(main, "is_empty", False):
            continue
        main_centroid = main.centroid
        main_boundary = main.boundary

        neigh = brow["neighborhood"].iloc[0]
        if neigh is None or getattr(neigh, "is_empty", False):
            continue

        for up_idx, up in enumerate(polygon_parts(neigh)):
            try:
                is_ccw = up.exterior.is_ccw
            except Exception:  # noqa: BLE001 - a degenerate ring defaults to CCW
                is_ccw = True

            rings = [(up.exterior, is_ccw)]
            for hole in up.interiors:
                rings.append((hole, not is_ccw))

            eid = 0
            for ring, ring_ccw in rings:
                ring_poly = Polygon(ring)
                for edge in iter_cleaned_edges(ring_poly, min_edge_len_m=0.05, angle_tol_deg=1.0):
                    score = facing_score(edge, main_centroid, ring_ccw)
                    dist_to_main = edge.distance(main_boundary)
                    if (score > 0.0) or (dist_to_main <= max_gap_m):
                        records.append(
                            {
                                "sample_id": sid,
                                "neighbor_id": f"union_{sid}_{up_idx}",
                                "edge_id": eid,
                                "facing_score": float(score),
                                "dist_to_main": float(dist_to_main),
                                "geometry": edge,
                            }
                        )
                    eid += 1

    if records:
        return gpd.GeoDataFrame(records, geometry="geometry", crs=buildings_m.crs)
    return empty_gdf(
        ["sample_id", "neighbor_id", "edge_id", "facing_score", "dist_to_main"],
        crs=buildings_m.crs,
    )


def build_rays(
    buildings_m: gpd.GeoDataFrame,
    shades_gdf: gpd.GeoDataFrame,
    *,
    interval_m: float,
    geom_col: str = "geom_local",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Connect points along each shading edge to the nearest point on the building.

    The rays are what the next step tests for obstruction. They target the
    shrunken footprint where one exists, so a ray aimed at a wall does not count
    that same wall as the thing blocking it.

    Returns the neighbour sample points, the building sample points, and the rays.
    """
    nei_pts_records, main_pts_records, rays_records = [], [], []
    edges_by_sample = dict(tuple(shades_gdf.groupby("sample_id", sort=False))) if not shades_gdf.empty else {}

    for sid, brow in buildings_m.groupby("sample_id", sort=False):
        main = brow[geom_col].iloc[0]
        shr = brow["geom_shrunk"].iloc[0] if "geom_shrunk" in brow.columns else None
        tgt = shr if (shr is not None and not getattr(shr, "is_empty", False)) else main

        for i, p in enumerate(sample_points_on_polygon_exterior(tgt, interval_m)):
            main_pts_records.append(
                {"sample_id": sid, "pt_id": f"main_{sid}_{i}", "geometry": Point(p.x, p.y)}
            )

        esub = edges_by_sample.get(sid)
        if esub is None:
            continue
        for idx, erow in esub.iterrows():
            for sp in sample_points_along_linestring(erow.geometry, interval_m):
                sp_pt = Point(sp.x, sp.y)
                nei_pts_records.append(
                    {"sample_id": sid, "edge_index": int(idx), "edge_id": erow["edge_id"], "geometry": sp_pt}
                )
                try:
                    _, on_main = nearest_points(sp_pt, tgt)
                except Exception:  # noqa: BLE001 - degenerate target, aim at its centre
                    on_main = tgt.centroid
                rays_records.append(
                    {
                        "sample_id": sid,
                        "edge_index": int(idx),
                        "edge_id": erow["edge_id"],
                        "geometry": LineString([(sp_pt.x, sp_pt.y), (on_main.x, on_main.y)]),
                    }
                )

    crs = buildings_m.crs
    return (
        gpd.GeoDataFrame(nei_pts_records, geometry="geometry", crs=crs)
        if nei_pts_records
        else empty_gdf(["sample_id", "edge_index", "edge_id"], crs=crs),
        gpd.GeoDataFrame(main_pts_records, geometry="geometry", crs=crs)
        if main_pts_records
        else empty_gdf(["sample_id", "pt_id"], crs=crs),
        gpd.GeoDataFrame(rays_records, geometry="geometry", crs=crs)
        if rays_records
        else empty_gdf(["sample_id", "edge_index", "edge_id"], crs=crs),
    )


def prune_blocked_edges(
    buildings_m: gpd.GeoDataFrame,
    shades_gdf: gpd.GeoDataFrame,
    rays_gdf: gpd.GeoDataFrame,
    *,
    endpoint_tol: float,
    mask_eps: float,
) -> tuple[gpd.GeoDataFrame, int]:
    """Drop shading edges that another neighbour hides completely.

    An edge survives if any one of its rays reaches the building through open
    space. Rays are trimmed at both ends before the test, because a ray starts
    on the occluder's own surface and ends on the building's, and untrimmed it
    would always register as blocked.

    Returns the surviving edges and how many were removed.
    """
    mask_by_sample = dict(zip(buildings_m["sample_id"], buildings_m["neighborhood"]))
    rays_by_edge = {int(k): g for k, g in rays_gdf.groupby("edge_index")} if not rays_gdf.empty else {}
    kept_idx = []

    for sid, edge_sub in shades_gdf.groupby("sample_id"):
        mask = mask_by_sample.get(sid)
        if mask is None or getattr(mask, "is_empty", False):
            kept_idx.extend(list(edge_sub.index))
            continue

        try:
            mask_geom = mask.buffer(mask_eps, join_style=2)
            if mask_geom.is_empty:
                mask_geom = mask
        except Exception:  # noqa: BLE001 - buffering failed, test the raw mask
            mask_geom = mask
        mask_prep = prep(mask_geom)

        for idx in edge_sub.index:
            rays_sub = rays_by_edge.get(int(idx))
            if (rays_sub is None) or rays_sub.empty:
                kept_idx.append(idx)
                continue

            for ray in rays_sub.geometry:
                if ray is None or getattr(ray, "is_empty", False):
                    continue
                ray_trim = trim_segment(ray, endpoint_tol)
                if ray_trim is None or getattr(ray_trim, "is_empty", False):
                    kept_idx.append(idx)
                    break
                if not mask_prep.intersects(ray_trim):
                    kept_idx.append(idx)
                    break

    kept = shades_gdf.loc[[i for i in kept_idx if i in shades_gdf.index]].copy()
    return kept, len(shades_gdf) - len(kept)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowLimits:
    """When a stretch of wall is open enough to carry a window.

    ``min_clearance_m`` is the space required to the side, ``min_front_m`` the
    space required straight ahead. Both are needed: a wall facing a courtyard
    two metres wide has clearance but no frontage.
    """

    min_wall_len_m: float = 1.0
    min_clearance_m: float = 2.0
    clearance_eps_m: float = 0.01
    touch_eps_m: float = 0.05
    trim_ends_m: float = 0.20
    min_visible_len_m: float = 1.0
    min_visible_frac: float = 0.30
    min_front_m: float = 4.0


def visible_subsegments(seg_trim, clearance_buffer, min_visible_len: float, min_visible_frac: float):
    """The stretches of a wall segment that lie outside the clearance buffer.

    A long wall is rarely all blocked or all open. Subtracting the buffer leaves
    the open runs; runs too short to hold a window are discarded, either in
    absolute terms or as a fraction of the wall.
    """
    if seg_trim is None or getattr(seg_trim, "is_empty", False):
        return []
    if clearance_buffer is None or getattr(clearance_buffer, "is_empty", False):
        return [seg_trim]

    raw = seg_trim.difference(clearance_buffer)
    if raw is None or getattr(raw, "is_empty", False):
        return []

    if isinstance(raw, LineString):
        parts = [raw]
    elif isinstance(raw, (MultiLineString, GeometryCollection)):
        parts = [g for g in raw.geoms if isinstance(g, LineString) and not g.is_empty]
    else:
        parts = []
    if not parts:
        return []

    trimmed_len = float(seg_trim.length)
    visible = []
    for ls in parts:
        length = float(ls.length)
        if (length >= min_visible_len) or (trimmed_len > 0 and (length / trimmed_len) >= min_visible_frac):
            visible.append(ls)
    return visible


def segment_outward_unit(seg, main_centroid):
    """The unit normal of a wall segment that points away from the building."""
    if seg is None or getattr(seg, "is_empty", False):
        return None
    (x1, y1), (x2, y2) = seg.coords[0], seg.coords[-1]
    ex, ey = x2 - x1, y2 - y1
    el = math.hypot(ex, ey)
    if el == 0:
        return None
    tx, ty = ex / el, ey / el
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    vx, vy = main_centroid.x - mx, main_centroid.y - my
    # Of the two normals, keep the one pointing against the centroid.
    nx1, ny1 = ty, -tx
    nx2, ny2 = -ty, tx
    return (nx1, ny1) if (nx1 * vx + ny1 * vy) < 0 else (nx2, ny2)


def has_front_clear(seg, mask_prep, neigh_mask, main_centroid, front_clear_m: float) -> bool:
    """Whether a segment has open space straight out in front of it."""
    if seg is None or getattr(seg, "is_empty", False):
        return False
    if neigh_mask is None or getattr(neigh_mask, "is_empty", False):
        return True
    if main_centroid is None or getattr(main_centroid, "is_empty", False):
        return True
    if mask_prep is None:
        return True

    nu = segment_outward_unit(seg, main_centroid)
    if nu is None:
        return False
    ux, uy = nu
    mid = seg.interpolate(0.5, normalized=True)
    ray = LineString([(mid.x, mid.y), (mid.x + front_clear_m * ux, mid.y + front_clear_m * uy)])
    return not mask_prep.intersects(ray)


def find_window_candidates(
    buildings_m: gpd.GeoDataFrame,
    *,
    limits: WindowLimits,
    mask_eps: float,
    geom_col: str = "geom_local",
) -> gpd.GeoDataFrame:
    """Every stretch of exterior wall that could carry a window.

    Each wall is tested in one of four ways, recorded in ``touch_type``:
    ``interior_clear`` when the whole wall is far enough from everything,
    ``corner_only`` when it merely meets a neighbour at a corner,
    ``partial_clear`` when only part of it is open and that part is kept,
    ``mid_clear`` for a wall too short to trim, judged at its midpoint. A wall
    with no neighbours at all is ``none``. Every candidate must also pass the
    frontage check.
    """
    records = []
    for sid, brow in buildings_m.groupby("sample_id"):
        main = brow[geom_col].iloc[0]
        if main is None or getattr(main, "is_empty", False):
            continue
        main_centroid = main.centroid
        neigh = brow["neighborhood"].iloc[0] if "neighborhood" in brow.columns else None

        if neigh is not None and not getattr(neigh, "is_empty", False):
            try:
                neigh_mask = neigh.buffer(mask_eps, join_style=shapely.JOIN_STYLE.mitre)
                if neigh_mask.is_empty:
                    neigh_mask = neigh
            except Exception:  # noqa: BLE001 - buffering failed, use the raw union
                neigh_mask = neigh
            mask_prep = prep(neigh_mask)
            try:
                clearance_buffer = neigh_mask.buffer(limits.min_clearance_m, join_style=shapely.JOIN_STYLE.mitre)
            except Exception:  # noqa: BLE001 - fall back to no clearance margin
                clearance_buffer = neigh_mask
        else:
            neigh_mask = None
            mask_prep = None
            clearance_buffer = None

        clear_threshold = limits.min_clearance_m - limits.clearance_eps_m

        for seg_id, seg in enumerate(exterior_edges(main)):
            seg_len = float(seg.length)
            if seg_len < limits.min_wall_len_m:
                continue

            seg_trim = trim_segment(seg, limits.trim_ends_m)

            if seg_trim is None or getattr(seg_trim, "is_empty", False):
                if neigh_mask is None:
                    records.append(_window_record(sid, seg_id, seg_len, float("inf"), "none", seg))
                    continue
                d_mid = seg.interpolate(0.5, normalized=True).distance(neigh_mask)
                if d_mid >= clear_threshold and has_front_clear(
                    seg, mask_prep, neigh_mask, main_centroid, limits.min_front_m
                ):
                    records.append(_window_record(sid, seg_id, seg_len, d_mid, "mid_clear", seg))
                continue

            if neigh_mask is None:
                records.append(_window_record(sid, seg_id, seg_len, float("inf"), "none", seg))
                continue

            d_int = seg_trim.distance(neigh_mask)
            interior_touches = mask_prep.intersects(seg_trim) or (d_int <= limits.touch_eps_m)
            p0, p1 = Point(*seg.coords[0]), Point(*seg.coords[-1])
            endpoint_touches = (p0.distance(neigh_mask) <= limits.touch_eps_m) or (
                p1.distance(neigh_mask) <= limits.touch_eps_m
            )

            if (not interior_touches) and endpoint_touches:
                if has_front_clear(seg_trim, mask_prep, neigh_mask, main_centroid, limits.min_front_m):
                    records.append(_window_record(sid, seg_id, seg_len, d_int, "corner_only", seg))
                continue

            if (not interior_touches) and (d_int >= clear_threshold):
                if has_front_clear(seg_trim, mask_prep, neigh_mask, main_centroid, limits.min_front_m):
                    records.append(_window_record(sid, seg_id, seg_len, d_int, "interior_clear", seg))
                continue

            for vp in visible_subsegments(
                seg_trim, clearance_buffer, limits.min_visible_len_m, limits.min_visible_frac
            ):
                d_vp = float(vp.distance(neigh_mask))
                if d_vp >= clear_threshold and has_front_clear(
                    vp, mask_prep, neigh_mask, main_centroid, limits.min_front_m
                ):
                    records.append(
                        _window_record(sid, seg_id, float(vp.length), d_vp, "partial_clear", vp)
                    )

    if not records:
        return empty_gdf(
            ["sample_id", "seg_id", "length_m", "min_dist_m", "touch_type"], crs=buildings_m.crs
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=buildings_m.crs)


def _window_record(sid, seg_id, length_m, min_dist_m, touch_type, geometry) -> dict:
    return {
        "sample_id": sid,
        "seg_id": seg_id,
        "length_m": length_m,
        "min_dist_m": min_dist_m,
        "touch_type": touch_type,
        "geometry": geometry,
    }


def select_final_windows(
    buildings_m: gpd.GeoDataFrame,
    window_walls_gdf: gpd.GeoDataFrame,
    *,
    long_min_m: float,
    fraction_cap: float,
    geom_col: str = "geom_local",
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Narrow the candidates down to the windows a building actually gets.

    Without a cap a free-standing building would get a window on every wall,
    which is not how buildings are built. The cap is a fraction of the
    building's own long walls, so it scales with the footprint instead of being
    a fixed number, and the longest candidates win the places available.

    Returns the selected windows and a per-sample summary of the arithmetic.
    """
    final_win_records, summary_rows = [], []
    candidates_by_sample = (
        dict(tuple(window_walls_gdf.groupby("sample_id", sort=False))) if not window_walls_gdf.empty else {}
    )

    for sid, brow in buildings_m.groupby("sample_id"):
        main = brow[geom_col].iloc[0]
        bld_edges_len = [float(e.length) for e in exterior_edges(main)]
        n_bld_all = len(bld_edges_len)
        n_bld_long = sum(1 for length in bld_edges_len if length >= long_min_m)

        cand_sub = candidates_by_sample.get(sid)
        n_cand_all = 0 if cand_sub is None else len(cand_sub)
        cand_long = (
            []
            if cand_sub is None
            else [(g, float(g.length)) for g in cand_sub.geometry if float(g.length) >= long_min_m]
        )
        n_cand_long = len(cand_long)

        cap = int(math.floor(max(0.0, min(1.0, float(fraction_cap))) * n_bld_long))
        cand_long.sort(key=lambda t: t[1], reverse=True)
        kept_long = cand_long[:cap] if n_cand_long > cap else cand_long

        for geom, length in kept_long:
            final_win_records.append({"sample_id": sid, "length_m": length, "geometry": geom})

        summary_rows.append(
            {
                "sample_id": sid,
                "bld_edges_all": n_bld_all,
                "bld_edges_long": n_bld_long,
                "cand_edges_all": n_cand_all,
                "cand_edges_long_initial": n_cand_long,
                "cand_edges_long_kept": len(kept_long),
                "cap_fraction": fraction_cap,
                "cap_long_edges": cap,
            }
        )

    if final_win_records:
        final_windows_gdf = gpd.GeoDataFrame(final_win_records, geometry="geometry", crs=buildings_m.crs)
    else:
        final_windows_gdf = empty_gdf(["sample_id", "length_m"], crs=buildings_m.crs)
    summary_df = pd.DataFrame(summary_rows).sort_values("sample_id")
    return final_windows_gdf, summary_df
