"""Put every sampled building into its own local metric frame.

Stage 4 measures everything in metres: wall lengths, clearances, ray distances,
zone widths. Degrees will not do, and one Europe-wide projection distorts too
much between Lisbon and Helsinki. So each city is reprojected to its own UTM
zone, and each building is then translated so that the centre of its bounding
box sits at the origin. Every later stage works in that local frame, where a
coordinate is a metre and a building sits within a few tens of metres of (0, 0).

The frames are built first, then the footprints are simplified inside them:
normalization collapses the surveying noise OSM footprints carry, edge
alignment squares up walls that are almost parallel, and a shrink-expand pass
erases notches too thin to matter. A neighbour that still overlaps its main
building afterwards has its shared wall projected onto the main building's wall
line, which is what makes shared walls in a terrace actually coincide.

Import from a notebook as::

    from helpers.local_frames import build_local_coordinate_frames

"""

from __future__ import annotations

import math

import geopandas as gpd
import pandas as pd
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from tqdm import tqdm

from scripts.ConvexDecomp.osm_footprint_normalizer import (
    FootprintNormalizationConfig,
    LinearArtifactFilterConfig,
    annotate_linear_artifact_flags,
    normalize_buildings_gdf,
)
from scripts.ConvexDecomp.polygon_edge_alignment import (
    EdgeAlignmentConfig,
    align_buildings_gdf,
    align_neighbors_gdf,
)
from .utils import (
    count_exterior_vertices,
    frame_with_object_geometry_columns,
    make_geometry_valid,
    metric_crs_str_for_lonlat,
    polygon_parts,
)

__all__ = [
    "align_neighbour_walls",
    "build_local_coordinate_frames",
    "city_lookup_from_assets",
    "enforce_wall_colinearity",
    "metric_crs_for_city_key",
    "preprocess_local_geometries",
    "remove_residual_overlap",
    "sync_geometry",
]


def city_lookup_from_assets(city_assets_df: pd.DataFrame) -> pd.DataFrame:
    """One row per city_key with its longitude and latitude, indexed by city_key."""
    return (
        city_assets_df[["city_key", "lon", "lat"]]
        .dropna(subset=["city_key"])
        .drop_duplicates(subset=["city_key"])
        .assign(city_key=lambda df: df["city_key"].astype(str))
        .set_index("city_key")
    )


def metric_crs_for_city_key(
    city_key: str,
    city_lookup_df: pd.DataFrame,
    fallback_geom: BaseGeometry | None = None,
) -> str:
    """The UTM CRS for one city, from its registered centre or its own geometry."""
    city_key = str(city_key)
    if city_key in city_lookup_df.index:
        row = city_lookup_df.loc[city_key]
        if pd.notna(row.get("lon")) and pd.notna(row.get("lat")):
            return metric_crs_str_for_lonlat(float(row["lon"]), float(row["lat"]))
    if fallback_geom is not None and not getattr(fallback_geom, "is_empty", True):
        pt = fallback_geom.representative_point()
        return metric_crs_str_for_lonlat(float(pt.x), float(pt.y))
    raise KeyError(f"Could not determine a metric CRS for city_key={city_key!r}.")


def build_local_coordinate_frames(
    buildings_4326: gpd.GeoDataFrame,
    neighbors_4326: gpd.GeoDataFrame,
    city_lookup_df: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, str]]:
    """Reproject city by city, then translate each sample to its own origin.

    Neighbours are translated by their main building's offset, not their own,
    so a sample and its neighbours stay in one frame. The WGS84 and city-metric
    geometries are kept alongside the local one, which is what lets stage 5 put
    a finished model back on the map.

    Returns the buildings, the neighbours, and the CRS chosen per city.
    """
    buildings_work = buildings_4326.copy()
    neighbors_work = neighbors_4326.copy()

    buildings_work["geometry_wgs84"] = pd.Series(
        list(buildings_work.geometry.values), index=buildings_work.index, dtype="object"
    )
    neighbors_work["geometry_wgs84"] = pd.Series(
        list(neighbors_work.geometry.values), index=neighbors_work.index, dtype="object"
    )

    sample_to_city = (
        buildings_work[["sample_id", "city_key"]]
        .drop_duplicates(subset=["sample_id"])
        .set_index("sample_id")["city_key"]
        .astype(str)
        .to_dict()
    )
    neighbors_work["city_key"] = neighbors_work["sample_id"].map(sample_to_city)
    neighbors_work = neighbors_work[neighbors_work["city_key"].notna()].copy()

    building_chunks = []
    neighbor_chunks = []
    metric_crs_map: dict[str, str] = {}

    city_groups = list(buildings_work.groupby("city_key", sort=False))
    for city_key, buildings_city in tqdm(city_groups, desc="Project cities to local metric frames"):
        raw_parts = [
            geom for geom in buildings_city.geometry if geom is not None and not getattr(geom, "is_empty", True)
        ]
        fallback_geom = unary_union(raw_parts) if raw_parts else None
        city_crs = metric_crs_for_city_key(city_key, city_lookup_df, fallback_geom=fallback_geom)
        metric_crs_map[str(city_key)] = city_crs

        buildings_city = buildings_city.to_crs(city_crs).copy()
        bounds = buildings_city.geometry.bounds
        buildings_city["bbox_cx_m"] = (bounds["minx"] + bounds["maxx"]) / 2.0
        buildings_city["bbox_cy_m"] = (bounds["miny"] + bounds["maxy"]) / 2.0
        buildings_city["metric_crs"] = city_crs
        buildings_city["geometry_metric"] = pd.Series(
            list(buildings_city.geometry.values), index=buildings_city.index, dtype="object"
        )
        buildings_city["geom_local_raw"] = [
            translate(geom, xoff=-float(cx), yoff=-float(cy))
            for geom, cx, cy in zip(
                buildings_city.geometry.values,
                buildings_city["bbox_cx_m"].to_numpy(),
                buildings_city["bbox_cy_m"].to_numpy(),
            )
        ]
        building_chunks.append(
            frame_with_object_geometry_columns(
                buildings_city,
                geom_cols=["geometry", "geometry_wgs84", "geometry_metric", "geom_local_raw"],
            )
        )

        neighbors_city = neighbors_work[neighbors_work["city_key"] == city_key].copy()
        if neighbors_city.empty:
            continue

        # Each neighbour moves with the building it belongs to.
        center_map = buildings_city.set_index("sample_id")[["bbox_cx_m", "bbox_cy_m"]].to_dict("index")
        neighbors_city = neighbors_city.to_crs(city_crs).copy()
        neighbors_city["metric_crs"] = city_crs
        neighbors_city["geometry_metric"] = pd.Series(
            list(neighbors_city.geometry.values), index=neighbors_city.index, dtype="object"
        )
        n_x = neighbors_city["sample_id"].map(lambda sid: center_map.get(sid, {}).get("bbox_cx_m", 0.0)).to_numpy()
        n_y = neighbors_city["sample_id"].map(lambda sid: center_map.get(sid, {}).get("bbox_cy_m", 0.0)).to_numpy()
        neighbors_city["geom_local_raw"] = [
            translate(geom, xoff=-float(cx), yoff=-float(cy))
            for geom, cx, cy in zip(neighbors_city.geometry.values, n_x, n_y)
        ]
        neighbor_chunks.append(
            frame_with_object_geometry_columns(
                neighbors_city,
                geom_cols=["geometry", "geometry_wgs84", "geometry_metric", "geom_local_raw"],
            )
        )

    buildings_local = gpd.GeoDataFrame(
        pd.concat(building_chunks, ignore_index=True), geometry="geom_local_raw", crs=None
    )
    if neighbor_chunks:
        neighbors_local = gpd.GeoDataFrame(
            pd.concat(neighbor_chunks, ignore_index=True), geometry="geom_local_raw", crs=None
        )
    else:
        empty_neighbors = neighbors_work.copy()
        empty_neighbors["metric_crs"] = pd.Series(dtype="object")
        empty_neighbors["geometry_metric"] = pd.Series(dtype="object")
        empty_neighbors["geom_local_raw"] = pd.Series(dtype="object")
        neighbors_local = gpd.GeoDataFrame(empty_neighbors.iloc[0:0].copy(), geometry="geom_local_raw", crs=None)

    return buildings_local, neighbors_local, metric_crs_map


def preprocess_local_geometries(
    gdf: gpd.GeoDataFrame,
    *,
    kind_label: str,
    normalization_cfg: FootprintNormalizationConfig,
    alignment_cfg: EdgeAlignmentConfig,
    min_area_m2: float,
    linear_filter_cfg: LinearArtifactFilterConfig | None = None,
    apply_linear_filter: bool = False,
    reference_buildings_gdf: gpd.GeoDataFrame | None = None,
    proximity_threshold_m: float = 2.0,
) -> gpd.GeoDataFrame:
    """Normalize and square up local footprints, and drop what is left too small.

    Pass ``reference_buildings_gdf`` for neighbours: their edges are then
    aligned to the main building's axes rather than to their own, so a terrace
    ends up on one grid. ``apply_linear_filter`` drops neighbour footprints that
    are really fences or walls rather than buildings.

    Writes ``geom_local_preray`` and the per-row diagnostics the notebook prints.
    """
    if gdf.empty:
        return gpd.GeoDataFrame(gdf.copy(), geometry="geom_local_raw", crs=None)

    work = gdf.copy()
    work["geometry"] = work["geom_local_raw"]
    work = gpd.GeoDataFrame(work, geometry="geometry", crs=None)

    work = normalize_buildings_gdf(work, geometry_col="geometry", cfg=normalization_cfg)
    work["geometry_before_align"] = work["geometry"]
    work["area_before_align_m2"] = work["geometry"].apply(_area_or_nan)

    if reference_buildings_gdf is not None:
        work = align_neighbors_gdf(
            work,
            reference_buildings_gdf,
            geometry_col="geometry",
            building_geometry_col="geom_local_preray",
            sample_id_col="sample_id",
            cfg=alignment_cfg,
            proximity_threshold_m=proximity_threshold_m,
        )
    else:
        work = align_buildings_gdf(work, geometry_col="geometry", cfg=alignment_cfg)

    work["n_vertices_preray"] = work["geometry"].apply(count_exterior_vertices)
    work["area_preray_m2"] = work["geometry"].apply(_area_or_nan)

    if apply_linear_filter:
        work = annotate_linear_artifact_flags(work, geometry_col="geometry", cfg=linear_filter_cfg)
        before_linear = len(work)
        work = work[~work["is_linear_artifact_hard"].fillna(False)].copy()
        print(f"[{kind_label}] dropped hard linear artifacts: {before_linear - len(work)}")
    else:
        work["is_linear_artifact_hard"] = False
        work["is_linear_artifact_review"] = False

    keep_mask = work["geometry"].notna() & ~work["geometry"].apply(lambda geom: getattr(geom, "is_empty", True))
    before_empty = len(work)
    work = work[keep_mask].copy()
    dropped_empty = before_empty - len(work)

    before_area = len(work)
    work = work[work["area_preray_m2"].fillna(0.0) >= float(min_area_m2)].copy()
    dropped_small = before_area - len(work)

    work["geom_local_preray"] = work["geometry"]

    n_norm_changed = int((work["vertex_delta_norm"] != 0).sum()) if "vertex_delta_norm" in work.columns else 0
    n_align_changed = int(work["alignment_changed"].fillna(False).sum()) if "alignment_changed" in work.columns else 0
    print(
        f"[{kind_label}] rows={len(work)} | normalized={n_norm_changed} | aligned={n_align_changed} | "
        f"dropped_empty={dropped_empty} | dropped_small={dropped_small}"
    )

    return gpd.GeoDataFrame(work, geometry="geometry", crs=None)


def _area_or_nan(geom) -> float:
    if geom is None or getattr(geom, "is_empty", True):
        return float("nan")
    return float(geom.area)


# ---------------------------------------------------------------------------
# Shared walls
# ---------------------------------------------------------------------------


def _edge_normal_form(p, q) -> dict | None:
    """An edge as ``(alpha, rho)``: its normal's angle and signed distance to the origin.

    Two edges lie on the same line when their alphas and rhos match, which is
    cheaper and more stable than comparing endpoints. The normal is flipped into
    the upper half-plane so that an edge and its reverse get the same alpha.
    """
    dx, dy = q[0] - p[0], q[1] - p[1]
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return None
    nx, ny = dy / length, -dx / length
    rho = nx * p[0] + ny * p[1]
    alpha = math.atan2(ny, nx)
    if alpha < 0:
        alpha += math.pi
        rho = -rho
        nx, ny = -nx, -ny
    return {"alpha": alpha, "rho": rho, "nx": nx, "ny": ny, "length": length}


def enforce_wall_colinearity(
    nei: Polygon,
    main: Polygon,
    *,
    angle_tol_deg: float = 5.0,
    max_rho_m: float = 0.25,
    max_shift_m: float | None = None,
) -> Polygon:
    """Project a neighbour's vertices onto the main building's matching wall lines.

    A shared wall in a terrace is usually mapped twice in OSM, a few centimetres
    apart. Normalization and alignment do not close that gap because each
    footprint is regularized on its own. Here each neighbour edge is matched to
    a main-building edge by angle and perpendicular offset, and the neighbour's
    vertices are moved onto the matched lines: a vertex between two matched
    edges goes to their intersection, a vertex with one match drops onto it.

    Vertices never move further than ``max_shift_m``, and a result that is
    invalid, collapsed, or overlaps the main building more than the input did is
    discarded in favour of the original.
    """
    if max_shift_m is None:
        max_shift_m = max_rho_m

    mc = list(main.exterior.coords[:-1])
    main_edges = []
    for i in range(len(mc)):
        j = (i + 1) % len(mc)
        ef = _edge_normal_form(mc[i], mc[j])
        if ef is not None:
            main_edges.append(ef)

    nc = list(nei.exterior.coords[:-1])
    n = len(nc)
    angle_tol = math.radians(angle_tol_deg)

    # For each neighbour edge, the closest near-parallel main edge, if any.
    edge_match: list[dict | None] = [None] * n
    for i in range(n):
        j = (i + 1) % n
        ef = _edge_normal_form(nc[i], nc[j])
        if ef is None:
            continue
        best, best_rd = None, float("inf")
        for me in main_edges:
            ad = abs(ef["alpha"] - me["alpha"])
            if ad > math.pi / 2:
                ad = math.pi - ad
            if ad >= angle_tol:
                continue
            rd = abs(me["nx"] * nc[i][0] + me["ny"] * nc[i][1] - me["rho"])
            if rd < max_rho_m and rd < best_rd:
                best, best_rd = me, rd
        if best is not None:
            edge_match[i] = best

    new_c = list(nc)
    n_moved = 0
    for i in range(n):
        mp = edge_match[(i - 1) % n]
        mc_ = edge_match[i]
        if mp is None and mc_ is None:
            continue

        if mp is not None and mc_ is not None:
            det = mp["nx"] * mc_["ny"] - mc_["nx"] * mp["ny"]
            if abs(det) > 1e-9:
                proposed = (
                    (mp["rho"] * mc_["ny"] - mc_["rho"] * mp["ny"]) / det,
                    (mp["nx"] * mc_["rho"] - mc_["nx"] * mp["rho"]) / det,
                )
            else:
                # The two matched lines are parallel, so drop onto the later one.
                d = mc_["nx"] * nc[i][0] + mc_["ny"] * nc[i][1] - mc_["rho"]
                proposed = (nc[i][0] - d * mc_["nx"], nc[i][1] - d * mc_["ny"])
        else:
            ref = mc_ if mc_ is not None else mp
            d = ref["nx"] * nc[i][0] + ref["ny"] * nc[i][1] - ref["rho"]
            proposed = (nc[i][0] - d * ref["nx"], nc[i][1] - d * ref["ny"])

        if math.hypot(proposed[0] - nc[i][0], proposed[1] - nc[i][1]) > max_shift_m:
            continue

        new_c[i] = proposed
        n_moved += 1

    if n_moved == 0:
        return nei

    result = Polygon(new_c + [new_c[0]])
    if not result.is_valid:
        result = make_geometry_valid(result)
        if not isinstance(result, Polygon):
            parts = polygon_parts(result)
            result = max(parts, key=lambda p: p.area) if parts else nei
    if result.is_empty or result.area < 1.0:
        return nei
    if result.intersection(main).area > nei.intersection(main).area + 1e-6:
        return nei
    return result


def remove_residual_overlap(nei: Polygon, main: Polygon, *, min_simplify_m: float = 0.05) -> Polygon:
    """Cut whatever overlap projection could not close, then simplify the cut edge.

    Subtracting the main building leaves a ragged boundary with more vertices
    than the neighbour started with, so the result is simplified at a tolerance
    scaled to how deep the overlap was, accepting the first tolerance that
    removes nine tenths of it.
    """
    if not nei.intersects(main):
        return nei
    overlap = nei.intersection(main)
    if overlap.is_empty or overlap.area < 1e-9:
        return nei

    orig_n = len(nei.exterior.coords) - 1
    diff = nei.difference(main)
    if diff is None or diff.is_empty:
        return nei
    if not diff.is_valid:
        diff = make_geometry_valid(diff)
    parts = polygon_parts(diff)
    if not parts:
        return nei
    diff_poly = max(parts, key=lambda p: p.area)

    best = diff_poly
    if len(diff_poly.exterior.coords) - 1 > orig_n:
        ow = (
            overlap.area / max(0.01, overlap.length)
            if hasattr(overlap, "length") and overlap.length > 0
            else 0.1
        )
        ow = max(ow, float(min_simplify_m))
        for factor in (1.0, 1.5, 2.0, 3.0):
            s = diff_poly.simplify(ow * factor, preserve_topology=True)
            if not isinstance(s, Polygon) or s.is_empty:
                continue
            if len(s.exterior.coords) - 1 < 3:
                continue
            if s.intersection(main).area < overlap.area * 0.1:
                best = s
                break

    result = best
    if not isinstance(result, Polygon) or result.is_empty or result.area < 1.0:
        rp = polygon_parts(result)
        result = max(rp, key=lambda p: p.area) if rp else best
    return result


def align_neighbour_walls(
    buildings_m: gpd.GeoDataFrame,
    neighbors_m: gpd.GeoDataFrame,
    *,
    geom_col: str = "geom_local",
) -> tuple[int, int, int]:
    """Close every neighbour-on-building overlap, in place on ``neighbors_m``.

    Projection is tried first because it preserves the neighbour's shape;
    cutting is the fallback for overlaps projection cannot reach, such as walls
    that meet at an angle.

    Returns how many neighbours were projected, how many were cut, and how many
    were changed at all.
    """
    main_geom_by_sid = dict(zip(buildings_m["sample_id"].astype(str), buildings_m[geom_col]))

    n_projected = n_cut = 0
    updated: dict = {}
    for idx in neighbors_m.index:
        sid = str(neighbors_m.loc[idx, "sample_id"])
        nei = neighbors_m.loc[idx, geom_col]
        main = main_geom_by_sid.get(sid)
        if nei is None or main is None:
            continue
        if not isinstance(nei, Polygon) or getattr(nei, "is_empty", True):
            continue
        if not isinstance(main, Polygon) or getattr(main, "is_empty", True):
            continue
        if not nei.intersects(main):
            continue
        overlap = nei.intersection(main)
        if overlap.is_empty or overlap.area < 1e-9:
            continue

        projected = enforce_wall_colinearity(nei, main)
        changed = projected is not nei
        proj_overlap = projected.intersection(main).area if changed else overlap.area

        if proj_overlap < 1e-6:
            updated[idx] = projected
            n_projected += 1
            continue

        result = remove_residual_overlap(projected if changed else nei, main)
        if result is not nei and result is not projected:
            updated[idx] = result
            if changed:
                n_projected += 1
            n_cut += 1
        elif changed:
            updated[idx] = projected
            n_projected += 1

    for idx, geom in updated.items():
        neighbors_m.at[idx, geom_col] = geom

    return n_projected, n_cut, len(updated)


def sync_geometry(gdf: gpd.GeoDataFrame, geom_col: str = "geom_local") -> gpd.GeoDataFrame:
    """Mirror a named geometry column into ``geometry`` and make it active.

    Stage 4 keeps several geometries per row and advances a different one at
    each step, but the decomposition modules, the review app and stage 5 all
    read plain ``geometry``. This keeps that column pointing at whichever
    version is current.
    """
    gdf["geometry"] = gdf[geom_col]
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs=None)
