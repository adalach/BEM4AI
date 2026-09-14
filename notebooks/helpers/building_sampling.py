"""Select OSM building footprints and their surroundings for the BEM4AI dataset.

Stage 3 draws a population-weighted sample of buildings from the per-city OSM
extracts, keeps the footprints that can become a usable model, and collects the
neighbouring buildings that shade them.

This module holds the mechanical half: reading an extract, the largest-remainder
allocation, reading a raster at a point, the footprint quality filters, the
identifier scheme and the neighbour search. What makes a building an office or a
dwelling, and where the quality thresholds sit, are decisions and stay in the
notebook.

Import from a notebook as::

    from helpers.building_sampling import FootprintLimits, allocate_by_population

"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyrosm import OSM
from shapely.geometry import MultiPoint, Point
from tqdm import tqdm

from .utils import (
    as_int_or_none,
    count_exterior_vertices,
    count_polygon_holes,
    empty_gdf,
    ensure_wgs84,
    extract_vertices,
    make_geometry_valid,
    min_rotated_rect_dims,
)

__all__ = [
    "WGS84",
    "FootprintLimits",
    "NeighborSearch",
    "load_city_buildings",
    "make_source_sample_id",
    "raster_value_at",
    "sample_rasters_at_point",
    "allocate_by_population",
    "filter_footprints",
    "assign_sample_ids",
    "build_neighbors",
]

WGS84 = "EPSG:4326"


@dataclass(frozen=True)
class FootprintLimits:
    """What a footprint has to look like to be worth modelling.

    A sliver - long, thin and a couple of metres wide - is almost always a wall
    or a fence mapped as a building, so it is removed before the area test.
    """

    area_min_m2: float
    area_max_m2: float
    max_holes: int
    max_exterior_vertices: int
    sliver_max_width_m: float
    sliver_min_aspect_ratio: float


@dataclass(frozen=True)
class NeighborSearch:
    """How far around a footprint to look, and when a candidate is the same building."""

    radius_m: float
    overlap_area_eps_m2: float


def load_city_buildings(pbf_path: Path | str) -> gpd.GeoDataFrame:
    """Polygonal OSM buildings of one city extract, in WGS84."""
    buildings = OSM(str(pbf_path)).get_buildings()
    if buildings is None or buildings.empty:
        return empty_gdf()
    buildings = ensure_wgs84(buildings, "OSM buildings")
    usable = buildings.geometry.notna() & ~buildings.geometry.is_empty
    return buildings.loc[
        usable & buildings.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy()


def make_source_sample_id(city_key: str, building_id) -> str:
    """`<city>_<osm id>`, reduced to characters that are safe in a filename."""
    building_id = as_int_or_none(building_id)
    if building_id is None:
        raise ValueError(f"No usable OSM id for city={city_key!r}")
    return re.sub(r"[^A-Za-z0-9_-]", "_", f"{city_key}_{building_id}")


def raster_value_at(dataset, x: float, y: float):
    """One raster value at a point, or None where the raster has no data."""
    values = next(dataset.sample([(x, y)]), None)
    if values is None or len(values) == 0:
        return None
    value = values[0]
    if np.ma.isMaskedArray(values) and values.mask[0]:
        return None
    if dataset.nodata is not None and value == dataset.nodata:
        return None
    if isinstance(value, np.floating) and np.isnan(value):
        return None
    return value.item() if hasattr(value, "item") else value


def sample_rasters_at_point(point_wgs84, rasters: dict, required: tuple[str, ...] = ()) -> dict | None:
    """Read every open raster at one WGS84 point, each in its own CRS.

    Returns None when a raster named in `required` does not cover the point, so
    a building outside the chip is skipped rather than carrying a null value
    into the dataset. Optional rasters simply come back as None.
    """
    values = {}
    point = gpd.GeoSeries([point_wgs84], crs=WGS84)
    for name, dataset in rasters.items():
        if dataset is None:
            values[name] = None
            continue
        projected = point.to_crs(dataset.crs).iloc[0]
        bounds = dataset.bounds
        inside = (bounds.left <= projected.x <= bounds.right
                  and bounds.bottom <= projected.y <= bounds.top)
        values[name] = raster_value_at(dataset, projected.x, projected.y) if inside else None
        if name in required and values[name] is None:
            return None
    return values


def allocate_by_population(total: int, cities: pd.DataFrame) -> pd.Series:
    """Split `total` across cities in proportion to population.

    Uses largest remainders, so the parts sum exactly to `total`, and breaks ties
    on population then city name - the same split on every run and machine.
    """
    pool = cities[["city_key", "population"]].drop_duplicates("city_key").copy()
    pool["population"] = pd.to_numeric(pool["population"], errors="coerce").fillna(0.0)
    pool = pool.loc[pool["population"] > 0]

    allocations = pd.Series(0, index=pool["city_key"].astype(str), dtype=int)
    if int(total) <= 0 or pool.empty:
        return allocations

    exact = int(total) * pool["population"] / float(pool["population"].sum())
    pool["base"] = np.floor(exact).astype(int)
    pool["fraction"] = exact - pool["base"]

    remainder = int(total) - int(pool["base"].sum())
    if remainder > 0:
        winners = pool.sort_values(
            ["fraction", "population", "city_key"], ascending=[False, False, True]
        ).index[:remainder]
        pool.loc[winners, "base"] += 1
    return pool.set_index("city_key")["base"].astype(int)


def filter_footprints(
    buildings: gpd.GeoDataFrame,
    limits: FootprintLimits,
    metric_crs_for: callable,
    *,
    label: str = "footprints",
) -> gpd.GeoDataFrame:
    """Keep the footprints that can become a model, and report what went.

    `metric_crs_for(city_key, fallback_gdf)` supplies the projected CRS each
    city's areas and widths are measured in.
    """
    if buildings is None or buildings.empty:
        return empty_gdf(columns=["geometry"])

    kept = gpd.GeoDataFrame(buildings.copy(), geometry="geometry", crs=WGS84)
    start = len(kept)

    kept = kept.drop_duplicates("source_sample_id").reset_index(drop=True)
    kept["geometry"] = kept.geometry.apply(make_geometry_valid)
    kept = kept.loc[kept.geometry.notna() & ~kept.geometry.is_empty]
    # A multipolygon footprint has no single outline to extrude, so it is dropped here.
    kept = kept.loc[kept.geometry.type == "Polygon"].copy()

    if kept.empty:
        return _nothing_kept(kept, start, label)

    # Areas and widths are both measured on the projected geometry: a rectangle fitted
    # to degrees would be a few 1e-5 wide everywhere, and stretched by latitude.
    projected = _project_by_city(kept, metric_crs_for)
    kept["area_m2"] = projected.area

    dimensions = projected.apply(min_rotated_rect_dims)
    width = dimensions.map(lambda pair: pair[0])
    aspect = dimensions.map(lambda pair: pair[1]) / width
    sliver = width.lt(limits.sliver_max_width_m) & aspect.gt(limits.sliver_min_aspect_ratio)
    kept = kept.loc[~sliver.fillna(False)].copy()
    if kept.empty:
        return _nothing_kept(kept, start, label)

    kept["n_holes"] = kept.geometry.apply(count_polygon_holes)
    kept["n_vertices"] = kept.geometry.apply(count_exterior_vertices)
    kept = kept.loc[
        kept["n_holes"].le(limits.max_holes)
        & kept["n_vertices"].le(limits.max_exterior_vertices)
        & kept["area_m2"].between(limits.area_min_m2, limits.area_max_m2)
    ].reset_index(drop=True)

    print(f"[{label}] footprint filters kept {len(kept)}/{start}")
    return gpd.GeoDataFrame(kept, geometry="geometry", crs=WGS84)


def _nothing_kept(frame, start: int, label: str) -> gpd.GeoDataFrame:
    """An empty result, reported the same way as a full one.

    `GeoSeries.apply` on an empty frame comes back with geometry dtype, so the
    numeric tests further down would raise a confusing TypeError instead.
    """
    print(f"[{label}] footprint filters kept 0/{start}")
    return empty_gdf(columns=list(frame.columns))


def _project_by_city(buildings: gpd.GeoDataFrame, metric_crs_for: callable) -> gpd.GeoSeries:
    """Every footprint in its own city's projected CRS, so metres mean metres.

    The cities are far enough apart that no single projection serves them all, so
    each is converted separately and the pieces are reassembled in row order.
    """
    projected = gpd.GeoSeries(index=buildings.index, dtype="geometry")
    for city_key, indices in buildings.groupby("city_key").groups.items():
        subset = buildings.loc[list(indices)]
        crs = metric_crs_for(city_key, fallback_gdf=subset)
        projected.loc[subset.index] = subset.to_crs(crs).geometry.values
    return projected


def assign_sample_ids(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Number the sample `<city_key>_0001` upwards, in a stable order.

    The order is fixed by the OSM id rather than by sampling order, so the same
    building keeps the same `sample_id` whenever the stage is re-run.
    """
    ordered = buildings.copy()
    ordered["_osm_id_sort"] = pd.to_numeric(ordered["building_id"], errors="coerce")
    ordered = ordered.sort_values(
        ["city_key", "_osm_id_sort", "building_type_id", "source_sample_id"], kind="mergesort"
    ).reset_index(drop=True)

    ordered["sample_id"] = None
    for city_key, indices in ordered.groupby("city_key", sort=False).groups.items():
        indices = list(indices)
        width = max(4, len(str(len(indices))))
        for sequence, index in enumerate(indices, start=1):
            ordered.at[index, "sample_id"] = f"{city_key}_{sequence:0{width}d}"
    return ordered.drop(columns="_osm_id_sort")


def build_neighbors(
    buildings: gpd.GeoDataFrame,
    pbf_dir: Path,
    metric_crs_for: callable,
    search: NeighborSearch,
) -> gpd.GeoDataFrame:
    """Buildings standing close enough to each sample to shade it.

    Closeness is measured vertex to vertex rather than centroid to centroid, so a
    long block counts as a neighbour when any part of it comes within
    `search.radius_m`. A candidate that overlaps the sample is the same building
    mapped twice and is dropped.
    """
    records = []
    overlapping = 0

    for city_key, indices in tqdm(
        list(buildings.groupby("city_key").groups.items()), desc="Neighbours"
    ):
        samples = buildings.loc[list(indices)]
        pbf_path = pbf_dir / Path(samples["osm_pbf"].iloc[0]).name
        city = load_city_buildings(pbf_path)
        if city.empty:
            continue
        city["geometry"] = city.geometry.make_valid()
        city = city.loc[city.geometry.notna() & ~city.geometry.is_empty].reset_index(drop=True)

        crs = metric_crs_for(city_key, fallback_gdf=samples)
        city_m = city.to_crs(crs)
        centroids_m = city_m.geometry.centroid

        osm_id_to_row = {}
        if "id" in city.columns:
            for position, value in city["id"].items():
                osm_id = as_int_or_none(value)
                if osm_id is not None:
                    osm_id_to_row.setdefault(osm_id, int(position))

        # One point per vertex of every building in the city, in one spatial index.
        vertex_rows, vertex_points = [], []
        for position, geometry in enumerate(city_m.geometry):
            for x, y in extract_vertices(geometry):
                vertex_points.append(Point(x, y))
                vertex_rows.append(position)
        if not vertex_points:
            continue
        vertices = gpd.GeoDataFrame(
            {"row": vertex_rows}, geometry=gpd.GeoSeries(vertex_points, crs=crs)
        )
        vertex_index = vertices.sindex
        centroid_fallbacks = 0

        for sample in samples.itertuples(index=False):
            sample_m = gpd.GeoSeries([sample.geometry], crs=WGS84).to_crs(crs).iloc[0]
            row = osm_id_to_row.get(as_int_or_none(sample.building_id))
            if row is None:
                # The extract was re-cut since sampling; fall back to the closest footprint.
                row = int(centroids_m.distance(sample_m.centroid).idxmin())
                centroid_fallbacks += 1

            outline = extract_vertices(city_m.geometry.iloc[row])
            if not outline:
                continue
            nearby = list(vertex_index.query(MultiPoint(outline).buffer(search.radius_m).envelope))
            if not nearby:
                continue

            outline_points = gpd.GeoSeries([Point(x, y) for x, y in outline], crs=crs)
            candidates = set()
            for candidate in vertices.iloc[nearby].itertuples(index=False):
                if outline_points.distance(candidate.geometry).min() <= search.radius_m:
                    candidates.add(int(candidate.row))
            candidates.discard(row)

            main_geometry = city_m.geometry.iloc[row]
            for candidate_row in candidates:
                candidate_m = city_m.geometry.iloc[candidate_row]
                if float(candidate_m.intersection(main_geometry).area) > search.overlap_area_eps_m2:
                    overlapping += 1
                    continue
                records.append({
                    "sample_id": str(sample.sample_id),
                    "geometry": gpd.GeoSeries([candidate_m], crs=crs).to_crs(WGS84).iloc[0],
                })

        if centroid_fallbacks:
            print(f"[warn] {city_key}: matched {centroid_fallbacks} samples by centroid, not by OSM id")

    if not records:
        return empty_gdf(["sample_id", "geometry"])

    neighbors = gpd.GeoDataFrame(records, geometry="geometry", crs=WGS84)
    # The same building can be a neighbour of one sample through several vertices.
    neighbors["_wkb"] = neighbors.geometry.map(lambda geometry: geometry.wkb_hex)
    neighbors = (
        neighbors.drop_duplicates(["sample_id", "_wkb"]).drop(columns="_wkb").reset_index(drop=True)
    )
    print(f"Neighbour geometries: {len(neighbors)}; overlapping candidates dropped: {overlapping}")
    return neighbors
