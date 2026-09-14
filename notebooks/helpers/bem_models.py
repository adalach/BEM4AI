"""Build Honeybee rooms and add the energy and context properties for Stage 5."""

from __future__ import annotations

from collections.abc import Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from tqdm import tqdm

from honeybee.face import Face
from honeybee.model import Model
from honeybee.room import Room
from honeybee.shade import Shade
from honeybee_energy.hvac.idealair import IdealAirSystem
from honeybee_energy.lib.programtypes import program_type_by_identifier
from ladybug_geometry.geometry3d import Face3D, Point3D

from .utils import angle_between_deg, polygon_parts

__all__ = [
    "add_neighbor_shades",
    "add_windows",
    "assign_energy_properties",
    "build_geometry_models",
]


def _horizontal_face(coordinates, elevation: float) -> Face3D:
    """Create a horizontal Face3D from a closed or open 2D polygon ring."""
    points = np.asarray(coordinates, dtype=float)
    if len(points) >= 3 and not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    return Face3D([Point3D(x, y, elevation) for x, y in points])


def _wall_face(start, end, bottom: float, top: float) -> Face3D:
    """Create one vertical wall face from a 2D footprint edge."""
    return Face3D(
        [
            Point3D(start[0], start[1], bottom),
            Point3D(end[0], end[1], bottom),
            Point3D(end[0], end[1], top),
            Point3D(start[0], start[1], top),
        ]
    )


def build_geometry_models(
    buildings: gpd.GeoDataFrame,
) -> dict[str, Model]:
    """Extrude Stage 4 zones, assemble models, and solve room adjacency."""
    models = {}
    for _, row in tqdm(
        buildings.iterrows(),
        total=len(buildings),
        desc="Building Honeybee geometry",
        unit="building",
    ):
        sample_id = str(row["sample_id"])
        zone_geometries = (
            row["zones"] if isinstance(row["zones"], (list, tuple)) else [row["zones"]]
        )
        rooms = []
        for level in range(int(row["levels_est"])):
            bottom = level * float(row["level_height_m"])
            top = (level + 1) * float(row["level_height_m"])
            for zone_index, geometry in enumerate(zone_geometries):
                parts = list(polygon_parts(geometry))
                for part_index, polygon in enumerate(parts):
                    if polygon.interiors:
                        raise ValueError(
                            f"{sample_id} zone {zone_index} contains holes; rerun Stage 4."
                        )
                    token = (
                        f"z{zone_index}"
                        if len(parts) == 1
                        else f"z{zone_index}_p{part_index}"
                    )
                    exterior = np.asarray(polygon.exterior.coords)[:, :2]
                    floor = _horizontal_face(exterior, bottom)
                    roof = _horizontal_face(exterior, top)
                    if floor.normal.z >= 0:
                        floor = Face3D(list(reversed(floor.boundary)))
                    if roof.normal.z <= 0:
                        roof = Face3D(list(reversed(roof.boundary)))
                    faces = [
                        Face(f"{sample_id}_{token}_L{level}_floor", floor),
                        Face(f"{sample_id}_{token}_L{level}_roof", roof),
                    ]
                    for edge_index in range(len(exterior) - 1):
                        wall = _wall_face(
                            exterior[edge_index], exterior[edge_index + 1], bottom, top
                        )
                        faces.append(
                            Face(f"{sample_id}_{token}_L{level}_wall_{edge_index}", wall)
                        )
                    rooms.append(Room(f"{sample_id}_{token}_L{level}", faces))

        try:
            Room.intersect_adjacency(rooms, tolerance=0.01, angle_tolerance=1)
            Room.solve_adjacency(rooms, tolerance=0.01)
        except Exception as error:
            raise RuntimeError(f"Could not solve room adjacency for {sample_id}") from error
        model = Model(identifier=sample_id, rooms=[])
        model.add_rooms(rooms)
        models[sample_id] = model
    return models


def _residential_type_ids(building_types: pd.DataFrame) -> set[str]:
    """Return canonical building-type IDs classified as residential."""
    identifiers = {
        str(int(row.id))
        for row in building_types.itertuples(index=False)
        if str(row.sector).casefold() == "residential"
    }
    if not identifiers:
        raise RuntimeError("building_types must define at least one residential row.")
    return identifiers


def _program_candidates(
    row: pd.Series,
    residential_ids: set[str],
    program_config: Mapping,
    highrise_floor: int,
) -> tuple[list[str], str]:
    """Build the preferred and compatible fallback program IDs for one building."""
    vintage = str(row.get("hb_vintage") or "").strip() or "2016"
    try:
        building_type_id = str(int(row["building_type_id"]))
    except Exception:
        building_type_id = None

    default = program_config.get("default", {})
    # This is because the imported library doesn't have this template
    if building_type_id in residential_ids:
        archetype = (
            "HighriseApartment"
            if int(row["levels_est"]) >= highrise_floor
            else "MidriseApartment"
        )
        alternative = (
            "MidriseApartment" if archetype == "HighriseApartment" else "HighriseApartment"
        )
        space_type = "Apartment"
    else:
        mapped = program_config.get(building_type_id, {}) if building_type_id else {}
        archetype = mapped.get("archetype", default.get("archetype", "MidriseApartment"))
        alternative = None
        space_type = mapped.get("space_type", default.get("space_type", "Apartment"))

    candidates = [f"{vintage}::{archetype}::{space_type}"]
    if vintage != "2016":
        candidates.append(f"2016::{archetype}::{space_type}")
    if alternative:
        candidates.append(f"{vintage}::{alternative}::{space_type}")
        if vintage != "2016":
            candidates.append(f"2016::{alternative}::{space_type}")
    return list(dict.fromkeys(candidates)), space_type


def _first_installed_program(candidates: list[str]) -> str | None:
    """Return the first candidate available in the installed standards library."""
    for candidate in candidates:
        try:
            program_type_by_identifier(candidate)
            return candidate
        except Exception:
            continue
    return None


def assign_energy_properties(
    models: Mapping[str, Model],
    buildings: gpd.GeoDataFrame,
    construction_sets: Mapping[str, object],
    building_types: pd.DataFrame,
    config: Mapping,
) -> gpd.GeoDataFrame:
    """Assign a valid program, custom construction set, and ideal HVAC to every room."""
    result = buildings.copy()
    residential_ids = _residential_type_ids(building_types)
    program_config = config.get("hb_program_mapping", {})
    highrise_floor = int(
        config.get("hb_program_classification", {}).get("highrise_floor_threshold", 9)
    )
    resolved = []
    unresolved = []
    fallback_count = 0

    for _, row in result.iterrows():
        sample_id = str(row["sample_id"])
        candidates, space_type = _program_candidates(
            row, residential_ids, program_config, highrise_floor
        )
        chosen = _first_installed_program(candidates)
        if chosen is None:
            unresolved.append((sample_id, candidates))
            resolved.append((None, None, space_type))
            continue
        if chosen != candidates[0]:
            fallback_count += 1
        resolved.append((chosen, chosen.split("::", 2)[1], space_type))

    if unresolved:
        examples = "; ".join(
            f"{sample_id}: {choices}" for sample_id, choices in unresolved[:3]
        )
        raise RuntimeError(
            f"No installed Honeybee program matched {len(unresolved)} buildings. {examples}"
        )

    result["hb_program_identifier"] = [value[0] for value in resolved]
    result["hb_program_type"] = [value[1] for value in resolved]
    result["hb_program_space_type"] = [value[2] for value in resolved]
    building_index = result.assign(sample_id=result["sample_id"].astype(str)).set_index(
        "sample_id"
    )

    assigned_rooms = 0
    for sample_id, model in models.items():
        program_id = building_index.at[sample_id, "hb_program_identifier"]
        construction_set = construction_sets[sample_id]
        for room in model.rooms:
            room.properties.energy.program_type = program_type_by_identifier(program_id)
            room.properties.energy.construction_set = construction_set
            room.properties.energy.hvac = IdealAirSystem(
                identifier=f"ideal_{room.identifier}",
                economizer_type="DifferentialDryBulb",
                demand_controlled_ventilation=False,
                sensible_heat_recovery=0,
                latent_heat_recovery=0,
                heating_air_temperature=50,
                cooling_air_temperature=13,
            )
            assigned_rooms += 1
    message = f"Assigned programs, constructions, and ideal HVAC to {assigned_rooms} rooms"
    if fallback_count:
        message += f"; {fallback_count} programs used a compatible fallback"
    print(message)
    return result


def _is_outdoor_vertical(face: Face) -> bool:
    """Return whether a Honeybee face can legally host an exterior aperture."""
    boundary = getattr(face, "boundary_condition", None)
    return (
        abs(face.geometry.normal.z) < 0.1
        and boundary is not None
        and boundary.__class__.__name__ == "Outdoors"
    )


def _bottom_edge(face: Face) -> LineString:
    """Project the lowest horizontal edge of a vertical Honeybee face to XY."""
    points = face.geometry.boundary
    elevations = np.array([point.z for point in points], dtype=float)
    bottom = float(elevations.min())
    for index, elevation in enumerate(elevations):
        following = (index + 1) % len(points)
        if abs(elevation - bottom) < 1e-6 and abs(points[following].z - bottom) < 1e-6:
            return LineString(
                [(points[index].x, points[index].y), (points[following].x, points[following].y)]
            )
    lowest = np.argsort(elevations)[:2]
    first, second = points[int(lowest[0])], points[int(lowest[1])]
    return LineString([(first.x, first.y), (second.x, second.y)])


def _edges_overlap(first_edge: LineString, second_edge: LineString) -> bool:
    """Match nearly collinear wall segments using Stage 4's geometric tolerances."""
    first = np.asarray(first_edge.coords, dtype=float)
    second = np.asarray(second_edge.coords, dtype=float)
    if len(first) < 2 or len(second) < 2:
        return False
    directions = (first[-1, :2] - first[0, :2], second[-1, :2] - second[0, :2])
    if angle_between_deg(*directions, undirected=True) > 5.0:
        return False
    if first_edge.distance(second_edge) > 0.10:
        return False
    overlap = first_edge.buffer(0.10, cap_style="flat").intersection(
        second_edge.buffer(0.10, cap_style="flat")
    )
    return overlap.length >= 0.50


def _line_parts(geometry):
    """Yield non-empty LineStrings from a line or multiline geometry."""
    if isinstance(geometry, LineString) and not geometry.is_empty:
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from (line for line in geometry.geoms if line is not None and not line.is_empty)


def _window_parameters(config: Mapping) -> tuple[dict[int, dict[str, float]], int, int]:
    """Read valid building-type parameters and the residential/service fallbacks."""
    window_config = config.get("window_characteristics", {})
    defaults = window_config.get("defaults", {})
    residential_fallback = int(defaults.get("residential_fallback_bt_id", 3))
    service_fallback = int(defaults.get("service_fallback_bt_id", 4))
    parameters = {}
    for key, values in window_config.items():
        if key == "defaults":
            continue
        try:
            parameters[int(key)] = {
                name: float(values[name])
                for name in (
                    "aperture_height",
                    "sill_height",
                    "horizontal_separation",
                    "ratio",
                )
            }
        except (KeyError, TypeError, ValueError):
            continue
    if not parameters:
        raise RuntimeError("hb_mappings.json contains no valid window characteristics.")
    return parameters, residential_fallback, service_fallback


def add_windows(
    models: Mapping[str, Model],
    buildings: gpd.GeoDataFrame,
    window_edges: gpd.GeoDataFrame,
    building_types: pd.DataFrame,
    config: Mapping,
) -> dict[str, int]:
    """Add rectangular apertures where Stage 4 selected an exterior wall edge."""
    if not isinstance(window_edges, gpd.GeoDataFrame) or window_edges.empty:
        raise ValueError("Stage 4 produced no final window edges. Re-run Stage 4.")
    parameters, residential_fallback, service_fallback = _window_parameters(config)
    residential_ids = {int(identifier) for identifier in _residential_type_ids(building_types)}
    building_type_by_sample = (
        buildings.assign(sample_id=buildings["sample_id"].astype(str))
        .set_index("sample_id")["building_type_id"]
        .astype("Int64")
        .to_dict()
    )
    grouped_edges = {
        str(sample_id): [line for geometry in group.geometry for line in _line_parts(geometry)]
        for sample_id, group in window_edges.groupby("sample_id")
    }

    counts = {}
    for sample_id, model in tqdm(
        models.items(), total=len(models), desc="Adding windows", unit="building"
    ):
        edges = grouped_edges.get(sample_id, [])
        building_type_id = building_type_by_sample.get(sample_id)
        fallback = (
            residential_fallback if building_type_id in residential_ids else service_fallback
        )
        values = parameters.get(building_type_id, parameters.get(fallback))
        if values is None:
            raise RuntimeError(
                f"No window parameters for building type {building_type_id} or fallback {fallback}."
            )

        added = 0
        for room in model.rooms:
            for face in room.faces:
                if not _is_outdoor_vertical(face) or face.apertures:
                    continue
                wall_edge = _bottom_edge(face)
                if wall_edge.length < 1e-6 or not any(
                    _edges_overlap(edge, wall_edge) for edge in edges
                ):
                    continue
                try:
                    face.apertures_by_ratio_rectangle(
                        ratio=values["ratio"],
                        aperture_height=values["aperture_height"],
                        sill_height=values["sill_height"],
                        horizontal_separation=values["horizontal_separation"],
                        vertical_separation=0.0,
                        tolerance=0.01,
                    )
                    added += len(face.apertures)
                except AssertionError:
                    continue
        counts[sample_id] = added
    return counts


def add_neighbor_shades(
    models: Mapping[str, Model],
    buildings: gpd.GeoDataFrame,
    shade_edges: gpd.GeoDataFrame | None,
    *,
    enabled: bool,
) -> dict[str, int]:
    """Extrude Stage 4 neighboring edges to the target building height."""
    counts = {sample_id: 0 for sample_id in models}
    if not enabled or not isinstance(shade_edges, gpd.GeoDataFrame) or shade_edges.empty:
        return counts

    indexed = buildings.assign(sample_id=buildings["sample_id"].astype(str)).set_index(
        "sample_id"
    )
    heights = indexed["levels_est"].astype(float) * indexed["level_height_m"].astype(float)
    for sample_id, group in tqdm(
        shade_edges.groupby("sample_id"),
        total=shade_edges["sample_id"].nunique(),
        desc="Adding neighboring shades",
        unit="building",
    ):
        sample_id = str(sample_id)
        model = models.get(sample_id)
        if model is None:
            continue
        shades = []
        for geometry in group.geometry:
            for line in _line_parts(geometry):
                coordinates = list(line.coords)
                if len(coordinates) != 2:
                    raise ValueError(
                        f"Expected one two-point shading edge for {sample_id}, got {len(coordinates)}."
                    )
                (x0, y0), (x1, y1) = coordinates
                top = float(heights.loc[sample_id])
                if np.hypot(x1 - x0, y1 - y0) < 1e-6 or top <= 0:
                    continue
                geometry = Face3D(
                    [
                        Point3D(x0, y0, 0.0),
                        Point3D(x1, y1, 0.0),
                        Point3D(x1, y1, top),
                        Point3D(x0, y0, top),
                    ]
                )
                shades.append(Shade(f"{sample_id}_shade_{len(shades)}", geometry))
        if shades:
            model.add_shades(shades)
        counts[sample_id] = len(shades)
    return counts
