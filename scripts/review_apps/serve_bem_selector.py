#!/usr/bin/env python3
"""Serve the Stage 5 browser app for reviewing generated Honeybee models."""

from __future__ import annotations

import argparse
import runpy
import sys
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse

import pandas as pd
import plotly
import plotly.graph_objects as go
from honeybee.model import Model
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import triangulate

if __package__:
    from .server_common import (
        ReviewHTTPServer,
        ReviewRequestHandler,
        PaginatedDataset,
        SelectionStore,
        natural_sort_key,
        read_sample_ids,
        repository_relative_path,
        start_server_thread,
    )
else:
    from server_common import (
        ReviewHTTPServer,
        ReviewRequestHandler,
        PaginatedDataset,
        SelectionStore,
        natural_sort_key,
        read_sample_ids,
        repository_relative_path,
        start_server_thread,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = Path(__file__).resolve().parent / "bem_selector"
DEFAULT_HBJSONS_FULL_PATH = PROJECT_ROOT / "data" / "interim" / "hbjsons_df_full.pkl"
DEFAULT_HBJSONS_SMALL_PATH = PROJECT_ROOT / "data" / "interim" / "hbjsons_df.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "hbjsons"
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "interim" / "selected_bad_bem_models.csv"
DEFAULT_TRANSFORMED_PATH = PROJECT_ROOT / "data" / "interim" / "buildings_transformed.pkl"
PLOTLY_JS_PATH = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
MAX_PAGE_SIZE = 24
DEFAULT_PAGE_SIZE = 6
MODEL_INCLUDE_TYPES = (
    "exterior_floor",
    "roof",
    "exterior_wall",
    "aperture",
    "door",
    "outdoor_shade",
    "shade_mesh",
)
CONTEXT_TRACE_NAMES = {"context_neighbor", "context_neighbor_wire"}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from notebooks.helpers.pickle_compat import read_pickle_compat
from scripts.hbjson_paths import resolve_hbjson_path

PLOT_HELPER_DIR = PROJECT_ROOT / "scripts" / "jupyter-honeybee-plot"
PLOT_HELPER_PATH = PLOT_HELPER_DIR / "plot_honeybee_model.py"
if not PLOT_HELPER_PATH.is_file():
    raise FileNotFoundError(f"Missing Honeybee plot helper: {PLOT_HELPER_PATH}")
if str(PLOT_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_HELPER_DIR))

from honeybee_plot_geometry import polygon_normal as _newell_normal

plot_honeybee_model = runpy.run_path(str(PLOT_HELPER_PATH))["plot_honeybee_model"]


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def _clean_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        try:
            as_float = float(value)
            return int(as_float) if float(as_float).is_integer() else None
        except Exception:
            return None


def _sample_city_key(sample_id: str) -> str:
    parts = str(sample_id).rsplit("_", 1)
    return parts[0] if parts else str(sample_id)


def _count_model_apertures(model: Model) -> int:
    count = 0
    for room in getattr(model, "rooms", []) or []:
        for face in getattr(room, "faces", []) or []:
            count += len(getattr(face, "apertures", []) or [])
    count += len(getattr(model, "orphaned_apertures", []) or [])
    return count


def _iter_unique_model_shades(model: Model) -> Iterable[object]:
    """Yield each shade once, including shades attached to faces and openings."""
    seen: set[tuple[str, object]] = set()
    candidates: list[object] = []

    for room in getattr(model, "rooms", []) or []:
        for face in getattr(room, "faces", []) or []:
            candidates.extend(getattr(face, "outdoor_shades", []) or [])
            for aperture in getattr(face, "apertures", []) or []:
                candidates.extend(getattr(aperture, "outdoor_shades", []) or [])
            for door in getattr(face, "doors", []) or []:
                candidates.extend(getattr(door, "outdoor_shades", []) or [])

    candidates.extend(getattr(model, "orphaned_shades", []) or [])
    candidates.extend(getattr(model, "shades", []) or [])

    for shade in candidates:
        identifier = _clean_text(getattr(shade, "identifier", None))
        token = ("identifier", identifier) if identifier else ("object", id(shade))
        if token in seen:
            continue
        seen.add(token)
        yield shade


def _count_model_outdoor_shades(model: Model) -> int:
    return sum(1 for _ in _iter_unique_model_shades(model))


def _model_max_z(model: Model) -> float:
    z_max = 0.0
    for room in getattr(model, "rooms", []) or []:
        for face in getattr(room, "faces", []) or []:
            boundary = getattr(getattr(face, "geometry", None), "boundary", None) or []
            for pt in boundary:
                try:
                    z_max = max(z_max, float(pt.z))
                except Exception:
                    continue
    return z_max if z_max > 0 else 10.0


def _model_bounds(model: Model) -> tuple[float, float, float, float, float, float] | None:
    """Return the axis-aligned bounds of all room faces in a model."""
    x_min = y_min = z_min = float("inf")
    x_max = y_max = z_max = float("-inf")

    for room in getattr(model, "rooms", []) or []:
        for face in getattr(room, "faces", []) or []:
            boundary = getattr(getattr(face, "geometry", None), "boundary", None) or []
            for pt in boundary:
                try:
                    x = float(pt.x)
                    y = float(pt.y)
                    z = float(pt.z)
                except Exception:
                    continue
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
                z_min = min(z_min, z)
                z_max = max(z_max, z)

    if x_min == float("inf"):
        return None
    return x_min, x_max, y_min, y_max, z_min, z_max


def _iter_polygons(geom) -> Iterable[Polygon]:
    if geom is None:
        return
    if isinstance(geom, Polygon):
        if not geom.is_empty:
            yield geom
        return
    if isinstance(geom, MultiPolygon):
        for sub_geom in geom.geoms:
            if sub_geom is not None and not sub_geom.is_empty:
                yield sub_geom
        return
    if isinstance(geom, GeometryCollection):
        for sub_geom in geom.geoms:
            yield from _iter_polygons(sub_geom)


def _iter_triangle_polygons(poly: Polygon) -> Iterable[Polygon]:
    """Yield triangles clipped to a possibly concave or holed polygon."""
    for tri in triangulate(poly):
        clipped = tri.intersection(poly)
        for candidate in _iter_polygons(clipped):
            if candidate.area <= 1e-8:
                continue
            coords = list(candidate.exterior.coords)
            if len(coords) == 4:
                yield candidate
            else:
                for sub_tri in triangulate(candidate):
                    sub_clipped = sub_tri.intersection(candidate)
                    for final_candidate in _iter_polygons(sub_clipped):
                        final_coords = list(final_candidate.exterior.coords)
                        if final_candidate.area > 1e-8 and len(final_coords) == 4:
                            yield final_candidate


def _build_context_neighbor_traces(
    neighbor_geometries: list[object],
    z_top: float,
    *,
    wall_eps: float = 0.08,
) -> list[go.BaseTraceType]:
    """Extrude neighboring footprints into translucent context traces."""
    if not neighbor_geometries or z_top <= 0:
        return []

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    wire_x: list[float | None] = []
    wire_y: list[float | None] = []
    wire_z: list[float | None] = []

    def add_triangle(points: list[tuple[float, float, float]]) -> None:
        start = len(vertices)
        vertices.extend(points)
        faces.append((start, start + 1, start + 2))

    for geom in neighbor_geometries:
        for poly in _iter_polygons(geom):
            ext = list(poly.exterior.coords)
            if len(ext) < 4:
                continue

            for tri in _iter_triangle_polygons(poly):
                tri_coords = list(tri.exterior.coords)[:-1]
                if len(tri_coords) != 3:
                    continue
                add_triangle([(float(x), float(y), float(z_top)) for x, y in tri_coords])

            ring = [(float(x), float(y)) for x, y in ext]
            for idx in range(len(ring) - 1):
                x0, y0 = ring[idx]
                x1, y1 = ring[idx + 1]
                dx = x1 - x0
                dy = y1 - y0
                edge_len = (dx * dx + dy * dy) ** 0.5
                if edge_len <= 1e-9:
                    continue

                # Draw the context wall as two mirrored shells so a coplanar
                # neighbor wall remains visible on either side of the original plane.
                nx = -dy / edge_len
                ny = dx / edge_len
                for direction in (-1.0, 1.0):
                    ox = direction * wall_eps * nx
                    oy = direction * wall_eps * ny
                    add_triangle(
                        [
                            (x0 + ox, y0 + oy, 0.0),
                            (x1 + ox, y1 + oy, 0.0),
                            (x1 + ox, y1 + oy, float(z_top)),
                        ]
                    )
                    add_triangle(
                        [
                            (x0 + ox, y0 + oy, 0.0),
                            (x1 + ox, y1 + oy, float(z_top)),
                            (x0 + ox, y0 + oy, float(z_top)),
                        ]
                    )

            for x, y in ring:
                wire_x.append(float(x))
                wire_y.append(float(y))
                wire_z.append(float(z_top))
            wire_x.append(None)
            wire_y.append(None)
            wire_z.append(None)

    if not vertices or not faces:
        return []

    mesh = go.Mesh3d(
        x=[pt[0] for pt in vertices],
        y=[pt[1] for pt in vertices],
        z=[pt[2] for pt in vertices],
        i=[tri[0] for tri in faces],
        j=[tri[1] for tri in faces],
        k=[tri[2] for tri in faces],
        name="context_neighbor",
        color="#d8d8d8",
        opacity=0.22,
        hoverinfo="skip",
        showscale=False,
        flatshading=True,
    )
    wire = go.Scatter3d(
        x=wire_x,
        y=wire_y,
        z=wire_z,
        mode="lines",
        name="context_neighbor_wire",
        hoverinfo="skip",
        line=dict(color="#aeb5bc", width=2),
        opacity=0.6,
    )
    return [mesh, wire]


def _context_xy_bounds(neighbor_geometries: list[object]) -> tuple[float, float, float, float] | None:
    """Return combined XY bounds for the available neighboring footprints."""
    if not neighbor_geometries:
        return None

    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")
    found = False

    for geom in neighbor_geometries:
        for poly in _iter_polygons(geom):
            bx_min, by_min, bx_max, by_max = poly.bounds
            x_min = min(x_min, float(bx_min))
            x_max = max(x_max, float(bx_max))
            y_min = min(y_min, float(by_min))
            y_max = max(y_max, float(by_max))
            found = True

    if not found:
        return None
    return x_min, x_max, y_min, y_max


def _load_context_neighbor_lookup(paths: list[Path]) -> dict[str, list[object]]:
    """Load Stage 4 neighboring footprints keyed by sample ID."""
    lookup: dict[str, list[object]] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = read_pickle_compat(path)
        except Exception as exc:
            print(f"Could not load transformed payload for BEM context: {path} ({exc})")
            continue
        if not isinstance(payload, dict):
            continue
        neighbors = payload.get("neighbors_m")
        if not isinstance(neighbors, pd.DataFrame) or neighbors.empty or "sample_id" not in neighbors.columns:
            continue
        geom_col = "geom_local" if "geom_local" in neighbors.columns else "geometry"
        if geom_col not in neighbors.columns:
            continue
        subset = neighbors.loc[neighbors[geom_col].notna(), ["sample_id", geom_col]].copy()
        for row in subset.itertuples(index=False):
            lookup.setdefault(str(row.sample_id), []).append(getattr(row, geom_col))
    return lookup


def _build_aperture_outlines_from_model(
    model: Model,
    extrude_eps: float = 0.02,
    color: str = "#1060A8",
    width: float = 1.5,
) -> "go.Scatter3d | None":
    """Outline apertures at the front face of their rendered slabs."""
    wx: list[float | None] = []
    wy: list[float | None] = []
    wz: list[float | None] = []

    def _add_ring(raw_pts) -> None:
        pts = [(float(p.x), float(p.y), float(p.z)) for p in raw_pts]
        if len(pts) < 3:
            return
        nx, ny, nz = _newell_normal(pts)
        ring = [
            (x + nx * extrude_eps, y + ny * extrude_eps, z + nz * extrude_eps)
            for x, y, z in pts
        ]
        for p in ring:
            wx.append(p[0])
            wy.append(p[1])
            wz.append(p[2])
        wx.append(ring[0][0])
        wy.append(ring[0][1])
        wz.append(ring[0][2])
        wx.append(None)
        wy.append(None)
        wz.append(None)

    def _visit_aperture(ap) -> None:
        geom = getattr(ap, "geometry", None)
        if geom is None:
            return
        raw = getattr(geom, "boundary", None) or getattr(geom, "vertices", None)
        if raw:
            _add_ring(raw)

    for room in getattr(model, "rooms", []) or []:
        for face in getattr(room, "faces", []) or []:
            for ap in getattr(face, "apertures", []) or []:
                _visit_aperture(ap)

    for ap in getattr(model, "orphaned_apertures", []) or []:
        _visit_aperture(ap)

    if not wx:
        return None

    return go.Scatter3d(
        x=wx,
        y=wy,
        z=wz,
        mode="lines",
        name="aperture_outline",
        hoverinfo="skip",
        line=dict(color=color, width=width),
        opacity=0.95,
    )


def _build_bidirectional_shade_traces_from_model(
    model: Model,
    extrude_eps: float = 0.08,
    color: str = "#7c5cc4",
    outline_color: str = "#4c2f91",
) -> list[go.BaseTraceType]:
    """Render shades as two-sided solids that remain visible from either side."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    wire_x: list[float | None] = []
    wire_y: list[float | None] = []
    wire_z: list[float | None] = []

    def add_triangle(points: list[tuple[float, float, float]]) -> None:
        start = len(vertices)
        vertices.extend(points)
        faces.append((start, start + 1, start + 2))

    def add_ring(points: list[tuple[float, float, float]]) -> None:
        if len(points) < 3:
            return
        for x, y, z in points:
            wire_x.append(x)
            wire_y.append(y)
            wire_z.append(z)
        wire_x.append(points[0][0])
        wire_y.append(points[0][1])
        wire_z.append(points[0][2])
        wire_x.append(None)
        wire_y.append(None)
        wire_z.append(None)

    for shade in _iter_unique_model_shades(model):
        geom = getattr(shade, "geometry", None)
        raw = getattr(geom, "boundary", None) or getattr(geom, "vertices", None)
        if not raw:
            continue

        pts = [(float(p.x), float(p.y), float(p.z)) for p in raw]
        if len(pts) >= 2 and pts[0] == pts[-1]:
            pts = pts[:-1]
        if len(pts) < 3:
            continue

        normal = getattr(geom, "normal", None)
        if normal is not None:
            nx, ny, nz = float(normal.x), float(normal.y), float(normal.z)
        else:
            nx, ny, nz = _newell_normal(pts)
        mag = (nx * nx + ny * ny + nz * nz) ** 0.5
        if mag <= 1e-12:
            continue
        nx, ny, nz = nx / mag, ny / mag, nz / mag

        front = [(x + nx * extrude_eps, y + ny * extrude_eps, z + nz * extrude_eps) for x, y, z in pts]
        back = [(x - nx * extrude_eps, y - ny * extrude_eps, z - nz * extrude_eps) for x, y, z in pts]

        for idx in range(1, len(front) - 1):
            add_triangle([front[0], front[idx], front[idx + 1]])
            add_triangle([back[0], back[idx + 1], back[idx]])

        for idx in range(len(front)):
            nxt = (idx + 1) % len(front)
            add_triangle([front[idx], front[nxt], back[nxt]])
            add_triangle([front[idx], back[nxt], back[idx]])

        add_ring(front)
        add_ring(back)
        for idx in range(len(front)):
            wire_x.extend([front[idx][0], back[idx][0], None])
            wire_y.extend([front[idx][1], back[idx][1], None])
            wire_z.extend([front[idx][2], back[idx][2], None])

    if not vertices or not faces:
        return []

    mesh = go.Mesh3d(
        x=[pt[0] for pt in vertices],
        y=[pt[1] for pt in vertices],
        z=[pt[2] for pt in vertices],
        i=[tri[0] for tri in faces],
        j=[tri[1] for tri in faces],
        k=[tri[2] for tri in faces],
        name="shade_overlay",
        color=color,
        opacity=0.72,
        hoverinfo="skip",
        showscale=False,
        flatshading=True,
    )
    wire = go.Scatter3d(
        x=wire_x,
        y=wire_y,
        z=wire_z,
        mode="lines",
        name="shade_overlay_wire",
        hoverinfo="skip",
        line=dict(color=outline_color, width=3),
        opacity=0.95,
    )
    return [mesh, wire]


def _count_model_floors(model: Model) -> int:
    """Count building storeys by clustering distinct room-floor z-levels."""
    floor_zs: list[float] = []
    for room in getattr(model, "rooms", []) or []:
        z_min = float("inf")
        for face in getattr(room, "faces", []) or []:
            for pt in getattr(getattr(face, "geometry", None), "boundary", None) or []:
                try:
                    z_min = min(z_min, float(pt.z))
                except Exception:
                    continue
        if z_min != float("inf"):
            floor_zs.append(z_min)

    if not floor_zs:
        return 1

    unique_levels: list[float] = []
    for z in sorted(floor_zs):
        if not unique_levels or z - unique_levels[-1] > 0.5:
            unique_levels.append(z)

    return max(1, len(unique_levels))


def _best_camera_eye(model: Model) -> tuple[float, float, float]:
    """Orient the camera toward the first exterior wall that contains a window."""
    for room in getattr(model, "rooms", []) or []:
        for face in getattr(room, "faces", []) or []:
            if not (getattr(face, "apertures", []) or []):
                continue
            geom = getattr(face, "geometry", None)
            if geom is None:
                continue
            normal = getattr(geom, "normal", None) or getattr(face, "normal", None)
            if normal is not None:
                nx, ny, nz = float(normal.x), float(normal.y), float(normal.z)
            else:
                raw = getattr(geom, "boundary", None) or getattr(geom, "vertices", None)
                if not raw or len(raw) < 3:
                    continue
                pts = [(float(p.x), float(p.y), float(p.z)) for p in raw]
                nx, ny, nz = _newell_normal(pts)
            # Skip roof/floor faces (near-horizontal normals).
            if abs(nz) > 0.7:
                continue
            h = (nx * nx + ny * ny) ** 0.5
            if h < 1e-6:
                continue
            # Flatten to horizontal and place the eye in the outward direction.
            nx, ny = nx / h, ny / h
            return (nx * 1.8, ny * 1.8, 0.75)
    return (1.55, -1.45, 0.92)


def _style_figure(
    fig,
    sample_id: str,
    *,
    model_bounds: tuple[float, float, float, float, float, float] | None = None,
    context_bounds: tuple[float, float, float, float] | None = None,
    camera_eye: tuple[float, float, float] | None = None,
) -> None:
    """Apply the fixed review-app colors, lighting, bounds, and camera."""
    # Favor ambient light so faces remain legible from every review angle.
    _SURFACE_LIGHTING = dict(ambient=0.9, diffuse=0.4, specular=0.05, roughness=0.5, fresnel=0.0)

    for trace in fig.data:
        name = (getattr(trace, "name", "") or "").lower()
        if name in {"outdoor_shade", "shade_mesh", "shade"}:
            trace.opacity = 0.28
            trace.color = "#7c5cc4"
            try:
                trace.lighting = _SURFACE_LIGHTING
            except Exception:
                pass
        elif name == "context_neighbor":
            trace.opacity = 0.22
            trace.color = "#d8d8d8"
        elif name == "context_neighbor_wire":
            trace.opacity = 0.6
            trace.line = {"width": 2, "color": "#aeb5bc"}
        elif name == "aperture":
            trace.opacity = 1.0
            trace.color = "#30C3FF"
            try:
                trace.lighting = _SURFACE_LIGHTING
            except Exception:
                pass
        elif name == "door":
            trace.opacity = 0.88
            trace.color = "#d7b38c"
            try:
                trace.lighting = _SURFACE_LIGHTING
            except Exception:
                pass
        elif name == "roof":
            trace.opacity = 1.0
            trace.color = "#7c1616"
            try:
                trace.lighting = _SURFACE_LIGHTING
            except Exception:
                pass
        elif name == "exterior_floor":
            trace.opacity = 0.92
            trace.color = "#ffe4b8"
            try:
                trace.lighting = _SURFACE_LIGHTING
            except Exception:
                pass
        elif name == "exterior_wall":
            trace.opacity = 1.0
            trace.color = "#FFA830"
            try:
                trace.lighting = _SURFACE_LIGHTING
            except Exception:
                pass
        elif name == "wireframe":
            trace.line = {"width": 2, "color": "#293845"}

    scene = dict(
        xaxis=dict(visible=False, showbackground=False),
        yaxis=dict(visible=False, showbackground=False),
        zaxis=dict(visible=False, showbackground=False),
        aspectmode="data",
        dragmode="turntable",
        camera=dict(
            eye=dict(
                x=camera_eye[0] if camera_eye else 1.55,
                y=camera_eye[1] if camera_eye else -1.45,
                z=camera_eye[2] if camera_eye else 0.92,
            ),
            up=dict(x=0, y=0, z=1),
        ),
    )

    if model_bounds is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = model_bounds
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        cz = 0.5 * (z_min + z_max)

        all_x_min, all_x_max = x_min, x_max
        all_y_min, all_y_max = y_min, y_max
        if context_bounds is not None:
            ctx_x_min, ctx_x_max, ctx_y_min, ctx_y_max = context_bounds
            all_x_min = min(all_x_min, ctx_x_min)
            all_x_max = max(all_x_max, ctx_x_max)
            all_y_min = min(all_y_min, ctx_y_min)
            all_y_max = max(all_y_max, ctx_y_max)

        half_x = max(abs(all_x_min - cx), abs(all_x_max - cx), 0.5 * (x_max - x_min), 3.0)
        half_y = max(abs(all_y_min - cy), abs(all_y_max - cy), 0.5 * (y_max - y_min), 3.0)
        half_xy = 1.08 * max(half_x, half_y)
        z_span = max(z_max - z_min, 3.0)
        z_pad_below = max(0.03 * z_span, 0.15)
        z_pad_above = max(0.2 * z_span, 2.0)

        scene["xaxis"]["range"] = [cx - half_xy, cx + half_xy]
        scene["yaxis"]["range"] = [cy - half_xy, cy + half_xy]
        scene["zaxis"]["range"] = [z_min - z_pad_below, z_max + z_pad_above]

    fig.update_layout(
        title=None,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=False,
        uirevision=sample_id,
        autosize=True,
        scene=scene,
    )
    # Let Plotly size the figure to the review card.
    fig.layout.width = None
    fig.layout.height = None


@dataclass(frozen=True)
class BemRecord:
    sample_id: str
    file_path: Path
    city_key: str
    model_id: str | None
    hb_program_identifier: str | None
    hb_construction_set_id: str | None
    building_type_id: int | None
    n_zones: int | None
    n_windows: int | None


class BemDataset(PaginatedDataset):
    """Load model metadata and build browser payloads on demand."""

    max_page_size = MAX_PAGE_SIZE

    def __init__(
        self,
        *,
        hbjsons_full_path: Path,
        hbjsons_small_path: Path,
        output_dir: Path,
        context_neighbor_lookup: dict[str, list[object]] | None = None,
        sample_id_filter: list[str] | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> None:
        self.hbjsons_full_path = hbjsons_full_path
        self.hbjsons_small_path = hbjsons_small_path
        self.output_dir = output_dir
        self.context_neighbor_lookup = context_neighbor_lookup or {}
        self.sample_id_filter = list(sample_id_filter) if sample_id_filter else None
        self.filtered_out_sample_ids: list[str] = []
        self.page_size = page_size
        self.records = self._load_records()
        self.record_lookup = {record.sample_id: record for record in self.records}

    def _apply_sample_id_filter(self, records: list["BemRecord"]) -> list["BemRecord"]:
        """Keep requested samples in request order and record missing IDs."""
        if not self.sample_id_filter:
            self.filtered_out_sample_ids = []
            return records

        record_lookup = {record.sample_id: record for record in records}
        filtered_records: list[BemRecord] = []
        missing_ids: list[str] = []
        seen: set[str] = set()

        for sample_id in self.sample_id_filter:
            if sample_id in seen:
                continue
            seen.add(sample_id)

            record = record_lookup.get(sample_id)
            if record is None:
                missing_ids.append(sample_id)
                continue
            filtered_records.append(record)

        self.filtered_out_sample_ids = missing_ids
        return filtered_records

    def _load_records(self) -> list[BemRecord]:
        """Load the canonical model index, falling back to exported HBJSON files."""
        df = None
        if self.hbjsons_full_path.exists():
            df = read_pickle_compat(self.hbjsons_full_path)
        elif self.hbjsons_small_path.exists():
            df = read_pickle_compat(self.hbjsons_small_path)

        if isinstance(df, pd.DataFrame) and not df.empty:
            rows = df.copy()
            if "sample_id" not in rows.columns:
                raise KeyError(f"{self.hbjsons_full_path} is missing 'sample_id'.")

            rows["sample_id"] = rows["sample_id"].astype(str)
            if "city_key" not in rows.columns:
                rows["city_key"] = rows["sample_id"].map(_sample_city_key)
            if "model_id" not in rows.columns:
                rows["model_id"] = rows["sample_id"]

            rows["file_path"] = [
                resolve_hbjson_path(row, self.output_dir)
                for row in rows.itertuples(index=False)
            ]
            rows = rows.dropna(subset=["file_path"]).copy()
            rows["file_exists"] = rows["file_path"].map(lambda path: Path(path).is_file())
            rows = rows[rows["file_exists"]].copy()
            rows = rows.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)
            rows = rows.sort_values("sample_id", key=lambda col: col.map(natural_sort_key), kind="mergesort")

            records = [
                BemRecord(
                    sample_id=str(row.sample_id),
                    file_path=Path(row.file_path),
                    city_key=_clean_text(row.city_key) or _sample_city_key(row.sample_id),
                    model_id=_clean_text(getattr(row, "model_id", None)),
                    hb_program_identifier=_clean_text(getattr(row, "hb_program_identifier", None)),
                    hb_construction_set_id=_clean_text(getattr(row, "hb_construction_set_id", None)),
                    building_type_id=_clean_int(getattr(row, "building_type_id", None)),
                    n_zones=_clean_int(getattr(row, "n_zones", None)),
                    n_windows=_clean_int(getattr(row, "n_windows", None)),
                )
                for row in rows.itertuples(index=False)
            ]
            return self._apply_sample_id_filter(records)

        hbjson_paths = sorted(self.output_dir.glob("*.hbjson"), key=lambda path: natural_sort_key(path.stem))
        records = [
            BemRecord(
                sample_id=path.stem,
                file_path=path.resolve(),
                city_key=_sample_city_key(path.stem),
                model_id=path.stem,
                hb_program_identifier=None,
                hb_construction_set_id=None,
                building_type_id=None,
                n_zones=None,
                n_windows=None,
            )
            for path in hbjson_paths
        ]
        return self._apply_sample_id_filter(records)

    def build_page_payload(
        self,
        page_index: int,
        saved_ids: set[str],
        csv_path: Path,
        page_size: int | None = None,
    ) -> dict[str, object]:
        """Build metadata for one paginated group of review cards."""
        effective_page_size = self._normalize_page_size(page_size)
        page_count = self.page_count_for(effective_page_size)
        page_index = self.normalize_page(page_index, effective_page_size)
        start = page_index * effective_page_size
        end = min(start + effective_page_size, len(self.records))
        page_records = self.records[start:end]

        return {
            "page": page_index,
            "page_count": page_count,
            "page_size": effective_page_size,
            "total_count": self.total_count,
            "start_index": start,
            "end_index": end,
            "saved_count": len(saved_ids),
            "csv_path": repository_relative_path(csv_path),
            "filter_requested_count": len(self.sample_id_filter or []),
            "filter_missing_count": len(self.filtered_out_sample_ids),
            "filter_missing_preview": self.filtered_out_sample_ids[:20],
            "samples": [
                {
                    "sample_id": record.sample_id,
                    "saved": record.sample_id in saved_ids,
                    "city_key": record.city_key,
                    "model_id": record.model_id,
                    "building_type_id": record.building_type_id,
                    "hb_program_identifier": record.hb_program_identifier,
                    "hb_construction_set_id": record.hb_construction_set_id,
                    "n_zones": record.n_zones,
                    "n_windows": record.n_windows,
                }
                for record in page_records
            ],
        }

    @lru_cache(maxsize=256)
    def build_sample_payload(self, sample_id: str) -> dict[str, object]:
        """Load and render one model for the browser review card."""
        record = self.record_lookup[sample_id]
        model = Model.from_hbjson(str(record.file_path))
        fig = plot_honeybee_model(
            model,
            title=None,
            show=False,
            show_wireframe=True,
            surface_opacity=1.0,
            include_types=list(MODEL_INCLUDE_TYPES),
            background_color="#ffffff",
            figsize=(420, 320),
            aspectmode="data",
        )
        if fig is None:
            raise RuntimeError(f"Plot creation returned no figure for {sample_id}.")

        context_neighbors = self.context_neighbor_lookup.get(sample_id, [])
        model_bounds = _model_bounds(model)
        context_bounds = _context_xy_bounds(context_neighbors)
        if context_neighbors:
            context_traces = _build_context_neighbor_traces(context_neighbors, _model_max_z(model))
            if context_traces:
                fig_json = fig.to_plotly_json()
                fig_json["data"] = [trace.to_plotly_json() for trace in context_traces] + list(fig_json.get("data", []))
                fig = go.Figure(fig_json)
        camera_eye = _best_camera_eye(model)
        _style_figure(fig, sample_id, model_bounds=model_bounds, context_bounds=context_bounds, camera_eye=camera_eye)

        for shade_trace in _build_bidirectional_shade_traces_from_model(model):
            fig.add_trace(shade_trace)

        aperture_outline = _build_aperture_outlines_from_model(model)
        if aperture_outline is not None:
            fig.add_trace(aperture_outline)

        zone_count = record.n_zones if record.n_zones is not None else len(getattr(model, "rooms", []) or [])
        window_count = record.n_windows if record.n_windows is not None else _count_model_apertures(model)
        shade_count = _count_model_outdoor_shades(model)
        floor_count = _count_model_floors(model)

        return {
            "sample_id": record.sample_id,
            "city_key": record.city_key,
            "model_id": record.model_id,
            "file_path": repository_relative_path(record.file_path),
            "hb_program_identifier": record.hb_program_identifier,
            "hb_construction_set_id": record.hb_construction_set_id,
            "building_type_id": record.building_type_id,
            "n_zones": zone_count,
            "n_windows": window_count,
            "n_floors": floor_count,
            "n_outdoor_shades": shade_count,
            "n_context_neighbors": len(context_neighbors),
            "figure": fig.to_plotly_json(),
        }


class AppState:
    """Hold the immutable model dataset and persistent review selections."""

    def __init__(
        self,
        *,
        hbjsons_full_path: Path,
        hbjsons_small_path: Path,
        output_dir: Path,
        csv_path: Path,
        transformed_paths: list[Path],
        sample_id_filter: list[str] | None = None,
    ) -> None:
        context_neighbor_lookup = _load_context_neighbor_lookup(transformed_paths)
        self.dataset = BemDataset(
            hbjsons_full_path=hbjsons_full_path,
            hbjsons_small_path=hbjsons_small_path,
            output_dir=output_dir,
            context_neighbor_lookup=context_neighbor_lookup,
            sample_id_filter=sample_id_filter,
        )
        self.store = SelectionStore(csv_path=csv_path)
        self.csv_path = csv_path


class AppHandler(ReviewRequestHandler):
    """Serve the BEM review frontend and its model-data endpoints."""

    static_root = STATIC_ROOT
    server_version = "BEM4AIBEMReview"

    def do_GET(self) -> None:
        """Serve frontend assets, page metadata, or one rendered model."""
        parsed = urlparse(self.path)

        if parsed.path in {"/", "/index.html"}:
            self._serve_static_file("index.html")
            return

        if parsed.path in {"/app.js", "/styles.css"}:
            self._serve_static_file(parsed.path.lstrip("/"))
            return

        if parsed.path == "/vendor/plotly.min.js":
            self._serve_file(PLOTLY_JS_PATH)
            return

        if parsed.path == "/api/page":
            pagination = self._parse_page_query(parsed.query)
            if pagination is None:
                return
            page_index, page_size = pagination

            state = self.server.app_state
            payload = state.dataset.build_page_payload(
                page_index=page_index,
                saved_ids=state.store.snapshot(),
                csv_path=state.csv_path,
                page_size=page_size,
            )
            self._send_json(payload)
            return

        if parsed.path == "/api/sample":
            query_values = parse_qs(parsed.query)
            sample_values = query_values.get("sample_id", [""])
            sample_id = sample_values[0].strip()
            state = self.server.app_state
            if not sample_id or sample_id not in state.dataset.record_lookup:
                self._send_json({"error": "Unknown sample_id."}, status=404)
                return

            try:
                payload = state.dataset.build_sample_payload(sample_id)
            except Exception as exc:
                self._send_json(
                    {"sample_id": sample_id, "error": f"Could not load/render model: {exc}"},
                    status=500,
                )
                return

            self._send_json(payload)
            return

        self._send_json({"error": "Not found."}, status=404)


def parse_args() -> argparse.Namespace:
    """Read command-line paths and server settings."""
    parser = argparse.ArgumentParser(description="Serve the BEM model review app.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the local server to.")
    parser.add_argument("--port", default=8001, type=int, help="Port to bind the local server to.")
    parser.add_argument(
        "--hbjsons-full",
        default=str(DEFAULT_HBJSONS_FULL_PATH),
        help="Path to hbjsons_df_full.pkl.",
    )
    parser.add_argument(
        "--hbjsons-small",
        default=str(DEFAULT_HBJSONS_SMALL_PATH),
        help="Path to hbjsons_df.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="HBJSON output directory used when file paths must be reconstructed.",
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV_PATH),
        help="Path to the CSV file used to persist rejected sample IDs.",
    )
    parser.add_argument(
        "--transformed",
        default=str(DEFAULT_TRANSFORMED_PATH),
        help="Path to the unified Stage 4 payload used for original-neighbor context overlays.",
    )
    parser.add_argument(
        "--sample-id-csv",
        default=None,
        help="Optional CSV with a 'sample_id' column used to restrict the dataset to a review subset.",
    )
    return parser.parse_args()


def create_server(
    *,
    hbjsons_full_path: Path = DEFAULT_HBJSONS_FULL_PATH,
    hbjsons_small_path: Path = DEFAULT_HBJSONS_SMALL_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    csv_path: Path = DEFAULT_CSV_PATH,
    transformed_path: Path = DEFAULT_TRANSFORMED_PATH,
    sample_id_filter: list[str] | None = None,
    host: str = "127.0.0.1",
    port: int = 8001,
) -> ReviewHTTPServer:
    """Create a configured review server without starting its event loop."""
    app_state = AppState(
        hbjsons_full_path=Path(hbjsons_full_path).expanduser().resolve(),
        hbjsons_small_path=Path(hbjsons_small_path).expanduser().resolve(),
        output_dir=Path(output_dir).expanduser().resolve(),
        csv_path=Path(csv_path).expanduser().resolve(),
        transformed_paths=[Path(transformed_path).expanduser().resolve()],
        sample_id_filter=sample_id_filter,
    )
    return ReviewHTTPServer((host, int(port)), AppHandler, app_state=app_state)


def start_server(
    *,
    hbjsons_full_path: Path = DEFAULT_HBJSONS_FULL_PATH,
    hbjsons_small_path: Path = DEFAULT_HBJSONS_SMALL_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    csv_path: Path = DEFAULT_CSV_PATH,
    transformed_path: Path = DEFAULT_TRANSFORMED_PATH,
    sample_id_filter: list[str] | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
) -> tuple[ReviewHTTPServer, threading.Thread, str]:
    """Start the review server in a background thread and return its URL."""
    server = create_server(
        hbjsons_full_path=hbjsons_full_path,
        hbjsons_small_path=hbjsons_small_path,
        output_dir=output_dir,
        csv_path=csv_path,
        transformed_path=transformed_path,
        sample_id_filter=sample_id_filter,
        host=host,
        port=port,
    )
    return start_server_thread(server, "bem4ai-bem-selector")


def main() -> None:
    """Run the review server until interrupted."""
    args = parse_args()
    sample_id_csv_path = Path(args.sample_id_csv).expanduser().resolve() if args.sample_id_csv else None
    sample_id_filter = read_sample_ids(sample_id_csv_path) if sample_id_csv_path else None
    server = create_server(
        hbjsons_full_path=Path(args.hbjsons_full),
        hbjsons_small_path=Path(args.hbjsons_small),
        output_dir=Path(args.output_dir),
        csv_path=Path(args.csv),
        transformed_path=Path(args.transformed),
        sample_id_filter=sample_id_filter,
        host=args.host,
        port=args.port,
    )
    bound_host, bound_port = server.server_address[:2]

    print(f"Serving BEM selector at http://{bound_host}:{bound_port}")
    print(f"HBJSONs full: {server.app_state.dataset.hbjsons_full_path}")
    print(f"HBJSONs small: {server.app_state.dataset.hbjsons_small_path}")
    print(f"HBJSON output dir: {server.app_state.dataset.output_dir}")
    print(f"Context payload: {Path(args.transformed).expanduser().resolve()}")
    if sample_id_csv_path is not None:
        print(f"Sample filter CSV: {sample_id_csv_path}")
        print(f"Requested filtered samples: {len(sample_id_filter or [])}")
        print(f"Missing filtered samples: {len(server.app_state.dataset.filtered_out_sample_ids)}")
        if server.app_state.dataset.filtered_out_sample_ids:
            preview = server.app_state.dataset.filtered_out_sample_ids[:10]
            print(f"Missing filtered sample preview: {preview}")
    print(f"Models available: {server.app_state.dataset.total_count}")
    print(f"Selections CSV: {server.app_state.csv_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
