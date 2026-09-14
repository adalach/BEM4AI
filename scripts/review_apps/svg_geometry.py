"""Convert Shapely geometry to the SVG paths used by the 2D review apps."""

from __future__ import annotations

from collections.abc import Iterable

from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
)
from shapely.geometry.base import BaseGeometry


def format_number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def ring_to_svg_path(coords) -> str:
    points = list(coords)
    if not points:
        return ""

    start_x, start_y = points[0]
    commands = [f"M {format_number(start_x)} {format_number(-start_y)}"]
    commands.extend(
        f"L {format_number(x_value)} {format_number(-y_value)}"
        for x_value, y_value in points[1:]
    )
    commands.append("Z")
    return " ".join(commands)


def line_to_svg_path(coords) -> str:
    points = list(coords)
    if not points:
        return ""

    start_x, start_y = points[0]
    commands = [f"M {format_number(start_x)} {format_number(-start_y)}"]
    commands.extend(
        f"L {format_number(x_value)} {format_number(-y_value)}"
        for x_value, y_value in points[1:]
    )
    return " ".join(commands)


def geometry_to_svg_path(geometry: BaseGeometry | None) -> str:
    if geometry is None or geometry.is_empty:
        return ""

    if isinstance(geometry, Polygon):
        parts = [ring_to_svg_path(geometry.exterior.coords)]
        parts.extend(ring_to_svg_path(interior.coords) for interior in geometry.interiors)
        return " ".join(part for part in parts if part)

    if isinstance(geometry, LineString):
        return line_to_svg_path(geometry.coords)

    if isinstance(geometry, (MultiPolygon, MultiLineString, GeometryCollection)):
        return " ".join(
            path
            for part in geometry.geoms
            if part is not None and not part.is_empty
            for path in [geometry_to_svg_path(part)]
            if path
        )

    return ""


def compute_view_box(*geometries: BaseGeometry | None) -> str:
    valid = [geometry for geometry in geometries if geometry is not None and not geometry.is_empty]
    if not valid:
        return "-10 -10 20 20"

    min_x = min(geometry.bounds[0] for geometry in valid)
    min_y = min(geometry.bounds[1] for geometry in valid)
    max_x = max(geometry.bounds[2] for geometry in valid)
    max_y = max(geometry.bounds[3] for geometry in valid)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    padding = max(width, height) * 0.08

    return " ".join(
        format_number(value)
        for value in (
            min_x - padding,
            -(max_y + padding),
            width + 2 * padding,
            height + 2 * padding,
        )
    )


def pack_geometries(geometries: Iterable[BaseGeometry]) -> BaseGeometry | None:
    valid = tuple(
        geometry for geometry in geometries if geometry is not None and not geometry.is_empty
    )
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    return GeometryCollection(valid)
