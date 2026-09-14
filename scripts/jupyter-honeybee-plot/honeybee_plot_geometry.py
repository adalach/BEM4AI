"""Backend-independent polygon helpers shared by the Honeybee renderers."""

from __future__ import annotations

from itertools import chain

import numpy as np

__all__ = [
    "add_mesh_like",
    "aperture_category",
    "boundary_condition_name",
    "build_plane_basis",
    "collect_model_geometry",
    "door_category",
    "point3d_tuple",
    "polygon_normal",
    "project_to_plane",
    "remove_collinear_points",
    "remove_consecutive_duplicates",
    "renderer_color_map",
    "surface_category",
    "triangulate_polygon",
]

FALLBACK_COLORS = (
    "#8da0cb",
    "#fc8d62",
    "#66c2a5",
    "#ffd92f",
    "#a6d854",
    "#e78ac3",
    "#8dd3c7",
    "#bebada",
)


def _palette_hex(colorset, index: int, fallback: str = "#cccccc") -> str:
    if colorset is None:
        return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]
    try:
        palette_source = None
        for getter in ("openstudio_palette", "openstudio"):
            if hasattr(colorset, getter):
                palette_source = getattr(colorset, getter)()
                break
        if palette_source is None:
            return fallback
        palette = getattr(palette_source, "colors", palette_source)
        color = palette[index % len(palette)]
        return color.to_hex() if hasattr(color, "to_hex") else getattr(color, "hex", str(color))
    except (AttributeError, IndexError, TypeError, ValueError):
        return fallback


def renderer_color_map(colorset) -> dict[str, str]:
    """Return the surface colors shared by both Honeybee renderers."""
    indices = {
        "exterior_wall": 0,
        "interior_wall": 1,
        "roof": 3,
        "ceiling": 4,
        "exterior_floor": 6,
        "interior_floor": 7,
        "air_wall": 12,
        "aperture": 9,
        "interior_aperture": 9,
        "door": 10,
        "interior_door": 10,
        "outdoor_shade": 11,
        "indoor_shade": 11,
        "shade_mesh": 11,
        "shade": 11,
        "default": 0,
    }
    return {name: _palette_hex(colorset, index) for name, index in indices.items()}


def point3d_tuple(point) -> tuple[float, float, float]:
    """Return a Ladybug point or three-item sequence as an XYZ tuple."""
    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
        return float(point.x), float(point.y), float(point.z)
    return float(point[0]), float(point[1]), float(point[2])


def add_mesh_like(
    mesh_geometry,
    key: str,
    *,
    include_types,
    categories: dict[str, dict],
) -> None:
    """Add a pre-meshed Honeybee geometry object to one renderer category."""
    if include_types is not None and key not in include_types:
        return
    if mesh_geometry is None:
        return

    vertices = (
        getattr(mesh_geometry, "vertices", None)
        or getattr(mesh_geometry, "points", None)
        or getattr(mesh_geometry, "vertices_xyz", None)
    )
    faces = (
        getattr(mesh_geometry, "faces", None)
        or getattr(mesh_geometry, "triangles", None)
        or getattr(mesh_geometry, "indices", None)
    )
    if not vertices or not faces:
        return

    points = [point3d_tuple(point) for point in vertices]
    category = categories.setdefault(key, {"pts": [], "idx": {}, "I": [], "J": [], "K": []})
    for point in points:
        rounded = tuple(round(coordinate, 6) for coordinate in point)
        if rounded not in category["idx"]:
            category["idx"][rounded] = len(category["pts"])
            category["pts"].append(rounded)

    for face in faces:
        try:
            indices = list(face)
        except TypeError:
            continue
        if len(indices) < 3:
            continue
        first = indices[0]
        for offset in range(1, len(indices) - 1):
            triangle = (points[first], points[indices[offset]], points[indices[offset + 1]])
            packed = [
                category["idx"][tuple(round(coordinate, 6) for coordinate in point)]
                for point in triangle
            ]
            category["I"].append(packed[0])
            category["J"].append(packed[1])
            category["K"].append(packed[2])


def remove_consecutive_duplicates(points, tolerance: float = 1e-9):
    """Remove repeated neighboring XYZ vertices, including a repeated closing point."""
    if not points:
        return []
    cleaned = [tuple(points[0])]
    for point in points[1:]:
        previous = cleaned[-1]
        if all(abs(previous[index] - point[index]) <= tolerance for index in range(3)):
            continue
        cleaned.append(tuple(point))
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def polygon_normal(points):
    """Return a stable unit normal using Newell's polygon method."""
    nx = ny = nz = 0.0
    count = len(points)
    for index in range(count):
        x1, y1, z1 = points[index]
        x2, y2, z2 = points[(index + 1) % count]
        nx += (y1 - y2) * (z1 + z2)
        ny += (z1 - z2) * (x1 + x2)
        nz += (x1 - x2) * (y1 + y2)
    normal = np.array((nx, ny, nz), dtype=float)
    magnitude = np.linalg.norm(normal)
    return normal / magnitude if magnitude > 1e-12 else np.array([0.0, 0.0, 1.0])


def build_plane_basis(normal):
    """Build stable orthogonal axes in the plane defined by `normal`."""
    normal = np.asarray(normal, dtype=float)
    reference = (
        np.array([1.0, 0.0, 0.0])
        if abs(normal[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    first = np.cross(normal, reference)
    first_magnitude = np.linalg.norm(first)
    first = (
        np.array([1.0, 0.0, 0.0])
        if first_magnitude < 1e-12
        else first / first_magnitude
    )
    second = np.cross(normal, first)
    second_magnitude = np.linalg.norm(second)
    second = (
        np.array([0.0, 0.0, 1.0])
        if second_magnitude < 1e-12
        else second / second_magnitude
    )
    return first, second


def project_to_plane(points, origin, first_axis, second_axis):
    """Project XYZ points to coordinates in a local two-dimensional plane."""
    origin = np.asarray(origin, dtype=float)
    return [
        (
            float(np.dot(np.asarray(point) - origin, first_axis)),
            float(np.dot(np.asarray(point) - origin, second_axis)),
        )
        for point in points
    ]


def remove_collinear_points(points, indices, relative_tolerance: float = 1e-12):
    """Drop nearly collinear 2D vertices while preserving original vertex indices."""
    if len(points) < 4:
        return points[:], indices[:]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    tolerance = relative_tolerance * scale
    cleaned_points = []
    cleaned_indices = []
    count = len(points)
    for index in range(count):
        previous = points[(index - 1) % count]
        current = points[index]
        following = points[(index + 1) % count]
        area = (
            (current[0] - previous[0]) * (following[1] - previous[1])
            - (current[1] - previous[1]) * (following[0] - previous[0])
        )
        if abs(area) <= tolerance:
            continue
        cleaned_points.append(current)
        cleaned_indices.append(indices[index])
    if len(cleaned_points) > 1 and cleaned_points[0] == cleaned_points[-1]:
        cleaned_points.pop()
        cleaned_indices.pop()
    if len(cleaned_points) >= 3:
        return cleaned_points, cleaned_indices

    # Degenerate polygons still need three distinct vertices for a fallback triangle.
    unique_points = []
    unique_indices = []
    seen = set()
    for point, index in zip(points, indices):
        key = round(point[0], 12), round(point[1], 12)
        if key in seen:
            continue
        seen.add(key)
        unique_points.append(point)
        unique_indices.append(index)
        if len(unique_points) == 3:
            break
    if len(unique_points) >= 3:
        return unique_points, unique_indices
    return points[:], indices[:]


def _signed_triangle_area(first, second, third) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _inside_triangle(point, first, second, third) -> bool:
    areas = (
        _signed_triangle_area(point, first, second),
        _signed_triangle_area(point, second, third),
        _signed_triangle_area(point, third, first),
    )
    return not (any(area < -1e-12 for area in areas) and any(area > 1e-12 for area in areas))


def triangulate_polygon(points, indices):
    """Triangulate a simple 2D polygon while preserving its original vertex indices."""
    count = len(points)
    if count < 3:
        return []
    if count == 3:
        return [(int(indices[0]), int(indices[1]), int(indices[2]))]

    points = list(points)
    indices = list(indices)
    signed_area = sum(
        points[index][0] * points[(index + 1) % count][1]
        - points[(index + 1) % count][0] * points[index][1]
        for index in range(count)
    )
    if signed_area < 0:
        points.reverse()
        indices.reverse()

    remaining = list(range(len(points)))
    triangles = []
    attempts = 0
    maximum_attempts = len(remaining) ** 2
    while len(remaining) > 3 and attempts < maximum_attempts:
        attempts += 1
        ear_found = False
        for position in range(len(remaining)):
            previous = remaining[(position - 1) % len(remaining)]
            current = remaining[position]
            following = remaining[(position + 1) % len(remaining)]
            triangle = points[previous], points[current], points[following]
            area = _signed_triangle_area(*triangle)
            if area <= 1e-12:
                continue
            if any(
                _inside_triangle(points[index], *triangle)
                for index in remaining
                if index not in (previous, current, following)
            ):
                continue
            triangles.append(
                (int(indices[previous]), int(indices[current]), int(indices[following]))
            )
            remaining.pop(position)
            ear_found = True
            break
        if not ear_found:
            triangles.extend(
                (
                    int(indices[remaining[0]]),
                    int(indices[remaining[index]]),
                    int(indices[remaining[index + 1]]),
                )
                for index in range(1, len(remaining) - 1)
            )
            remaining = []
    if len(remaining) == 3:
        triangles.append(tuple(int(indices[index]) for index in remaining))
    return triangles


def boundary_condition_name(boundary_condition) -> str:
    """Return a normalized Honeybee boundary-condition name."""
    if boundary_condition is None:
        return ""
    for attribute in ("name", "type", "display_name"):
        value = getattr(boundary_condition, attribute, None)
        if value:
            return str(value).lower()
    if getattr(boundary_condition, "boundary_condition_objects", None):
        return "surface"
    return ""


def surface_category(face) -> str:
    """Map a Honeybee face to the shared renderer category names."""
    face_type = (
        str(getattr(face.type, "name", "")).lower() if getattr(face, "type", None) else ""
    )
    boundary = boundary_condition_name(getattr(face, "boundary_condition", None))
    is_interior = "surface" in boundary or "adiabatic" in boundary
    if "airboundary" in face_type:
        return "air_wall"
    if "roof" in face_type or "roofceiling" in face_type:
        return "ceiling" if is_interior else "roof"
    if "floor" in face_type:
        return "exterior_floor" if "outdoor" in boundary or "ground" in boundary else "interior_floor"
    if "wall" in face_type:
        return "exterior_wall" if "outdoor" in boundary else "interior_wall"
    return "default"


def aperture_category(aperture, host_category: str) -> str:
    """Classify a Honeybee aperture as exterior or interior."""
    boundary = boundary_condition_name(getattr(aperture, "boundary_condition", None))
    if "outdoor" in boundary:
        return "aperture"
    if "surface" in boundary or getattr(
        getattr(aperture, "boundary_condition", None), "boundary_condition_objects", None
    ):
        return "interior_aperture"
    return "interior_aperture" if host_category == "interior_wall" else "aperture"


def door_category(door, host_category: str) -> str:
    """Classify a Honeybee door as exterior or interior."""
    boundary = boundary_condition_name(getattr(door, "boundary_condition", None))
    if "outdoor" in boundary:
        return "door"
    if "surface" in boundary or getattr(
        getattr(door, "boundary_condition", None), "boundary_condition_objects", None
    ):
        return "interior_door"
    return "interior_door" if host_category == "interior_wall" else "door"


def collect_model_geometry(
    model,
    *,
    extrude_eps: float,
    include_types=None,
    room_ids=None,
    want_wire: bool = False,
):
    """Collect Honeybee surfaces into renderer-neutral triangle and edge buckets."""
    categories: dict[str, dict] = {}
    structural_edges = []
    opening_edges = []
    include_types = set(include_types) if include_types is not None else None

    opening_keys = {"aperture", "interior_aperture", "door", "interior_door"}
    shade_keys = {"outdoor_shade", "indoor_shade", "shade_mesh", "shade"}

    def ensure_category(key: str) -> dict:
        return categories.setdefault(
            key, {"pts": [], "idx": {}, "I": [], "J": [], "K": []}
        )

    def add_wire_polygon(points, key: str) -> None:
        if not want_wire or key in shade_keys or len(points) < 2:
            return
        target = opening_edges if key in opening_keys else structural_edges
        target.extend(
            (point3d_tuple(points[index]), point3d_tuple(points[(index + 1) % len(points)]))
            for index in range(len(points))
        )

    def add_fan(category: dict, indices: list[int], *, reverse: bool = False) -> None:
        for offset in range(1, len(indices) - 1):
            triangle = (
                (indices[0], indices[offset + 1], indices[offset])
                if reverse
                else (indices[0], indices[offset], indices[offset + 1])
            )
            category["I"].append(triangle[0])
            category["J"].append(triangle[1])
            category["K"].append(triangle[2])

    def add_triangles(category: dict, indices: list[int]) -> None:
        vertices = [category["pts"][index] for index in indices]
        try:
            normal = polygon_normal(vertices)
            first_axis, second_axis = build_plane_basis(normal)
            projected = project_to_plane(vertices, np.asarray(vertices[0]), first_axis, second_axis)
            projected, source_indices = remove_collinear_points(projected, indices)
            triangles = triangulate_polygon(projected, source_indices)
        except (ArithmeticError, IndexError, TypeError, ValueError):
            triangles = []
        if not triangles:
            add_fan(category, indices)
            return
        for first, second, third in triangles:
            category["I"].append(first)
            category["J"].append(second)
            category["K"].append(third)

    def add_polygon(points, key: str) -> None:
        if include_types is not None and key not in include_types:
            return
        points = remove_consecutive_duplicates(points)
        if len(points) < 3:
            return
        category = ensure_category(key)
        indices = []
        for point in points:
            rounded = tuple(round(coordinate, 6) for coordinate in point3d_tuple(point))
            if rounded not in category["idx"]:
                category["idx"][rounded] = len(category["pts"])
                category["pts"].append(rounded)
            indices.append(category["idx"][rounded])
        add_triangles(category, indices)
        add_wire_polygon(points, key)

    def add_extruded_polygon(points, key: str) -> None:
        if include_types is not None and key not in include_types:
            return
        points = remove_consecutive_duplicates(points)
        if len(points) < 3:
            return
        normal = polygon_normal(points)
        category = ensure_category(key)
        front = []
        back = []
        for point in points:
            point = point3d_tuple(point)
            offset_points = (
                tuple(round(point[index] + normal[index] * extrude_eps, 6) for index in range(3)),
                tuple(round(point[index] - normal[index] * extrude_eps, 6) for index in range(3)),
            )
            for offset_point, indices in zip(offset_points, (front, back)):
                if offset_point not in category["idx"]:
                    category["idx"][offset_point] = len(category["pts"])
                    category["pts"].append(offset_point)
                indices.append(category["idx"][offset_point])

        add_triangles(category, front)
        add_fan(category, back, reverse=True)
        for index in range(len(front)):
            following = (index + 1) % len(front)
            first, second = front[index], front[following]
            third, fourth = back[following], back[index]
            category["I"].extend((first, first))
            category["J"].extend((second, third))
            category["K"].extend((third, fourth))
        add_wire_polygon(points, key)

    def geometry_points(geometry):
        if geometry is None:
            return None
        for attribute in ("boundary", "vertices", "points", "coordinates"):
            value = getattr(geometry, attribute, None)
            if value is not None:
                return value
        return None

    def add_geometry(geometry, key: str, *, extruded: bool = False) -> None:
        points = geometry_points(geometry)
        if points is not None:
            add = add_extruded_polygon if extruded else add_polygon
            add([point3d_tuple(point) for point in points], key)
        else:
            add_mesh_like(
                geometry,
                key,
                include_types=include_types,
                categories=categories,
            )

    def add_shades(owner) -> None:
        shades = chain(
            getattr(owner, "outdoor_shades", []) or (),
            getattr(owner, "indoor_shades", []) or (),
        )
        for shade in shades:
            key = "indoor_shade" if getattr(shade, "is_indoor", False) else "outdoor_shade"
            add_geometry(getattr(shade, "geometry", None), key)

    def add_openings(face, host_category: str) -> None:
        for aperture in getattr(face, "apertures", []) or []:
            add_shades(aperture)
            add_geometry(
                getattr(aperture, "geometry", None),
                aperture_category(aperture, host_category),
                extruded=True,
            )
        for door in getattr(face, "doors", []) or []:
            add_shades(door)
            add_geometry(
                getattr(door, "geometry", None),
                door_category(door, host_category),
                extruded=True,
            )

    selected_room_ids = set(room_ids) if room_ids else None
    rooms = [
        room
        for room in (getattr(model, "rooms", []) or [])
        if selected_room_ids is None or getattr(room, "identifier", None) in selected_room_ids
    ]
    for room in rooms:
        for face in getattr(room, "faces", []) or []:
            key = surface_category(face)
            add_geometry(getattr(face, "geometry", None), key)
            add_shades(face)
            add_openings(face, key)

    for shade in getattr(model, "orphaned_shades", []) or []:
        key = "indoor_shade" if getattr(shade, "is_indoor", False) else "outdoor_shade"
        add_geometry(getattr(shade, "geometry", None), key)
    for shade_mesh in getattr(model, "shade_meshes", []) or []:
        geometry = getattr(shade_mesh, "geometry", None)
        if geometry is None:
            geometry = getattr(shade_mesh, "mesh", None)
        add_geometry(geometry if geometry is not None else shade_mesh, "shade_mesh")

    for face in getattr(model, "orphaned_faces", []) or []:
        key = surface_category(face)
        add_geometry(getattr(face, "geometry", None), key)
        add_shades(face)
        add_openings(face, key)
    for aperture in getattr(model, "orphaned_apertures", []) or []:
        add_geometry(
            getattr(aperture, "geometry", None),
            aperture_category(aperture, "exterior_wall"),
            extruded=True,
        )
    for door in getattr(model, "orphaned_doors", []) or []:
        add_geometry(
            getattr(door, "geometry", None),
            door_category(door, "exterior_wall"),
            extruded=True,
        )

    return categories, structural_edges, opening_edges
