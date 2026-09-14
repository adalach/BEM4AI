"""
Fast hole removal for Shapely polygons.

Public API
----------
    holeless_polygons(geom, area_tol=1e-6, max_edges_per_hole=None,
                      max_selected_splits_per_hole=2)

Given a Polygon / MultiPolygon / GeometryCollection, returns a list of
Polygon objects **without interior rings** whose union approximates the
original geometry (respecting holes).

Algorithm
---------
For each polygon that has holes:

1. Pick one hole at a time.
2. Within that hole, test collinear split lines for its edges.
3. Keep only the valid cuts that:
   - produce >= 2 polygon pieces,
   - conserve area within `area_tol`,
   - strictly reduce the total number of holes.
4. Score those valid cuts by the total in-polygon overlap length of the line.
5. Keep the `max_selected_splits_per_hole` shortest valid cuts and apply them
   sequentially as long as they continue reducing hole count.

If no such split is found for that polygon:
   - fall back to triangulate() when no cut works, which always gives
     hole-less pieces.

Processing is iterative (using a work stack) and stops as soon as all
output polygons are holeless. The goal is to minimize the number of
splits on average rather than exploring many alternative variants.
"""

from __future__ import annotations

from typing import Any, List, Optional

import math
from shapely.geometry import (
    Polygon,
    MultiPolygon,
    GeometryCollection,
    LineString,
    base,
)
from shapely.ops import split, triangulate


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_holes(p: Polygon) -> bool:
    """Return True if polygon has one or more interior rings."""
    return isinstance(p, Polygon) and len(p.interiors) > 0


def _collinear_split_line(
    edge: tuple[tuple[float, float], tuple[float, float]],
    bounds: tuple[float, float, float, float],
    margin_factor: float = 1.2,
) -> Optional[LineString]:
    """
    Build a long line collinear with the given edge that crosses the whole polygon.

    Parameters
    ----------
    edge : ((x1, y1), (x2, y2))
        One edge of a hole ring.
    bounds : (minx, miny, maxx, maxy)
        Bounding box of the polygon being split.
    margin_factor : float, optional
        Multiplier for the line length relative to the bounding-box diagonal.

    Returns
    -------
    LineString or None
        A line that extends beyond the polygon in both directions, or None
        if the edge is degenerate.
    """
    (x1, y1), (x2, y2) = edge
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-12:
        # Degenerate edge
        return None

    ux = dx / length
    uy = dy / length

    # Midpoint of the edge as anchor point
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    minx, miny, maxx, maxy = bounds
    diag = math.hypot(maxx - minx, maxy - miny)
    if diag <= 0.0:
        # Pathological bounding box; fall back to some length
        diag = length

    half_len = 0.5 * diag * margin_factor

    p1 = (cx - ux * half_len, cy - uy * half_len)
    p2 = (cx + ux * half_len, cy + uy * half_len)

    return LineString([p1, p2])


def _hole_edge_split_candidates(
    poly: Polygon,
    area_tol: float,
    max_edges_per_hole: Optional[int] = None,
    max_selected_splits_per_hole: int = 2,
) -> list[dict[str, Any]]:
    """
    Build valid split candidates for each hole based on collinear hole-edge lines.

    A candidate is considered useful if:
        - it produces at least 2 polygon pieces,
        - the total area is conserved within `area_tol`,
        - the total number of interior rings across all pieces is strictly
          smaller than before the split.

    For each hole, the returned candidate list is already truncated to the
    `max_selected_splits_per_hole` shortest valid cuts, scored by the total
    in-polygon overlap length of the line.

    Parameters
    ----------
    poly : Polygon
        A polygon that has at least one interior ring.
    area_tol : float
        Absolute tolerance on area conservation.
    max_edges_per_hole : int or None, optional
        If given, at most this many edges per hole are tried.
    max_selected_splits_per_hole : int, optional
        Maximum number of shortest valid cuts to keep per hole.

    Returns
    -------
    list[dict]
        Candidate metadata dictionaries. Each entry contains `hole_idx`,
        `edge_idx`, `line`, `overlap_len`, `holes_after`, and `parts`.
    """
    if not _has_holes(poly):
        return []

    holes_before = len(poly.interiors)
    if holes_before == 0:
        return []

    bounds = poly.bounds
    area_before = poly.area
    all_candidates: list[dict[str, Any]] = []

    for hole_idx, interior in enumerate(poly.interiors):
        coords = list(interior.coords)
        if len(coords) < 2:
            continue

        edges = [(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]
        if max_edges_per_hole is not None and max_edges_per_hole > 0:
            edges = edges[:max_edges_per_hole]

        hole_candidates: list[dict[str, Any]] = []
        for edge_idx, edge in enumerate(edges):
            line = _collinear_split_line(edge, bounds)
            if line is None:
                continue

            try:
                parts = split(poly, line)
            except Exception:
                # Numerical issue or invalid geometry interaction; skip this edge.
                continue

            polys = [
                p for p in parts.geoms if isinstance(p, Polygon) and not p.is_empty
            ]
            if len(polys) < 2:
                continue

            area_after = sum(p.area for p in polys)
            if abs(area_after - area_before) > area_tol:
                continue

            holes_after = sum(len(p.interiors) for p in polys)
            if holes_after >= holes_before:
                continue

            overlap = poly.intersection(line)
            overlap_len = float(overlap.length) if not overlap.is_empty else math.inf
            hole_candidates.append(
                {
                    "hole_idx": hole_idx,
                    "edge_idx": edge_idx,
                    "line": line,
                    "overlap_len": overlap_len,
                    "holes_after": holes_after,
                    "parts": polys,
                }
            )

        hole_candidates.sort(key=lambda d: (d["overlap_len"], d["edge_idx"]))
        all_candidates.extend(hole_candidates[:max_selected_splits_per_hole])

    return all_candidates


def _split_once_along_hole_edge(
    poly: Polygon,
    area_tol: float,
    max_edges_per_hole: Optional[int] = None,
    max_selected_splits_per_hole: int = 2,
) -> Optional[List[Polygon]]:
    """
    Attempt a single useful hole-removal step by applying up to the two
    shortest valid collinear hole-edge cuts per hole.

    The selected cuts are computed on the original polygon, then applied
    sequentially. A cut is kept only if it still reduces the total hole
    count of the current piece it is applied to.
    """
    if not _has_holes(poly):
        return None

    candidates = _hole_edge_split_candidates(
        poly,
        area_tol=area_tol,
        max_edges_per_hole=max_edges_per_hole,
        max_selected_splits_per_hole=max_selected_splits_per_hole,
    )
    if not candidates:
        return None

    pieces: list[Polygon] = [poly]
    any_used = False

    for candidate in candidates:
        new_pieces: list[Polygon] = []
        used_here = False

        for piece in pieces:
            if not _has_holes(piece):
                new_pieces.append(piece)
                continue

            try:
                split_result = split(piece, candidate["line"])
            except Exception:
                new_pieces.append(piece)
                continue

            polys = [
                p for p in split_result.geoms if isinstance(p, Polygon) and not p.is_empty
            ]
            if len(polys) < 2:
                new_pieces.append(piece)
                continue

            area_after = sum(p.area for p in polys)
            if abs(area_after - piece.area) > area_tol:
                new_pieces.append(piece)
                continue

            holes_before = len(piece.interiors)
            holes_after = sum(len(p.interiors) for p in polys)
            if holes_after >= holes_before:
                new_pieces.append(piece)
                continue

            new_pieces.extend(polys)
            used_here = True
            any_used = True

        pieces = new_pieces
        if not any(_has_holes(piece) for piece in pieces):
            break

        if not used_here:
            continue

    return pieces if any_used else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def holeless_polygons(
    geom: base.BaseGeometry,
    area_tol: float = 1e-6,
    max_edges_per_hole: Optional[int] = None,
    max_selected_splits_per_hole: int = 2,
) -> List[Polygon]:
    """
    Decompose a geometry into a list of Polygon objects without holes.

    For each polygon component with interior rings:
      - Build valid collinear hole-edge cuts for each hole.
      - Keep up to `max_selected_splits_per_hole` shortest valid cuts per hole.
      - Apply those cuts sequentially while they keep reducing hole count.
      - Recurse on the resulting pieces until no holes remain.

    When no such split can be found, the polygon is triangulated instead. The
    triangles are uglier than a clean cut, but they are certainly hole-less,
    which is the contract this function has to keep.

    Parameters
    ----------
    geom : shapely geometry
        Input geometry (Polygon, MultiPolygon, or GeometryCollection). Other
        geometry types are ignored.
    area_tol : float, optional
        Absolute tolerance used when checking area conservation after splits.
    max_edges_per_hole : int or None, optional
        Maximum number of edges per hole to try for split candidates.
    max_selected_splits_per_hole : int, optional
        Maximum number of shortest valid collinear cuts to keep per hole.

    Returns
    -------
    list[Polygon]
        List of polygons without interior rings. The union of these polygons
        approximates the original polygonal area (excluding holes).
    """
    out: List[Polygon] = []

    if geom is None or geom.is_empty:
        return out

    # Handle MultiPolygon and GeometryCollection by iterating components
    if isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            out.extend(
                holeless_polygons(
                    p,
                    area_tol=area_tol,
                    max_edges_per_hole=max_edges_per_hole,
                    max_selected_splits_per_hole=max_selected_splits_per_hole,
                )
            )
        return out

    if isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            out.extend(
                holeless_polygons(
                    g,
                    area_tol=area_tol,
                    max_edges_per_hole=max_edges_per_hole,
                    max_selected_splits_per_hole=max_selected_splits_per_hole,
                )
            )
        return out

    if not isinstance(geom, Polygon):
        # Non-polygonal geometries are ignored; only polygon area is decomposed.
        return out

    # Work stack for polygons needing hole processing
    work: List[Polygon] = [geom]

    while work:
        poly = work.pop()

        if not _has_holes(poly):
            # Already holeless → final result
            out.append(poly)
            continue

        # Try a single useful split along a hole edge
        pieces = _split_once_along_hole_edge(
            poly,
            area_tol=area_tol,
            max_edges_per_hole=max_edges_per_hole,
            max_selected_splits_per_hole=max_selected_splits_per_hole,
        )

        if pieces is not None:
            # We found a split that reduces holes; process the pieces further.
            work.extend(pieces)
            continue

        # No edge-based split worked. Triangulating is the last resort: the
        # pieces are uglier than a clean cut but they are certainly hole-less,
        # which is the contract this function has to keep.
        for triangle in triangulate(poly):
            if isinstance(triangle, Polygon) and not triangle.is_empty:
                out.append(triangle)

    return out
