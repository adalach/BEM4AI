"""Where a polygon stops being convex.

Everything the decomposer does starts from a reflex vertex - a corner that
turns the wrong way relative to the rest of the ring. This module finds them,
and answers the question the search asks most often: is this polygon convex
yet. Results are memoised on a rounded ring key, because the search revisits
the same pieces constantly.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon

__all__ = [
    "convexity_mask",
    "is_convex_polygon",
    "polygon_turn_crosses",
    "raw_ring_key",
]


_CONVEX_CACHE: Dict[Tuple[Tuple[Tuple[float, float], ...], int, float, bool], bool] = {}


def _clean_exterior_coords(poly: Polygon, tol: float = 1e-12) -> np.ndarray:
    """Return exterior coordinates without consecutive duplicate vertices."""
    if not isinstance(poly, Polygon) or poly.is_empty:
        return np.empty((0, 2), dtype=float)

    raw = np.asarray(poly.exterior.coords[:-1], dtype=float)
    if len(raw) == 0:
        return raw

    cleaned = [raw[0]]
    for point in raw[1:]:
        if np.hypot(*(point - cleaned[-1])) <= tol:
            continue
        cleaned.append(point)
    if len(cleaned) > 1 and np.hypot(*(cleaned[0] - cleaned[-1])) <= tol:
        cleaned.pop()
    return np.asarray(cleaned, dtype=float)


def raw_ring_key(poly: Polygon, ndigits: int = 9) -> Tuple[Tuple[float, float], ...]:
    """Return a rounded cache key for the exterior ring of a polygon."""
    if not isinstance(poly, Polygon) or poly.is_empty:
        return tuple()
    coords = _clean_exterior_coords(poly)
    return tuple((round(float(x), ndigits), round(float(y), ndigits)) for x, y in coords)


def polygon_turn_crosses(poly: Polygon) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
    """
    Return `(coords, cross, orientation)` for the exterior ring of a polygon.

    `orientation` is `+1.0` for CCW rings and `-1.0` for CW rings.
    """
    if not isinstance(poly, Polygon) or poly.is_empty:
        return None, np.empty(0, dtype=float), 1.0

    coords = _clean_exterior_coords(poly)
    if len(coords) <= 2:
        return coords, np.empty(0, dtype=float), 1.0

    prev = np.roll(coords, 1, axis=0)
    nxt = np.roll(coords, -1, axis=0)
    v1 = coords - prev
    v2 = nxt - coords
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    area2 = float(
        np.sum(coords[:, 0] * np.roll(coords[:, 1], -1) - coords[:, 1] * np.roll(coords[:, 0], -1))
    )
    orientation = 1.0 if area2 >= 0.0 else -1.0
    return coords, cross, orientation


def convexity_mask(
    polys: Iterable[Polygon],
    tol: float = 0.0,
    *,
    holes_are_nonconvex: bool = True,
) -> np.ndarray:
    """
    Batch convexity check for polygons, grouped by exterior vertex count.

    ``tol`` is the cross-product magnitude below which a turn counts as
    straight rather than reflex. The pipeline uses the strict ``0.0``: a
    footprint is cleaned of near-collinear vertices before it gets here, so a
    turn that survives is a real one.

    Empty / non-polygon inputs are treated as convex so callers can pass mixed
    geometry lists without prefiltering. Invalid polygons and polygons with
    holes (when `holes_are_nonconvex=True`) are marked non-convex.
    """
    polys = list(polys)
    if not polys:
        return np.zeros(0, dtype=bool)

    mask = np.zeros(len(polys), dtype=bool)
    grouped = {}

    for idx, poly in enumerate(polys):
        if not isinstance(poly, Polygon) or poly.is_empty:
            mask[idx] = True
            continue
        if holes_are_nonconvex and len(poly.interiors) > 0:
            continue
        if not poly.is_valid:
            continue

        coords = _clean_exterior_coords(poly)
        if len(coords) <= 3:
            mask[idx] = True
            continue
        grouped.setdefault(len(coords), []).append((idx, coords))

    for items in grouped.values():
        indices = np.array([idx for idx, _ in items], dtype=int)
        batch = np.stack([coords for _, coords in items], axis=0)
        prev = np.roll(batch, 1, axis=1)
        nxt = np.roll(batch, -1, axis=1)
        v1 = batch - prev
        v2 = nxt - batch
        cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
        pos = cross > tol
        neg = cross < -tol
        mask[indices] = ~(pos.any(axis=1) & neg.any(axis=1))

    return mask


def is_convex_polygon(
    poly: Polygon,
    tol: float = 0.0,
    *,
    holes_are_nonconvex: bool = True,
    use_cache: bool = True,
) -> bool:
    """Return `True` if a polygon is convex under the shared strict turn-sign test."""
    if not isinstance(poly, Polygon) or poly.is_empty:
        return True

    cache_key = (
        raw_ring_key(poly),
        len(poly.interiors),
        round(float(tol), 12),
        bool(holes_are_nonconvex),
    )
    if use_cache and cache_key in _CONVEX_CACHE:
        return _CONVEX_CACHE[cache_key]

    value = bool(convexity_mask([poly], tol=tol, holes_are_nonconvex=holes_are_nonconvex)[0])
    if use_cache:
        _CONVEX_CACHE[cache_key] = value
    return value
