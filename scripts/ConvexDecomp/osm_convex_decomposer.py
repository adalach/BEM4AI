"""
Convex decomposition of 2D polygons (e.g. OSM building parts) using
recursive splits along extended edges at reflex vertices.

Design:
- Polygon-centric: this module does NOT know about buildings, perimeters, etc.
- You call `decompose_polygon_with_stats(poly, cfg)` once per Polygon.
- Decomposition is modelled as a search in "state space" where a state is
  a list of current pieces [P1, P2, ...]. We refine states by splitting
  one non-convex piece at a time.

Key ideas:
- We explore the state space with a persistent adaptive beam search.
- Each state is canonicalised to a key (multiset of polygon keys) so that
  if two different split sequences lead to the same configuration, we
  process it only once (state dedup).
- At each refinement step we choose ONE non-convex piece and split it
  at a small number of promising reflex vertices.
- Reflex vertices are ordered by how close their *interior* angle is to
  270 degrees (a strong "quarter-turn" reflex). Equivalently, we sort by
  concavity = interior_angle - 180° being close to 90°.
- We cap the number of distinct final variants at max_variants_per_polygon.

Public API:
    from scripts.ConvexDecomp.osm_convex_decomposer import (
        ConvexDecompositionConfig,
        extract_polygons,
        decompose_polygon_with_stats,
        decompose_polygon_best,
    )
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import split

from .convex_decomposition import (
    AdaptiveSearchConfig,
    DecompositionSearchConfig,
    ExpandedState,
    StateExpansionResult,
    run_polygon_state_search,
    score_polygon_partition,
)
from helpers.utils import polygon_parts as extract_polygons

from .polygon_convexity import is_convex_polygon as _shared_is_convex_polygon

__all__ = [
    "ConvexDecompositionConfig",
    "extract_polygons",
    "decompose_polygon_with_stats",
    "decompose_polygon_best",
]

# ---------------------------------------------------------------------------
# Global tuning constants
# ---------------------------------------------------------------------------

# Minimum area for any polygon piece to be kept during splitting.
# Raising this prunes tiny slivers early (faster, but may lose very skinny zones).
MIN_PART_AREA: float = 0.5


def _default_osm_adaptive_search() -> AdaptiveSearchConfig:
    # OSM perimeter remainders often need many sequential splits; keep the
    # beam narrow by default and spend budget on depth first.
    return AdaptiveSearchConfig(
        enabled=True,
        initial_depth=4,
        initial_width=3,
        max_depth=24,
        max_width=4,
        depth_step=1,
        width_step=1,
    )


# ---------------------------------------------------------------------------
# Configuration dataclass ("knobs")
# ---------------------------------------------------------------------------

@dataclass
class ConvexDecompositionConfig:
    """
    Parameters controlling the convex decomposition search.

    Attributes
    ----------
    search_depth : int
        Maximum recursion depth (max number of split steps) for exploring
        the state space.
    search_width : int
        Maximum number of open states kept in the active beam before the
        remaining scored states are parked for later widening.
    max_variants_per_polygon : int
        Hard cap on the number of distinct fully convex decomposition variants
        we are willing to keep for a single polygon. Only states where all
        pieces are convex count toward this cap.
    max_failed_states : Optional[int]
        Hard cap on the number of unsuccessful terminal states (dead-ends
        where at least one piece is non-convex and we either hit the depth
        limit or have no further splits). If None, no explicit limit is
        applied to failed states.
    min_area_m2 : float
        Area threshold for "too small" zones in the final result stats
        (used only for has_small_zone).
    weight_parts : float
        Weight for penalising many parts in the scoring function.
    weight_compactness : float
        Weight for rewarding compact shapes in the scoring function.
    min_concavity_deg : float
        Minimum concavity (in degrees) for a vertex to be treated as a
        meaningful reflex. Smaller concavities are considered "almost straight".
    max_reflex_per_node : Optional[int]
        Extra cap on how many reflex vertices per polygon we consider at a node.
        If None, it becomes ``max(search_width * 2, 8)``.
    length_factor : float
        How far to extend the splitting line relative to the polygon's diagonal.
    adaptive_search : AdaptiveSearchConfig
        Optional outer retry schedule that starts from a smaller search budget
        and increases depth first, then width, until the first successful
        parameter setting is found.
    track_time : bool
        If True, decompose_polygon_with_stats measures wall-clock time per
        polygon and adds a "time_seconds" field to the result dict.
    """

    search_depth: int = 24
    search_width: int = 4
    max_variants_per_polygon: int = 64
    max_failed_states: Optional[int] = None
    min_area_m2: float = 2.0
    weight_parts: float = 1.0
    weight_compactness: float = 20.0
    min_concavity_deg: float = 2.0
    max_reflex_per_node: Optional[int] = None
    length_factor: float = 10.0
    track_time: bool = False
    use_bisector_dead_end_fallback: bool = True
    cut_snap_vertex_tolerance_m: float = 0.1
    adaptive_search: AdaptiveSearchConfig = field(default_factory=_default_osm_adaptive_search)

    def __post_init__(self) -> None:
        if self.max_reflex_per_node is None:
            self.max_reflex_per_node = max(int(self.search_width) * 2, 8)



# ---------------------------------------------------------------------------
# Small caches (convexity + polygon key for states)
# ---------------------------------------------------------------------------

# Cache canonical polygon keys for states keyed by raw ring key
_POLY_KEY_CACHE: Dict[Tuple[Tuple[float, float], ...], Tuple[Tuple[float, float], ...]] = {}


# ---------------------------------------------------------------------------
# Basic geometry helpers
# ---------------------------------------------------------------------------

def _polygon_orientation(coords: Sequence[Tuple[float, float]]) -> bool:
    """
    Return True if ring is CCW, False if CW.
    """
    area2 = 0.0
    n = len(coords)
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        area2 += x1 * y2 - y1 * x2
    return area2 > 0.0


def _raw_ring_key(poly: Polygon, ndigits: int = 6) -> Tuple[Tuple[float, float], ...]:
    """
    Raw key for a polygon's exterior ring (before normalisation).
    Used only for caching.
    """
    if not isinstance(poly, Polygon) or poly.is_empty:
        return tuple()
    coords = list(poly.exterior.coords)
    return tuple((round(x, ndigits), round(y, ndigits)) for (x, y) in coords)


def is_convex_polygon(poly: Polygon, tol: float = 0.0) -> bool:
    """
    Thin wrapper around the shared strict convexity helper.
    """
    return _shared_is_convex_polygon(poly, tol=tol)


def normalize_polygon(
    poly: Polygon,
    snap_tol: float = 1e-6,
    collinear_tol: float = 1e-10,
) -> Polygon:
    """
    Snap very close vertices and remove almost-collinear vertices on the exterior ring.
    Interiors (holes) are ignored here; this cleaner is aimed at perimeter-style shapes.

    Returns a cleaned Polygon; if cleaning fails or changes the polygon too much,
    returns the original polygon.
    """
    if not isinstance(poly, Polygon) or poly.is_empty:
        return poly

    coords = list(poly.exterior.coords)
    if len(coords) <= 4:  # triangle + closing coord
        return poly

    # 1) Snap very close vertices on the original ring
    snapped: List[Tuple[float, float]] = []
    for x, y in coords:
        if not snapped:
            snapped.append((x, y))
            continue
        x_prev, y_prev = snapped[-1]
        dx, dy = x - x_prev, y - y_prev
        if dx * dx + dy * dy < snap_tol * snap_tol:
            continue
        snapped.append((x, y))

    if len(snapped) < 3:
        return poly
    if snapped[0] != snapped[-1]:
        snapped.append(snapped[0])

    # 2) Remove almost-collinear interior vertices
    def _is_collinear(p0, p1, p2) -> bool:
        (x0, y0), (x1, y1), (x2, y2) = p0, p1, p2
        v1x, v1y = x1 - x0, y1 - y0
        v2x, v2y = x2 - x1, y2 - y1
        cross = v1x * v2y - v1y * v2x
        return abs(cross) < collinear_tol

    ring = snapped[:-1]
    n = len(ring)
    if n < 3:
        return poly

    cleaned: List[Tuple[float, float]] = []
    for i in range(n):
        p_prev = ring[(i - 1) % n]
        p_curr = ring[i]
        p_next = ring[(i + 1) % n]
        if _is_collinear(p_prev, p_curr, p_next):
            continue
        cleaned.append(p_curr)

    if len(cleaned) < 3:
        return poly

    # 3) Merge very close neighbours again
    dedup: List[Tuple[float, float]] = []
    for x, y in cleaned:
        if not dedup:
            dedup.append((x, y))
            continue
        x_prev, y_prev = dedup[-1]
        dx, dy = x - x_prev, y - y_prev
        if dx * dx + dy * dy < snap_tol * snap_tol:
            continue
        dedup.append((x, y))

    if len(dedup) < 3:
        return poly
    if dedup[0] != dedup[-1]:
        dedup.append(dedup[0])

    try:
        cleaned_poly = Polygon(dedup)
        if not cleaned_poly.is_valid or cleaned_poly.is_empty:
            return poly
        if poly.area > 0:
            rel_err = abs(cleaned_poly.area - poly.area) / poly.area
            if rel_err > 1e-4:
                return poly
        return cleaned_poly
    except Exception:
        return poly


def _ring_canonical_key(
    coords: Sequence[Tuple[float, float]],
    ndigits: int = 3,
) -> Tuple[Tuple[float, float], ...]:
    """
    Canonical key for a ring:
    - round coords
    - ignore closing point
    - invariant to rotation and orientation
    """
    pts = [(round(x, ndigits), round(y, ndigits)) for (x, y) in coords[:-1]]
    m = len(pts)
    if m == 0:
        return tuple()

    def rotations(seq):
        for i in range(len(seq)):
            yield tuple(seq[i:] + seq[:i])

    seq_fwd = pts
    seq_rev = list(reversed(pts))

    candidates = list(rotations(seq_fwd)) + list(rotations(seq_rev))
    return min(candidates)


def _poly_key_for_state(poly: Polygon, ndigits: int = 3) -> Tuple[Tuple[float, float], ...]:
    """
    Canonical key for a polygon when used inside a state.

    - Normalise polygon (snap + drop collinear vertices).
    - Drop very small polygons (area < MIN_PART_AREA).
    - Canonicalise exterior ring w.r.t. rotation/orientation.
    - Results are cached by the raw ring key for speed.
    """
    if not isinstance(poly, Polygon) or poly.is_empty:
        return tuple()
    if poly.area < MIN_PART_AREA:
        return tuple()

    raw_key = _raw_ring_key(poly)
    if raw_key in _POLY_KEY_CACHE:
        return _POLY_KEY_CACHE[raw_key]

    norm = normalize_polygon(poly)
    if norm.is_empty or norm.area < MIN_PART_AREA:
        key: Tuple[Tuple[float, float], ...] = tuple()
    else:
        coords = list(norm.exterior.coords)
        key = _ring_canonical_key(coords, ndigits=ndigits)

    _POLY_KEY_CACHE[raw_key] = key
    return key


# ---------------------------------------------------------------------------
# Reflex detection (ordered by "closeness" to 90° concavity)
# ---------------------------------------------------------------------------

def find_reflex_vertices_scored(
    poly: Polygon,
    cfg: ConvexDecompositionConfig,
    tol: float = 1e-9,
) -> List[Tuple[int, float]]:
    """
    Return list of (vertex_index, concavity_rad) for reflex vertices on the exterior ring.

    concavity_rad = π - exterior_angle (0 for straight, larger = more reflex).

    Heuristic:
    - Only keep reflex vertices with concavity ≥ cfg.min_concavity_deg,
      where concavity_deg = interior_angle_deg - 180°.
    - Order them by how close their interior angle is to 270°:
      equivalently, we sort by |concavity_deg - 90°| ascending.
      Vertices whose interior angle is closest to 270° are tried first.
    """
    if not isinstance(poly, Polygon) or poly.is_empty:
        return []

    coords = list(poly.exterior.coords)[:-1]
    n = len(coords)
    if n <= 3:
        return []

    ccw = _polygon_orientation(coords)
    reflex_scored: List[Tuple[int, float]] = []

    min_concavity_rad = math.radians(cfg.min_concavity_deg)
    target_concavity_rad = math.pi / 2.0  # 90 degrees

    for i in range(n):
        x0, y0 = coords[i - 1]
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]

        v1x, v1y = x1 - x0, y1 - y0
        v2x, v2y = x2 - x1, y2 - y1
        cross = v1x * v2y - v1y * v2x

        if abs(cross) <= tol:
            continue

        is_reflex = (ccw and cross < -tol) or ((not ccw) and cross > tol)
        if not is_reflex:
            continue

        # vectors pointing into the vertex
        v_in1 = (x0 - x1, y0 - y1)
        v_in2 = (x2 - x1, y2 - y1)
        len1 = math.hypot(*v_in1)
        len2 = math.hypot(*v_in2)
        if len1 <= tol or len2 <= tol:
            continue

        u1 = (v_in1[0] / len1, v_in1[1] / len1)
        u2 = (v_in2[0] / len2, v_in2[1] / len2)
        dot = max(-1.0, min(1.0, u1[0] * u2[0] + u1[1] * u2[1]))
        exterior_ang = math.acos(dot)  # in (0, π)
        concavity = math.pi - exterior_ang  # 0 = straight, larger = more reflex

        if concavity < min_concavity_rad:
            continue

        reflex_scored.append((i, concavity))

    # Order by "closeness" to 90 degrees concavity
    reflex_scored.sort(key=lambda t: abs(t[1] - target_concavity_rad))

    max_r = int(cfg.max_reflex_per_node) if cfg.max_reflex_per_node is not None else max(int(cfg.search_width), 1)

    if len(reflex_scored) > max_r:
        reflex_scored = reflex_scored[:max_r]

    return reflex_scored


# ---------------------------------------------------------------------------
# Split helper (extends an incident edge and calls shapely.split)
# ---------------------------------------------------------------------------

def _split_polygon_with_cutter(
    poly: Polygon,
    cutter: BaseGeometry,
    tol_area: float = 1e-12,
) -> List[Polygon]:
    """
    Split `poly` with `cutter` and return cleaned polygon parts.

    If no meaningful split occurs, returns `[poly]`.
    """
    if not isinstance(poly, Polygon) or poly.is_empty:
        return [poly]

    try:
        res = split(poly, cutter)
    except Exception:
        return [poly]

    parts_raw = [
        g for g in getattr(res, "geoms", [res])
        if isinstance(g, Polygon) and (not g.is_empty) and (g.area > tol_area)
    ]
    if len(parts_raw) <= 1:
        return [poly]

    parts: List[Polygon] = []
    for p in parts_raw:
        q = normalize_polygon(p)
        if isinstance(q, Polygon) and (not q.is_empty) and q.area >= MIN_PART_AREA:
            parts.append(q)

    if len(parts) <= 1:
        return [poly]

    sum_area = sum(p.area for p in parts)
    if poly.area > 0 and sum_area < 0.9 * poly.area:
        return [poly]

    return parts


def _try_snap_cut_to_existing_vertex(
    poly: Polygon,
    coords: Sequence[Tuple[float, float]],
    origin_idx: int,
    ux: float,
    uy: float,
    snap_tol: float,
    exclude_indices: set[int],
    tol_area: float = 1e-12,
) -> Optional[List[Polygon]]:
    """
    If the proposed cut line runs almost through an existing vertex, try using
    the exact vertex-to-vertex segment instead of the near-miss cut.
    """
    if snap_tol <= 0.0:
        return None

    from shapely.geometry import LineString

    x1, y1 = coords[origin_idx]
    candidates: List[Tuple[float, float, int]] = []

    for j, (xj, yj) in enumerate(coords):
        if j in exclude_indices:
            continue
        perp_dist = abs((xj - x1) * uy - (yj - y1) * ux)
        if perp_dist > snap_tol:
            continue
        seg_len = math.hypot(xj - x1, yj - y1)
        if seg_len <= tol_area:
            continue
        candidates.append((perp_dist, seg_len, j))

    candidates.sort()

    for _, _, j in candidates:
        cutter = LineString([coords[origin_idx], coords[j]])
        parts = _split_polygon_with_cutter(poly, cutter, tol_area=tol_area)
        if len(parts) > 1:
            return parts

    return None

def cut_polygon_at_vertex(
    poly: Polygon,
    vidx: int,
    cfg: ConvexDecompositionConfig,
    use_prev: bool = True,
    tol_area: float = 1e-12,
) -> List[Polygon]:
    """
    Extend an incident edge at vertex `vidx` to a long line and split the polygon.

    Returns the resulting polygon parts. If no real split occurs or results are
    degenerate, returns [poly].

    Microscopic parts (area < MIN_PART_AREA) are dropped.
    """
    if not isinstance(poly, Polygon) or poly.is_empty:
        return [poly]

    coords = list(poly.exterior.coords)[:-1]
    n = len(coords)
    if n <= 3 or vidx < 0 or vidx >= n:
        return [poly]

    x1, y1 = coords[vidx]

    if use_prev:
        x0, y0 = coords[vidx - 1]
        dx, dy = x1 - x0, y1 - y0
    else:
        x2, y2 = coords[(vidx + 1) % n]
        dx, dy = x2 - x1, y2 - y1

    length = math.hypot(dx, dy)
    if length <= tol_area:
        return [poly]

    ux, uy = dx / length, dy / length
    snap_parts = _try_snap_cut_to_existing_vertex(
        poly,
        coords,
        origin_idx=vidx,
        ux=ux,
        uy=uy,
        snap_tol=cfg.cut_snap_vertex_tolerance_m,
        exclude_indices={
            vidx,
            (vidx - 1) % n if use_prev else (vidx + 1) % n,
        },
        tol_area=tol_area,
    )
    if snap_parts is not None:
        return snap_parts

    from shapely.geometry import LineString

    minx, miny, maxx, maxy = poly.bounds
    diag = math.hypot(maxx - minx, maxy - miny)
    L = max(diag * cfg.length_factor, diag + 1.0)

    p_start = (x1 - ux * L, y1 - uy * L)
    p_end = (x1 + ux * L, y1 + uy * L)
    return _split_polygon_with_cutter(poly, LineString([p_start, p_end]), tol_area=tol_area)


def cut_polygon_at_vertex_bisector(
    poly: Polygon,
    vidx: int,
    cfg: ConvexDecompositionConfig,
    tol_area: float = 1e-12,
) -> List[Polygon]:
    """
    Split a polygon with the angle-bisector line through vertex `vidx`.

    This is a fallback for cases where extending an incident edge either
    creates slivers or fails to progress. The line orientation is based on
    the sum of the unit vectors pointing from the vertex to its adjacent
    vertices, which defines the bisector line up to sign.
    """
    if not isinstance(poly, Polygon) or poly.is_empty:
        return [poly]

    coords = list(poly.exterior.coords)[:-1]
    n = len(coords)
    if n <= 3 or vidx < 0 or vidx >= n:
        return [poly]

    x1, y1 = coords[vidx]
    x0, y0 = coords[vidx - 1]
    x2, y2 = coords[(vidx + 1) % n]

    vx0, vy0 = x0 - x1, y0 - y1
    vx2, vy2 = x2 - x1, y2 - y1
    len0 = math.hypot(vx0, vy0)
    len2 = math.hypot(vx2, vy2)
    if len0 <= tol_area or len2 <= tol_area:
        return [poly]

    ux0, uy0 = vx0 / len0, vy0 / len0
    ux2, uy2 = vx2 / len2, vy2 / len2
    dx, dy = ux0 + ux2, uy0 + uy2
    if math.hypot(dx, dy) <= tol_area:
        dx, dy = -(uy2 - uy0), (ux2 - ux0)

    length = math.hypot(dx, dy)
    if length <= tol_area:
        return [poly]

    ux, uy = dx / length, dy / length
    snap_parts = _try_snap_cut_to_existing_vertex(
        poly,
        coords,
        origin_idx=vidx,
        ux=ux,
        uy=uy,
        snap_tol=cfg.cut_snap_vertex_tolerance_m,
        exclude_indices={vidx, (vidx - 1) % n, (vidx + 1) % n},
        tol_area=tol_area,
    )
    if snap_parts is not None:
        return snap_parts

    from shapely.geometry import LineString

    minx, miny, maxx, maxy = poly.bounds
    diag = math.hypot(maxx - minx, maxy - miny)
    L = max(diag * cfg.length_factor, diag + 1.0)

    line = LineString([(x1 - ux * L, y1 - uy * L), (x1 + ux * L, y1 + uy * L)])
    return _split_polygon_with_cutter(poly, line, tol_area=tol_area)


def cut_polygon_at_vertex_pair(
    poly: Polygon,
    start_idx: int,
    end_idx: int,
    tol_area: float = 1e-12,
) -> List[Polygon]:
    """
    Split a polygon with the exact diagonal between two existing exterior vertices.
    """
    from shapely.geometry import LineString

    if not isinstance(poly, Polygon) or poly.is_empty:
        return [poly]

    coords = list(poly.exterior.coords)[:-1]
    n = len(coords)
    if n <= 3:
        return [poly]
    if start_idx < 0 or end_idx < 0 or start_idx >= n or end_idx >= n:
        return [poly]
    if start_idx == end_idx:
        return [poly]
    if abs(start_idx - end_idx) == 1 or {start_idx, end_idx} == {0, n - 1}:
        return [poly]

    return _split_polygon_with_cutter(
        poly,
        LineString([coords[start_idx], coords[end_idx]]),
        tol_area=tol_area,
    )


# ---------------------------------------------------------------------------
# State representation and scoring
# ---------------------------------------------------------------------------

def _state_key(pieces: List[Polygon]) -> Tuple[Tuple[Tuple[float, float], ...], ...]:
    """
    Canonical key for a state (unordered list of polygon pieces).

    - Each piece → canonical polygon key.
    - State key = sorted tuple of polygon keys.

    If any piece fails to produce a key, we consider the state invalid
    and return an empty tuple.
    """
    keys: List[Tuple[float, float]] = []
    for p in pieces:
        k = _poly_key_for_state(p)
        if not k:
            return tuple()
        keys.append(k)
    # sort keys for order-invariance
    return tuple(sorted(keys))


def _score_polys(
    polys: List[Polygon],
    w_parts: float,
    w_compact: float,
) -> float:
    return score_polygon_partition(
        polys,
        weight_parts=w_parts,
        weight_compactness=w_compact,
    )


# ---------------------------------------------------------------------------
# Core state-space search for one polygon
# ---------------------------------------------------------------------------

def _enumerate_states_for_polygon(
    poly: Polygon,
    cfg: ConvexDecompositionConfig,
) -> Dict[str, Any]:
    """
    Explore the decomposition state space for a single polygon.

    Returns a list of variants; each variant is a list[Polygon] of parts,
    and **every returned variant is fully convex** (all pieces convex).

    Performance control:
    - Depth limit: cfg.search_depth.
    - A 'terminal' state is either:
        * a fully convex configuration (success), or
        * a non-convex configuration where we cannot or do not want to split
          further (no reflex vertices or depth limit reached).
    - We maintain two counters:
        * n_success: number of fully convex terminal states (variants).
        * n_failed:  number of non-convex terminal states (dead-ends).
    - We stop exploring when:
        * n_success >= cfg.max_variants_per_polygon, or
        * n_failed  >= cfg.max_failed_states (if not None).

    So:
    - Only fully convex terminal states are returned as variants and are used
      in scoring.
    - Unsuccessful terminal states are still "seen" and used to stop the
      search, but they are not returned or counted as variants.
    """
    if not isinstance(poly, Polygon) or poly.is_empty:
        return {
            "variants": [],
            "n_success": 0,
            "n_failed": 0,
            "n_depth_limit_dead_ends": 0,
            "n_no_reflex_dead_ends": 0,
            "n_min_area_dead_ends": 0,
            "n_states_seen": 0,
            "terminated_by": "invalid_or_empty_input",
        }

    # Normalise root once
    root = normalize_polygon(poly)
    if root.is_empty or root.area < MIN_PART_AREA:
        return {
            "variants": [],
            "n_success": 0,
            "n_failed": 0,
            "n_depth_limit_dead_ends": 0,
            "n_no_reflex_dead_ends": 0,
            "n_min_area_dead_ends": 0,
            "n_states_seen": 0,
            "terminated_by": "invalid_root",
        }

    def _expand_state(pieces: List[Polygon]) -> StateExpansionResult:
        result = StateExpansionResult()
        children_by_key: Dict[Any, ExpandedState] = {}
        best_idx: Optional[int] = None
        diagonal_fallback_idx: Optional[int] = None
        best_reflexes: List[Tuple[int, float]] = []
        best_area = -1.0
        diagonal_fallback_area = -1.0

        for idx, candidate in enumerate(pieces):
            if not isinstance(candidate, Polygon) or candidate.is_empty or is_convex_polygon(candidate):
                continue
            if candidate.area > diagonal_fallback_area:
                diagonal_fallback_idx = idx
                diagonal_fallback_area = candidate.area
            reflexes = find_reflex_vertices_scored(candidate, cfg)
            if not reflexes:
                continue
            if candidate.area > best_area:
                best_area = candidate.area
                best_idx = idx
                best_reflexes = reflexes

        def _register_child(
            target_idx: int,
            parts: Sequence[Polygon],
            *,
            fallback_label: str | None = None,
        ) -> None:
            new_pieces = pieces[:target_idx] + list(parts) + pieces[target_idx + 1 :]
            child_key = _state_key(new_pieces)
            if not child_key:
                return
            child = ExpandedState(tuple(new_pieces), (fallback_label,) if fallback_label else ())
            existing = children_by_key.get(child_key)
            if existing is None or (existing.fallback_labels and not child.fallback_labels):
                children_by_key[child_key] = child

        if best_idx is None or not best_reflexes:
            min_area_pruned = False
            if diagonal_fallback_idx is not None:
                target_poly = pieces[diagonal_fallback_idx]
                coords = list(target_poly.exterior.coords)[:-1]
                diagonal_candidates: List[Tuple[bool, float, List[Polygon]]] = []

                for i in range(len(coords)):
                    for j in range(i + 1, len(coords)):
                        parts = cut_polygon_at_vertex_pair(target_poly, i, j)
                        if len(parts) != 2:
                            continue
                        if any(part.area < cfg.min_area_m2 for part in parts):
                            min_area_pruned = True
                            continue
                        all_convex_parts = all(is_convex_polygon(part) for part in parts)
                        score = _score_polys(parts, cfg.weight_parts, cfg.weight_compactness)
                        diagonal_candidates.append((all_convex_parts, score, parts))

                if diagonal_candidates:
                    diagonal_candidates.sort(key=lambda item: (not item[0], item[1]))
                    max_diagonal_splits = max(2, cfg.search_width * 2)
                    for _, _, parts in diagonal_candidates[:max_diagonal_splits]:
                        _register_child(diagonal_fallback_idx, parts, fallback_label="diagonal_no_reflex")
                    if children_by_key:
                        result.children.extend(children_by_key.values())
                        return result

            result.dead_end_counts.update(["min_area_dead_end" if min_area_pruned else "no_reflex_dead_end"])
            return result

        target_poly = pieces[best_idx]
        min_area_pruned = False

        for vidx, _ in best_reflexes:
            for use_prev in (True, False):
                result.metrics.update(["search_split_attempts"])
                parts = cut_polygon_at_vertex(target_poly, vidx, cfg, use_prev=use_prev)
                if len(parts) != 2:
                    continue
                if any(part.area < cfg.min_area_m2 for part in parts):
                    min_area_pruned = True
                    continue

                result.metrics.update(["search_successful_splits"])
                _register_child(best_idx, parts)

        if children_by_key:
            result.children.extend(children_by_key.values())
            return result

        if cfg.use_bisector_dead_end_fallback:
            result.metrics.update(["bisector_fallback_attempts"])
            bisector_child_count_before = len(children_by_key)
            for vidx, _ in best_reflexes:
                parts = cut_polygon_at_vertex_bisector(target_poly, vidx, cfg)
                if len(parts) != 2:
                    continue
                if any(part.area < cfg.min_area_m2 for part in parts):
                    min_area_pruned = True
                    continue

                _register_child(best_idx, parts, fallback_label="bisector_dead_end")
            if len(children_by_key) > bisector_child_count_before:
                result.metrics.update(["bisector_fallback_successes"])

        if children_by_key:
            result.children.extend(children_by_key.values())
            return result

        result.dead_end_counts.update(["min_area_dead_end" if min_area_pruned else "no_reflex_dead_end"])
        return result

    search = run_polygon_state_search(
        [root],
        cfg=DecompositionSearchConfig(
            max_depth=cfg.search_depth,
            max_variants=cfg.max_variants_per_polygon,
            max_failed_states=cfg.max_failed_states,
        ),
        is_convex_piece=is_convex_polygon,
        state_key_fn=_state_key,
        expand_state_fn=_expand_state,
        adaptive_cfg=cfg.adaptive_search,
        default_width=cfg.search_width,
        state_score_fn=lambda pieces: _score_polys(pieces, cfg.weight_parts, cfg.weight_compactness),
        min_piece_area=cfg.min_area_m2,
    )
    dead_end_counts = search.get("dead_end_counts") or {}
    return {
        "variants": list(search.get("variants") or []),
        "n_success": int(search.get("n_success", 0)),
        "n_failed": int(search.get("n_failed", 0)),
        "n_depth_limit_dead_ends": int(dead_end_counts.get("depth_limit_dead_end", 0)),
        "n_no_reflex_dead_ends": int(dead_end_counts.get("no_reflex_dead_end", 0)),
        "n_min_area_dead_ends": int(dead_end_counts.get("min_area_dead_end", 0)),
        "n_states_seen": int(search.get("n_states_seen", 0)),
        "search_total_states": int(search.get("search_total_states", 0)),
        "search_max_depth": int(search.get("search_max_depth", 0)),
        "search_max_width": int(search.get("search_max_width", 0)),
        "search_split_attempts": int(search.get("search_split_attempts", 0)),
        "search_successful_splits": int(search.get("search_successful_splits", 0)),
        "bisector_fallback_attempts": int(search.get("bisector_fallback_attempts", 0)),
        "bisector_fallback_successes": int(search.get("bisector_fallback_successes", 0)),
        "terminated_by": str(search.get("terminated_by", "queue_exhausted")),
        "search_depth_used": search.get("search_depth_used"),
        "search_width_used": search.get("search_width_used"),
        "search_attempt_count": int(search.get("search_attempt_count", 0)),
        "search_attempt_history": list(search.get("search_attempt_history") or []),
    }

# ---------------------------------------------------------------------------
# Polygon-level public API
# ---------------------------------------------------------------------------

def decompose_polygon_with_stats(
    poly: Polygon,
    cfg: ConvexDecompositionConfig,
) -> Dict[str, Any]:
    """
    Run convex decomposition on a single polygon and return detailed statistics.

    Returns a dict with keys:
        - convex_success : bool
            True iff at least one fully convex decomposition variant was found.
        - n_variants : int
            Number of fully convex variants found (capped by cfg.max_variants_per_polygon).
        - best_variant : list[Polygon]
            Chosen configuration (fully convex if convex_success is True,
            otherwise best_variant returns an empty list and n_parts is 0).
        - fully_convex_best : bool
            True if best_variant is fully convex (i.e., convex_success is True).
        - has_small_zone : bool
            True if any part in best_variant has area < cfg.min_area_m2.
        - n_parts : int
            len(best_variant).
        - time_seconds : float
            Only if cfg.track_time is True.
        - search_terminated_by : str
            Why the search loop stopped.
        - n_failed_terminal_states : int
            Number of non-convex terminal dead-end states encountered.
        - n_depth_limit_dead_ends : int
            Failed terminal states caused by hitting `search_depth`.
        - n_no_reflex_dead_ends : int
            Failed terminal states where no further reflex-driven split was available.
        - n_min_area_dead_ends : int
            Failed terminal states where every available split branch created a
            sub-polygon below `cfg.min_area_m2`.
        - n_states_seen : int
            Number of unique decomposition states explored.
    """
    t0 = time.perf_counter() if getattr(cfg, "track_time", False) else None

    def _with_time(res: Dict[str, Any]) -> Dict[str, Any]:
        if t0 is not None:
            out = dict(res)
            out["time_seconds"] = time.perf_counter() - t0
            return out
        return res

    # Early exit: invalid / empty polygon
    if not isinstance(poly, Polygon) or poly.is_empty:
        return _with_time({
            "convex_success": False,
            "n_variants": 0,
            "best_variant": [],
            "fully_convex_best": False,
            "has_small_zone": False,
            "n_parts": 0,
            "search_terminated_by": "invalid_or_empty_input",
            "n_failed_terminal_states": 0,
            "n_depth_limit_dead_ends": 0,
            "n_no_reflex_dead_ends": 0,
            "n_min_area_dead_ends": 0,
            "n_states_seen": 0,
            "search_total_states": 0,
            "search_max_depth": 0,
            "search_max_width": 0,
            "search_split_attempts": 0,
            "search_successful_splits": 0,
            "bisector_fallback_attempts": 0,
            "bisector_fallback_successes": 0,
            "search_depth_used": None,
            "search_width_used": None,
            "search_attempt_count": 0,
            "search_attempt_history": [],
            "fallback_used": None,
        })

    # Early exit: polygon already convex → treat as one successful variant
    if is_convex_polygon(poly):
        return _with_time({
            "convex_success": True,
            "n_variants": 1,
            "best_variant": [poly],
            "fully_convex_best": True,
            "has_small_zone": poly.area < cfg.min_area_m2,
            "n_parts": 1,
            "search_terminated_by": "already_convex",
            "n_failed_terminal_states": 0,
            "n_depth_limit_dead_ends": 0,
            "n_no_reflex_dead_ends": 0,
            "n_min_area_dead_ends": 0,
            "n_states_seen": 1,
            "search_total_states": 1,
            "search_max_depth": 0,
            "search_max_width": 0,
            "search_split_attempts": 0,
            "search_successful_splits": 0,
            "bisector_fallback_attempts": 0,
            "bisector_fallback_successes": 0,
            "search_depth_used": None,
            "search_width_used": None,
            "search_attempt_count": 0,
            "search_attempt_history": [],
            "fallback_used": None,
        })

    # Enumerate ONLY fully convex variants.
    search = _enumerate_states_for_polygon(poly, cfg)
    variants = [record for record in search["variants"] if record.get("parts")]
    n_variants = len(variants)

    if n_variants == 0:
        # No fully convex decomposition found. The caller discards a building
        # whose pieces did not all succeed, so there is no partial result to keep.
        return _with_time({
            "convex_success": False,
            "n_variants": 0,
            "best_variant": [],
            "fully_convex_best": False,
            "has_small_zone": False,
            "n_parts": 0,
            "search_terminated_by": search.get("terminated_by", "no_fully_convex_variant"),
            "n_failed_terminal_states": int(search.get("n_failed", 0)),
            "n_depth_limit_dead_ends": int(search.get("n_depth_limit_dead_ends", 0)),
            "n_no_reflex_dead_ends": int(search.get("n_no_reflex_dead_ends", 0)),
            "n_min_area_dead_ends": int(search.get("n_min_area_dead_ends", 0)),
            "n_states_seen": int(search.get("n_states_seen", 0)),
            "search_total_states": int(search.get("search_total_states", 0)),
            "search_max_depth": int(search.get("search_max_depth", 0)),
            "search_max_width": int(search.get("search_max_width", 0)),
            "search_split_attempts": int(search.get("search_split_attempts", 0)),
            "search_successful_splits": int(search.get("search_successful_splits", 0)),
            "bisector_fallback_attempts": int(search.get("bisector_fallback_attempts", 0)),
            "bisector_fallback_successes": int(search.get("bisector_fallback_successes", 0)),
            "search_depth_used": search.get("search_depth_used"),
            "search_width_used": search.get("search_width_used"),
            "search_attempt_count": int(search.get("search_attempt_count", 0)),
            "search_attempt_history": list(search.get("search_attempt_history") or []),
            "fallback_used": None,
        })

    # At this point, every variant in `variants` is fully convex by construction.
    def _pick_best(var_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        best_idx = 0
        best_score = float("inf")
        best_uses_fallback = True
        for i, var in enumerate(var_list):
            parts = var["parts"]
            score = _score_polys(parts, cfg.weight_parts, cfg.weight_compactness)
            uses_fallback = bool(var.get("fallback_used"))
            if (score, uses_fallback) < (best_score, best_uses_fallback):
                best_score = score
                best_uses_fallback = uses_fallback
                best_idx = i
        return var_list[best_idx]

    best_variant_record = _pick_best(variants)
    best_variant = best_variant_record["parts"]
    has_small_zone = any(p.area < cfg.min_area_m2 for p in best_variant)

    return _with_time({
        "convex_success": True,
        "n_variants": n_variants,
        "best_variant": best_variant,
        "fully_convex_best": True,  # by definition, since all parts are convex
        "has_small_zone": has_small_zone,
        "n_parts": len(best_variant),
        "search_terminated_by": search.get("terminated_by", "queue_exhausted"),
        "n_failed_terminal_states": int(search.get("n_failed", 0)),
        "n_depth_limit_dead_ends": int(search.get("n_depth_limit_dead_ends", 0)),
        "n_no_reflex_dead_ends": int(search.get("n_no_reflex_dead_ends", 0)),
        "n_min_area_dead_ends": int(search.get("n_min_area_dead_ends", 0)),
        "n_states_seen": int(search.get("n_states_seen", 0)),
        "search_total_states": int(search.get("search_total_states", 0)),
        "search_max_depth": int(search.get("search_max_depth", 0)),
        "search_max_width": int(search.get("search_max_width", 0)),
        "search_split_attempts": int(search.get("search_split_attempts", 0)),
        "search_successful_splits": int(search.get("search_successful_splits", 0)),
        "bisector_fallback_attempts": int(search.get("bisector_fallback_attempts", 0)),
        "bisector_fallback_successes": int(search.get("bisector_fallback_successes", 0)),
        "search_depth_used": search.get("search_depth_used"),
        "search_width_used": search.get("search_width_used"),
        "search_attempt_count": int(search.get("search_attempt_count", 0)),
        "search_attempt_history": list(search.get("search_attempt_history") or []),
        "fallback_used": best_variant_record.get("fallback_used"),
    })



def decompose_polygon_best(poly: Polygon, cfg: ConvexDecompositionConfig) -> List[Polygon]:
    """
    Convenience wrapper: return only the best decomposition variant (list[Polygon]).
    """
    return decompose_polygon_with_stats(poly, cfg)["best_variant"]
