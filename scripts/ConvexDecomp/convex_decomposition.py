"""The bounded search that cuts a polygon into convex pieces.

A decomposition is found by search, not by construction. A state is a set of
polygons; a move cuts one of them at a reflex vertex; a state is terminal when
every polygon in it is convex. The search is best-first over states scored by
`score_polygon_partition`, and bounded in two directions - depth, how many cuts
deep it will go, and width, how many cuts it will consider at each step -
because the space is far too large to exhaust.

Three pools hold the frontier. `active` is what the current budget allows,
`overflow` holds states the width bound pushed aside, and `deferred` holds
states the depth bound stopped. Widening the budget moves states back from the
other two rather than starting again, so raising the bounds costs only the work
the wider bounds actually add.

Nothing here knows about buildings. It works on any simple polygon.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon

from helpers.utils import compactness as compactness_polsby_popper

from .polygon_convexity import polygon_turn_crosses, raw_ring_key

__all__ = [
    "AdaptiveSearchConfig",
    "DecompositionSearchConfig",
    "ExpandedState",
    "StateExpansionResult",
    "compactness_polsby_popper",
    "run_polygon_state_search",
    "score_polygon_partition",
]


@dataclass(frozen=True)
class AdaptiveSearchConfig:
    """Outer retry schedule for the shared polygon search."""

    enabled: bool = False
    initial_depth: Optional[int] = None
    initial_width: Optional[int] = None
    max_depth: Optional[int] = None
    max_width: Optional[int] = None
    depth_step: int = 1
    width_step: int = 1


@dataclass(frozen=True)
class DecompositionSearchConfig:
    max_depth: int = 8
    max_variants: int = 64
    max_failed_states: Optional[int] = None


@dataclass(frozen=True)
class ExpandedState:
    pieces: Tuple[Polygon, ...]
    fallback_labels: Tuple[str, ...] = ()


@dataclass
class StateExpansionResult:
    children: List[ExpandedState] = field(default_factory=list)
    dead_end_counts: Counter = field(default_factory=Counter)
    metrics: Counter = field(default_factory=Counter)


@dataclass
class _QueuedState:
    key: Hashable
    state: ExpandedState
    depth: int
    priority: Tuple[Any, ...]
    order: int
    pool: str


def score_polygon_partition(
    polys: Sequence[Polygon],
    *,
    weight_parts: float,
    weight_compactness: float,
) -> float:
    """Score a decomposition. Lower is better.

    Two things make a zoning bad, and they pull against each other. Cutting
    more produces rounder pieces; cutting less produces fewer of them. So the
    score is ``weight_parts * n_pieces + weight_compactness * (1 - C)``, where
    ``C`` is Polsby-Popper compactness averaged over the pieces and weighted by
    their area, so a large ugly piece counts for more than a small one.

    The ratio between the two weights is the real parameter. At the shipped
    values, 1 and 20, one extra piece is worth about a five-hundredth of
    compactness - the search will happily cut again to square something up, and
    will not cut to no purpose.
    """
    if not polys:
        return 0.0

    areas = np.array([poly.area for poly in polys], dtype=float)
    total_area = float(areas.sum()) if areas.size else 1.0
    if total_area <= 0.0:
        return float(weight_parts * len(polys))

    compactness = np.array([compactness_polsby_popper(poly) for poly in polys], dtype=float)
    area_weighted_compactness = float((areas * compactness).sum() / max(total_area, 1e-12))
    return weight_parts * len(polys) + weight_compactness * (1.0 - area_weighted_compactness)


def _resolve_adaptive_limits(
    *,
    adaptive_cfg: Optional[AdaptiveSearchConfig],
    default_depth: int,
    default_width: Optional[int],
) -> Tuple[bool, int, int, Optional[int], Optional[int], int, int]:
    adaptive_enabled = bool(adaptive_cfg and adaptive_cfg.enabled)
    max_depth = max(
        1,
        int(adaptive_cfg.max_depth if adaptive_cfg and adaptive_cfg.max_depth is not None else default_depth),
    )
    depth = int(adaptive_cfg.initial_depth) if adaptive_cfg and adaptive_cfg.initial_depth is not None else int(default_depth)
    depth = min(max(1, depth), max_depth)

    if adaptive_cfg and adaptive_cfg.max_width is not None:
        max_width = int(adaptive_cfg.max_width)
    else:
        max_width = int(default_width) if default_width is not None else None

    if adaptive_cfg and adaptive_cfg.initial_width is not None:
        width = int(adaptive_cfg.initial_width)
    else:
        width = int(default_width) if default_width is not None else None

    if max_width is not None and width is not None:
        width = min(max(1, width), max_width)

    depth_step = max(1, int(adaptive_cfg.depth_step if adaptive_cfg else 1))
    width_step = max(1, int(adaptive_cfg.width_step if adaptive_cfg else 1))
    return adaptive_enabled, depth, max_depth, width, max_width, depth_step, width_step


def _piece_cache_key(poly: Polygon) -> Tuple[Tuple[Tuple[float, float], ...], int]:
    return raw_ring_key(poly), int(len(poly.interiors))


def _count_reflex_vertices(poly: Polygon, *, tol: float = 0.0) -> int:
    coords, cross, orientation = polygon_turn_crosses(poly)
    if coords is None or len(coords) <= 3 or cross.size == 0:
        return 0
    return int(np.count_nonzero(cross * orientation < -tol))


def run_polygon_state_search(
    initial_pieces: Sequence[Polygon],
    *,
    cfg: DecompositionSearchConfig,
    is_convex_piece: Callable[[Polygon], bool],
    state_key_fn: Callable[[List[Polygon]], Hashable],
    expand_state_fn: Callable[[List[Polygon]], StateExpansionResult],
    adaptive_cfg: Optional[AdaptiveSearchConfig] = None,
    default_width: Optional[int] = None,
    state_score_fn: Optional[Callable[[List[Polygon]], Any]] = None,
    min_piece_area: Optional[float] = None,
) -> Dict[str, Any]:
    initial_state = ExpandedState(tuple(initial_pieces), ())
    variants: List[Dict[str, Any]] = []

    metrics = Counter(
        {
            "search_total_states": 0,
            "search_max_depth": 0,
            "search_max_width": 0,
            "search_max_active_frontier": 0,
            "search_max_overflow_frontier": 0,
            "search_max_deferred_frontier": 0,
        }
    )

    if cfg.max_variants is None:
        max_success = float("inf")
    else:
        max_success = max(1, int(cfg.max_variants))

    if cfg.max_failed_states is None:
        max_failed = float("inf")
    else:
        max_failed = max(1, int(cfg.max_failed_states))

    n_success = 0
    n_failed = 0
    dead_end_counts: Counter = Counter()
    terminated_by = "invalid_initial_state"

    adaptive_enabled, current_depth_limit, max_depth_limit, current_width_limit, max_width_limit, depth_step, width_step = _resolve_adaptive_limits(
        adaptive_cfg=adaptive_cfg,
        default_depth=cfg.max_depth,
        default_width=default_width,
    )

    active: Dict[Hashable, _QueuedState] = {}
    overflow: Dict[Hashable, _QueuedState] = {}
    deferred: Dict[Hashable, _QueuedState] = {}
    frontier_nodes: Dict[Hashable, _QueuedState] = {}
    closed_states: set[Hashable] = set()
    priority_cache: Dict[Hashable, Tuple[Any, ...]] = {}
    piece_stats_cache: Dict[Tuple[Tuple[Tuple[float, float], ...], int], Tuple[bool, int, float]] = {}
    history: List[Dict[str, Any]] = []
    insertion_order = 0

    def _node_sort_key(node: _QueuedState) -> Tuple[Any, ...]:
        return (*node.priority, node.order)

    def _update_frontier_metrics() -> None:
        metrics["search_max_active_frontier"] = max(metrics["search_max_active_frontier"], len(active))
        metrics["search_max_overflow_frontier"] = max(metrics["search_max_overflow_frontier"], len(overflow))
        metrics["search_max_deferred_frontier"] = max(metrics["search_max_deferred_frontier"], len(deferred))

    def _remove_from_current_pool(node: _QueuedState) -> None:
        if node.pool == "active":
            active.pop(node.key, None)
        elif node.pool == "overflow":
            overflow.pop(node.key, None)
        elif node.pool == "deferred":
            deferred.pop(node.key, None)

    def _assign_to_pool(node: _QueuedState, pool: str) -> None:
        _remove_from_current_pool(node)
        node.pool = pool
        if pool == "active":
            active[node.key] = node
        elif pool == "overflow":
            overflow[node.key] = node
        elif pool == "deferred":
            deferred[node.key] = node
        _update_frontier_metrics()

    def _trim_active_to_width() -> None:
        if current_width_limit is None:
            _update_frontier_metrics()
            return
        while len(active) > current_width_limit:
            worst = max(active.values(), key=_node_sort_key)
            _assign_to_pool(worst, "overflow")
        _update_frontier_metrics()

    def _piece_stats(poly: Polygon) -> Tuple[bool, int, float]:
        cache_key = _piece_cache_key(poly)
        cached = piece_stats_cache.get(cache_key)
        if cached is not None:
            return cached
        convex = bool(is_convex_piece(poly))
        reflexes = 0 if convex else _count_reflex_vertices(poly)
        stats = (convex, reflexes, float(poly.area))
        piece_stats_cache[cache_key] = stats
        return stats

    def _state_priority(key: Hashable, pieces: List[Polygon], fallback_labels: Tuple[str, ...], depth: int) -> Tuple[Any, ...]:
        cached = priority_cache.get(key)
        if cached is None:
            n_nonconvex = 0
            total_reflex = 0
            near_min_area_penalty = 0.0
            min_area_value = float("inf")
            for piece in pieces:
                convex, reflexes, area = _piece_stats(piece)
                if not convex:
                    n_nonconvex += 1
                    total_reflex += reflexes
                min_area_value = min(min_area_value, area)
                if min_piece_area is not None and min_piece_area > 0.0:
                    soft_limit = 1.5 * float(min_piece_area)
                    if area < soft_limit:
                        near_min_area_penalty += (soft_limit - area) / float(min_piece_area)
            partition_score = state_score_fn(pieces) if state_score_fn is not None else float(len(pieces))
            if isinstance(partition_score, tuple):
                partition_key = tuple(partition_score)
            else:
                partition_key = (float(partition_score),)
            cached = (
                int(n_nonconvex),
                int(total_reflex),
                round(float(near_min_area_penalty), 6),
                *partition_key,
                int(len(pieces)),
                round(-float(min_area_value if math.isfinite(min_area_value) else 0.0), 6),
            )
            priority_cache[key] = cached
        # For satisficing decomposition, a tie between equally promising states
        # should prefer the deeper branch because it is closer to a full split.
        return (*cached, 1 if fallback_labels else 0, -int(depth))

    def _record_attempt(stage_reason: str) -> None:
        history.append(
            {
                "depth": int(current_depth_limit),
                "width": int(current_width_limit) if current_width_limit is not None else None,
                "success": bool(variants),
                "n_variants": int(len(variants)),
                "terminated_by": stage_reason,
                "n_failed": int(n_failed),
                "dead_end_counts": dict(dead_end_counts),
                "n_states_seen": int(len(closed_states)),
            }
        )

    def _admit_existing_node(node: _QueuedState) -> None:
        nonlocal n_failed
        is_goal = bool(node.priority and node.priority[0] == 0)
        if not is_goal and node.depth >= current_depth_limit:
            if adaptive_enabled and current_depth_limit < max_depth_limit:
                _assign_to_pool(node, "deferred")
            else:
                frontier_nodes.pop(node.key, None)
                _remove_from_current_pool(node)
                closed_states.add(node.key)
                n_failed += 1
                dead_end_counts.update(["depth_limit_dead_end"])
            return

        _assign_to_pool(node, "active")
        _trim_active_to_width()

    def _enqueue_state(state: ExpandedState, depth: int) -> None:
        nonlocal insertion_order
        pieces = list(state.pieces)
        key = state_key_fn(pieces)
        if not key or key in closed_states:
            return

        priority = _state_priority(key, pieces, state.fallback_labels, depth)
        candidate = _QueuedState(
            key=key,
            state=state,
            depth=int(depth),
            priority=priority,
            order=insertion_order,
            pool="active",
        )
        insertion_order += 1

        existing = frontier_nodes.get(key)
        if existing is not None:
            if _node_sort_key(existing) <= _node_sort_key(candidate):
                return
            _remove_from_current_pool(existing)
            frontier_nodes.pop(key, None)

        frontier_nodes[key] = candidate
        _admit_existing_node(candidate)

    def _activate_depth_budget() -> None:
        for node in sorted(list(deferred.values()), key=_node_sort_key):
            _admit_existing_node(node)

    def _promote_width_budget() -> None:
        if not overflow:
            return
        for node in sorted(list(overflow.values()), key=_node_sort_key):
            if current_width_limit is not None and len(active) >= current_width_limit:
                break
            if node.depth >= current_depth_limit and node.priority[0] != 0:
                continue
            _assign_to_pool(node, "active")
        _trim_active_to_width()

    def _advance_budget_until_active() -> bool:
        nonlocal current_depth_limit, current_width_limit
        while not active:
            if current_depth_limit < max_depth_limit:
                current_depth_limit = min(max_depth_limit, current_depth_limit + depth_step)
                _activate_depth_budget()
                if active:
                    return True
                continue
            if current_width_limit is not None and max_width_limit is not None and current_width_limit < max_width_limit:
                current_width_limit = min(max_width_limit, current_width_limit + width_step)
                _promote_width_budget()
                if active:
                    return True
                continue
            return False
        return True

    _enqueue_state(initial_state, 0)

    while True:
        if n_success >= max_success:
            terminated_by = "max_success_variants"
            break
        if n_failed >= max_failed:
            terminated_by = "max_failed_states"
            break
        if not active:
            if variants:
                terminated_by = "queue_exhausted"
                break
            if not adaptive_enabled:
                terminated_by = "budget_exhausted" if overflow or deferred else "queue_exhausted"
                break
            if not _advance_budget_until_active():
                terminated_by = "budget_exhausted" if overflow or deferred else "queue_exhausted"
                break

        stage_reason = "frontier_exhausted"

        while active:
            if n_success >= max_success:
                stage_reason = "max_success_variants"
                break
            if n_failed >= max_failed:
                stage_reason = "max_failed_states"
                break

            node = min(active.values(), key=_node_sort_key)
            _remove_from_current_pool(node)
            frontier_nodes.pop(node.key, None)

            pieces = list(node.state.pieces)
            metrics["search_total_states"] += 1
            metrics["search_max_depth"] = max(metrics["search_max_depth"], node.depth)

            if all(is_convex_piece(piece) for piece in pieces):
                closed_states.add(node.key)
                fallback_used = ",".join(sorted(set(node.state.fallback_labels))) if node.state.fallback_labels else None
                variants.append(
                    {
                        "parts": pieces,
                        "fallback_used": fallback_used,
                    }
                )
                n_success += 1
                continue

            if node.depth >= current_depth_limit:
                if adaptive_enabled and current_depth_limit < max_depth_limit:
                    frontier_nodes[node.key] = node
                    _assign_to_pool(node, "deferred")
                    continue
                closed_states.add(node.key)
                n_failed += 1
                dead_end_counts.update(["depth_limit_dead_end"])
                continue

            closed_states.add(node.key)
            expansion = expand_state_fn(pieces)
            if expansion.metrics:
                metrics.update(expansion.metrics)
            metrics["search_max_width"] = max(metrics["search_max_width"], len(expansion.children))

            if expansion.children:
                for child in expansion.children:
                    combined_labels = node.state.fallback_labels + tuple(label for label in child.fallback_labels if label)
                    _enqueue_state(ExpandedState(child.pieces, combined_labels), node.depth + 1)
                continue

            n_failed += 1
            if expansion.dead_end_counts:
                dead_end_counts.update(expansion.dead_end_counts)
            else:
                dead_end_counts.update(["no_split_dead_end"])

        _record_attempt(stage_reason)
        if stage_reason != "frontier_exhausted":
            terminated_by = stage_reason
            break

    return {
        "variants": variants,
        "n_success": n_success,
        "n_failed": n_failed,
        "dead_end_counts": dict(dead_end_counts),
        "n_states_seen": len(closed_states),
        "terminated_by": terminated_by,
        "search_depth_used": int(current_depth_limit),
        "search_width_used": int(current_width_limit) if current_width_limit is not None else None,
        "search_attempt_count": int(len(history)),
        "search_attempt_history": history,
        "adaptive_search_enabled": adaptive_enabled,
        **dict(metrics),
    }
