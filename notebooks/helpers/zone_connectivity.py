"""Reject buildings whose zones do not form one body.

Every other check in stage 4 looks at a zone: is it convex, is it wide enough,
is it too long for its width. This one looks at the set. A building can pass
zone by zone and still be unusable, because its zones have ended up in two
groups that touch nowhere, or touch only at a point.

That happens for real reasons. A footprint mapped as two wings joined by a
corridor loses the corridor to the minimum-area filter. An L with a very thin
waist is pinched apart by the shrink-expand cleanup. In both cases every
surviving zone is a perfectly good room and the building is still two buildings.
EnergyPlus would solve it, and the answer would describe a thing that does not
exist.

Adjacency is tested on a small buffer rather than on exact contact, because two
zones cut from the same polygon meet along an edge that floating-point
arithmetic has moved by a nanometre. Sharing only a corner does not count: a
point of contact carries no heat.

Import from a notebook as::

    from helpers.zone_connectivity import remove_disconnected_buildings

"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from .utils import polygon_parts

__all__ = ["connected_zone_groups", "remove_disconnected_buildings"]


def connected_zone_groups(zones, *, gap_tol_m: float = 0.05, min_shared_edge_m: float = 0.01) -> list[list[int]]:
    """Group a building's zones by whether heat could pass between them.

    Two zones are joined when their buffered outlines overlap along at least
    ``min_shared_edge_m`` of shared boundary. The buffer absorbs the numerical
    gap between zones cut from one polygon; the shared-length test is what
    stops two zones that merely touch at a corner from counting as joined.

    Returns the indices of ``zones``, grouped. One group means the building
    holds together.
    """
    polys = []
    for zone in zones or []:
        parts = polygon_parts(zone)
        if parts:
            polys.append(max(parts, key=lambda poly: poly.area))
    if len(polys) <= 1:
        return [[0]] if polys else []

    buffered = [poly.buffer(gap_tol_m, join_style=2) for poly in polys]

    parent = list(range(len(polys)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if not buffered[i].intersects(buffered[j]):
                continue
            shared = buffered[i].intersection(buffered[j])
            # A corner contact has negligible extent; a shared wall does not.
            if float(getattr(shared, "length", 0.0)) < min_shared_edge_m and float(shared.area) <= 0.0:
                continue
            parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(polys)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def remove_disconnected_buildings(
    buildings_m: gpd.GeoDataFrame,
    *,
    zones_col: str = "zones",
    gap_tol_m: float = 0.05,
    min_shared_edge_m: float = 0.01,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split a frame into the buildings that hold together and those that do not.

    Adds ``n_zone_groups`` to both frames, so a rejection can be read without
    re-running the test.

    Returns the survivors and the rejects, in that order.
    """
    if buildings_m.empty:
        out = buildings_m.copy()
        out["n_zone_groups"] = pd.Series(dtype="int64")
        return out, out.copy()

    work = buildings_m.copy()
    work["n_zone_groups"] = [
        len(connected_zone_groups(zones, gap_tol_m=gap_tol_m, min_shared_edge_m=min_shared_edge_m))
        for zones in work[zones_col]
    ]

    whole = work["n_zone_groups"] <= 1
    return work[whole].copy(), work[~whole].copy()
