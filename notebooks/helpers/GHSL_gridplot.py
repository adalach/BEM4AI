"""Grid previews for aligned building, zone, window, and context geometries."""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmaps
try:
    # Shapely 2.x
    from shapely.geometry.base import BaseGeometry  # type: ignore
except Exception:
    # Fallback for older shapely
    from shapely.geometry import base as _shp_base  # type: ignore
    BaseGeometry = getattr(_shp_base, 'BaseGeometry', object)  # type: ignore


def _default_color_for(role: str) -> str:
    if role == "main":
        return "#1f77b4"  # matplotlib default blue
    return "#d3d3d3"      # light grey for neighbors


def _default_cmap_for(role: str):
    if role == "main":
        return cmaps.get("viridis")   # returns a Colormap object
    return cmaps.get("Greys")


def _normalize_cmap(cmap_like, role: str):
    """Accepts a Colormap or a string name; returns a Colormap.
       Falls back to the role default if the name is invalid."""
    if cmap_like is None:
        return _default_cmap_for(role)
    if isinstance(cmap_like, str):
        try:
            return cmaps.get(cmap_like)
        except Exception:
            return _default_cmap_for(role)
    return cmap_like


def _iter_parts(geom_or_parts, parts_col_present: bool) -> Iterable[BaseGeometry]:
    if parts_col_present:
        parts = geom_or_parts
        if parts is None:
            return []
        try:
            return [g for g in parts if g is not None]
        except TypeError:
            return [geom_or_parts]
    else:
        return [geom_or_parts]


def GHSL_gridplot(
    sources: Sequence[dict],
    ids: Optional[Sequence] = None,
    *,
    id_col: str = "sample_id",
    ncols: int = 5,
    figsize_scale: float = 3.5,
    title_fn: Optional[Callable[[object, int], str]] = None,
    edgecolor_main: str = "black",
    edgecolor_neighbor: str = "none",
    linewidth: float = 0.5,
    background_color: Optional[str] = None,
    tight_layout: bool = True,
    share_aspect_equal: bool = True,
):
    """Plot the same sample IDs across aligned geospatial data sources."""
    if not sources:
        raise ValueError("sources must be a non-empty list of source dicts")

    norm_sources = []
    for s in sources:
        if "gdf" not in s:
            raise ValueError("Each source dict must contain a 'gdf' key (GeoDataFrame)")
        role = s.get("role", "neighbor")
        name = s.get("name", role)
        gdf: gpd.GeoDataFrame = s["gdf"]
        src_id_col = s.get("id_col", id_col)
        geometry_col = s.get(
            "geometry_col",
            getattr(gdf, "_geometry_column_name", None) or (gdf.geometry.name if hasattr(gdf, "geometry") else "geometry"),
        )
        parts_col = s.get("parts_col")
        color = s.get("color")
        cmap = s.get("cmap")
        alpha = s.get("alpha", 1.0)

        src_linewidth = s.get("linewidth", linewidth)
        src_edgecolor = s.get("edgecolor", (edgecolor_main if role == "main" else edgecolor_neighbor))
        src_zorder = s.get("zorder", None)

        norm_sources.append({
            "name": name,
            "gdf": gdf,
            "role": role,
            "id_col": src_id_col,
            "geometry_col": geometry_col,
            "parts_col": parts_col,
            "color": color,
            "cmap": cmap,
            "alpha": alpha,
            "linewidth": src_linewidth,
            "edgecolor": src_edgecolor,
            "zorder": src_zorder,
        })

    if ids is None:
        anchor = norm_sources[0]
        if anchor["id_col"] not in anchor["gdf"].columns:
            raise KeyError(f"Anchor source '{anchor['name']}' is missing id column '{anchor['id_col']}'")
        ids = list(anchor["gdf"][anchor["id_col"]])
    ids = list(ids)

    n = len(ids)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols)) if n > 0 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_scale*ncols, figsize_scale*nrows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    if background_color is not None:
        fig.patch.set_facecolor(background_color)

    # Draw order: neighbors first, then mains
    draw_order = [s for s in norm_sources if s["role"] != "main"] + [s for s in norm_sources if s["role"] == "main"]

    for ax_idx, ax in enumerate(axes[:n]):
        cur_id = ids[ax_idx]

        if share_aspect_equal:
            ax.set_aspect("equal")

        for src in draw_order:
            gdf = src["gdf"]
            if src["id_col"] not in gdf.columns:
                continue
            mask = gdf[src["id_col"]] == cur_id
            sub = gdf[mask]
            if sub.empty:
                continue

            role = src["role"]
            parts_col = src["parts_col"]
            has_parts = parts_col is not None and (parts_col in sub.columns)

            if has_parts:
                cmap = _normalize_cmap(src.get("cmap"), role)
                color = None  # ignored for parts rendering
            else:
                color = src.get("color", None)
                if color is None:
                    color = _default_color_for(role)
                cmap = None  # not used

            lw = float(src["linewidth"])
            ec = src["edgecolor"]
            z = src.get("zorder", None)

            if has_parts:
                parts_series = sub[parts_col]
                total_parts = sum(len(_iter_parts(parts, True)) for parts in parts_series)
                denom = max(1, total_parts - 1)
                i_part = 0
                for _, row in sub.iterrows():
                    parts = _iter_parts(row[parts_col], True)
                    for part in parts:
                        rgba = cmap(i_part / denom) if hasattr(cmap, "__call__") else cmap
                        gpd.GeoSeries([part], crs=sub.crs).plot(
                            ax=ax,
                            color=rgba,
                            edgecolor=ec,
                            linewidth=lw,
                            alpha=src["alpha"],
                            zorder=z,
                        )
                        i_part += 1
            else:
                geom_col = src["geometry_col"]
                geoms = sub[geom_col]
                gpd.GeoSeries(geoms.values, crs=sub.crs).plot(
                    ax=ax,
                    color=color,
                    edgecolor=ec,
                    linewidth=lw,
                    alpha=src["alpha"],
                    zorder=z,
                )

        title = title_fn(cur_id, ax_idx) if callable(title_fn) else str(cur_id)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    if tight_layout:
        plt.tight_layout()

    return fig, axes


def preview_samples(
    samples: "gpd.GeoDataFrame",
    layers: Sequence[dict],
    *,
    ids: Optional[Sequence] = None,
    n_show: int = 20,
    id_col: str = "sample_id",
    ncols: int = 5,
    figsize_scale: float = 3.5,
    title_fn: Optional[Callable[[object, int], str]] = None,
    enabled: bool = True,
    **gridplot_kwargs,
):
    """Draw one panel per sample, layer by layer.

    Stage 4 previews the same handful of samples after every step, changing only
    which geometries are drawn on top. This does the bookkeeping those previews
    share: pick the panels from `samples`, cut each layer's frame down to them,
    point it at the right geometry column and drop the rows with nothing in it.

    Each layer is a dict of ``gdf`` plus, optionally, ``column`` (which geometry
    column to draw, default the frame's active one), ``parts_col`` (draw a list
    of geometries per row, coloured by a colormap), and any styling key
    `GHSL_gridplot` accepts: ``role``, ``color``, ``edgecolor``, ``linewidth``,
    ``alpha``, ``zorder``, ``cmap``, ``name``.

    With ``enabled=False`` nothing is drawn, which is how the notebook's
    plot switch turns the previews off for a headless run.
    """
    if not enabled:
        return None

    if ids is None:
        ids = samples[id_col].head(n_show).tolist()
    ids = list(ids)
    if not ids:
        print("No samples to preview.")
        return None

    sources = []
    for layer in layers:
        spec = dict(layer)
        gdf = spec.pop("gdf")
        column = spec.pop("column", None)
        parts_col = spec.get("parts_col")
        if gdf is None or id_col not in getattr(gdf, "columns", []):
            continue

        sub = gdf[gdf[id_col].isin(ids)]
        if sub.empty:
            continue

        if parts_col is None:
            geometry_col = column or sub.geometry.name
            sub = gpd.GeoDataFrame(sub.copy()).set_geometry(geometry_col, crs=gdf.crs)
            sub = sub[sub[geometry_col].notna() & ~sub.geometry.is_empty]
            if sub.empty:
                continue
            spec["geometry_col"] = geometry_col

        spec["gdf"] = sub
        spec.setdefault("id_col", id_col)
        sources.append(spec)

    if not sources:
        print("No geometry to preview.")
        return None

    return GHSL_gridplot(
        sources,
        ids=ids,
        id_col=id_col,
        ncols=ncols,
        figsize_scale=figsize_scale,
        title_fn=title_fn or (lambda sid, i: str(sid)),
        **gridplot_kwargs,
    )
