"""Interactive Plotly renderer for one Honeybee model."""

from __future__ import annotations

import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from honeybee_plot_geometry import collect_model_geometry, renderer_color_map


# ``None`` hides the title, so a separate sentinel represents an omitted title.
_TITLE_OMITTED = object()


def plot_honeybee_model(
    model,
    title=_TITLE_OMITTED,
    *,
    extrude_eps: float = 0.02,
    figsize: tuple[int, int] = (640, 420),
    show: bool = True,
    show_wireframe: bool = False,
    surface_opacity: float = 0.5,
    show_legend: bool = False,
    include_types: list[str] | None = None,
    room_ids: list[str] | None = None,
    background_color: str | None = "white",
    aspectmode: str = "data",
):
    """Render one Honeybee model as an interactive Plotly figure.

    Apertures and doors receive a small thickness so that they remain visible on
    their host surfaces. When ``show`` is false, return the figure for embedding
    or further styling; otherwise display it and return ``None``.
    """

    import numpy as np

    # Colors follow the Ladybug/OpenStudio palette when it is available.
    try:
        from ladybug.color import Colorset  # optional
    except Exception:
        Colorset = None  # type: ignore

    COLOR_MAP = renderer_color_map(Colorset)
    if title is _TITLE_OMITTED:
        title_val = getattr(model, "display_name", None) or getattr(model, "identifier", None)
    else:
        title_val = title

    categories, edge_segments, _ = collect_model_geometry(
        model,
        extrude_eps=extrude_eps,
        include_types=include_types,
        room_ids=room_ids,
        want_wire=show_wireframe,
    )

    import plotly.graph_objects as go

    fig = go.Figure()
    if include_types is not None:
        include_types = set(include_types)

    draw_order = [
        "exterior_floor",
        "interior_floor",
        "roof",
        "ceiling",
        "exterior_wall",
        "interior_wall",
        "air_wall",
        "aperture",
        "interior_aperture",
        "door",
        "interior_door",
        "outdoor_shade",
        "indoor_shade",
        "shade_mesh",
        "shade",
        "default",
    ]

    for key in draw_order:
        if include_types is not None and key not in include_types:
            continue
        data = categories.get(key)
        if not data or not data["pts"] or not data["I"]:
            continue
        X, Y, Z = zip(*data["pts"])
        fig.add_trace(
            go.Mesh3d(
                x=X,
                y=Y,
                z=Z,
                i=data["I"],
                j=data["J"],
                k=data["K"],
                color=COLOR_MAP.get(key, COLOR_MAP["default"]),
                opacity=float(np.clip(surface_opacity, 0.0, 1.0)),
                flatshading=True,
                name=key,
                hoverinfo="skip",
                lighting=dict(ambient=0.6, diffuse=0.4),
                showlegend=show_legend,
            )
        )

    for key, data in categories.items():
        if key in draw_order:
            continue
        if include_types is not None and key not in include_types:
            continue
        if not data["pts"] or not data["I"]:
            continue
        X, Y, Z = zip(*data["pts"])
        fig.add_trace(
            go.Mesh3d(
                x=X,
                y=Y,
                z=Z,
                i=data["I"],
                j=data["J"],
                k=data["K"],
                color=COLOR_MAP.get(key, COLOR_MAP["default"]),
                opacity=float(np.clip(surface_opacity, 0.0, 1.0)),
                flatshading=True,
                name=key,
                hoverinfo="skip",
                lighting=dict(ambient=0.6, diffuse=0.4),
                showlegend=show_legend,
            )
        )

    if show_wireframe and edge_segments:
        xs, ys, zs = [], [], []
        for (p, q) in edge_segments:
            xs += [p[0], q[0], None]
            ys += [p[1], q[1], None]
            zs += [p[2], q[2], None]
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                name="wireframe",
                line=dict(width=2, color="black"),
                hoverinfo="skip",
                showlegend=show_legend,
            )
        )

    layout_title = str(title_val) if title_val is not None else None
    fig.update_layout(
        title=layout_title,
        width=int(figsize[0]) if figsize and len(figsize) >= 1 else 640,
        height=int(figsize[1]) if figsize and len(figsize) >= 2 else 420,
        margin=dict(l=0, r=0, t=30 if layout_title else 6, b=0),
        paper_bgcolor=background_color or "white",
        plot_bgcolor=background_color or "white",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode=aspectmode,
        ),
        showlegend=show_legend,
    )

    if show:
        fig.show(config={"scrollZoom": True, "displaylogo": False})
        return None
    return fig
