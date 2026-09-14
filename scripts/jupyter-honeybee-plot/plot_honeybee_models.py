"""Static PyVista and Pillow renderer for multiple Honeybee models."""

from __future__ import annotations

import sys
import warnings
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import TqdmWarning

# PyVista imports an optional automatic progress bar that is not used here.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="IProgress not found.*", category=TqdmWarning)
    import pyvista as pv


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from honeybee_plot_geometry import collect_model_geometry, renderer_color_map

try:
    from ladybug.color import Colorset
except Exception:
    Colorset = None

_MODEL_TITLES_OMITTED = object()

VIEW_PRESETS: Dict[str, Union[str, Tuple[float, float]]] = {
    "iso": "iso",
    "top": (90.0, 0.0),
    "bottom": (-90.0, 0.0),
    "front": (0.0, 0.0),
    "back": (0.0, 180.0),
    "left": (0.0, -90.0),
    "right": (0.0, 90.0),
}


def plot_honeybee_models(
    models: Iterable[Any],
    model_titles: Any = _MODEL_TITLES_OMITTED,
    main_title: Optional[str] = None,
    grid: Optional[Tuple[int, int]] = (6, 2),
    extrude_eps: float = 0.02,
    figsize: Tuple[int, int] = (640, 420),
    total_width: int = 1200,
    base_model_height: int = 600,
    plot_main_legend: bool = True,
    *,
    show=True,
    view: Any = "iso",
    camera_zoom: float = 1.25,
    background_color: Optional[str] = None,
    legend_labels: Optional[List[str]] = None,
    shades_opacity: float = 0.5,
    **plot_kwargs,
) -> Optional[Image.Image]:
    """Render Honeybee models off-screen and combine them in one image grid.

    Apertures and doors receive a small thickness to prevent z-fighting. When
    ``show`` is false, return the composed Pillow image instead of displaying it.
    ``grid`` is given as ``(columns, rows)``.
    """

    COLOR_MAP = renderer_color_map(Colorset)
    def _default_titles(_models, _model_titles):
        if _model_titles is None:
            return [None] * len(_models)
        if _model_titles is _MODEL_TITLES_OMITTED:
            out = []
            for i, m in enumerate(_models):
                t = getattr(m, "display_name", None) or getattr(m, "identifier", None) or f"Model_{i+1}"
                out.append(str(t))
            return out
        if len(_model_titles) != len(_models):
            raise ValueError("model_titles must match number of models.")
        return [None if t is None else str(t) for t in _model_titles]




    def _auto_grid(n_models: int, _figsize: Tuple[int, int], _total_width: int):
        ref_w = max(1, int(_figsize[0]))
        cols = max(1, min(n_models, int(round(_total_width / float(ref_w))) or 1))
        rows = ceil(n_models / cols)
        return cols, rows

    def _per_cell_size(cols: int, _figsize: Tuple[int, int], _total_width: int, _base_model_height: int):
        cw = max(1, int(round(_total_width / float(cols))))
        scale = cw / float(max(1, int(_figsize[0])))
        ch = max(1, int(round(_base_model_height * scale)))
        return cw, ch

    def _polydata_from_category(cat):
        pts = np.asarray(cat["pts"], dtype=np.float32)
        if pts.size == 0 or len(cat["I"]) == 0:
            return None
        I = np.asarray(cat["I"], dtype=np.int64)
        J = np.asarray(cat["J"], dtype=np.int64)
        K = np.asarray(cat["K"], dtype=np.int64)
        faces = np.column_stack([np.full(I.shape[0], 3, dtype=np.int64), I, J, K]).ravel()
        return pv.PolyData(pts, faces=faces)

    def _wire_polydata(edges):
        """Build a line-only mesh so PyVista does not draw vertex points."""
        if not edges:
            return None
        idx = {}
        pts = []
        lines = []

        def addp(p):
            k = (round(p[0], 6), round(p[1], 6), round(p[2], 6))
            if k in idx:
                return idx[k]
            i = len(pts)
            idx[k] = i
            pts.append([p[0], p[1], p[2]])
            return i

        for a, b in edges:
            ia = addp(a)
            ib = addp(b)
            lines.extend([2, ia, ib])
        pts_arr = np.asarray(pts, np.float32)
        lines_arr = np.asarray(lines, np.int64)
        pd = pv.PolyData(pts_arr, lines=lines_arr)
        try:
            pd.verts = np.empty(0, dtype=np.int64)
            pd.faces = np.empty(0, dtype=np.int64)
        except Exception:
            pass
        return pd


    models = list(models)
    if not models:
        raise ValueError("No models provided.")

    show_wireframe = bool(plot_kwargs.get("show_wireframe", False))
    surface_opacity = float(plot_kwargs.get("surface_opacity", 1))
    include_types = set(plot_kwargs["include_types"]) if plot_kwargs.get("include_types") is not None else None
    room_ids = plot_kwargs.get("room_ids", None)
    wire_color = plot_kwargs.get("wireframe_color", "black")
    wire_width = float(plot_kwargs.get("wireframe_width", 1.5))
    wire_open_color = plot_kwargs.get("wireframe_openings_color", wire_color)
    wire_open_width = float(plot_kwargs.get("wireframe_openings_width", max(0.5, wire_width * 0.7)))
    wire_show_vertices = bool(plot_kwargs.get("wireframe_show_vertices", False))
    bg_color = background_color or plot_kwargs.get("background_color", "white")
    aspectmode = plot_kwargs.get("aspectmode", "data")

    if grid is None:
        cols, rows = _auto_grid(len(models), figsize, total_width)
    else:
        cols, rows = int(grid[0]), int(grid[1])
        max_cells = max(1, cols * rows)
        if max_cells < len(models):
            models = models[:max_cells]
    cell_w, cell_h = _per_cell_size(cols, figsize, total_width, base_model_height)
    total_h = rows * cell_h

    titles = _default_titles(models, model_titles)

    def _load_truetype_font(size: int):
        # Try several common fonts across platforms so the title can be drawn large
        candidates = [
            "DejaVuSans.ttf",
            "Arial.ttf",
            "LiberationSans-Regular.ttf",
            "FreeSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/Library/Fonts/Arial.ttf",
            "C:/Windows/Fonts/Arial.ttf",
        ]
        for name in candidates:
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        return None

    title_font_size = max(18, int(min(total_width, total_h) / 16))
    title_font = _load_truetype_font(title_font_size)
    if title_font is None:
        title_font = ImageFont.load_default()
        title_font_size = getattr(title_font, "size", 12) * max(6, title_font_size // 12)

    title_h = 0
    if main_title:
        test_img = Image.new("RGBA", (10, 10))
        test_draw = ImageDraw.Draw(test_img)
        try:
            bbox = test_draw.textbbox((0, 0), str(main_title), font=title_font)
            title_h = bbox[3] - bbox[1] + 32  # base padding for title
        except Exception:
            try:
                _, h = test_draw.textsize(str(main_title), font=title_font)
                title_h = h + 32
            except Exception:
                title_h = title_font_size + 32
        title_grid_margin = max(20, title_font_size // 2)  # breathing room below title
        title_h += title_grid_margin

    legend_h = 0
    legend_padding = 10
    legend_item_gap = 12
    legend_square = 12
    try:
        legend_font_size = max(12, int(title_font_size * 0.55))
        legend_font = ImageFont.truetype("DejaVuSans.ttf", legend_font_size)
    except Exception:
        legend_font = ImageFont.load_default()
        legend_font_size = getattr(legend_font, "size", 12)

    default_order = [
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
    keys_in_map = list(COLOR_MAP.keys())
    ordered = []
    for k in default_order:
        if k in keys_in_map and k not in ordered:
            ordered.append(k)
    for k in keys_in_map:
        if k not in ordered:
            ordered.append(k)
    legend_names = legend_labels if legend_labels is not None else ordered

    if plot_main_legend:
        n_items = len(legend_names)
        target_col_w = 220  # target width per legend column
        safe_width = max(1, int(total_width))
        num_cols = max(1, min(4, n_items, safe_width // target_col_w if target_col_w else 1))
        per_col = int(np.ceil(n_items / float(num_cols))) if num_cols else n_items
        test_img = Image.new("RGBA", (10, 10))
        test_draw = ImageDraw.Draw(test_img)
        max_txt_w = 0
        max_txt_h = 0
        for name in legend_names:
            txt = str(name)
            try:
                bbox = test_draw.textbbox((0, 0), txt, font=legend_font)
                txt_w = bbox[2] - bbox[0]
                txt_h = bbox[3] - bbox[1]
            except Exception:
                try:
                    txt_w, txt_h = test_draw.textsize(txt, font=legend_font)
                except Exception:
                    txt_w = int(len(txt) * legend_font_size * 0.6)
                    txt_h = legend_font_size
            max_txt_w = max(max_txt_w, txt_w)
            max_txt_h = max(max_txt_h, txt_h)
        item_h = max(legend_square, max_txt_h) + 6
        legend_h = per_col * item_h + 2 * legend_padding
    else:
        num_cols = 0
        per_col = 0
        item_h = 0




    canvas_w = int(total_width)
    canvas_h = int(title_h + total_h + legend_h)
    try:
        big_img = Image.new("RGBA", (canvas_w, canvas_h), color=bg_color)
    except Exception:
        big_img = Image.new("RGBA", (canvas_w, canvas_h), color=(255, 255, 255, 255))

    for i, m in enumerate(models):
        r, c = divmod(i, cols)
        p = pv.Plotter(off_screen=True, window_size=(int(cell_w), int(cell_h)))
        p.set_background(bg_color)
        try:
            p.enable_depth_peeling(occlusion_ratio=0.1, number_of_peels=100)
        except Exception:
            pass

        cats, edges_struct, edges_openings = collect_model_geometry(
            m,
            extrude_eps=extrude_eps,
            include_types=include_types,
            room_ids=room_ids,
            want_wire=show_wireframe,
        )

        SHADE_KEYS = {"outdoor_shade", "indoor_shade", "shade_mesh", "shade"}

        for key in sorted(cats.keys()):
            cat = cats[key]
            if not cat["pts"] or not cat["I"]:
                continue
            pd = _polydata_from_category(cat)
            if pd is None:
                continue

            is_shade = key in SHADE_KEYS
            base_color = COLOR_MAP.get(key, COLOR_MAP["default"])
            op = float(np.clip(shades_opacity if is_shade else surface_opacity, 0.0, 1.0))

            if not is_shade:
                p.add_mesh(
                    pd,
                    color=base_color,
                    opacity=op,
                    smooth_shading=False,
                    lighting=False,
                    show_edges=False,
                    edge_color="black",
                    line_width=1.0,
                    reset_camera=False,
                    show_vertices=False,
                    render_points_as_spheres=False,
                    point_size=0,
                )
            else:
                # Opposite offsets keep thin shades visible from both sides.
                op_half = op * 0.5

                a1 = p.add_mesh(
                    pd,
                    color=base_color,
                    opacity=op_half,
                    smooth_shading=False,
                    lighting=False,
                    show_edges=False,
                    reset_camera=False,
                    show_vertices=False,
                    render_points_as_spheres=False,
                    point_size=0,
                )
                try:
                    m1 = a1.mapper
                    m1.SetResolveCoincidentTopologyToPolygonOffset()
                    m1.SetResolveCoincidentTopologyPolygonOffsetParameters(1, 1)
                except Exception:
                    pass

                a2 = p.add_mesh(
                    pd,
                    color=base_color,
                    opacity=op_half,
                    smooth_shading=False,
                    lighting=False,
                    show_edges=False,
                    reset_camera=False,
                    show_vertices=False,
                    render_points_as_spheres=False,
                    point_size=0,
                )
                try:
                    m2 = a2.mapper
                    m2.SetResolveCoincidentTopologyToPolygonOffset()
                    m2.SetResolveCoincidentTopologyPolygonOffsetParameters(-1, -1)
                except Exception:
                    pass

        if show_wireframe:
            if edges_struct:
                wpd_s = _wire_polydata(edges_struct)
                if wpd_s is not None:
                    p.add_mesh(
                        wpd_s,
                        color=wire_color,
                        line_width=wire_width,
                        render_lines_as_tubes=False,
                        show_vertices=wire_show_vertices,
                        render_points_as_spheres=False,
                        point_size=0,
                        lighting=False,
                    )
            if edges_openings:
                wpd_o = _wire_polydata(edges_openings)
                if wpd_o is not None:
                    p.add_mesh(
                        wpd_o,
                        color=wire_open_color,
                        line_width=wire_open_width,
                        render_lines_as_tubes=False,
                        show_vertices=wire_show_vertices,
                        render_points_as_spheres=False,
                        point_size=0,
                        lighting=False,
                    )

        if isinstance(view, str):
            preset = VIEW_PRESETS.get(view.lower(), "iso")
            if preset == "iso":
                p.view_isometric()
            else:
                elev, azim = preset  # type: ignore[assignment]
                try:
                    p.camera.elevation(float(elev))
                    p.camera.azimuth(float(azim))
                except Exception:
                    p.view_isometric()
        elif isinstance(view, (tuple, list)) and len(view) == 2:
            elev, azim = view
            try:
                p.camera.elevation(float(elev))
                p.camera.azimuth(float(azim))
            except Exception:
                p.view_isometric()
        else:
            p.view_isometric()

        # PyVista zoom values above one zoom in, opposite to this function's API.
        p.camera.zoom(1.0 / float(camera_zoom))

        if aspectmode == "cube":
            b = p.bounds
            if b:
                sx, sy, sz = b[1] - b[0], b[3] - b[2], b[5] - b[4]
                s = max(sx, sy, sz)
                cx, cy, cz = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2
                cube = pv.Cube(center=(cx, cy, cz), x_length=s, y_length=s, z_length=s)
                p.add_mesh(cube, opacity=0.0, reset_camera=False)

        if titles[i]:
            p.add_text(
                str(titles[i]),
                position="upper_left",
                font_size=max(10, int(0.8 * min(cell_w, cell_h) / 40)),
            )

        cell_arr = p.screenshot(return_img=True)
        p.close()
        cell_img = Image.fromarray(cell_arr).convert("RGBA")
        x0 = int(c * cell_w)
        y0 = int(title_h + r * cell_h)
        big_img.paste(cell_img, (x0, y0), cell_img)

    if main_title:
        draw = ImageDraw.Draw(big_img)
        font = title_font
        txt = str(main_title)
        try:
            bbox = draw.textbbox((0, 0), txt, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            try:
                w, h = draw.textsize(txt, font=font)
            except Exception:
                w = int(len(txt) * (getattr(font, "size", 12) * 0.6))
                h = getattr(font, "size", 12)
        x = max(20, (canvas_w - w) // 2)
        y = max(12, (title_h - h) // 2)
        draw.text((x, y), txt, fill="black", font=font)

    if plot_main_legend:
        draw = ImageDraw.Draw(big_img)
        names = list(legend_names)
        start_y = int(title_h + total_h + legend_padding)
        col_w = int((canvas_w - 2 * legend_padding) / float(num_cols)) if num_cols > 0 else canvas_w - 2 * legend_padding
        for col in range(max(1, num_cols)):
            col_x = legend_padding + col * (col_w if num_cols > 0 else (canvas_w - 2 * legend_padding))
            for row_i in range(per_col if num_cols > 0 else len(names)):
                idx = col * (per_col if num_cols > 0 else len(names)) + row_i
                if idx >= len(names):
                    break
                name = names[idx]
                label = str(name)
                try:
                    bbox = draw.textbbox((0, 0), label, font=legend_font)
                    txt_w = bbox[2] - bbox[0]
                    txt_h = bbox[3] - bbox[1]
                except Exception:
                    try:
                        txt_w, txt_h = draw.textsize(label, font=legend_font)
                    except Exception:
                        txt_w = int(len(label) * legend_font_size * 0.6)
                        txt_h = legend_font_size
                item_y = start_y + row_i * item_h if num_cols > 0 else start_y
                if num_cols == 0:
                    item_y = start_y + idx * (max(legend_square, txt_h) + 6)
                sq_x0 = col_x
                sq_y0 = item_y + (max(legend_square, txt_h) - legend_square) // 2
                sq_x1 = sq_x0 + legend_square
                sq_y1 = sq_y0 + legend_square
                color = COLOR_MAP.get(name, COLOR_MAP.get("default", "#cccccc"))
                try:
                    draw.rectangle([sq_x0, sq_y0, sq_x1, sq_y1], fill=color, outline="black")
                except Exception:
                    draw.rectangle([sq_x0, sq_y0, sq_x1, sq_y1], fill=(200, 200, 200), outline="black")
                tx = sq_x1 + 6
                ty = sq_y0 + max(0, (legend_square - txt_h) // 2)
                draw.text((tx, ty), label, fill="black", font=legend_font)

    fig: Image.Image = big_img

    if show:
        try:
            from IPython.display import display as _display
            _display(fig)
        except Exception:
            try:
                fig.show()
            except Exception:
                pass
        return None

    return fig
