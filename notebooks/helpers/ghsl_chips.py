"""Clip raster layers to per-city chips, and preview them.

Stage 1 cuts every raster layer to the window of each city, the 2x2 km box around
its centre: GHSL Built-C (10 m settlement classes, a tile set), Built-H (100 m
average net building height, one global raster), AGE (100 m construction epoch,
one global raster) and the Copernicus building heights (10 m, one file per city
area). The layers differ only in the layout of their source files, their dtype and
their palette, so the clipping, the on-disk caching and the previews are shared.

Every layer keeps its own CRS and nodata value.

Every chip is cut on its source's own pixel grid: the city window is expanded
outward to whole source pixels before reading, so each chip pixel is exactly one
source pixel, at its true position. Pixels that do not touch the OSM window are
then set to nodata, so a chip holds data for exactly that window. Chips carry their
grid version (`CHIP_GRID_TAG`) and window (`CHIP_WINDOW_TAG`); a chip without the
current ones is rebuilt.

`north_up_crs`, `draw_chip` and `plot_chip_alignment` show chips of different grids
together, north-up and pixel by pixel, without resampling them.

Built-C is read from JRC's 1000 km tiles. `tiles_for_cities`, `tile_status` and
`tile_download_table` list the tiles the city windows need, with download links.

Import from a notebook as::

    from helpers.ghsl_chips import build_city_chips, preview_chips, BUILTC_CMAP

"""

from __future__ import annotations

import contextlib
import html
import math
import os
from collections.abc import Mapping
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.errors import WindowError
from rasterio.features import geometry_mask
from rasterio.merge import merge
from rasterio.warp import transform_bounds, transform_geom
from shapely.geometry import box, mapping
from rasterio.windows import Window, from_bounds

from .paths import display_path
from .utils import bbox_around_latlon, hex_to_rgba, rgb255_to_rgba

__all__ = [
    # palettes
    "BUILTC_COLORS",
    "BUILTC_LABELS",
    "BUILTC_CMAP",
    "BUILTC_NORM",
    "BUILTC_NODATA",
    "AGE_COLORS",
    "AGE_LABELS",
    "AGE_CMAP",
    "AGE_NORM",
    # chips
    "build_city_chips",
    "chips_to_write",
    "preview_chips",
    "chip_is_complete",
    "CHIP_GRID_TAG",
    "CHIP_WINDOW_TAG",
    "window_tag",
    "window_polygon",
    "crop_to_window",
    # display
    "north_up_crs",
    "draw_chip",
    "plot_chip_alignment",
    # tiles
    "GHSL_GRID_ORIGIN",
    "GHSL_TILE_SIZE_M",
    "tiles_for_cities",
    "tile_status",
    "tile_download_table",
]

CRS_WGS84 = CRS.from_epsg(4326)

#: GeoTIFF tag of every chip cut on its source pixel grid and cropped to the OSM
#: window. A chip with another value, or none, is rebuilt instead of reused.
CHIP_GRID_TAG = "BEM4AI_CHIP_GRID"
CHIP_GRID_VERSION = "source-aligned-cropped-2"
UNCROPPED_GRID_VERSION = "source-aligned-uncropped-2"   # chips cut with `crop=False`

#: GeoTIFF tag holding the window a chip was cut for, as written by `window_tag`.
#: A chip whose window has moved is rebuilt instead of reused.
CHIP_WINDOW_TAG = "BEM4AI_WINDOW"


def window_tag(lat, lon, km_half=1.0) -> str:
    """The window of a city as ``"<lat>,<lon>,<km_half>"``: centre in WGS84 degrees, 7 decimals."""
    return f"{float(lat):.7f},{float(lon):.7f},{float(km_half):g}"


def _chip_tags(lat, lon, km_half, crop, tags):
    return {**(tags or {}), CHIP_GRID_TAG: CHIP_GRID_VERSION if crop else UNCROPPED_GRID_VERSION,
            CHIP_WINDOW_TAG: window_tag(lat, lon, km_half)}


# --- Built-C palette (GHS_BUILT_C_MSZ classes; undefined codes stay transparent) ---
BUILTC_NODATA = 255

BUILTC_COLORS = [
    (0, 0, 0, 0),             # 0  unused
    hex_to_rgba("#B9E3B5"),   # 1  open space, low vegetation
    hex_to_rgba("#6FBE6B"),   # 2  open space, medium vegetation
    hex_to_rgba("#2E7D32"),   # 3  open space, high vegetation
    hex_to_rgba("#4F8DDF"),   # 4  open space, water
    hex_to_rgba("#fa382a"),   # 5  open space, roads
    *[(0, 0, 0, 0)] * 5,      # 6-10 unused
    hex_to_rgba("#FDD835"),   # 11 residential <= 3 m      (gold ->)
    hex_to_rgba("#F6C026"),   # 12 residential 3-6 m
    hex_to_rgba("#E6951C"),   # 13 residential 6-15 m
    hex_to_rgba("#B86B0B"),   # 14 residential 15-30 m
    hex_to_rgba("#804528"),   # 15 residential > 30 m      (-> brown)
    *[(0, 0, 0, 0)] * 5,      # 16-20 unused
    hex_to_rgba("#F8BBD0"),   # 21 non-residential <= 3 m  (pink ->)
    hex_to_rgba("#F48FB1"),   # 22 non-residential 3-6 m
    hex_to_rgba("#F06292"),   # 23 non-residential 6-15 m
    hex_to_rgba("#BA68C8"),   # 24 non-residential 15-30 m
    hex_to_rgba("#8e2ea3"),   # 25 non-residential > 30 m  (-> violet)
]

BUILTC_LABELS = {
    1: "Open spaces, low vegetation (NDVI <= 0.3)",
    2: "Open spaces, medium vegetation (0.3 < NDVI <= 0.5)",
    3: "Open spaces, high vegetation (NDVI > 0.5)",
    4: "Open spaces, water (LAND < 0.5)",
    5: "Open spaces, roads",
    11: "Residential <= 3 m",
    12: "Residential 3-6 m",
    13: "Residential 6-15 m",
    14: "Residential 15-30 m",
    15: "Residential > 30 m",
    21: "Non-residential <= 3 m",
    22: "Non-residential 3-6 m",
    23: "Non-residential 6-15 m",
    24: "Non-residential 15-30 m",
    25: "Non-residential > 30 m",
}

BUILTC_CMAP = ListedColormap(BUILTC_COLORS, name="ghsl_builtc")
BUILTC_NORM = BoundaryNorm(np.arange(-0.5, 26.5, 1), BUILTC_CMAP.N)


# --- AGE palette (GHS_AGE construction epochs, official turbo-like ramp) ---
AGE_COLORS = [
    rgb255_to_rgba(48, 18, 59),     # 0  not built-up
    rgb255_to_rgba(68, 88, 203),    # 1  < 1975
    rgb255_to_rgba(62, 155, 254),   # 2  1975-1980
    rgb255_to_rgba(24, 213, 204),   # 3  1980-1985
    rgb255_to_rgba(70, 247, 131),   # 4  1985-1990
    rgb255_to_rgba(164, 252, 59),   # 5  1990-1995
    rgb255_to_rgba(225, 220, 55),   # 6  1995-2000
    rgb255_to_rgba(253, 163, 48),   # 7  2000-2005
    rgb255_to_rgba(239, 90, 17),    # 8  2005-2010
    rgb255_to_rgba(195, 36, 2),     # 9  2010-2015
    rgb255_to_rgba(122, 4, 2),      # 10 2015-2020
]

AGE_LABELS = {
    0: "not built-up",
    1: "< 1975",
    2: "1975-1980",
    3: "1980-1985",
    4: "1985-1990",
    5: "1990-1995",
    6: "1995-2000",
    7: "2000-2005",
    8: "2005-2010",
    9: "2010-2015",
    10: "2015-2020",
}

AGE_CMAP = ListedColormap(AGE_COLORS, name="ghsl_age")
AGE_NORM = BoundaryNorm(np.arange(-0.5, len(AGE_COLORS) + 0.5, 1), AGE_CMAP.N)


# --- chip I/O ---


def _chip_status(path, expected_tags=None) -> str:
    """'current', 'outdated' (opens, but without the `expected_tags`) or 'missing'.

    `expected_tags` defaults to the current `CHIP_GRID_TAG` of a cropped chip. A chip
    that is absent, empty or truncated counts as missing.
    """
    path = Path(path)
    expected = expected_tags or {CHIP_GRID_TAG: CHIP_GRID_VERSION}
    if not path.is_file() or path.stat().st_size == 0:
        return "missing"
    try:
        with rasterio.open(path) as ds:
            if not (ds.count >= 1 and ds.width > 0 and ds.height > 0):
                return "missing"
            tags = ds.tags()
            return "current" if all(tags.get(k) == v for k, v in expected.items()) else "outdated"
    except Exception:
        return "missing"


def chips_to_write(cities: pd.DataFrame, out_dir, suffix: str, *, km_half: float = 1.0,
                   tags: dict | None = None, crop: bool = True, overwrite: bool = False) -> list:
    """`city_key` of every city whose chip `build_city_chips` would write with these arguments:
    missing, unreadable, cut for another window, or without the current tags."""
    out_dir = Path(out_dir)
    return [key for key, lat, lon in zip(cities["city_key"], cities["lat"], cities["lon"])
            if overwrite or _chip_status(out_dir / f"{key}_{suffix}.tif",
                                         _chip_tags(lat, lon, km_half, crop, tags)) != "current"]


def chip_is_complete(path, tags: dict | None = None) -> bool:
    """True if a cropped per-city chip is on disk, opens cleanly and carries the current
    grid tag and the `tags` passed to `build_city_chips`."""
    return _chip_status(path, {**(tags or {}), CHIP_GRID_TAG: CHIP_GRID_VERSION}) == "current"


def write_chip_atomically(out_path, profile, array, tags=None) -> str:
    """Write a single-band GeoTIFF through a temporary file, then rename it."""
    out_path = str(out_path)
    tmp_path = out_path + ".tmp"
    try:
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(array, 1)
            if tags:
                dst.update_tags(**tags)
        os.replace(tmp_path, out_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return out_path


def _wgs84_bbox_to_bounds(bbox_wgs84, target_crs):
    """Project a WGS84 bbox into the raster CRS, as an axis-aligned bounding box."""
    return transform_bounds(CRS_WGS84, target_crs, *bbox_wgs84, densify_pts=21)


def _grid_of(transform):
    """``(x0, y0, xres, yres)`` of a north-up pixel grid; anything else is rejected."""
    if transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0:
        raise RuntimeError(f"Expected a north-up raster without rotation, got {transform}")
    return transform.c, transform.f, transform.a, -transform.e


def _snap_bounds(bounds, grid, tol=1e-6):
    """Expand `bounds` outward to the pixel edges of `grid`.

    A window starting inside a pixel gives chip pixels that straddle two source
    pixels and hold the value of one of them at a shifted position.
    """
    x0, y0, xres, yres = grid
    minx, miny, maxx, maxy = bounds
    col_start = math.floor((minx - x0) / xres + tol)
    col_end = math.ceil((maxx - x0) / xres - tol)
    row_start = math.floor((y0 - maxy) / yres + tol)
    row_end = math.ceil((y0 - miny) / yres - tol)
    return (x0 + col_start * xres, y0 - row_end * yres,
            x0 + col_end * xres, y0 - row_start * yres)


def _on_grid(transform, grid, tol=1e-6) -> bool:
    """True if `transform` has the grid's resolution and starts on one of its pixel edges."""
    x0, y0, xres, yres = grid
    cols = (transform.c - x0) / xres
    rows = (y0 - transform.f) / yres
    return (math.isclose(transform.a, xres, rel_tol=tol)
            and math.isclose(-transform.e, yres, rel_tol=tol)
            and transform.b == 0 and transform.d == 0
            and abs(cols - round(cols)) < tol and abs(rows - round(rows)) < tol)


def _list_tifs(base: Path) -> list[Path]:
    tifs = sorted(base.rglob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No .tif files found under {base}")
    return tifs


def _clip_tiles(tile_bounds, bounds, nodata, res):
    """Mosaic the tiles overlapping grid-snapped `bounds` into one array (Built-C)."""
    minx, miny, maxx, maxy = bounds
    overlapping = [
        path for path, b in tile_bounds
        if not (maxx <= b.left or minx >= b.right or maxy <= b.bottom or miny >= b.top)
    ]
    if not overlapping:
        return None
    with contextlib.ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(p)) for p in overlapping]
        # `bounds` lie on the tile grid and `res` is the tile resolution, so merge
        # copies whole pixels instead of resampling onto a grid of its own.
        mosaic, transform = merge(sources, bounds=bounds, res=res, nodata=nodata,
                                  method="first")
        profile = sources[0].profile.copy()
    return mosaic[0], transform, profile


def _clip_window(source: Path, bounds):
    """Read grid-snapped `bounds` as a clamped window from one global raster (Built-H, AGE)."""
    with rasterio.open(source) as ds:
        window = from_bounds(*bounds, ds.transform)
        # `bounds` lie on the raster grid, so rounding only removes float noise and
        # the read copies whole pixels; a fractional window would be resampled.
        window = Window(round(window.col_off), round(window.row_off),
                        round(window.width), round(window.height))
        try:
            window = window.intersection(Window(0, 0, ds.width, ds.height))
        except WindowError:  # the city lies outside the raster
            return None
        if window.width <= 0 or window.height <= 0:
            return None
        return ds.read(1, window=window), ds.window_transform(window), ds.profile.copy()


def window_polygon(lat: float, lon: float, km_half: float = 1.0, step_deg: float = 0.0002):
    """The OSM window: the ``km_half`` box in degrees (WGS84), edges densified for reprojection."""
    return box(*bbox_around_latlon(float(lat), float(lon), km_half)).segmentize(step_deg)


def crop_to_window(array, transform, crs, window, fill):
    """Copy of `array` with every pixel that does not touch `window` (WGS84) set to `fill`.

    The grid is unchanged, so kept pixels keep their exact values and positions,
    and the kept pixels cover the whole window.
    """
    shape = transform_geom(CRS_WGS84, crs, mapping(window))
    keep = geometry_mask([shape], out_shape=array.shape, transform=transform,
                         all_touched=True, invert=True)
    out = array.copy()
    out[~keep] = fill
    return out


def north_up_crs(lat: float, lon: float) -> CRS:
    """Transverse Mercator centred on the city: north is up within 0.02 deg over a
    2x2 km window, and the frame is conformal and true to scale there."""
    return CRS.from_proj4(f"+proj=tmerc +lat_0={lat} +lon_0={lon} +k=1 +x_0=0 +y_0=0 "
                          f"+ellps=GRS80 +units=m +no_defs")


def draw_chip(ax, path, frame_crs=None, mask_values=(), **style):
    """Draw a chip pixel by pixel; returns the mesh for a colorbar.

    Without `frame_crs` the chip is drawn in its own CRS (square pixels). With it,
    every pixel is drawn as its exact quadrilateral in that frame, so chips of
    different grids can be compared without resampling them. Nodata and the values
    in `mask_values` are left empty.
    """
    with rasterio.open(path) as ds:
        array, t, crs, nodata = ds.read(1), ds.transform, ds.crs, ds.nodata
    hidden = np.isin(array, list(mask_values))
    if nodata is not None:
        hidden |= array == nodata
    rows, cols = np.mgrid[0:array.shape[0] + 1, 0:array.shape[1] + 1]
    xs, ys = t.c + cols * t.a, t.f + rows * t.e
    if frame_crs is not None:
        xs, ys = Transformer.from_crs(crs, frame_crs, always_xy=True).transform(xs, ys)
    return ax.pcolormesh(xs, ys, np.ma.masked_array(array, hidden), shading="flat", **style)


def _clip_to(artist, ax, polygon):
    """Draw `artist` only inside `polygon` (in the axes' data coordinates)."""
    patch = mpatches.PathPatch(MplPath(np.asarray(polygon.exterior.coords)), transform=ax.transData)
    artist.set_clip_path(patch)


def plot_chip_alignment(lat, lon, buildings, layers, *, km_half: float = 1.0,
                        zoom_m: float = 400.0, title: str = ""):
    """Figure with four rows for one city; returns the figure.

    Rows: every chip on its own grid with the OSM window and footprints transformed
    into it; all chips north-up before cropping; north-up, cropped and shown only
    inside the OSM window; and a `zoom_m` close-up of the centre. `layers` holds
    ``(title, uncropped_chip_or_None, cropped_chip, style)``; `style` goes to
    `draw_chip`. `buildings` are the OSM footprints (any CRS).
    """
    window = gpd.GeoSeries([window_polygon(lat, lon, km_half)], crs=4326)
    frame = north_up_crs(lat, lon)
    window_f = window.to_crs(frame).iloc[0]
    fp = buildings.to_crs(frame)
    fig, axes = plt.subplots(4, len(layers) + 1, figsize=(4.0 * (len(layers) + 1), 16.5),
                             constrained_layout=True)

    # Row 1: own grids
    buildings.to_crs(4326).plot(ax=axes[0, 0], color="#333333", lw=0)
    window.boundary.plot(ax=axes[0, 0], color="#C4071B", lw=1.2)
    axes[0, 0].set_aspect(1 / math.cos(math.radians(lat)))
    axes[0, 0].set_title("OSM footprints")
    extents = [window_f.bounds]
    for j, (name, uncropped, cropped, style) in enumerate(layers, start=1):
        chip = uncropped or cropped
        with rasterio.open(chip) as ds:
            crs, b = ds.crs, ds.bounds
        draw_chip(axes[0, j], chip, **style)
        buildings.to_crs(crs).boundary.plot(ax=axes[0, j], color="black", lw=0.15)
        window.to_crs(crs).boundary.plot(ax=axes[0, j], color="#C4071B", lw=1.2)
        axes[0, j].set_aspect("equal")
        axes[0, j].set_title(name)
        extents.append(gpd.GeoSeries([box(*b)], crs=crs).segmentize(20).to_crs(frame).total_bounds)

    # Rows 2-4: north-up
    full = (min(e[0] for e in extents) - 60, max(e[2] for e in extents) + 60,
            min(e[1] for e in extents) - 60, max(e[3] for e in extents) + 60)
    w = window_f.bounds
    square = (w[0] - 60, w[2] + 60, w[1] - 60, w[3] + 60)
    cx, cy = window_f.centroid.x, window_f.centroid.y
    zoom = (cx - zoom_m / 2, cx + zoom_m / 2, cy - zoom_m / 2, cy + zoom_m / 2)
    fp_clipped = gpd.clip(fp, window_f)
    for r, limits, clip in [(1, full, False), (2, square, True), (3, zoom, True)]:
        shown = fp_clipped if clip else fp
        shown.plot(ax=axes[r, 0], color="#333333", lw=0)
        for j, (name, uncropped, cropped, style) in enumerate(layers, start=1):
            chip = uncropped if r == 1 else cropped
            if chip is None:
                axes[r, j].text(0.5, 0.5, "source raster not available", ha="center",
                                transform=axes[r, j].transAxes)
                continue
            mesh = draw_chip(axes[r, j], chip, frame_crs=frame, **style)
            if clip:
                _clip_to(mesh, axes[r, j], window_f)
            shown.boundary.plot(ax=axes[r, j], color="white" if r == 3 else "black",
                                lw=0.9 if r == 3 else 0.15)
        for ax in axes[r]:
            gpd.GeoSeries([window_f]).boundary.plot(ax=ax, color="#C4071B", lw=1.2)
            ax.set_xlim(limits[0], limits[1])
            ax.set_ylim(limits[2], limits[3])
            ax.set_aspect("equal")
    for r, label in enumerate(["own grid of each source", "north-up, before cropping",
                               "north-up, cropped (saved)", f"north-up, cropped, {zoom_m:.0f} m zoom"]):
        axes[r, 0].set_ylabel(label, fontsize=12)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{title}: red = OSM window; outlines = OSM footprints", fontsize=13)
    return fig


#: GHSL tiling of its Mollweide (ESRI:54009) grids: 1000 x 1000 km tiles named
#: ``R<row>_C<col>``, numbered from 1 at the grid's top-left corner. JRC trims each
#: tile to its data and publishes no tile where there is none.
GHSL_GRID_ORIGIN = (-18041000.0, 9000000.0)
GHSL_TILE_SIZE_M = 1_000_000.0


def tiles_for_cities(cities: pd.DataFrame, *, crs="ESRI:54009", res: float = 10.0,
                     km_half: float = 1.0) -> pd.DataFrame:
    """GHSL tiles under the chip windows of `cities`, one row per tile.

    The window is the one `build_city_chips` reads: the ``km_half`` degree box
    projected into `crs` and widened to the `res` pixel grid. Returns the columns
    ``tile`` (``"R4_C19"``) and ``cities`` (list of `city_key`), north to south and
    west to east.
    """
    x0, y0 = GHSL_GRID_ORIGIN
    size = GHSL_TILE_SIZE_M
    crs = CRS.from_user_input(crs)
    per_tile: dict[tuple[int, int], list] = {}
    for key, lat, lon in zip(cities["city_key"], cities["lat"], cities["lon"]):
        minx, miny, maxx, maxy = _snap_bounds(
            _wgs84_bbox_to_bounds(bbox_around_latlon(float(lat), float(lon), km_half), crs),
            (x0, y0, res, res),
        )
        for row in range(math.floor((y0 - maxy) / size) + 1, math.ceil((y0 - miny) / size) + 1):
            for col in range(math.floor((minx - x0) / size) + 1, math.ceil((maxx - x0) / size) + 1):
                per_tile.setdefault((row, col), []).append(key)
    return pd.DataFrame(
        [{"tile": f"R{row}_C{col}", "cities": keys} for (row, col), keys in sorted(per_tile.items())],
        columns=["tile", "cities"],
    )


def tile_status(tiles: pd.DataFrame, tile_dir, product: str) -> pd.DataFrame:
    """Add ``on_disk`` to `tiles`: True where ``<product>_<tile>.tif`` lies anywhere under `tile_dir`.

    `product` is the release name the tile files start with, e.g.
    ``GHS_BUILT_C_MSZ_E2018_GLOBE_R2023A_54009_10_V1_0``.
    """
    tile_dir = Path(tile_dir)
    names = {path.name for path in tile_dir.rglob("*.tif")} if tile_dir.is_dir() else set()
    status = tiles.copy()
    status["on_disk"] = [f"{product}_{tile}.tif" in names for tile in status["tile"]]
    return status


def tile_download_table(status: pd.DataFrame, base_url: str, product: str,
                        max_cities: int = 6) -> str:
    """HTML table of `status` whose tile names link to their zips under `base_url`."""
    rows = []
    for tile, cities, on_disk in zip(status["tile"], status["cities"], status["on_disk"]):
        url = html.escape(f"{base_url.rstrip('/')}/{product}_{tile}.zip", quote=True)
        shown = ", ".join(html.escape(str(c)) for c in cities[:max_cities])
        more = f" and {len(cities) - max_cities} more" if len(cities) > max_cities else ""
        state, colour = ("yes", "inherit") if on_disk else ("missing", "#C4071B")
        rows.append(
            f'<tr><td><a href="{url}" target="_blank" rel="noopener">{tile}</a></td>'
            f'<td style="color:{colour}">{state}</td>'
            f"<td style='text-align:right'>{len(cities)}</td>"
            f"<td style='text-align:left'>{shown}{more}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Tile (click to download)</th><th>On disk</th>"
        "<th>Cities</th><th style='text-align:left'>City keys</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def build_city_chips(
    cities: pd.DataFrame,
    source,
    out_dir,
    suffix: str,
    *,
    label: str | None = None,
    km_half: float = 1.0,
    nodata=None,
    dtype=None,
    overwrite: bool = False,
    crop: bool = True,
    verbose: bool = True,
    tags: dict | None = None,
) -> pd.Series:
    """Clip one raster layer to a chip around every city in `cities`.

    `source` is a directory of tiles (mosaicked, as for Built-C), a single
    raster (windowed read, as for Built-H and AGE), or a mapping from `city_key`
    to the raster file or files of that city (as for the Copernicus BBH files;
    several files are mosaicked and must share one pixel grid). Cities missing from
    the mapping get no chip. Chips are named ``<city_key>_<suffix>.tif``; one that
    is already on disk with the current `CHIP_GRID_TAG` is reused, so a restarted
    notebook only writes what is missing, and a run whose chips are all present
    never opens the source rasters at all.

    The window is the ``km_half`` box around the centroid in degrees (the OSM
    window), projected into the layer CRS as an axis-aligned bounding box and then
    expanded outward to whole source pixels. Every chip pixel is exactly one source
    pixel, so a value read at a point is the value of the source pixel under that
    point. With `crop`, pixels that do not touch the OSM window are set to nodata,
    so the chip holds data for exactly that window; the file keeps the source grid.
    `tags` are written into every new chip, e.g. the attribution a licence asks for,
    together with the grid version and the window (`CHIP_WINDOW_TAG`); a chip on disk
    without them, or cut for another window, counts as outdated and is rewritten.

    Returns a Series mapping `city_key` to the chip filename.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = label or suffix
    per_city = isinstance(source, Mapping)
    layers = {}   # source files -> CRS, nodata, grid, tile bounds; read once and only if needed

    def files_for(key):
        if not per_city:
            return (Path(source),)
        value = source.get(key)
        if value is None:
            return None
        return tuple(Path(v) for v in ([value] if isinstance(value, (str, os.PathLike)) else value))

    def open_layer(files):
        if files in layers:
            return layers[files]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"{label}: chips have to be written but {display_path(missing[0])} is missing. "
                f"Download it as described in the notebook."
            )
        tiled = len(files) > 1 or files[0].is_dir()
        paths = _list_tifs(files[0]) if len(files) == 1 and files[0].is_dir() else list(files)
        bounds, crs, grid, layer_nodata = [], None, None, None
        for path in paths:
            with rasterio.open(path) as ds:
                if crs is None:
                    crs, grid, layer_nodata = ds.crs, _grid_of(ds.transform), ds.nodata
                elif ds.crs != crs:
                    raise RuntimeError(f"{path} has CRS {ds.crs}, expected {crs}")
                elif not _on_grid(ds.transform, grid):
                    raise RuntimeError(f"{path} is not on the pixel grid of {paths[0]}")
                bounds.append((path, ds.bounds))
        if crs is None:
            raise RuntimeError(f"CRS missing in {paths[0]}")
        layers[files] = dict(crs=crs, nodata=layer_nodata, grid=grid, tiled=tiled,
                             tile_bounds=bounds, path=paths[0])
        return layers[files]

    names, reused, written, outdated, uncovered = {}, 0, 0, 0, []

    for key, lat, lon in zip(cities["city_key"], cities["lat"], cities["lon"]):
        chip_name = f"{key}_{suffix}.tif"
        out_path = out_dir / chip_name
        chip_tags = _chip_tags(lat, lon, km_half, crop, tags)
        status = _chip_status(out_path, chip_tags)
        if not overwrite and status == "current":
            names[key] = chip_name
            reused += 1
            continue
        files = files_for(key)
        if files is None:
            uncovered.append(key)
            continue
        outdated += status == "outdated"

        layer = open_layer(files)
        crs, grid = layer["crs"], layer["grid"]
        fill = nodata if nodata is not None else layer["nodata"]
        if crop and fill is None:
            raise ValueError(f"{label}: cropping needs a nodata value; pass `nodata`.")
        bounds = _snap_bounds(
            _wgs84_bbox_to_bounds(bbox_around_latlon(float(lat), float(lon), km_half), crs),
            grid,
        )
        clipped = (
            _clip_tiles(layer["tile_bounds"], bounds, fill, res=grid[2:]) if layer["tiled"]
            else _clip_window(layer["path"], bounds)
        )
        if clipped is None:
            uncovered.append(key)
            continue

        array, transform, profile = clipped
        if not _on_grid(transform, grid):
            raise RuntimeError(f"{label} chip of {key} is not on the source pixel grid: {transform}")
        if dtype is not None and array.dtype != dtype:
            array = array.astype(dtype)
        if crop:
            array = crop_to_window(array, transform, crs, window_polygon(lat, lon, km_half), fill)
        profile.update(
            driver="GTiff",
            count=1,
            height=array.shape[0],
            width=array.shape[1],
            dtype=array.dtype,
            transform=transform,
            crs=crs,
            nodata=fill,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        write_chip_atomically(out_path, profile, array, tags=chip_tags)
        names[key] = chip_name
        written += 1

    if verbose:
        print(f"{label} chips in {display_path(out_dir)}: {len(names)} "
              f"(reused {reused}, newly written {written})")
        if outdated:
            print(f"  {outdated} of them replaced chips cut for another window or without the current tags")
        if uncovered:
            print(f"  no raster coverage for {len(uncovered)} cities: "
                  f"{', '.join(uncovered[:10])}{' ...' if len(uncovered) > 10 else ''}")
    return pd.Series(names, dtype="object")


def preview_chips(
    chips: pd.Series,
    chip_dir,
    *,
    cmap,
    norm=None,
    labels=None,
    vmin=None,
    vmax=None,
    colorbar_label=None,
    legend_title=None,
    n: int = 5,
):
    """Show the first `n` chips as miniatures, with a legend or a colorbar."""
    selected = list(chips.items())[:n]
    chip_dir = Path(chip_dir)
    width = 3.4 * len(selected) + (3.0 if labels else 0.0)
    fig, axes = plt.subplots(1, len(selected), figsize=(width, 3.5), constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, (city_key, chip_name) in zip(axes, selected):
        with rasterio.open(chip_dir / chip_name) as ds:
            array, chip_nodata = ds.read(1), ds.nodata
        array = (np.ma.masked_equal(array, chip_nodata) if chip_nodata is not None
                 else np.ma.masked_invalid(array))
        image = ax.imshow(array, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                          interpolation="nearest")
        ax.set_title(city_key, fontsize=9)
        ax.axis("off")

    if labels:
        fig.legend(
            handles=[mpatches.Patch(color=cmap.colors[code], label=text)
                     for code, text in labels.items()],
            loc="center right", bbox_to_anchor=(1.18, 0.5),
            fontsize=7, title=legend_title, title_fontsize=8,
        )
    elif colorbar_label:
        fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04).set_label(colorbar_label)

    plt.show()
