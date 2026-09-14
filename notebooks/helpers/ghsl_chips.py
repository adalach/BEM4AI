"""Clip GHSL raster layers to per-city chips, and preview them.

Stage 1 cuts three GHSL layers to the same 2x2 km window around every city
centroid: Built-C (10 m settlement classes, shipped as a tile set), Built-H
(100 m average net building height, one global raster) and AGE (100 m
construction epoch, one global raster). The three differ only in the layout of
their source files, their dtype and their palette, so the clipping, the on-disk
caching and the previews live here once instead of three times in the notebook.

Every layer keeps its own CRS and nodata value; nothing is carried over from one
layer to the next.

Import from a notebook as::

    from helpers.ghsl_chips import build_city_chips, preview_chips, BUILTC_CMAP

"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from rasterio.crs import CRS
from rasterio.errors import WindowError
from rasterio.merge import merge
from rasterio.warp import transform_bounds
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
    "preview_chips",
    "chip_is_complete",
]

CRS_WGS84 = CRS.from_epsg(4326)


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


def chip_is_complete(path) -> bool:
    """True if a per-city chip is on disk and opens cleanly; a truncated one is not."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as ds:
            return ds.count >= 1 and ds.width > 0 and ds.height > 0
    except Exception:
        return False


def write_chip_atomically(out_path, profile, array) -> str:
    """Write a single-band GeoTIFF through a temporary file, then rename it."""
    out_path = str(out_path)
    tmp_path = out_path + ".tmp"
    try:
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(array, 1)
        os.replace(tmp_path, out_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return out_path


def _wgs84_bbox_to_bounds(bbox_wgs84, target_crs):
    """Project a WGS84 bbox into the raster CRS, as an axis-aligned bounding box."""
    return transform_bounds(CRS_WGS84, target_crs, *bbox_wgs84, densify_pts=21)


def _list_tifs(base: Path) -> list[Path]:
    tifs = sorted(base.rglob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No .tif files found under {base}")
    return tifs


def _clip_tiles(tile_bounds, bounds, nodata):
    """Mosaic the tiles overlapping `bounds` into one array (Built-C)."""
    minx, miny, maxx, maxy = bounds
    overlapping = [
        path for path, b in tile_bounds
        if not (maxx <= b.left or minx >= b.right or maxy <= b.bottom or miny >= b.top)
    ]
    if not overlapping:
        return None
    with contextlib.ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(p)) for p in overlapping]
        mosaic, transform = merge(sources, bounds=bounds, nodata=nodata, method="first")
        profile = sources[0].profile.copy()
    return mosaic[0], transform, profile


def _clip_window(source: Path, bounds):
    """Read `bounds` as a clamped window from one global raster (Built-H, AGE)."""
    with rasterio.open(source) as ds:
        try:
            window = from_bounds(*bounds, ds.transform).intersection(
                Window(0, 0, ds.width, ds.height)
            )
        except WindowError:  # the city lies outside the raster
            return None
        if window.width <= 0 or window.height <= 0:
            return None
        return ds.read(1, window=window), ds.window_transform(window), ds.profile.copy()


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
) -> pd.Series:
    """Clip one GHSL layer to a chip around every city in `cities`.

    `source` is either a directory of tiles (mosaicked, as for Built-C) or a
    single global raster (windowed read, as for Built-H and AGE). Chips are
    named ``<city_key>_<suffix>.tif``; one that is already on disk and opens
    cleanly is reused, so a restarted notebook only writes what is missing, and
    a run whose chips are all present never opens the source rasters at all.

    The window is the ``km_half`` box around the centroid in degrees, projected
    into the layer CRS as an axis-aligned bounding box. Reprojection makes that
    box a little larger than the nominal 2x2 km, by an amount that grows with
    distance from the central meridian, so the chip always covers the OSM
    extract cut from the same degree box.

    Returns a Series mapping `city_key` to the chip filename.
    """
    source = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = label or suffix
    tiled = source.is_dir()

    layer = {}   # filled on the first chip that has to be written

    def open_source():
        """CRS, nodata and (for a tile set) tile bounds, read once and only if needed."""
        if layer:
            return layer
        if tiled:
            bounds, crs = [], None
            for path in _list_tifs(source):
                with rasterio.open(path) as ds:
                    if crs is None:
                        crs = ds.crs
                    elif ds.crs != crs:
                        raise RuntimeError(f"{path} has CRS {ds.crs}, expected {crs}")
                    bounds.append((path, ds.bounds))
            layer.update(crs=crs, nodata=None, tile_bounds=bounds)
        else:
            with rasterio.open(source) as ds:
                layer.update(crs=ds.crs, nodata=ds.nodata, tile_bounds=None)
        if layer["crs"] is None:
            raise RuntimeError(f"CRS missing in {source}")
        return layer

    names, reused, written, uncovered = {}, 0, 0, []

    for key, lat, lon in zip(cities["city_key"], cities["lat"], cities["lon"]):
        chip_name = f"{key}_{suffix}.tif"
        out_path = out_dir / chip_name
        if not overwrite and chip_is_complete(out_path):
            names[key] = chip_name
            reused += 1
            continue

        crs = open_source()["crs"]
        fill = nodata if nodata is not None else layer["nodata"]
        bounds = _wgs84_bbox_to_bounds(
            bbox_around_latlon(float(lat), float(lon), km_half), crs
        )
        clipped = (
            _clip_tiles(layer["tile_bounds"], bounds, fill) if tiled
            else _clip_window(source, bounds)
        )
        if clipped is None:
            uncovered.append(key)
            continue

        array, transform, profile = clipped
        if dtype is not None and array.dtype != dtype:
            array = array.astype(dtype)
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
        write_chip_atomically(out_path, profile, array)
        names[key] = chip_name
        written += 1

    print(f"{label} chips in {display_path(out_dir)}: {len(names)} "
          f"(reused {reused}, newly written {written})")
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
