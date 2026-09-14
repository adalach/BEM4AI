"""Small, project-agnostic helpers shared by the BEM4AI notebooks.

This module deliberately contains only *mechanical* utilities: string cleaning,
missing-value coercion, CRS bookkeeping, polygon unwrapping, edge and point
sampling, pickle I/O and formatting. The substantive pipeline logic (sampling,
decomposition, Honeybee model assembly, result interpretation) lives in the
per-stage helper modules beside this one.

Import from a notebook as::

    from helpers.utils import clean_city_name, metric_crs_for_lonlat

"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from textwrap import wrap
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import (
    LineString,
    MultiPolygon,
    Polygon,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring, unary_union

try:  # shapely >= 2.0
    from shapely.validation import make_valid
except ImportError:  # pragma: no cover - older shapely falls back to buffer(0)
    make_valid = None

__all__ = [
    # text
    "TRANSLIT_MAP",
    "collapse_whitespace",
    "clean_city_name",
    "safe_name",
    "clean_label",
    "slug",
    "shorten",
    # scalars and missing values
    "is_missing",
    "as_scalar",
    "as_int_or_none",
    "as_number_or_none",
    "as_dict",
    "parse_json_like",
    "round_floats",
    "pct",
    "fmt",
    "fmt_int",
    "safe_len",
    "join_reasons",
    "brief",
    # dataframes
    "find_col",
    "assign_ids",
    "check_unique",
    "check_fk",
    "read_pickle_if_exists",
    "series_stats",
    "frame_with_object_geometry_columns",
    # CRS and GeoDataFrames
    "WGS84_EPSG",
    "WGS84_CRS",
    "utm_epsg_for_lonlat",
    "metric_crs_for_lonlat",
    "metric_crs_str_for_lonlat",
    "metric_crs_for_gdf",
    "metric_crs_for_geometry",
    "metric_crs_for_row",
    "ensure_wgs84",
    "empty_gdf",
    "concat_gdfs",
    # geometry
    "polygon_parts",
    "extract_polygons",
    "make_geometry_valid",
    "union_polygons",
    "keep_polygons_min_area",
    "count_exterior_vertices",
    "count_polygon_holes",
    "extract_vertices",
    "min_rotated_rect_dims",
    "aspect_ratio",
    "compactness",
    "total_line_length",
    "clean_ring_coords",
    "iter_cleaned_edges",
    "exterior_edges",
    "trim_segment",
    "clean_polygonal",
    "sample_points_along_linestring",
    "sample_points_on_polygon_exterior",
    "unit_vector",
    "angle_between_deg",
    "haversine_m",
    "bbox_around_latlon",
    # plotting helpers
    "hex_to_rgba",
    "rgb255_to_rgba",
    "wrap_labels",
    "cut_counts",
    "integer_counts",
]


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
#
# Note: repository-root resolution does NOT live here. ``helpers.paths`` owns it:
# it derives ``PROJECT_ROOT`` from its own file location and puts the repository
# on ``sys.path``, so every notebook starts with a single
# ``from helpers.paths import ...`` line. Use ``helpers.paths.display_path`` when
# printing a path.

#: Letters that Unicode decomposition alone does not reduce to ASCII.
TRANSLIT_MAP = str.maketrans(
    {
        "Ł": "L",
        "ł": "l",
        "ß": "ss",
        "Æ": "AE",
        "æ": "ae",
        "Œ": "OE",
        "œ": "oe",
        "Ø": "O",
        "ø": "o",
        "Ð": "D",
        "ð": "d",
        "Þ": "Th",
        "þ": "th",
    }
)


def collapse_whitespace(value: Any) -> str:
    """Strip and collapse internal runs of whitespace to a single space."""
    return " ".join(str(value).strip().split())


def clean_city_name(value: Any) -> str:
    """ASCII, underscore-separated city key (``"Elx / Elche"`` -> ``"Elx_Elche"``).

    Accents are stripped, special letters transliterated, slashes and hyphens
    treated as separators, brackets dropped but their contents kept, and
    apostrophes removed outright.
    """
    if not value:
        return ""

    text = str(value).strip()

    # 1) Decompose and drop combining marks (é -> e, ń -> n, ż -> z).
    decomposed = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))

    # 2) Transliterate letters that decomposition does not handle.
    text = text.translate(TRANSLIT_MAP)

    # 3) Slashes and hyphens are separators.
    text = re.sub(r"[\/\-]+", " ", text)

    # 4) Drop bracket characters but keep what is inside them.
    text = re.sub(r"[\[\]\(\)]", " ", text)

    # 5) Apostrophes vanish (O'Connor -> OConnor).
    text = text.replace("'", "")

    # 6) Any run of non-alphanumerics becomes a single underscore.
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)

    # 7) Collapse repeated underscores and trim the ends.
    return re.sub(r"_+", "_", text).strip("_")


def safe_name(value: Any, fallback: str = "") -> str:
    """Filesystem-safe name: non-word characters collapse to underscores.

    Unlike :func:`clean_city_name` this keeps non-ASCII word characters, so use
    it for file names derived from already-clean labels rather than for keys.
    """
    text = str(value).strip()
    if not text:
        text = str(fallback)
    return re.sub(r"[^\w\-]+", "_", text)


def clean_label(
    value: Any,
    unknown: str = "Unknown",
    replacements: Mapping[str, str] | None = None,
) -> str:
    """Human-readable label, mapping blanks and ``"nan"`` to ``unknown``.

    ``replacements`` applies caller-specific substitutions to the cleaned text,
    e.g. ``{"synthetic / pvc": "PVC"}`` for the material composition tables.
    """
    if not isinstance(value, str):
        return unknown
    text = value.strip()
    if not text or text.lower() == "nan":
        return unknown
    for old, new in (replacements or {}).items():
        text = text.replace(old, new)
    return text


def slug(value: Any, replacements: Mapping[str, str] | None = None) -> str:
    """Underscore-joined identifier built from :func:`clean_label`."""
    return re.sub(r"\s+", "_", clean_label(value, replacements=replacements))


def shorten(text: str, max_chars: int = 1200) -> str:
    """Truncate long text for printing, with an explicit marker."""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated) ..."


# ---------------------------------------------------------------------------
# Scalars and missing values
# ---------------------------------------------------------------------------


def is_missing(value: Any) -> bool:
    """True for ``None``, NaN, and blank/whitespace-only strings."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def as_scalar(value: Any) -> Any:
    """Reduce a numpy scalar or 1-element container to a plain Python value.

    Returns ``None`` when the value is empty or NaN. Useful when a ``.get()``
    on a lookup table may hand back a ``Series`` instead of a scalar.
    """
    if isinstance(value, np.generic):
        value = value.item()
    elif isinstance(value, (pd.Series, pd.Index, np.ndarray, list, tuple)):
        if len(value) == 0:
            return None
        value = value[0]
        if isinstance(value, np.generic):
            value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def as_int_or_none(value: Any) -> int | None:
    """Coerce an identifier to ``int``; ``None`` when it is missing or not a whole number.

    Exists because a pandas column of OSM ids becomes ``float64`` as soon as it
    holds one NaN, so ids arrive as ``1.234e9`` floats that must survive and as
    NaN that must not. A value with a fractional part is rejected rather than
    truncated: an id is either exact or it is not an id.
    """
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(as_float) or not as_float.is_integer():
        return None
    return int(as_float)


def as_number_or_none(value: Any) -> int | float | None:
    """Coerce to ``int``/``float``, accepting comma decimal separators."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return value
    if isinstance(value, str):
        try:
            number = float(value.replace(",", "."))
        except ValueError:
            return None
        return int(number) if number.is_integer() else number
    return None


def as_dict(value: Any, *, strict: bool = False) -> dict:
    """Return ``value`` when it is a dictionary, otherwise an empty one.

    With ``strict=True`` anything that is neither a dictionary nor a missing
    value raises, which is what the composition tables want.
    """
    if isinstance(value, dict):
        return value
    if strict and not (value is None or (isinstance(value, float) and pd.isna(value))):
        raise TypeError(f"Expected a composition dictionary, got {type(value).__name__}.")
    return {}


def parse_json_like(value: Any) -> dict | list | None:
    """Parse a field that may already be a dict, strict JSON, or a Python repr."""
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return json.loads(value.replace("'", '"'))
            except json.JSONDecodeError:
                return None
    return None


def round_floats(obj: Any, ndigits: int = 2) -> Any:
    """Recursively round floats inside nested dicts and lists."""
    if isinstance(obj, dict):
        return {key: round_floats(val, ndigits) for key, val in obj.items()}
    if isinstance(obj, list):
        return [round_floats(val, ndigits) for val in obj]
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), ndigits)
    return obj


def pct(numerator: float, denominator: float) -> float:
    """Percentage, returning ``0.0`` instead of dividing by zero."""
    return 0.0 if not denominator else 100.0 * numerator / denominator


def fmt(value: Any, ndigits: int = 1, missing: str = "n/a") -> str:
    """Fixed-decimal string, or ``missing`` when the value is absent."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return missing
    return f"{value:.{ndigits}f}"


def fmt_int(value: Any, missing: str = "n/a") -> str:
    """Thousands-separated integer string, or ``missing`` when absent."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return missing
    return f"{int(value):,}"


def safe_len(obj: Any, missing: str = "n/a") -> int | str:
    """``len(obj)`` where it is defined, otherwise ``missing``."""
    try:
        return len(obj)
    except TypeError:
        return missing


def join_reasons(values: Any, sep: str = "; ") -> str:
    """Join a list of reason strings; anything else becomes an empty string."""
    if isinstance(values, (list, tuple)):
        return sep.join(str(v) for v in values)
    return ""


def brief(obj: Any) -> str:
    """One-line shape summary for logging tables and frames."""
    if isinstance(obj, gpd.GeoDataFrame):
        return (
            f"GeoDataFrame(rows={len(obj)}, cols={obj.shape[1]}, crs={obj.crs}, "
            f"geom={getattr(obj.geometry, 'name', None)})"
        )
    try:
        return f"{type(obj).__name__}(shape={obj.shape})"
    except AttributeError:
        return type(obj).__name__


# ---------------------------------------------------------------------------
# DataFrames
# ---------------------------------------------------------------------------


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """First column matching one of ``candidates`` (exact, then substring)."""
    wanted = [str(c).lower() for c in candidates]
    for col in df.columns:
        if str(col).lower() in wanted:
            return col
    for col in df.columns:
        if any(token in str(col).lower() for token in wanted):
            return col
    return None


def assign_ids(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """Attach deterministic ``1..N`` integer ids to an already-ordered table."""
    out = df.reset_index(drop=True).copy()
    out.insert(0, id_col, np.arange(1, len(out) + 1, dtype="int64"))
    return out


def check_unique(df: pd.DataFrame, cols: Sequence[str], table: str) -> None:
    """Raise when ``cols`` do not form a unique key over ``df``."""
    if df.empty:
        return
    duplicated = df.duplicated(subset=list(cols), keep=False)
    if duplicated.any():
        sample = df.loc[duplicated, list(cols)].head(10).to_string(index=False)
        raise ValueError(f"{table}: duplicate keys on {list(cols)}:\n{sample}")


def check_fk(
    df: pd.DataFrame,
    col: str,
    parent: pd.DataFrame,
    table: str,
    parent_name: str,
    parent_col: str = "id",
) -> None:
    """Raise when ``df[col]`` references ids absent from ``parent[parent_col]``."""
    if df.empty:
        return
    orphans = set(df[col].dropna().astype("int64")) - set(
        parent[parent_col].astype("int64")
    )
    if orphans:
        raise ValueError(
            f"{table}.{col} references unknown {parent_name} ids: {sorted(orphans)[:10]}"
        )


def read_pickle_if_exists(path: Path | str, *, warn: bool = True):
    """Read a pickle, returning ``None`` if it is missing or unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_pickle(path)
    except Exception as exc:  # noqa: BLE001 - a corrupt cache must not stop a run
        if warn:
            print(f"[WARN] Could not read {path}: {exc}")
        return None


def series_stats(series: pd.Series) -> dict[str, float]:
    """Min/max/mean/median/std of a numeric series, as a plain dictionary."""
    return {
        "min": series.min(),
        "max": series.max(),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
    }


def frame_with_object_geometry_columns(
    df: pd.DataFrame, geom_cols: Sequence[str]
) -> pd.DataFrame:
    """Plain ``DataFrame`` whose geometry columns hold objects, not geometries.

    Needed before pickling frames that carry several geometry columns, so that
    pandas does not try to treat them as a GeoSeries.
    """
    plain = pd.DataFrame(df.copy())
    for col in geom_cols:
        if col in plain.columns:
            plain[col] = pd.Series(list(plain[col].values), index=plain.index, dtype="object")
    return plain


# ---------------------------------------------------------------------------
# CRS and GeoDataFrames
# ---------------------------------------------------------------------------

WGS84_EPSG = 4326
WGS84_CRS = CRS.from_epsg(WGS84_EPSG)

#: Column names searched by :func:`metric_crs_for_row`, in priority order.
DEFAULT_LON_COLUMNS: tuple[str, ...] = ("lon", "GC_UCC_LON_2025")
DEFAULT_LAT_COLUMNS: tuple[str, ...] = ("lat", "GC_UCC_LAT_2025")


def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing ``(lon, lat)``."""
    zone = int(math.floor((float(lon) + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    return (32600 if float(lat) >= 0.0 else 32700) + zone


def metric_crs_for_lonlat(lon: float, lat: float) -> CRS:
    """Local metric (UTM) CRS for a longitude/latitude pair."""
    return CRS.from_epsg(utm_epsg_for_lonlat(lon, lat))


def metric_crs_str_for_lonlat(lon: float, lat: float) -> str:
    """Same as :func:`metric_crs_for_lonlat` but as an ``"EPSG:32633"`` string."""
    return f"EPSG:{utm_epsg_for_lonlat(lon, lat)}"


def metric_crs_for_gdf(gdf: "gpd.GeoDataFrame", default_epsg: int = 32633) -> CRS:
    """Local metric CRS for the centre of a GeoDataFrame's WGS84 extent."""
    if gdf is None or len(gdf) == 0:
        return CRS.from_epsg(default_epsg)
    xmin, ymin, xmax, ymax = gdf.total_bounds
    return metric_crs_for_lonlat((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)


def metric_crs_for_geometry(geom: BaseGeometry) -> CRS:
    """Local metric CRS for a geometry's representative point (WGS84 input)."""
    if geom is None or getattr(geom, "is_empty", True):
        raise ValueError("Cannot derive a metric CRS from an empty geometry.")
    point = geom.representative_point()
    return metric_crs_for_lonlat(float(point.x), float(point.y))


def metric_crs_for_row(
    row: pd.Series,
    lon_cols: Sequence[str] = DEFAULT_LON_COLUMNS,
    lat_cols: Sequence[str] = DEFAULT_LAT_COLUMNS,
) -> CRS:
    """Local metric CRS from the first populated lon/lat pair in a table row."""
    lon = lat = None
    for col in lon_cols:
        if col in row and pd.notna(row[col]):
            lon = float(row[col])
            break
    for col in lat_cols:
        if col in row and pd.notna(row[col]):
            lat = float(row[col])
            break
    if lon is None or lat is None:
        raise KeyError(f"Could not infer lon/lat from columns {list(lon_cols)}/{list(lat_cols)}.")
    return metric_crs_for_lonlat(lon, lat)


def ensure_wgs84(gdf: "gpd.GeoDataFrame", label: str = "GeoDataFrame") -> "gpd.GeoDataFrame":
    """Return ``gdf`` in EPSG:4326, assigning the CRS when it is missing.

    Columns and row count are preserved, including for an empty frame — callers
    that want an empty frame replaced should check that themselves.
    """
    if gdf is None:
        raise ValueError(f"{label}: expected a GeoDataFrame, got None.")
    if gdf.crs is None:
        return gdf.set_crs(WGS84_CRS, allow_override=True)
    try:
        if CRS.from_user_input(gdf.crs) == WGS84_CRS:
            return gdf
    except Exception as exc:  # noqa: BLE001 - malformed CRS is reported below
        raise ValueError(f"{label}: unreadable CRS {gdf.crs!r} ({exc})") from exc
    try:
        return gdf.to_crs(WGS84_CRS)
    except Exception as exc:  # noqa: BLE001 - surface the frame that failed
        raise ValueError(f"{label}: could not reproject to EPSG:{WGS84_EPSG} ({exc})") from exc


def empty_gdf(
    columns: Sequence[str] | None = None,
    crs: Any = "EPSG:4326",
    geometry: str = "geometry",
) -> "gpd.GeoDataFrame":
    """Empty GeoDataFrame with the given columns, guaranteeing a geometry column."""
    columns = list(columns) if columns else []
    if geometry not in columns:
        columns.append(geometry)
    return gpd.GeoDataFrame(pd.DataFrame(columns=columns), geometry=geometry, crs=crs)


def concat_gdfs(
    frames: Iterable[pd.DataFrame],
    columns: Sequence[str] | None = None,
    crs: Any = "EPSG:4326",
    geometry: str = "geometry",
) -> "gpd.GeoDataFrame":
    """Concatenate non-empty frames into one GeoDataFrame, or return an empty one."""
    valid = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
    if not valid:
        return empty_gdf(columns, crs=crs, geometry=geometry)
    out = pd.concat(valid, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry=geometry, crs=crs)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def polygon_parts(value: Any) -> list[Polygon]:
    """Flatten any geometry into a list of real ``Polygon`` parts.

    Accepts a ``Polygon``, a ``MultiPolygon``, a ``GeometryCollection``, or a
    list/tuple of any of those, and recurses through nesting. Non-polygonal
    parts are dropped, and so is anything degenerate: ``None``, empty
    geometries, and zero-area slivers. Degenerate geometry never enters the
    pipeline through this function, so nothing downstream has to re-check.
    """
    if isinstance(value, (list, tuple)):
        parts: list[Polygon] = []
        for item in value:
            parts.extend(polygon_parts(item))
        return parts
    if value is None or getattr(value, "is_empty", True):
        return []
    if isinstance(value, Polygon):
        return [value] if float(value.area) > 0.0 else []
    parts = []
    for part in getattr(value, "geoms", ()):
        parts.extend(polygon_parts(part))
    return parts


#: Backwards-compatible alias; :func:`polygon_parts` already accepts sequences.
extract_polygons = polygon_parts


def make_geometry_valid(geom: Any, *, drop_invalid: bool = False) -> Any:
    """Repair an invalid geometry, preferring ``shapely.make_valid``.

    ``make_valid`` is used first because it keeps every lobe of a
    self-intersecting ring, where the older ``buffer(0)`` trick silently keeps
    only one of them. ``buffer(0)`` remains the fallback on older shapely.

    With ``drop_invalid=True`` an empty or unrepairable geometry becomes
    ``None`` instead of being passed through unchanged.
    """
    if geom is None or getattr(geom, "is_empty", True):
        return None if drop_invalid else geom
    try:
        if geom.is_valid:
            return geom
    except Exception:  # noqa: BLE001 - fall through to the repair attempts
        pass
    if make_valid is not None:
        try:
            repaired = make_valid(geom)
            if repaired is not None and not repaired.is_empty:
                return repaired
        except Exception:  # noqa: BLE001 - fall back to the buffer trick
            pass
    try:
        repaired = geom.buffer(0)
    except Exception:  # noqa: BLE001 - a stubborn geometry is reported below
        return None if drop_invalid else geom
    if repaired is None or repaired.is_empty:
        return None if drop_invalid else geom
    return repaired


def union_polygons(value: Any) -> BaseGeometry | None:
    """Union every polygonal part of ``value``; ``None`` when there is nothing to union."""
    parts = polygon_parts(value)
    if not parts:
        return None
    merged = unary_union(parts)
    return None if getattr(merged, "is_empty", True) else merged


def keep_polygons_min_area(geom: Any, min_area: float) -> Polygon | MultiPolygon | None:
    """Drop polygonal parts below ``min_area``; ``None`` when nothing survives."""
    parts = [p for p in polygon_parts(geom) if p.area >= float(min_area) and p.is_valid]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def count_exterior_vertices(geom: Any) -> int:
    """Number of distinct exterior-ring vertices across all polygonal parts."""
    return int(sum(max(0, len(p.exterior.coords) - 1) for p in extract_polygons(geom)))


def count_polygon_holes(geom: Any) -> int:
    """Number of interior rings across all polygonal parts."""
    return int(sum(len(p.interiors) for p in extract_polygons(geom)))


def extract_vertices(geom: Any) -> list[tuple[float, float]]:
    """All coordinates of every ring (exterior and interior) of every part."""
    vertices: list[tuple[float, float]] = []
    for poly in extract_polygons(geom):
        vertices.extend(list(poly.exterior.coords))
        for ring in poly.interiors:
            vertices.extend(list(ring.coords))
    return vertices


def min_rotated_rect_dims(geom: Any) -> tuple[float, float]:
    """``(short_side, long_side)`` of the minimum rotated rectangle, in CRS units."""
    if geom is None or getattr(geom, "is_empty", True):
        return float("nan"), float("nan")
    try:
        coords = list(geom.minimum_rotated_rectangle.exterior.coords)
    except (AttributeError, ValueError):
        return float("nan"), float("nan")
    if len(coords) < 5:
        return float("nan"), float("nan")
    lengths = sorted(
        length
        for length in (
            math.hypot(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1])
            for i in range(4)
        )
        if length > 0.0
    )
    if not lengths:
        return float("nan"), float("nan")
    return float(lengths[0]), float(lengths[-1])


def aspect_ratio(geom: Any) -> float:
    """Long/short side ratio of the minimum rotated rectangle (``inf`` if degenerate)."""
    short, long = min_rotated_rect_dims(geom)
    if not np.isfinite(short) or not np.isfinite(long) or short <= 1e-9:
        return float("inf")
    return float(long / short)


def compactness(geom: Any) -> float:
    """Polsby-Popper compactness ``4*pi*A / P^2``; ``0.0`` for degenerate input."""
    if geom is None or getattr(geom, "is_empty", True):
        return 0.0
    area, perimeter = float(geom.area), float(geom.length)
    if area <= 0.0 or perimeter <= 0.0:
        return 0.0
    return (4.0 * math.pi * area) / (perimeter * perimeter)


def total_line_length(geom: Any) -> float:
    """Summed length of every ``LineString`` inside a geometry or collection."""
    if geom is None or getattr(geom, "is_empty", True):
        return 0.0
    if isinstance(geom, LineString):
        return float(geom.length)
    return float(
        sum(part.length for part in getattr(geom, "geoms", ()) if isinstance(part, LineString))
    )


def clean_ring_coords(coords, *, min_edge_len_m: float = 0.05, angle_tol_deg: float = 1.0):
    """Drop a ring's tiny edges and near-collinear vertices.

    Works on the closed coordinate list of one ring and returns a closed list.
    Removing a vertex changes its two neighbouring edges, so the pass restarts
    after every removal rather than walking the ring once. If cleaning would
    leave fewer than three points the original coordinates come back unchanged.
    """
    pts = [tuple(map(float, pt)) for pt in list(coords)[:-1]]
    if len(pts) < 3:
        return list(coords)

    changed = True
    while changed and len(pts) >= 3:
        changed = False
        n_pts = len(pts)
        for idx in range(n_pts):
            prev_pt = np.asarray(pts[(idx - 1) % n_pts], dtype=float)
            cur_pt = np.asarray(pts[idx], dtype=float)
            next_pt = np.asarray(pts[(idx + 1) % n_pts], dtype=float)

            prev_edge = cur_pt - prev_pt
            next_edge = next_pt - cur_pt

            if float(np.linalg.norm(prev_edge)) < float(min_edge_len_m):
                del pts[idx]
                changed = True
                break
            if float(np.linalg.norm(next_edge)) < float(min_edge_len_m):
                del pts[(idx + 1) % n_pts]
                changed = True
                break

            if angle_between_deg(prev_edge, next_edge) <= float(angle_tol_deg):
                del pts[idx]
                changed = True
                break

    if len(pts) < 3:
        return list(coords)
    return [*pts, pts[0]]


def iter_cleaned_edges(poly: Polygon, *, min_edge_len_m: float = 0.05, angle_tol_deg: float = 1.0):
    """Yield the exterior edges of one polygon after cleaning its ring."""
    coords = clean_ring_coords(poly.exterior.coords, min_edge_len_m=min_edge_len_m, angle_tol_deg=angle_tol_deg)
    for idx in range(len(coords) - 1):
        edge = LineString([coords[idx], coords[idx + 1]])
        if float(edge.length) >= float(min_edge_len_m):
            yield edge


def exterior_edges(geom: Any, *, min_edge_len_m: float = 0.05, angle_tol_deg: float = 1.0) -> list[LineString]:
    """The cleaned exterior edges of every polygonal part of a geometry."""
    segments: list[LineString] = []
    for poly in polygon_parts(geom):
        segments.extend(
            iter_cleaned_edges(poly, min_edge_len_m=min_edge_len_m, angle_tol_deg=angle_tol_deg)
        )
    return segments


def trim_segment(seg: LineString, trim: float) -> LineString | None:
    """A segment with ``trim`` metres cut off each end, or None if nothing is left."""
    length = float(seg.length)
    if length <= 0.0 or 2.0 * float(trim) >= length:
        return None
    return substring(seg, float(trim), length - float(trim), normalized=False)


def clean_polygonal(geom: Any, shrink: float, expand: float, min_area: float) -> Any:
    """Shrink then re-expand a footprint to erase notches thinner than ``shrink``.

    Any step that empties the geometry returns the input unchanged, so the
    cleanup can never lose a building.
    """
    if geom is None or getattr(geom, "is_empty", False):
        return geom
    out = geom
    if float(shrink) > 0.0:
        out = out.buffer(-float(shrink), join_style=2)
    if out is None or getattr(out, "is_empty", False):
        return geom
    if float(expand) > 0.0:
        out = out.buffer(float(expand), join_style=2)
    out = keep_polygons_min_area(out, float(min_area)) or geom
    return out


def sample_points_along_linestring(ls: LineString, interval_m: float) -> list:
    """Points spaced at most ``interval_m`` apart along a line, both ends included."""
    if ls is None or getattr(ls, "is_empty", False):
        return []
    length = float(ls.length)
    if length <= 0.0:
        return []
    n_steps = max(1, int(math.ceil(length / float(interval_m))))
    fractions = [idx / n_steps for idx in range(n_steps + 1)]
    return [ls.interpolate(frac, normalized=True) for frac in fractions]


def sample_points_on_polygon_exterior(poly: Any, interval_m: float) -> list:
    """The same sampling, walked around every part's exterior ring."""
    points = []
    for part in polygon_parts(poly):
        ring = LineString(list(part.exterior.coords))
        points.extend(sample_points_along_linestring(ring, interval_m))
    return points


def unit_vector(vec: Any) -> np.ndarray:
    """Normalised copy of a vector; the zero vector is returned unchanged."""
    array = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 0.0 else array


def angle_between_deg(vec_a: Any, vec_b: Any, *, undirected: bool = False) -> float:
    """Angle between two vectors in degrees.

    ``undirected=True`` folds the result into ``[0, 90]``, which is what you want
    when comparing the orientation of two lines rather than two directions.
    A zero-length input yields ``180.0`` (or ``90.0`` when ``undirected``).
    """
    a, b = np.asarray(vec_a, dtype=float), np.asarray(vec_b, dtype=float)
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 90.0 if undirected else 180.0
    dot = float(np.dot(a / norm_a, b / norm_b))
    if undirected:
        dot = abs(dot)
    return float(math.degrees(math.acos(float(np.clip(dot, -1.0, 1.0)))))


def haversine_m(lat1, lon1, lat2, lon2, radius_m: float = 6371000.0):
    """Great-circle distance in metres. Inputs in **radians**; broadcasts."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * radius_m * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def bbox_around_latlon(lat: float, lon: float, km_half: float = 1.0):
    """``(west, south, east, north)`` WGS84 bbox of half-width ``km_half`` km."""
    dlat = km_half / 111.0
    dlon = km_half / (111.0 * max(abs(math.cos(math.radians(lat))), 1e-6))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


# ---------------------------------------------------------------------------
# Plotting and binning helpers
# ---------------------------------------------------------------------------


def hex_to_rgba(hex_str: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """``"#RRGGBB"`` to a matplotlib ``(r, g, b, a)`` tuple in ``0..1``."""
    digits = hex_str.lstrip("#")
    return (
        int(digits[0:2], 16) / 255.0,
        int(digits[2:4], 16) / 255.0,
        int(digits[4:6], 16) / 255.0,
        alpha,
    )


def rgb255_to_rgba(r: float, g: float, b: float, alpha: float = 1.0):
    """``0..255`` channels to a matplotlib ``(r, g, b, a)`` tuple in ``0..1``."""
    return (r / 255.0, g / 255.0, b / 255.0, alpha)


def wrap_labels(values: Iterable[Any], width: int = 8) -> list[str]:
    """Wrap tick labels with ``<br>`` breaks, as Plotly expects."""
    return ["<br>".join(wrap(str(v), width=width)) for v in values]


def cut_counts(series: pd.Series, bins, labels, right: bool = False) -> pd.Series | None:
    """Counts per bin for a numeric series, reindexed onto ``labels``."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    categories = pd.cut(values, bins=bins, labels=labels, include_lowest=True, right=right)
    return categories.value_counts().reindex(labels, fill_value=0)


def integer_counts(
    series: pd.Series,
    min_val: int,
    max_val: int,
    under_label: str | None = None,
    over_label: str | None = None,
) -> pd.Series | None:
    """Counts for each integer in ``[min_val, max_val]`` plus optional tail bins."""
    values = pd.to_numeric(series, errors="coerce").dropna().round().astype(int)
    if values.empty:
        return None
    core = values.value_counts().reindex(range(min_val, max_val + 1), fill_value=0)
    parts = []
    if under_label is not None:
        parts.append(pd.Series({under_label: int((values < min_val).sum())}))
    parts.append(core)
    if over_label is not None:
        parts.append(pd.Series({over_label: int((values > max_val).sum())}))
    counts = pd.concat(parts)
    counts.index = [str(label) for label in counts.index]
    return counts
