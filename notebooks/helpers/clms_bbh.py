"""Copernicus Urban Atlas Building Block Height 2021 (CLMS, 10 m) for the BEM4AI city windows.

The product is one GeoTIFF per European city core area (EPSG:3035, 10 m, uint16
metres, nodata 65535 for non-building pixels), published on the Copernicus Data
Space Ecosystem (CDSE). This module finds the files under the city windows in the
public CDSE catalogue, records them in a manifest, and downloads them with a free
CDSE account. Stage 1 cuts per-city chips from them with
`helpers.ghsl_chips.build_city_chips`, passing `sources_for_cities(...)`.

Licence: Copernicus data policy (full, open and free access). Chips may be
redistributed with the attribution "European Union's Copernicus Land Monitoring
Service information" and the dataset DOI 10.2909/c5215520-98e9-4035-be56-d28156cd4dcd,
stating that they are clipped to the city windows; `CHIP_TAGS` writes this into
every chip.

Import from a notebook as::

    from helpers.clms_bbh import build_manifest, sources_for_cities, coverage_for_cities

"""

from __future__ import annotations

import json
import os
import urllib.parse
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import requests
from shapely import wkt
from shapely.geometry import box, shape
from shapely.ops import unary_union

from .ghsl_chips import window_polygon, window_tag

__all__ = [
    "PRODUCT_PREFIX",
    "DOI",
    "CHIP_TAGS",
    "query_products",
    "build_manifest",
    "sources_for_cities",
    "coverage_for_cities",
    "missing_files",
    "download_files",
]

PRODUCT_PREFIX = "CLMS_UA_BBH_S2021"
DOI = "10.2909/c5215520-98e9-4035-be56-d28156cd4dcd"
CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products({id})/$value"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
USER_AGENT = "BEM4AI-dataset-pipeline (CDSE download)"
EQUAL_AREA = "EPSG:3035"

#: Written into every BBH chip, so the attribution the Copernicus data policy asks for
#: travels with the file.
CHIP_TAGS = {
    "SOURCE": "European Union's Copernicus Land Monitoring Service information",
    "DATASET": "Urban Atlas Building Block Height 2021 (raster 10 m), Europe, 3-yearly",
    "DOI": DOI,
    "MODIFICATION": ("Cut to the 2x2 km window of one BEM4AI city, widened to whole pixels of the "
                     "product's own 10 m grid (EPSG:3035), without resampling; a window spanning several "
                     "product files is mosaicked from them. In cropped chips (BEM4AI_CHIP_GRID "
                     "source-aligned-cropped-*) the pixels outside the window are set to nodata (65535)."),
    "LICENSE": "Copernicus data policy, https://land.copernicus.eu/en/data-policy",
}


def query_products(prefix: str = PRODUCT_PREFIX) -> gpd.GeoDataFrame:
    """Every product whose name starts with `prefix`, latest version per city area.

    Uses the public CDSE catalogue (no login). Columns: ``name``, ``id``, ``bytes``,
    ``area`` (e.g. ``AT001L3_WIEN``), ``version`` and the footprint (EPSG:4326).
    """
    rows, skip = [], 0
    query = f"startswith(Name,'{prefix}')"
    while True:
        r = requests.get(CATALOGUE, params={"$filter": query, "$top": 1000, "$skip": skip},
                         headers={"Accept": "application/json", "User-Agent": USER_AGENT}, timeout=120)
        r.raise_for_status()
        page = r.json()["value"]
        rows += [{"name": p["Name"], "id": p["Id"], "bytes": p.get("ContentLength"),
                  "geometry": shape(p["GeoFootprint"])} for p in page if p.get("GeoFootprint")]
        if len(page) < 1000:
            break
        skip += 1000
    products = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
    products["area"] = products["name"].str.extract(r"_R10m_(.+)_03035_")[0]
    products["version"] = products["name"].str.extract(r"_(V\d+_R\d+)_")[0]
    return (products.sort_values(["area", "version", "name"]).groupby("area").tail(1)
            .reset_index(drop=True))


def _windows(cities: pd.DataFrame, km_half: float) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        cities[["city_key"]].reset_index(drop=True),
        geometry=[window_polygon(lat, lon, km_half) for lat, lon in zip(cities["lat"], cities["lon"])],
        crs=4326,
    )


def _raster_extent(path) -> object:
    """The extent of a raster file as a polygon in EPSG:4326."""
    with rasterio.open(path) as ds:
        return gpd.GeoSeries([box(*ds.bounds)], crs=ds.crs).segmentize(100).to_crs(4326).iloc[0]


def build_manifest(cities: pd.DataFrame, path, *, km_half: float = 1.0, refresh: bool = False,
                   products: gpd.GeoDataFrame | None = None, files_dir=None) -> dict:
    """Files under the city windows, written to and read from the JSON manifest at `path`.

    The manifest is reused while it holds the window of every city in `cities`, as
    `ghsl_chips.window_tag` writes it; otherwise the catalogue is queried again (or
    `products` is used). The catalogue footprints can reach beyond the data, so the
    footprint of every file found in `files_dir` is cut to the extent of its raster; a
    manifest built without the files is rebuilt once they are there. A city is listed
    under a file only if its window overlaps the footprint with a positive area.
    Returns ``{"km_half", "windows", "footprints", "files": [{name, id, bytes, cities,
    footprint}]}``, with `windows` mapping `city_key` to its window, `footprints` saying
    whether they were cut to the rasters, and the footprints as WKT in EPSG:4326.
    """
    path = Path(path)
    windows = {key: window_tag(lat, lon, km_half)
               for key, lat, lon in zip(cities["city_key"], cities["lat"], cities["lon"])}
    have_files = files_dir is not None and any(Path(files_dir).glob(f"{PRODUCT_PREFIX}*.tif"))
    if path.exists() and not refresh:
        manifest = json.loads(path.read_text())
        stored = manifest.get("windows", {})
        if (all(stored.get(key) == window for key, window in windows.items())
                and (manifest.get("footprints") == "raster extent" or not have_files)):
            return manifest
    products = query_products() if products is None else products.copy()
    if have_files:
        files_dir = Path(files_dir)
        products["geometry"] = [
            geometry.intersection(_raster_extent(files_dir / f"{name}.tif"))
            if (files_dir / f"{name}.tif").is_file() else geometry
            for name, geometry in zip(products["name"], products.geometry)
        ]
        products = products[~products.geometry.is_empty]
    city_windows = _windows(cities, km_half)
    hits = gpd.sjoin(products, city_windows, predicate="intersects")
    overlap = [gpd.GeoSeries([footprint.intersection(city_windows.geometry.loc[i])], crs=4326)
               .to_crs(EQUAL_AREA).area.iloc[0] for footprint, i in zip(hits.geometry, hits["index_right"])]
    hits = hits[[area > 0 for area in overlap]]
    files = [
        {"name": name, "id": group["id"].iloc[0], "bytes": int(group["bytes"].iloc[0] or 0),
         "cities": sorted(group["city_key"]), "footprint": group.geometry.iloc[0].wkt}
        for name, group in hits.groupby("name")
    ]
    manifest = {"km_half": km_half, "windows": dict(sorted(windows.items())),
                "footprints": "raster extent" if have_files else "catalogue", "files": files}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1))
    return manifest


def sources_for_cities(manifest: dict, files_dir) -> dict[str, list[Path]]:
    """``city_key`` -> the BBH GeoTIFFs under its window, for `build_city_chips`."""
    files_dir = Path(files_dir)
    sources: dict[str, list[Path]] = {}
    for f in manifest["files"]:
        for key in f["cities"]:
            sources.setdefault(key, []).append(files_dir / f"{f['name']}.tif")
    return sources


def coverage_for_cities(manifest: dict, cities: pd.DataFrame, km_half: float = 1.0) -> pd.Series:
    """Share of each city window inside the product footprints (0 for cities without files)."""
    footprints: dict[str, list] = {}
    for f in manifest["files"]:
        for key in f["cities"]:
            footprints.setdefault(key, []).append(wkt.loads(f["footprint"]))
    windows = _windows(cities, km_half).to_crs(EQUAL_AREA)
    shares = {}
    for key, window in zip(windows["city_key"], windows.geometry):
        if key not in footprints:
            shares[key] = 0.0
            continue
        covered = gpd.GeoSeries([unary_union(footprints[key])], crs=4326).to_crs(EQUAL_AREA).iloc[0]
        shares[key] = min(1.0, window.intersection(covered).area / window.area)
    return pd.Series(shares, name="bbh_coverage")


def missing_files(manifest: dict, files_dir, cities=None) -> list[dict]:
    """Manifest entries whose GeoTIFF is not in `files_dir` yet.

    With `cities` (city keys), only the files under those cities' windows count.
    """
    files_dir = Path(files_dir)
    wanted = None if cities is None else set(cities)
    return [f for f in manifest["files"]
            if not (files_dir / f"{f['name']}.tif").is_file()
            and (wanted is None or wanted & set(f["cities"]))]


def _token(credentials: dict, totp: str | None) -> str:
    form = {"grant_type": "password", "client_id": "cdse-public",
            "username": credentials["username"], "password": credentials["password"]}
    if totp:
        form["totp"] = totp
    r = requests.post(TOKEN_URL, data=form, timeout=60, headers={"User-Agent": USER_AGENT})
    if r.status_code != 200:
        raise RuntimeError(f"CDSE login failed: HTTP {r.status_code}")
    return r.json()["access_token"]


def download_files(manifest: dict, files_dir, credentials_path, *, totp: str | None = None,
                   progress=print) -> int:
    """Download the missing GeoTIFFs of the manifest from CDSE; returns how many were added.

    `credentials_path` is a JSON file with ``username`` and ``password`` of a CDSE
    account, kept outside the repository; it is never printed. Each product is
    fetched as a zip, its GeoTIFF unpacked into `files_dir` and the zip removed.
    The bearer token is sent on redirects only to CDSE hosts.
    """
    files_dir = Path(files_dir)
    files_dir.mkdir(parents=True, exist_ok=True)
    todo = missing_files(manifest, files_dir)
    if not todo:
        return 0
    credentials = json.loads(Path(credentials_path).read_text())
    http = requests.Session()
    http.headers["User-Agent"] = USER_AGENT
    token, added = _token(credentials, totp), 0
    for n, f in enumerate(todo, start=1):
        url = DOWNLOAD.format(id=f["id"])
        for attempt in range(2):
            target = url
            for _ in range(5):
                r = http.get(target, headers={"Authorization": f"Bearer {token}"},
                             allow_redirects=False, stream=True, timeout=600)
                if r.status_code not in (301, 302, 303, 307, 308):
                    break
                target = urllib.parse.urljoin(target, r.headers["Location"])
                if not urllib.parse.urlparse(target).hostname.endswith(".dataspace.copernicus.eu"):
                    raise RuntimeError(f"Refusing to send the CDSE token to {target}")
            if r.status_code == 401 and attempt == 0:
                token = _token(credentials, totp)     # expired: log in again and re-send
                continue
            break
        if r.status_code != 200:
            progress(f"  [{n}/{len(todo)}] {f['name']}: HTTP {r.status_code}")
            continue
        archive = files_dir / f"{f['name']}.zip.part"
        with open(archive, "wb") as out:
            for chunk in r.iter_content(1 << 20):
                out.write(chunk)
        with zipfile.ZipFile(archive) as zf:
            tif = next(m for m in zf.namelist() if m.lower().endswith((".tif", ".tiff")))
            partial = files_dir / f"{f['name']}.tif.part"
            partial.write_bytes(zf.read(tif))
            os.replace(partial, files_dir / f"{f['name']}.tif")
        archive.unlink()
        added += 1
        if n % 25 == 0 or n == len(todo):
            progress(f"  [{n}/{len(todo)}] downloaded")
    return added
