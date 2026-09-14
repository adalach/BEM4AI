#!/usr/bin/env python3
from __future__ import annotations

import argparse
from functools import lru_cache
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

if __package__:
    from .server_common import (
        GeometryReviewRequestHandler,
        ReviewHTTPServer,
        PaginatedDataset,
        SelectionStore,
        natural_sort_key,
        start_server_thread,
    )
    from .svg_geometry import compute_view_box, geometry_to_svg_path, pack_geometries
else:
    from server_common import (
        GeometryReviewRequestHandler,
        ReviewHTTPServer,
        PaginatedDataset,
        SelectionStore,
        natural_sort_key,
        start_server_thread,
    )
    from svg_geometry import compute_view_box, geometry_to_svg_path, pack_geometries


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = Path(__file__).resolve().parent / "building_selector"
DEFAULT_TRANSFORMED_PATH = PROJECT_ROOT / "data" / "interim" / "buildings_transformed.pkl"
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "interim" / "selected_bad_building_samples.csv"
PAGE_COLUMNS = 12
PAGE_ROWS = 8
PAGE_SIZE = PAGE_COLUMNS * PAGE_ROWS
MAX_PAGE_SIZE = 240


def _prepare_pickle_import_paths() -> None:
    for path in (PROJECT_ROOT, PROJECT_ROOT.parent, PROJECT_ROOT / "notebooks"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_prepare_pickle_import_paths()

from notebooks.helpers.pickle_compat import read_pickle_compat


def flatten_polygon_geometries(value: object) -> tuple[Polygon, ...]:
    if value is None:
        return ()

    if isinstance(value, Polygon):
        return () if value.is_empty else (value,)

    if isinstance(value, MultiPolygon):
        return tuple(geom for geom in value.geoms if geom is not None and not geom.is_empty)

    if isinstance(value, GeometryCollection):
        parts: list[Polygon] = []
        for geom in value.geoms:
            parts.extend(flatten_polygon_geometries(geom))
        return tuple(parts)

    if isinstance(value, (list, tuple)):
        parts: list[Polygon] = []
        for item in value:
            parts.extend(flatten_polygon_geometries(item))
        return tuple(parts)

    return ()


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    zone_geometries: tuple[Polygon, ...]
    main_geometry: BaseGeometry | None
    neighbor_geometries: tuple[BaseGeometry, ...]
    shade_geometries: tuple[BaseGeometry, ...]
    window_geometries: tuple[BaseGeometry, ...]


class SampleDataset(PaginatedDataset):
    max_page_size = MAX_PAGE_SIZE

    def __init__(
        self,
        *,
        transformed_path: Path,
        page_size: int = PAGE_SIZE,
    ) -> None:
        self.transformed_path = transformed_path
        self.page_size = page_size
        self.records = self._load_records()
        self.record_lookup = {record.sample_id: record for record in self.records}

    def _load_records(self) -> list[SampleRecord]:
        payload = self._prepare_payload(self.transformed_path)
        buildings = payload["buildings"]
        if buildings.empty:
            return []

        buildings = buildings.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
        ordered_rows = sorted(buildings.to_dict("records"), key=lambda row: natural_sort_key(row["sample_id"]))

        records: list[SampleRecord] = []
        for row in ordered_rows:
            sample_id = row["sample_id"]
            zone_geometries = self._extract_zone_geometries(row)
            main_geometry = self._extract_main_geometry(row)
            if not zone_geometries and main_geometry is not None and not main_geometry.is_empty:
                zone_geometries = flatten_polygon_geometries(main_geometry)

            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    zone_geometries=zone_geometries,
                    main_geometry=main_geometry,
                    neighbor_geometries=payload["neighbors_lookup"].get(sample_id, ()),
                    shade_geometries=payload["shades_lookup"].get(sample_id, ()),
                    window_geometries=payload["windows_lookup"].get(sample_id, ()),
                )
            )

        return records

    @staticmethod
    def _prepare_payload(path: Path) -> dict[str, object]:
        payload = read_pickle_compat(path)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} does not contain the expected stage-4 payload dictionary.")

        buildings = payload.get("buildings_m")
        if not isinstance(buildings, pd.DataFrame):
            raise KeyError(f"{path} is missing the buildings_m table.")

        out_buildings = buildings.dropna(subset=["sample_id"]).copy()
        out_buildings["sample_id"] = out_buildings["sample_id"].astype(str)

        return {
            "buildings": out_buildings.reset_index(drop=True),
            "neighbors_lookup": SampleDataset._build_geometry_lookup(
                payload.get("neighbors_m"),
                geometry_candidates=("geom_local", "geometry"),
            ),
            "windows_lookup": SampleDataset._build_geometry_lookup(
                payload.get("final_windows_gdf"),
                geometry_candidates=("geometry",),
            ),
            "shades_lookup": SampleDataset._build_geometry_lookup(
                payload.get("shades_gdf"),
                geometry_candidates=("geometry",),
            ),
        }

    @staticmethod
    def _build_geometry_lookup(frame: object, *, geometry_candidates: tuple[str, ...]) -> dict[str, tuple[BaseGeometry, ...]]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return {}
        if "sample_id" not in frame.columns:
            raise KeyError("Expected 'sample_id' in transformed payload geometry table.")

        geometry_col = next((col for col in geometry_candidates if col in frame.columns), None)
        if geometry_col is None:
            raise KeyError(f"Could not find any geometry column from {geometry_candidates}.")

        out = frame.dropna(subset=["sample_id", geometry_col]).copy()
        out["sample_id"] = out["sample_id"].astype(str)

        lookup: dict[str, tuple[BaseGeometry, ...]] = {}
        for sample_id, group in out.groupby("sample_id", sort=False):
            geometries = tuple(
                geometry
                for geometry in group[geometry_col]
                if isinstance(geometry, BaseGeometry) and not geometry.is_empty
            )
            if geometries:
                lookup[str(sample_id)] = geometries
        return lookup

    @staticmethod
    def _extract_zone_geometries(row: dict[str, object]) -> tuple[Polygon, ...]:
        for key in ("zones", "zones_perim", "convex_parts_geom", "geom_local", "geometry"):
            polygons = flatten_polygon_geometries(row.get(key))
            if polygons:
                return polygons
        return ()

    @staticmethod
    def _extract_main_geometry(row: dict[str, object]) -> BaseGeometry | None:
        for key in ("geom_local", "convex_parts_geom", "geometry"):
            geometry = row.get(key)
            if isinstance(geometry, BaseGeometry) and not geometry.is_empty:
                return geometry
        return None

    @lru_cache(maxsize=2048)
    def _render_sample_geometry(self, sample_id: str) -> dict[str, object]:
        record = self.record_lookup[sample_id]
        neighbor_geometry = pack_geometries(record.neighbor_geometries)
        shade_geometry = pack_geometries(record.shade_geometries)
        window_geometry = pack_geometries(record.window_geometries)
        zone_geometries = tuple(geometry for geometry in record.zone_geometries if geometry is not None and not geometry.is_empty)

        zone_paths: list[str] = []
        for geometry in zone_geometries:
            path = geometry_to_svg_path(geometry)
            if path:
                zone_paths.append(path)
        neighbor_paths: list[str] = []
        for geometry in record.neighbor_geometries:
            path = geometry_to_svg_path(geometry)
            if path:
                neighbor_paths.append(path)
        return {
            "zone_paths": zone_paths,
            "main_path": geometry_to_svg_path(record.main_geometry),
            "neighbor_paths": neighbor_paths,
                "shade_path": geometry_to_svg_path(shade_geometry),
            "window_path": geometry_to_svg_path(window_geometry),
            "view_box": compute_view_box(
                neighbor_geometry,
                shade_geometry,
                window_geometry,
                pack_geometries(zone_geometries),
                record.main_geometry,
            ),
        }

class AppState:
    def __init__(
        self,
        *,
        transformed_path: Path,
        csv_path: Path,
    ) -> None:
        self.dataset = SampleDataset(
            transformed_path=transformed_path,
        )
        self.store = SelectionStore(csv_path=csv_path)
        self.csv_path = csv_path


class AppHandler(GeometryReviewRequestHandler):
    static_root = STATIC_ROOT
    server_version = "BEM4AIBuildingReview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the transformed-building selection web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the local server to.")
    parser.add_argument("--port", default=8002, type=int, help="Port to bind the local server to.")
    parser.add_argument(
        "--transformed",
        default=str(DEFAULT_TRANSFORMED_PATH),
        help="Path to the unified Stage 4 transformed payload pickle.",
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV_PATH),
        help="Path to the CSV file used to persist rejected sample IDs.",
    )
    return parser.parse_args()


def create_server(
    *,
    transformed_path: Path = DEFAULT_TRANSFORMED_PATH,
    csv_path: Path = DEFAULT_CSV_PATH,
    host: str = "127.0.0.1",
    port: int = 8002,
) -> ReviewHTTPServer:
    """Create a configured selector server without starting its event loop."""
    transformed_path = Path(transformed_path).expanduser().resolve()
    csv_path = Path(csv_path).expanduser().resolve()

    if not transformed_path.is_file():
        raise FileNotFoundError(
            "Run Stage 4 before starting the review app. Missing: "
            + str(transformed_path)
        )

    app_state = AppState(
        transformed_path=transformed_path,
        csv_path=csv_path,
    )
    return ReviewHTTPServer((host, int(port)), AppHandler, app_state=app_state)


def start_server(
    *,
    transformed_path: Path = DEFAULT_TRANSFORMED_PATH,
    csv_path: Path = DEFAULT_CSV_PATH,
    host: str = "127.0.0.1",
    port: int = 0,
) -> tuple[ReviewHTTPServer, threading.Thread, str]:
    """Start the selector in a daemon thread and return its local URL."""
    server = create_server(
        transformed_path=transformed_path,
        csv_path=csv_path,
        host=host,
        port=port,
    )
    return start_server_thread(server, "bem4ai-building-selector")


def main() -> None:
    args = parse_args()
    transformed_path = Path(args.transformed).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve()

    server = create_server(
        transformed_path=transformed_path,
        csv_path=csv_path,
        host=args.host,
        port=args.port,
    )

    bound_host, bound_port = server.server_address[:2]
    print(f"Serving building selector at http://{bound_host}:{bound_port}")
    print(f"Transformed payload: {transformed_path}")
    print(f"Selections CSV: {csv_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
