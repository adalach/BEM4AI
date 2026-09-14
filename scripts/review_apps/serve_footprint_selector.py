#!/usr/bin/env python3
from __future__ import annotations

import argparse
from functools import lru_cache
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

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
STATIC_ROOT = Path(__file__).resolve().parent / "footprint_selector"
DEFAULT_BUILDINGS_PATH = PROJECT_ROOT / "data" / "interim" / "buildings_gdf.pkl"
DEFAULT_NEIGHBORS_PATH = PROJECT_ROOT / "data" / "interim" / "neighbors_gdf.pkl"
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "interim" / "selected_bad_footprint_samples.csv"
PAGE_COLUMNS = 12
PAGE_ROWS = 8
PAGE_SIZE = PAGE_COLUMNS * PAGE_ROWS
MAX_PAGE_SIZE = 240


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    main_geometry: BaseGeometry | None
    neighbor_geometries: tuple[BaseGeometry, ...]


class SampleDataset(PaginatedDataset):
    max_page_size = MAX_PAGE_SIZE

    def __init__(
        self,
        *,
        buildings_path: Path,
        neighbors_path: Path,
        page_size: int = PAGE_SIZE,
    ) -> None:
        self.buildings_path = buildings_path
        self.neighbors_path = neighbors_path
        self.page_size = page_size
        self.records = self._load_records()
        self.record_lookup = {record.sample_id: record for record in self.records}

    def _load_records(self) -> list[SampleRecord]:
        buildings = pd.read_pickle(self.buildings_path)
        if not isinstance(buildings, pd.DataFrame):
            raise ValueError(f"{self.buildings_path} does not contain a tabular buildings payload.")
        required = {"sample_id", "geometry"}
        missing = required.difference(buildings.columns)
        if missing:
            raise KeyError(f"{self.buildings_path} is missing columns: {sorted(missing)}")

        buildings = buildings.dropna(subset=["sample_id", "geometry"]).copy()
        buildings["sample_id"] = buildings["sample_id"].astype(str)
        buildings = buildings.drop_duplicates(subset=["sample_id"], keep="first")
        buildings = buildings.sort_values(
            "sample_id", key=lambda column: column.map(natural_sort_key), kind="mergesort"
        ).reset_index(drop=True)
        neighbors_lookup = self._load_neighbors_lookup()

        return [
            SampleRecord(
                sample_id=row["sample_id"],
                main_geometry=row["geometry"],
                neighbor_geometries=tuple(neighbors_lookup.get(row["sample_id"], ())),
            )
            for _, row in buildings.iterrows()
        ]

    def _load_neighbors_lookup(self) -> dict[str, list[BaseGeometry]]:
        if not self.neighbors_path.exists():
            return {}
        neighbors = pd.read_pickle(self.neighbors_path)
        if not isinstance(neighbors, pd.DataFrame) or neighbors.empty:
            return {}
        required = {"sample_id", "geometry"}
        missing = required.difference(neighbors.columns)
        if missing:
            raise KeyError(f"{self.neighbors_path} is missing columns: {sorted(missing)}")

        neighbors = neighbors.dropna(subset=["sample_id", "geometry"]).copy()
        neighbors["sample_id"] = neighbors["sample_id"].astype(str)
        lookup: dict[str, list[BaseGeometry]] = {}
        for sample_id, group in neighbors.groupby("sample_id", sort=False):
            lookup[str(sample_id)] = [
                geometry
                for geometry in group["geometry"]
                if isinstance(geometry, BaseGeometry) and not geometry.is_empty
            ]
        return lookup

    @staticmethod
    def _project_to_local(
        main_geometry: BaseGeometry | None,
        neighbor_geometry: BaseGeometry | None,
    ) -> tuple[BaseGeometry | None, BaseGeometry | None]:
        if main_geometry is None or main_geometry.is_empty:
            return None, neighbor_geometry

        anchor = main_geometry.representative_point()
        local_crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={anchor.y:.10f} +lon_0={anchor.x:.10f} +datum=WGS84 +units=m +no_defs"
        )
        transformer = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
        project = transformer.transform

        projected_main = transform(project, main_geometry)
        projected_neighbors = transform(project, neighbor_geometry) if neighbor_geometry is not None else None
        return projected_main, projected_neighbors

    @lru_cache(maxsize=2048)
    def _render_sample_geometry(self, sample_id: str) -> dict[str, str]:
        record = self.record_lookup[sample_id]
        neighbor_geometry = pack_geometries(record.neighbor_geometries)
        main_local, neighbors_local = self._project_to_local(record.main_geometry, neighbor_geometry)
        return {
            "main_path": geometry_to_svg_path(main_local),
            "neighbor_path": geometry_to_svg_path(neighbors_local),
            "view_box": compute_view_box(neighbors_local, main_local),
        }

class AppState:
    def __init__(
        self,
        *,
        buildings_path: Path,
        neighbors_path: Path,
        csv_path: Path,
    ) -> None:
        self.dataset = SampleDataset(
            buildings_path=buildings_path,
            neighbors_path=neighbors_path,
        )
        self.store = SelectionStore(csv_path=csv_path)
        self.csv_path = csv_path


class AppHandler(GeometryReviewRequestHandler):
    static_root = STATIC_ROOT
    server_version = "BEM4AIFootprintReview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the footprint selection web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the local server to.")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind the local server to.")
    parser.add_argument(
        "--buildings",
        default=str(DEFAULT_BUILDINGS_PATH),
        help="Path to the unified Stage 3 buildings GeoDataFrame pickle.",
    )
    parser.add_argument(
        "--neighbors",
        default=str(DEFAULT_NEIGHBORS_PATH),
        help="Path to the unified Stage 3 neighbors GeoDataFrame pickle.",
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV_PATH),
        help="Path to the CSV file used to persist rejected sample IDs.",
    )
    return parser.parse_args()


def create_server(
    *,
    buildings_path: Path = DEFAULT_BUILDINGS_PATH,
    neighbors_path: Path = DEFAULT_NEIGHBORS_PATH,
    csv_path: Path = DEFAULT_CSV_PATH,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ReviewHTTPServer:
    app_state = AppState(
        buildings_path=Path(buildings_path).expanduser().resolve(),
        neighbors_path=Path(neighbors_path).expanduser().resolve(),
        csv_path=Path(csv_path).expanduser().resolve(),
    )
    return ReviewHTTPServer((host, int(port)), AppHandler, app_state=app_state)


def start_server(
    *,
    buildings_path: Path = DEFAULT_BUILDINGS_PATH,
    neighbors_path: Path = DEFAULT_NEIGHBORS_PATH,
    csv_path: Path = DEFAULT_CSV_PATH,
    host: str = "127.0.0.1",
    port: int = 0,
) -> tuple[ReviewHTTPServer, threading.Thread, str]:
    server = create_server(
        buildings_path=buildings_path,
        neighbors_path=neighbors_path,
        csv_path=csv_path,
        host=host,
        port=port,
    )
    return start_server_thread(server, "bem4ai-footprint-selector")


def main() -> None:
    args = parse_args()
    server = create_server(
        buildings_path=Path(args.buildings),
        neighbors_path=Path(args.neighbors),
        csv_path=Path(args.csv),
        host=args.host,
        port=args.port,
    )
    bound_host, bound_port = server.server_address[:2]

    print(f"Serving footprint selector at http://{bound_host}:{bound_port}")
    print(f"Buildings: {server.app_state.dataset.buildings_path}")
    print(f"Neighbors: {server.app_state.dataset.neighbors_path}")
    print(f"Selections CSV: {server.app_state.csv_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
