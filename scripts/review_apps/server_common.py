"""Shared HTTP and selection persistence for the local review apps."""

from __future__ import annotations

import csv
import json
import mimetypes
import os
import re
import threading
from collections.abc import Iterable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


SAMPLE_ID_COLUMN = "sample_id"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def repository_relative_path(path: str | Path) -> str:
    """Return a portable display path relative to the repository root."""
    resolved = Path(path).expanduser().resolve()
    try:
        return Path(os.path.relpath(resolved, PROJECT_ROOT)).as_posix()
    except ValueError:
        return resolved.name


def natural_sort_key(value: object) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def read_sample_ids(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        return []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if SAMPLE_ID_COLUMN not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} must contain a '{SAMPLE_ID_COLUMN}' column.")
        return normalize_sample_ids(row.get(SAMPLE_ID_COLUMN) for row in reader)


def normalize_sample_ids(values: Iterable[object], valid_ids: set[str] | None = None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        sample_id = value.strip()
        if not sample_id or sample_id in seen:
            continue
        if valid_ids is not None and sample_id not in valid_ids:
            continue
        normalized.append(sample_id)
        seen.add(sample_id)
    return normalized


class PaginatedDataset:
    """Pagination shared by review datasets backed by an ordered record list."""

    page_size: int
    max_page_size: int
    records: Sequence[object]

    @property
    def total_count(self) -> int:
        return len(self.records)

    @property
    def page_count(self) -> int:
        return self.page_count_for(self.page_size)

    def _normalize_page_size(self, page_size: int | None) -> int:
        requested = self.page_size if page_size is None else int(page_size)
        return max(1, min(requested, self.max_page_size))

    def page_count_for(self, page_size: int | None = None) -> int:
        effective_page_size = self._normalize_page_size(page_size)
        return max(1, (len(self.records) + effective_page_size - 1) // effective_page_size)

    def normalize_page(self, page_index: int, page_size: int | None = None) -> int:
        return max(0, min(page_index, self.page_count_for(page_size) - 1))

    def _render_sample_geometry(self, sample_id: str) -> dict[str, object]:
        raise NotImplementedError

    def build_page_payload(
        self,
        page_index: int,
        saved_ids: set[str],
        csv_path: Path,
        page_size: int | None = None,
    ) -> dict[str, object]:
        """Build one page for a two-dimensional geometry review app."""
        effective_page_size = self._normalize_page_size(page_size)
        page_count = self.page_count_for(effective_page_size)
        page_index = self.normalize_page(page_index, effective_page_size)
        start = page_index * effective_page_size
        end = min(start + effective_page_size, len(self.records))

        return {
            "page": page_index,
            "page_count": page_count,
            "page_size": effective_page_size,
            "total_count": self.total_count,
            "start_index": start,
            "end_index": end,
            "saved_count": len(saved_ids),
            "csv_path": repository_relative_path(csv_path),
            "samples": [
                {
                    "sample_id": record.sample_id,
                    "saved": record.sample_id in saved_ids,
                    **self._render_sample_geometry(record.sample_id),
                }
                for record in self.records[start:end]
            ],
        }


class SelectionStore:
    """Thread-safe persistence for rejected sample identifiers."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self._lock = threading.Lock()
        self._saved_ids = read_sample_ids(csv_path)
        self._saved_set = set(self._saved_ids)

    def _write(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.csv_path.with_name(f".{self.csv_path.name}.tmp")
        try:
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[SAMPLE_ID_COLUMN])
                writer.writeheader()
                writer.writerows({SAMPLE_ID_COLUMN: sample_id} for sample_id in self._saved_ids)
            os.replace(temporary, self.csv_path)
        finally:
            temporary.unlink(missing_ok=True)

    def snapshot(self) -> set[str]:
        with self._lock:
            return set(self._saved_set)

    def count(self) -> int:
        with self._lock:
            return len(self._saved_ids)

    def add_many(self, sample_ids: list[str]) -> list[str]:
        added: list[str] = []
        with self._lock:
            for sample_id in sample_ids:
                if sample_id not in self._saved_set:
                    self._saved_ids.append(sample_id)
                    self._saved_set.add(sample_id)
                    added.append(sample_id)
            if added:
                self._write()
        return added

    def remove(self, sample_id: str) -> bool:
        with self._lock:
            if sample_id not in self._saved_set:
                return False
            self._saved_set.remove(sample_id)
            self._saved_ids = [saved_id for saved_id in self._saved_ids if saved_id != sample_id]
            self._write()
            return True


class ReviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, handler_class, app_state) -> None:
        super().__init__(server_address, handler_class)
        self.app_state = app_state


class ReviewRequestHandler(BaseHTTPRequestHandler):
    """Base request handler for static files, pagination, and selections."""

    static_root: Path
    server_version = "BEM4AIReview"

    def do_POST(self) -> None:
        """Persist a replacement set of rejected sample IDs."""
        parsed = urlparse(self.path)
        data = self._read_json_body()
        if data is None:
            return

        state = self.server.app_state
        if parsed.path == "/api/save":
            values = data.get("sample_ids")
            if not isinstance(values, list):
                self._send_json({"error": "Expected 'sample_ids' to be a list."}, status=400)
                return
            valid_ids = set(state.dataset.record_lookup)
            added = state.store.add_many(normalize_sample_ids(values, valid_ids))
            self._send_json({"added": added, "saved_count": state.store.count()})
            return

        if parsed.path == "/api/remove":
            value = data.get("sample_id")
            if not isinstance(value, str) or not value.strip():
                self._send_json({"error": "Expected non-empty 'sample_id'."}, status=400)
                return
            removed = state.store.remove(value.strip())
            self._send_json({"removed": removed, "saved_count": state.store.count()})
            return

        self._send_json({"error": "Not found."}, status=404)

    def log_message(self, format: str, *args) -> None:
        return

    def _parse_page_query(self, query: str) -> tuple[int, int | None] | None:
        values = parse_qs(query)
        try:
            page_index = int(values.get("page", ["0"])[0])
            page_size = int(values["page_size"][0]) if values.get("page_size") else None
        except ValueError:
            self._send_json({"error": "Page and page_size must be integers."}, status=400)
            return None
        return page_index, page_size

    def _read_json_body(self) -> dict[str, object] | None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "Invalid Content-Length header."}, status=400)
            return None

        try:
            value = json.loads(self.rfile.read(content_length).decode("utf-8") if content_length else "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json({"error": "Request body must be valid JSON."}, status=400)
            return None
        if not isinstance(value, dict):
            self._send_json({"error": "Request body must be a JSON object."}, status=400)
            return None
        return value

    def _serve_static_file(self, relative_path: str) -> None:
        target = (self.static_root / relative_path).resolve()
        if target != self.static_root and self.static_root not in target.parents:
            self._send_json({"error": "Invalid static path."}, status=400)
            return
        self._serve_file(target)

    def _serve_file(self, path: Path) -> None:
        if not path.is_file():
            self._send_json({"error": "Static file not found."}, status=404)
            return

        payload = path.read_bytes()
        mime_type, _ = mimetypes.guess_type(path.name)
        self.send_response(200)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)


class GeometryReviewRequestHandler(ReviewRequestHandler):
    """GET routes shared by the footprint and transformed-building apps."""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path in {"/", "/index.html"}:
            self._serve_static_file("index.html")
            return

        if parsed.path in {"/app.js", "/styles.css"}:
            self._serve_static_file(parsed.path.lstrip("/"))
            return

        if parsed.path == "/api/page":
            pagination = self._parse_page_query(parsed.query)
            if pagination is None:
                return
            page_index, page_size = pagination
            state = self.server.app_state
            self._send_json(
                state.dataset.build_page_payload(
                    page_index=page_index,
                    saved_ids=state.store.snapshot(),
                    csv_path=state.csv_path,
                    page_size=page_size,
                )
            )
            return

        self._send_json({"error": "Not found."}, status=404)


def start_server_thread(server: ReviewHTTPServer, name: str) -> tuple[ReviewHTTPServer, threading.Thread, str]:
    thread = threading.Thread(target=server.serve_forever, name=name, daemon=True)
    thread.start()
    bound_host, bound_port = server.server_address[:2]
    display_host = "127.0.0.1" if bound_host in {"0.0.0.0", "::"} else bound_host
    return server, thread, f"http://{display_host}:{bound_port}"
