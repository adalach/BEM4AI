#!/usr/bin/env python3
"""Extract OneBuilding EPW/DDY archives into canonical BEM4AI weather paths.

This is the pipeline-side copy of the helper shipped with the published BEM4AI
dataset (``weather/extract_weather.py``). The logic is identical; only the
default paths differ, so that it resolves the repository layout used by
``notebooks/BEM4AI_6_Weather.ipynb``:

    data/interim/weather_download_links.csv   <- written by notebook 5
    output/weather/downloads/                 <- you put the ZIP archives here
    output/weather/epw/  output/weather/ddy/  <- written by this script

Standard library only; Python 3.9 or newer; Windows, macOS and Linux.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LINKS = PROJECT_ROOT / "data" / "interim" / "weather_download_links.csv"
DEFAULT_WEATHER_DIR = PROJECT_ROOT / "output" / "weather"
REQUIRED_COLUMNS = {"city_key", "download_link"}


def display_path(path: Path) -> str:
    """Format repository paths portably while preserving external paths."""
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class WeatherRow:
    city_key: str
    archive_name: str
    download_link: str


@dataclass(frozen=True)
class ArchiveMembers:
    epw: str
    ddy: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate downloaded OneBuilding ZIP archives, extract one "
            "EPW and one DDY from each archive, and rename them to the canonical "
            "BEM4AI city filenames."
        )
    )
    parser.add_argument(
        "--links",
        type=Path,
        default=DEFAULT_LINKS,
        help="Weather-link CSV (default: data/interim/weather_download_links.csv).",
    )
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=DEFAULT_WEATHER_DIR / "downloads",
        help="Directory holding the downloaded ZIPs (default: output/weather/downloads).",
    )
    parser.add_argument(
        "--epw-dir",
        type=Path,
        default=DEFAULT_WEATHER_DIR / "epw",
        help="EPW output directory (default: output/weather/epw).",
    )
    parser.add_argument(
        "--ddy-dir",
        type=Path,
        default=DEFAULT_WEATHER_DIR / "ddy",
        help="DDY output directory (default: output/weather/ddy).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the downloaded archives without creating EPW/DDY outputs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing expected EPW/DDY outputs.",
    )
    return parser.parse_args()


def archive_name_from_url(url: str) -> str:
    name = unquote(Path(urlsplit(url).path).name)
    if not name or not name.lower().endswith(".zip"):
        raise ValueError(f"Download link does not end in a ZIP filename: {url!r}")
    return name


def validate_city_key(city_key: str) -> None:
    if not city_key or city_key in {".", ".."}:
        raise ValueError(f"Invalid city_key: {city_key!r}")
    if Path(city_key).name != city_key or "/" in city_key or "\\" in city_key:
        raise ValueError(f"Unsafe city_key: {city_key!r}")


def load_rows(path: Path) -> list[WeatherRow]:
    if not path.is_file():
        raise FileNotFoundError(f"Weather-link CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Weather-link CSV is missing columns: " + ", ".join(sorted(missing))
            )
        source_rows = list(reader)

    if not source_rows:
        raise ValueError(f"Weather-link CSV is empty: {path}")

    rows: list[WeatherRow] = []
    seen_cities: set[str] = set()
    archive_urls: dict[str, str] = {}
    for line_number, source in enumerate(source_rows, start=2):
        city_key = (source.get("city_key") or "").strip()
        download_link = (source.get("download_link") or "").strip()
        validate_city_key(city_key)
        if city_key in seen_cities:
            raise ValueError(f"Duplicate city_key {city_key!r} at CSV line {line_number}")
        seen_cities.add(city_key)

        archive_name = archive_name_from_url(download_link)
        previous_url = archive_urls.get(archive_name)
        if previous_url is not None and previous_url != download_link:
            raise ValueError(
                f"Archive filename {archive_name!r} refers to multiple URLs: "
                f"{previous_url!r} and {download_link!r}"
            )
        archive_urls[archive_name] = download_link
        rows.append(WeatherRow(city_key, archive_name, download_link))

    return rows


def choose_member(names: list[str], suffix: str, archive_name: str) -> str:
    candidates = [name for name in names if name.lower().endswith(suffix)]
    if not candidates:
        raise ValueError(f"No {suffix} member found")
    if len(candidates) == 1:
        return candidates[0]

    expected_basename = f"{Path(archive_name).stem}{suffix}".lower()
    exact = [name for name in candidates if Path(name).name.lower() == expected_basename]
    if len(exact) == 1:
        return exact[0]
    listed = ", ".join(sorted(candidates))
    raise ValueError(f"Ambiguous {suffix} members: {listed}")


def validate_member_content(
    archive: zipfile.ZipFile, member: str, kind: str, archive_name: str
) -> None:
    with archive.open(member, "r") as source:
        sample = source.read(128 * 1024).decode("utf-8", errors="replace")
    if kind == "EPW" and not sample.lstrip("\ufeff\r\n ").upper().startswith("LOCATION,"):
        raise ValueError(f"{archive_name}: selected EPW does not begin with LOCATION")
    if kind == "DDY" and "SIZINGPERIOD:DESIGNDAY" not in sample.upper():
        raise ValueError(f"{archive_name}: selected DDY has no SizingPeriod:DesignDay")


def preflight_archives(
    rows: list[WeatherRow], downloads_dir: Path
) -> dict[str, ArchiveMembers]:
    if not downloads_dir.is_dir():
        raise FileNotFoundError(f"Downloads directory not found: {downloads_dir}")

    expected_names = sorted({row.archive_name for row in rows})
    missing = [name for name in expected_names if not (downloads_dir / name).is_file()]
    if missing:
        preview = "\n  ".join(missing[:20])
        remainder = len(missing) - min(len(missing), 20)
        suffix = f"\n  ... and {remainder} more" if remainder else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} required ZIP archive(s) in {downloads_dir}:\n  "
            f"{preview}{suffix}"
        )

    selected: dict[str, ArchiveMembers] = {}
    errors: list[str] = []
    for archive_name in expected_names:
        archive_path = downloads_dir / archive_name
        try:
            with zipfile.ZipFile(archive_path, "r") as archive:
                corrupt_member = archive.testzip()
                if corrupt_member is not None:
                    raise ValueError(f"CRC failure in member {corrupt_member!r}")
                names = [info.filename for info in archive.infolist() if not info.is_dir()]
                epw_member = choose_member(names, ".epw", archive_name)
                ddy_member = choose_member(names, ".ddy", archive_name)
                validate_member_content(archive, epw_member, "EPW", archive_name)
                validate_member_content(archive, ddy_member, "DDY", archive_name)
                selected[archive_name] = ArchiveMembers(epw_member, ddy_member)
        except Exception as exc:
            errors.append(f"{archive_name}: {exc}")

    if errors:
        raise ValueError(
            f"{len(errors)} archive(s) failed validation:\n  " + "\n  ".join(errors)
        )
    return selected


def validate_output_directory(
    directory: Path, expected_names: set[str], force: bool
) -> None:
    if not directory.exists():
        return
    if not directory.is_dir():
        raise ValueError(f"Output path is not a directory: {directory}")

    existing = {entry.name for entry in directory.iterdir()}
    unexpected = sorted(existing - expected_names)
    if unexpected:
        raise ValueError(
            f"Output directory contains unexpected files: {directory}\n  "
            + "\n  ".join(unexpected[:20])
        )
    if existing and not force:
        raise FileExistsError(
            f"Expected outputs already exist in {directory}. Use --force to overwrite them."
        )


def copy_member_atomic(
    archive: zipfile.ZipFile, member: str, destination: Path
) -> None:
    temporary = destination.with_name(f".{destination.name}.extracting")
    try:
        with archive.open(member, "r") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def extract(
    rows: list[WeatherRow],
    selected: dict[str, ArchiveMembers],
    downloads_dir: Path,
    epw_dir: Path,
    ddy_dir: Path,
    force: bool,
) -> None:
    expected_epw = {f"{row.city_key}.epw" for row in rows}
    expected_ddy = {f"{row.city_key}.ddy" for row in rows}
    validate_output_directory(epw_dir, expected_epw, force)
    validate_output_directory(ddy_dir, expected_ddy, force)
    epw_dir.mkdir(parents=True, exist_ok=True)
    ddy_dir.mkdir(parents=True, exist_ok=True)

    rows_by_archive: dict[str, list[WeatherRow]] = defaultdict(list)
    for row in rows:
        rows_by_archive[row.archive_name].append(row)

    for archive_name in sorted(rows_by_archive):
        members = selected[archive_name]
        with zipfile.ZipFile(downloads_dir / archive_name, "r") as archive:
            for row in rows_by_archive[archive_name]:
                copy_member_atomic(
                    archive, members.epw, epw_dir / f"{row.city_key}.epw"
                )
                copy_member_atomic(
                    archive, members.ddy, ddy_dir / f"{row.city_key}.ddy"
                )

    actual_epw = {entry.name for entry in epw_dir.iterdir() if entry.is_file()}
    actual_ddy = {entry.name for entry in ddy_dir.iterdir() if entry.is_file()}
    if actual_epw != expected_epw or actual_ddy != expected_ddy:
        raise RuntimeError("Output-name verification failed after extraction")


def main() -> int:
    args = parse_args()
    try:
        rows = load_rows(args.links)
        archive_count = len({row.archive_name for row in rows})
        print(f"City outputs expected: {len(rows)}")
        print(f"ZIP archives expected: {archive_count}")
        selected = preflight_archives(rows, args.downloads_dir)
        print(f"ZIP archives valid:    {len(selected)}")

        if args.check_only:
            print("Check complete; no EPW/DDY files were written.")
            return 0

        extract(
            rows=rows,
            selected=selected,
            downloads_dir=args.downloads_dir,
            epw_dir=args.epw_dir,
            ddy_dir=args.ddy_dir,
            force=args.force,
        )
        print(f"EPW files written:     {len(rows)} -> {display_path(args.epw_dir)}")
        print(f"DDY files written:     {len(rows)} -> {display_path(args.ddy_dir)}")
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
