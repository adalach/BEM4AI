#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sqlite3
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable


IDEAL_LOADS_COOLING_VAR = "Zone Ideal Loads Zone Total Cooling Rate"
IDEAL_LOADS_HEATING_VAR = "Zone Ideal Loads Zone Total Heating Rate"
IDEAL_LOADS_SUFFIX = " IDEAL LOADS AIR SYSTEM"

ENERGY_TO_KWH = {
    "J": 1.0 / 3_600_000.0,
    "kJ": 1.0 / 3_600.0,
    "MJ": 1.0 / 3.6,
    "GJ": 1_000.0 / 3.6,
    "Wh": 0.001,
    "kWh": 1.0,
    "MWh": 1_000.0,
}

SIM_COLUMNS = [
    "building_id",
    "eui_kwh_per_m2",
    "total_energy_kwh",
    "total_floor_area_m2",
    "conditioned_floor_area_m2",
    "heating_eui_kwh_per_m2",
    "cooling_eui_kwh_per_m2",
    "ref_floor_area_m2",
    "annual_heating_kwh",
    "annual_cooling_kwh",
    "annual_heating_plus_cooling_kwh",
    "peak_cooling_kw",
    "peak_heating_kw",
    "max_zone_cooling_kw",
    "max_zone_heating_kw",
    "n_cooling_zones",
    "n_heating_zones",
    "peak_cooling_kw_per_m2",
    "peak_heating_kw_per_m2",
    "peak_cooling_sim_time",
    "peak_heating_sim_time",
]

ZONE_COLUMNS = [
    "building_id",
    "zone_id",
    "zone_area_m2",
    "zone_peak_cooling_kw",
    "zone_peak_cooling_time",
    "zone_cooling_kw_at_building_peak",
    "building_total_peak_cooling_kw",
    "building_peak_cooling_time",
    "zone_peak_heating_kw",
    "zone_peak_heating_time",
    "zone_heating_kw_at_building_peak",
    "building_total_peak_heating_kw",
    "building_peak_heating_time",
]

HOURLY_COLUMNS = [
    "building_id",
    "zone_id",
    "month_day_hour",
    "cooling_kw",
    "heating_kw",
]

ERROR_COLUMNS = ["building_id", "path", "error"]
LOAD_VAR_TO_LABEL = {
    IDEAL_LOADS_COOLING_VAR: "cooling",
    IDEAL_LOADS_HEATING_VAR: "heating",
}


def default_jobs() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open each EnergyPlus SQL file once and extract building-level "
            "simulation metrics plus merged per-zone cooling/heating peak loads."
        )
    )
    parser.add_argument(
        "--sql-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing EnergyPlus .sql files. Default: current working directory.",
    )
    parser.add_argument(
        "--sim-output",
        type=Path,
        default=Path.cwd() / "sim_metrics.csv",
        help="Output CSV path for building-level metrics.",
    )
    parser.add_argument(
        "--zone-output",
        type=Path,
        default=Path.cwd() / "zone_peak_loads.csv",
        help="Output CSV path for merged per-zone cooling/heating peak loads.",
    )
    parser.add_argument(
        "--errors-output",
        type=Path,
        default=Path.cwd() / "extract_energyplus_sql_metrics_errors.csv",
        help="Output CSV path for files that failed to parse.",
    )
    parser.add_argument(
        "--hourly-zone-parquet",
        type=Path,
        default=None,
        help=(
            "Optional output directory for full-year hourly zone loads in long-format "
            "parquet files, one file per building "
            "(building_id, zone_id, month_day_hour, cooling_kw, heating_kw)."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=default_jobs(),
        help="Number of SQL files to process in parallel. Default: up to 8 workers.",
    )
    parser.add_argument(
        "--executor",
        choices=("auto", "process", "thread"),
        default="auto",
        help="Parallel executor type. 'auto' prefers processes and falls back to threads.",
    )
    parser.add_argument(
        "--flush-every-models",
        type=int,
        default=10,
        help="Flush CSV and parquet writers after this many processed SQL files. Default: 10.",
    )
    parser.add_argument(
        "--parquet-compression",
        choices=("snappy", "zstd", "gzip", "brotli", "lz4", "none"),
        default="snappy",
        help="Compression codec for optional parquet output. Default: snappy.",
    )
    parser.add_argument(
        "--resume",
        choices=("ask", "continue", "overwrite", "error"),
        default="ask",
        help=(
            "How to handle existing outputs. "
            "'ask' prompts to continue or overwrite, 'continue' resumes from the "
            "last common completed building, 'overwrite' clears prior outputs, "
            "and 'error' stops if outputs already exist."
        ),
    )
    return parser.parse_args()


def normalize_building_id(sql_path: Path) -> str:
    stem = sql_path.stem
    if stem.endswith("out"):
        return stem[:-3]
    return stem


def normalize_zone_id(key_value: str) -> str:
    if key_value.endswith(IDEAL_LOADS_SUFFIX):
        return key_value[: -len(IDEAL_LOADS_SUFFIX)]
    return key_value


def parse_float(value: object) -> float:
    text = str(value).strip().replace(",", "")
    if not text:
        raise ValueError("Empty numeric value.")
    return float(text)


def energy_value_to_kwh(value: object, units: object) -> float:
    unit = str(units).strip()
    factor = ENERGY_TO_KWH.get(unit)
    if factor is None:
        raise ValueError(f"Unsupported energy unit: {unit!r}")
    return parse_float(value) * factor


def find_sql_files(sql_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in sql_dir.glob("*.sql")
        if path.is_file() and not path.name.startswith("._")
    )


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def read_csv_building_ids(csv_path: Path, column_name: str) -> list[str]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return unique_in_order(str(row[column_name]) for row in reader if row.get(column_name))


def count_csv_data_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def hourly_dataset_file_path(dataset_dir: Path, building_id: str) -> Path:
    return dataset_dir / f"{building_id}.parquet"


def read_hourly_dataset_building_ids(dataset_dir: Path) -> list[str]:
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return []
    return sorted(path.stem for path in dataset_dir.glob("*.parquet") if path.is_file())


def determine_resume_prefix(
    sql_files: list[Path],
    sim_output_path: Path,
    zone_output_path: Path,
    hourly_zone_parquet_path: Path | None,
) -> list[str]:
    sim_ids = set(read_csv_building_ids(sim_output_path, "building_id"))
    zone_ids = set(read_csv_building_ids(zone_output_path, "building_id"))
    hourly_ids = (
        set(read_hourly_dataset_building_ids(hourly_zone_parquet_path))
        if hourly_zone_parquet_path is not None
        else None
    )

    completed_prefix: list[str] = []
    for sql_path in sql_files:
        building_id = normalize_building_id(sql_path)
        if building_id not in sim_ids or building_id not in zone_ids:
            break
        if hourly_ids is not None and building_id not in hourly_ids:
            break
        completed_prefix.append(building_id)
    return completed_prefix


def rewrite_csv_with_allowed_buildings(
    csv_path: Path,
    fieldnames: list[str],
    building_column: str,
    allowed_buildings: set[str],
) -> None:
    if not csv_path.exists():
        return
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get(building_column) in allowed_buildings]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def remove_outputs_for_nonprefix_buildings(
    completed_prefix: list[str],
    sim_output_path: Path,
    zone_output_path: Path,
    errors_output_path: Path,
    hourly_zone_parquet_path: Path | None,
) -> None:
    allowed_buildings = set(completed_prefix)
    rewrite_csv_with_allowed_buildings(sim_output_path, SIM_COLUMNS, "building_id", allowed_buildings)
    rewrite_csv_with_allowed_buildings(zone_output_path, ZONE_COLUMNS, "building_id", allowed_buildings)
    if errors_output_path.exists():
        errors_output_path.unlink()

    if hourly_zone_parquet_path is not None and hourly_zone_parquet_path.exists():
        hourly_zone_parquet_path.mkdir(parents=True, exist_ok=True)
        for path in hourly_zone_parquet_path.glob("*.parquet"):
            if path.stem not in allowed_buildings:
                path.unlink()


def clear_output_paths(
    sim_output_path: Path,
    zone_output_path: Path,
    errors_output_path: Path,
    hourly_zone_parquet_path: Path | None,
) -> None:
    for path in (sim_output_path, zone_output_path, errors_output_path):
        if path.exists():
            path.unlink()
    if hourly_zone_parquet_path is not None and hourly_zone_parquet_path.exists():
        if hourly_zone_parquet_path.is_dir():
            shutil.rmtree(hourly_zone_parquet_path)
        else:
            hourly_zone_parquet_path.unlink()


def outputs_exist(
    sim_output_path: Path,
    zone_output_path: Path,
    errors_output_path: Path,
    hourly_zone_parquet_path: Path | None,
) -> bool:
    return any(
        path.exists()
        for path in (
            sim_output_path,
            zone_output_path,
            errors_output_path,
            hourly_zone_parquet_path,
        )
        if path is not None
    )


def resolve_resume_mode(
    resume_mode: str,
    completed_prefix: list[str],
    total_sql_files: int,
) -> str:
    if resume_mode != "ask":
        return resume_mode

    prompt = (
        f"Found existing outputs with {len(completed_prefix)} common completed model(s) "
        f"out of {total_sql_files}. Continue from the next model? [Y/n]: "
    )
    if not sys.stdin.isatty():
        print(prompt + "Y")
        return "continue"

    answer = input(prompt).strip().lower()
    if answer in ("", "y", "yes"):
        return "continue"
    return "overwrite"


def format_sim_time(row: sqlite3.Row) -> str:
    return (
        f"{int(row['Year']):04d}-{int(row['Month']):02d}-{int(row['Day']):02d} "
        f"{int(row['Hour']):02d}:{int(row['Minute']):02d}"
    )


def month_day_hour_code_from_row(row: sqlite3.Row) -> int:
    return int(row["Month"]) * 10000 + int(row["Day"]) * 100 + int(row["Hour"])


def get_annual_run_period_index(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """
        SELECT EnvironmentPeriodIndex
        FROM EnvironmentPeriods
        WHERE EnvironmentType = 3
        ORDER BY EnvironmentPeriodIndex
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise ValueError(
            "Could not find annual run period (EnvironmentType = 3). "
            "This extractor reads the annual EPW run, not DDY design days."
        )
    return int(row[0])


def load_zone_areas(conn: sqlite3.Connection) -> dict[str, float]:
    rows = conn.execute(
        """
        SELECT RowName, Value
        FROM TabularDataWithStrings
        WHERE ReportName = 'InputVerificationandResultsSummary'
          AND TableName = 'Zone Summary'
          AND ColumnName = 'Area'
          AND RowName NOT IN ('Total', 'Conditioned Total', 'Unconditioned Total', 'Not Part of Total')
        """
    ).fetchall()
    areas: dict[str, float] = {}
    for zone_id, value in rows:
        try:
            areas[str(zone_id)] = parse_float(value)
        except ValueError:
            continue
    return areas


def extract_eui_dict(conn: sqlite3.Connection) -> dict[str, object]:
    area_rows = conn.execute(
        """
        SELECT RowName, Value
        FROM TabularDataWithStrings
        WHERE TableName = 'Building Area'
          AND ColumnName = 'Area'
        """
    ).fetchall()
    area_map = {str(row_name): parse_float(value) for row_name, value in area_rows}
    try:
        total_floor_area = area_map["Total Building Area"]
        conditioned_floor_area = area_map["Net Conditioned Building Area"]
    except KeyError as exc:
        raise ValueError('Failed to find the "Building Area" table in the .sql file.') from exc

    end_uses_kwh: OrderedDict[str, float] = OrderedDict()
    total_energy_kwh = 0.0
    end_use_rows = conn.execute(
        """
        SELECT RowName, Units, Value
        FROM TabularDataWithStrings
        WHERE TableName = 'End Uses By Subcategory'
        """
    ).fetchall()
    for row_name, units, value in end_use_rows:
        if str(units).strip() not in ENERGY_TO_KWH:
            continue
        total_use_kwh = energy_value_to_kwh(value, units)
        if total_use_kwh == 0:
            continue
        total_energy_kwh += total_use_kwh
        category, _, sub_category = str(row_name).partition(":")
        end_use_name = category if sub_category in ("", "General", "Other") else sub_category
        end_uses_kwh[end_use_name] = end_uses_kwh.get(end_use_name, 0.0) + total_use_kwh

    if total_floor_area:
        return {
            "eui": round(total_energy_kwh / total_floor_area, 3),
            "total_floor_area": total_floor_area,
            "conditioned_floor_area": conditioned_floor_area,
            "total_energy": round(total_energy_kwh, 3),
            "end_uses": OrderedDict(
                (key, round(value / total_floor_area, 3))
                for key, value in end_uses_kwh.items()
            ),
        }

    return {
        "eui": 0.0,
        "total_floor_area": total_floor_area,
        "conditioned_floor_area": conditioned_floor_area,
        "total_energy": round(total_energy_kwh, 3),
        "end_uses": OrderedDict((key, 0.0) for key in end_uses_kwh.keys()),
    }


def load_peak_summary(conn: sqlite3.Connection) -> dict[str, dict[str, object]]:
    run_period_index = get_annual_run_period_index(conn)
    rows = conn.execute(
        """
        WITH annual_times AS (
            SELECT TimeIndex, Year, Month, Day, Hour, Minute
            FROM Time
            WHERE EnvironmentPeriodIndex = ?
              AND IntervalType = 1
        ),
        filtered AS (
            SELECT
                rdd.Name,
                rdd.KeyValue,
                rd.TimeIndex,
                rd.Value,
                at.Year,
                at.Month,
                at.Day,
                at.Hour,
                at.Minute
            FROM ReportData rd
            JOIN annual_times at
                ON rd.TimeIndex = at.TimeIndex
            JOIN ReportDataDictionary rdd
                ON rd.ReportDataDictionaryIndex = rdd.ReportDataDictionaryIndex
            WHERE rdd.ReportingFrequency = 'Hourly'
              AND rdd.Name IN (?, ?)
        ),
        zone_ranked AS (
            SELECT
                Name,
                KeyValue,
                Value AS zone_peak_w,
                Year,
                Month,
                Day,
                Hour,
                Minute,
                ROW_NUMBER() OVER (
                    PARTITION BY Name, KeyValue
                    ORDER BY Value DESC, Year, Month, Day, Hour, Minute
                ) AS rn
            FROM filtered
        ),
        time_ranked AS (
            SELECT
                Name,
                TimeIndex,
                SUM(Value) AS building_total_w,
                Year,
                Month,
                Day,
                Hour,
                Minute,
                ROW_NUMBER() OVER (
                    PARTITION BY Name
                    ORDER BY SUM(Value) DESC, Year, Month, Day, Hour, Minute
                ) AS rn
            FROM filtered
            GROUP BY Name, TimeIndex
        )
        SELECT
            'zone_peak' AS row_kind,
            Name,
            KeyValue,
            zone_peak_w AS zone_value_w,
            NULL AS building_total_w,
            Year,
            Month,
            Day,
            Hour,
            Minute
        FROM zone_ranked
        WHERE rn = 1
        UNION ALL
        SELECT
            'building_peak_zone' AS row_kind,
            f.Name,
            f.KeyValue,
            f.Value AS zone_value_w,
            tr.building_total_w,
            tr.Year,
            tr.Month,
            tr.Day,
            tr.Hour,
            tr.Minute
        FROM filtered f
        JOIN time_ranked tr
            ON f.Name = tr.Name
           AND f.TimeIndex = tr.TimeIndex
        WHERE tr.rn = 1
        ORDER BY Name, row_kind, KeyValue
        """,
        (
            run_period_index,
            IDEAL_LOADS_COOLING_VAR,
            IDEAL_LOADS_HEATING_VAR,
        ),
    )

    states = {
        IDEAL_LOADS_COOLING_VAR: {
            "building_peak_sum_w": None,
            "building_peak_time_text": None,
            "building_peak_zone_values_w": {},
            "zone_peaks": {},
        },
        IDEAL_LOADS_HEATING_VAR: {
            "building_peak_sum_w": None,
            "building_peak_time_text": None,
            "building_peak_zone_values_w": {},
            "zone_peaks": {},
        },
    }

    for row in rows:
        load_name = str(row["Name"])
        state = states[load_name]
        zone_id = normalize_zone_id(str(row["KeyValue"]))
        zone_value_w = float(row["zone_value_w"])
        time_text = format_sim_time(row)

        if str(row["row_kind"]) == "zone_peak":
            zone_peaks = state["zone_peaks"]  # type: ignore[assignment]
            zone_peaks[zone_id] = (zone_value_w, time_text)
            continue

        if state["building_peak_sum_w"] is None:
            state["building_peak_sum_w"] = float(row["building_total_w"])
            state["building_peak_time_text"] = time_text
        building_peak_zone_values_w = state["building_peak_zone_values_w"]  # type: ignore[assignment]
        building_peak_zone_values_w[zone_id] = zone_value_w

    for load_name, state in states.items():
        if state["building_peak_sum_w"] is None:
            raise ValueError(f"Could not find hourly peak data for {LOAD_VAR_TO_LABEL[load_name]}.")

    return states


def query_hourly_load_rows(conn: sqlite3.Connection):
    run_period_index = get_annual_run_period_index(conn)
    return conn.execute(
        """
        WITH annual_times AS (
            SELECT TimeIndex, Year, Month, Day, Hour, Minute
            FROM Time
            WHERE EnvironmentPeriodIndex = ?
              AND IntervalType = 1
        )
        SELECT
            rdd.Name,
            rdd.KeyValue,
            rd.TimeIndex,
            rd.Value,
            at.Year,
            at.Month,
            at.Day,
            at.Hour,
            at.Minute
        FROM ReportData rd
        JOIN annual_times at
            ON rd.TimeIndex = at.TimeIndex
        JOIN ReportDataDictionary rdd
            ON rd.ReportDataDictionaryIndex = rdd.ReportDataDictionaryIndex
        WHERE rdd.ReportingFrequency = 'Hourly'
          AND rdd.Name IN (?, ?)
        ORDER BY rd.TimeIndex, rdd.KeyValue, rdd.Name
        """,
        (
            run_period_index,
            IDEAL_LOADS_COOLING_VAR,
            IDEAL_LOADS_HEATING_VAR,
        ),
    )


def _empty_detailed_peak_state() -> dict[str, object]:
    return {
        "building_peak_sum_w": None,
        "building_peak_time_text": None,
        "building_peak_zone_values_w": {},
        "zone_peaks": {},
    }


def _update_zone_peak(
    zone_peaks: dict[str, tuple[float, str]],
    zone_id: str,
    value_w: float,
    time_text: str,
) -> None:
    previous = zone_peaks.get(zone_id)
    if previous is None or value_w > previous[0]:
        zone_peaks[zone_id] = (value_w, time_text)


def _finalize_hour_group(
    current_time_text: str | None,
    current_month_day_hour: int | None,
    current_zone_values: dict[str, dict[str, float]],
    current_sums: dict[str, float],
    states: dict[str, dict[str, object]],
    hourly_columns: dict[str, list[object]] | None,
    building_id: str,
) -> None:
    if current_time_text is None:
        return

    for zone_id, loads in current_zone_values.items():
        cooling_w = loads["cooling_w"]
        heating_w = loads["heating_w"]
        if hourly_columns is not None and current_month_day_hour is not None:
            hourly_columns["building_id"].append(building_id)
            hourly_columns["zone_id"].append(zone_id)
            hourly_columns["month_day_hour"].append(current_month_day_hour)
            hourly_columns["cooling_kw"].append(cooling_w / 1000.0)
            hourly_columns["heating_kw"].append(heating_w / 1000.0)

    for load_name, label in LOAD_VAR_TO_LABEL.items():
        state = states[load_name]
        candidate_sum = current_sums[label]
        peak_sum = state["building_peak_sum_w"]
        if peak_sum is None or candidate_sum > peak_sum:
            state["building_peak_sum_w"] = candidate_sum
            state["building_peak_time_text"] = current_time_text
            state["building_peak_zone_values_w"] = {
                zone_id: loads[f"{label}_w"] for zone_id, loads in current_zone_values.items()
            }


def extract_hourly_rows_and_peak_summary(
    conn: sqlite3.Connection,
    building_id: str,
    include_hourly_rows: bool,
) -> tuple[dict[str, dict[str, object]], dict[str, list[object]] | None]:
    rows = query_hourly_load_rows(conn)
    states = {
        IDEAL_LOADS_COOLING_VAR: _empty_detailed_peak_state(),
        IDEAL_LOADS_HEATING_VAR: _empty_detailed_peak_state(),
    }
    hourly_columns = {column: [] for column in HOURLY_COLUMNS} if include_hourly_rows else None

    current_time_index: int | None = None
    current_time_text: str | None = None
    current_month_day_hour: int | None = None
    current_zone_values: dict[str, dict[str, float]] = {}
    current_sums = {"cooling": 0.0, "heating": 0.0}

    for row in rows:
        time_index = int(row["TimeIndex"])
        if current_time_index is not None and time_index != current_time_index:
            _finalize_hour_group(
                current_time_text,
                current_month_day_hour,
                current_zone_values,
                current_sums,
                states,
                hourly_columns,
                building_id,
            )
            current_zone_values = {}
            current_sums = {"cooling": 0.0, "heating": 0.0}

        if current_time_index != time_index:
            current_time_index = time_index
            current_time_text = format_sim_time(row)
            current_month_day_hour = month_day_hour_code_from_row(row)

        zone_id = normalize_zone_id(str(row["KeyValue"]))
        value_w = float(row["Value"])
        label = LOAD_VAR_TO_LABEL[str(row["Name"])]

        zone_values = current_zone_values.setdefault(zone_id, {"cooling_w": 0.0, "heating_w": 0.0})
        zone_values[f"{label}_w"] = value_w
        current_sums[label] += value_w

        zone_peaks = states[str(row["Name"])]["zone_peaks"]  # type: ignore[assignment]
        _update_zone_peak(zone_peaks, zone_id, value_w, current_time_text)

    _finalize_hour_group(
        current_time_text,
        current_month_day_hour,
        current_zone_values,
        current_sums,
        states,
        hourly_columns,
        building_id,
    )

    for load_name, state in states.items():
        if state["building_peak_sum_w"] is None:
            raise ValueError(f"Could not find hourly peak data for {LOAD_VAR_TO_LABEL[load_name]}.")

    return states, hourly_columns


def build_sim_row(
    building_id: str,
    eui_dict: dict[str, object],
    peak_states: dict[str, dict[str, object]],
) -> dict[str, float | int | str | None]:
    cooling_state = peak_states[IDEAL_LOADS_COOLING_VAR]
    heating_state = peak_states[IDEAL_LOADS_HEATING_VAR]

    total_floor_area = eui_dict.get("total_floor_area")
    conditioned_floor_area = eui_dict.get("conditioned_floor_area")
    ref_floor_area = conditioned_floor_area or total_floor_area or 0.0
    end_uses = eui_dict.get("end_uses") or {}
    heating_eui = end_uses.get("Heating", 0.0)
    cooling_eui = end_uses.get("Cooling", 0.0)
    annual_heating = heating_eui * ref_floor_area
    annual_cooling = cooling_eui * ref_floor_area

    return {
        "building_id": building_id,
        "eui_kwh_per_m2": eui_dict.get("eui"),
        "total_energy_kwh": eui_dict.get("total_energy"),
        "total_floor_area_m2": total_floor_area,
        "conditioned_floor_area_m2": conditioned_floor_area,
        "heating_eui_kwh_per_m2": heating_eui,
        "cooling_eui_kwh_per_m2": cooling_eui,
        "ref_floor_area_m2": ref_floor_area,
        "annual_heating_kwh": annual_heating,
        "annual_cooling_kwh": annual_cooling,
        "annual_heating_plus_cooling_kwh": annual_heating + annual_cooling,
        "peak_cooling_kw": float(cooling_state["building_peak_sum_w"]) / 1000.0,
        "peak_heating_kw": float(heating_state["building_peak_sum_w"]) / 1000.0,
        "max_zone_cooling_kw": max(
            (peak[0] for peak in cooling_state["zone_peaks"].values()),  # type: ignore[union-attr]
            default=0.0,
        )
        / 1000.0,
        "max_zone_heating_kw": max(
            (peak[0] for peak in heating_state["zone_peaks"].values()),  # type: ignore[union-attr]
            default=0.0,
        )
        / 1000.0,
        "n_cooling_zones": len(cooling_state["zone_peaks"]),  # type: ignore[arg-type]
        "n_heating_zones": len(heating_state["zone_peaks"]),  # type: ignore[arg-type]
        "peak_cooling_kw_per_m2": (float(cooling_state["building_peak_sum_w"]) / 1000.0 / ref_floor_area)
        if ref_floor_area
        else None,
        "peak_heating_kw_per_m2": (float(heating_state["building_peak_sum_w"]) / 1000.0 / ref_floor_area)
        if ref_floor_area
        else None,
        "peak_cooling_sim_time": cooling_state["building_peak_time_text"],
        "peak_heating_sim_time": heating_state["building_peak_time_text"],
    }


def build_zone_rows(
    building_id: str,
    zone_areas: dict[str, float],
    peak_states: dict[str, dict[str, object]],
) -> list[dict[str, float | str | None]]:
    cooling_state = peak_states[IDEAL_LOADS_COOLING_VAR]
    heating_state = peak_states[IDEAL_LOADS_HEATING_VAR]

    cooling_zone_peaks = cooling_state["zone_peaks"]  # type: ignore[assignment]
    heating_zone_peaks = heating_state["zone_peaks"]  # type: ignore[assignment]
    cooling_peak_snapshot = cooling_state["building_peak_zone_values_w"]  # type: ignore[assignment]
    heating_peak_snapshot = heating_state["building_peak_zone_values_w"]  # type: ignore[assignment]

    zone_ids = sorted(
        set(zone_areas)
        | set(cooling_zone_peaks)
        | set(heating_zone_peaks)
        | set(cooling_peak_snapshot)
        | set(heating_peak_snapshot)
    )

    zone_rows: list[dict[str, float | str | None]] = []
    for zone_id in zone_ids:
        cooling_zone_peak = cooling_zone_peaks.get(zone_id)
        heating_zone_peak = heating_zone_peaks.get(zone_id)
        cooling_at_building_peak = cooling_peak_snapshot.get(zone_id)
        heating_at_building_peak = heating_peak_snapshot.get(zone_id)
        zone_rows.append(
            {
                "building_id": building_id,
                "zone_id": zone_id,
                "zone_area_m2": zone_areas.get(zone_id),
                "zone_peak_cooling_kw": (cooling_zone_peak[0] / 1000.0) if cooling_zone_peak is not None else None,
                "zone_peak_cooling_time": cooling_zone_peak[1] if cooling_zone_peak is not None else None,
                "zone_cooling_kw_at_building_peak": (
                    cooling_at_building_peak / 1000.0 if cooling_at_building_peak is not None else None
                ),
                "building_total_peak_cooling_kw": float(cooling_state["building_peak_sum_w"]) / 1000.0,
                "building_peak_cooling_time": cooling_state["building_peak_time_text"],
                "zone_peak_heating_kw": (heating_zone_peak[0] / 1000.0) if heating_zone_peak is not None else None,
                "zone_peak_heating_time": heating_zone_peak[1] if heating_zone_peak is not None else None,
                "zone_heating_kw_at_building_peak": (
                    heating_at_building_peak / 1000.0 if heating_at_building_peak is not None else None
                ),
                "building_total_peak_heating_kw": float(heating_state["building_peak_sum_w"]) / 1000.0,
                "building_peak_heating_time": heating_state["building_peak_time_text"],
            }
        )
    return zone_rows


def _hourly_columns_to_arrow_table(hourly_columns: dict[str, list[object]]):
    import pyarrow as pa

    if not hourly_columns or not hourly_columns["building_id"]:
        return None

    return pa.table(
        {
            "building_id": pa.array(hourly_columns["building_id"]),
            "zone_id": pa.array(hourly_columns["zone_id"]),
            "month_day_hour": pa.array(hourly_columns["month_day_hour"], type=pa.int32()),
            "cooling_kw": pa.array(hourly_columns["cooling_kw"], type=pa.float32()),
            "heating_kw": pa.array(hourly_columns["heating_kw"], type=pa.float32()),
        }
    )


def extract_from_sql(
    sql_path: Path,
    include_hourly_rows: bool,
) -> tuple[dict[str, float | int | str | None], list[dict[str, float | str | None]], object | None]:
    building_id = normalize_building_id(sql_path)
    conn = sqlite3.connect(sql_path)
    try:
        conn.row_factory = sqlite3.Row
        eui_dict = extract_eui_dict(conn)
        zone_areas = load_zone_areas(conn)
        if include_hourly_rows:
            peak_states, hourly_columns = extract_hourly_rows_and_peak_summary(conn, building_id, True)
            hourly_table = _hourly_columns_to_arrow_table(hourly_columns or {})
        else:
            peak_states = load_peak_summary(conn)
            hourly_table = None
    finally:
        conn.close()

    sim_row = build_sim_row(building_id, eui_dict, peak_states)
    zone_rows = build_zone_rows(building_id, zone_areas, peak_states)
    return sim_row, zone_rows, hourly_table


def _extract_sql_safe(sql_path_str: str, include_hourly_rows: bool) -> dict[str, object]:
    sql_path = Path(sql_path_str)
    building_id = normalize_building_id(sql_path)
    try:
        sim_row, zone_rows, hourly_table = extract_from_sql(sql_path, include_hourly_rows)
        return {
            "ok": True,
            "sim_row": sim_row,
            "zone_rows": zone_rows,
            "hourly_table": hourly_table,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "ok": False,
            "error": {
                "building_id": building_id,
                "path": str(sql_path),
                "error": repr(exc),
            },
        }


def iter_results_with_executor(
    worker: Callable[..., dict[str, object]],
    item_strs: list[str],
    jobs: int,
    executor_kind: str,
    include_hourly_rows: bool,
):
    if executor_kind == "thread":
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(worker, item_str, include_hourly_rows)
                for item_str in item_strs
            ]
            for future in as_completed(futures):
                yield future.result()
        return

    if executor_kind == "process":
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(worker, item_str, include_hourly_rows)
                for item_str in item_strs
            ]
            for future in as_completed(futures):
                yield future.result()
        return

    try:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(worker, item_str, include_hourly_rows)
                for item_str in item_strs
            ]
            for future in as_completed(futures):
                yield future.result()
    except (PermissionError, OSError):
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(worker, item_str, include_hourly_rows)
                for item_str in item_strs
            ]
            for future in as_completed(futures):
                yield future.result()


class CsvStreamWriter:
    def __init__(self, output_path: Path, columns: list[str], append: bool) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self._handle = output_path.open(mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=columns)
        if not append:
            self._writer.writeheader()

    def write_row(self, row: dict[str, object]) -> None:
        self._writer.writerow(row)

    def write_rows(self, rows: Iterable[dict[str, object]]) -> None:
        for row in rows:
            self._writer.writerow(row)

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class ParquetStreamWriter:
    def __init__(self, output_path: Path, compression: str) -> None:
        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Optional parquet export requires pyarrow. "
                "Install it in the active environment and rerun with --hourly-zone-parquet."
            ) from exc

        if output_path.exists() and not output_path.is_dir():
            raise ValueError(
                f"Hourly parquet path {output_path} already exists as a file. "
                "Use a directory path or delete the existing file."
            )
        output_path.mkdir(parents=True, exist_ok=True)
        self._output_path = output_path
        self._compression = None if compression == "none" else compression

    def write_model_table(self, building_id: str, table: object | None) -> None:
        if table is None:
            return
        import pyarrow.parquet as pq

        pq.write_table(
            table,
            hourly_dataset_file_path(self._output_path, building_id),
            compression=self._compression,
        )

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def run_extraction(
    sql_dir: Path,
    sim_output_path: Path,
    zone_output_path: Path,
    errors_output_path: Path,
    hourly_zone_parquet_path: Path | None,
    jobs: int,
    executor_kind: str,
    flush_every_models: int,
    parquet_compression: str,
    resume_mode: str,
) -> int:
    sql_files = find_sql_files(sql_dir)
    if not sql_files:
        raise FileNotFoundError(f"No .sql files found in {sql_dir}")

    include_hourly_rows = hourly_zone_parquet_path is not None
    completed_prefix: list[str] = []
    if outputs_exist(sim_output_path, zone_output_path, errors_output_path, hourly_zone_parquet_path):
        completed_prefix = determine_resume_prefix(
            sql_files,
            sim_output_path,
            zone_output_path,
            hourly_zone_parquet_path if include_hourly_rows else None,
        )
        chosen_resume_mode = resolve_resume_mode(resume_mode, completed_prefix, len(sql_files))
        if chosen_resume_mode == "error":
            raise FileExistsError(
                "Output files already exist. Use --resume continue, --resume overwrite, or --resume ask."
            )
        if chosen_resume_mode == "overwrite":
            clear_output_paths(
                sim_output_path,
                zone_output_path,
                errors_output_path,
                hourly_zone_parquet_path,
            )
            completed_prefix = []
        else:
            remove_outputs_for_nonprefix_buildings(
                completed_prefix,
                sim_output_path,
                zone_output_path,
                errors_output_path,
                hourly_zone_parquet_path if include_hourly_rows else None,
            )

    remaining_sql_files = sql_files[len(completed_prefix) :]
    if not remaining_sql_files:
        print("All SQL files are already completed; nothing to do.")
        return 0

    if completed_prefix:
        print(
            f"Continuing from model {len(completed_prefix) + 1} of {len(sql_files)} "
            f"after completed prefix ending at {completed_prefix[-1]}."
        )
    elif outputs_exist(sim_output_path, zone_output_path, errors_output_path, hourly_zone_parquet_path):
        print("Existing outputs did not contain a common completed prefix; restarting from the beginning.")

    effective_executor = executor_kind
    if include_hourly_rows and executor_kind in ("auto", "process"):
        effective_executor = "thread"
        print(
            "Hourly parquet export enabled; using thread executor to avoid "
            "serializing large hourly batches between worker processes."
        )

    append_mode = bool(completed_prefix)
    sim_writer = CsvStreamWriter(sim_output_path, SIM_COLUMNS, append=append_mode)
    zone_writer = CsvStreamWriter(zone_output_path, ZONE_COLUMNS, append=append_mode)
    error_writer = CsvStreamWriter(errors_output_path, ERROR_COLUMNS, append=False)
    parquet_writer = (
        ParquetStreamWriter(hourly_zone_parquet_path, parquet_compression)
        if include_hourly_rows
        else None
    )

    used_executor = "serial" if jobs == 1 else effective_executor
    n_sim_rows = len(completed_prefix)
    n_zone_rows = count_csv_data_rows(zone_output_path) if append_mode else 0
    n_bad_rows = 0

    def handle_result(result: dict[str, object]) -> None:
        nonlocal n_sim_rows, n_zone_rows, n_bad_rows
        if result["ok"]:
            sim_row = result["sim_row"]  # type: ignore[assignment]
            zone_rows = result["zone_rows"]  # type: ignore[assignment]
            hourly_table = result.get("hourly_table")
            sim_writer.write_row(sim_row)
            zone_writer.write_rows(zone_rows)
            n_sim_rows += 1
            n_zone_rows += len(zone_rows)
            if parquet_writer is not None:
                parquet_writer.write_model_table(str(sim_row["building_id"]), hourly_table)
        else:
            error_writer.write_row(result["error"])  # type: ignore[arg-type]
            n_bad_rows += 1

    try:
        if jobs == 1:
            processed = 0
            for sql_path in remaining_sql_files:
                handle_result(_extract_sql_safe(str(sql_path), include_hourly_rows))
                processed += 1
                if processed % flush_every_models == 0:
                    sim_writer.flush()
                    zone_writer.flush()
                    error_writer.flush()
                    if parquet_writer is not None:
                        parquet_writer.flush()
        else:
            for processed, result in enumerate(
                iter_results_with_executor(
                    _extract_sql_safe,
                    [str(path) for path in remaining_sql_files],
                    jobs,
                    effective_executor,
                    include_hourly_rows,
                ),
                start=1,
            ):
                handle_result(result)
                if processed % flush_every_models == 0:
                    sim_writer.flush()
                    zone_writer.flush()
                    error_writer.flush()
                    if parquet_writer is not None:
                        parquet_writer.flush()
    finally:
        sim_writer.close()
        zone_writer.close()
        error_writer.close()
        if parquet_writer is not None:
            parquet_writer.close()

    if n_bad_rows:
        print(
            f"Extracted {n_sim_rows} building rows to {sim_output_path} and "
            f"{n_zone_rows} zone rows to {zone_output_path} using {used_executor}. "
            f"{n_bad_rows} files failed; details written to {errors_output_path}."
        )
        return 1

    message = (
        f"Extracted {n_sim_rows} building rows to {sim_output_path} and "
        f"{n_zone_rows} zone rows to {zone_output_path} using {used_executor}."
    )
    if hourly_zone_parquet_path is not None:
        message += f" Hourly zone loads written to {hourly_zone_parquet_path}."
    print(message)
    return 0


def main() -> int:
    args = parse_args()
    return run_extraction(
        sql_dir=args.sql_dir.expanduser().resolve(),
        sim_output_path=args.sim_output.expanduser().resolve(),
        zone_output_path=args.zone_output.expanduser().resolve(),
        errors_output_path=args.errors_output.expanduser().resolve(),
        hourly_zone_parquet_path=(
            args.hourly_zone_parquet.expanduser().resolve()
            if args.hourly_zone_parquet is not None
            else None
        ),
        jobs=max(1, int(args.jobs)),
        executor_kind=args.executor,
        flush_every_models=max(1, int(args.flush_every_models)),
        parquet_compression=args.parquet_compression,
        resume_mode=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
