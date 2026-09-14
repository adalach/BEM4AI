# Extract EnergyPlus SQL Metrics

`extract_energyplus_sql_metrics.py` is the project-level script for extracting
building and zone metrics from EnergyPlus `.sql` outputs.

It opens each SQL file once and writes:

- `sim_metrics.csv`
  One row per building
- `zone_peak_loads.csv`
  One row per zone, with both cooling and heating peak summaries
- optionally, a long-format parquet with full-year hourly zone loads

## Peak Definition

This script reads the annual weather-file run, not the DDY design-day runs.

Concretely, it filters EnergyPlus SQL outputs to:

- `EnvironmentType = 3`
- `Time.IntervalType = 1`

So all peak loads come from the annual EPW simulation at hourly resolution.

## Usage

Basic:

```bash
python3 scripts/extract_energyplus_sql_metrics.py --sql-dir /path/to/sql_results
```

Explicit outputs:

```bash
python3 scripts/extract_energyplus_sql_metrics.py \
  --sql-dir /path/to/sql_results \
  --sim-output /path/to/output/sim_metrics.csv \
  --zone-output /path/to/output/zone_peak_loads.csv \
  --errors-output /path/to/output/extract_energyplus_sql_metrics_errors.csv \
  --jobs 8
```

With optional hourly parquet export:

```bash
python3 scripts/extract_energyplus_sql_metrics.py \
  --sql-dir /path/to/sql_results \
  --sim-output /path/to/output/sim_metrics.csv \
  --zone-output /path/to/output/zone_peak_loads.csv \
  --hourly-zone-parquet /path/to/output/zone_hourly_loads_dataset \
  --jobs 8
```

With resumable outputs:

```bash
python3 scripts/extract_energyplus_sql_metrics.py \
  --sql-dir /path/to/sql_results \
  --sim-output /path/to/output/sim_metrics.csv \
  --zone-output /path/to/output/zone_peak_loads.csv \
  --hourly-zone-parquet /path/to/output/zone_hourly_loads_dataset \
  --resume continue
```

## Outputs

`sim_metrics.csv` includes:

- annual EUI and total energy
- total and conditioned floor area
- annual heating and cooling energy
- coincident building peak cooling and heating loads
- maximum zone peak cooling and heating loads
- timestamps of the annual-run coincident building peaks

`zone_peak_loads.csv` includes:

- `building_id`
- `zone_id`
- `zone_area_m2`
- `zone_peak_cooling_kw`
- `zone_peak_cooling_time`
- `zone_cooling_kw_at_building_peak`
- `building_total_peak_cooling_kw`
- `building_peak_cooling_time`
- `zone_peak_heating_kw`
- `zone_peak_heating_time`
- `zone_heating_kw_at_building_peak`
- `building_total_peak_heating_kw`
- `building_peak_heating_time`

If `--hourly-zone-parquet` is set, the parquet output is a directory of one parquet file per
building, which makes resume and partial reruns much safer. Each parquet file uses long format
with:

- `building_id`
- `zone_id`
- `month_day_hour`
- `cooling_kw`
- `heating_kw`

The optional parquet export requires `pyarrow` in the active Python environment.
`month_day_hour` is an integer code in `MMDDHH` format, for example `10101`
for January 1st at 01:00 and `70416` for July 4th at 16:00.

## Resume Behavior

- `--resume ask` is the default.
- If outputs already exist, the script asks whether to continue from the next model. Pressing Enter means `yes`.
- Answering `n` overwrites prior outputs and starts from the beginning.
- `--resume continue` skips the prompt and resumes automatically.
- `--resume overwrite` skips the prompt and clears prior outputs first.
- `--resume error` fails immediately if any output already exists.

Resume is based on the last common completed building found across:

- `sim_metrics.csv`
- `zone_peak_loads.csv`
- the hourly parquet dataset, if enabled

If later buildings exist in some outputs but not all of them, the script trims those partial tail
results before resuming so the outputs stay consistent.

## Performance Notes

- Default `--jobs` is capped at `8`.
- `--executor auto` prefers processes and falls back to threads if needed.
- When hourly parquet export is enabled, the script switches to threads for parallel mode to avoid sending very large hourly batches between processes.
- Hourly parquet stores `cooling_kw` and `heating_kw` as `float32` and uses compact `MMDDHH` integer time codes to reduce size and write overhead.
- Parallel workers are consumed as they finish, so one slow SQL file does not block all later completed results from being written.
- Writers are flushed in batches, default every `10` models, so the script does not need to hold the full dataset in memory before writing.
- This script is designed to be much more scalable than running separate extractors for each output.
