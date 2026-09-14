# Extract Weather Files

`extract_weather.py` turns downloaded [Climate.OneBuilding](https://climate.onebuilding.org/)
ZIP archives into the per-city EPW and DDY files that `BEM4AI_7_Simulation.ipynb` expects.

BEM4AI does **not** redistribute weather files, and this script does **not** download them.
Obtaining the archives is a manual step, exactly as in the published dataset. The script
only validates what you downloaded and renames it.

## Workflow

1. Run `notebooks/BEM4AI_6_Weather.ipynb` up to *Write the download list*. It matches each
   sampled city to its nearest station and writes
   `data/interim/weather_download_links.csv` with columns `city_key` and `download_link`.
2. Download every distinct `download_link` into `output/weather/downloads/`, keeping the
   original ZIP filenames. Do not unpack or rename them.
3. Run this script, or the remaining notebook cells, which call it for you.

## Usage

```bash
python3 scripts/extract_weather.py --check-only   # validate the downloads, write nothing
python3 scripts/extract_weather.py                # extract
python3 scripts/extract_weather.py --force        # replace a complete existing extraction
```

Defaults resolve relative to the repository root:

| Option | Default |
| --- | --- |
| `--links` | `data/interim/weather_download_links.csv` |
| `--downloads-dir` | `output/weather/downloads` |
| `--epw-dir` | `output/weather/epw` |
| `--ddy-dir` | `output/weather/ddy` |

## What it checks

Before writing anything, every required archive must be present, open as a ZIP with no CRC
error, and contain exactly one identifiable EPW and DDY. The EPW must start with `LOCATION,`
and the DDY must contain a `SizingPeriod:DesignDay` object. A city whose archive fails any of
these is an error, not a warning, so a missing weather file surfaces here rather than as a
failed EnergyPlus run hours later.

Outputs are written through a temporary name and renamed, so an interrupted run leaves no
half-written file. Existing outputs are never overwritten without `--force`. Where several
cities share a station, the same file is written under each `city_key`.

Standard library only, Python 3.9 or newer, on Windows, macOS and Linux.

## Relationship to the published dataset

This is the pipeline-side copy of `weather/extract_weather.py` shipped inside the published
BEM4AI dataset. The logic is identical; only the default paths differ. Keep the two in sync.

Climate.OneBuilding refreshes station archives periodically, so a download made today can
differ from the files used for the published simulations. The layout and filenames are
reproducible; the exact historical file versions are not.
