# Convert HBJSON models to IDF and EPJSON

`convert_hbjson_exports_to_idf_epjson.py` converts the validated and reviewed
Honeybee models from Stage 5 into the two EnergyPlus model formats published
with the dataset. Stage 7 calls the same code before simulation, so the IDF,
EPJSON, and SQL files use one set of simulation settings and output requests.

## When it runs

Run this conversion only after both prerequisite stages are complete:

1. Stage 5 has written the reviewed models to `output/hbjsons/` and their index
   to `data/interim/hbjsons_df.pkl`.
2. Stage 6 has written `data/interim/city_weather_assets.pkl` and the assigned
   EPW and DDY files under `output/weather/`.

The normal entry point is the conversion cell in
`notebooks/BEM4AI_7_Simulation.ipynb`. The script can also be run directly from
the repository root.

## Inputs and outputs

| Path | Role |
| --- | --- |
| `data/interim/hbjsons_df.pkl` | Stage 5 model index, joined by `sample_id` and `city_key`. |
| `output/hbjsons/<sample_id>.hbjson` | Reviewed Honeybee model to translate. |
| `data/interim/city_weather_assets.pkl` | Stage 6 weather assignment for each city. |
| `output/weather/epw/` | EPW files used for the model location and annual simulation. |
| `output/weather/ddy/` | DDY files used to add heating and cooling design days. |
| `output/idf/<sample_id>.idf` | Final EnergyPlus IDF model. |
| `output/epjson/<sample_id>.epjson` | Final EnergyPlus JSON model. |
| `data/interim/hbjson_to_idf_epjson.csv` | Per-model conversion, validation, and structure summary. |

When the Stage 5 index is missing, the script can reconstruct a minimal index
from the HBJSON filenames. The normal publication workflow should use the index
because it carries the authoritative `city_key` needed for weather assignment.

## Conversion workflow

For each model, the converter performs the following operations:

1. Resolve the HBJSON, EPW, and DDY paths. The HBJSON resolver accepts the
   portable Stage 5 filename and also supports older index rows containing an
   absolute path.
2. Build a Honeybee `SimulationParameter`. It enables annual and sizing-period
   runs, SQLite output, hourly reporting, and the design days selected from the
   assigned DDY file.
3. Translate HBJSON through the Honeybee/OpenStudio measures-only workflow.
   This expands the ideal-loads templates into the zone-level EnergyPlus HVAC
   objects required by the dataset.
4. Replace `Site:Location` in the translated IDF with the location from the
   assigned EPW file.
5. Replace the IDF reporting objects with the BEM4AI output contract described
   below.
6. Inspect the IDF structure and reject a non-canonical HVAC representation.
7. Run EnergyPlus with `--convert-only` to validate the IDF and create EPJSON.
8. Inspect the EPJSON structure, then convert it back once with EnergyPlus to
   verify that the published EPJSON is readable.
9. Record the final status, elapsed time, file paths, format versions, and
   structural object counts in the summary CSV.

OpenStudio is used for semantic HBJSON translation. EnergyPlus
`--convert-only` is used for format conversion and validation. Neither step
runs the annual energy simulation.

## EnergyPlus output contract

EnergyPlus writes time-series values to SQL only for variables requested by
`Output:Variable` objects in the IDF. Enabling SQLite output by itself does not
request the surface, window, Zone, weather, and ventilation variables needed
by the result tables and graph datasets.

`apply_dataset_output_requests_to_idf` therefore replaces any existing
reporting block with a deterministic one containing:

- `Output:SQLite = SimpleAndTabular`
- `Output:Table:SummaryReports = AllSummary`
- an IDF variable dictionary
- the dataset reporting tolerances
- 33 variables requested at both annual and hourly frequency
- two enclosure solar variables requested at hourly frequency

The replacement is intentional and idempotent. Running it more than once does
not duplicate output objects. The variable names are declared near the top of
the converter as `DATASET_COMMON_OUTPUT_VARIABLES` and
`DATASET_HOURLY_ONLY_OUTPUT_VARIABLES` so the SQL contract can be reviewed
without reading the conversion branches. This is a dataset requirement, not a
compatibility step for an earlier operating system or an individual model.

## Structural validation

The publication models use zone-based ideal-loads HVAC. By default, both IDF
and EPJSON must contain:

- `ZoneHVAC:IdealLoadsAirSystem`
- `ZoneHVAC:EquipmentConnections`

They must not contain:

- `HVACTemplate:Zone:IdealLoadsAirSystem`
- `SpaceHVAC:EquipmentConnections`

`Space` and `SpaceList` objects are recorded in the summary but are not a pass
or fail condition. They are optional metadata and do not define how HVAC is
attached. `--allow-template-idf` disables this structural guard for diagnostic
runs only; it should not be used for the publication export.

## Existing and partial outputs

Without `--overwrite`, the converter handles existing files as follows:

| Existing files | Action |
| --- | --- |
| IDF and EPJSON | Inspect their structure and keep them. With `--validate-existing`, also validate both through EnergyPlus. |
| IDF only | Apply the dataset output contract, validate it, and recreate EPJSON. |
| EPJSON only | Recreate IDF, apply the dataset output contract, and regenerate EPJSON from the normalized IDF. |
| Neither, or a structurally invalid pair | Translate the HBJSON again and create a fresh pair. |

Temporary OpenStudio workflow folders are removed after a successful attempt.
Use `--keep-temp` when diagnosing a failed translation.

## Usage

Convert every indexed model:

```bash
python scripts/convert_hbjson_exports_to_idf_epjson.py
```

Convert a small test batch:

```bash
python scripts/convert_hbjson_exports_to_idf_epjson.py --limit 5
```

Replace existing outputs and use four worker processes:

```bash
python scripts/convert_hbjson_exports_to_idf_epjson.py --overwrite --workers 4
```

Validate complete existing pairs without recreating them:

```bash
python scripts/convert_hbjson_exports_to_idf_epjson.py --validate-existing
```

If EnergyPlus cannot be found automatically, pass its executable explicitly or
set `ENERGYPLUS_EXE`:

```bash
python scripts/convert_hbjson_exports_to_idf_epjson.py \
  --energyplus-exe /path/to/energyplus
```

The repository pins `honeybee-energy[openstudio]` in `requirements.txt`. This
installs the OpenStudio 3.10 Python bindings used to generate EnergyPlus 25.1
models. Keep this dependency aligned with the EnergyPlus 25.1 executable used
by Stage 7. The converter checks the generated IDF version against the selected
EnergyPlus executable before validating the model and reports a clear version
mismatch if the two toolchains differ.

The OpenStudio CLI must also be available as `openstudio` on `PATH`.

## Summary statuses

The summary CSV always contains one row per selected model. Important statuses
include:

| Status | Meaning |
| --- | --- |
| `converted` | A new IDF/EPJSON pair passed structural and format validation. |
| `skipped_existing` | A complete, structurally valid pair already existed. |
| `validated_existing` | An existing pair also passed EnergyPlus validation. |
| `restored_missing_idf` | IDF was recreated from EPJSON and both files were normalized. |
| `restored_missing_epjson` | EPJSON was recreated from IDF. |
| `missing_hbjson`, `missing_epw`, `missing_ddy` | A required input was absent. |
| `idf_structure_failed`, `epjson_structure_failed` | The zone-based HVAC contract was not met. |
| `idf_validation_failed`, `epjson_validation_failed` | EnergyPlus could not read one format. |
| `conversion_failed` | Translation raised another runtime error; details are in `message`. |

The structural count columns make failures inspectable without opening every
model. The `*_has_expanded_space_hvac` columns are retained as aliases for the
current `*_has_zone_based_hvac` result so older local summaries remain readable.
