# Scripts

The notebooks are the main BEM4AI workflow. This directory contains executable
utilities, browser applications, and documented bundled components that are
used by more than one notebook cell or can be run independently. Small
stage-specific operations remain in `notebooks/helpers/`.

## Called by notebooks

| File | Called from | Role |
| --- | --- | --- |
| `review_apps/serve_footprint_selector.py` | Stage 3 | Serves the sampled-footprint review and records rejected `sample_id` values. |
| `ConvexDecomp/` | Stage 4 | Cuts cleaned OSM footprints into the convex zones an EnergyPlus model is built from: normalization, edge alignment, perimeter and core, and the bounded convex-decomposition search. See `ConvexDecomp/README.md`. |
| `review_apps/serve_building_selector.py` | Stage 4 | Serves the transformed-building review and records rejected `sample_id` values. |
| `jupyter-honeybee-plot/plot_honeybee_model.py` | Stage 5 | Creates an interactive preview of one Honeybee model. |
| `jupyter-honeybee-plot/plot_honeybee_models.py` | Stage 5 | Creates the three-model static previews. Both renderers use the shared Honeybee traversal and geometry kernel bundled in the same directory. |
| `review_apps/serve_bem_selector.py` | Stage 5 | Serves the generated-BEM review and records rejected `sample_id` values. |
| `extract_weather.py` | Stage 6 | Validates downloaded Climate.OneBuilding archives and extracts one EPW/DDY pair per sampled city. |
| `convert_hbjson_exports_to_idf_epjson.py` | Stage 7 | Converts reviewed HBJSON models to IDF and EPJSON after weather assignment and provides the shared simulation-parameter and EnergyPlus-output contract. See `README_convert_hbjson_exports_to_idf_epjson.md`. |

The browser files used by the selector servers are kept under `review_apps/`
because they are not standalone user entry points. See
`review_apps/README.md` for the review workflow. Shared HTTP and selection-file
handling for the three servers is implemented in `review_apps/server_common.py`.
Their two-dimensional renderers share `review_apps/svg_geometry.py`.
The notebooks apply saved selections through
`notebooks/helpers/review_selections.py`.

`hbjson_paths.py` provides the single HBJSON path-resolution rule used by the
Stage 7 converter and the generated-model review app. It remains under
`scripts/` because both consumers are scripts; `notebooks/helpers/paths.py`
instead owns the repository directory constants imported directly by notebooks.

## Run after simulation

| File | Role |
| --- | --- |
| `extract_energyplus_sql_metrics.py` | Builds publication-level building summaries, Zone peak-load tables, and optional hourly Zone-load Parquet files from EnergyPlus SQL outputs. Stage 8 also reads SQL but produces a different exploratory results table. |

No script in this public directory is an alternative to running the notebook
pipeline. Detailed usage for the extraction tools is documented in their
matching `README_*.md` files.
