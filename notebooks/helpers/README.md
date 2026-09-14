# Notebook helpers

The notebooks contain the visible workflow, user settings, previews, and
outputs. This directory contains reusable operations that would otherwise make
the notebooks difficult to read or repeat the same implementation across
stages.

## Modules by stage

Rows are ordered by the first notebook stage that imports each module.

| Module | Called by | Responsibility |
| --- | --- | --- |
| `config.py` | Stages 1-3 | Loads JSON configuration files and reports missing or malformed configuration clearly. |
| `ghsl_chips.py` | Stage 1 | Clips the global GHSL rasters to city extents, validates cached chips, and draws chip previews. |
| `paths.py` | Stages 1-8 | Defines repository, data, interim, notebook, and script paths independently of the current working directory. It also makes both `helpers` and `scripts` importable from notebooks. |
| `utils.py` | Stages 1-6 | Provides shared naming, validation, tabular, coordinate, and geometry operations used across several stages. |
| `hb_constructions.py` | Stages 2 and 5 | Generates Honeybee material and construction JSON from the building-stock data, then reconstructs custom construction sets for each Stage 5 model. |
| `building_sampling.py` | Stage 3 | Allocates samples by population, loads and filters OSM footprints, samples GHSL rasters, assigns `sample_id`, and collects neighbors. |
| `GHSL_gridplot.py` | Stages 3-5 | Draws comparable footprint, transformed-geometry, and Honeybee-input previews. |
| `review_app.py` | Stages 3-5 | Connects notebook launch cells to the browser servers under `scripts/review_apps/`, keeps one server per app, and returns the local browser link. |
| `review_selections.py` | Stages 3-5 | Applies saved `sample_id` exclusions consistently to each stage's tables and generated files while preserving removed records. |
| `local_frames.py` | Stage 4 | Creates per-city metric coordinate frames, aligns neighboring walls, removes residual overlaps, and synchronizes local and WGS84 geometry. |
| `pickle_compat.py` | Stages 5 and 6 | Reads project pickles containing path or NumPy classes written on another supported platform or dependency layout. |
| `shading_and_windows.py` | Stage 4 | Selects neighboring shading edges and exterior wall segments that can host windows. |
| `zone_connectivity.py` | Stage 4 | Detects disconnected Zone groups and removes buildings whose Zone decomposition is not connected. |
| `zone_quality.py` | Stage 4 | Scores Zone geometry and applies the whole-building fallback when a decomposition is unusable. |
| `bem_inputs.py` | Stage 5 | Loads Stage 4 and Stage 2 artifacts, attaches city, age, climate, and material mappings, filters incomplete inputs, and estimates floor counts. |
| `bem_models.py` | Stage 5 | Extrudes Zones into Honeybee Rooms, solves adjacency, assigns programs and ideal HVAC, and adds windows and neighboring shades. |
| `bem_outputs.py` | Stage 5 | Loads the preview renderers, draws compact model grids, validates HBJSON exports, removes models outside the publication limits, and writes the model indexes. |

## Package boundary

`__init__.py` intentionally contains no imports. Its presence makes `helpers`
an explicit Python package, which keeps imports such as
`from helpers import bem_inputs` and relative imports such as
`from .utils import ...` reliable in Jupyter, tests, and IDEs. Importing the
package itself therefore does not pull in GeoPandas, Shapely, or Honeybee.

The helper boundary is notebook-facing. Code used only by executable scripts
stays under `scripts/`. For example, `scripts/hbjson_paths.py` is shared by the
Stage 7 converter and the BEM review server, whereas `helpers/paths.py` defines
the project directories used directly by every notebook.

## Review applications

The browser applications live under `scripts/review_apps/`. They own the HTTP
server, page rendering, pagination, and selection persistence.
`review_app.py` is only the notebook connector: it imports the requested server,
starts it in a background thread, stops an earlier instance when the launch
cell is rerun, and returns a clickable link. The following notebook cell uses
`review_selections.py` to apply the saved choices to the pipeline artifacts.

## Shared geometry and plotting

`GHSL_gridplot.py` provides the two-dimensional sample previews used before
model export. The Honeybee renderers under `scripts/jupyter-honeybee-plot/`
remain separate because one builds an interactive Plotly figure and the other
builds a static PyVista/Pillow grid. Their shared Honeybee traversal, surface
classification, polygon cleanup, and triangulation live beside them in
`scripts/jupyter-honeybee-plot/honeybee_plot_geometry.py`.
