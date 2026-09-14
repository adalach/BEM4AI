# Honeybee model previews

This directory contains the renderer subset used by
`notebooks/BEM4AI_5_BEMModels.ipynb`. The complete project, usage examples,
and current source are available in the
[jupyter-honeybee-plot repository](https://github.com/adalach/jupyter-honeybee-plot).

| File | Role |
| --- | --- |
| `plot_honeybee_model.py` | Builds an interactive Plotly figure for one Honeybee model. |
| `plot_honeybee_models.py` | Renders several Honeybee models with PyVista and combines them into one static Pillow image. |
| `honeybee_plot_geometry.py` | Traverses Honeybee models once and supplies both renderers with classified triangle and wireframe geometry. |

The two renderer files remain separate because their figure and layout code is
backend-specific. Model traversal, polygon cleanup, projection, triangulation,
and Honeybee surface classification are shared through
`honeybee_plot_geometry.py`, so the bundled renderer does not depend on
notebook-private modules or maintain two geometry implementations.

Stage 5 loads both functions with `helpers.bem_outputs.load_plotters`. They are
not installed as a Python package and do not download example models or other
files.

## Dependencies

The interactive renderer requires Plotly. The static renderer requires
PyVista, Pillow, and a working off-screen VTK renderer. These dependencies are
included in the repository `requirements.txt`.

Both functions accept in-memory `honeybee.model.Model` objects. Stage 5 shows
three models in each static preview and one model in the interactive preview.
