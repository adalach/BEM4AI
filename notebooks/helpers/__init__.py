"""Shared helpers for the BEM4AI notebooks.

Kept import-free on purpose: `helpers.paths` must be importable without pulling
in geopandas, shapely or honeybee, because it is the first thing every notebook
imports. Import submodules explicitly, e.g. `from helpers.utils import ...`.
"""
