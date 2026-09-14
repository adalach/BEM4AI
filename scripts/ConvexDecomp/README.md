# ConvexDecomp

Convex decomposition of OSM building footprints, used by
`notebooks/BEM4AI_4_ModelTransformations.ipynb` to turn a cleaned footprint
into the convex zones an EnergyPlus model is built from.

Derived from **ConvexDecomp**, which accompanies *Workflow-Oriented Convex
Decomposition of 2D Building Geometry* (Dalach, Chen and Borrmann, 33rd
International Workshop on Intelligent Computing in Engineering, 2026,
<https://mediatum.ub.tum.de/node?id=1855368>). The complete source project is
available at <https://github.com/adalach/ConvexDecomp>. This is a reduced and
extended copy, not a snapshot: the ResPlan floorplan adapters and the paper's
figure and diagnostic code are not here, the library's own duplicates of
geometry helpers are replaced by imports from `notebooks/helpers/utils.py`, and
two things this pipeline needs were added - alignment of a neighbour's edges to
a reference building, and a runner that labels every zone with the part of the
building it came from, so that perimeter and core zones can be told apart
downstream. The code included here is the reproducible source of truth for the
BEM4AI pipeline rather than an implicit dependency on an upstream revision.

## Modules

Read from the bottom up: the first three are geometry, the rest are the
footprint workflow built on them.

| Module | Role |
| --- | --- |
| `polygon_convexity.py` | Whether a ring turns consistently, and where it does not. The reflex vertices everything else works from. |
| `convex_decomposition.py` | The bounded search itself: cut a polygon at a reflex vertex, recurse, score the variants, keep the best. Dataset-agnostic. |
| `hole_splitter.py` | Cuts a polygon with holes into holeless pieces, since the search assumes simple rings. |
| `osm_footprint_normalizer.py` | Removes the surveying noise an OSM footprint carries: duplicate vertices, centimetre edges, almost-collinear runs. Also flags footprints too long and thin to be buildings. |
| `polygon_edge_alignment.py` | Squares up walls that are nearly parallel or nearly colinear, within guards that stop the building from being reshaped. Can align a footprint to a reference building's axes instead of its own. |
| `osm_perimeter_builder.py` | Offsets a footprint inward to separate the daylit perimeter band from the core, and decides when a footprint has no core at all. |
| `osm_perimeter_subdivider.py` | Carves the obvious trapezoids and corner triangles out of a perimeter band before the general search sees it. |
| `osm_convex_decomposer.py` | The footprint-facing entry point to the search, with the adaptive depth and width schedule and the per-polygon statistics. |
| `decomposition_runner.py` | Runs the whole thing over a GeoDataFrame of buildings: works out what each building is decomposed from, decomposes it, and collects the diagnostics. |
