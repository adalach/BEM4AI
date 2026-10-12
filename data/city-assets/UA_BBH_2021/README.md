# Copernicus building heights per city (UA BBH 2021)

One GeoTIFF per BEM4AI city whose 2x2 km window lies in the coverage of the
Copernicus Urban Atlas Building Block Height 2021, named
`<city_key>_UA_BBH_2021_3035.tif`. Each holds the building heights in
metres (uint16) on the product's own 10 m grid in EPSG:3035; pixels without a
building, and pixels outside the city window, are nodata (65535).

## Source and licence

Contains modified European Union's Copernicus Land Monitoring Service information:
Urban Atlas Building Block Height 2021 (raster 10 m), Europe, 3-yearly,
<https://doi.org/10.2909/c5215520-98e9-4035-be56-d28156cd4dcd>. The data are
provided under the [Copernicus data policy](https://land.copernicus.eu/en/data-policy)
(full, open and free access). Their use here does not imply any endorsement by the
European Union.

## Modifications

- Cut to the 2x2 km window of one city, widened to whole pixels of the source grid,
  without resampling.
- Pixels that do not touch the window set to nodata (65535).
- A window spanning the files of several city areas mosaicked from exactly those
  files, which share one grid.

Every file carries this attribution and modification statement in its GeoTIFF tags
(`SOURCE`, `DATASET`, `DOI`, `MODIFICATION`, `LICENSE`), and its grid version in
`BEM4AI_CHIP_GRID`.

## How they are made

`notebooks/BEM4AI_1_Cities.ipynb`, section "Copernicus building heights", cuts them
from the files listed in `data/raw/CLMS/UA_BBH_2021/manifest.json`. A chip that is
present and current is reused, so the source files (a free Copernicus Data Space
account, `scripts/download_clms_bbh2021.py`) are needed only to cut chips for new
cities or a changed window.
