# Uncropped chips of the example cities

The chips of Lisbon, Luxembourg, Munich and Oulu for every raster layer, cut to the
city window and widened to whole source pixels, but not cropped to the window
(`BEM4AI_CHIP_GRID` = `source-aligned-uncropped-2`). The chip alignment check at the
end of `notebooks/BEM4AI_1_Cities.ipynb` draws them next to the saved, cropped chips;
they are not used by any later stage.

## Sources and licences

- `*_GHS_BUILT_C_moll_uncropped.tif`, `*_GHS_BUILT_H_moll_uncropped.tif`,
  `*_GHS_AGE_moll_uncropped.tif`: cut from the Global Human Settlement Layer
  (European Commission, Joint Research Centre) products GHS-BUILT-C R2023A,
  GHS-BUILT-H R2023A and GHS-AGE R2025A, CC BY 4.0, on their own Mollweide grids
  (ESRI:54009) without resampling.
- `*_UA_BBH_2021_3035_uncropped.tif`: contains modified European Union's
  Copernicus Land Monitoring Service information, Urban Atlas Building Block Height
  2021, <https://doi.org/10.2909/c5215520-98e9-4035-be56-d28156cd4dcd>, under the
  [Copernicus data policy](https://land.copernicus.eu/en/data-policy); cut to the
  window on its own 10 m grid (EPSG:3035) without resampling. No endorsement by the
  European Union is implied. The attribution is also in each file's GeoTIFF tags.
