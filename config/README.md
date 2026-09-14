# Configuration

Configuration files hold stable source mappings and BEM assumptions shared by
more than one notebook. Runtime choices such as sample counts, overwrite flags,
simulation batch size, and geometry QA controls remain visible in the notebooks.

| File | Used by | Contents |
| --- | --- | --- |
| `indicator_attribute_mapping.csv` | Stage 1 | UCDB workbook sheets and attributes retained in the city table. |
| `geography.json` | Stages 1 and 2 | Building-stock aggregate-country rows, country-name aliases, and Geofabrik PBF locations. |
| `building_types.json` | Stages 2, 3, and 8; Stage 5 via the Stage 2 lookup table | Canonical building-type IDs, labels, sectors, source aliases, GHSL classes, OSM non-residential rules, fabric topics, and age classes. |
| `hb_material_rules.json` | Stage 2 | Rules that translate building-stock material labels into Honeybee opaque and glazing constructions. |
| `hb_mappings.json` | Stage 5 | GHSL age and climate mappings, Honeybee program assignment, the default floor-height bounds, window properties, and the residential high-rise floor threshold. |

The IDs in `building_types.json` are canonical and must remain aligned with the
`building_types` table written by Stage 2.
