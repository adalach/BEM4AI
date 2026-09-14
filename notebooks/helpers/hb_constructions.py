"""Create and reconstruct the Honeybee constructions used by BEM4AI.

Stage 2 sizes one Honeybee construction per archetype element - one wall, roof,
floor or window, for one building type, of one age class, in one country - so that
its U-value matches the value reported by the Hotmaps
building stock. Two mechanisms are used:

* **Opaque elements** (wall, floor, roof) start from a Honeybee Energy Standards
  construction. Every layer except the insulation is kept as published, and the
  R-value of a single insulation layer is solved so the assembly reaches the
  target U-value. The template is chosen per element by the notebook.
* **Windows** use ``EnergyWindowMaterialSimpleGlazSys``, whose U-factor is set
  directly, so the target is met exactly and only SHGC and VT are assumed.

The module also reconstructs the stored JSON into complete construction sets
when Stage 5 creates the Honeybee models. Template selection and SHGC/VT
assumptions remain visible in Stage 2 and ``config/hb_material_rules.json``.

Import from a notebook as::

    from helpers.hb_constructions import OpaqueStandards, build_construction_sets

"""

from __future__ import annotations

import copy
import json
from importlib import resources
from typing import Callable

import pandas as pd

from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.construction.window import WindowConstruction
from honeybee_energy.lib.constructionsets import construction_set_by_identifier
from honeybee_energy.lib.materials import opaque_material_by_identifier
from honeybee_energy.material.glazing import EnergyWindowMaterialSimpleGlazSys
from honeybee_energy.material.opaque import EnergyMaterialNoMass

from .utils import parse_json_like

__all__ = [
    "MATERIAL_KEYS",
    "HONEYBEE_JSON",
    "HONEYBEE_NUMERIC",
    "HONEYBEE_COLUMNS",
    "OpaqueStandards",
    "element_targets",
    "build_opaque",
    "build_windows",
    "apply_results",
    "build_construction_sets",
]

#: The natural key of one archetype element, shared by every table in the bundle.
MATERIAL_KEYS = ["country_id", "building_type_id", "age_class_id", "element_id"]

HONEYBEE_JSON = ["honeybee_material", "honeybee_construction"]
HONEYBEE_NUMERIC = ["honeybee_u_value", "u_diff"]
HONEYBEE_COLUMNS = HONEYBEE_JSON + HONEYBEE_NUMERIC

STANDARDS_PACKAGE = "honeybee_energy_standards.constructions"


class OpaqueStandards:
    """The Honeybee opaque standards library, plus the constructions generated from it.

    Materials and constructions created for an element are registered back into the
    library, so :meth:`summarize` can report the U-value actually achieved.
    """

    def __init__(self, package: str = STANDARDS_PACKAGE):
        self.package = package
        try:
            self.materials = self._read("opaque_material.json")
            self.constructions = self._read("opaque_construction.json")
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                f"{package} is not installed. The pinned version is part of the generation "
                "method; install it with: python -m pip install -r requirements.txt"
            ) from error

    def _read(self, filename: str) -> dict:
        with resources.files(self.package).joinpath(filename).open(encoding="utf-8") as stream:
            return json.load(stream)

    def layer_r_value(self, name: str) -> float | None:
        """R-value of one material: as published, or thickness / conductivity."""
        material = self.materials.get(name)
        if not isinstance(material, dict):
            return None
        if material.get("r_value") is not None:
            return float(material["r_value"])
        thickness, conductivity = material.get("thickness"), material.get("conductivity")
        if thickness is None or not conductivity:
            return None
        return float(thickness) / float(conductivity)

    def summarize(self, construction: str) -> dict:
        """Total R-value and U-value of one construction, layer by layer."""
        entry = self.constructions.get(construction)
        if entry is None:
            raise KeyError(f"Construction {construction!r} is not in the standards library.")
        layers = list(entry.get("materials", []))
        r_values = [self.layer_r_value(name) for name in layers]
        if any(r is None for r in r_values):
            unknown = [n for n, r in zip(layers, r_values) if r is None]
            raise ValueError(f"{construction!r} has layers without an R-value: {unknown}")
        r_total = sum(r_values)
        return {
            "construction": construction,
            "layers": layers,
            "r_values": r_values,
            "r_total": r_total,
            "u_value": (1.0 / r_total) if r_total else None,
        }

    def insulation_for(
        self,
        construction: str,
        u_target: float,
        *,
        insulation_mode: str = "replace",
        insulation_prefixes: tuple[str, ...] = ("Typical Insulation",),
        donor_insulation: str | None = None,
        insert_after: int | None = None,
        min_r_value: float = 1e-4,
    ) -> dict:
        """Solve the R-value of the one insulation layer that hits `u_target`.

        ``replace`` swaps the template's existing insulation layer; ``insert``
        adds one to a template that has none. When the fixed layers alone
        already exceed the target R-value, the insulation is clamped to
        `min_r_value` and the result is flagged, because a construction cannot
        be made to conduct more heat than its own structure does.
        """
        summary = self.summarize(construction)
        layers, r_values = summary["layers"], summary["r_values"]

        if insulation_mode == "replace":
            found = [i for i, name in enumerate(layers)
                     if any(str(name).startswith(prefix) for prefix in insulation_prefixes)]
            if len(found) != 1:
                raise ValueError(
                    f"Expected exactly one insulation layer in {construction!r}, found {len(found)}."
                )
            index = found[0]
            r_fixed = sum(r for i, r in enumerate(r_values) if i != index)
            donor = layers[index]
            position = None
        elif insulation_mode == "insert":
            donor = donor_insulation or insulation_prefixes[0]
            r_fixed = sum(r_values)
            index = None
            position = (len(layers) if insert_after is None else int(insert_after) + 1)
        else:
            raise ValueError(f"Unsupported insulation_mode: {insulation_mode!r}")

        if donor not in self.materials:
            raise KeyError(f"Donor insulation {donor!r} is not in the standards library.")

        r_target = 1.0 / float(u_target)
        r_insulation = r_target - r_fixed
        clamped = r_insulation <= min_r_value
        return {
            "construction": construction,
            "donor": donor,
            "index": index,
            "position": position,
            "r_fixed": r_fixed,
            "r_target": r_target,
            "r_insulation": float(min_r_value) if clamped else float(r_insulation),
            "clamped": clamped,
            "insulation_mode": insulation_mode,
        }

    def register(self, material: dict, construction: dict) -> None:
        """Add a generated material and construction to the library."""
        self.materials[material["identifier"]] = material
        self.constructions[construction["identifier"]] = construction

    def check_templates(self, template_library: dict) -> None:
        """Fail early if the configured templates or donors are not in the library."""
        for element_code, families in template_library.items():
            for family, config in families.items():
                where = f"{element_code}/{family}"
                if config["construction"] not in self.constructions:
                    raise KeyError(f"Template {config['construction']!r} for {where} is not in the library.")
                donor = config.get("donor_insulation")
                if donor and donor not in self.materials:
                    raise KeyError(f"Donor insulation {donor!r} for {where} is not in the library.")


def element_targets(materials: pd.DataFrame, dimensions: dict, element_code: str) -> pd.DataFrame:
    """Archetype elements of one element type that carry a target U-value, with their labels.

    `dimensions` holds the `countries`, `building_types`, `age_classes` and
    `elements` tables. The composition dictionaries are carried along, because
    template selection and window classification read them.
    """
    elements = dimensions["elements"]

    element_id = elements.loc[elements["code"] == element_code, "id"]
    if element_id.empty:
        raise KeyError(
            f"Unknown element code {element_code!r}. Known: {', '.join(elements['code'])}"
        )

    rows = materials.loc[
        materials["element_id"].isin(element_id) & materials["u_value"].notna(),
        [*MATERIAL_KEYS, "u_value", "materials", "methodologies"],
    ]

    labelled = (
        rows.merge(dimensions["countries"].rename(columns={"id": "country_id", "name": "country"}),
                   on="country_id", how="left", validate="many_to_one")
            .merge(dimensions["building_types"].rename(columns={
                       "id": "building_type_id", "code": "building_code", "label": "building_label"}),
                   on="building_type_id", how="left", validate="many_to_one")
            .merge(dimensions["age_classes"].rename(columns={"id": "age_class_id", "label": "age_label"})[
                       ["age_class_id", "age_label"]],
                   on="age_class_id", how="left", validate="many_to_one")
            .merge(elements.rename(columns={"id": "element_id", "code": "element_code"})[
                       ["element_id", "element_code"]],
                   on="element_id", how="left", validate="many_to_one")
            .rename(columns={"u_value": "u_target"})
    )

    ordered = [
        "country_id", "country",
        "building_type_id", "building_code", "building_label", "sector",
        "age_class_id", "age_label",
        "element_id", "element_code", "u_target",
        "materials", "methodologies",
    ]
    return (
        labelled[ordered]
        .sort_values(["country", "building_code", "age_label"])
        .reset_index(drop=True)
    )


def build_opaque(
    targets: pd.DataFrame,
    standards: OpaqueStandards,
    select_template: Callable[[pd.Series], dict],
    *,
    label: str,
    min_r_value: float = 1e-4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One Honeybee construction per archetype element, insulation sized to `u_target`.

    `select_template` is called with each row and returns the chosen entry of the
    template library, extended with a `family` key. Returns the generated rows
    and the rows that could not be generated.
    """
    generated, failures = [], []

    for row in targets.itertuples(index=False):
        keys = {key: int(getattr(row, key)) for key in MATERIAL_KEYS}
        u_target = float(row.u_target)
        name = f"{row.country} {row.building_code} {row.age_label}"
        construction_id = f"BS {label} {name}"
        insulation_id = f"BS {label} Insulation {name}"

        try:
            template = select_template(pd.Series(row._asdict()))
            solved = standards.insulation_for(
                template["construction"],
                u_target,
                insulation_mode=template.get("insulation_mode", "replace"),
                donor_insulation=template.get("donor_insulation"),
                insert_after=template.get("insert_after"),
                min_r_value=min_r_value,
            )

            insulation = copy.deepcopy(standards.materials[solved["donor"]])
            insulation["identifier"] = insulation_id
            insulation["r_value"] = solved["r_insulation"]

            construction = copy.deepcopy(standards.constructions[solved["construction"]])
            construction["identifier"] = construction_id
            layers = list(construction.get("materials", []))
            if solved["insulation_mode"] == "replace":
                layers[solved["index"]] = insulation_id
            else:
                layers.insert(solved["position"], insulation_id)
            construction["materials"] = layers

            standards.register(insulation, construction)

            generated.append({
                **keys,
                "honeybee_material": insulation,
                "honeybee_construction": construction,
                "u_target": u_target,
                "honeybee_u_value": standards.summarize(construction_id)["u_value"],
                "clamped": solved["clamped"],
                "template_family": template["family"],
                "template": template["construction"],
            })
        except Exception as error:
            failures.append({**keys, "u_target": u_target, "error": str(error)})

    generated_df = pd.DataFrame(generated)
    failures_df = pd.DataFrame(failures)

    if not generated_df.empty:
        generated_df["u_diff"] = (generated_df["honeybee_u_value"] - generated_df["u_target"]).abs()
        families = generated_df["template_family"].value_counts().to_dict()
        print(f"[{label}] {len(generated_df)} constructions | "
              f"max |dU| {generated_df['u_diff'].max():.3f} W/m2K | "
              f"clamped {int(generated_df['clamped'].sum())} | templates {families}")
    if not failures_df.empty:
        print(f"[{label}] {len(failures_df)} failed: "
              f"{failures_df['error'].value_counts().head(3).to_dict()}")

    return generated_df, failures_df


def build_windows(
    targets: pd.DataFrame,
    optics: Callable[[pd.Series], dict],
    *,
    label: str = "Window",
) -> pd.DataFrame:
    """One simple glazing system per archetype element, with its U-factor set to `u_target`.

    `optics` is called with each row and returns the `shgc`, `vt` and the
    `name` fragment describing the frame and glazing. Because the U-factor is an
    input to ``EnergyWindowMaterialSimpleGlazSys``, the target is met exactly and
    `u_diff` is zero by construction.
    """
    generated = []

    for row in targets.itertuples(index=False):
        keys = {key: int(getattr(row, key)) for key in MATERIAL_KEYS}
        u_target = float(row.u_target)
        chosen = optics(pd.Series(row._asdict()))
        name = f"{chosen['name']} {row.country} {row.building_code} {row.age_label}"
        glazing_id = f"BS Glazing {name}"

        glazing = {
            "type": "EnergyWindowMaterialSimpleGlazSys",
            "identifier": glazing_id,
            "u_factor": u_target,
            "shgc": float(chosen["shgc"]),
            "vt": float(chosen["vt"]),
        }
        construction = {
            "type": "WindowConstructionAbridged",
            "identifier": f"BS Window {name}",
            "materials": [glazing_id],
        }

        generated.append({
            **keys,
            "honeybee_material": glazing,
            "honeybee_construction": construction,
            "u_target": u_target,
            "honeybee_u_value": u_target,
            "u_diff": 0.0,
            "glazing_type": chosen.get("glazing_type"),
            "frame_material": chosen.get("frame_material"),
        })

    generated_df = pd.DataFrame(generated)
    print(f"[{label}] {len(generated_df)} glazing systems, U-factor set to the target")
    return generated_df


def apply_results(materials: pd.DataFrame, generated: pd.DataFrame, label: str) -> pd.DataFrame:
    """Write the Honeybee columns of `generated` into `materials`, keyed on the element.

    Returns a new materials table; rows that `generated` does not cover keep what
    they had.
    """
    if generated.empty:
        print(f"[{label}] nothing to write back.")
        return materials

    missing = [c for c in (*MATERIAL_KEYS, *HONEYBEE_COLUMNS) if c not in generated.columns]
    if missing:
        raise KeyError(f"[{label}] generated frame is missing columns: {missing}")

    payload = generated[[*MATERIAL_KEYS, *HONEYBEE_COLUMNS]].drop_duplicates(
        subset=MATERIAL_KEYS, keep="last"
    )
    unknown = payload.set_index(MATERIAL_KEYS).index.difference(materials.set_index(MATERIAL_KEYS).index)
    if len(unknown):
        raise KeyError(
            f"[{label}] {len(unknown)} generated rows are not in the materials table, "
            f"e.g. {list(unknown[:3])}"
        )

    columns = list(materials.columns)
    merged = materials.merge(payload, on=MATERIAL_KEYS, how="left", suffixes=("", "_new"))
    for column in HONEYBEE_COLUMNS:
        incoming = merged[f"{column}_new"]
        merged[column] = incoming.where(incoming.notna(), merged[column])

    print(f"[{label}] wrote {len(payload)} rows into the materials table.")
    return merged[columns]


def _opaque_construction(
    construction_value,
    material_value,
    element: str,
    sample_id: str,
) -> OpaqueConstruction:
    """Rebuild one generated opaque construction from its stored JSON fields."""
    construction_json = parse_json_like(construction_value)
    material_json = parse_json_like(material_value)
    if not isinstance(construction_json, dict) or not isinstance(material_json, dict):
        raise ValueError(f"invalid {element} construction or material JSON")

    insulation = EnergyMaterialNoMass.from_dict(material_json)
    material_ids = list(construction_json.get("materials") or [])
    if insulation.identifier not in material_ids:
        raise ValueError(
            f"{element} construction does not reference generated material "
            f"{insulation.identifier!r}"
        )

    layers = []
    for material_id in material_ids:
        if material_id == insulation.identifier:
            layers.append(insulation)
        else:
            try:
                layers.append(opaque_material_by_identifier(material_id))
            except Exception as error:
                raise ValueError(
                    f"unknown standard {element} material {material_id!r}"
                ) from error
    if not layers:
        raise ValueError(f"{element} construction has no material layers")
    return OpaqueConstruction(
        identifier=construction_json.get("identifier", f"{sample_id}_{element}"),
        materials=layers,
    )


def _window_construction(
    construction_value,
    material_value,
    sample_id: str,
) -> WindowConstruction:
    """Rebuild one generated window construction from its stored JSON fields."""
    construction_json = parse_json_like(construction_value)
    material_json = parse_json_like(material_value)
    if not isinstance(construction_json, dict) or not isinstance(material_json, dict):
        raise ValueError("invalid window construction or material JSON")
    glazing = EnergyWindowMaterialSimpleGlazSys.from_dict(material_json)
    return WindowConstruction(
        identifier=construction_json.get("identifier", f"{sample_id}_window"),
        materials=[glazing],
    )


def build_construction_sets(
    buildings: pd.DataFrame,
    excluded: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Rebuild one complete custom Honeybee construction set per building."""
    result = buildings.copy()
    construction_sets = {}
    serialized = []
    errors = {}

    for _, row in result.iterrows():
        sample_id = str(row["sample_id"])
        try:
            base_identifier = row["hb_construction_set_id"]
            construction_set = construction_set_by_identifier(base_identifier).duplicate()
            construction_set.identifier = f"cs_{sample_id.rsplit('_', 1)[-1]}"
            construction_set.wall_set.exterior_construction = _opaque_construction(
                row["wall_hb_construction"], row["wall_hb_material"], "wall", sample_id
            )
            construction_set.roof_ceiling_set.exterior_construction = _opaque_construction(
                row["roof_hb_construction"], row["roof_hb_material"], "roof", sample_id
            )
            construction_set.floor_set.ground_construction = _opaque_construction(
                row["floor_hb_construction"], row["floor_hb_material"], "floor", sample_id
            )
            window = _window_construction(
                row["window_hb_construction"], row["window_hb_material"], sample_id
            )
            construction_set.aperture_set.window_construction = window
            construction_set.aperture_set.skylight_construction = window
            construction_set.aperture_set.operable_construction = window
            construction_sets[sample_id] = construction_set
            try:
                construction_dict = construction_set.to_dict(abridged=True)
            except TypeError:
                construction_dict = construction_set.to_dict()
            serialized.append(json.dumps(construction_dict, ensure_ascii=False))
        except Exception as error:
            errors[sample_id] = str(error)
            serialized.append(None)

    result["hb_construction_set_djson"] = serialized
    if errors:
        failed_ids = set(errors)
        failed = result[result["sample_id"].astype(str).isin(failed_ids)].copy()
        failed["bem_exclusion_reasons"] = [
            [*reasons, "invalid_honeybee_construction"]
            for reasons in failed["bem_exclusion_reasons"]
        ]
        failed["construction_error"] = failed["sample_id"].astype(str).map(errors)
        excluded = pd.concat([excluded, failed], ignore_index=True)
        result = result[~result["sample_id"].astype(str).isin(failed_ids)].copy()

    if result.empty:
        raise RuntimeError("No buildings remain after construction-set validation.")
    return result, excluded, construction_sets
