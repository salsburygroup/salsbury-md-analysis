"""Topology-derived standard chemistry definitions for self-service analysis.

The routines in this module infer broadly useful, reviewable observables from
atom identity and one reference structure.  They do not infer biological
importance.  Every generated definition carries an inference record so a
project can override unusual residue, ligand, symmetry, or ion chemistry.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

from .atom_mapping import AtomMappingError, AtomRecord, read_pdb_atoms
from .chemical_identity import (
    ANION_ELEMENTS,
    CATION_ELEMENTS,
    ION_RESIDUES,
    NUCLEIC_RESIDUES,
    PROTEIN_RESIDUES,
    WATER_RESIDUES,
)
from .coordinates import CoordinateReadError, iter_coordinate_frames


class AutomaticChemistryError(ValueError):
    """Raised when topology-derived standard chemistry is ambiguous or invalid."""


_PURINE_ORDER = ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4")
_PYRIMIDINE_ORDER = ("N1", "C2", "N3", "C4", "C5", "C6")


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((float(right) - float(left)) ** 2 for left, right in zip(first, second)))


def _residue_groups(atoms: Sequence[AtomRecord]) -> list[tuple[tuple[object, ...], list[AtomRecord]]]:
    groups: Dict[tuple[object, ...], list[AtomRecord]] = {}
    for atom in atoms:
        key = (
            atom.chain_id, atom.residue_number, atom.insertion_code,
            atom.residue_name,
        )
        groups.setdefault(key, []).append(atom)
    return list(groups.items())


def _safe_token(value: object) -> str:
    text = "".join(character.lower() if character.isalnum() else "-" for character in str(value))
    return "-".join(part for part in text.split("-") if part) or "unnamed"


def _module_budget(
    maximum_frames_by_module: Mapping[str, int], module_id: str, default: int
) -> int:
    value = maximum_frames_by_module.get(module_id, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AutomaticChemistryError(f"invalid inferred frame budget for {module_id}")
    return value


def _ion_cutoff(element: str) -> float:
    return {
        "MG": 3.0, "ZN": 3.0, "MN": 3.0, "FE": 3.0,
        "CO": 3.0, "NI": 3.0, "CU": 3.0,
        "CA": 3.5, "LI": 3.0, "NA": 3.5, "K": 4.0,
    }.get(element.upper(), 3.5)


def _ring_rows(
    atoms: Sequence[AtomRecord], coordinates: Sequence[Sequence[float]]
) -> tuple[list[Dict[str, object]], list[Dict[str, object]], list[Dict[str, object]]]:
    rings: list[Dict[str, object]] = []
    identities: list[Dict[str, object]] = []
    centroids: Dict[str, tuple[float, float, float]] = {}
    for key, residue_atoms in _residue_groups(atoms):
        names = {atom.atom_name.upper(): atom for atom in residue_atoms if not atom.altloc}
        residue_name = str(key[3]).upper()
        order = (
            _PURINE_ORDER if "N9" in names and "N7" in names
            else _PYRIMIDINE_ORDER if "N1" in names and "C6" in names
            else ()
        )
        indices = [names[name].atom_index for name in order if name in names]
        if (
            len(indices) < 4
            or (residue_name not in NUCLEIC_RESIDUES and "C1'" not in names)
        ):
            continue
        ring_id = "base-{}-{}{}-{}".format(
            _safe_token(key[0] or "chain"), key[1], _safe_token(key[2]),
            _safe_token(residue_name),
        )
        rings.append({"ring_id": ring_id, "atom_indices": indices})
        identities.append({
            "ring_id": ring_id, "chain_id": str(key[0]),
            "residue_number": int(key[1]), "insertion_code": str(key[2]),
            "residue_name": residue_name,
        })
        centroids[ring_id] = tuple(
            sum(float(coordinates[index][axis]) for index in indices) / len(indices)
            for axis in range(3)
        )
    plane_pairs: list[Dict[str, object]] = []
    by_chain: Dict[str, list[Dict[str, object]]] = defaultdict(list)
    for row in identities:
        by_chain[str(row["chain_id"])].append(row)
    for chain_id, rows in sorted(by_chain.items()):
        ordered = sorted(rows, key=lambda row: (int(row["residue_number"]), str(row["insertion_code"])))
        for first, second in zip(ordered, ordered[1:]):
            plane_pairs.append({
                "pair_id": f"adjacent-{_safe_token(chain_id)}-{first['residue_number']}-{second['residue_number']}",
                "first_ring_id": first["ring_id"],
                "second_ring_id": second["ring_id"],
                "interpretation": "base_stacking",
            })
    for left_index, left in enumerate(identities[:-1]):
        for right in identities[left_index + 1:]:
            if left["chain_id"] == right["chain_id"]:
                continue
            separation = _distance(centroids[str(left["ring_id"])], centroids[str(right["ring_id"])])
            if separation <= 8.0:
                plane_pairs.append({
                    "pair_id": f"cross-{_safe_token(left['chain_id'])}-{left['residue_number']}-{_safe_token(right['chain_id'])}-{right['residue_number']}",
                    "first_ring_id": left["ring_id"],
                    "second_ring_id": right["ring_id"],
                    "interpretation": "base_stacking",
                })
    unique = {str(row["pair_id"]): row for row in plane_pairs}
    return rings, [unique[key] for key in sorted(unique)], identities


def infer_standard_chemistry_definitions(
    pdb_path: Path,
    *,
    maximum_frames_by_module: Mapping[str, int],
    total_source_frames: int,
    dssr_executable: str | None = None,
    ion_site_classification_enabled: bool = True,
) -> Dict[str, object]:
    """Infer standard ion, nucleic-acid, scalar, and RDF definitions.

    The returned definitions are intended for one topology-compatible system.
    Cross-system dependent analyses are generated only after their upstream
    reports exist.
    """

    source = Path(pdb_path).expanduser().resolve(strict=True)
    try:
        atoms = read_pdb_atoms(source)
        coordinates = next(iter_coordinate_frames(source, "angstrom")).coordinates_angstrom
    except (AtomMappingError, CoordinateReadError, StopIteration) as exc:
        raise AutomaticChemistryError(str(exc)) from exc
    if len(atoms) != len(coordinates):
        raise AutomaticChemistryError("reference atom and coordinate counts differ")

    water_oxygens = [
        atom.atom_index for atom in atoms
        if atom.residue_name.upper() in WATER_RESIDUES and atom.element.upper() == "O"
    ]
    ion_atoms = [
        atom for atom in atoms
        if atom.residue_name.upper() in ION_RESIDUES
        and atom.element.upper() in CATION_ELEMENTS
    ]
    solute_ligands = [
        atom for atom in atoms
        if atom.element.upper() in {"N", "O", "S"}
        and atom.residue_name.upper() not in WATER_RESIDUES
        and atom.residue_name.upper() not in ION_RESIDUES
    ]
    atmosphere_ions = [
        atom for atom in atoms
        if atom.residue_name.upper() in ION_RESIDUES
        and atom.element.upper() in (CATION_ELEMENTS | ANION_ELEMENTS)
    ]
    polar_solute_atoms = [
        atom for atom in atoms
        if atom.element.upper() in {"N", "O", "S", "P"}
        and atom.residue_name.upper() not in WATER_RESIDUES
        and atom.residue_name.upper() not in ION_RESIDUES
    ]
    ion_sites: list[Dict[str, object]] = []
    selected_ion_atoms: list[AtomRecord] = []
    site_inference: list[Dict[str, object]] = []
    for ordinal, ion in enumerate(ion_atoms, start=1):
        ranked = sorted(
            ((_distance(coordinates[ion.atom_index], coordinates[atom.atom_index]), atom.atom_index)
             for atom in solute_ligands),
            key=lambda row: (row[0], row[1]),
        )
        if not ranked:
            continue
        site_id = f"{_safe_token(ion.element or ion.residue_name)}-{ordinal}"
        cutoff = _ion_cutoff(ion.element)
        nearest_distance = float(ranked[0][0])
        candidate_screening_distance = (
            max(6.0, cutoff + 2.0)
            if ion_site_classification_enabled else cutoff
        )
        retained = nearest_distance <= candidate_screening_distance
        candidates = [
            index for distance, index in ranked
            if distance <= candidate_screening_distance
        ][:16]
        inference_row = {
            "site_id": site_id,
            "ion_atom_index": ion.atom_index,
            "ion_identity": {
                "chain_id": ion.chain_id, "residue_number": ion.residue_number,
                "residue_name": ion.residue_name, "atom_name": ion.atom_name,
                "element": ion.element,
            },
            "reference_nearest_solute_ligand_distance_angstrom": nearest_distance,
            "reference_candidate_ligand_count": len(candidates),
            "candidate_screening_distance_angstrom": candidate_screening_distance,
            # Reference-distant bulk ions remain in the audit inventory, but
            # must not enter the expensive trajectory classifier.  Only the
            # screened candidates are propagated into ion sites and pairs.
            "trajectory_classification_requested": bool(
                ion_site_classification_enabled and retained
            ),
            "screening_status": (
                "retained_trajectory_binding_candidate"
                if retained and ion_site_classification_enabled
                else "retained_reference_inner_shell"
                if retained else "excluded_bulk_mobile_candidate"
            ),
        }
        site_inference.append(inference_row)
        if not retained:
            continue
        ion_sites.append({
            "site_id": site_id,
            "ion_atom_index": ion.atom_index,
            "candidate_ligand_atom_indices": candidates,
            "coordination_cutoff_angstrom": cutoff,
            "geometry_templates": ["tetrahedral", "square_planar", "octahedral"],
        })
        selected_ion_atoms.append(ion)

    ion_pairs: list[Dict[str, object]] = []
    for first_index, first in enumerate(ion_sites[:-1]):
        for second in ion_sites[first_index + 1:]:
            ion_pairs.append({
                "pair_id": f"{first['site_id']}_{second['site_id']}",
                "first_ion_atom_index": first["ion_atom_index"],
                "second_ion_atom_index": second["ion_atom_index"],
            })

    feature_rows: list[Dict[str, object]] = []
    threshold_rows: list[Dict[str, object]] = []
    distribution_rows: list[Dict[str, object]] = []
    for pair in ion_pairs:
        feature_id = str(pair["pair_id"])
        indices = [int(pair["first_ion_atom_index"]), int(pair["second_ion_atom_index"])]
        feature_rows.append({"feature_id": feature_id, "kind": "pair_distance", "atom_indices": indices})
        cutoff = 4.0
        threshold_rows.append({
            "state_analysis_id": f"{feature_id}-contact-state",
            "question": f"How sensitive is the close-pair population for {feature_id}?",
            "feature_id": feature_id, "value_index": 0,
            "operator": "less_than_or_equal", "threshold": cutoff,
            "sensitivity_thresholds": [cutoff - 0.5, cutoff, cutoff + 0.5],
            "meets_threshold_label": "close_pair",
            "does_not_meet_threshold_label": "separated_pair",
        })
        distribution_rows.append({
            "distribution_id": f"{feature_id}-scott", "feature_id": feature_id,
            "value_index": 0, "binning_rule": "scott", "minimum_bins": 5,
            "maximum_bins": 200, "padding_fraction": 0.05,
            "question": f"How is the inferred ion-pair distance {feature_id} distributed?",
        })
    if not feature_rows:
        for site in ion_sites:
            feature_id = f"{site['site_id']}-nearest-solute-ligand"
            feature_rows.append({
                "feature_id": feature_id, "kind": "group_minimum_distance",
                "group_a_atom_indices": [int(site["ion_atom_index"])],
                "group_b_atom_indices": list(site["candidate_ligand_atom_indices"]),
            })
            cutoff = float(site["coordination_cutoff_angstrom"])
            threshold_rows.append({
                "state_analysis_id": f"{feature_id}-association-state",
                "question": f"How sensitive is inferred association of {site['site_id']}?",
                "feature_id": feature_id, "value_index": 0,
                "operator": "less_than_or_equal", "threshold": cutoff,
                "sensitivity_thresholds": [cutoff - 0.25, cutoff, cutoff + 0.25],
                "meets_threshold_label": "inner_shell_associated",
                "does_not_meet_threshold_label": "not_inner_shell_associated",
            })
            distribution_rows.append({
                "distribution_id": f"{feature_id}-scott", "feature_id": feature_id,
                "value_index": 0, "binning_rule": "scott", "minimum_bins": 5,
                "maximum_bins": 200, "padding_fraction": 0.05,
                "question": f"How is {feature_id} distributed?",
            })

    rings, plane_pairs, ring_identities = _ring_rows(atoms, coordinates)
    definitions: Dict[str, object] = {}
    applicable: list[str] = []
    not_applicable: Dict[str, str] = {}
    if atmosphere_ions and polar_solute_atoms:
        atmosphere_frames = _module_budget(
            maximum_frames_by_module, "ion_atmosphere", total_source_frames
        )
        by_species: Dict[str, list[int]] = defaultdict(list)
        for atom in atmosphere_ions:
            by_species[atom.element.upper()].append(atom.atom_index)
        target_groups: list[Dict[str, object]] = [{
            "target_id": "all_solute",
            "atom_indices": [atom.atom_index for atom in polar_solute_atoms],
        }]
        class_rows = {
            "protein": [
                atom.atom_index for atom in polar_solute_atoms
                if atom.residue_name.upper() in PROTEIN_RESIDUES
            ],
            "nucleic_acid": [
                atom.atom_index for atom in polar_solute_atoms
                if atom.residue_name.upper() in NUCLEIC_RESIDUES
                or "'" in atom.atom_name
                or atom.atom_name.upper() in {"P", "OP1", "OP2", "O1P", "O2P"}
            ],
        }
        assigned = set(class_rows["protein"]) | set(class_rows["nucleic_acid"])
        class_rows["ligand_or_cofactor"] = [
            atom.atom_index for atom in polar_solute_atoms
            if atom.atom_index not in assigned
        ]
        target_groups.extend(
            {"target_id": target_id, "atom_indices": indices}
            for target_id, indices in class_rows.items() if indices
        )
        definitions["ion_atmosphere"] = {
            "frame_stride": 1,
            "maximum_frames": atmosphere_frames,
            "shell_cutoffs_angstrom": [3.5, 5.0, 6.0],
            "ion_groups": [{
                "species": species,
                "charge_class": (
                    "anion" if species in ANION_ELEMENTS else "cation"
                ),
                "atom_indices": indices,
            } for species, indices in sorted(by_species.items())],
            "target_groups": target_groups,
        }
        applicable.append("ion_atmosphere")
    else:
        not_applicable["ion_atmosphere"] = (
            "supported ions and polar non-solvent solute atoms were not both detected"
        )
    if ion_sites:
        ion_frames = _module_budget(
            maximum_frames_by_module, "ion_coordination_geometry", total_source_frames
        )
        definitions["ion_coordination_geometry"] = {
            "frame_stride": 1, "maximum_frames": ion_frames,
            "ion_sites": ion_sites, "ion_pairs": ion_pairs,
            "block_count": 10, "histogram_rule": "scott",
            "histogram_padding_fraction": 0.05,
            "minimum_histogram_bins": 5, "maximum_histogram_bins": 200,
        }
        maximum_values = max(total_source_frames, total_source_frames * max(1, len(feature_rows)))
        definitions["trajectory_features"] = {
            "frame_stride": 1, "maximum_feature_values": maximum_values,
            "features": feature_rows,
        }
        definitions["scalar_feature_distributions"] = {
            "source": "trajectory_features", "maximum_observations": maximum_values,
            "distributions": distribution_rows,
        }
        definitions["scalar_threshold_states"] = {
            "source": "trajectory_features", "maximum_observations": maximum_values,
            "states": threshold_rows,
        }
        not_applicable["optional_observables"] = (
            "automatic ion-distance questions are represented once by "
            "trajectory_features and its scalar distribution/state consumers; "
            "optional_observables remains available for distinct user-declared questions"
        )
        applicable.extend([
            "ion_coordination_geometry", "trajectory_features",
            "scalar_feature_distributions",
            "scalar_threshold_states",
        ])
    else:
        for module_id in (
            "ion_coordination_geometry", "trajectory_features",
            "optional_observables", "scalar_feature_distributions",
            "scalar_threshold_states",
        ):
            not_applicable[module_id] = "no supported cation atoms were detected"

    if rings:
        nucleic_frames = _module_budget(
            maximum_frames_by_module, "nucleic_acid_geometry", total_source_frames
        )
        definitions["nucleic_acid_geometry"] = {
            "frame_stride": 1, "maximum_frames": nucleic_frames,
            "rings": rings, "plane_pairs": plane_pairs, "block_count": 10,
            "histogram_rule": "scott", "histogram_padding_fraction": 0.05,
            "minimum_histogram_bins": 5, "maximum_histogram_bins": 200,
        }
        applicable.append("nucleic_acid_geometry")
        if dssr_executable:
            dssr_frames = _module_budget(
                maximum_frames_by_module, "nucleic_acid_structure", min(total_source_frames, 1000)
            )
            definitions["nucleic_acid_structure"] = {
                "method": "x3dna-dssr-json", "executable": dssr_executable,
                "frame_stride": 1, "maximum_frames": dssr_frames,
                "timeout_seconds": 120.0,
                "json_collection_fields": [
                    "pairs", "helices", "stems", "junctions", "multiplets",
                ],
                "numeric_queries": [],
            }
            applicable.append("nucleic_acid_structure")
        else:
            not_applicable["nucleic_acid_structure"] = (
                "nucleic acid is present but an x3dna-dssr executable was not supplied"
            )
    else:
        not_applicable["nucleic_acid_geometry"] = "no complete nucleobase rings were detected"
        not_applicable["nucleic_acid_structure"] = "no complete nucleobase rings were detected"

    if ion_sites and water_oxygens:
        rdf_frames = _module_budget(
            maximum_frames_by_module, "radial_distribution_functions", total_source_frames
        )
        rdf_stride = max(1, math.ceil(total_source_frames / rdf_frames))
        rdf_selected_frames = math.ceil(total_source_frames / rdf_stride)
        by_element: Dict[str, list[int]] = defaultdict(list)
        for site, atom in zip(ion_sites, selected_ion_atoms):
            by_element[atom.element.upper()].append(int(site["ion_atom_index"]))
        rdf_features = [{
            "feature_id": f"{_safe_token(element)}-water-oxygen-rdf",
            "question": f"How is water oxygen distributed around inferred {element} ion candidates?",
            "group_a_atom_indices": indices,
            "group_b_atom_indices": water_oxygens,
            "minimum_radius_angstrom": 0.0,
            "maximum_radius_angstrom": 10.0,
            "bin_width_angstrom": 0.1,
        } for element, indices in sorted(by_element.items())]
        definitions["radial_distribution_functions"] = {
            "frame_stride": 1,
            "frame_selection": (
                {"mode": "fixed_stride_v1"}
                if rdf_stride == 1 else {
                    "mode": "integer_stride_per_replica_v1",
                    "stride": rdf_stride,
                }
            ),
            "maximum_observations": max(
                rdf_selected_frames * len(water_oxygens) * max(1, len(ion_sites)), 1
            ),
            "features": rdf_features,
        }
        applicable.append("radial_distribution_functions")
    else:
        not_applicable["radial_distribution_functions"] = (
            "standard cation candidates and water oxygens were not both detected"
        )

    return {
        "definitions": definitions,
        "applicable_modules": sorted(applicable),
        "not_applicable_modules": not_applicable,
        "inference": {
            "inference_schema": "salsbury-standard-chemical-context-v1",
            "reference_structure": str(source),
            "ion_site_classification_enabled": bool(ion_site_classification_enabled),
            "ion_candidates": site_inference,
            "retained_ion_candidate_count": len(ion_sites),
            "excluded_bulk_mobile_ion_count": sum(
                row["screening_status"] == "excluded_bulk_mobile_candidate"
                for row in site_inference
            ),
            "water_oxygen_count": len(water_oxygens),
            "ion_atmosphere_species": sorted({
                atom.element.upper() for atom in atmosphere_ions
            }),
            "ion_atmosphere_ion_count": len(atmosphere_ions),
            "ion_atmosphere_target_polar_atom_count": len(polar_solute_atoms),
            "nucleic_ring_identities": ring_identities,
            "nucleic_plane_pair_count": len(plane_pairs),
            "limitations": [
                "Topology identity and one reference geometry define candidates; trajectory modules validate persistence.",
                "Persistent site binding is descriptive and does not establish biological function or binding free energy.",
                "Nonstandard chemistry and ambiguous equivalence remain overrideable project inputs.",
            ],
        },
    }
