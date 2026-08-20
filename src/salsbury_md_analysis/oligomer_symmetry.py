"""Outcome-independent equivalent-oligomer planning and paired-member statistics."""

from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .atom_mapping import AtomRecord, read_topology_atoms
from .chemical_identity import ION_RESIDUES, WATER_RESIDUES
from .context import compile_project_context_file
from .coordinates import iter_coordinate_frames
from .frame_sampling import (
    FrameSelectionPlan,
    frame_selected,
    plan_frame_selection,
    reader_frame_indices,
)
from .geometry import apply_transform, best_fit_transform
from .manifests import load_json, resolve_manifest_path, sha256_file
from .moments import sample_summary
from .pca_math import (
    CartesianCovariance,
    PCAResult,
    mixture_covariance,
    principal_components,
    project as project_scores,
    randomized_truncated_pca,
)
from .periodic import PeriodicFrameProcessor
from .reporting import atom_identity_record, issue_record
from .trajectory_contracts import (
    frame_axis_value,
    normalize_segment_axis,
    require_periodic_policy,
)
from .upstream_cache import project_module_contract_sha256


Coordinate = Tuple[float, float, float]
ResidueKey = Tuple[str, int, str]

_PROTEIN_BACKBONE = {"N", "CA", "C", "O"}
_NUCLEIC_BACKBONE = {
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C1'", "C2'",
    "O2'", "C3'", "O3'",
}


class OligomerSymmetryError(ValueError):
    """Raised when equivalent-member identity is ambiguous or invalid."""


def _heavy(atom: AtomRecord) -> bool:
    return not (
        atom.element.upper() == "H"
        or atom.atom_name.lstrip("0123456789").upper().startswith("H")
    )


def _residues(atoms: Sequence[AtomRecord]) -> Dict[ResidueKey, List[AtomRecord]]:
    rows: Dict[ResidueKey, List[AtomRecord]] = defaultdict(list)
    for atom in atoms:
        rows[(atom.chain_id, atom.residue_number, atom.insertion_code)].append(atom)
    return rows


def _chain_classes(
    atoms: Sequence[AtomRecord],
) -> Tuple[Dict[str, List[ResidueKey]], Dict[str, List[ResidueKey]]]:
    residues = _residues(atoms)
    protein: Dict[str, List[ResidueKey]] = defaultdict(list)
    nucleic: Dict[str, List[ResidueKey]] = defaultdict(list)
    for key, rows in residues.items():
        names = {atom.atom_name.upper() for atom in rows}
        residue_name = rows[0].residue_name.upper()
        if residue_name in WATER_RESIDUES or residue_name in ION_RESIDUES:
            continue
        if {"N", "CA", "C"}.issubset(names):
            protein[key[0]].append(key)
        elif "C1'" in names and ({"P", "O4'", "C4'"} & names):
            nucleic[key[0]].append(key)
    return (
        {chain: sorted(keys, key=lambda key: (key[1], key[2])) for chain, keys in protein.items()},
        {chain: sorted(keys, key=lambda key: (key[1], key[2])) for chain, keys in nucleic.items()},
    )


def _chain_signature(
    chain_keys: Sequence[ResidueKey],
    residues: Mapping[ResidueKey, Sequence[AtomRecord]],
    *,
    include_residue_names: bool = True,
) -> Tuple[object, ...]:
    signature: List[object] = []
    for ordinal, key in enumerate(chain_keys):
        rows = sorted(residues[key], key=lambda atom: atom.atom_index)
        signature.append((
            ordinal,
            rows[0].residue_name.upper() if include_residue_names else None,
            tuple((atom.atom_name.upper(), atom.altloc, atom.element.upper()) for atom in rows),
        ))
    return tuple(signature)


def _minimum_heavy_distance_squared(
    left_keys: Sequence[ResidueKey],
    right_keys: Sequence[ResidueKey],
    residues: Mapping[ResidueKey, Sequence[AtomRecord]],
    coordinates: Sequence[Coordinate],
) -> float:
    left = [atom for key in left_keys for atom in residues[key] if _heavy(atom)]
    right = [atom for key in right_keys for atom in residues[key] if _heavy(atom)]
    if not left or not right:
        raise OligomerSymmetryError("member-chain distance requires heavy atoms")
    best = math.inf
    for first in left:
        x, y, z = coordinates[first.atom_index]
        for second in right:
            a, b, c = coordinates[second.atom_index]
            squared = (x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2
            if squared < best:
                best = squared
    return best


def _canonical_chain_atoms(
    chain_keys: Sequence[ResidueKey],
    residues: Mapping[ResidueKey, Sequence[AtomRecord]],
    component_role: str,
    *,
    heavy_only: bool,
    backbone_only: bool,
    include_residue_names: bool,
) -> Dict[Tuple[object, ...], AtomRecord]:
    result: Dict[Tuple[object, ...], AtomRecord] = {}
    for residue_ordinal, key in enumerate(chain_keys):
        rows = sorted(residues[key], key=lambda atom: atom.atom_index)
        residue_name = rows[0].residue_name.upper()
        for atom in rows:
            name = atom.atom_name.upper()
            if heavy_only and not _heavy(atom):
                continue
            if backbone_only:
                allowed = (
                    _PROTEIN_BACKBONE
                    if component_role == "protein"
                    else _NUCLEIC_BACKBONE
                )
                if name not in allowed:
                    continue
            identity = (
                component_role,
                residue_ordinal,
                residue_name if include_residue_names else None,
                name,
                atom.altloc,
                atom.element.upper(),
            )
            if identity in result:
                raise OligomerSymmetryError(
                    f"duplicate canonical member atom identity {identity!r}"
                )
            result[identity] = atom
    return result


def _member_atom_maps(
    member: Mapping[str, object],
    residues: Mapping[ResidueKey, Sequence[AtomRecord]],
    protein: Mapping[str, Sequence[ResidueKey]],
    nucleic: Mapping[str, Sequence[ResidueKey]],
    *,
    include_residue_names: bool,
) -> Tuple[Dict[Tuple[object, ...], AtomRecord], Dict[Tuple[object, ...], AtomRecord]]:
    protein_chain = str(member["protein_chain_id"])
    nucleic_chains = member.get("nucleic_chain_ids", [])
    if not isinstance(nucleic_chains, list):
        raise OligomerSymmetryError("nucleic_chain_ids must be an array")
    if protein_chain not in protein:
        raise OligomerSymmetryError(
            f"member protein chain {protein_chain!r} is absent"
        )
    analysis = _canonical_chain_atoms(
        protein[protein_chain], residues, "protein",
        heavy_only=True, backbone_only=False,
        include_residue_names=include_residue_names,
    )
    alignment = _canonical_chain_atoms(
        protein[protein_chain], residues, "protein",
        heavy_only=True, backbone_only=True,
        include_residue_names=include_residue_names,
    )
    for role_index, chain_id in enumerate(nucleic_chains, start=1):
        chain = str(chain_id)
        if chain not in nucleic:
            raise OligomerSymmetryError(
                f"member nucleic-acid chain {chain!r} is absent"
            )
        role = f"nucleic_{role_index}"
        for destination, backbone_only in ((analysis, False), (alignment, True)):
            rows = _canonical_chain_atoms(
                nucleic[chain], residues, role,
                heavy_only=True, backbone_only=backbone_only,
                include_residue_names=include_residue_names,
            )
            overlap = set(destination).intersection(rows)
            if overlap:
                raise OligomerSymmetryError(
                    f"duplicate canonical member identities for {role}"
                )
            destination.update(rows)
    if len(alignment) < 3:
        raise OligomerSymmetryError(
            "each oligomer member requires at least three alignment atoms"
        )
    return analysis, alignment


def plan_equivalent_oligomer_members(
    atoms: Sequence[AtomRecord],
    coordinates_angstrom: Sequence[Coordinate],
    *,
    maximum_assignment_distance_angstrom: float = 15.0,
    minimum_member_count: int = 2,
    maximum_member_count: int = 12,
) -> Dict[str, object]:
    """Detect strictly equivalent protein-centered oligomer members.

    The decision uses only the reference topology, reference coordinates, and
    declared gates. Nucleic-acid chains are assigned to their uniquely nearest
    protein chain. Ambiguous assignments or non-equivalent members remain
    explicitly inapplicable rather than being guessed.
    """

    if len(atoms) != len(coordinates_angstrom):
        raise OligomerSymmetryError(
            "topology and reference coordinate atom counts differ"
        )
    if maximum_assignment_distance_angstrom <= 0.0:
        raise OligomerSymmetryError(
            "maximum_assignment_distance_angstrom must be positive"
        )
    if minimum_member_count < 2 or maximum_member_count < minimum_member_count:
        raise OligomerSymmetryError("oligomer member-count gates are invalid")
    residues = _residues(atoms)
    protein, nucleic = _chain_classes(atoms)
    protein_chains = sorted(protein)
    if not minimum_member_count <= len(protein_chains) <= maximum_member_count:
        return {
            "planning_schema": "salsbury-equivalent-oligomer-plan-v1",
            "applicable": False,
            "reason": (
                f"detected {len(protein_chains)} protein chains; automatic equivalent-"
                f"oligomer analysis requires {minimum_member_count}--{maximum_member_count}"
            ),
            "detected_protein_chain_ids": protein_chains,
        }
    protein_signatures = {
        chain: _chain_signature(protein[chain], residues)
        for chain in protein_chains
    }

    if len(set(protein_signatures.values())) != 1:
        return {
            "planning_schema": "salsbury-equivalent-oligomer-plan-v1",
            "applicable": False,
            "reason": "protein chains are not strictly topology-equivalent",
            "detected_protein_chain_ids": protein_chains,
        }

    assigned: Dict[str, List[str]] = {chain: [] for chain in protein_chains}
    assignment_evidence = []
    cutoff2 = maximum_assignment_distance_angstrom ** 2
    for nucleic_chain in sorted(nucleic):
        distances = sorted(
            (
                _minimum_heavy_distance_squared(
                    nucleic[nucleic_chain], protein[protein_chain],
                    residues, coordinates_angstrom,
                ),
                protein_chain,
            )
            for protein_chain in protein_chains
        )
        best_squared, best_chain = distances[0]
        if best_squared > cutoff2:
            return {
                "planning_schema": "salsbury-equivalent-oligomer-plan-v1",
                "applicable": False,
                "reason": (
                    f"nucleic-acid chain {nucleic_chain!r} is not within the declared "
                    "member-assignment distance of a protein chain"
                ),
                "detected_protein_chain_ids": protein_chains,
            }
        tolerance = max(1.0e-8, best_squared * 1.0e-8)
        if len(distances) > 1 and abs(distances[1][0] - best_squared) <= tolerance:
            return {
                "planning_schema": "salsbury-equivalent-oligomer-plan-v1",
                "applicable": False,
                "reason": (
                    f"nucleic-acid chain {nucleic_chain!r} has an ambiguous nearest "
                    "protein member in the reference structure"
                ),
                "detected_protein_chain_ids": protein_chains,
            }
        assigned[best_chain].append(nucleic_chain)
        assignment_evidence.append({
            "nucleic_chain_id": nucleic_chain,
            "protein_chain_id": best_chain,
            "minimum_heavy_atom_distance_angstrom": math.sqrt(best_squared),
        })

    # Canonicalize nucleic roles by topology signature. Duplicate signatures
    # within one member are ambiguous because chain labels are intentionally
    # removed from the pooled canonical representation.
    members = []
    member_nucleic_signatures = []
    for member_index, protein_chain in enumerate(protein_chains, start=1):
        rows = []
        seen = set()
        for chain in assigned[protein_chain]:
            signature = _chain_signature(nucleic[chain], residues)
            if signature in seen:
                return {
                    "planning_schema": "salsbury-equivalent-oligomer-plan-v1",
                    "applicable": False,
                    "reason": (
                        f"member {protein_chain!r} contains topology-identical nucleic-acid "
                        "chains whose canonical roles are ambiguous"
                    ),
                    "detected_protein_chain_ids": protein_chains,
                }
            seen.add(signature)
            rows.append((signature, chain))
        rows.sort(key=lambda row: (row[0], row[1]))
        member_nucleic_signatures.append(tuple(row[0] for row in rows))
        members.append({
            "member_id": f"member-{member_index}",
            "protein_chain_id": protein_chain,
            "nucleic_chain_ids": [row[1] for row in rows],
        })
    if len(set(member_nucleic_signatures)) != 1:
        return {
            "planning_schema": "salsbury-equivalent-oligomer-plan-v1",
            "applicable": False,
            "reason": "protein-centered members do not carry equivalent nucleic-acid chain sets",
            "detected_protein_chain_ids": protein_chains,
        }

    analysis_maps = []
    alignment_maps = []
    for member in members:
        analysis, alignment = _member_atom_maps(
            member, residues, protein, nucleic, include_residue_names=True
        )
        analysis_maps.append(analysis)
        alignment_maps.append(alignment)
    analysis_keys = tuple(sorted(analysis_maps[0]))
    alignment_keys = tuple(sorted(alignment_maps[0]))
    if any(tuple(sorted(rows)) != analysis_keys for rows in analysis_maps[1:]):
        raise OligomerSymmetryError(
            "detected members do not share one strict heavy-atom identity"
        )
    if any(tuple(sorted(rows)) != alignment_keys for rows in alignment_maps[1:]):
        raise OligomerSymmetryError(
            "detected members do not share one strict alignment identity"
        )
    for member, analysis, alignment in zip(members, analysis_maps, alignment_maps):
        member["analysis_atom_indices"] = [analysis[key].atom_index for key in analysis_keys]
        member["alignment_atom_indices"] = [alignment[key].atom_index for key in alignment_keys]

    return {
        "planning_schema": "salsbury-equivalent-oligomer-plan-v1",
        "applicable": True,
        "detection_basis": (
            "reference topology equivalence plus uniquely nearest reference-coordinate "
            "assignment of nucleic-acid chains; no trajectory outcomes inspected"
        ),
        "oligomer_order": len(members),
        "member_count": len(members),
        "member_kind": (
            "protein_nucleic_acid_complex" if nucleic else "protein_protomer"
        ),
        "members": members,
        "canonical_reference_member_id": members[0]["member_id"],
        "analysis_atom_count_per_member": len(analysis_keys),
        "alignment_atom_count_per_member": len(alignment_keys),
        "nucleic_chain_assignment": assignment_evidence,
        "maximum_assignment_distance_angstrom": maximum_assignment_distance_angstrom,
        "observation_contract": {
            "physical_frame_multiplier": 1,
            "member_observation_multiplier": len(members),
            "independent_sampling_unit": "original simulation replica and physical time block",
            "member_observations_are_independent_replicas": False,
        },
        "alignment_contract": (
            "each member is independently least-squares aligned to the canonical reference "
            "member on equivalent protein/nucleic-acid backbone atoms before pooled analysis"
        ),
    }


def restrict_equivalent_member_plan(
    plan: Mapping[str, object],
    selected_atom_indices: Sequence[int],
    *,
    selection_id: str,
) -> Dict[str, object]:
    """Restrict every equivalent member to one symmetrized canonical atom subset."""

    if plan.get("applicable") is not True:
        raise OligomerSymmetryError("cannot restrict an inapplicable oligomer plan")
    members = plan.get("members")
    if not isinstance(members, list) or len(members) < 2:
        raise OligomerSymmetryError("restricted oligomer plan requires members")
    selected = {int(index) for index in selected_atom_indices}
    if not selected:
        raise OligomerSymmetryError("member-interface selection is empty")
    arrays = []
    for member in members:
        if not isinstance(member, dict) or not isinstance(
            member.get("analysis_atom_indices"), list
        ):
            raise OligomerSymmetryError("oligomer member lacks analysis atom indices")
        arrays.append([int(index) for index in member["analysis_atom_indices"]])
    if len({len(values) for values in arrays}) != 1:
        raise OligomerSymmetryError("oligomer member analysis maps differ in width")
    positions = [
        position for position in range(len(arrays[0]))
        if any(values[position] in selected for values in arrays)
    ]
    if not positions:
        raise OligomerSymmetryError(
            "whole-system interface selection has no atoms in equivalent members"
        )
    restricted = deepcopy(dict(plan))
    restricted_members = restricted.get("members")
    assert isinstance(restricted_members, list)
    for member, values in zip(restricted_members, arrays):
        assert isinstance(member, dict)
        member["analysis_atom_indices"] = [values[position] for position in positions]
    explicit_identities = restricted.get("analysis_identity_keys")
    if isinstance(explicit_identities, list):
        if len(explicit_identities) != len(arrays[0]):
            raise OligomerSymmetryError(
                "oligomer analysis_identity_keys width differs from member maps"
            )
        restricted["analysis_identity_keys"] = [
            explicit_identities[position] for position in positions
        ]
        restricted.pop("analysis_position_indices", None)
    else:
        restricted["analysis_position_indices"] = positions
    restricted.update({
        "analysis_atom_count_per_member": len(positions),
        "analysis_selection_id": selection_id,
        "analysis_selection_policy": (
            "union of topology-derived selected canonical positions across equivalent "
            "members, applied symmetrically to every member"
        ),
    })
    return restricted


def validate_member_plan(
    atoms: Sequence[AtomRecord], plan: Mapping[str, object], *, policy: str = "strict"
) -> Dict[str, object]:
    """Resolve one stored plan against a topology and return canonical index maps."""

    if policy not in {"strict", "position"}:
        raise OligomerSymmetryError("member mapping policy must be strict or position")
    members = plan.get("members")
    if not isinstance(members, list) or len(members) < 2:
        raise OligomerSymmetryError("equivalent oligomer plan requires at least two members")
    residues = _residues(atoms)
    protein, nucleic = _chain_classes(atoms)
    analyses = []
    alignments = []
    for raw in members:
        if not isinstance(raw, dict):
            raise OligomerSymmetryError("oligomer members must be objects")
        analysis, alignment = _member_atom_maps(
            raw, residues, protein, nucleic,
            include_residue_names=policy == "strict",
        )
        analyses.append(analysis)
        alignments.append(alignment)
    analysis_keys = sorted(set.intersection(*(set(row) for row in analyses)))
    alignment_keys = sorted(set.intersection(*(set(row) for row in alignments)))
    declared_analysis_keys = plan.get("analysis_identity_keys")
    if declared_analysis_keys is not None:
        if (
            not isinstance(declared_analysis_keys, list)
            or not declared_analysis_keys
            or any(not isinstance(row, list) for row in declared_analysis_keys)
        ):
            raise OligomerSymmetryError(
                "oligomer analysis_identity_keys must be a nonempty array of arrays"
            )
        declared = [tuple(row) for row in declared_analysis_keys]
        if len(set(declared)) != len(declared):
            raise OligomerSymmetryError(
                "oligomer analysis_identity_keys contain duplicates"
            )
        missing = [key for key in declared if any(key not in row for row in analyses)]
        if missing:
            raise OligomerSymmetryError(
                "declared comparative member analysis identities are absent"
            )
        analysis_keys = declared
    declared_alignment_keys = plan.get("alignment_identity_keys")
    if declared_alignment_keys is not None:
        if (
            not isinstance(declared_alignment_keys, list)
            or len(declared_alignment_keys) < 3
            or any(not isinstance(row, list) for row in declared_alignment_keys)
        ):
            raise OligomerSymmetryError(
                "oligomer alignment_identity_keys must contain at least three arrays"
            )
        declared = [tuple(row) for row in declared_alignment_keys]
        if len(set(declared)) != len(declared):
            raise OligomerSymmetryError(
                "oligomer alignment_identity_keys contain duplicates"
            )
        missing = [key for key in declared if any(key not in row for row in alignments)]
        if missing:
            raise OligomerSymmetryError(
                "declared comparative member alignment identities are absent"
            )
        alignment_keys = declared
    if not analysis_keys or len(alignment_keys) < 3:
        raise OligomerSymmetryError("oligomer member mapping has insufficient common atoms")
    position_indices = (
        None
        if declared_analysis_keys is not None
        else plan.get("analysis_position_indices")
    )
    if position_indices is not None:
        if (
            not isinstance(position_indices, list)
            or not position_indices
            or any(
                isinstance(position, bool) or not isinstance(position, int)
                or position < 0 or position >= len(analysis_keys)
                for position in position_indices
            )
            or len(set(position_indices)) != len(position_indices)
        ):
            raise OligomerSymmetryError(
                "oligomer analysis_position_indices are invalid"
            )
        analysis_keys = [analysis_keys[position] for position in position_indices]
    reference_analysis_count = (
        len(declared_analysis_keys)
        if isinstance(declared_analysis_keys, list)
        else len(position_indices) if isinstance(position_indices, list)
        else len(analyses[0])
    )
    coverage = len(analysis_keys) / reference_analysis_count
    resolved = []
    for raw, analysis, alignment in zip(members, analyses, alignments):
        resolved.append({
            "member_id": str(raw["member_id"]),
            "analysis_atom_indices": [analysis[key].atom_index for key in analysis_keys],
            "alignment_atom_indices": [alignment[key].atom_index for key in alignment_keys],
        })
    return {
        "members": resolved,
        "member_count": len(resolved),
        "analysis_atom_count_per_member": len(analysis_keys),
        "alignment_atom_count_per_member": len(alignment_keys),
        "reference_analysis_atom_coverage": coverage,
        "analysis_identity_keys": [list(key) for key in analysis_keys],
        "alignment_identity_keys": [list(key) for key in alignment_keys],
    }


def restrict_resolved_member_analysis(
    resolved: Mapping[str, object], identity_keys: Sequence[Sequence[object]]
) -> Dict[str, object]:
    """Restrict one resolved topology to an ordered cross-topology atom identity."""

    raw_keys = resolved.get("analysis_identity_keys")
    members = resolved.get("members")
    if not isinstance(raw_keys, list) or not isinstance(members, list):
        raise OligomerSymmetryError("resolved member mapping is invalid")
    positions = {tuple(key): index for index, key in enumerate(raw_keys)}
    requested = [tuple(key) for key in identity_keys]
    if not requested or any(key not in positions for key in requested):
        raise OligomerSymmetryError(
            "cross-topology member analysis identity is absent from a resolved topology"
        )
    restricted = deepcopy(dict(resolved))
    restricted_members = restricted.get("members")
    assert isinstance(restricted_members, list)
    selected_positions = [positions[key] for key in requested]
    for member in restricted_members:
        if not isinstance(member, dict) or not isinstance(
            member.get("analysis_atom_indices"), list
        ):
            raise OligomerSymmetryError("resolved member lacks analysis atom indices")
        indices = member["analysis_atom_indices"]
        member["analysis_atom_indices"] = [
            indices[position] for position in selected_positions
        ]
    restricted["analysis_identity_keys"] = [list(key) for key in requested]
    restricted["analysis_atom_count_per_member"] = len(requested)
    return restricted


def align_member_coordinates(
    coordinates: Sequence[Coordinate],
    member: Mapping[str, object],
    reference_coordinates: Sequence[Coordinate],
    reference_member: Mapping[str, object],
) -> Tuple[Coordinate, ...]:
    """Independently align one member and return its canonical analysis coordinates."""

    analysis = member.get("analysis_atom_indices")
    alignment = member.get("alignment_atom_indices")
    reference_alignment = reference_member.get("alignment_atom_indices")
    if not all(isinstance(value, list) for value in (analysis, alignment, reference_alignment)):
        raise OligomerSymmetryError("resolved member plan lacks atom-index arrays")
    try:
        mobile_fit = tuple(coordinates[int(index)] for index in alignment)  # type: ignore[arg-type]
        reference_fit = tuple(
            reference_coordinates[int(index)] for index in reference_alignment  # type: ignore[arg-type]
        )
        mobile_analysis = tuple(coordinates[int(index)] for index in analysis)  # type: ignore[arg-type]
    except IndexError as exc:
        raise OligomerSymmetryError("member atom index exceeds coordinate count") from exc
    transform = best_fit_transform(mobile_fit, reference_fit)
    return apply_transform(mobile_analysis, transform)


def paired_member_score_correlations(
    records: Sequence[Mapping[str, object]],
    *,
    minimum_paired_frames: int = 20,
) -> Dict[str, object]:
    """Calculate within-physical-frame member score coupling by replica.

    Records require system, replica, segment, source frame, member, and PCA
    scores. Correlations are calculated across physical frames, never across
    the symmetry-expanded member-observation rows themselves.
    """

    if minimum_paired_frames < 3:
        raise OligomerSymmetryError("minimum_paired_frames must be at least three")
    grouped: Dict[Tuple[str, str, str], Dict[int, Dict[str, Tuple[float, ...]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for index, row in enumerate(records):
        required = {
            "system_id", "replica_id", "segment_id", "source_frame_index",
            "member_id", "scores_angstrom",
        }
        if not required.issubset(row):
            raise OligomerSymmetryError(
                f"paired score record {index} lacks required provenance"
            )
        scores = row["scores_angstrom"]
        if not isinstance(scores, list) or not scores:
            raise OligomerSymmetryError("paired member scores must be a nonempty array")
        vector = tuple(float(value) for value in scores)
        if not all(math.isfinite(value) for value in vector):
            raise OligomerSymmetryError("paired member scores contain non-finite values")
        key = (str(row["system_id"]), str(row["replica_id"]), str(row["segment_id"]))
        frame = int(row["source_frame_index"])
        member = str(row["member_id"])
        if member in grouped[key][frame]:
            raise OligomerSymmetryError("duplicate member score for one physical frame")
        grouped[key][frame][member] = vector

    reports = []
    for (system_id, replica_id, segment_id), frames in sorted(grouped.items()):
        member_ids = sorted({member for rows in frames.values() for member in rows})
        for left_index, left in enumerate(member_ids[:-1]):
            for right in member_ids[left_index + 1:]:
                paired = [
                    (rows[left], rows[right])
                    for _, rows in sorted(frames.items())
                    if left in rows and right in rows
                ]
                if not paired:
                    continue
                component_count = len(paired[0][0])
                if any(len(a) != component_count or len(b) != component_count for a, b in paired):
                    raise OligomerSymmetryError("paired member score dimensions differ")
                matrix = []
                for left_component in range(component_count):
                    row_values = []
                    for right_component in range(component_count):
                        x = [row[0][left_component] for row in paired]
                        y = [row[1][right_component] for row in paired]
                        x_mean = sum(x) / len(x)
                        y_mean = sum(y) / len(y)
                        numerator = sum(
                            (a - x_mean) * (b - y_mean) for a, b in zip(x, y)
                        )
                        left_ss = sum((a - x_mean) ** 2 for a in x)
                        right_ss = sum((b - y_mean) ** 2 for b in y)
                        value = (
                            numerator / math.sqrt(left_ss * right_ss)
                            if left_ss > 0.0 and right_ss > 0.0 else None
                        )
                        row_values.append(value)
                    matrix.append(row_values)
                reports.append({
                    "system_id": system_id,
                    "replica_id": replica_id,
                    "segment_id": segment_id,
                    "left_member_id": left,
                    "right_member_id": right,
                    "paired_physical_frame_count": len(paired),
                    "status": (
                        "complete" if len(paired) >= minimum_paired_frames
                        else "insufficient_paired_frames"
                    ),
                    "score_cross_correlation_matrix": matrix,
                    "same_component_correlations": [
                        matrix[index][index] for index in range(component_count)
                    ],
                })
    return {
        "correlation_schema": "salsbury-paired-oligomer-member-score-correlation-v1",
        "minimum_paired_frames": minimum_paired_frames,
        "pair_reports": reports,
        "interpretation": (
            "zero-lag Pearson correlations between independently aligned member PCA "
            "scores across matched physical frames; correlation does not establish "
            "causality, direct interaction, or an independent-replica count"
        ),
    }


def _flatten(coordinates: Sequence[Coordinate]) -> Tuple[float, ...]:
    return tuple(value for coordinate in coordinates for value in coordinate)


def _pca_payload(
    mean: Sequence[float], solution: PCAResult, atoms: Sequence[AtomRecord]
) -> Dict[str, object]:
    mean_rows = []
    for index, atom in enumerate(atoms):
        offset = 3 * index
        identity = atom_identity_record(atom, index)
        # Chain identity is deliberately normalized in a member-canonical basis.
        identity["source_reference_chain_id"] = identity.pop("chain_id")
        identity["canonical_component_identity"] = "member_local"
        mean_rows.append({
            **identity,
            "mean_x_angstrom": mean[offset],
            "mean_y_angstrom": mean[offset + 1],
            "mean_z_angstrom": mean[offset + 2],
        })
    components = []
    for component in solution.components:
        loadings = []
        for index, atom in enumerate(atoms):
            offset = 3 * index
            identity = atom_identity_record(atom, index)
            identity["source_reference_chain_id"] = identity.pop("chain_id")
            identity["canonical_component_identity"] = "member_local"
            loadings.append({
                **identity,
                "loading_x": component.vector[offset],
                "loading_y": component.vector[offset + 1],
                "loading_z": component.vector[offset + 2],
            })
        components.append({
            "component_index": component.component_index,
            "eigenvalue_angstrom2": component.eigenvalue_angstrom2,
            "explained_variance_fraction": component.explained_variance_fraction,
            "cumulative_explained_variance_fraction": component.cumulative_explained_variance_fraction,
            "residual_norm_angstrom2": component.residual_norm_angstrom2,
            "iteration_count": component.iteration_count,
            "converged": component.converged,
            "loadings": loadings,
        })
    return {
        "atom_count": len(atoms),
        "feature_count": 3 * len(atoms),
        "total_variance_angstrom2": solution.total_variance_angstrom2,
        "requested_component_count": solution.requested_component_count,
        "returned_component_count": len(solution.components),
        "numerical_rank_lower_bound": solution.numerical_rank_lower_bound,
        "mean_structure": mean_rows,
        "components": components,
    }


def _projection_values(
    reports: Sequence[Mapping[str, object]], component_count: int
) -> List[List[float]]:
    result = [[] for _ in range(component_count)]
    for report in reports:
        projections = report.get("projections")
        if not isinstance(projections, list):
            continue
        for row in projections:
            if not isinstance(row, dict) or not isinstance(row.get("scores_angstrom"), list):
                continue
            for index, value in enumerate(row["scores_angstrom"]):
                result[index].append(float(value))
    return result


def _scan_symmetry_replica(
    *,
    system_id: str,
    replica_id: str,
    replica: Mapping[str, object],
    topology_atoms: Sequence[AtomRecord],
    resolved: Mapping[str, object],
    reference_coordinates: Sequence[Coordinate],
    reference_member: Mapping[str, object],
    project: Mapping[str, object],
    system_path: Path,
    coordinate_unit: str,
    time_unit: str | None,
    periodic_policy: str,
    frame_stride: int,
    frame_selection_plan: FrameSelectionPlan,
    state: CartesianCovariance | None = None,
    mean: Sequence[float] | None = None,
    solution: PCAResult | None = None,
    vector_sink: object = None,
    inventory_by_path: Mapping[str, Mapping[str, object]] | None = None,
) -> List[Dict[str, object]]:
    if (mean is None) != (solution is None):
        raise OligomerSymmetryError(
            "projection mean and PCA solution must be supplied together"
        )
    members = resolved["members"]
    if not isinstance(members, list):
        raise OligomerSymmetryError("resolved oligomer plan lacks members")
    reconstruction = tuple(sorted({
        int(index)
        for member in members if isinstance(member, dict)
        for field in ("analysis_atom_indices", "alignment_atom_indices")
        for index in member[field]  # type: ignore[index]
    }))
    processor = PeriodicFrameProcessor.from_replica(
        project, replica, system_path, len(topology_atoms)
    )
    segments = replica["segments"]
    if not isinstance(segments, list):
        raise OligomerSymmetryError("replica segments must be an array")
    reports = []
    for segment in segments:
        if not isinstance(segment, dict):
            raise OligomerSymmetryError("replica segment must be an object")
        segment_id = str(segment["segment_id"])
        location = f"{system_id}/{replica_id}/{segment_id}"
        trajectory = resolve_manifest_path(str(segment["trajectory"]), system_path)
        selected_indices = frame_selection_plan[(system_id, replica_id, segment_id)]
        reader_indices = reader_frame_indices(selected_indices, processor.policy)
        axis = normalize_segment_axis(segment, time_unit)
        before = trajectory.stat()
        observed = 0
        physical_evaluated = 0
        member_observations = 0
        periodic = 0
        projections = []
        first_axis = None
        last_axis = None
        processor.begin_segment(bool(segment.get("continuous_with_previous", False)))
        for raw_frame in iter_coordinate_frames(
            trajectory, coordinate_unit, reader_indices
        ):
            selected = frame_selected(raw_frame.frame_index, selected_indices, frame_stride)
            if not selected and processor.policy != "unwrap_continuous":
                continue
            frame = processor.process(
                raw_frame,
                f"{location}/frame-{raw_frame.frame_index}",
                reconstruction,
            )
            observed += 1
            periodic += int(frame.periodic_cell_present)
            if frame.atom_count != len(topology_atoms):
                raise OligomerSymmetryError(
                    f"{location} frame {frame.frame_index} atom count differs from topology"
                )
            if not selected:
                continue
            physical_evaluated += 1
            axis_value = frame_axis_value(axis, frame.frame_index)
            first_axis = axis_value if first_axis is None else first_axis
            last_axis = axis_value
            for member in members:
                if not isinstance(member, dict):
                    raise OligomerSymmetryError("resolved oligomer member is invalid")
                aligned = align_member_coordinates(
                    frame.coordinates_angstrom,
                    member,
                    reference_coordinates,
                    reference_member,
                )
                vector = _flatten(aligned)
                if state is not None:
                    state.update(vector)
                if callable(vector_sink):
                    vector_sink(vector, system_id, replica_id, str(member["member_id"]))
                member_observations += 1
                if mean is not None and solution is not None:
                    row: Dict[str, object] = {
                        "source_frame_index": frame.frame_index,
                        "member_id": str(member["member_id"]),
                        "scores_angstrom": list(project_scores(vector, mean, solution.components)),
                    }
                    if axis["kind"] == "physical_time":
                        row.update({"time": axis_value, "time_unit": time_unit})
                    else:
                        row["sample_index"] = axis_value
                    projections.append(row)
        if observed == 0:
            raise OligomerSymmetryError(f"{location} trajectory contains no frames")
        after = trajectory.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise OligomerSymmetryError(f"{location} trajectory changed during analysis")
        inventory = (inventory_by_path or {}).get(str(trajectory), {})
        report: Dict[str, object] = {
            "segment_id": segment_id,
            "trajectory_path": str(trajectory),
            "trajectory_sha256": inventory.get("sha256"),
            "source_fingerprint": {
                "size_bytes": before.st_size,
                "modified_time_ns": before.st_mtime_ns,
            },
            "observed_frame_count": observed,
            "physical_evaluated_frame_count": physical_evaluated,
            "evaluated_frame_count": member_observations,
            "evaluated_member_observation_count": member_observations,
            "member_count": len(members),
            "periodic_cell_frame_count": periodic,
            "periodic_reconstruction_replica_cumulative": processor.report(),
            "frame_axis": axis,
            "evaluated_axis_range": (
                {
                    "start": first_axis,
                    "end": last_axis,
                    "unit": time_unit if axis["kind"] == "physical_time" else "sample",
                }
                if first_axis is not None else None
            ),
            "projections": projections if solution is not None else None,
        }
        reports.append(report)
    return reports


def symmetry_expanded_common_pca_project(
    project_source: Path,
    settings: Mapping[str, object],
    *,
    hash_content: bool = False,
) -> Dict[str, object]:
    """Fit a shared PCA basis to independently aligned equivalent members."""

    source = Path(project_source).expanduser().resolve(strict=False)
    project_data = load_json(source)
    raw_plan = settings.get("symmetry_expansion")
    if not isinstance(raw_plan, dict) or raw_plan.get("applicable") is not True:
        raise OligomerSymmetryError(
            "common_pca symmetry_expansion requires an applicable equivalent-oligomer plan"
        )
    policy = str(project_data.get("common_atom_policy"))
    context = compile_project_context_file(source, hash_content=hash_content)
    contract = context["contract"]
    if not isinstance(contract, dict):
        raise OligomerSymmetryError("compiled project contract is invalid")
    units = contract["units"]
    if not isinstance(units, dict):
        raise OligomerSymmetryError("compiled units are invalid")
    coordinate_unit = str(units["coordinates"])
    time_value = units.get("time")
    time_unit = str(time_value) if isinstance(time_value, str) else None
    periodic_policy = require_periodic_policy(contract.get("periodic_coordinate_policy"))
    reference_path = resolve_manifest_path(str(project_data["reference_structure"]), source)
    reference_format, reference_atoms = read_topology_atoms(reference_path)
    reference_raw = next(iter_coordinate_frames(reference_path, coordinate_unit))
    reference_processor = PeriodicFrameProcessor.from_reference(
        project_data, source, len(reference_atoms)
    )
    reference_frame = reference_processor.process(reference_raw, str(reference_path))
    reference_resolved = validate_member_plan(reference_atoms, raw_plan, policy=policy)
    minimum_coverage = float(settings["minimum_reference_coverage"])
    if float(reference_resolved["reference_analysis_atom_coverage"]) < minimum_coverage:
        raise OligomerSymmetryError(
            "canonical member reference coverage is below minimum_reference_coverage"
        )
    system_path = Path(str(context["system_manifest_path"]))
    system_manifest = load_json(system_path)
    systems = system_manifest["systems"]
    if not isinstance(systems, list):
        raise OligomerSymmetryError("system manifest systems must be an array")
    basis_plan, basis_report = plan_frame_selection(
        system_manifest, system_path, coordinate_unit,
        settings["frame_selection"],  # type: ignore[arg-type]
        frame_stride=int(settings["frame_stride"]),
        error_type=OligomerSymmetryError,
    )
    projection_plan, projection_report = plan_frame_selection(
        system_manifest, system_path, coordinate_unit,
        settings["projection_frame_selection"],  # type: ignore[arg-type]
        frame_stride=int(settings["projection_frame_stride"]),
        error_type=OligomerSymmetryError,
    )
    inventory = context["input_inventory"]
    if not isinstance(inventory, dict) or not isinstance(inventory.get("entries"), list):
        raise OligomerSymmetryError("compiled input inventory is invalid")
    inventory_by_path = {
        str(row["resolved_path"]): row
        for row in inventory["entries"] if isinstance(row, dict)
    }
    replica_plans = []
    for system in systems:
        if not isinstance(system, dict) or not isinstance(system.get("replicas"), list):
            raise OligomerSymmetryError("system replicas must be an array")
        for replica in system["replicas"]:
            if not isinstance(replica, dict):
                raise OligomerSymmetryError("replica must be an object")
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            topology_format, topology_atoms = read_topology_atoms(topology_path)
            resolved = validate_member_plan(topology_atoms, raw_plan, policy=policy)
            if resolved["alignment_identity_keys"] != reference_resolved["alignment_identity_keys"]:
                raise OligomerSymmetryError(
                    "target topology does not share the canonical member alignment identity"
                )
            replica_plans.append({
                "system_id": str(system["system_id"]),
                "replica_id": str(replica["replica_id"]),
                "replica": replica,
                "topology_path": topology_path,
                "topology_format": topology_format,
                "topology_atoms": topology_atoms,
                "resolved": resolved,
            })
    if not replica_plans:
        raise OligomerSymmetryError("system manifest contains no replicas")
    reference_analysis_keys = reference_resolved["analysis_identity_keys"]
    if not isinstance(reference_analysis_keys, list):
        raise OligomerSymmetryError("reference member analysis identity is invalid")
    common_analysis_keys = {
        tuple(key) for key in reference_analysis_keys
    }
    for plan in replica_plans:
        resolved_keys = plan["resolved"]["analysis_identity_keys"]
        if not isinstance(resolved_keys, list):
            raise OligomerSymmetryError("target member analysis identity is invalid")
        common_analysis_keys.intersection_update(tuple(key) for key in resolved_keys)
    ordered_common_analysis_keys = [
        key for key in reference_analysis_keys if tuple(key) in common_analysis_keys
    ]
    if not ordered_common_analysis_keys:
        raise OligomerSymmetryError(
            "equivalent oligomer topologies share no member analysis atoms"
        )
    reference_analysis_atom_count = len(reference_analysis_keys)
    reference_resolved = restrict_resolved_member_analysis(
        reference_resolved, ordered_common_analysis_keys
    )
    for plan in replica_plans:
        plan["resolved"] = restrict_resolved_member_analysis(
            plan["resolved"], ordered_common_analysis_keys
        )
    reference_members = reference_resolved["members"]
    if not isinstance(reference_members, list):
        raise OligomerSymmetryError("reference member mapping is invalid")
    reference_member = reference_members[0]
    if not isinstance(reference_member, dict):
        raise OligomerSymmetryError("canonical reference member is invalid")
    reference_analysis_indices = reference_member["analysis_atom_indices"]
    if not isinstance(reference_analysis_indices, list):
        raise OligomerSymmetryError("canonical analysis indices are invalid")
    analysis_atoms = [reference_atoms[int(index)] for index in reference_analysis_indices]
    feature_count = 3 * len(analysis_atoms)
    if feature_count > int(settings["maximum_features"]):
        raise OligomerSymmetryError(
            f"member view contains {feature_count} Cartesian features; maximum_features "
            f"is {settings['maximum_features']}"
        )
    if int(settings["component_count"]) > feature_count:
        raise OligomerSymmetryError("component_count exceeds member Cartesian features")
    member_count = int(reference_resolved["member_count"])
    solver = settings["solver"]
    if not isinstance(solver, dict):
        raise OligomerSymmetryError("PCA solver settings are invalid")
    solver_method = str(solver["method"])
    basis_counts = []
    physical_basis_counts = []
    first_passes = []
    weighting = str(settings["basis_weighting"])
    if solver_method == "dense_covariance_v1":
        states = []
        for plan in replica_plans:
            state = CartesianCovariance(feature_count)
            reports = _scan_symmetry_replica(
                system_id=plan["system_id"], replica_id=plan["replica_id"],
                replica=plan["replica"], topology_atoms=plan["topology_atoms"],
                resolved=plan["resolved"], reference_coordinates=reference_frame.coordinates_angstrom,
                reference_member=reference_member, project=project_data, system_path=system_path,
                coordinate_unit=coordinate_unit, time_unit=time_unit,
                periodic_policy=periodic_policy, frame_stride=int(settings["frame_stride"]),
                frame_selection_plan=basis_plan, state=state,
                inventory_by_path=inventory_by_path,
            )
            physical_count = sum(int(row["physical_evaluated_frame_count"]) for row in reports)
            if physical_count < int(settings["minimum_evaluated_frames_per_replica"]):
                raise OligomerSymmetryError(
                    f"{plan['system_id']}/{plan['replica_id']} has {physical_count} physical "
                    "basis frames, below minimum_evaluated_frames_per_replica"
                )
            states.append(state)
            basis_counts.append(state.count)
            physical_basis_counts.append(physical_count)
            first_passes.append(reports)
        if weighting == "frame":
            pooled = CartesianCovariance(feature_count)
            for state in states:
                pooled.merge(state)
            mean = pooled.mean()
            covariance = pooled.population_covariance()
            weights = [state.count / pooled.count for state in states]
        else:
            weights = [1.0 / len(states)] * len(states)
            mean, covariance = mixture_covariance(states, weights)
        solution = principal_components(
            covariance,
            int(settings["component_count"]),
            eigenvalue_tolerance_angstrom2=1.0e-12,
            solver_tolerance=1.0e-10,
            maximum_relative_residual=1.0e-8,
            maximum_iterations=10_000,
        )
        solver_diagnostics = {
            "method": solver_method,
            "sample_count": sum(basis_counts),
            "physical_frame_count": sum(physical_basis_counts),
            "feature_count": feature_count,
        }
    elif solver_method == "randomized_truncated_svd_v1":
        selected_physical = int(basis_report["selected_frame_count"])
        selected_observations = selected_physical * member_count
        sample_elements = selected_observations * feature_count
        if sample_elements > int(solver["maximum_sample_matrix_elements"]):
            raise OligomerSymmetryError(
                f"symmetry-expanded randomized PCA requires {sample_elements} sample "
                "matrix elements; reduce the physical basis-frame budget, not projection coverage"
            )
        samples = np.empty((selected_observations, feature_count), dtype=float)
        cursor = [0]
        replica_slices = []
        for plan in replica_plans:
            start = cursor[0]

            def store(vector: Sequence[float], *_: object) -> None:
                if cursor[0] >= selected_observations:
                    raise OligomerSymmetryError(
                        "symmetry-expanded basis exceeded the planned sample matrix"
                    )
                samples[cursor[0], :] = vector
                cursor[0] += 1

            reports = _scan_symmetry_replica(
                system_id=plan["system_id"], replica_id=plan["replica_id"],
                replica=plan["replica"], topology_atoms=plan["topology_atoms"],
                resolved=plan["resolved"], reference_coordinates=reference_frame.coordinates_angstrom,
                reference_member=reference_member, project=project_data, system_path=system_path,
                coordinate_unit=coordinate_unit, time_unit=time_unit,
                periodic_policy=periodic_policy, frame_stride=int(settings["frame_stride"]),
                frame_selection_plan=basis_plan, vector_sink=store,
                inventory_by_path=inventory_by_path,
            )
            physical_count = sum(int(row["physical_evaluated_frame_count"]) for row in reports)
            if physical_count < int(settings["minimum_evaluated_frames_per_replica"]):
                raise OligomerSymmetryError(
                    f"{plan['system_id']}/{plan['replica_id']} has insufficient physical basis frames"
                )
            count = cursor[0] - start
            basis_counts.append(count)
            physical_basis_counts.append(physical_count)
            replica_slices.append(slice(start, cursor[0]))
            first_passes.append(reports)
        if cursor[0] != selected_observations:
            raise OligomerSymmetryError(
                f"collected {cursor[0]} member observations; planner declared {selected_observations}"
            )
        sample_weights = np.empty(selected_observations, dtype=float)
        if weighting == "frame":
            sample_weights.fill(1.0 / selected_observations)
            weights = [count / selected_observations for count in basis_counts]
        else:
            weights = [1.0 / len(replica_slices)] * len(replica_slices)
            for replica_slice, count in zip(replica_slices, basis_counts):
                sample_weights[replica_slice] = 1.0 / (len(replica_slices) * count)
        mean, solution, solver_diagnostics = randomized_truncated_pca(
            samples, sample_weights, int(settings["component_count"]),
            oversampling=int(solver["oversampling"]),
            power_iterations=int(solver["power_iterations"]),
            power_iteration_schedule=solver["power_iteration_schedule"],
            random_seed=int(solver["random_seed"]),
            eigenvalue_tolerance_angstrom2=1.0e-12,
            maximum_relative_residual=float(solver["maximum_relative_residual"]),
        )
        solver_diagnostics.update({
            "sample_matrix_elements": sample_elements,
            "sample_matrix_bytes_float64": sample_elements * 8,
            "physical_frame_count": sum(physical_basis_counts),
        })
    else:
        raise OligomerSymmetryError("unsupported PCA solver for oligomer symmetry view")

    issues = [row for row in context.get("issues", []) if isinstance(row, dict)]
    excluded_reference_analysis_atoms = (
        reference_analysis_atom_count - len(ordered_common_analysis_keys)
    )
    if excluded_reference_analysis_atoms:
        issues.append(issue_record(
            "warning", "SYMMETRY_COMMON_ANALYSIS_INTERSECTION_EXCLUDES_VARIANT_ATOMS",
            str(source),
            f"cross-topology oligomer-member analysis uses "
            f"{len(ordered_common_analysis_keys)} of {reference_analysis_atom_count} "
            "canonical reference-member atoms; variant-specific atoms are excluded",
        ))
    if len(solution.components) < int(settings["component_count"]):
        issues.append(issue_record(
            "warning", "NUMERICAL_RANK_LIMIT", str(source),
            f"requested {settings['component_count']} components but returned {len(solution.components)}",
        ))
    replicas_out: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    all_projection_records = []
    replica_contributions = []
    for plan, basis_count, physical_basis_count, basis_weight in zip(
        replica_plans, basis_counts, physical_basis_counts, weights
    ):
        reports = _scan_symmetry_replica(
            system_id=plan["system_id"], replica_id=plan["replica_id"],
            replica=plan["replica"], topology_atoms=plan["topology_atoms"],
            resolved=plan["resolved"], reference_coordinates=reference_frame.coordinates_angstrom,
            reference_member=reference_member, project=project_data, system_path=system_path,
            coordinate_unit=coordinate_unit, time_unit=time_unit,
            periodic_policy=periodic_policy, frame_stride=int(settings["projection_frame_stride"]),
            frame_selection_plan=projection_plan, mean=mean, solution=solution,
            inventory_by_path=inventory_by_path,
        )
        scores = _projection_values(reports, len(solution.components))
        summaries = [sample_summary(values) for values in scores]
        for segment in reports:
            projections = segment["projections"]
            if isinstance(projections, list):
                all_projection_records.extend({
                    "system_id": plan["system_id"],
                    "replica_id": plan["replica_id"],
                    "segment_id": segment["segment_id"],
                    **row,
                } for row in projections if isinstance(row, dict))
        topology_path = plan["topology_path"]
        topology_inventory = inventory_by_path.get(str(topology_path), {})
        physical_projection_count = sum(
            int(row["physical_evaluated_frame_count"]) for row in reports
        )
        observation_projection_count = sum(
            int(row["evaluated_member_observation_count"]) for row in reports
        )
        replicas_out[plan["system_id"]].append({
            "replica_id": plan["replica_id"],
            "topology_path": str(topology_path),
            "topology_format": plan["topology_format"],
            "topology_sha256": topology_inventory.get("sha256"),
            "topology_atom_count": len(plan["topology_atoms"]),
            "physical_evaluated_frame_count": physical_projection_count,
            "evaluated_frame_count": observation_projection_count,
            "projection_evaluated_frame_count": observation_projection_count,
            "basis_evaluated_frame_count": basis_count,
            "physical_basis_evaluated_frame_count": physical_basis_count,
            "member_count": member_count,
            "basis_weight": basis_weight,
            "member_mapping": plan["resolved"],
            "projection_summaries_angstrom": summaries,
            "segments": reports,
        })
        replica_contributions.append({
            "system_id": plan["system_id"],
            "replica_id": plan["replica_id"],
            "physical_basis_frame_count": physical_basis_count,
            "member_observation_count": basis_count,
            "basis_weight": basis_weight,
        })
    systems_out = []
    reference_system = str(contract["reference_system"])
    reference_means = None
    for system_id, replicas in replicas_out.items():
        segments = [segment for replica in replicas for segment in replica["segments"]]
        summaries = [
            sample_summary(values)
            for values in _projection_values(segments, len(solution.components))
        ]
        if system_id == reference_system:
            reference_means = [row["mean"] for row in summaries]
        systems_out.append({
            "system_id": system_id,
            "physical_frame_pooled_projection_summaries_angstrom": summaries,
            "frame_pooled_projection_summaries_angstrom": summaries,
            "replicas": replicas,
        })
    if reference_means is None:
        raise OligomerSymmetryError("reference system produced no member projections")
    for system in systems_out:
        system["projection_mean_difference_from_reference_angstrom"] = [
            None if row["mean"] is None or reference is None
            else float(row["mean"]) - float(reference)
            for row, reference in zip(
                system["frame_pooled_projection_summaries_angstrom"], reference_means
            )
        ]
    if int(basis_report["selected_frame_count"]) < int(basis_report["source_frame_count"]):
        issues.append(issue_record(
            "warning", "FRAME_SUBSAMPLING", str(source),
            f"oligomer PCA fitted {basis_report['selected_frame_count']} of "
            f"{basis_report['source_frame_count']} physical source frames, producing "
            f"{int(basis_report['selected_frame_count']) * member_count} member observations",
        ))
    error_count = sum(row.get("severity") == "error" for row in issues)
    warning_count = sum(row.get("severity") == "warning" for row in issues)
    physical_source = int(projection_report["source_frame_count"])
    physical_selected = int(projection_report["selected_frame_count"])
    return {
        "module_id": "common_pca",
        "analysis_mode": "symmetry_expanded_equivalent_oligomer_members_v1",
        "technical_status": "failed" if error_count else "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": context["system_manifest_path"],
        "system_manifest_sha256": context["system_manifest_sha256"],
        "contract_signature_sha256": context["contract_signature_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "module_contract_sha256": project_module_contract_sha256("common_pca", source),
        "reference": {
            "path": str(reference_path),
            "format": reference_format,
            "sha256": sha256_file(reference_path) if hash_content else None,
            "atom_count": len(reference_atoms),
        },
        "reference_system_id": reference_system,
        "settings": dict(settings),
        "frame_selection": basis_report,
        "basis_frame_selection": basis_report,
        "projection_frame_selection": projection_report,
        "common_atom_policy": policy,
        "periodic_coordinate_policy": periodic_policy,
        "sampling_mode": contract["sampling_mode"],
        "frame_axis_kind": "sample_index" if contract["sampling_mode"] == "AI_ENSEMBLE" else "physical_time",
        "time_unit": time_unit,
        "symmetry_expansion": raw_plan,
        "common_member_analysis_identity": {
            "policy": "exact ordered intersection across every resolved topology",
            "reference_analysis_atom_count_per_member": reference_analysis_atom_count,
            "common_analysis_atom_count_per_member": len(ordered_common_analysis_keys),
            "excluded_reference_analysis_atom_count_per_member": excluded_reference_analysis_atoms,
            "reference_coverage": (
                len(ordered_common_analysis_keys) / reference_analysis_atom_count
            ),
            "identity_keys": ordered_common_analysis_keys,
        },
        "observation_accounting": {
            "source_physical_frame_count": physical_source,
            "selected_physical_frame_count": physical_selected,
            "member_count": member_count,
            "symmetry_expanded_observation_count": physical_selected * member_count,
            "basis_selected_physical_frame_count": int(basis_report["selected_frame_count"]),
            "basis_member_observation_count": int(basis_report["selected_frame_count"]) * member_count,
            "independent_sampling_unit": "original simulation replica and physical time block",
            "member_observations_are_independent_replicas": False,
        },
        "basis": {
            "basis_weighting": weighting,
            "replica_count": len(replica_plans),
            "physical_evaluated_frame_count": sum(physical_basis_counts),
            "evaluated_frame_count": sum(basis_counts),
            "member_observation_count": sum(basis_counts),
            "replica_contributions": replica_contributions,
            "solver_diagnostics": solver_diagnostics,
            "pca": _pca_payload(mean, solution, analysis_atoms),
        },
        "paired_member_correlation": paired_member_score_correlations(
            all_projection_records,
            minimum_paired_frames=max(3, int(settings["minimum_evaluated_frames_per_replica"])),
        ),
        "systems": systems_out,
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": issues,
        "limitations": [
            "Symmetry expansion multiplies member observations, not independent physical frames or simulation replicas.",
            "Each member is independently aligned; whole-assembly relative translation and rotation are intentionally removed from the pooled member conformation view.",
            "Whole-assembly PCA and symmetry-expanded member PCA answer different questions and both remain reported.",
            "Paired-member correlations use matched physical frames and do not establish causality, direct interaction, or mechanism.",
            "Basis subsampling is computational and does not establish scientific convergence.",
            "Technical completion does not establish equilibration, metastability, kinetics, mechanism, or scientific validity.",
        ],
    }
