"""Backbone and complete conventional side-chain circular distributions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from .atom_mapping import AtomMappingError, AtomRecord, read_topology_atoms
from .context import compile_project_context_file
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .frame_sampling import (
    frame_selected,
    integer_stride_indices,
    reader_frame_indices,
    source_frame_count,
)
from .geometry import distance3
from .hydrogen_bond_chemistry import NUCLEIC_RESIDUES
from .manifests import ManifestValidationError, load_json, resolve_manifest_path
from .periodic import PeriodicFrameProcessor, PeriodicReconstructionError
from .trajectory_contracts import (
    TrajectoryContractError,
    frame_axis_value,
    normalize_segment_axis,
)


class DihedralAnalysisError(ValueError):
    """Raised when a declared dihedral analysis cannot be evaluated safely."""


def dihedral_degrees(
    first: Sequence[float], second: Sequence[float],
    third: Sequence[float], fourth: Sequence[float],
) -> float:
    """Return the signed torsion in degrees on [-180, 180)."""

    b0 = tuple(first[index] - second[index] for index in range(3))
    b1 = tuple(third[index] - second[index] for index in range(3))
    b2 = tuple(fourth[index] - third[index] for index in range(3))
    norm = math.sqrt(sum(value * value for value in b1))
    if norm <= 1.0e-15:
        raise DihedralAnalysisError("dihedral central bond has zero length")
    axis = tuple(value / norm for value in b1)
    projection0 = sum(b0[index] * axis[index] for index in range(3))
    projection2 = sum(b2[index] * axis[index] for index in range(3))
    v = tuple(b0[index] - projection0 * axis[index] for index in range(3))
    w = tuple(b2[index] - projection2 * axis[index] for index in range(3))
    v_norm = math.sqrt(sum(value * value for value in v))
    w_norm = math.sqrt(sum(value * value for value in w))
    if min(v_norm, w_norm) <= 1.0e-15:
        raise DihedralAnalysisError("dihedral contains collinear bonds")
    x = sum(v[index] * w[index] for index in range(3))
    cross = (
        axis[1] * v[2] - axis[2] * v[1],
        axis[2] * v[0] - axis[0] * v[2],
        axis[0] * v[1] - axis[1] * v[0],
    )
    y = sum(cross[index] * w[index] for index in range(3))
    angle = math.degrees(math.atan2(y, x))
    return angle if angle < 180.0 else angle - 360.0


def circular_summary(values_degrees: Sequence[float], bin_count: int) -> Dict[str, object]:
    if not values_degrees:
        return {
            "count": 0, "mean_angle_degrees": None, "mean_resultant_length": None,
            "circular_variance": None, "histogram": [],
        }
    radians = [math.radians(value) for value in values_degrees]
    cosine = sum(math.cos(value) for value in radians) / len(radians)
    sine = sum(math.sin(value) for value in radians) / len(radians)
    resultant = math.sqrt(cosine * cosine + sine * sine)
    mean = math.degrees(math.atan2(sine, cosine))
    width = 360.0 / bin_count
    counts = [0 for _ in range(bin_count)]
    for value in values_degrees:
        index = min(bin_count - 1, int((value + 180.0) / width))
        counts[index] += 1
    return {
        "count": len(values_degrees),
        "mean_angle_degrees": mean,
        "mean_resultant_length": resultant,
        "circular_variance": 1.0 - resultant,
        "histogram": [
            {
                "lower_degrees": -180.0 + index * width,
                "upper_degrees": -180.0 + (index + 1) * width,
                "count": count,
                "fraction": count / len(values_degrees),
            }
            for index, count in enumerate(counts)
        ],
    }


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("dihedral_distributions") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict):
        raise DihedralAnalysisError("definitions.dihedral_distributions must be an object")
    required = {
        "angle_types", "frame_stride", "histogram_bins",
        "maximum_reference_peptide_bond_angstrom", "maximum_observations",
    }
    optional = {"maximum_reference_phosphodiester_bond_angstrom"}
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required | optional))
    if missing:
        raise DihedralAnalysisError("dihedral settings missing: " + ", ".join(missing))
    if unknown:
        raise DihedralAnalysisError("dihedral settings contain unknown fields: " + ", ".join(unknown))
    angles = raw["angle_types"]
    allowed = {
        "phi", "psi", "omega", "chi1", "chi2", "chi3", "chi4", "chi5",
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi",
        "nu0", "nu1", "nu2", "nu3", "nu4",
    }
    if (
        not isinstance(angles, list) or not angles or any(value not in allowed for value in angles)
        or len(set(angles)) != len(angles)
    ):
        raise DihedralAnalysisError(
            "angle_types must contain unique protein and/or nucleic-acid torsion names"
        )
    frame_stride = raw["frame_stride"]
    bins = raw["histogram_bins"]
    maximum_observations = raw["maximum_observations"]
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (frame_stride, bins, maximum_observations)):
        raise DihedralAnalysisError("frame_stride, histogram_bins, and maximum_observations must be positive integers")
    if bins < 4 or bins > 360:
        raise DihedralAnalysisError("histogram_bins must be between 4 and 360")
    bond = raw["maximum_reference_peptide_bond_angstrom"]
    if isinstance(bond, bool) or not isinstance(bond, (int, float)) or not math.isfinite(float(bond)) or float(bond) <= 0.0:
        raise DihedralAnalysisError("maximum_reference_peptide_bond_angstrom must be finite and positive")
    phosphodiester = raw.get("maximum_reference_phosphodiester_bond_angstrom", 2.2)
    if (
        isinstance(phosphodiester, bool)
        or not isinstance(phosphodiester, (int, float))
        or not math.isfinite(float(phosphodiester))
        or float(phosphodiester) <= 0.0
    ):
        raise DihedralAnalysisError(
            "maximum_reference_phosphodiester_bond_angstrom must be finite and positive"
        )
    return {
        "angle_types": list(angles), "frame_stride": frame_stride,
        "histogram_bins": bins,
        "maximum_reference_peptide_bond_angstrom": float(bond),
        "maximum_reference_phosphodiester_bond_angstrom": float(phosphodiester),
        "maximum_observations": maximum_observations,
    }


def _residue_atoms(atoms: Sequence[AtomRecord]) -> List[Tuple[Tuple[object, ...], Dict[str, int]]]:
    residues: List[Tuple[Tuple[object, ...], Dict[str, int]]] = []
    current_key = None
    current: Dict[str, int] = {}
    for atom in atoms:
        key = (atom.chain_id, atom.residue_number, atom.insertion_code, atom.residue_name)
        if key != current_key:
            if current_key is not None and {"N", "CA", "C"}.issubset(current):
                residues.append((current_key, current))
            current_key = key
            current = {}
        name = atom.atom_name.upper()
        if name not in current and not atom.altloc:
            current[name] = atom.atom_index
    if current_key is not None and {"N", "CA", "C"}.issubset(current):
        residues.append((current_key, current))
    return residues


def _nucleic_residue_atoms(
    atoms: Sequence[AtomRecord],
) -> List[Tuple[Tuple[object, ...], Dict[str, int]]]:
    """Return ordered standard nucleic-acid residues with normalized prime names."""

    residues: List[Tuple[Tuple[object, ...], Dict[str, int]]] = []
    current_key = None
    current: Dict[str, int] = {}
    for atom in atoms:
        key = (atom.chain_id, atom.residue_number, atom.insertion_code, atom.residue_name)
        if key != current_key:
            if (
                current_key is not None
                and str(current_key[3]).upper() in NUCLEIC_RESIDUES
                and "C1'" in current
            ):
                residues.append((current_key, current))
            current_key = key
            current = {}
        name = atom.atom_name.upper().replace("*", "'")
        if name not in current and not atom.altloc:
            current[name] = atom.atom_index
    if (
        current_key is not None
        and str(current_key[3]).upper() in NUCLEIC_RESIDUES
        and "C1'" in current
    ):
        residues.append((current_key, current))
    return residues


_NUCLEIC_ANGLE_TYPES = (
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi",
    "nu0", "nu1", "nu2", "nu3", "nu4",
)
_PURINE_RESIDUES = frozenset({
    "A", "G", "DA", "DG", "RA", "RG", "ADE", "GUA", "8OG", "8OX", "OX3",
})


def _torsion_specs(
    atoms: Sequence[AtomRecord], reference_coordinates: Sequence[Sequence[float]],
    settings: Mapping[str, object],
) -> List[Dict[str, object]]:
    residues = _residue_atoms(atoms)
    specs = []
    sidechain_definitions = {
        "ARG": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ"), ("CD", "NE", "CZ", "NH1")),
        "ASN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
        "ASP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
        "CYS": (("N", "CA", "CB", "SG"),),
        "GLN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")),
        "GLU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")),
        "HIS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")),
        "HID": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")),
        "HIE": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")),
        "HIP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")),
        "ILE": (("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")),
        "LEU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
        "LYS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")),
        "MET": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE")),
        "PHE": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
        "PRO": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
        "SER": (("N", "CA", "CB", "OG"),),
        "THR": (("N", "CA", "CB", "OG1"),),
        "TRP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
        "TYR": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
        "VAL": (("N", "CA", "CB", "CG1"),),
    }
    angles = set(settings["angle_types"])
    for index, (identity, names) in enumerate(residues):
        residue = {
            "chain_id": identity[0], "residue_number": identity[1],
            "insertion_code": identity[2], "residue_name": identity[3],
        }
        residue_name = str(identity[3]).upper()
        for chi_index, atom_names in enumerate(sidechain_definitions.get(residue_name, ()), start=1):
            angle_type = f"chi{chi_index}"
            if angle_type in angles and all(name in names for name in atom_names):
                specs.append({
                    "angle_type": angle_type, "residue": residue,
                    "atom_indices": [names[name] for name in atom_names],
                })
        if index > 0:
            previous_identity, previous = residues[index - 1]
            linked = (
                previous_identity[0] == identity[0]
                and distance3(reference_coordinates[previous["C"]], reference_coordinates[names["N"]])
                <= float(settings["maximum_reference_peptide_bond_angstrom"])
            )
            if linked and "phi" in angles:
                specs.append({
                    "angle_type": "phi", "residue": residue,
                    "atom_indices": [previous["C"], names["N"], names["CA"], names["C"]],
                })
        if index + 1 < len(residues):
            next_identity, following = residues[index + 1]
            linked = (
                next_identity[0] == identity[0]
                and distance3(reference_coordinates[names["C"]], reference_coordinates[following["N"]])
                <= float(settings["maximum_reference_peptide_bond_angstrom"])
            )
            if linked and "psi" in angles:
                specs.append({
                    "angle_type": "psi", "residue": residue,
                    "atom_indices": [names["N"], names["CA"], names["C"], following["N"]],
                })
            if linked and "omega" in angles:
                specs.append({
                    "angle_type": "omega", "residue": residue,
                    "atom_indices": [names["CA"], names["C"], following["N"], following["CA"]],
                })
    nucleic_residues = _nucleic_residue_atoms(atoms)
    local_nucleic = {
        "beta": ("P", "O5'", "C5'", "C4'"),
        "gamma": ("O5'", "C5'", "C4'", "C3'"),
        "delta": ("C5'", "C4'", "C3'", "O3'"),
        "nu0": ("C4'", "O4'", "C1'", "C2'"),
        "nu1": ("O4'", "C1'", "C2'", "C3'"),
        "nu2": ("C1'", "C2'", "C3'", "C4'"),
        "nu3": ("C2'", "C3'", "C4'", "O4'"),
        "nu4": ("C3'", "C4'", "O4'", "C1'"),
    }
    for index, (identity, names) in enumerate(nucleic_residues):
        residue = {
            "chain_id": identity[0], "residue_number": identity[1],
            "insertion_code": identity[2], "residue_name": identity[3],
        }
        for angle_type, atom_names in local_nucleic.items():
            if angle_type in angles and all(name in names for name in atom_names):
                specs.append({
                    "angle_type": angle_type,
                    "residue": residue,
                    "atom_indices": [names[name] for name in atom_names],
                })
        residue_name = str(identity[3]).upper()
        chi_names = (
            ("O4'", "C1'", "N9", "C4")
            if residue_name in _PURINE_RESIDUES
            else ("O4'", "C1'", "N1", "C2")
        )
        if "chi" in angles and all(name in names for name in chi_names):
            specs.append({
                "angle_type": "chi",
                "residue": residue,
                "atom_indices": [names[name] for name in chi_names],
            })
        if index > 0:
            previous_identity, previous = nucleic_residues[index - 1]
            linked = (
                previous_identity[0] == identity[0]
                and "O3'" in previous
                and "P" in names
                and distance3(
                    reference_coordinates[previous["O3'"]],
                    reference_coordinates[names["P"]],
                ) <= float(settings["maximum_reference_phosphodiester_bond_angstrom"])
            )
            alpha_names = ("O3'", "P", "O5'", "C5'")
            if linked and "alpha" in angles and all(name in names for name in alpha_names[1:]):
                specs.append({
                    "angle_type": "alpha",
                    "residue": residue,
                    "atom_indices": [
                        previous["O3'"], names["P"], names["O5'"], names["C5'"]
                    ],
                })
        if index + 1 < len(nucleic_residues):
            next_identity, following = nucleic_residues[index + 1]
            linked = (
                next_identity[0] == identity[0]
                and "O3'" in names
                and "P" in following
                and distance3(
                    reference_coordinates[names["O3'"]],
                    reference_coordinates[following["P"]],
                ) <= float(settings["maximum_reference_phosphodiester_bond_angstrom"])
            )
            if linked and "epsilon" in angles and all(
                name in names for name in ("C4'", "C3'", "O3'")
            ):
                specs.append({
                    "angle_type": "epsilon",
                    "residue": residue,
                    "atom_indices": [
                        names["C4'"], names["C3'"], names["O3'"], following["P"]
                    ],
                })
            if linked and "zeta" in angles and all(
                name in names for name in ("C3'", "O3'")
            ) and "O5'" in following:
                specs.append({
                    "angle_type": "zeta",
                    "residue": residue,
                    "atom_indices": [
                        names["C3'"], names["O3'"], following["P"], following["O5'"]
                    ],
                })
    return specs


def dihedral_distributions_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    context = compile_project_context_file(source, hash_content=hash_content)
    system_path = Path(context["system_manifest_path"])
    system = load_json(system_path)
    coordinate_unit = str(project["coordinate_unit"])
    output_time_unit = project.get("time_unit")
    periodic_policy = str(project["periodic_coordinate_policy"])
    groups: Dict[Tuple[str, str, str, int, str, str], List[float]] = {}
    observations = 0
    issues = [issue for issue in context.get("warnings", []) if isinstance(issue, dict)]
    segment_reports = []
    for raw_system in system["systems"]:
        system_id = str(raw_system["system_id"])
        for replica in raw_system["replicas"]:
            replica_id = str(replica["replica_id"])
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology_path)
            processor = PeriodicFrameProcessor.from_replica(
                project, replica, system_path, len(atoms)
            )
            raw_reference = next(iter_coordinate_frames(topology_path, coordinate_unit))
            reference = processor.process(raw_reference, str(topology_path))
            specs = _torsion_specs(atoms, reference.coordinates_angstrom, settings)
            if (
                not specs
                and _nucleic_residue_atoms(atoms)
                and not set(settings["angle_types"]).intersection(_NUCLEIC_ANGLE_TYPES)
            ):
                inferred_settings = dict(settings)
                inferred_settings["angle_types"] = list(_NUCLEIC_ANGLE_TYPES)
                specs = _torsion_specs(
                    atoms, reference.coordinates_angstrom, inferred_settings
                )
                issues.append({
                    "severity": "warning",
                    "code": "LEGACY_GENERIC_NUCLEIC_TORSIONS_INFERRED",
                    "location": f"{system_id}/{replica_id}",
                    "message": (
                        "The legacy generic project requested only protein torsions for a "
                        "DNA/RNA-only topology; standard alpha/beta/gamma/delta/epsilon/zeta, "
                        "glycosidic chi, and nu0-nu4 sugar torsions were inferred."
                    ),
                })
            if not specs:
                raise DihedralAnalysisError(f"{system_id}/{replica_id} produced no requested torsion definitions")
            reconstruction_atom_indices = tuple(sorted({
                int(index) for spec in specs for index in spec["atom_indices"]
            }))
            for segment in replica["segments"]:
                segment_id = str(segment["segment_id"])
                trajectory_path = resolve_manifest_path(str(segment["trajectory"]), system_path)
                source_frames = source_frame_count(
                    trajectory_path, coordinate_unit, error_type=DihedralAnalysisError
                )
                selected_indices = (
                    integer_stride_indices(
                        source_frames,
                        int(settings["frame_stride"]),
                        error_type=DihedralAnalysisError,
                    )
                    if int(settings["frame_stride"]) > 1 else None
                )
                axis = normalize_segment_axis(segment, str(output_time_unit) if output_time_unit else None)
                evaluated_frames = 0
                decoded_frames = 0
                periodic_frames = 0
                processor.begin_segment(
                    bool(segment.get("continuous_with_previous", False))
                )
                for raw_frame in iter_coordinate_frames(
                    trajectory_path,
                    coordinate_unit,
                    reader_frame_indices(selected_indices, periodic_policy),
                ):
                    frame = processor.process(
                        raw_frame,
                        f"{system_id}/{replica_id}/{segment_id}/frame-{raw_frame.frame_index}",
                        reconstruction_atom_indices,
                    )
                    if frame.atom_count != len(atoms):
                        raise DihedralAnalysisError("trajectory/topology atom count mismatch")
                    decoded_frames += 1
                    periodic_frames += int(frame.periodic_cell_present)
                    if not frame_selected(
                        frame.frame_index,
                        selected_indices,
                        int(settings["frame_stride"]),
                    ):
                        continue
                    evaluated_frames += 1
                    frame_axis_value(axis, frame.frame_index)
                    for spec in specs:
                        indices = spec["atom_indices"]
                        coordinates = [frame.coordinates_angstrom[index] for index in indices]
                        try:
                            angle = dihedral_degrees(*coordinates)
                        except DihedralAnalysisError:
                            continue
                        residue = spec["residue"]
                        key = (
                            system_id, replica_id, str(residue["chain_id"]),
                            int(residue["residue_number"]), str(residue["insertion_code"]),
                            str(spec["angle_type"]),
                        )
                        groups.setdefault(key, []).append(angle)
                        observations += 1
                        if observations > int(settings["maximum_observations"]):
                            raise DihedralAnalysisError("maximum_observations gate exceeded")
                if periodic_frames and periodic_policy == "allow_wrapped_diagnostic":
                    issues.append({
                        "severity": "warning", "code": "PERIODIC_COORDINATES_NOT_UNWRAPPED",
                        "location": f"{system_id}/{replica_id}/{segment_id}",
                        "message": f"{periodic_frames} periodic frames were evaluated without molecule reconstruction",
                    })
                segment_reports.append({
                    "system_id": system_id, "replica_id": replica_id,
                    "segment_id": segment_id, "torsion_definition_count": len(specs),
                    "source_frame_count": source_frames,
                    "decoded_frame_count": decoded_frames,
                    "evaluated_frame_count": evaluated_frames,
                    "periodic_cell_frame_count": periodic_frames,
                })
    summaries = []
    for key in sorted(groups):
        system_id, replica_id, chain_id, residue_number, insertion_code, angle_type = key
        summaries.append({
            "system_id": system_id, "replica_id": replica_id,
            "chain_id": chain_id, "residue_number": residue_number,
            "insertion_code": insertion_code, "angle_type": angle_type,
            **circular_summary(groups[key], int(settings["histogram_bins"])),
        })
    return {
        "module_id": "dihedral_distributions",
        "technical_status": "complete", "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "contract_signature_sha256": context["contract_signature_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content, "settings": settings,
        "segment_reports": segment_reports, "observation_count": observations,
        "series_count": len(summaries), "circular_summaries": summaries,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Peptide adjacency is inferred only when consecutive topology residues pass the declared reference C-N distance gate.",
            "Nucleic-acid alpha/epsilon/zeta adjacency is inferred only when consecutive same-chain residues pass the declared reference O3'-P distance gate.",
            "Nucleic-acid torsions include alpha-zeta, glycosidic chi, and the five endocyclic nu torsions; pseudorotation phase/amplitude require a derived sugar-pucker analysis.",
            "Chi1-chi5 use explicit conventional protein residue atom-name definitions; nonstandard residues require a future declared torsion adapter.",
            "Periodic production analysis requires explicit connectivity and make_whole or unwrap_continuous; wrapped analysis remains diagnostic.",
            "Circular distributions do not by themselves establish state populations, convergence, mechanism, or statistical significance.",
        ],
    }


def dihedral_distributions_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return dihedral_distributions_project(project_path, hash_content=hash_content)
    except (
        ManifestValidationError, DihedralAnalysisError, AtomMappingError,
        CoordinateReadError, PeriodicReconstructionError, TrajectoryContractError, OSError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "dihedral_distributions", "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [{"severity": "error", "code": "DIHEDRAL_INVALID", "message": message} for message in messages],
        }
