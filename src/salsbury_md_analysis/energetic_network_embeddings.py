"""Experimental Cowan/Thayer-style residue interaction-energy networks.

The implementation reproduces the cpptraj ``pairwise`` energy definition used
by Cowan and Thayer from Amber, CHARMM, or standard OpenMM nonbonded parameters,
without invoking cpptraj. It intentionally analyzes only common protein
residues and retains only the electrostatic heat-kernel/Wasserstein workflow
described in that work.
"""

from __future__ import annotations

import math
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.stats import wasserstein_distance

from .atom_mapping import AtomMappingError, AtomRecord, read_topology_atoms
from .chemical_identity import PROTEIN_RESIDUES
from .context import compile_project_context_file
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .frame_sampling import (
    frame_selected, normalize_frame_selection, plan_frame_selection,
    reader_frame_indices,
)
from .manifests import (
    ManifestValidationError, load_json, resolve_manifest_path, sha256_file,
)
from .periodic import PeriodicFrameProcessor, PeriodicReconstructionError
from .periodic import load_connectivity
from .trajectory_contracts import require_periodic_policy
from .validation import positive_integer


class EnergeticNetworkError(ValueError):
    """Raised when the energetic-network contract is invalid."""


AMBER_CHARGE_SCALE = 18.2223
CPPTRAJ_COULOMB_FACTOR_KCAL_ANGSTROM_PER_MOL_E2 = 332.0522173


@dataclass(frozen=True)
class PairwiseParameters:
    parameter_source: str
    parameter_files: Tuple[str, ...]
    atom_names: Tuple[str, ...]
    residue_names: Tuple[str, ...]
    residue_indices: np.ndarray
    charges_e: np.ndarray
    atom_type_indices: np.ndarray
    nonbonded_parameter_indices: np.ndarray
    lennard_jones_a: np.ndarray
    lennard_jones_b: np.ndarray
    atom_sigma_angstrom: Optional[np.ndarray]
    atom_epsilon_kcal_per_mol: Optional[np.ndarray]
    pair_parameter_overrides: Mapping[Tuple[int, int], Tuple[float, float, float]]
    excluded_pairs: frozenset[Tuple[int, int]]
    atom_count: int
    atom_type_count: int
    bond_count: int
    nbfix_pair_type_count: int


# Backward-compatible public name retained for callers of the first Amber-only
# experimental implementation.
AmberPairwiseParameters = PairwiseParameters


_FORMAT_RE = re.compile(
    r"%FORMAT\(\s*(\d+)([A-Za-z])(\d+)(?:\.\d+)?\s*\)", re.IGNORECASE
)


def _amber_sections(path: Path) -> Dict[str, List[object]]:
    """Read fixed-width Amber topology sections without an Amber dependency."""

    source = Path(path).expanduser().resolve(strict=False)
    sections: Dict[str, List[object]] = {}
    current: Optional[str] = None
    kind: Optional[str] = None
    width: Optional[int] = None
    try:
        with source.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if stripped.startswith("%FLAG "):
                    current = stripped[6:].strip().upper()
                    sections.setdefault(current, [])
                    kind = None
                    width = None
                    continue
                if stripped.startswith("%FORMAT"):
                    match = _FORMAT_RE.fullmatch(stripped)
                    if match is None:
                        raise EnergeticNetworkError(
                            f"unsupported Amber format at line {line_number}: {stripped}"
                        )
                    kind = match.group(2).upper()
                    width = int(match.group(3))
                    continue
                if not stripped or current is None or kind is None or width is None:
                    continue
                fields = [
                    line[index:index + width]
                    for index in range(0, len(line.rstrip("\n")), width)
                ]
                for field in fields:
                    text = field.strip()
                    if not text:
                        continue
                    try:
                        if kind == "A":
                            value: object = text
                        elif kind == "I":
                            value = int(text)
                        elif kind in {"E", "F", "D"}:
                            value = float(text.replace("D", "E").replace("d", "e"))
                        else:
                            raise EnergeticNetworkError(
                                f"unsupported Amber field kind {kind!r} in {current}"
                            )
                    except ValueError as exc:
                        raise EnergeticNetworkError(
                            f"Amber {current} contains a malformed value {text!r}"
                        ) from exc
                    sections[current].append(value)
    except (OSError, UnicodeError) as exc:
        raise EnergeticNetworkError(f"{source}: {exc}") from exc
    return sections


def _excluded_pairs_from_bonds(
    atom_count: int, bonds: Sequence[Tuple[int, int]],
) -> frozenset[Tuple[int, int]]:
    """Match cpptraj SetupExcluded(..., TgtDist=4) from a bond graph."""

    adjacency: List[Set[int]] = [set() for _ in range(atom_count)]
    normalized: Set[Tuple[int, int]] = set()
    for first_raw, second_raw in bonds:
        first, second = int(first_raw), int(second_raw)
        if min(first, second) < 0 or max(first, second) >= atom_count or first == second:
            raise EnergeticNetworkError("bond graph contains an invalid bond")
        pair = (min(first, second), max(first, second))
        normalized.add(pair)
        adjacency[first].add(second)
        adjacency[second].add(first)
    if atom_count > 1 and not normalized:
        raise EnergeticNetworkError("parameter source contains no bond graph for exclusions")
    excluded: Set[Tuple[int, int]] = set()
    for atom_index in range(atom_count):
        frontier = {atom_index}
        visited = {atom_index}
        for _ in range(3):
            next_frontier = {
                neighbor
                for current in frontier
                for neighbor in adjacency[current]
                if neighbor not in visited
            }
            for partner in next_frontier:
                excluded.add((min(atom_index, partner), max(atom_index, partner)))
            visited.update(next_frontier)
            frontier = next_frontier
    return frozenset(excluded)


def read_amber_pairwise_parameters(path: Path) -> PairwiseParameters:
    """Read the Amber fields required for cpptraj-style pair energies."""

    sections = _amber_sections(path)
    required = {
        "POINTERS", "ATOM_NAME", "CHARGE", "ATOM_TYPE_INDEX",
        "NONBONDED_PARM_INDEX", "RESIDUE_LABEL", "RESIDUE_POINTER",
        "LENNARD_JONES_ACOEF", "LENNARD_JONES_BCOEF",
        "BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN",
    }
    missing = sorted(required.difference(sections))
    if missing:
        raise EnergeticNetworkError(
            "Amber parameter topology lacks required pairwise fields: "
            + ", ".join(missing)
        )
    pointers = [int(value) for value in sections["POINTERS"]]
    if len(pointers) < 2 or pointers[0] <= 0 or pointers[1] <= 0:
        raise EnergeticNetworkError("Amber POINTERS must declare NATOM and NTYPES")
    atom_count, atom_type_count = pointers[0], pointers[1]

    def exact(name: str, count: int) -> List[object]:
        values = sections[name]
        if len(values) != count:
            raise EnergeticNetworkError(
                f"Amber {name} contains {len(values)} values; expected {count}"
            )
        return values

    atom_names = tuple(str(value).strip() for value in exact("ATOM_NAME", atom_count))
    charges = np.asarray(exact("CHARGE", atom_count), dtype=float) / AMBER_CHARGE_SCALE
    atom_types = np.asarray(exact("ATOM_TYPE_INDEX", atom_count), dtype=np.int64) - 1
    if not np.isfinite(charges).all() or np.any(atom_types < 0) or np.any(
        atom_types >= atom_type_count
    ):
        raise EnergeticNetworkError("Amber charges or atom-type indices are invalid")
    nb_index = np.asarray(
        exact("NONBONDED_PARM_INDEX", atom_type_count * atom_type_count),
        dtype=np.int64,
    ) - 1
    if np.any(nb_index < 0):
        raise EnergeticNetworkError(
            "Amber topology uses unsupported negative 10-12 nonbonded parameter indices"
        )
    lj_a = np.asarray(sections["LENNARD_JONES_ACOEF"], dtype=float)
    lj_b = np.asarray(sections["LENNARD_JONES_BCOEF"], dtype=float)
    if (
        len(lj_a) != len(lj_b) or not len(lj_a)
        or int(np.max(nb_index)) >= len(lj_a)
        or not np.isfinite(lj_a).all() or not np.isfinite(lj_b).all()
    ):
        raise EnergeticNetworkError("Amber Lennard-Jones tables are malformed")

    residue_names = tuple(str(value).strip() for value in sections["RESIDUE_LABEL"])
    pointers_raw = [int(value) for value in sections["RESIDUE_POINTER"]]
    if (
        not residue_names or len(residue_names) != len(pointers_raw)
        or pointers_raw[0] != 1 or any(
            right <= left for left, right in zip(pointers_raw, pointers_raw[1:])
        ) or pointers_raw[-1] > atom_count
    ):
        raise EnergeticNetworkError("Amber residue labels/pointers are malformed")
    residue_indices = np.empty(atom_count, dtype=np.int64)
    atom_residue_names: List[str] = [""] * atom_count
    for residue_index, start in enumerate(pointers_raw):
        end = pointers_raw[residue_index + 1] if residue_index + 1 < len(pointers_raw) else atom_count + 1
        residue_indices[start - 1:end - 1] = residue_index
        atom_residue_names[start - 1:end - 1] = [residue_names[residue_index]] * (end - start)

    bonds: Set[Tuple[int, int]] = set()
    for name in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
        values = [int(value) for value in sections[name]]
        if len(values) % 3:
            raise EnergeticNetworkError(f"Amber {name} is malformed")
        for index in range(0, len(values), 3):
            first_pointer, second_pointer = values[index:index + 2]
            if first_pointer % 3 or second_pointer % 3:
                raise EnergeticNetworkError(
                    f"Amber {name} atom pointers are not multiples of three"
                )
            first, second = first_pointer // 3, second_pointer // 3
            if min(first, second) < 0 or max(first, second) >= atom_count or first == second:
                raise EnergeticNetworkError(f"Amber {name} contains an invalid bond")
            pair = (min(first, second), max(first, second))
            bonds.add(pair)
    excluded = _excluded_pairs_from_bonds(atom_count, sorted(bonds))
    return PairwiseParameters(
        parameter_source="amber_prmtop_v1",
        parameter_files=(str(Path(path).expanduser().resolve(strict=False)),),
        atom_names=atom_names,
        residue_names=tuple(atom_residue_names),
        residue_indices=residue_indices,
        charges_e=charges,
        atom_type_indices=atom_types,
        nonbonded_parameter_indices=nb_index,
        lennard_jones_a=lj_a,
        lennard_jones_b=lj_b,
        atom_sigma_angstrom=None,
        atom_epsilon_kcal_per_mol=None,
        pair_parameter_overrides={},
        excluded_pairs=excluded,
        atom_count=atom_count,
        atom_type_count=atom_type_count,
        bond_count=len(bonds),
        nbfix_pair_type_count=0,
    )


def _as_float(text: str, label: str) -> float:
    try:
        value = float(text.replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise EnergeticNetworkError(f"{label} is not numeric: {text!r}") from exc
    if not math.isfinite(value):
        raise EnergeticNetworkError(f"{label} must be finite")
    return value


def _read_charmm_psf_atoms(path: Path) -> Dict[str, object]:
    source = Path(path).expanduser().resolve(strict=False)
    try:
        lines = source.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EnergeticNetworkError(f"{source}: {exc}") from exc
    declaration = next(
        (index for index, line in enumerate(lines) if "!NATOM" in line.upper()),
        None,
    )
    if declaration is None:
        raise EnergeticNetworkError("CHARMM PSF contains no !NATOM declaration")
    try:
        atom_count = int(lines[declaration].split()[0])
    except (IndexError, ValueError) as exc:
        raise EnergeticNetworkError("CHARMM PSF !NATOM declaration is malformed") from exc
    atom_rows = [
        line for line in lines[declaration + 1:] if line.strip()
    ][:atom_count]
    if atom_count <= 0 or len(atom_rows) != atom_count:
        raise EnergeticNetworkError("CHARMM PSF atom table is incomplete")
    atom_names: List[str] = []
    residue_names: List[str] = []
    atom_types: List[str] = []
    charges: List[float] = []
    for expected_index, line in enumerate(atom_rows, start=1):
        fields = line.split()
        if len(fields) < 8:
            raise EnergeticNetworkError(
                f"CHARMM PSF atom row {expected_index} has fewer than eight fields"
            )
        try:
            observed_index = int(fields[0])
        except ValueError as exc:
            raise EnergeticNetworkError(
                f"CHARMM PSF atom row {expected_index} has a malformed index"
            ) from exc
        if observed_index != expected_index:
            raise EnergeticNetworkError(
                "CHARMM PSF atom indices must be consecutive and one-based"
            )
        residue_names.append(fields[3].strip())
        atom_names.append(fields[4].strip())
        atom_types.append(fields[5].strip().upper())
        charges.append(_as_float(fields[6], f"PSF atom {expected_index} charge"))
    bonds, identity = load_connectivity(source, atom_count)
    return {
        "atom_count": atom_count,
        "atom_names": tuple(atom_names),
        "residue_names": tuple(residue_names),
        "atom_types": tuple(atom_types),
        "charges_e": np.asarray(charges, dtype=float),
        "bonds": bonds,
        "bond_count": int(identity["bond_count"]),
    }


_CHARMM_SECTIONS = {
    "ATOMS", "BONDS", "ANGLES", "THETAS", "DIHEDRALS", "PHI",
    "IMPROPER", "IMPHI", "CMAP", "NONBONDED", "NBONDED", "NBFIX",
    "HBOND", "END",
}
_CHARMM_NONBONDED_OPTIONS = {
    "NBXMOD", "ATOM", "CDIEL", "RDIE", "SHIFT", "VATOM", "VDISTANCE",
    "VSWITCH", "CUTNB", "CTOFNB", "CTONNB", "EPS", "E14FAC", "WMIN",
}


def _read_charmm_lj_tables(
    paths: Sequence[Path], atom_types: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    definitions: Dict[str, Tuple[float, float]] = {}
    nbfix: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for path in paths:
        source = Path(path).expanduser().resolve(strict=False)
        try:
            lines = source.read_text(encoding="utf-8", errors="strict").splitlines()
        except (OSError, UnicodeError) as exc:
            raise EnergeticNetworkError(f"{source}: {exc}") from exc
        section: Optional[str] = None
        for line_number, raw_line in enumerate(lines, start=1):
            text = raw_line.split("!", 1)[0].strip()
            if not text or text.startswith("*"):
                continue
            fields = text.split()
            keyword = fields[0].upper()
            if keyword.startswith("NONB") or keyword.startswith("NBON"):
                section = "NONBONDED"
                continue
            if keyword in _CHARMM_SECTIONS:
                section = keyword
                continue
            if section == "NONBONDED":
                if keyword in _CHARMM_NONBONDED_OPTIONS or len(fields) < 4:
                    continue
                try:
                    ignored = _as_float(fields[1], "CHARMM NONBONDED ignored field")
                    epsilon_raw = _as_float(fields[2], "CHARMM NONBONDED epsilon")
                    rmin_half = _as_float(fields[3], "CHARMM NONBONDED Rmin/2")
                except EnergeticNetworkError:
                    continue
                if epsilon_raw > 0.0 or rmin_half <= 0.0:
                    raise EnergeticNetworkError(
                        f"{source}:{line_number}: CHARMM epsilon must be nonpositive "
                        "and Rmin/2 must be positive"
                    )
                definitions[fields[0].upper()] = (abs(epsilon_raw), rmin_half)
            elif section == "NBFIX":
                if len(fields) < 4:
                    continue
                epsilon_raw = _as_float(fields[2], "CHARMM NBFIX epsilon")
                rmin = _as_float(fields[3], "CHARMM NBFIX Rmin")
                if epsilon_raw > 0.0 or rmin <= 0.0:
                    raise EnergeticNetworkError(
                        f"{source}:{line_number}: CHARMM NBFIX epsilon must be "
                        "nonpositive and Rmin must be positive"
                    )
                key = tuple(sorted((fields[0].upper(), fields[1].upper())))
                nbfix[key] = (abs(epsilon_raw), rmin)
    unique_types = tuple(dict.fromkeys(str(value) for value in atom_types))
    missing = sorted(set(unique_types).difference(definitions))
    if missing:
        raise EnergeticNetworkError(
            "CHARMM parameter files lack NONBONDED entries for atom types: "
            + ", ".join(missing)
        )
    type_lookup = {name: index for index, name in enumerate(unique_types)}
    type_indices = np.asarray([type_lookup[name] for name in atom_types], dtype=np.int64)
    type_count = len(unique_types)
    table_indices = np.arange(type_count * type_count, dtype=np.int64)
    lj_a = np.zeros(type_count * type_count, dtype=float)
    lj_b = np.zeros_like(lj_a)
    used_nbfix = 0
    for left, left_name in enumerate(unique_types):
        left_epsilon, left_rmin_half = definitions[left_name]
        for right, right_name in enumerate(unique_types):
            key = tuple(sorted((left_name, right_name)))
            if key in nbfix:
                epsilon, rmin = nbfix[key]
                used_nbfix += int(left <= right)
            else:
                right_epsilon, right_rmin_half = definitions[right_name]
                epsilon = math.sqrt(left_epsilon * right_epsilon)
                rmin = left_rmin_half + right_rmin_half
            index = left * type_count + right
            lj_a[index] = epsilon * rmin ** 12
            lj_b[index] = 2.0 * epsilon * rmin ** 6
    return type_indices, table_indices, lj_a, lj_b, used_nbfix


def read_charmm_pairwise_parameters(
    psf_path: Path, parameter_paths: Sequence[Path],
) -> PairwiseParameters:
    if not parameter_paths:
        raise EnergeticNetworkError("CHARMM input requires at least one parameter file")
    psf = _read_charmm_psf_atoms(psf_path)
    atom_types = psf["atom_types"]
    assert isinstance(atom_types, tuple)
    type_indices, table_indices, lj_a, lj_b, used_nbfix = _read_charmm_lj_tables(
        parameter_paths, atom_types,
    )
    atom_count = int(psf["atom_count"])
    bonds = psf["bonds"]
    assert isinstance(bonds, tuple)
    return PairwiseParameters(
        parameter_source="charmm_psf_parameter_files_v1",
        parameter_files=tuple(
            str(Path(path).expanduser().resolve(strict=False))
            for path in parameter_paths
        ),
        atom_names=psf["atom_names"],  # type: ignore[arg-type]
        residue_names=psf["residue_names"],  # type: ignore[arg-type]
        residue_indices=np.zeros(atom_count, dtype=np.int64),
        charges_e=psf["charges_e"],  # type: ignore[arg-type]
        atom_type_indices=type_indices,
        nonbonded_parameter_indices=table_indices,
        lennard_jones_a=lj_a,
        lennard_jones_b=lj_b,
        atom_sigma_angstrom=None,
        atom_epsilon_kcal_per_mol=None,
        pair_parameter_overrides={},
        excluded_pairs=_excluded_pairs_from_bonds(atom_count, bonds),
        atom_count=atom_count,
        atom_type_count=len(set(atom_types)),
        bond_count=int(psf["bond_count"]),
        nbfix_pair_type_count=used_nbfix,
    )


def _xml_local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _xml_number(element: ET.Element, names: Sequence[str], label: str) -> float:
    for name in names:
        if name in element.attrib:
            return _as_float(element.attrib[name], label)
    raise EnergeticNetworkError(
        f"OpenMM System XML {label} lacks any of: {', '.join(names)}"
    )


def read_openmm_system_pairwise_parameters(
    xml_path: Path, connectivity_path: Path, atoms: Sequence[AtomRecord],
) -> PairwiseParameters:
    source = Path(xml_path).expanduser().resolve(strict=False)
    try:
        root = ET.parse(source).getroot()
    except (OSError, ET.ParseError) as exc:
        raise EnergeticNetworkError(f"{source}: {exc}") from exc
    forces = []
    custom_nonbonded = []
    for element in root.iter():
        local = _xml_local_name(element)
        type_name = str(element.attrib.get("type", local))
        if type_name.endswith("CustomNonbondedForce"):
            custom_nonbonded.append(type_name)
        if type_name.endswith("NonbondedForce") and not type_name.endswith(
            "CustomNonbondedForce"
        ):
            forces.append(element)
    if custom_nonbonded:
        raise EnergeticNetworkError(
            "OpenMM System XML uses CustomNonbondedForce; native extraction cannot "
            "safely infer its Lennard-Jones/NBFIX energy expression"
        )
    if len(forces) != 1:
        raise EnergeticNetworkError(
            f"OpenMM System XML must contain exactly one NonbondedForce; found {len(forces)}"
        )
    force = forces[0]
    if any(
        _xml_local_name(element) in {"ParticleOffsets", "ExceptionOffsets"}
        and len(element) > 0
        for element in force
    ):
        raise EnergeticNetworkError(
            "OpenMM parameter offsets are unsupported because their active global "
            "parameter values are not encoded unambiguously in this analysis input"
        )
    particle_container = next(
        (element for element in force if _xml_local_name(element) == "Particles"),
        None,
    )
    if particle_container is None:
        raise EnergeticNetworkError("OpenMM NonbondedForce contains no Particles table")
    particles = [
        element for element in particle_container
        if _xml_local_name(element) == "Particle"
    ]
    if len(particles) != len(atoms):
        raise EnergeticNetworkError(
            f"OpenMM NonbondedForce has {len(particles)} particles; topology has {len(atoms)} atoms"
        )
    charges = np.asarray([
        _xml_number(row, ("q", "charge"), f"particle {index} charge")
        for index, row in enumerate(particles)
    ], dtype=float)
    sigma = np.asarray([
        10.0 * _xml_number(row, ("sig", "sigma"), f"particle {index} sigma")
        for index, row in enumerate(particles)
    ], dtype=float)
    epsilon = np.asarray([
        _xml_number(row, ("eps", "epsilon"), f"particle {index} epsilon") / 4.184
        for index, row in enumerate(particles)
    ], dtype=float)
    if np.any(sigma < 0.0) or np.any(epsilon < 0.0):
        raise EnergeticNetworkError("OpenMM particle sigma and epsilon must be nonnegative")
    bonds, connectivity_identity = load_connectivity(connectivity_path, len(atoms))
    overrides: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    exception_container = next(
        (element for element in force if _xml_local_name(element) == "Exceptions"),
        None,
    )
    if exception_container is not None:
        for index, row in enumerate(exception_container):
            if _xml_local_name(row) != "Exception":
                continue
            try:
                first = int(row.attrib["p1"])
                second = int(row.attrib["p2"])
            except (KeyError, ValueError) as exc:
                raise EnergeticNetworkError(
                    f"OpenMM exception {index} has malformed particle indices"
                ) from exc
            if min(first, second) < 0 or max(first, second) >= len(atoms) or first == second:
                raise EnergeticNetworkError(f"OpenMM exception {index} has invalid indices")
            charge_product = _xml_number(
                row, ("q", "chargeProd", "charge"),
                f"exception {index} charge product",
            )
            pair_sigma = 10.0 * _xml_number(
                row, ("sig", "sigma"), f"exception {index} sigma",
            )
            pair_epsilon = _xml_number(
                row, ("eps", "epsilon"), f"exception {index} epsilon",
            ) / 4.184
            if pair_sigma < 0.0 or pair_epsilon < 0.0:
                raise EnergeticNetworkError(
                    f"OpenMM exception {index} sigma and epsilon must be nonnegative"
                )
            overrides[(min(first, second), max(first, second))] = (
                charge_product,
                4.0 * pair_epsilon * pair_sigma ** 12,
                4.0 * pair_epsilon * pair_sigma ** 6,
            )
    atom_names = tuple(atom.atom_name.strip() for atom in atoms)
    residue_names = tuple(atom.residue_name.strip() for atom in atoms)
    return PairwiseParameters(
        parameter_source="openmm_serialized_system_xml_v1",
        parameter_files=(str(source),),
        atom_names=atom_names,
        residue_names=residue_names,
        residue_indices=np.zeros(len(atoms), dtype=np.int64),
        charges_e=charges,
        atom_type_indices=np.zeros(len(atoms), dtype=np.int64),
        nonbonded_parameter_indices=np.zeros(0, dtype=np.int64),
        lennard_jones_a=np.zeros(0, dtype=float),
        lennard_jones_b=np.zeros(0, dtype=float),
        atom_sigma_angstrom=sigma,
        atom_epsilon_kcal_per_mol=epsilon,
        pair_parameter_overrides=overrides,
        excluded_pairs=_excluded_pairs_from_bonds(len(atoms), bonds),
        atom_count=len(atoms),
        atom_type_count=0,
        bond_count=int(connectivity_identity["bond_count"]),
        nbfix_pair_type_count=0,
    )


def _finite_number(
    value: object, name: str, *, minimum: Optional[float] = None,
    maximum: Optional[float] = None, strictly_positive: bool = False,
) -> float:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise EnergeticNetworkError(f"{name} must be finite")
    result = float(value)
    if strictly_positive and result <= 0.0:
        raise EnergeticNetworkError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise EnergeticNetworkError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise EnergeticNetworkError(f"{name} must be at most {maximum}")
    return result


_SETTING_FIELDS = {
    "parameter_source", "atom_scope", "periodic_pair_treatment",
    "electrostatic_reporting_threshold_kcal_per_mol",
    "vdw_reporting_threshold_kcal_per_mol", "network_edge_threshold",
    "heat_diffusion_time", "embedding_component_count", "frame_stride",
    "frame_selection", "minimum_evaluated_frames_per_system",
    "maximum_common_protein_residues", "maximum_selected_atom_pairs",
    "maximum_atom_pair_frame_evaluations", "pair_chunk_size",
    "maximum_heat_kernel_elements", "maximum_vdw_to_electrostatic_ratio",
}


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = (
        definitions.get("energetic_network_embeddings")
        if isinstance(definitions, dict) else None
    )
    if not isinstance(raw, dict):
        raise EnergeticNetworkError(
            "definitions.energetic_network_embeddings must be an object"
        )
    missing = sorted(_SETTING_FIELDS.difference(raw))
    unknown = sorted(set(raw).difference(_SETTING_FIELDS))
    if missing or unknown:
        raise EnergeticNetworkError(
            "energetic-network settings mismatch; missing=" + ",".join(missing)
            + "; unknown=" + ",".join(unknown)
        )
    if raw["parameter_source"] not in {
        "amber_prmtop_v1", "force_field_parameter_source_auto_v1",
    }:
        raise EnergeticNetworkError(
            "parameter_source must be force_field_parameter_source_auto_v1 "
            "(or legacy amber_prmtop_v1)"
        )
    fixed = {
        "atom_scope": "strict_common_complete_protein_residues_v1",
        "periodic_pair_treatment": "nonperiodic_made_whole_cpptraj_pairwise_v1",
    }
    for name, expected in fixed.items():
        if raw[name] != expected:
            raise EnergeticNetworkError(f"{name} must be {expected}")
    for name in (
        "minimum_evaluated_frames_per_system", "maximum_common_protein_residues",
        "maximum_selected_atom_pairs", "maximum_atom_pair_frame_evaluations",
        "pair_chunk_size", "maximum_heat_kernel_elements",
    ):
        positive_integer(raw[name], name, error_type=EnergeticNetworkError)
    components = positive_integer(
        raw["embedding_component_count"], "embedding_component_count",
        error_type=EnergeticNetworkError,
    )
    if components != 3:
        raise EnergeticNetworkError(
            "embedding_component_count must be 3 for the Cowan/Thayer protocol"
        )
    positive_integer(raw["frame_stride"], "frame_stride", error_type=EnergeticNetworkError)
    _finite_number(
        raw["electrostatic_reporting_threshold_kcal_per_mol"],
        "electrostatic_reporting_threshold_kcal_per_mol", minimum=0.0,
    )
    _finite_number(
        raw["vdw_reporting_threshold_kcal_per_mol"],
        "vdw_reporting_threshold_kcal_per_mol", minimum=0.0,
    )
    _finite_number(raw["network_edge_threshold"], "network_edge_threshold", minimum=0.0)
    _finite_number(raw["heat_diffusion_time"], "heat_diffusion_time", strictly_positive=True)
    _finite_number(
        raw["maximum_vdw_to_electrostatic_ratio"],
        "maximum_vdw_to_electrostatic_ratio", minimum=0.0,
    )
    result = dict(raw)
    result["frame_selection"] = normalize_frame_selection(
        raw["frame_selection"], int(raw["frame_stride"]),
        error_type=EnergeticNetworkError,
    )
    return result


def _residue_key(atom: AtomRecord) -> Tuple[str, int, str, str]:
    return (
        atom.chain_id, atom.residue_number, atom.insertion_code,
        atom.residue_name.upper(),
    )


def _complete_common_protein_layout(
    atom_sets: Sequence[Sequence[AtomRecord]], maximum_residues: int,
) -> Tuple[List[Dict[str, object]], List[np.ndarray], Dict[str, object]]:
    """Retain whole protein residues with identical atom identities everywhere."""

    if not atom_sets:
        raise EnergeticNetworkError("no topology atom sets were supplied")
    per_topology: List[Dict[Tuple[str, int, str, str], List[AtomRecord]]] = []
    for atoms in atom_sets:
        groups: Dict[Tuple[str, int, str, str], List[AtomRecord]] = {}
        for atom in atoms:
            key = _residue_key(atom)
            if key[3] in PROTEIN_RESIDUES:
                groups.setdefault(key, []).append(atom)
        per_topology.append(groups)
    common = set(per_topology[0])
    for groups in per_topology[1:]:
        common.intersection_update(groups)
    retained = []
    incomplete = []
    for key in per_topology[0]:
        if key not in common:
            continue
        signatures = [
            tuple(sorted((atom.atom_name, atom.altloc) for atom in groups[key]))
            for groups in per_topology
        ]
        if len(set(signatures)) != 1 or len(signatures[0]) != len(set(signatures[0])):
            incomplete.append(key)
            continue
        retained.append(key)
    if len(retained) < 2:
        raise EnergeticNetworkError(
            "fewer than two complete common protein residues remain after strict mapping"
        )
    if len(retained) > maximum_residues:
        raise EnergeticNetworkError(
            f"{len(retained)} common protein residues exceed maximum_common_protein_residues={maximum_residues}"
        )
    nodes = [{
        "node_index": index,
        "node_id": f"{key[0] or '_'}:{key[1]}{key[2]}:{key[3]}",
        "chain_id": key[0], "residue_number": key[1],
        "insertion_code": key[2], "residue_name": key[3],
    } for index, key in enumerate(retained)]
    indices: List[np.ndarray] = []
    for groups in per_topology:
        selected = []
        for key in retained:
            by_identity = {
                (atom.atom_name, atom.altloc): atom.atom_index
                for atom in groups[key]
            }
            for identity in sorted(by_identity):
                selected.append(by_identity[identity])
        indices.append(np.asarray(selected, dtype=np.int64))
    all_reference_protein = set(per_topology[0])
    return nodes, indices, {
        "reference_protein_residue_count": len(all_reference_protein),
        "common_residue_count": len(retained),
        "excluded_noncommon_or_mutated_residue_count": len(all_reference_protein - common),
        "excluded_incomplete_atom_identity_residue_count": len(incomplete),
        "contract": "whole residues only; solvent, ions, nucleic acid, ligands, mutations, and atom-incomplete residues are excluded",
    }


def _validate_parameter_identity(
    atoms: Sequence[AtomRecord], parameters: PairwiseParameters, label: str,
) -> None:
    if len(atoms) != parameters.atom_count:
        raise EnergeticNetworkError(
            f"{label}: topology has {len(atoms)} atoms but interaction parameters "
            f"have {parameters.atom_count}"
        )
    for index, (atom, amber_name, amber_residue) in enumerate(zip(
        atoms, parameters.atom_names, parameters.residue_names
    )):
        if atom.atom_name.strip() != amber_name or atom.residue_name.strip().upper() != amber_residue.upper():
            raise EnergeticNetworkError(
                f"{label}: parameter/topology atom-order identity mismatch at atom {index}: "
                f"{amber_residue}:{amber_name} versus {atom.residue_name}:{atom.atom_name}"
            )


def _selected_pair_terms(
    parameters: PairwiseParameters,
    selected_atom_indices: np.ndarray,
    selected_residue_indices: np.ndarray,
    maximum_pairs: int,
) -> Dict[str, np.ndarray]:
    atom_count = len(selected_atom_indices)
    possible = atom_count * (atom_count - 1) // 2
    if possible > maximum_pairs:
        raise EnergeticNetworkError(
            f"{possible} selected atom pairs exceed maximum_selected_atom_pairs={maximum_pairs}"
        )
    left, right = np.triu_indices(atom_count, 1)
    full_left = selected_atom_indices[left]
    full_right = selected_atom_indices[right]
    same_residue = selected_residue_indices[left] == selected_residue_indices[right]
    excluded = np.fromiter(
        (
            (min(int(i), int(j)), max(int(i), int(j))) in parameters.excluded_pairs
            for i, j in zip(full_left, full_right)
        ), dtype=bool, count=len(left),
    )
    keep = ~(same_residue | excluded)
    left, right = left[keep], right[keep]
    full_left, full_right = full_left[keep], full_right[keep]
    if (
        parameters.atom_sigma_angstrom is not None
        and parameters.atom_epsilon_kcal_per_mol is not None
    ):
        sigma = 0.5 * (
            parameters.atom_sigma_angstrom[full_left]
            + parameters.atom_sigma_angstrom[full_right]
        )
        epsilon = np.sqrt(
            parameters.atom_epsilon_kcal_per_mol[full_left]
            * parameters.atom_epsilon_kcal_per_mol[full_right]
        )
        lj_a = 4.0 * epsilon * sigma ** 12
        lj_b = 4.0 * epsilon * sigma ** 6
    else:
        type_left = parameters.atom_type_indices[full_left]
        type_right = parameters.atom_type_indices[full_right]
        table_index = parameters.nonbonded_parameter_indices[
            type_left * parameters.atom_type_count + type_right
        ]
        lj_a = parameters.lennard_jones_a[table_index].copy()
        lj_b = parameters.lennard_jones_b[table_index].copy()
    charge_product = (
        parameters.charges_e[full_left] * parameters.charges_e[full_right]
    )
    if parameters.pair_parameter_overrides:
        for pair_index, (first, second) in enumerate(zip(full_left, full_right)):
            override = parameters.pair_parameter_overrides.get(
                (min(int(first), int(second)), max(int(first), int(second)))
            )
            if override is not None:
                charge_product[pair_index], lj_a[pair_index], lj_b[pair_index] = override
    return {
        "left": left, "right": right,
        "residue_left": selected_residue_indices[left],
        "residue_right": selected_residue_indices[right],
        "charge_product": charge_product,
        "lj_a": lj_a,
        "lj_b": lj_b,
    }


def cpptraj_style_residue_energy_matrices(
    coordinates_angstrom: Sequence[Sequence[float]],
    pair_terms: Mapping[str, np.ndarray], residue_count: int,
    *, pair_chunk_size: int = 250_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return absolute electrostatic and VDW residue-pair weight matrices.

    This intentionally uses direct nonperiodic distances, force-field exclusions,
    ``332.0522173*q_i*q_j/r``, and ``A/r^12-B/r^6``. Reporting cutoffs do not
    alter cpptraj's map output and therefore do not alter these matrices.
    """

    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.isfinite(coordinates).all():
        raise EnergeticNetworkError("coordinates must be a finite atom-by-three matrix")
    chunk_size = positive_integer(
        pair_chunk_size, "pair_chunk_size", error_type=EnergeticNetworkError
    )
    electrostatic = np.zeros((residue_count, residue_count), dtype=float)
    vdw = np.zeros_like(electrostatic)
    left = pair_terms["left"]
    right = pair_terms["right"]
    for start in range(0, len(left), chunk_size):
        stop = min(len(left), start + chunk_size)
        distances = np.linalg.norm(
            coordinates[left[start:stop]] - coordinates[right[start:stop]], axis=1
        )
        if np.any(distances <= 0.0):
            raise EnergeticNetworkError("an included atom pair has zero separation")
        inv_r = 1.0 / distances
        elec = (
            CPPTRAJ_COULOMB_FACTOR_KCAL_ANGSTROM_PER_MOL_E2
            * pair_terms["charge_product"][start:stop] * inv_r
        )
        inv_r6 = inv_r ** 6
        lj = (
            pair_terms["lj_a"][start:stop] * inv_r6 * inv_r6
            - pair_terms["lj_b"][start:stop] * inv_r6
        )
        ri = pair_terms["residue_left"][start:stop]
        rj = pair_terms["residue_right"][start:stop]
        np.add.at(electrostatic, (ri, rj), np.abs(elec))
        np.add.at(vdw, (ri, rj), np.abs(lj))
    electrostatic += electrostatic.T
    vdw += vdw.T
    return electrostatic, vdw


def locally_normalized_energy_network(
    absolute_residue_energy: Sequence[Sequence[float]], threshold: float = 0.003,
) -> np.ndarray:
    """Apply the local maximum normalization and strict edge threshold."""

    matrix = np.asarray(absolute_residue_energy, dtype=float)
    if (
        matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[0] != matrix.shape[1]
        or not np.isfinite(matrix).all() or np.any(matrix < 0.0)
        or not np.allclose(matrix, matrix.T)
    ):
        raise EnergeticNetworkError("absolute residue energy must be finite, nonnegative, square, and symmetric")
    edge_threshold = _finite_number(threshold, "threshold", minimum=0.0)
    row_sums = matrix.sum(axis=1)
    normalized = np.zeros_like(matrix)
    for left in range(len(matrix) - 1):
        for right in range(left + 1, len(matrix)):
            value = matrix[left, right]
            if value <= 0.0:
                continue
            fractions = []
            if row_sums[left] > 0.0:
                fractions.append(value / row_sums[left])
            if row_sums[right] > 0.0:
                fractions.append(value / row_sums[right])
            retained = max(fractions, default=0.0)
            if retained > edge_threshold:
                normalized[left, right] = normalized[right, left] = retained
    return normalized


def heat_kernel(network: Sequence[Sequence[float]], diffusion_time: float = 6.0) -> np.ndarray:
    """Return the normalized-graph-Laplacian heat kernel."""

    adjacency = np.asarray(network, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise EnergeticNetworkError("network must be square")
    degrees = adjacency.sum(axis=1)
    inverse_root = np.zeros_like(degrees)
    positive = degrees > 0.0
    inverse_root[positive] = 1.0 / np.sqrt(degrees[positive])
    laplacian = np.eye(len(adjacency)) - (
        inverse_root[:, None] * adjacency * inverse_root[None, :]
    )
    laplacian[~positive, ~positive] = 0.0
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    kernel = (eigenvectors * np.exp(-float(diffusion_time) * eigenvalues)) @ eigenvectors.T
    return np.round(kernel, decimals=6)


def _pca_scores_per_kernel(kernel: np.ndarray, component_count: int) -> np.ndarray:
    """Match the supplement's deterministic per-frame PCA of heat-kernel rows."""

    centered = kernel - kernel.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    components = vectors[:component_count].copy()
    # Match scikit-learn's deterministic svd_flip(u_based_decision=True).
    scores = centered @ components.T
    for component in range(component_count):
        column = scores[:, component]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0.0:
            scores[:, component] *= -1.0
    return scores


def compare_embedding_ensembles(
    kernels_by_system: Mapping[str, Sequence[np.ndarray]], component_count: int = 3,
) -> List[Dict[str, object]]:
    """Sum per-component one-dimensional Wasserstein distances per residue."""

    embeddings: Dict[str, np.ndarray] = {}
    residue_count: Optional[int] = None
    for system_id, kernels in kernels_by_system.items():
        if not kernels:
            raise EnergeticNetworkError(f"system {system_id} contains no heat kernels")
        scores = np.stack([
            _pca_scores_per_kernel(np.asarray(kernel, dtype=float), component_count)
            for kernel in kernels
        ], axis=0)
        if residue_count is None:
            residue_count = scores.shape[1]
        elif scores.shape[1] != residue_count:
            raise EnergeticNetworkError("systems do not share one residue-node count")
        embeddings[system_id] = scores
    comparisons = []
    system_ids = sorted(embeddings)
    for left_index, left_id in enumerate(system_ids[:-1]):
        for right_id in system_ids[left_index + 1:]:
            left, right = embeddings[left_id], embeddings[right_id]
            rows = []
            assert residue_count is not None
            for residue_index in range(residue_count):
                components = [
                    float(wasserstein_distance(
                        left[:, residue_index, component],
                        right[:, residue_index, component],
                    ))
                    for component in range(component_count)
                ]
                rows.append({
                    "node_index": residue_index,
                    "component_wasserstein_distances": components,
                    "summed_wasserstein_distance": float(sum(components)),
                })
            comparisons.append({
                "system_i": left_id, "system_j": right_id,
                "residue_distances": rows,
            })
    return comparisons


def _resolved_force_field_spec(
    replica: Mapping[str, object], system_path: Path,
) -> Optional[Dict[str, object]]:
    raw = replica.get("force_field_parameters")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise EnergeticNetworkError("force_field_parameters must be an object")
    format_name = raw.get("format")
    files = raw.get("files")
    if not isinstance(format_name, str) or not isinstance(files, list) or not files:
        raise EnergeticNetworkError(
            "force_field_parameters requires a format and nonempty files array"
        )
    return {
        "format": format_name,
        "files": [
            resolve_manifest_path(str(value), system_path) for value in files
        ],
    }


def _load_replica_pairwise_parameters(
    atoms: Sequence[AtomRecord], connectivity_path: Path,
    force_field_spec: Optional[Mapping[str, object]],
) -> PairwiseParameters:
    if force_field_spec is None:
        if connectivity_path.suffix.lower() not in {".prmtop", ".parm7"}:
            raise EnergeticNetworkError(
                "PSF and bond-only connectivity require either CHARMM parameter "
                "files or a serialized OpenMM System XML"
            )
        return read_amber_pairwise_parameters(connectivity_path)
    format_name = str(force_field_spec.get("format", ""))
    raw_files = force_field_spec.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise EnergeticNetworkError("force-field parameter files are missing")
    files = [Path(str(value)).expanduser().resolve(strict=False) for value in raw_files]
    if format_name == "charmm_parameter_files_v1":
        if connectivity_path.suffix.lower() != ".psf":
            raise EnergeticNetworkError(
                "CHARMM parameter files require atom-order-matched PSF connectivity"
            )
        return read_charmm_pairwise_parameters(connectivity_path, files)
    if format_name == "openmm_system_xml_v1":
        if len(files) != 1:
            raise EnergeticNetworkError(
                "OpenMM System XML parameter input requires exactly one file"
            )
        return read_openmm_system_pairwise_parameters(
            files[0], connectivity_path, atoms,
        )
    if format_name == "gromacs_tpr_v1":
        executable = shutil.which("gmx") or shutil.which("gmx_mpi")
        state = "installed" if executable else "not installed"
        raise EnergeticNetworkError(
            "raw GROMACS TPR extraction is not available: GROMACS is " + state
            + ", and a version-independent decomposition of its compiled pair "
            "tables is not implemented; serialize the constructed system as "
            "OpenMM System XML instead"
        )
    raise EnergeticNetworkError(
        f"unsupported force-field parameter format {format_name!r}"
    )


def probe_energetic_parameter_source(
    topology_path: Path, connectivity_path: Path,
    force_field_spec: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Fail-closed preparation probe for one replica parameter source."""

    topology = Path(topology_path).expanduser().resolve(strict=False)
    connectivity = Path(connectivity_path).expanduser().resolve(strict=False)
    try:
        _, atoms = read_topology_atoms(topology)
        parameters = _load_replica_pairwise_parameters(
            atoms, connectivity, force_field_spec,
        )
        _validate_parameter_identity(atoms, parameters, "parameter-source probe")
    except (
        EnergeticNetworkError, AtomMappingError, PeriodicReconstructionError,
        OSError, KeyError, TypeError, ValueError,
    ) as exc:
        return {
            "availability_status": "not_available",
            "availability_reason": str(exc),
            "topology": str(topology),
            "connectivity": str(connectivity),
            "force_field_parameters": (
                {
                    "format": force_field_spec.get("format"),
                    "files": [str(value) for value in force_field_spec.get("files", [])],
                }
                if isinstance(force_field_spec, Mapping) else None
            ),
        }
    return {
        "availability_status": "available",
        "availability_reason": None,
        "parameter_source": parameters.parameter_source,
        "parameter_files": list(parameters.parameter_files),
        "atom_count": parameters.atom_count,
        "bond_count": parameters.bond_count,
        "nbfix_pair_type_count": parameters.nbfix_pair_type_count,
        "topology": str(topology),
        "connectivity": str(connectivity),
    }


def _availability(project_path: Path) -> Tuple[bool, str, List[Dict[str, object]]]:
    context = compile_project_context_file(project_path, hash_content=False)
    system_path = Path(str(context["system_manifest_path"]))
    manifest = load_json(system_path)
    details = []
    for system in manifest["systems"]:
        for replica in system["replicas"]:
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            raw_connectivity = replica.get("connectivity")
            if not isinstance(raw_connectivity, str):
                probe = {
                    "availability_status": "not_available",
                    "availability_reason": "explicit connectivity is required",
                }
            else:
                connectivity_path = resolve_manifest_path(raw_connectivity, system_path)
                spec = _resolved_force_field_spec(replica, system_path)
                probe = probe_energetic_parameter_source(
                    topology_path, connectivity_path, spec,
                )
            details.append({
                "system_id": str(system["system_id"]),
                "replica_id": str(replica["replica_id"]),
                **probe,
            })
    incompatible = [
        row for row in details
        if row.get("availability_status") != "available"
    ]
    if incompatible:
        first = str(incompatible[0].get("availability_reason", "unknown reason"))
        return False, "not available: " + first, details
    return True, "available", details


def energetic_network_embeddings_project(
    project_path: Path, hash_content: bool = False,
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    context = compile_project_context_file(source, hash_content=hash_content)
    available, reason, availability_details = _availability(source)
    base = {
        "module_id": "energetic_network_embeddings",
        "technical_status": "complete", "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": context["system_manifest_path"],
        "system_manifest_sha256": context["system_manifest_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content, "settings": settings,
        "availability_details": availability_details,
    }
    if not available:
        issue = {
            "severity": "warning", "code": "ENERGETIC_NETWORK_NOT_AVAILABLE",
            "message": reason,
        }
        return {
            **base, "availability_status": "not_available",
            "availability_reason": reason, "analysis_performed": False,
            "nodes": [], "systems": [], "pairwise_system_comparisons": [],
            "error_count": 0, "warning_count": 1, "issues": [issue],
            "limitations": [
                "The module does not infer force-field charges or Lennard-Jones parameters from coordinates or bond connectivity."
            ],
        }

    contract = context["contract"]
    assert isinstance(contract, dict)
    units = contract["units"]
    assert isinstance(units, dict)
    coordinate_unit = str(units["coordinates"])
    require_periodic_policy(contract.get("periodic_coordinate_policy"))
    system_path = Path(str(context["system_manifest_path"]))
    manifest = load_json(system_path)
    systems = manifest["systems"]
    frame_plan, frame_report = plan_frame_selection(
        manifest, system_path, coordinate_unit, settings["frame_selection"],
        frame_stride=int(settings["frame_stride"]),
        error_type=EnergeticNetworkError,
    )

    records: List[Dict[str, object]] = []
    for system in systems:
        for replica in system["replicas"]:
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            connectivity_path = resolve_manifest_path(
                str(replica["connectivity"]), system_path
            )
            _, atoms = read_topology_atoms(topology_path)
            force_field_spec = _resolved_force_field_spec(replica, system_path)
            parameters = _load_replica_pairwise_parameters(
                atoms, connectivity_path, force_field_spec,
            )
            label = f"{system['system_id']}/{replica['replica_id']}"
            _validate_parameter_identity(atoms, parameters, label)
            records.append({
                "key": (str(system["system_id"]), str(replica["replica_id"])),
                "atoms": atoms, "parameters": parameters,
                "connectivity_path": connectivity_path,
            })
    nodes, selected_indices, mapping_report = _complete_common_protein_layout(
        [record["atoms"] for record in records],  # type: ignore[list-item]
        int(settings["maximum_common_protein_residues"]),
    )
    residue_count = len(nodes)
    if residue_count < int(settings["embedding_component_count"]):
        raise EnergeticNetworkError(
            "common protein residue count is smaller than embedding_component_count"
        )
    atom_count = len(selected_indices[0])
    possible_pairs = atom_count * (atom_count - 1) // 2
    selected_pair_frame_evaluations = 0
    kernels_by_system: Dict[str, List[np.ndarray]] = {
        str(system["system_id"]): [] for system in systems
    }
    system_reports = []
    record_by_key = {record["key"]: (record, indices) for record, indices in zip(records, selected_indices)}
    total_kernel_elements = 0
    for system in systems:
        system_id = str(system["system_id"])
        frame_rows = []
        electrostatic_total = 0.0
        vdw_total = 0.0
        segment_reports = []
        for replica in system["replicas"]:
            replica_id = str(replica["replica_id"])
            record, indices = record_by_key[(system_id, replica_id)]
            atoms = record["atoms"]
            parameters = record["parameters"]
            assert isinstance(atoms, list) and isinstance(parameters, PairwiseParameters)
            node_index_by_key = {
                (
                    str(node["chain_id"]), int(node["residue_number"]),
                    str(node["insertion_code"]), str(node["residue_name"]),
                ): node_index
                for node_index, node in enumerate(nodes)
            }
            selected_residue_indices = np.asarray([
                node_index_by_key[_residue_key(atoms[int(atom_index)])]
                for atom_index in indices
            ], dtype=np.int64)
            if len(selected_residue_indices) != len(indices):
                raise EnergeticNetworkError("internal common-residue atom indexing mismatch")
            pair_terms = _selected_pair_terms(
                parameters, indices, selected_residue_indices,
                int(settings["maximum_selected_atom_pairs"]),
            )
            processor = PeriodicFrameProcessor.from_replica(
                project, replica, system_path, len(atoms)
            )
            for segment in replica["segments"]:
                segment_id = str(segment["segment_id"])
                trajectory_path = resolve_manifest_path(str(segment["trajectory"]), system_path)
                selected = frame_plan[(system_id, replica_id, segment_id)]
                evaluated = 0
                processor.begin_segment(bool(segment.get("continuous_with_previous", False)))
                for raw_frame in iter_coordinate_frames(
                    trajectory_path, coordinate_unit,
                    reader_frame_indices(selected, processor.policy),
                ):
                    use = frame_selected(
                        raw_frame.frame_index, selected, int(settings["frame_stride"])
                    )
                    if not use and processor.policy != "unwrap_continuous":
                        continue
                    frame = processor.process(
                        raw_frame, f"{system_id}/{replica_id}/{segment_id}/frame-{raw_frame.frame_index}",
                        tuple(int(index) for index in indices),
                    )
                    if not use:
                        continue
                    selected_pair_frame_evaluations += len(pair_terms["left"])
                    if selected_pair_frame_evaluations > int(settings["maximum_atom_pair_frame_evaluations"]):
                        raise EnergeticNetworkError(
                            "selected atom-pair/frame evaluations exceed "
                            "maximum_atom_pair_frame_evaluations"
                        )
                    coordinates = np.asarray(frame.coordinates_angstrom, dtype=float)[indices]
                    electrostatic, vdw = cpptraj_style_residue_energy_matrices(
                        coordinates, pair_terms, residue_count,
                        pair_chunk_size=int(settings["pair_chunk_size"]),
                    )
                    electrostatic_total += float(np.sum(np.triu(electrostatic, 1)))
                    vdw_total += float(np.sum(np.triu(vdw, 1)))
                    network = locally_normalized_energy_network(
                        electrostatic, float(settings["network_edge_threshold"])
                    )
                    kernel = heat_kernel(network, float(settings["heat_diffusion_time"]))
                    total_kernel_elements += kernel.size
                    if total_kernel_elements > int(settings["maximum_heat_kernel_elements"]):
                        raise EnergeticNetworkError(
                            "retained heat-kernel elements exceed maximum_heat_kernel_elements"
                        )
                    kernels_by_system[system_id].append(kernel)
                    frame_rows.append({
                        "replica_id": replica_id, "segment_id": segment_id,
                        "source_frame_index": raw_frame.frame_index,
                        "retained_network_edge_count": int(np.count_nonzero(np.triu(network, 1))),
                    })
                    evaluated += 1
                segment_reports.append({
                    "replica_id": replica_id, "segment_id": segment_id,
                    "trajectory_path": str(trajectory_path),
                    "trajectory_sha256": sha256_file(trajectory_path) if hash_content else None,
                    "evaluated_frame_count": evaluated,
                })
        frame_count = len(kernels_by_system[system_id])
        if frame_count < int(settings["minimum_evaluated_frames_per_system"]):
            raise EnergeticNetworkError(
                f"system {system_id} produced {frame_count} frames; minimum_evaluated_frames_per_system is {settings['minimum_evaluated_frames_per_system']}"
            )
        ratio = vdw_total / electrostatic_total if electrostatic_total > 0.0 else math.inf
        vdw_status = (
            "passed" if ratio <= float(settings["maximum_vdw_to_electrostatic_ratio"])
            else "failed"
        )
        system_reports.append({
            "system_id": system_id, "evaluated_frame_count": frame_count,
            "segments": segment_reports, "frame_network_summaries": frame_rows,
            "parameter_sources": [
                {
                    "replica_id": str(replica["replica_id"]),
                    "parameter_source": record_by_key[
                        (system_id, str(replica["replica_id"]))
                    ][0]["parameters"].parameter_source,
                    "parameter_files": list(record_by_key[
                        (system_id, str(replica["replica_id"]))
                    ][0]["parameters"].parameter_files),
                    "nbfix_pair_type_count": record_by_key[
                        (system_id, str(replica["replica_id"]))
                    ][0]["parameters"].nbfix_pair_type_count,
                }
                for replica in system["replicas"]
            ],
            "total_absolute_electrostatic_weight_kcal_per_mol": electrostatic_total,
            "total_absolute_vdw_weight_kcal_per_mol": vdw_total,
            "vdw_to_electrostatic_weight_ratio": ratio,
            "vdw_negligibility_status": vdw_status,
        })

    comparisons = compare_embedding_ensembles(
        kernels_by_system, int(settings["embedding_component_count"])
    )
    for comparison in comparisons:
        for row in comparison["residue_distances"]:
            row["node_id"] = nodes[int(row["node_index"])]["node_id"]
    issues = [issue for issue in context.get("issues", []) if isinstance(issue, dict)]
    for row in system_reports:
        if row["vdw_negligibility_status"] == "failed":
            issues.append({
                "severity": "warning", "code": "VDW_NOT_NEGLIGIBLE",
                "location": row["system_id"],
                "message": "VDW/electrostatic absolute-weight ratio exceeds the configured Cowan/Thayer compatibility gate",
            })
    return {
        **base, "availability_status": "available", "availability_reason": None,
        "analysis_performed": True, "nodes": nodes,
        "common_protein_mapping": mapping_report,
        "pair_energy_contract": {
            "implementation": "native_cpptraj_pairwise_compatible_multiformat_v2",
            "supported_parameter_sources": [
                "amber_prmtop_v1", "charmm_psf_parameter_files_v1",
                "openmm_serialized_system_xml_v1",
            ],
            "charmm_nonbonded_rule": (
                "epsilon geometric mean and additive Rmin/2; explicit NBFIX overrides"
            ),
            "openmm_nonbonded_rule": (
                "standard NonbondedForce Lorentz-Berthelot particles plus explicit exceptions"
            ),
            "electrostatic_equation": "332.0522173*q_i*q_j/r (kcal/mol; charges in e, r in angstrom)",
            "vdw_equation": "A/r^12-B/r^6 (kcal/mol)",
            "periodic_treatment": "none after declared made-whole coordinate processing",
            "exclusions": "derived from the Amber bond graph at topological distances 1, 2, and 3, matching cpptraj SetupExcluded(..., TgtDist=4)",
            "one_four_scaling": "excluded, matching cpptraj pairwise SetupExcluded behavior; no separate scaled 1-4 term",
            "energy_aggregation": "sum absolute atom-pair energies into symmetric residue-pair weights",
            "solvent_ions_nucleic_acids_ligands": "excluded",
            "complete_pme_decomposition_claimed": False,
            "reporting_cutoff_note": "cpptraj cuteelec/cutevdw affect printed atom-pair reports, not emapout/vmapout matrices; they are provenance-only here",
        },
        "network_embedding_contract": {
            "normalization": "max(Eij/row_sum_i,Eij/row_sum_j)",
            "edge_retention": "strictly greater than network_edge_threshold",
            "channel": "electrostatic",
            "laplacian": "symmetric normalized graph Laplacian",
            "heat_kernel": "exp(-tL), rounded to six decimals",
            "embedding": "deterministic three-component PCA of each frame heat-kernel row matrix",
            "comparison": "per-residue sum of three one-dimensional Wasserstein distances",
        },
        "frame_selection": frame_report,
        "resource_accounting": {
            "common_protein_atom_count": atom_count,
            "possible_selected_atom_pair_count": possible_pairs,
            "evaluated_included_atom_pair_frame_count": selected_pair_frame_evaluations,
            "retained_heat_kernel_element_count": total_kernel_elements,
        },
        "systems": system_reports, "pairwise_system_comparisons": comparisons,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "This is a residue-pair interaction-energy network, not a unique decomposition of the complete reciprocal-space PME energy.",
            "Direct pair energies are nonperiodic after declared made-whole coordinate processing, matching cpptraj pairwise rather than production PME periodic electrostatics.",
            "Solvent, ions, nucleic acids, ligands, and non-common or atom-incomplete protein residues are excluded to match the protein-only comparison protocol.",
            "OpenMM System XML with parameter offsets or CustomNonbondedForce expressions is rejected; raw GROMACS TPR extraction is unavailable.",
            "Absolute energies discard favorable-versus-unfavorable sign information; no signed-energy extension is implemented.",
            "Per-frame PCA and summed marginal Wasserstein distances reproduce the published supplement but do not preserve multivariate dependence and may be sensitive to near-degenerate PCA axes.",
            "Residue rankings are descriptive effect sizes, not p-values, binding affinities, causal pathways, or proof of allostery.",
        ],
    }


def energetic_network_embeddings_project_safe(
    project_path: Path, hash_content: bool = False,
) -> Dict[str, object]:
    try:
        return energetic_network_embeddings_project(
            project_path, hash_content=hash_content
        )
    except (
        EnergeticNetworkError, AtomMappingError, CoordinateReadError,
        ManifestValidationError, PeriodicReconstructionError, OSError,
        KeyError, TypeError, ValueError, np.linalg.LinAlgError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "energetic_network_embeddings",
            "technical_status": "failed", "scientific_status": "not evaluated",
            "availability_status": "unknown",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [{
                "severity": "error", "code": "ENERGETIC_NETWORK_INVALID",
                "message": message,
            } for message in messages],
        }
