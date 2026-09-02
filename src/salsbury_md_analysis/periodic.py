"""Connectivity-aware periodic reconstruction for production coordinate analysis.

The implementation operates only on explicit bond topologies and preserves
compact NumPy-backed DCD coordinates. It never guesses covalent connectivity
from coordinate distances.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import deque
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .coordinates import CellVectors, Coordinate, CoordinateFrame
from .manifests import resolve_manifest_path


Bond = Tuple[int, int]
_RECONSTRUCTION_POLICIES = {"make_whole", "unwrap_continuous"}
_PREPROCESSED_POLICY = "preprocessed_make_whole"


class PeriodicReconstructionError(ValueError):
    """Raised when periodic reconstruction cannot be established safely."""


def _cross(first: Coordinate, second: Coordinate) -> Coordinate:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _dot(first: Sequence[float], second: Sequence[float]) -> float:
    return sum(left * right for left, right in zip(first, second))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _reciprocal_rows(cell: CellVectors) -> Tuple[Coordinate, Coordinate, Coordinate]:
    a, b, c = cell
    b_cross_c = _cross(b, c)
    determinant = _dot(a, b_cross_c)
    scale = max(_norm(a), _norm(b), _norm(c))
    if not math.isfinite(determinant) or abs(determinant) <= 1.0e-12 * scale ** 3:
        raise PeriodicReconstructionError("periodic cell is singular or numerically degenerate")
    c_cross_a = _cross(c, a)
    a_cross_b = _cross(a, b)
    return tuple(
        tuple(value / determinant for value in vector)
        for vector in (b_cross_c, c_cross_a, a_cross_b)
    )  # type: ignore[return-value]


def _cartesian(cell: CellVectors, fractional: Sequence[float]) -> Coordinate:
    return tuple(
        sum(fractional[index] * cell[index][axis] for index in range(3))
        for axis in range(3)
    )  # type: ignore[return-value]


def minimum_image_displacement(
    displacement: Sequence[float], cell: CellVectors, maximum_candidates: int = 100_000
) -> Coordinate:
    """Return the exact nearest lattice image for a three-dimensional cell.

    Bounds derived from the reciprocal cell make the finite enumeration exact,
    including for skewed triclinic cells.  Extremely ill-conditioned cells fail
    rather than silently falling back to component-wise fractional rounding.
    """

    if len(displacement) != 3 or not all(math.isfinite(value) for value in displacement):
        raise PeriodicReconstructionError("displacement must contain three finite values")
    reciprocal = _reciprocal_rows(cell)
    fractional = tuple(_dot(row, displacement) for row in reciprocal)
    nearest = tuple(math.floor(value + 0.5) for value in fractional)
    candidate = _cartesian(
        cell, tuple(fractional[index] - nearest[index] for index in range(3))
    )
    radius = _norm(candidate)
    ranges = []
    candidate_count = 1
    for value, reciprocal_row in zip(fractional, reciprocal):
        bound = _norm(reciprocal_row) * radius + 1.0e-12
        lower = math.ceil(value - bound)
        upper = math.floor(value + bound)
        values = range(lower, upper + 1)
        candidate_count *= len(values)
        ranges.append(values)
    if candidate_count > maximum_candidates:
        raise PeriodicReconstructionError(
            f"triclinic nearest-image search requires {candidate_count} candidates; "
            f"safety gate is {maximum_candidates}"
        )
    best = candidate
    best_squared = _dot(best, best)
    for lattice in product(*ranges):
        trial = _cartesian(
            cell, tuple(fractional[index] - lattice[index] for index in range(3))
        )
        squared = _dot(trial, trial)
        if squared < best_squared - 1.0e-20:
            best = trial
            best_squared = squared
    return best


def _validated_bonds(
    raw_bonds: Iterable[Sequence[int]], atom_count: int, label: str
) -> Tuple[Bond, ...]:
    bonds = set()
    for index, raw in enumerate(raw_bonds):
        if len(raw) != 2:
            raise PeriodicReconstructionError(f"{label} bond {index} must contain two indices")
        first, second = raw
        if isinstance(first, bool) or isinstance(second, bool) or not isinstance(first, int) or not isinstance(second, int):
            raise PeriodicReconstructionError(f"{label} bond {index} indices must be integers")
        if first == second:
            raise PeriodicReconstructionError(f"{label} bond {index} is a self bond")
        if min(first, second) < 0 or max(first, second) >= atom_count:
            raise PeriodicReconstructionError(
                f"{label} bond {index} exceeds atom count {atom_count}"
            )
        bonds.add((min(first, second), max(first, second)))
    if atom_count > 1 and not bonds:
        raise PeriodicReconstructionError(f"{label} contains no bonds")
    return tuple(sorted(bonds))


def _load_json_bonds(path: Path) -> Tuple[int, Tuple[Bond, ...]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PeriodicReconstructionError(f"{path}: {exc}") from exc
    required = {"format", "atom_count", "index_base", "bonds"}
    allowed = required | {"provenance"}
    if not isinstance(payload, dict) or not required.issubset(payload) or not set(payload).issubset(allowed):
        raise PeriodicReconstructionError(
            "bond JSON requires format, atom_count, index_base, and bonds; provenance is optional"
        )
    if "provenance" in payload and not isinstance(payload["provenance"], dict):
        raise PeriodicReconstructionError("bond JSON provenance must be an object")
    if payload["format"] != "salsbury-bonds-v1" or payload["index_base"] != 0:
        raise PeriodicReconstructionError(
            "bond JSON requires format=salsbury-bonds-v1 and index_base=0"
        )
    atom_count = payload["atom_count"]
    if isinstance(atom_count, bool) or not isinstance(atom_count, int) or atom_count <= 0:
        raise PeriodicReconstructionError("bond JSON atom_count must be a positive integer")
    raw_bonds = payload["bonds"]
    if not isinstance(raw_bonds, list):
        raise PeriodicReconstructionError("bond JSON bonds must be an array")
    return atom_count, _validated_bonds(raw_bonds, atom_count, str(path))


def _load_psf_bonds(path: Path) -> Tuple[int, Tuple[Bond, ...]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PeriodicReconstructionError(f"{path}: {exc}") from exc
    atom_count = None
    bond_count = None
    bond_values: List[int] = []
    collecting = False
    for line in lines:
        upper = line.upper()
        if "!NATOM" in upper:
            try:
                atom_count = int(line.split()[0])
            except (IndexError, ValueError) as exc:
                raise PeriodicReconstructionError("PSF !NATOM declaration is malformed") from exc
        if "!NBOND" in upper:
            try:
                bond_count = int(line.split()[0])
            except (IndexError, ValueError) as exc:
                raise PeriodicReconstructionError("PSF !NBOND declaration is malformed") from exc
            collecting = True
            continue
        if collecting:
            if "!" in line:
                break
            for field in line.split():
                try:
                    bond_values.append(int(field))
                except ValueError as exc:
                    raise PeriodicReconstructionError("PSF bond section contains a noninteger") from exc
    if atom_count is None or atom_count <= 0 or bond_count is None or bond_count < 0:
        raise PeriodicReconstructionError("PSF must declare positive !NATOM and nonnegative !NBOND")
    if len(bond_values) != 2 * bond_count:
        raise PeriodicReconstructionError(
            f"PSF declares {bond_count} bonds but contains {len(bond_values) // 2} pairs"
        )
    raw_bonds = [
        (bond_values[index] - 1, bond_values[index + 1] - 1)
        for index in range(0, len(bond_values), 2)
    ]
    return atom_count, _validated_bonds(raw_bonds, atom_count, str(path))


def _prmtop_sections(path: Path) -> Dict[str, List[int]]:
    sections: Dict[str, List[int]] = {}
    current: Optional[str] = None
    integer_width: Optional[int] = None
    required_sections = {"POINTERS", "BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"}
    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("%FLAG "):
                    current = stripped[6:].strip().upper()
                    sections.setdefault(current, [])
                    integer_width = None
                elif stripped.startswith("%FORMAT"):
                    match = re.fullmatch(r"%FORMAT\(\s*\d+I(\d+)\s*\)", stripped, re.IGNORECASE)
                    integer_width = int(match.group(1)) if match else None
                elif not stripped or current is None:
                    continue
                elif current in required_sections:
                    fields = (
                        [line[index : index + integer_width] for index in range(0, len(line.rstrip("\n")), integer_width)]
                        if integer_width is not None
                        else stripped.split()
                    )
                    for field in fields:
                        if not field.strip():
                            continue
                        try:
                            sections[current].append(int(field.strip()))
                        except ValueError as exc:
                            raise PeriodicReconstructionError(
                                f"Amber {current} section contains a noninteger"
                            ) from exc
    except (OSError, UnicodeError) as exc:
        raise PeriodicReconstructionError(f"{path}: {exc}") from exc
    return sections


def _load_prmtop_bonds(path: Path) -> Tuple[int, Tuple[Bond, ...]]:
    sections = _prmtop_sections(path)
    pointers = sections.get("POINTERS", [])
    if not pointers or pointers[0] <= 0:
        raise PeriodicReconstructionError("Amber POINTERS section does not declare NATOM")
    atom_count = pointers[0]
    raw_bonds = []
    for section in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
        values = sections.get(section)
        if values is None or len(values) % 3:
            raise PeriodicReconstructionError(f"Amber {section} section is absent or malformed")
        for index in range(0, len(values), 3):
            first, second = values[index], values[index + 1]
            if first % 3 or second % 3:
                raise PeriodicReconstructionError(
                    f"Amber {section} atom pointers are not multiples of three"
                )
            raw_bonds.append((first // 3, second // 3))
    return atom_count, _validated_bonds(raw_bonds, atom_count, str(path))


def load_connectivity(path: Path, expected_atom_count: int) -> Tuple[Tuple[Bond, ...], Dict[str, object]]:
    """Load a complete explicit bond topology and verify its coordinate cardinality."""

    source = Path(path).expanduser().resolve(strict=False)
    suffix = source.suffix.lower()
    if suffix == ".json":
        format_name = "salsbury-bonds-v1"
        atom_count, bonds = _load_json_bonds(source)
    elif suffix == ".psf":
        format_name = "psf"
        atom_count, bonds = _load_psf_bonds(source)
    elif suffix in {".prmtop", ".parm7"}:
        format_name = "amber-prmtop"
        atom_count, bonds = _load_prmtop_bonds(source)
    else:
        raise PeriodicReconstructionError(
            f"unsupported connectivity format {suffix or '<none>'}; supported: .json, .psf, .prmtop, .parm7"
        )
    if atom_count != expected_atom_count:
        raise PeriodicReconstructionError(
            f"connectivity has {atom_count} atoms; coordinates/topology have {expected_atom_count}"
        )
    try:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as exc:
        raise PeriodicReconstructionError(f"{source}: {exc}") from exc
    return bonds, {
        "path": str(source),
        "format": format_name,
        "sha256": digest,
        "atom_count": atom_count,
        "bond_count": len(bonds),
    }


def connected_components(atom_count: int, bonds: Sequence[Bond]) -> Tuple[Tuple[int, ...], ...]:
    adjacency = [[] for _ in range(atom_count)]
    for first, second in bonds:
        adjacency[first].append(second)
        adjacency[second].append(first)
    components = []
    unseen = set(range(atom_count))
    while unseen:
        root = min(unseen)
        queue = deque((root,))
        unseen.remove(root)
        component = []
        while queue:
            atom = queue.popleft()
            component.append(atom)
            for neighbor in sorted(adjacency[atom]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _make_whole_components(
    coordinates: Sequence[Coordinate],
    cell: CellVectors,
    adjacency: Sequence[Sequence[int]],
    bonds: Sequence[Bond],
    components: Sequence[Sequence[int]],
    maximum_bond_length_angstrom: float,
    cycle_closure_tolerance_angstrom: float,
) -> Sequence[Coordinate]:
    """Rebuild the supplied complete bonded components."""

    array_backed = isinstance(coordinates, np.ndarray)
    if array_backed:
        coordinate_array = np.asarray(coordinates, dtype=float)
        if coordinate_array.shape != (len(coordinates), 3):
            raise PeriodicReconstructionError(
                "coordinate array must have shape (atom_count, 3)"
            )
        rebuilt = coordinate_array.copy()
    else:
        rebuilt = list(coordinates)
    active_atoms = {atom for component in components for atom in component}
    visited = set()
    for component in components:
        root = component[0]
        rebuilt[root] = tuple(coordinates[root])
        visited.add(root)
        queue = deque((root,))
        while queue:
            parent = queue.popleft()
            parent_coordinate = rebuilt[parent]
            for child in sorted(adjacency[parent]):
                raw_delta = tuple(
                    coordinates[child][axis] - coordinates[parent][axis]
                    for axis in range(3)
                )
                delta = minimum_image_displacement(raw_delta, cell)
                length = _norm(delta)
                if length > maximum_bond_length_angstrom:
                    raise PeriodicReconstructionError(
                        f"bond {parent}-{child} minimum-image length {length:.6g} angstrom "
                        f"exceeds gate {maximum_bond_length_angstrom:.6g}"
                    )
                candidate = tuple(
                    parent_coordinate[axis] + delta[axis] for axis in range(3)
                )
                if child not in visited:
                    rebuilt[child] = candidate
                    visited.add(child)
                    queue.append(child)
                else:
                    residual = _norm(tuple(candidate[axis] - rebuilt[child][axis] for axis in range(3)))
                    if residual > cycle_closure_tolerance_angstrom:
                        raise PeriodicReconstructionError(
                            f"bond cycle closure residual {residual:.6g} angstrom exceeds gate "
                            f"{cycle_closure_tolerance_angstrom:.6g} at bond {parent}-{child}"
                        )
    result = rebuilt if array_backed else tuple(rebuilt)
    for first, second in bonds:
        if first not in active_atoms:
            continue
        length = _norm(tuple(result[second][axis] - result[first][axis] for axis in range(3)))
        if length > maximum_bond_length_angstrom:
            raise PeriodicReconstructionError(
                f"reconstructed bond {first}-{second} length {length:.6g} angstrom exceeds gate "
                f"{maximum_bond_length_angstrom:.6g}"
            )
    return result


def make_whole_coordinates(
    coordinates: Sequence[Coordinate],
    cell: CellVectors,
    bonds: Sequence[Bond],
    maximum_bond_length_angstrom: float,
    cycle_closure_tolerance_angstrom: float,
    required_atom_indices: Optional[Sequence[int]] = None,
) -> Tuple[Coordinate, ...]:
    """Rebuild relevant bonded components using minimum-image bond vectors.

    When ``required_atom_indices`` is supplied, every complete bonded component
    containing at least one required atom is rebuilt and all other coordinates
    are retained verbatim.  This is numerically identical for observables that
    access only the declared required atoms, while avoiding reconstruction of
    tens of thousands of unrelated solvent atoms.
    """

    atom_count = len(coordinates)
    adjacency = [[] for _ in range(atom_count)]
    for first, second in bonds:
        if max(first, second) >= atom_count:
            raise PeriodicReconstructionError("bond index exceeds coordinate atom count")
        adjacency[first].append(second)
        adjacency[second].append(first)
    components = connected_components(atom_count, bonds)
    if required_atom_indices is not None:
        required = set()
        for index in required_atom_indices:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < atom_count:
                raise PeriodicReconstructionError(
                    "required reconstruction atom indices must be valid integers"
                )
            required.add(index)
        if not required:
            raise PeriodicReconstructionError(
                "required reconstruction atom indices must be nonempty"
            )
        components = tuple(
            component for component in components
            if any(atom in required for atom in component)
        )
    return _make_whole_components(
        coordinates,
        cell,
        adjacency,
        bonds,
        components,
        maximum_bond_length_angstrom,
        cycle_closure_tolerance_angstrom,
    )


def reconstruction_settings(project: Mapping[str, object], policy: str) -> Dict[str, float]:
    if policy not in _RECONSTRUCTION_POLICIES:
        return {}
    raw = project.get("periodic_reconstruction")
    if not isinstance(raw, dict):
        raise PeriodicReconstructionError(
            "periodic_reconstruction settings are required for make_whole or unwrap_continuous"
        )
    required = {"maximum_bond_length_angstrom", "cycle_closure_tolerance_angstrom"}
    if policy == "unwrap_continuous":
        required.add("maximum_anchor_displacement_angstrom")
    unknown = sorted(set(raw).difference(required))
    missing = sorted(required.difference(raw))
    if missing or unknown:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise PeriodicReconstructionError("periodic_reconstruction settings invalid (" + "; ".join(details) + ")")
    result = {}
    for key in required:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise PeriodicReconstructionError(f"periodic_reconstruction.{key} must be finite and positive")
        result[key] = float(value)
    return result


class PeriodicFrameProcessor:
    """Stateful per-replica make-whole and continuous-unwrapping processor."""

    def __init__(
        self,
        policy: str,
        atom_count: int,
        bonds: Sequence[Bond] = (),
        settings: Optional[Mapping[str, float]] = None,
        connectivity_identity: Optional[Mapping[str, object]] = None,
        preprocessed_cache_identity: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.policy = policy
        self.atom_count = atom_count
        self.bonds = tuple(bonds)
        self.settings = dict(settings or {})
        self.connectivity_identity = dict(connectivity_identity or {})
        self.preprocessed_cache_identity = dict(preprocessed_cache_identity or {})
        adjacency = [[] for _ in range(atom_count)]
        for first, second in self.bonds:
            adjacency[first].append(second)
            adjacency[second].append(first)
        self.adjacency = tuple(tuple(sorted(neighbors)) for neighbors in adjacency)
        self.components = connected_components(atom_count, self.bonds)
        self._active_components_cache: Dict[
            Tuple[int, ...], Tuple[Tuple[int, ...], ...]
        ] = {}
        self._previous_anchors: Optional[Tuple[Coordinate, ...]] = None
        self._previous_component_signature: Optional[Tuple[int, ...]] = None
        self.periodic_frame_count = 0
        self.reconstructed_frame_count = 0
        self.reconstructed_component_count = 0
        self.reconstructed_atom_count = 0
        self.maximum_anchor_displacement_observed = 0.0

    @classmethod
    def from_replica(
        cls,
        project: Mapping[str, object],
        replica: Mapping[str, object],
        system_manifest_path: Path,
        atom_count: int,
        *,
        independent_frames: bool = False,
    ) -> "PeriodicFrameProcessor":
        declared_policy = str(project.get("periodic_coordinate_policy"))
        settings = reconstruction_settings(project, declared_policy)
        policy = (
            "make_whole"
            if independent_frames and declared_policy == "unwrap_continuous"
            else declared_policy
        )
        if policy == "make_whole" and declared_policy == "unwrap_continuous":
            settings = {
                key: settings[key]
                for key in (
                    "maximum_bond_length_angstrom",
                    "cycle_closure_tolerance_angstrom",
                )
            }
        if policy == _PREPROCESSED_POLICY:
            declared = project.get("preprocessed_coordinate_source")
            if not isinstance(declared, dict):
                raise PeriodicReconstructionError(
                    "preprocessed_make_whole requires preprocessed_coordinate_source"
                )
            report_value = declared.get("cache_report")
            expected_digest = declared.get("cache_report_sha256")
            if not isinstance(report_value, str) or not isinstance(expected_digest, str):
                raise PeriodicReconstructionError(
                    "preprocessed_coordinate_source requires cache_report and cache_report_sha256"
                )
            report_path = resolve_manifest_path(report_value, system_manifest_path)
            try:
                actual_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise PeriodicReconstructionError(str(exc)) from exc
            if actual_digest.lower() != expected_digest.lower():
                raise PeriodicReconstructionError(
                    "preprocessed coordinate cache report hash does not match"
                )
            if not isinstance(report, dict) or report.get("technical_status") != "complete":
                raise PeriodicReconstructionError(
                    "preprocessed coordinate cache report is not technically complete"
                )
            if report.get("coordinate_representation") != (
                "continuous_unwrap_unaligned_strided"
            ):
                raise PeriodicReconstructionError(
                    "preprocessed coordinate cache report has the wrong representation"
                )
            if report.get("selection") != "molecular_payload":
                raise PeriodicReconstructionError(
                    "preprocessed coordinate cache report has the wrong selection"
                )
            cached_manifest = report.get("cached_system_manifest")
            cached_manifest_digest = report.get("cached_system_manifest_sha256")
            if not isinstance(cached_manifest, str) or not isinstance(
                cached_manifest_digest, str
            ):
                raise PeriodicReconstructionError(
                    "preprocessed coordinate cache report lacks manifest identity"
                )
            reported_manifest_path = resolve_manifest_path(
                cached_manifest, report_path
            )
            actual_manifest_path = Path(system_manifest_path).expanduser().resolve(
                strict=False
            )
            if reported_manifest_path != actual_manifest_path:
                raise PeriodicReconstructionError(
                    "preprocessed cache report names a different system manifest"
                )
            try:
                manifest_digest = hashlib.sha256(
                    actual_manifest_path.read_bytes()
                ).hexdigest()
            except OSError as exc:
                raise PeriodicReconstructionError(str(exc)) from exc
            if manifest_digest.lower() != cached_manifest_digest.lower():
                raise PeriodicReconstructionError(
                    "preprocessed cache system manifest hash does not match"
                )
            return cls(
                policy,
                atom_count,
                settings=settings,
                preprocessed_cache_identity={
                    "cache_report": str(report_path),
                    "cache_report_sha256": actual_digest,
                    "cached_system_manifest_sha256": manifest_digest,
                    "coordinate_representation": report[
                        "coordinate_representation"
                    ],
                },
            )
        if policy not in _RECONSTRUCTION_POLICIES:
            return cls(policy, atom_count)
        connectivity = replica.get("connectivity")
        if not isinstance(connectivity, str) or not connectivity.strip():
            raise PeriodicReconstructionError(
                "replica connectivity path is required for make_whole or unwrap_continuous"
            )
        path = resolve_manifest_path(connectivity, system_manifest_path)
        bonds, identity = load_connectivity(path, atom_count)
        return cls(policy, atom_count, bonds, settings, identity)

    @classmethod
    def from_reference(
        cls,
        project: Mapping[str, object],
        project_manifest_path: Path,
        atom_count: int,
    ) -> "PeriodicFrameProcessor":
        policy = str(project.get("periodic_coordinate_policy"))
        settings = reconstruction_settings(project, policy)
        if policy not in _RECONSTRUCTION_POLICIES:
            return cls(policy, atom_count)
        connectivity = project.get("reference_connectivity")
        if not isinstance(connectivity, str) or not connectivity.strip():
            return cls(policy, atom_count, (), settings)
        path = resolve_manifest_path(connectivity, project_manifest_path)
        bonds, identity = load_connectivity(path, atom_count)
        return cls(policy, atom_count, bonds, settings, identity)

    def begin_segment(self, continuous_with_previous: bool) -> None:
        if not continuous_with_previous:
            self._previous_anchors = None
            self._previous_component_signature = None

    def checkpoint_state(self) -> Dict[str, object]:
        """Return the mutable reconstruction state at a frame boundary."""

        return {
            "schema": "periodic_frame_processor_checkpoint_v1",
            "policy": self.policy,
            "atom_count": self.atom_count,
            "previous_anchors": (
                [list(point) for point in self._previous_anchors]
                if self._previous_anchors is not None else None
            ),
            "previous_component_signature": (
                list(self._previous_component_signature)
                if self._previous_component_signature is not None else None
            ),
            "periodic_frame_count": self.periodic_frame_count,
            "reconstructed_frame_count": self.reconstructed_frame_count,
            "reconstructed_component_count": self.reconstructed_component_count,
            "reconstructed_atom_count": self.reconstructed_atom_count,
            "maximum_anchor_displacement_observed": (
                self.maximum_anchor_displacement_observed
            ),
        }

    def restore_checkpoint_state(self, state: Mapping[str, object]) -> None:
        """Restore a state produced by :meth:`checkpoint_state` fail-closed."""

        expected = {
            "schema", "policy", "atom_count", "previous_anchors",
            "previous_component_signature", "periodic_frame_count",
            "reconstructed_frame_count", "reconstructed_component_count",
            "reconstructed_atom_count", "maximum_anchor_displacement_observed",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise PeriodicReconstructionError(
                "periodic checkpoint state has an unsupported schema"
            )
        if (
            state["schema"] != "periodic_frame_processor_checkpoint_v1"
            or state["policy"] != self.policy
            or state["atom_count"] != self.atom_count
        ):
            raise PeriodicReconstructionError(
                "periodic checkpoint state does not match this processor"
            )
        anchors = state["previous_anchors"]
        if anchors is None:
            self._previous_anchors = None
        elif isinstance(anchors, list) and all(
            isinstance(point, list) and len(point) == 3
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in point
            )
            for point in anchors
        ):
            self._previous_anchors = tuple(
                tuple(float(value) for value in point) for point in anchors
            )
        else:
            raise PeriodicReconstructionError(
                "periodic checkpoint anchors are invalid"
            )
        signature = state["previous_component_signature"]
        if signature is None:
            self._previous_component_signature = None
        elif isinstance(signature, list) and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in signature
        ):
            self._previous_component_signature = tuple(signature)
        else:
            raise PeriodicReconstructionError(
                "periodic checkpoint component signature is invalid"
            )
        for field in (
            "periodic_frame_count", "reconstructed_frame_count",
            "reconstructed_component_count", "reconstructed_atom_count",
        ):
            value = state[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PeriodicReconstructionError(
                    f"periodic checkpoint {field} is invalid"
                )
            setattr(self, field, value)
        maximum = state["maximum_anchor_displacement_observed"]
        if (
            isinstance(maximum, bool) or not isinstance(maximum, (int, float))
            or not math.isfinite(float(maximum)) or float(maximum) < 0.0
        ):
            raise PeriodicReconstructionError(
                "periodic checkpoint maximum anchor displacement is invalid"
            )
        self.maximum_anchor_displacement_observed = float(maximum)

    def process(
        self,
        frame: CoordinateFrame,
        location: str,
        required_atom_indices: Optional[Sequence[int]] = None,
    ) -> CoordinateFrame:
        if frame.atom_count != self.atom_count:
            raise PeriodicReconstructionError(
                f"{location}: frame has {frame.atom_count} atoms; expected {self.atom_count}"
            )
        if not frame.periodic_cell_present:
            if self.policy == "unwrap_continuous":
                self._previous_anchors = None
            return frame
        self.periodic_frame_count += 1
        if self.policy == "reject":
            raise PeriodicReconstructionError(
                f"{location} declares a periodic cell; periodic_coordinate_policy=reject"
            )
        if self.policy == "allow_wrapped_diagnostic":
            return frame
        if self.policy == _PREPROCESSED_POLICY:
            if not self.preprocessed_cache_identity:
                raise PeriodicReconstructionError(
                    f"{location}: preprocessed coordinate cache was not verified"
                )
            if (
                frame.coordinate_representation
                != "made_whole_molecular_payload_cache"
            ):
                raise PeriodicReconstructionError(
                    f"{location}: trajectory is not a declared made-whole molecular-payload cache"
                )
            self.reconstructed_frame_count += 1
            return replace(
                frame, coordinate_representation="preprocessed_make_whole"
            )
        if self.policy not in _RECONSTRUCTION_POLICIES:
            raise PeriodicReconstructionError(f"unsupported periodic policy {self.policy!r}")
        if frame.cell_vectors_angstrom is None:
            raise PeriodicReconstructionError(
                f"{location} declares a periodic cell but provides no usable cell vectors"
            )
        if not self.bonds:
            raise PeriodicReconstructionError(
                f"{location} is periodic but no explicit connectivity was loaded"
            )
        required = None
        if required_atom_indices is not None:
            required = tuple(sorted(set(required_atom_indices)))
            if not required or any(
                isinstance(index, bool) or not isinstance(index, int)
                or index < 0 or index >= self.atom_count
                for index in required
            ):
                raise PeriodicReconstructionError(
                    f"{location}: required reconstruction atom indices must be nonempty valid integers"
                )
        required_set = set(required or ())
        if required is None:
            active_components = self.components
        else:
            active_components = self._active_components_cache.get(required)
            if active_components is None:
                active_components = tuple(
                    component for component in self.components
                    if any(atom in required_set for atom in component)
                )
                self._active_components_cache[required] = active_components
        component_signature = tuple(component[0] for component in active_components)
        if (
            self.policy == "unwrap_continuous"
            and self._previous_component_signature is not None
            and self._previous_component_signature != component_signature
        ):
            raise PeriodicReconstructionError(
                f"{location}: required reconstruction components changed during continuous unwrapping"
            )
        coordinates = _make_whole_components(
            frame.coordinates_angstrom,
            frame.cell_vectors_angstrom,
            self.adjacency,
            self.bonds,
            active_components,
            self.settings["maximum_bond_length_angstrom"],
            self.settings["cycle_closure_tolerance_angstrom"],
        )
        self.reconstructed_component_count = len(active_components)
        self.reconstructed_atom_count = sum(len(component) for component in active_components)
        representation = "make_whole"
        if self.policy == "unwrap_continuous":
            anchors = tuple(
                tuple(float(value) for value in coordinates[component[0]])
                for component in active_components
            )
            if self._previous_anchors is not None:
                array_backed = isinstance(coordinates, np.ndarray)
                shifted = coordinates.copy() if array_backed else list(coordinates)
                next_anchors = []
                for component, current, previous in zip(active_components, anchors, self._previous_anchors):
                    raw_delta = tuple(current[axis] - previous[axis] for axis in range(3))
                    delta = minimum_image_displacement(raw_delta, frame.cell_vectors_angstrom)
                    distance = _norm(delta)
                    gate = self.settings["maximum_anchor_displacement_angstrom"]
                    if distance > gate:
                        raise PeriodicReconstructionError(
                            f"{location}: component anchor displacement {distance:.6g} angstrom "
                            f"exceeds gate {gate:.6g}"
                        )
                    self.maximum_anchor_displacement_observed = max(
                        self.maximum_anchor_displacement_observed, distance
                    )
                    target = tuple(previous[axis] + delta[axis] for axis in range(3))
                    shift = tuple(target[axis] - current[axis] for axis in range(3))
                    for atom in component:
                        shifted[atom] = tuple(
                            coordinates[atom][axis] + shift[axis] for axis in range(3)
                        )
                    next_anchors.append(target)
                coordinates = shifted if array_backed else tuple(shifted)
                anchors = tuple(next_anchors)
            self._previous_anchors = anchors
            self._previous_component_signature = component_signature
            representation = "unwrap_continuous"
        self.reconstructed_frame_count += 1
        return replace(
            frame,
            coordinates_angstrom=coordinates,
            coordinate_representation=representation,
        )

    def report(self) -> Dict[str, object]:
        return {
            "policy": self.policy,
            "periodic_frame_count": self.periodic_frame_count,
            "reconstructed_frame_count": self.reconstructed_frame_count,
            "component_count": len(self.components) if self.bonds else None,
            "reconstructed_component_count": (
                self.reconstructed_component_count if self.reconstructed_frame_count else None
            ),
            "reconstructed_atom_count": (
                self.reconstructed_atom_count if self.reconstructed_frame_count else None
            ),
            "maximum_anchor_displacement_observed_angstrom": (
                self.maximum_anchor_displacement_observed
                if self.policy == "unwrap_continuous" and self.reconstructed_frame_count > 1
                else None
            ),
            "connectivity": self.connectivity_identity or None,
            "preprocessed_cache": self.preprocessed_cache_identity or None,
        }
