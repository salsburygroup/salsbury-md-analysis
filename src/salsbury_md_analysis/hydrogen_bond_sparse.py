"""Chunked, sparse direct-hydrogen-bond frame evaluation.

The sparse representation has an explicit-zero contract: candidate indices not
listed for a frame/cutoff are absent under that cutoff.  Chemistry defines the
candidate universe before coordinates are evaluated, so sparse storage does
not introduce occupancy-based feature selection.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .coordinates import CellVectors
from .hydrogen_bonds import angle_degrees, distance_angstrom
from .periodic import minimum_image_displacement


class SparseHydrogenBondError(ValueError):
    """Raised when a sparse hydrogen-bond representation is inconsistent."""


PACKED_EVENT_CODEC = "base64-little-endian-u32-u32-f32-f32-v1"
PACKED_CUTOFF_COUNT_CODEC = "base64-little-endian-u32-u8-u32-v1"
_PACKED_EVENT_DTYPE = np.dtype([
    ("candidate_index", "<u4"),
    ("cutoff_mask", "<u4"),
    ("donor_acceptor_distance_angstrom", "<f4"),
    ("donor_hydrogen_acceptor_angle_degrees", "<f4"),
])
_PACKED_CUTOFF_COUNT_DTYPE = np.dtype([
    ("candidate_index", "<u4"),
    ("cutoff_index", "u1"),
    ("present_frame_count", "<u4"),
])


def pack_sparse_present_geometry(
    present_geometry: Sequence[Mapping[str, object]],
    cutoff_definitions: Sequence[Mapping[str, object]],
    candidate_count: int,
) -> Dict[str, object]:
    """Pack exact present-event geometry into a bounded per-frame payload.

    One event stores one candidate index, one cutoff-membership bit mask, and
    two IEEE-754 single-precision geometry values.  This removes duplicated
    bond identifiers, per-cutoff index arrays, and Python dictionaries while
    retaining exact frame locators and reproducible cutoff membership.
    """

    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 1
        or candidate_count > np.iinfo(np.uint32).max
    ):
        raise SparseHydrogenBondError(
            "packed candidate count must fit an unsigned 32-bit integer"
        )
    cutoff_ids = [str(row.get("cutoff_id")) for row in cutoff_definitions]
    if (
        not cutoff_ids or len(cutoff_ids) > 32
        or len(set(cutoff_ids)) != len(cutoff_ids)
        or any(not value for value in cutoff_ids)
    ):
        raise SparseHydrogenBondError(
            "packed sparse frames require one to 32 unique cutoff identifiers"
        )
    cutoff_bits = {cutoff_id: 1 << index for index, cutoff_id in enumerate(cutoff_ids)}
    packed = np.empty(len(present_geometry), dtype=_PACKED_EVENT_DTYPE)
    prior_index = -1
    for row_index, row in enumerate(present_geometry):
        candidate_index = row.get("candidate_index")
        matched = row.get("present_cutoff_ids")
        distance = row.get("donor_acceptor_distance_angstrom")
        angle = row.get("donor_hydrogen_acceptor_angle_degrees")
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or candidate_index <= prior_index
            or candidate_index < 0
            or candidate_index >= candidate_count
        ):
            raise SparseHydrogenBondError(
                "packed present candidate indices must be sorted, unique, and in range"
            )
        if not isinstance(matched, list) or not matched:
            raise SparseHydrogenBondError(
                "packed present geometry requires nonempty cutoff membership"
            )
        mask = 0
        for cutoff_id in matched:
            if cutoff_id not in cutoff_bits:
                raise SparseHydrogenBondError(
                    "packed present geometry references an unknown cutoff"
                )
            mask |= cutoff_bits[str(cutoff_id)]
        if (
            isinstance(distance, bool) or not isinstance(distance, (int, float))
            or isinstance(angle, bool) or not isinstance(angle, (int, float))
            or not math.isfinite(float(distance)) or not math.isfinite(float(angle))
        ):
            raise SparseHydrogenBondError(
                "packed present-event geometry must be finite"
            )
        packed[row_index] = (
            candidate_index, mask, float(distance), float(angle),
        )
        prior_index = candidate_index
    return {
        "representation": "sparse_packed_v2",
        "packed_event_codec": PACKED_EVENT_CODEC,
        "packed_event_count": len(present_geometry),
        "cutoff_ids": cutoff_ids,
        "packed_present_events_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def unpack_sparse_present_events(frame: Mapping[str, object]) -> np.ndarray:
    """Validate and decode one packed sparse frame without copying its payload."""

    if frame.get("representation") != "sparse_packed_v2":
        raise SparseHydrogenBondError("frame is not sparse_packed_v2")
    if frame.get("packed_event_codec") != PACKED_EVENT_CODEC:
        raise SparseHydrogenBondError("packed sparse frame codec is unsupported")
    count = frame.get("packed_event_count")
    payload = frame.get("packed_present_events_b64")
    if (
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        or not isinstance(payload, str)
    ):
        raise SparseHydrogenBondError("packed sparse frame metadata is invalid")
    try:
        raw = base64.b64decode(payload.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise SparseHydrogenBondError("packed sparse frame is not valid base64") from exc
    if len(raw) != count * _PACKED_EVENT_DTYPE.itemsize:
        raise SparseHydrogenBondError("packed sparse frame byte count is inconsistent")
    events = np.frombuffer(raw, dtype=_PACKED_EVENT_DTYPE, count=count)
    candidate_count = frame.get("candidate_count")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 1
    ):
        raise SparseHydrogenBondError("packed sparse frame candidate count is invalid")
    indices = events["candidate_index"]
    if indices.size and (
        int(indices[-1]) >= candidate_count
        or np.any(indices[1:] <= indices[:-1])
    ):
        raise SparseHydrogenBondError(
            "packed sparse candidate indices are not sorted, unique, and in range"
        )
    cutoff_ids = frame.get("cutoff_ids")
    if (
        not isinstance(cutoff_ids, list) or not cutoff_ids
        or len(cutoff_ids) > 32 or len(set(cutoff_ids)) != len(cutoff_ids)
        or any(not isinstance(value, str) or not value for value in cutoff_ids)
    ):
        raise SparseHydrogenBondError("packed sparse cutoff identifiers are invalid")
    valid_mask = (1 << len(cutoff_ids)) - 1
    masks = events["cutoff_mask"]
    invalid_mask = np.uint32((~valid_mask) & 0xFFFFFFFF)
    if masks.size and (
        np.any(masks == 0) or np.any(np.bitwise_and(masks, invalid_mask) != 0)
    ):
        raise SparseHydrogenBondError("packed sparse cutoff mask is invalid")
    for field in (
        "donor_acceptor_distance_angstrom",
        "donor_hydrogen_acceptor_angle_degrees",
    ):
        if not np.isfinite(events[field]).all():
            raise SparseHydrogenBondError("packed sparse geometry is non-finite")
    return events


def packed_present_indices(
    frame: Mapping[str, object], cutoff_id: str,
) -> List[int]:
    """Return candidate indices present under one cutoff in a packed frame."""

    cutoff_ids = frame.get("cutoff_ids")
    if not isinstance(cutoff_ids, list) or cutoff_id not in cutoff_ids:
        raise SparseHydrogenBondError(
            f"packed sparse frame lacks requested cutoff {cutoff_id!r}"
        )
    bit = np.uint32(1 << cutoff_ids.index(cutoff_id))
    events = unpack_sparse_present_events(frame)
    selected = events["candidate_index"][(events["cutoff_mask"] & bit) != 0]
    return [int(value) for value in selected]


def pack_sparse_cutoff_counts(
    cutoff_counts: Sequence[Mapping[int, int]],
    cutoff_definitions: Sequence[Mapping[str, object]],
    candidate_count: int,
    evaluated_frame_count: int,
) -> Dict[str, object]:
    """Pack nonzero candidate/cutoff occupancy counts for one segment."""

    if (
        isinstance(candidate_count, bool) or not isinstance(candidate_count, int)
        or candidate_count < 1 or candidate_count > np.iinfo(np.uint32).max
        or isinstance(evaluated_frame_count, bool)
        or not isinstance(evaluated_frame_count, int)
        or evaluated_frame_count < 1
        or evaluated_frame_count > np.iinfo(np.uint32).max
    ):
        raise SparseHydrogenBondError("packed cutoff-count dimensions are invalid")
    if (
        not cutoff_counts or len(cutoff_counts) != len(cutoff_definitions)
        or len(cutoff_counts) > np.iinfo(np.uint8).max
    ):
        raise SparseHydrogenBondError(
            "packed cutoff counts must match one to 255 cutoff definitions"
        )
    rows = []
    for cutoff_index, counts in enumerate(cutoff_counts):
        if not isinstance(counts, Mapping):
            raise SparseHydrogenBondError("packed cutoff counts must be mappings")
        for candidate_index, count in sorted(counts.items()):
            if (
                isinstance(candidate_index, bool)
                or not isinstance(candidate_index, int)
                or candidate_index < 0 or candidate_index >= candidate_count
                or isinstance(count, bool) or not isinstance(count, int)
                or count < 1 or count > evaluated_frame_count
            ):
                raise SparseHydrogenBondError(
                    "packed cutoff occupancy count is invalid"
                )
            rows.append((candidate_index, cutoff_index, count))
    packed = np.asarray(rows, dtype=_PACKED_CUTOFF_COUNT_DTYPE)
    return {
        "representation": "sparse_packed_cutoff_counts_v1",
        "packed_count_codec": PACKED_CUTOFF_COUNT_CODEC,
        "packed_count_record_count": len(rows),
        "cutoff_ids": [str(row["cutoff_id"]) for row in cutoff_definitions],
        "candidate_count": candidate_count,
        "evaluated_frame_count": evaluated_frame_count,
        "packed_counts_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def unpack_sparse_cutoff_counts(segment: Mapping[str, object]) -> np.ndarray:
    """Validate and decode one packed segment cutoff-occupancy table."""

    if segment.get("representation") != "sparse_packed_cutoff_counts_v1":
        raise SparseHydrogenBondError("segment is not sparse_packed_cutoff_counts_v1")
    if segment.get("packed_count_codec") != PACKED_CUTOFF_COUNT_CODEC:
        raise SparseHydrogenBondError("packed cutoff-count codec is unsupported")
    count = segment.get("packed_count_record_count")
    payload = segment.get("packed_counts_b64")
    candidate_count = segment.get("candidate_count")
    evaluated = segment.get("evaluated_frame_count")
    cutoff_ids = segment.get("cutoff_ids")
    if (
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        or not isinstance(payload, str)
        or isinstance(candidate_count, bool) or not isinstance(candidate_count, int)
        or candidate_count < 1
        or isinstance(evaluated, bool) or not isinstance(evaluated, int)
        or evaluated < 1
        or not isinstance(cutoff_ids, list) or not cutoff_ids
        or len(cutoff_ids) > np.iinfo(np.uint8).max
        or any(not isinstance(value, str) or not value for value in cutoff_ids)
    ):
        raise SparseHydrogenBondError("packed cutoff-count metadata is invalid")
    try:
        raw = base64.b64decode(payload.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise SparseHydrogenBondError("packed cutoff counts are not valid base64") from exc
    if len(raw) != count * _PACKED_CUTOFF_COUNT_DTYPE.itemsize:
        raise SparseHydrogenBondError("packed cutoff-count byte count is inconsistent")
    rows = np.frombuffer(raw, dtype=_PACKED_CUTOFF_COUNT_DTYPE, count=count)
    if rows.size and (
        int(rows["candidate_index"].max()) >= candidate_count
        or int(rows["cutoff_index"].max()) >= len(cutoff_ids)
        or int(rows["present_frame_count"].min()) < 1
        or int(rows["present_frame_count"].max()) > evaluated
    ):
        raise SparseHydrogenBondError("packed cutoff-count record is out of range")
    return rows


def candidate_chunks(
    candidates: Sequence[Mapping[str, object]], chunk_size: int
) -> Iterable[tuple[int, Sequence[Mapping[str, object]]]]:
    """Yield deterministic contiguous candidate chunks."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise SparseHydrogenBondError("candidate chunk size must be a positive integer")
    for start in range(0, len(candidates), chunk_size):
        yield start, candidates[start:start + chunk_size]


def evaluate_sparse_frame(
    coordinates: Sequence[Sequence[float]],
    candidates: Sequence[Mapping[str, object]],
    cutoff_definitions: Sequence[Mapping[str, object]],
    *,
    cell: CellVectors | None,
    chunk_size: int,
) -> Dict[str, object]:
    """Evaluate one frame without materializing a dense candidate vector."""

    if not cutoff_definitions:
        raise SparseHydrogenBondError("at least one cutoff definition is required")
    present_by_cutoff: List[List[int]] = [[] for _ in cutoff_definitions]
    present_geometry = []
    evaluated = 0
    for start, chunk in candidate_chunks(candidates, chunk_size):
        for local_index, candidate in enumerate(chunk):
            candidate_index = start + local_index
            donor = int(candidate["donor_atom_index"])
            hydrogen = int(candidate["hydrogen_atom_index"])
            acceptor = int(candidate["acceptor_atom_index"])
            distance = distance_angstrom(
                coordinates[donor], coordinates[acceptor], cell
            )
            angle = angle_degrees(
                coordinates[donor], coordinates[hydrogen], coordinates[acceptor], cell
            )
            matched_cutoffs = []
            for cutoff_index, cutoff in enumerate(cutoff_definitions):
                if (
                    distance <= float(cutoff["maximum_donor_acceptor_distance_angstrom"])
                    and angle >= float(cutoff["minimum_donor_hydrogen_acceptor_angle_degrees"])
                ):
                    present_by_cutoff[cutoff_index].append(candidate_index)
                    matched_cutoffs.append(str(cutoff["cutoff_id"]))
            if matched_cutoffs:
                present_geometry.append({
                    "candidate_index": candidate_index,
                    "bond_id": str(candidate["bond_id"]),
                    "donor_acceptor_distance_angstrom": distance,
                    "donor_hydrogen_acceptor_angle_degrees": angle,
                    "present_cutoff_ids": matched_cutoffs,
                })
            evaluated += 1
    return {
        "representation": "sparse_implicit_zero_v1",
        "evaluated_candidate_count": evaluated,
        "present_candidate_indices_by_cutoff": present_by_cutoff,
        "present_geometry": present_geometry,
    }


def _orthorhombic_lengths(cell: CellVectors | None) -> np.ndarray | None:
    if cell is None:
        return None
    matrix = np.asarray(cell, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise SparseHydrogenBondError("periodic cell must contain nine finite values")
    diagonal = np.diag(matrix)
    scale = max(float(np.max(np.abs(diagonal))), 1.0)
    off_diagonal = matrix - np.diag(diagonal)
    if np.max(np.abs(off_diagonal)) > 1.0e-12 * scale:
        return None
    if np.any(diagonal <= 0.0):
        raise SparseHydrogenBondError("orthorhombic cell lengths must be positive")
    return diagonal


def _minimum_image_vectors(
    vectors: np.ndarray, cell: CellVectors | None, lengths: np.ndarray | None
) -> np.ndarray:
    if cell is None:
        return vectors
    if lengths is not None:
        return vectors - lengths * np.floor(vectors / lengths + 0.5)
    return np.asarray(
        [minimum_image_displacement(vector, cell) for vector in vectors],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class CompiledSparseHydrogenBondEvaluator:
    """Frozen candidate arrays for exact spatially screened frame evaluation.

    The complete topology-defined candidate universe remains explicit, but a
    periodic cell list identifies donor/acceptor endpoints inside the largest
    requested distance cutoff before hydrogen angles are evaluated. Candidates
    absent from that neighbor list are exact implicit zeros.
    """

    candidates: Sequence[Mapping[str, object]]
    cutoff_definitions: Sequence[Mapping[str, object]]
    donor_indices: np.ndarray
    hydrogen_indices: np.ndarray
    acceptor_indices: np.ndarray
    maximum_cutoff_distance: float
    chunk_size: int
    unique_donor_indices: Tuple[int, ...]
    unique_acceptor_indices: Tuple[int, ...]
    candidate_indices_by_endpoint_pair: Mapping[Tuple[int, int], Tuple[int, ...]]

    @classmethod
    def compile(
        cls,
        candidates: Sequence[Mapping[str, object]],
        cutoff_definitions: Sequence[Mapping[str, object]],
        chunk_size: int,
    ) -> "CompiledSparseHydrogenBondEvaluator":
        if not cutoff_definitions:
            raise SparseHydrogenBondError("at least one cutoff definition is required")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise SparseHydrogenBondError("candidate chunk size must be a positive integer")
        maximum = max(
            float(cutoff["maximum_donor_acceptor_distance_angstrom"])
            for cutoff in cutoff_definitions
        )
        endpoint_pairs: Dict[Tuple[int, int], List[int]] = {}
        for candidate_index, candidate in enumerate(candidates):
            key = (
                int(candidate["donor_atom_index"]),
                int(candidate["acceptor_atom_index"]),
            )
            endpoint_pairs.setdefault(key, []).append(candidate_index)
        return cls(
            candidates=candidates,
            cutoff_definitions=cutoff_definitions,
            donor_indices=np.fromiter(
                (int(candidate["donor_atom_index"]) for candidate in candidates),
                dtype=np.int64,
                count=len(candidates),
            ),
            hydrogen_indices=np.fromiter(
                (int(candidate["hydrogen_atom_index"]) for candidate in candidates),
                dtype=np.int64,
                count=len(candidates),
            ),
            acceptor_indices=np.fromiter(
                (int(candidate["acceptor_atom_index"]) for candidate in candidates),
                dtype=np.int64,
                count=len(candidates),
            ),
            maximum_cutoff_distance=maximum,
            chunk_size=chunk_size,
            unique_donor_indices=tuple(sorted({key[0] for key in endpoint_pairs})),
            unique_acceptor_indices=tuple(sorted({key[1] for key in endpoint_pairs})),
            candidate_indices_by_endpoint_pair={
                key: tuple(values) for key, values in endpoint_pairs.items()
            },
        )

    def evaluate(
        self,
        coordinates: Sequence[Sequence[float]],
        *,
        cell: CellVectors | None,
    ) -> Dict[str, object]:
        coordinate_array = np.asarray(coordinates, dtype=np.float64)
        if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 3:
            raise SparseHydrogenBondError("coordinates must be an N by 3 array")
        if not np.isfinite(coordinate_array).all():
            raise SparseHydrogenBondError("coordinates must be finite")
        if self.donor_indices.size and (
            min(
                int(self.donor_indices.min()),
                int(self.hydrogen_indices.min()),
                int(self.acceptor_indices.min()),
            ) < 0
            or max(
                int(self.donor_indices.max()),
                int(self.hydrogen_indices.max()),
                int(self.acceptor_indices.max()),
            ) >= coordinate_array.shape[0]
        ):
            raise SparseHydrogenBondError("candidate atom index is outside the coordinate array")

        # Reuse the exact periodic spatial-bin implementation already validated
        # for one-water networks. The import is intentionally local because the
        # water module imports this module's packed sparse codecs.
        from .water_mediated_hydrogen_bonds import (
            WaterMediatedHydrogenBondError,
            neighbor_pairs_within,
        )

        present_by_cutoff: List[List[int]] = [
            [] for _ in self.cutoff_definitions
        ]
        present_geometry: List[Dict[str, object]] = []
        try:
            nearby = neighbor_pairs_within(
                coordinate_array,
                self.unique_donor_indices,
                self.unique_acceptor_indices,
                self.maximum_cutoff_distance,
                cell,
                maximum_pairs=max(1, len(self.candidates)),
            )
        except WaterMediatedHydrogenBondError as exc:
            raise SparseHydrogenBondError(str(exc)) from exc
        evaluated_nearby_candidates = 0
        for donor, acceptor, distance in nearby:
            candidate_indices = self.candidate_indices_by_endpoint_pair.get(
                (donor, acceptor), ()
            )
            for candidate_index in candidate_indices:
                evaluated_nearby_candidates += 1
                candidate = self.candidates[candidate_index]
                angle = angle_degrees(
                    coordinate_array[donor],
                    coordinate_array[int(candidate["hydrogen_atom_index"])],
                    coordinate_array[acceptor],
                    cell,
                )
                cutoff_ids = []
                for cutoff_index, cutoff in enumerate(self.cutoff_definitions):
                    if (
                        distance <= float(cutoff["maximum_donor_acceptor_distance_angstrom"])
                        and angle >= float(cutoff["minimum_donor_hydrogen_acceptor_angle_degrees"])
                    ):
                        present_by_cutoff[cutoff_index].append(candidate_index)
                        cutoff_ids.append(str(cutoff["cutoff_id"]))
                if cutoff_ids:
                    present_geometry.append({
                        "candidate_index": candidate_index,
                        "bond_id": str(candidate["bond_id"]),
                        "donor_acceptor_distance_angstrom": float(distance),
                        "donor_hydrogen_acceptor_angle_degrees": float(angle),
                        "present_cutoff_ids": cutoff_ids,
                    })
        for indices in present_by_cutoff:
            indices.sort()
        present_geometry.sort(key=lambda row: int(row["candidate_index"]))
        return {
            "representation": "sparse_implicit_zero_v1",
            "evaluated_candidate_count": len(self.candidates),
            "spatial_neighbor_pair_count": len(nearby),
            "explicit_geometry_evaluation_count": evaluated_nearby_candidates,
            "present_candidate_indices_by_cutoff": present_by_cutoff,
            "present_geometry": present_geometry,
            "geometry_engine": "spatial_cell_list_exact_periodic_v1",
        }


@dataclass(frozen=True)
class LazySpatialHydrogenBondEvaluator:
    """Evaluate a compact donor/acceptor universe without its Cartesian product."""

    donor_hydrogen_pairs: Sequence[Mapping[str, object]]
    acceptors: Sequence[Mapping[str, object]]
    cutoff_definitions: Sequence[Mapping[str, object]]
    interaction_scope: str
    exclude_same_residue: bool
    maximum_neighbor_pairs: int

    def evaluate(
        self,
        coordinates: Sequence[Sequence[float]],
        *,
        cell: CellVectors | None,
    ) -> Dict[str, object]:
        coordinate_array = np.asarray(coordinates, dtype=np.float64)
        if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 3:
            raise SparseHydrogenBondError("coordinates must be an N by 3 array")
        if not np.isfinite(coordinate_array).all():
            raise SparseHydrogenBondError("coordinates must be finite")
        if not self.donor_hydrogen_pairs or not self.acceptors:
            raise SparseHydrogenBondError(
                "lazy spatial evaluation requires donors and acceptors"
            )
        if not self.cutoff_definitions:
            raise SparseHydrogenBondError("at least one cutoff definition is required")
        if self.maximum_neighbor_pairs < 1:
            raise SparseHydrogenBondError("maximum_neighbor_pairs must be positive")

        donor_groups: Dict[int, List[Mapping[str, object]]] = {}
        for row in self.donor_hydrogen_pairs:
            donor = int(row["donor_atom_index"])
            hydrogen = int(row["hydrogen_atom_index"])
            if min(donor, hydrogen) < 0 or max(donor, hydrogen) >= len(coordinate_array):
                raise SparseHydrogenBondError(
                    "donor or hydrogen atom index is outside the coordinate array"
                )
            donor_groups.setdefault(donor, []).append(row)
        acceptor_rows: Dict[int, Mapping[str, object]] = {}
        for row in self.acceptors:
            acceptor = int(row["acceptor_atom_index"])
            if acceptor < 0 or acceptor >= len(coordinate_array):
                raise SparseHydrogenBondError(
                    "acceptor atom index is outside the coordinate array"
                )
            if acceptor in acceptor_rows:
                raise SparseHydrogenBondError("acceptor atom index is duplicated")
            acceptor_rows[acceptor] = row

        from .water_mediated_hydrogen_bonds import (
            WaterMediatedHydrogenBondError,
            neighbor_pairs_within,
        )

        maximum_distance = max(
            float(row["maximum_donor_acceptor_distance_angstrom"])
            for row in self.cutoff_definitions
        )
        try:
            nearby = neighbor_pairs_within(
                coordinate_array,
                sorted(donor_groups),
                sorted(acceptor_rows),
                maximum_distance,
                cell,
                maximum_pairs=self.maximum_neighbor_pairs,
            )
        except WaterMediatedHydrogenBondError as exc:
            raise SparseHydrogenBondError(str(exc)) from exc

        events: List[Dict[str, object]] = []
        angle_evaluations = 0
        from .hydrogen_bond_chemistry import scope_allows
        for donor, acceptor, distance in nearby:
            acceptor_row = acceptor_rows[acceptor]
            for donor_row in donor_groups[donor]:
                if not scope_allows(
                    str(donor_row["entity_class"]),
                    str(acceptor_row["entity_class"]),
                    self.interaction_scope,
                ):
                    continue
                if (
                    self.exclude_same_residue
                    and donor_row["residue_key"] == acceptor_row["residue_key"]
                ):
                    continue
                angle_evaluations += 1
                hydrogen = int(donor_row["hydrogen_atom_index"])
                angle = angle_degrees(
                    coordinate_array[donor],
                    coordinate_array[hydrogen],
                    coordinate_array[acceptor],
                    cell,
                )
                matched = [
                    str(cutoff["cutoff_id"])
                    for cutoff in self.cutoff_definitions
                    if (
                        distance <= float(
                            cutoff["maximum_donor_acceptor_distance_angstrom"]
                        )
                        and angle >= float(
                            cutoff["minimum_donor_hydrogen_acceptor_angle_degrees"]
                        )
                    )
                ]
                if matched:
                    events.append({
                        "donor_atom_index": donor,
                        "hydrogen_atom_index": hydrogen,
                        "acceptor_atom_index": acceptor,
                        "donor_identity_key": donor_row["donor_identity_key"],
                        "hydrogen_identity_key": donor_row["hydrogen_identity_key"],
                        "acceptor_identity_key": acceptor_row["acceptor_identity_key"],
                        "interaction_stratum": (
                            f"{donor_row['entity_class']}_to_"
                            f"{acceptor_row['entity_class']}"
                        ),
                        "donor_acceptor_distance_angstrom": float(distance),
                        "donor_hydrogen_acceptor_angle_degrees": float(angle),
                        "present_cutoff_ids": matched,
                    })
        events.sort(key=lambda row: (
            row["donor_identity_key"],
            row["hydrogen_identity_key"],
            row["acceptor_identity_key"],
        ))
        return {
            "representation": "lazy_spatial_events_v1",
            "spatial_neighbor_pair_count": len(nearby),
            "explicit_geometry_evaluation_count": angle_evaluations,
            "present_events": events,
            "geometry_engine": "spatial_cell_list_exact_periodic_v1",
        }


def dense_primary_values(
    frame: Mapping[str, object], candidate_count: int
) -> List[int]:
    """Materialize a primary binary vector from either report representation."""

    if "binary_values" in frame:
        values = list(frame["binary_values"])  # type: ignore[arg-type]
        if len(values) != candidate_count or any(value not in {0, 1} for value in values):
            raise SparseHydrogenBondError("dense frame vector does not match candidate count")
        return [int(value) for value in values]
    if frame.get("representation") == "sparse_packed_v2":
        if frame.get("candidate_count") != candidate_count:
            raise SparseHydrogenBondError(
                "packed frame candidate count does not match the dictionary"
            )
        values = [0] * candidate_count
        for value in packed_present_indices(frame, "primary"):
            if value >= candidate_count:
                raise SparseHydrogenBondError(
                    "packed primary candidate index exceeds dictionary"
                )
            values[value] = 1
        return values
    raw = frame.get("primary_present_candidate_indices")
    if not isinstance(raw, list):
        raise SparseHydrogenBondError("sparse frame is missing primary candidate indices")
    values = [0] * candidate_count
    prior = -1
    for value in raw:
        if (
            isinstance(value, bool) or not isinstance(value, int)
            or value < 0 or value >= candidate_count or value <= prior
        ):
            raise SparseHydrogenBondError(
                "sparse primary candidate indices must be sorted, unique, and in range"
            )
        values[value] = 1
        prior = value
    return values
