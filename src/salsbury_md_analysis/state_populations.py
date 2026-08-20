"""Descriptive system and replica population tables for shared state labels."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple


class StatePopulationError(ValueError):
    """Raised when assignment records cannot support a state comparison."""


def _paired_member_state_coupling(
    rows: Sequence[Mapping[str, object]], state_key: str, state_ids: Sequence[int]
) -> Dict[str, object] | None:
    if not any("member_id" in row for row in rows):
        return None
    frames: Dict[
        Tuple[str, str], Dict[Tuple[str, int], Dict[str, int | None]]
    ] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if "member_id" not in row or "segment_id" not in row or "source_frame_index" not in row:
            raise StatePopulationError(
                "member-resolved assignments require member, segment, and source-frame identity"
            )
        key = (str(row["system_id"]), str(row["replica_id"]))
        frame = (str(row["segment_id"]), int(row["source_frame_index"]))
        member_id = str(row["member_id"])
        if member_id in frames[key][frame]:
            raise StatePopulationError("duplicate member state for one physical frame")
        value = row[state_key]
        frames[key][frame][member_id] = int(value) if value is not None else None

    pair_reports = []
    for (system_id, replica_id), physical_frames in sorted(frames.items()):
        member_ids = sorted({
            member_id for member_rows in physical_frames.values()
            for member_id in member_rows
        })
        for left_index, left_id in enumerate(member_ids[:-1]):
            for right_id in member_ids[left_index + 1:]:
                paired_all = [
                    (member_rows[left_id], member_rows[right_id])
                    for _, member_rows in sorted(physical_frames.items())
                    if left_id in member_rows and right_id in member_rows
                ]
                paired = [
                    (left, right) for left, right in paired_all
                    if left is not None and right is not None
                ]
                contingency = [
                    [sum(left == a and right == b for left, right in paired) for b in state_ids]
                    for a in state_ids
                ]
                count = len(paired)
                agreement = (
                    sum(left == right for left, right in paired) / count if count else None
                )
                cramers_v = None
                if count and len(state_ids) > 1:
                    row_totals = [sum(row) for row in contingency]
                    column_totals = [sum(contingency[i][j] for i in range(len(state_ids))) for j in range(len(state_ids))]
                    chi_squared = 0.0
                    for i in range(len(state_ids)):
                        for j in range(len(state_ids)):
                            expected = row_totals[i] * column_totals[j] / count
                            if expected > 0.0:
                                chi_squared += (contingency[i][j] - expected) ** 2 / expected
                    denominator = count * min(len(state_ids) - 1, len(state_ids) - 1)
                    cramers_v = math.sqrt(chi_squared / denominator) if denominator else None
                pair_reports.append({
                    "system_id": system_id,
                    "replica_id": replica_id,
                    "left_member_id": left_id,
                    "right_member_id": right_id,
                    "paired_physical_frame_count": len(paired_all),
                    "both_members_assigned_count": count,
                    "both_members_assigned_fraction": (
                        count / len(paired_all) if paired_all else None
                    ),
                    "same_state_fraction": agreement,
                    "cramers_v": cramers_v,
                    "state_ids": list(state_ids),
                    "contingency_counts": contingency,
                })
    return {
        "coupling_schema": "salsbury-paired-oligomer-member-state-coupling-v1",
        "pair_reports": pair_reports,
        "interpretation": (
            "within-physical-frame member state concordance and Cramer's V; these "
            "are descriptive association measures and do not establish causality, "
            "kinetic coupling, or additional independent replicas"
        ),
    }


def _population_row(
    rows: Sequence[Mapping[str, object]], state_key: str, state_ids: Sequence[int]
) -> Dict[str, object]:
    evaluated = len(rows)
    assigned = sum(row.get(state_key) is not None for row in rows)
    populations = []
    for state_id in state_ids:
        count = sum(row.get(state_key) == state_id for row in rows)
        populations.append({
            "state_id": state_id,
            "count": count,
            "fraction_of_all_evaluated": count / evaluated if evaluated else None,
            "fraction_of_assigned": count / assigned if assigned else None,
        })
    return {
        "evaluated_count": evaluated,
        "assigned_count": assigned,
        "unassigned_or_noise_count": evaluated - assigned,
        "assigned_coverage_fraction": assigned / evaluated if evaluated else None,
        "state_populations": populations,
    }


def summarize_state_populations(
    assignment_rows: Sequence[Mapping[str, object]], state_key: str
) -> Dict[str, object]:
    """Summarize labels shared across systems without asserting independence."""

    if not assignment_rows:
        raise StatePopulationError("state population comparison requires assignments")
    required = {"system_id", "replica_id", state_key}
    if any(not required.issubset(row) for row in assignment_rows):
        raise StatePopulationError(
            "state assignments require system_id, replica_id, and the state field"
        )
    state_values = [
        row[state_key] for row in assignment_rows if row[state_key] is not None
    ]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in state_values):
        raise StatePopulationError("state labels must be integer or null")
    state_ids = sorted(set(int(value) for value in state_values))
    system_rows: List[Dict[str, object]] = []
    systems = sorted({str(row["system_id"]) for row in assignment_rows})
    for system_id in systems:
        rows = [row for row in assignment_rows if str(row["system_id"]) == system_id]
        system_rows.append({
            "system_id": system_id,
            **_population_row(rows, state_key, state_ids),
        })
    replica_rows: List[Dict[str, object]] = []
    replicas = sorted({
        (str(row["system_id"]), str(row["replica_id"]))
        for row in assignment_rows
    })
    for system_id, replica_id in replicas:
        rows = [
            row for row in assignment_rows
            if str(row["system_id"]) == system_id
            and str(row["replica_id"]) == replica_id
        ]
        replica_rows.append({
            "system_id": system_id,
            "replica_id": replica_id,
            **_population_row(rows, state_key, state_ids),
        })

    member_rows: List[Dict[str, object]] = []
    member_keys = sorted({
        (str(row["system_id"]), str(row["replica_id"]), str(row["member_id"]))
        for row in assignment_rows if "member_id" in row
    })
    for system_id, replica_id, member_id in member_keys:
        rows = [
            row for row in assignment_rows
            if str(row["system_id"]) == system_id
            and str(row["replica_id"]) == replica_id
            and str(row.get("member_id")) == member_id
        ]
        member_rows.append({
            "system_id": system_id,
            "replica_id": replica_id,
            "member_id": member_id,
            **_population_row(rows, state_key, state_ids),
        })

    by_system = {row["system_id"]: row for row in system_rows}
    pairwise = []
    for left_index, left_id in enumerate(systems[:-1]):
        for right_id in systems[left_index + 1:]:
            left = by_system[left_id]
            right = by_system[right_id]
            left_states = {
                row["state_id"]: row for row in left["state_populations"]  # type: ignore[union-attr]
            }
            right_states = {
                row["state_id"]: row for row in right["state_populations"]  # type: ignore[union-attr]
            }
            pairwise.append({
                "left_system_id": left_id,
                "right_system_id": right_id,
                "state_fraction_differences": [
                    {
                        "state_id": state_id,
                        "left_fraction_of_all_evaluated": left_states[state_id]["fraction_of_all_evaluated"],
                        "right_fraction_of_all_evaluated": right_states[state_id]["fraction_of_all_evaluated"],
                        "left_minus_right_fraction_of_all_evaluated": (
                            float(left_states[state_id]["fraction_of_all_evaluated"])
                            - float(right_states[state_id]["fraction_of_all_evaluated"])
                        ),
                        "left_fraction_of_assigned": left_states[state_id]["fraction_of_assigned"],
                        "right_fraction_of_assigned": right_states[state_id]["fraction_of_assigned"],
                        "left_minus_right_fraction_of_assigned": (
                            None
                            if left_states[state_id]["fraction_of_assigned"] is None
                            or right_states[state_id]["fraction_of_assigned"] is None
                            else float(left_states[state_id]["fraction_of_assigned"])
                            - float(right_states[state_id]["fraction_of_assigned"])
                        ),
                    }
                    for state_id in state_ids
                ],
                "left_assigned_coverage_fraction": left["assigned_coverage_fraction"],
                "right_assigned_coverage_fraction": right["assigned_coverage_fraction"],
            })
    return {
        "state_field": state_key,
        "state_ids": state_ids,
        "system_populations": system_rows,
        "replica_populations": replica_rows,
        "member_populations": member_rows,
        "paired_member_state_coupling": _paired_member_state_coupling(
            assignment_rows, state_key, state_ids
        ),
        "pairwise_system_differences": pairwise,
        "observation_independence": (
            "member rows from one physical frame are paired symmetry representations, "
            "not independent replicas"
            if member_rows else
            "frames remain time-correlated within simulation replicas"
        ),
        "interpretation": (
            "descriptive frame fractions; replica and time-block uncertainty must "
            "be evaluated before inferential system comparisons"
        ),
    }
