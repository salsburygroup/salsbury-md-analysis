"""Chemical-identity comparison of direct hydrogen-bond discovery reports.

This postprocessor deliberately groups connectivity-declared donor hydrogens
before calculating occupancy.  A donor-heavy-atom/acceptor-heavy-atom pair is
therefore present at most once in a frame even when equivalent hydrogens swap.
Optional homolog mappings live in the request, never in package source.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from .hydrogen_bond_sparse import (
    SparseHydrogenBondError,
    packed_present_indices,
)
from .manifests import ManifestValidationError, load_json, resolve_manifest_path, sha256_file


class HydrogenBondComparisonError(ValueError):
    """Raised when hydrogen-bond reports cannot be compared unambiguously."""


_IDENTITY_FIELDS = (
    "chain_id", "residue_number", "insertion_code", "residue_name",
    "atom_name", "altloc", "element",
)


def _strict_object(
    value: object, label: str, *, required: Sequence[str], optional: Sequence[str] = ()
) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise HydrogenBondComparisonError(f"{label} must be an object")
    missing = sorted(set(required).difference(value))
    unknown = sorted(set(value).difference(set(required) | set(optional)))
    if missing:
        raise HydrogenBondComparisonError(f"{label} is missing: {', '.join(missing)}")
    if unknown:
        raise HydrogenBondComparisonError(f"{label} has unknown fields: {', '.join(unknown)}")
    return value


def _identity(value: object, label: str) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise HydrogenBondComparisonError(f"{label} must be an atom-identity object")
    missing = [field for field in _IDENTITY_FIELDS if field not in value]
    if missing:
        raise HydrogenBondComparisonError(
            f"{label} is missing identity fields: {', '.join(missing)}"
        )
    result = {field: value[field] for field in _IDENTITY_FIELDS}
    if isinstance(result["residue_number"], bool) or not isinstance(result["residue_number"], int):
        raise HydrogenBondComparisonError(f"{label}.residue_number must be an integer")
    for field in set(_IDENTITY_FIELDS) - {"residue_number"}:
        if not isinstance(result[field], str):
            raise HydrogenBondComparisonError(f"{label}.{field} must be a string")
    return result


def _identity_key(identity: Mapping[str, object]) -> Tuple[object, ...]:
    return tuple(identity[field] for field in (
        "chain_id", "residue_number", "insertion_code", "atom_name", "altloc",
        "element",
    ))


def _normalize_mappings(
    value: object, condition_ids: Sequence[str]
) -> List[Dict[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise HydrogenBondComparisonError("homolog_mappings must be an array")
    normalized = []
    allowed = set(_IDENTITY_FIELDS)
    for index, raw in enumerate(value):
        row = _strict_object(
            raw, f"homolog_mappings[{index}]",
            required=("condition_id", "match", "canonical_updates"),
        )
        condition_id = row["condition_id"]
        if condition_id not in condition_ids:
            raise HydrogenBondComparisonError(
                f"homolog_mappings[{index}].condition_id is not a declared condition"
            )
        match = row["match"]
        updates = row["canonical_updates"]
        if not isinstance(match, dict) or not match or set(match).difference(allowed):
            raise HydrogenBondComparisonError(
                f"homolog_mappings[{index}].match must use one or more chemical identity fields"
            )
        if not isinstance(updates, dict) or not updates or set(updates).difference(allowed):
            raise HydrogenBondComparisonError(
                f"homolog_mappings[{index}].canonical_updates must use chemical identity fields"
            )
        if "residue_number" in match and (
            isinstance(match["residue_number"], bool)
            or not isinstance(match["residue_number"], int)
        ):
            raise HydrogenBondComparisonError(
                f"homolog_mappings[{index}].match.residue_number must be an integer"
            )
        if "residue_number" in updates and (
            isinstance(updates["residue_number"], bool)
            or not isinstance(updates["residue_number"], int)
        ):
            raise HydrogenBondComparisonError(
                f"homolog_mappings[{index}].canonical_updates.residue_number must be an integer"
            )
        for container_name, container in (("match", match), ("canonical_updates", updates)):
            for field, field_value in container.items():
                if field != "residue_number" and not isinstance(field_value, str):
                    raise HydrogenBondComparisonError(
                        f"homolog_mappings[{index}].{container_name}.{field} must be a string"
                    )
        normalized.append({
            "condition_id": condition_id,
            "match": dict(match),
            "canonical_updates": dict(updates),
        })
    return normalized


def _canonicalize(
    identity: Mapping[str, object], condition_id: str,
    mappings: Sequence[Mapping[str, object]], matched_mapping_indices: set[int],
) -> Dict[str, object]:
    result = dict(identity)
    matched = []
    for index, mapping in enumerate(mappings):
        if mapping["condition_id"] != condition_id:
            continue
        selector = mapping["match"]
        assert isinstance(selector, dict)
        if all(identity.get(field) == expected for field, expected in selector.items()):
            matched.append(index)
    if len(matched) > 1:
        raise HydrogenBondComparisonError(
            f"atom identity matches multiple homolog mappings for condition {condition_id}: "
            + str(dict(identity))
        )
    if matched:
        matched_mapping_indices.add(matched[0])
        updates = mappings[matched[0]]["canonical_updates"]
        assert isinstance(updates, dict)
        result.update(updates)
    return result


def _candidate_identities(report: Mapping[str, object]) -> List[Tuple[Dict[str, object], Dict[str, object]]]:
    candidates = report.get("candidate_dictionary")
    if not isinstance(candidates, list) or not candidates:
        raise HydrogenBondComparisonError("candidate_dictionary must be a nonempty array")
    atom_dictionary = report.get("atom_dictionary")
    atoms: Dict[int, Dict[str, object]] = {}
    if isinstance(atom_dictionary, list):
        for index, raw in enumerate(atom_dictionary):
            if not isinstance(raw, dict) or "atom_index" not in raw or "identity" not in raw:
                raise HydrogenBondComparisonError(f"atom_dictionary[{index}] is malformed")
            atom_index = raw["atom_index"]
            if isinstance(atom_index, bool) or not isinstance(atom_index, int) or atom_index in atoms:
                raise HydrogenBondComparisonError("atom_dictionary indices must be unique integers")
            atoms[atom_index] = _identity(raw["identity"], f"atom_dictionary[{index}].identity")
    identities = []
    for index, raw in enumerate(candidates):
        if not isinstance(raw, dict):
            raise HydrogenBondComparisonError(f"candidate_dictionary[{index}] must be an object")
        try:
            donor_index = int(raw["donor_atom_index"])
            acceptor_index = int(raw["acceptor_atom_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HydrogenBondComparisonError(
                f"candidate_dictionary[{index}] lacks valid donor/acceptor indices"
            ) from exc
        donor_raw = raw.get("donor_identity", atoms.get(donor_index))
        acceptor_raw = raw.get("acceptor_identity", atoms.get(acceptor_index))
        identities.append((
            _identity(donor_raw, f"candidate_dictionary[{index}].donor_identity"),
            _identity(acceptor_raw, f"candidate_dictionary[{index}].acceptor_identity"),
        ))
    return identities


def _scope(report: Mapping[str, object]) -> str:
    settings = report.get("settings")
    if not isinstance(settings, dict) or not isinstance(settings.get("interaction_scope"), str):
        raise HydrogenBondComparisonError(
            "each automatic-discovery report must declare settings.interaction_scope"
        )
    return str(settings["interaction_scope"])


def _cutoff(report: Mapping[str, object], cutoff_id: str) -> Dict[str, object]:
    rows = report.get("cutoff_definitions")
    if not isinstance(rows, list):
        raise HydrogenBondComparisonError("cutoff_definitions must be an array")
    matches = [row for row in rows if isinstance(row, dict) and row.get("cutoff_id") == cutoff_id]
    if len(matches) != 1:
        raise HydrogenBondComparisonError(f"cutoff_id {cutoff_id!r} must occur exactly once")
    return dict(matches[0])


def _present_indices(frame: Mapping[str, object], cutoff_id: str) -> Sequence[int]:
    if frame.get("representation") == "sparse_packed_v2":
        try:
            return packed_present_indices(frame, cutoff_id)
        except SparseHydrogenBondError as exc:
            raise HydrogenBondComparisonError(str(exc)) from exc
    by_cutoff = frame.get("cutoff_present_candidate_indices")
    if isinstance(by_cutoff, dict) and cutoff_id in by_cutoff:
        raw = by_cutoff[cutoff_id]
    elif cutoff_id == "primary":
        raw = frame.get("primary_present_candidate_indices")
    else:
        raise HydrogenBondComparisonError(f"frame lacks requested cutoff {cutoff_id!r}")
    if not isinstance(raw, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in raw
    ):
        raise HydrogenBondComparisonError("present candidate indices must be nonnegative integers")
    return raw


def _resolved_system_report(
    report: Mapping[str, object], system_id: str | None,
) -> tuple[Mapping[str, object], str | None]:
    """Select a full per-system feature space from a v2 discovery report."""

    raw_views = report.get("system_feature_spaces")
    if not isinstance(raw_views, list):
        return report, system_id
    views = {
        str(row.get("system_id")): row
        for row in raw_views
        if isinstance(row, dict) and isinstance(row.get("system_id"), str)
    }
    if system_id is None:
        if len(views) != 1:
            raise HydrogenBondComparisonError(
                "source report contains multiple per-system feature spaces; "
                "each condition must declare system_id"
            )
        system_id = next(iter(views))
    if system_id not in views:
        available = ", ".join(sorted(views)) or "none"
        raise HydrogenBondComparisonError(
            f"requested system_id {system_id!r} is absent from source report; available: {available}"
        )
    selected = dict(report)
    selected.update(views[system_id])
    selected.pop("system_feature_spaces", None)
    return selected, system_id


def _condition_summary(
    condition_id: str, report: Mapping[str, object], cutoff_id: str,
    mappings: Sequence[Mapping[str, object]], matched_mapping_indices: set[int],
    *, system_id: str | None = None,
) -> Dict[str, object]:
    report, system_id = _resolved_system_report(report, system_id)
    if report.get("module_id") != "hydrogen_bond_discovery":
        raise HydrogenBondComparisonError("source report module_id must be hydrogen_bond_discovery")
    if report.get("technical_status") != "complete" or int(report.get("error_count", 0)) != 0:
        raise HydrogenBondComparisonError("source hydrogen-bond report is not technically complete")
    if report.get("frame_matrix_representation") not in {
        "sparse_implicit_zero_v1", "sparse_packed_v2",
    }:
        raise HydrogenBondComparisonError(
            "comparison requires a supported sparse frame matrix with an explicit-zero contract"
        )
    if not isinstance(report.get("sparse_zero_contract"), str):
        raise HydrogenBondComparisonError("source report lacks its sparse explicit-zero contract")
    candidates = _candidate_identities(report)
    canonical_candidates = []
    candidate_groups = set()
    candidate_group_identities = {}
    original_to_canonical: Dict[Tuple[str, Tuple[object, ...]], Tuple[object, ...]] = {}
    canonical_to_original: Dict[Tuple[str, Tuple[object, ...]], Tuple[object, ...]] = {}
    for donor, acceptor in candidates:
        canonical_donor = _canonicalize(donor, condition_id, mappings, matched_mapping_indices)
        canonical_acceptor = _canonicalize(acceptor, condition_id, mappings, matched_mapping_indices)
        for role, original, canonical in (
            ("donor", donor, canonical_donor), ("acceptor", acceptor, canonical_acceptor)
        ):
            original_key = _identity_key(original)
            canonical_key = _identity_key(canonical)
            original_to_canonical[(role, original_key)] = canonical_key
            collision_key = (role, canonical_key)
            prior = canonical_to_original.setdefault(collision_key, original_key)
            if prior != original_key:
                raise HydrogenBondComparisonError(
                    f"homolog mappings collapse distinct {role} atoms within condition {condition_id}"
                )
        canonical_candidates.append((canonical_donor, canonical_acceptor))
        group_key = (_identity_key(canonical_donor), _identity_key(canonical_acceptor))
        candidate_groups.add(group_key)
        candidate_group_identities.setdefault(
            group_key, (canonical_donor, canonical_acceptor)
        )

    frames = report.get("frame_bond_matrix")
    if not isinstance(frames, list) or not frames:
        raise HydrogenBondComparisonError("frame_bond_matrix must be a nonempty array")
    declared_system_ids = {
        raw_frame.get("system_id")
        for raw_frame in frames
        if isinstance(raw_frame, dict)
        and isinstance(raw_frame.get("system_id"), str)
        and raw_frame.get("system_id")
    }
    if system_id is None and len(declared_system_ids) > 1:
        raise HydrogenBondComparisonError(
            "source report contains multiple system_id values; each condition must declare system_id"
        )
    if system_id is not None and system_id not in declared_system_ids:
        available = ", ".join(sorted(str(value) for value in declared_system_ids)) or "none"
        raise HydrogenBondComparisonError(
            f"requested system_id {system_id!r} is absent from source report; available: {available}"
        )
    resolved_system_id = system_id
    if resolved_system_id is None and len(declared_system_ids) == 1:
        resolved_system_id = str(next(iter(declared_system_ids)))
    replica_frames: Dict[str, int] = defaultdict(int)
    replica_counts: Dict[str, Dict[Tuple[Tuple[object, ...], Tuple[object, ...]], int]] = defaultdict(lambda: defaultdict(int))
    group_identities: Dict[Tuple[Tuple[object, ...], Tuple[object, ...]], Tuple[Dict[str, object], Dict[str, object]]] = {}
    for frame_index, raw_frame in enumerate(frames):
        if not isinstance(raw_frame, dict):
            raise HydrogenBondComparisonError(f"frame_bond_matrix[{frame_index}] must be an object")
        if system_id is not None and raw_frame.get("system_id") != system_id:
            continue
        replica_id = raw_frame.get("replica_id")
        if not isinstance(replica_id, str) or not replica_id:
            raise HydrogenBondComparisonError(f"frame_bond_matrix[{frame_index}] lacks replica_id")
        replica_frames[replica_id] += 1
        present_groups = set()
        for candidate_index in _present_indices(raw_frame, cutoff_id):
            if candidate_index >= len(canonical_candidates):
                raise HydrogenBondComparisonError("present candidate index exceeds dictionary")
            donor, acceptor = canonical_candidates[candidate_index]
            key = (_identity_key(donor), _identity_key(acceptor))
            present_groups.add(key)
            group_identities.setdefault(key, (donor, acceptor))
        for key in present_groups:
            replica_counts[replica_id][key] += 1

    all_groups = set().union(*(set(values) for values in replica_counts.values()))
    group_rows = {}
    for key in all_groups:
        per_replica = []
        total_present = 0
        total_frames = 0
        for replica_id in sorted(replica_frames):
            present = replica_counts[replica_id].get(key, 0)
            frames_for_replica = replica_frames[replica_id]
            total_present += present
            total_frames += frames_for_replica
            per_replica.append({
                "replica_id": replica_id,
                "evaluated_frame_count": frames_for_replica,
                "present_frame_count": present,
                "occupancy_fraction": present / frames_for_replica,
            })
        donor, acceptor = group_identities[key]
        group_rows[key] = {
            "donor_identity": donor,
            "acceptor_identity": acceptor,
            "equal_replica_mean_occupancy_fraction": sum(
                row["occupancy_fraction"] for row in per_replica
            ) / len(per_replica),
            "pooled_frame_occupancy_fraction": total_present / total_frames,
            "per_replica": per_replica,
        }
    return {
        "condition_id": condition_id,
        "system_id": resolved_system_id,
        "interaction_scope": _scope(report),
        "candidate_count": len(candidates),
        "donor_acceptor_candidate_group_count": len(candidate_groups),
        "donor_acceptor_group_count_observed": len(group_rows),
        "replica_count": len(replica_frames),
        "evaluated_frame_count": sum(replica_frames.values()),
        "replica_frame_counts": dict(sorted(replica_frames.items())),
        "candidate_groups": candidate_groups,
        "candidate_group_identities": candidate_group_identities,
        "groups": group_rows,
    }


def compare_hydrogen_bond_reports(
    request: Mapping[str, object], *, request_path: Path | None = None
) -> Dict[str, object]:
    """Compare exactly two sparse discovery reports on chemical identity."""

    request = _strict_object(
        request, "request", required=("conditions",),
        optional=(
            "comparison_id", "cutoff_id", "group_donor_hydrogens",
            "expected_interaction_scope", "homolog_mappings", "top_n",
        ),
    )
    conditions_raw = request["conditions"]
    if not isinstance(conditions_raw, list) or len(conditions_raw) != 2:
        raise HydrogenBondComparisonError("conditions must contain exactly two reports")
    conditions = []
    for index, raw in enumerate(conditions_raw):
        row = _strict_object(
            raw, f"conditions[{index}]", required=("condition_id", "report"),
            optional=("system_id",),
        )
        if not isinstance(row["condition_id"], str) or not row["condition_id"].strip():
            raise HydrogenBondComparisonError(f"conditions[{index}].condition_id must be nonempty")
        if not isinstance(row["report"], str) or not row["report"].strip():
            raise HydrogenBondComparisonError(f"conditions[{index}].report must be a path string")
        if "system_id" in row and (
            not isinstance(row["system_id"], str) or not row["system_id"].strip()
        ):
            raise HydrogenBondComparisonError(f"conditions[{index}].system_id must be nonempty")
        conditions.append(dict(row))
    condition_ids = [str(row["condition_id"]) for row in conditions]
    if len(set(condition_ids)) != 2:
        raise HydrogenBondComparisonError("condition_id values must be unique")
    if request.get("group_donor_hydrogens", True) is not True:
        raise HydrogenBondComparisonError(
            "group_donor_hydrogens must be true; explicit-hydrogen comparison is intentionally excluded"
        )
    cutoff_id = request.get("cutoff_id", "primary")
    if not isinstance(cutoff_id, str) or not cutoff_id:
        raise HydrogenBondComparisonError("cutoff_id must be a nonempty string")
    top_n = request.get("top_n", 100)
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise HydrogenBondComparisonError("top_n must be a positive integer")
    mappings = _normalize_mappings(request.get("homolog_mappings"), condition_ids)
    matched_mapping_indices: set[int] = set()
    reports = []
    source_records = []
    cutoff_definitions = []
    for row in conditions:
        path = resolve_manifest_path(str(row["report"]), request_path)
        report = load_json(path)
        reports.append(report)
        source_records.append({
            "condition_id": row["condition_id"],
            "system_id": row.get("system_id"),
            "report_path": str(path),
            "report_sha256": sha256_file(path),
            "source_contract_signature_sha256": report.get("contract_signature_sha256"),
            "source_input_content_signature_sha256": report.get("input_content_signature_sha256"),
        })
        cutoff_definitions.append(_cutoff(report, cutoff_id))
    if cutoff_definitions[0] != cutoff_definitions[1]:
        raise HydrogenBondComparisonError("requested cutoff definitions differ between reports")

    summaries = [
        _condition_summary(
            condition_id, report, str(cutoff_id), mappings, matched_mapping_indices,
            system_id=str(row["system_id"]) if "system_id" in row else None,
        )
        for condition_id, report, row in zip(condition_ids, reports, conditions)
    ]
    scopes = [str(summary["interaction_scope"]) for summary in summaries]
    if scopes[0] != scopes[1]:
        raise HydrogenBondComparisonError("interaction_scope differs between reports")
    expected_scope = request.get("expected_interaction_scope")
    if expected_scope is not None and expected_scope != scopes[0]:
        raise HydrogenBondComparisonError(
            "expected_interaction_scope does not match the source reports"
        )
    unmatched = sorted(set(range(len(mappings))).difference(matched_mapping_indices))
    if unmatched:
        raise HydrogenBondComparisonError(
            "homolog mappings matched no donor or acceptor atom: "
            + ", ".join(str(index) for index in unmatched)
        )

    first_groups = summaries[0].pop("groups")
    second_groups = summaries[1].pop("groups")
    first_candidates = summaries[0].pop("candidate_groups")
    second_candidates = summaries[1].pop("candidate_groups")
    first_candidate_identities = summaries[0].pop("candidate_group_identities")
    second_candidate_identities = summaries[1].pop("candidate_group_identities")
    first_replica_frames = summaries[0].pop("replica_frame_counts")
    second_replica_frames = summaries[1].pop("replica_frame_counts")
    assert isinstance(first_groups, dict) and isinstance(second_groups, dict)
    assert isinstance(first_candidates, set) and isinstance(second_candidates, set)
    assert isinstance(first_candidate_identities, dict) and isinstance(second_candidate_identities, dict)
    assert isinstance(first_replica_frames, dict) and isinstance(second_replica_frames, dict)
    keys = set(first_candidates) | set(second_candidates)
    comparisons = []
    for key in keys:
        first = first_groups.get(key)
        second = second_groups.get(key)
        identity_source = (
            first if first is not None else second
            if second is not None else None
        )
        if identity_source is None:
            identities = first_candidate_identities.get(key, second_candidate_identities.get(key))
            assert identities is not None
            identity_source = {
                "donor_identity": identities[0], "acceptor_identity": identities[1]
            }
        assert isinstance(identity_source, dict)
        first_eligible = key in first_candidates
        second_eligible = key in second_candidates
        first_occupancy = (
            float(first["equal_replica_mean_occupancy_fraction"])
            if isinstance(first, dict) else 0.0 if first_eligible else None
        )
        second_occupancy = (
            float(second["equal_replica_mean_occupancy_fraction"])
            if isinstance(second, dict) else 0.0 if second_eligible else None
        )
        difference = (
            second_occupancy - first_occupancy
            if first_occupancy is not None and second_occupancy is not None
            else None
        )
        if difference is not None and not math.isfinite(difference):
            raise HydrogenBondComparisonError("non-finite occupancy difference")
        first_per_replica = (
            first["per_replica"] if isinstance(first, dict)
            else [
                {
                    "replica_id": replica_id,
                    "evaluated_frame_count": frame_count,
                    "present_frame_count": 0,
                    "occupancy_fraction": 0.0,
                }
                for replica_id, frame_count in sorted(first_replica_frames.items())
            ] if first_eligible else []
        )
        second_per_replica = (
            second["per_replica"] if isinstance(second, dict)
            else [
                {
                    "replica_id": replica_id,
                    "evaluated_frame_count": frame_count,
                    "present_frame_count": 0,
                    "occupancy_fraction": 0.0,
                }
                for replica_id, frame_count in sorted(second_replica_frames.items())
            ] if second_eligible else []
        )
        comparisons.append({
            "donor_identity": identity_source["donor_identity"],
            "acceptor_identity": identity_source["acceptor_identity"],
            "condition_occupancies": {
                condition_ids[0]: first_occupancy, condition_ids[1]: second_occupancy,
            },
            "occupancy_difference_second_minus_first": difference,
            "absolute_occupancy_difference": abs(difference) if difference is not None else None,
            "candidate_eligible_in_conditions": [
                condition_id for condition_id, eligible in zip(
                    condition_ids, (first_eligible, second_eligible)
                ) if eligible
            ],
            "chemically_comparable_between_conditions": first_eligible and second_eligible,
            "condition_feature_status": {
                condition_ids[0]: (
                    "observed" if first is not None else
                    "chemically_present_never_observed" if first_eligible else
                    "chemically_absent"
                ),
                condition_ids[1]: (
                    "observed" if second is not None else
                    "chemically_present_never_observed" if second_eligible else
                    "chemically_absent"
                ),
            },
            "observed_in_conditions": [
                condition_id for condition_id, row in zip(condition_ids, (first, second))
                if row is not None
            ],
            "per_replica": {
                condition_ids[0]: first_per_replica,
                condition_ids[1]: second_per_replica,
            },
        })
    comparisons.sort(key=lambda row: (
        row["absolute_occupancy_difference"] is None,
        -float(row["absolute_occupancy_difference"] or 0.0),
        str(row["donor_identity"]), str(row["acceptor_identity"]),
    ))
    ranked = [
        row for row in comparisons
        if row["chemically_comparable_between_conditions"]
    ]
    return {
        "module_id": "hydrogen_bond_comparison",
        "comparison_schema": "salsbury-grouped-hydrogen-bond-comparison-v2",
        "technical_status": "complete",
        "scientific_status": "descriptive comparison; convergence and mechanism not established",
        "comparison_id": request.get("comparison_id"),
        "grouping_contract": (
            "Each donor-heavy-atom/acceptor-heavy-atom pair is counted at most once per frame; "
            "connectivity-declared equivalent donor hydrogens are grouped before occupancy."
        ),
        "identity_mapping_contract": (
            "Donor and acceptor heavy atoms map by chain, residue number, insertion "
            "code, atom name, alternate location, and element. Residue names are "
            "reported but do not prevent homologous-position comparison; explicit "
            "project mappings are required when numbering or chain identity differs."
        ),
        "missing_value_contract": (
            "A group present in a condition's candidate universe but absent from its observed sparse union "
            "has occupancy zero. A group chemically ineligible for that topology has null occupancy and "
            "is excluded from numeric difference ranking."
        ),
        "interaction_scope": scopes[0],
        "cutoff_definition": cutoff_definitions[0],
        "condition_order": condition_ids,
        "condition_summaries": summaries,
        "source_reports": source_records,
        "homolog_mappings": mappings,
        "candidate_group_union_count": len(comparisons),
        "observed_group_union_count": sum(
            bool(row["observed_in_conditions"]) for row in comparisons
        ),
        "observed_group_shared_count": sum(
            len(row["observed_in_conditions"]) == 2 for row in comparisons
        ),
        "chemically_comparable_observed_union_count": len(ranked),
        "topology_specific_candidate_group_count": len(comparisons) - len(ranked),
        "topology_specific_observed_group_count": sum(
            not row["chemically_comparable_between_conditions"]
            and bool(row["observed_in_conditions"])
            for row in comparisons
        ),
        "group_comparisons": comparisons,
        "top_absolute_differences": ranked[:top_n],
        "error_count": 0,
        "warning_count": 0,
        "issues": [],
        "limitations": [
            "The full chemistry-defined candidate union is tabulated, including groups that were eligible but never observed.",
            "Topology-specific atoms are reported with null occupancy in the ineligible condition rather than converted to evaluated zeros.",
            "Homolog mappings are explicit project assertions and require chemical review; they are not inferred by this module.",
            "Equal-replica means prevent a longer replica from silently dominating, but do not establish independent sampling or convergence.",
            "Occupancy differences do not establish energy, affinity, causality, or mechanism.",
        ],
    }


def compare_hydrogen_bond_reports_file(path: Path) -> Dict[str, object]:
    source = Path(path).expanduser().resolve(strict=False)
    return compare_hydrogen_bond_reports(load_json(source), request_path=source)


def compare_hydrogen_bond_reports_file_safe(path: Path) -> Dict[str, object]:
    try:
        return compare_hydrogen_bond_reports_file(path)
    except (
        HydrogenBondComparisonError, ManifestValidationError, OSError,
        KeyError, TypeError, ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "hydrogen_bond_comparison",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "request_path": str(Path(path).expanduser().resolve(strict=False)),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "HYDROGEN_BOND_COMPARISON_INVALID", "message": message}
                for message in messages
            ],
        }
