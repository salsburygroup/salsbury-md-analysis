"""Transparent, deterministic prioritization of completed analysis findings."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .analysis_config import DEFAULT_DISABLED_MODULES
from .manifests import load_json


class FindingPickerError(ValueError):
    """Raised when finding prioritization cannot preserve evidence semantics."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_PRIORITY = {
    "free_energy_surface": 0,
    "free_energy_conformation": 1,
    "clustering": 2,
    "clustering_conformation": 3,
    "rmsf": 4,
    "coupled_interaction": 5,
    "other_physical": 6,
}


def _candidate(
    *, module_id: str, category: str, statement: str, report_path: Path,
    effect_value: float | None = None, p_value: float | None = None,
    evidence_level: str = "descriptive", systems: Sequence[str] = (),
    family: str = "single_system",
    report_paths: Sequence[Path] = (),
) -> Dict[str, object]:
    if effect_value is not None and not math.isfinite(effect_value):
        raise FindingPickerError("finding effect must be finite")
    if p_value is not None and (not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0):
        raise FindingPickerError("finding p value must be within zero and one")
    return {
        "module_id": module_id,
        "category": category,
        "evidence_level": evidence_level,
        "statement": statement,
        "system_ids": list(systems),
        "comparison_family": family,
        "effect_value": effect_value,
        "absolute_effect_value": abs(effect_value) if effect_value is not None else None,
        "p_value": p_value,
        "adjusted_p_value": None,
        "report_path": str(report_path),
        "report_paths": [str(value) for value in (report_paths or (report_path,))],
    }


def _state_differences(
    report: Mapping[str, object], module_id: str, path: Path
) -> List[Dict[str, object]]:
    comparison = report.get("state_population_comparison")
    if not isinstance(comparison, dict):
        return []
    category = "free_energy_surface" if module_id == "pca_fes_basins" else "clustering"
    findings = []
    pairs = comparison.get("pairwise_system_differences", [])
    if isinstance(pairs, list):
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            left = str(pair.get("left_system_id"))
            right = str(pair.get("right_system_id"))
            differences = pair.get("state_fraction_differences", [])
            if not isinstance(differences, list):
                continue
            for row in differences:
                if not isinstance(row, dict):
                    continue
                effect = row.get("left_minus_right_fraction_of_all_evaluated")
                if not isinstance(effect, (int, float)) or isinstance(effect, bool):
                    continue
                state_id = row.get("state_id")
                findings.append(_candidate(
                    module_id=module_id, category=category,
                    statement=(
                        f"State {state_id} frame fraction differs descriptively between "
                        f"{left} and {right} by {float(effect):+.4f} ({left} minus {right})."
                    ),
                    report_path=path, effect_value=float(effect), systems=(left, right),
                    family=f"{module_id}:state_population",
                ))
    coupling = comparison.get("paired_member_state_coupling")
    if isinstance(coupling, dict) and isinstance(coupling.get("pair_reports"), list):
        for row in coupling["pair_reports"]:
            if not isinstance(row, dict) or not isinstance(row.get("cramers_v"), (int, float)):
                continue
            value = float(row["cramers_v"])
            findings.append(_candidate(
                module_id=module_id, category="coupled_interaction",
                statement=(
                    f"Equivalent members {row.get('left_member_id')} and {row.get('right_member_id')} "
                    f"have within-frame state association Cramer's V={value:.3f} in "
                    f"{row.get('system_id')}/{row.get('replica_id')}."
                ),
                report_path=path, effect_value=value,
                systems=(str(row.get("system_id")),),
                family=f"{module_id}:oligomer_state_coupling",
            ))
    return findings


def _score_correlations(
    report: Mapping[str, object], module_id: str, path: Path
) -> List[Dict[str, object]]:
    raw = report.get("paired_member_correlation", report.get("paired_member_score_correlations"))
    if not isinstance(raw, dict) or not isinstance(raw.get("pair_reports"), list):
        return []
    findings = []
    for pair in raw["pair_reports"]:
        if not isinstance(pair, dict):
            continue
        values = pair.get("same_component_correlations")
        if not isinstance(values, list):
            continue
        for component, raw_value in enumerate(values, start=1):
            if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
                continue
            value = float(raw_value)
            findings.append(_candidate(
                module_id=module_id, category="coupled_interaction",
                statement=(
                    f"Equivalent-member component {component} scores have zero-lag paired "
                    f"correlation r={value:.3f} for {pair.get('system_id')}/"
                    f"{pair.get('replica_id')}/{pair.get('segment_id')}."
                ),
                report_path=path, effect_value=value,
                systems=(str(pair.get("system_id")),),
                family=f"{module_id}:oligomer_score_correlation",
            ))
    return findings


def _atom_label(identity: Mapping[str, object]) -> str:
    chain = str(identity.get("chain_id", "")).strip()
    residue_name = str(identity.get("residue_name", "UNK")).strip()
    residue_number = identity.get("residue_number", "?")
    insertion = str(identity.get("insertion_code", "")).strip()
    atom_name = str(identity.get("atom_name", "?")).strip()
    chain_label = chain if chain else "_"
    return f"{chain_label}:{residue_name}{residue_number}{insertion}:{atom_name}"


def _rmsf_candidates(
    report: Mapping[str, object], path: Path
) -> List[Dict[str, object]]:
    systems = report.get("systems")
    if not isinstance(systems, list):
        return []
    profiles: Dict[str, Dict[int, Mapping[str, object]]] = {}
    findings = []
    for system in systems:
        if not isinstance(system, dict) or not isinstance(system.get("atom_statistics"), list):
            continue
        system_id = str(system.get("system_id"))
        rows = [
            row for row in system["atom_statistics"]
            if isinstance(row, dict)
            and isinstance(row.get("common_atom_index"), int)
            and isinstance(row.get("frame_pooled_rmsf_angstrom"), (int, float))
            and not isinstance(row.get("frame_pooled_rmsf_angstrom"), bool)
        ]
        profiles[system_id] = {int(row["common_atom_index"]): row for row in rows}
        if rows:
            top = max(
                rows,
                key=lambda row: (
                    float(row["frame_pooled_rmsf_angstrom"]),
                    -int(row["common_atom_index"]),
                ),
            )
            value = float(top["frame_pooled_rmsf_angstrom"])
            findings.append(_candidate(
                module_id="pooled_rmsf", category="rmsf",
                statement=(
                    f"Highest frame-pooled RMSF in {system_id} is {_atom_label(top)} "
                    f"at {value:.3f} angstrom."
                ),
                report_path=path, effect_value=value, systems=(system_id,),
                family="pooled_rmsf:within_system_maximum",
            ))
    for left, right in itertools.combinations(sorted(profiles), 2):
        common = sorted(set(profiles[left]).intersection(profiles[right]))
        if not common:
            continue
        atom_index = max(
            common,
            key=lambda index: (
                abs(
                    float(profiles[left][index]["frame_pooled_rmsf_angstrom"])
                    - float(profiles[right][index]["frame_pooled_rmsf_angstrom"])
                ),
                -index,
            ),
        )
        identity = profiles[left][atom_index]
        effect = (
            float(identity["frame_pooled_rmsf_angstrom"])
            - float(profiles[right][atom_index]["frame_pooled_rmsf_angstrom"])
        )
        findings.append(_candidate(
            module_id="pooled_rmsf", category="rmsf",
            statement=(
                f"Largest descriptive atom-level RMSF difference between {left} and "
                f"{right} is {_atom_label(identity)}: {effect:+.3f} angstrom "
                f"({left} minus {right})."
            ),
            report_path=path, effect_value=effect, systems=(left, right),
            family="pooled_rmsf:pairwise_atom_difference",
        ))
    return findings


def _upper_triangle(matrix: Sequence[object]) -> List[tuple[int, int, float]]:
    values = []
    for left, raw_row in enumerate(matrix):
        if not isinstance(raw_row, list):
            continue
        for right in range(left + 1, len(raw_row)):
            value = raw_row[right]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if math.isfinite(numeric):
                    values.append((left, right, numeric))
    return values


def _dccm_candidates(
    report: Mapping[str, object], path: Path
) -> List[Dict[str, object]]:
    atoms = report.get("analysis_atoms")
    systems = report.get("systems")
    if not isinstance(atoms, list) or not isinstance(systems, list):
        return []
    profiles: Dict[str, Dict[tuple[int, int], float]] = {}
    findings = []
    for system in systems:
        if not isinstance(system, dict):
            continue
        payload = system.get("frame_pooled_dccm")
        matrix = payload.get("matrix") if isinstance(payload, dict) else None
        if not isinstance(matrix, list):
            continue
        system_id = str(system.get("system_id"))
        upper = _upper_triangle(matrix)
        profiles[system_id] = {(left, right): value for left, right, value in upper}
        if upper:
            left_index, right_index, value = max(
                upper, key=lambda row: (abs(row[2]), -row[0], -row[1])
            )
            if left_index < len(atoms) and right_index < len(atoms):
                left_atom = atoms[left_index]
                right_atom = atoms[right_index]
                if isinstance(left_atom, dict) and isinstance(right_atom, dict):
                    findings.append(_candidate(
                        module_id="dccm", category="coupled_interaction",
                        statement=(
                            f"Strongest absolute pooled DCCM entry in {system_id} is "
                            f"{_atom_label(left_atom)} with {_atom_label(right_atom)}: "
                            f"{value:+.3f}."
                        ),
                        report_path=path, effect_value=value, systems=(system_id,),
                        family="dccm:within_system_extreme",
                    ))
    for left, right in itertools.combinations(sorted(profiles), 2):
        keys = set(profiles[left]).intersection(profiles[right])
        if not keys:
            continue
        pair = max(
            keys,
            key=lambda key: (
                abs(profiles[left][key] - profiles[right][key]),
                -key[0], -key[1],
            ),
        )
        effect = profiles[left][pair] - profiles[right][pair]
        left_atom = atoms[pair[0]]
        right_atom = atoms[pair[1]]
        if isinstance(left_atom, dict) and isinstance(right_atom, dict):
            findings.append(_candidate(
                module_id="dccm", category="coupled_interaction",
                statement=(
                    f"Largest descriptive DCCM difference between {left} and {right} is "
                    f"{_atom_label(left_atom)} with {_atom_label(right_atom)}: "
                    f"{effect:+.3f} ({left} minus {right})."
                ),
                report_path=path, effect_value=effect, systems=(left, right),
                family="dccm:pairwise_extreme_difference",
            ))
    return findings


def _hydrogen_bond_candidates(
    report: Mapping[str, object], path: Path
) -> List[Dict[str, object]]:
    frame_rows = report.get("frame_bond_matrix")
    occupancy_rows = report.get("occupancies")
    candidates = report.get("candidate_dictionary")
    atom_rows = report.get("atom_dictionary")
    if not isinstance(frame_rows, list) or not isinstance(candidates, list):
        return []
    atom_by_index = {
        int(row["atom_index"]): row.get("identity", {})
        for row in atom_rows or []
        if isinstance(row, dict) and isinstance(row.get("atom_index"), int)
    } if isinstance(atom_rows, list) else {}
    candidate_by_id = {
        str(row["bond_id"]): row
        for row in candidates
        if isinstance(row, dict) and "bond_id" in row
    }
    counts: Dict[str, Dict[str, int]] = {}
    totals: Dict[str, int] = {}
    if isinstance(occupancy_rows, list):
        for row in frame_rows:
            if isinstance(row, dict) and isinstance(row.get("system_id"), str):
                system_id = str(row["system_id"])
                totals[system_id] = totals.get(system_id, 0) + 1
        for row in occupancy_rows:
            if not isinstance(row, dict):
                continue
            system_id = str(row.get("system_id"))
            bond_id = row.get("bond_id")
            evaluated = row.get("evaluated_frame_count")
            present = row.get("present_frame_count")
            if (
                not isinstance(bond_id, str)
                or isinstance(evaluated, bool) or not isinstance(evaluated, int)
                or isinstance(present, bool) or not isinstance(present, int)
                or evaluated < 1 or present < 0 or present > evaluated
            ):
                continue
            system_counts = counts.setdefault(system_id, {})
            system_counts[bond_id] = system_counts.get(bond_id, 0) + present
    else:
        for row in frame_rows:
            if not isinstance(row, dict) or not isinstance(row.get("present_bond_ids"), list):
                continue
            system_id = str(row.get("system_id"))
            totals[system_id] = totals.get(system_id, 0) + 1
            system_counts = counts.setdefault(system_id, {})
            for bond_id in row["present_bond_ids"]:
                key = str(bond_id)
                system_counts[key] = system_counts.get(key, 0) + 1

    def label(bond_id: str) -> str:
        candidate = candidate_by_id.get(bond_id, {})
        donor = atom_by_index.get(int(candidate.get("donor_atom_index", -1)), {})
        acceptor = atom_by_index.get(int(candidate.get("acceptor_atom_index", -1)), {})
        if isinstance(donor, dict) and isinstance(acceptor, dict) and donor and acceptor:
            return f"{_atom_label(donor)} to {_atom_label(acceptor)}"
        return bond_id

    findings = []
    for system_id in sorted(totals):
        if totals[system_id] and counts.get(system_id):
            bond_id, count = max(
                counts[system_id].items(), key=lambda row: (row[1], row[0])
            )
            occupancy = count / totals[system_id]
            findings.append(_candidate(
                module_id="hydrogen_bond_discovery", category="other_physical",
                statement=(
                    f"Most occupied discovered direct hydrogen bond in {system_id} is "
                    f"{label(bond_id)} at {occupancy:.1%} of evaluated frames."
                ),
                report_path=path, effect_value=occupancy, systems=(system_id,),
                family="hydrogen_bond_discovery:within_system_maximum",
            ))
    for left, right in itertools.combinations(sorted(totals), 2):
        bond_ids = set(counts.get(left, {})).union(counts.get(right, {}))
        if not bond_ids or not totals[left] or not totals[right]:
            continue
        bond_id = max(
            bond_ids,
            key=lambda key: (
                abs(
                    counts.get(left, {}).get(key, 0) / totals[left]
                    - counts.get(right, {}).get(key, 0) / totals[right]
                ),
                key,
            ),
        )
        effect = (
            counts.get(left, {}).get(bond_id, 0) / totals[left]
            - counts.get(right, {}).get(bond_id, 0) / totals[right]
        )
        findings.append(_candidate(
            module_id="hydrogen_bond_discovery", category="other_physical",
            statement=(
                f"Largest descriptive direct-hydrogen-bond occupancy difference between "
                f"{left} and {right} is {label(bond_id)}: {effect:+.1%} "
                f"({left} minus {right})."
            ),
            report_path=path, effect_value=effect, systems=(left, right),
            family="hydrogen_bond_discovery:pairwise_occupancy_difference",
        ))
    return findings


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _standardized_difference(left: Sequence[float], right: Sequence[float]) -> tuple[float, float | None]:
    raw = _mean(left) - _mean(right)
    if len(left) < 2 or len(right) < 2:
        return raw, None
    left_mean = _mean(left)
    right_mean = _mean(right)
    variance = (
        sum((value - left_mean) ** 2 for value in left) / (len(left) - 1)
        + sum((value - right_mean) ** 2 for value in right) / (len(right) - 1)
    ) / 2.0
    return raw, raw / math.sqrt(variance) if variance > 0.0 else None


def _replica_metric_candidates(
    report: Mapping[str, object], module_id: str, path: Path
) -> List[Dict[str, object]]:
    rows = report.get("replica_reports")
    if not isinstance(rows, list):
        return []
    means: Dict[str, Dict[str, List[float]]] = {}
    shifts: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        system_id = str(row.get("system_id"))
        summaries = row.get("metric_summaries")
        if isinstance(summaries, dict):
            for metric, summary in summaries.items():
                value = summary.get("mean") if isinstance(summary, dict) else None
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    means.setdefault(system_id, {}).setdefault(str(metric), []).append(float(value))
        raw_shifts = row.get("late_minus_early_metric_means")
        if isinstance(raw_shifts, dict):
            for metric, value in raw_shifts.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    shifts.setdefault(system_id, {}).setdefault(str(metric), []).append(float(value))
    findings = []
    for system_id, metric_rows in shifts.items():
        candidates = [
            (abs(_mean(values)), metric, _mean(values))
            for metric, values in metric_rows.items() if values
        ]
        if candidates:
            _, metric, value = max(candidates)
            findings.append(_candidate(
                module_id=module_id, category="other_physical",
                statement=(
                    f"Largest mean replica-level late-minus-early {module_id} change in "
                    f"{system_id} is {metric}: {value:+.4g} in the metric's declared units."
                ),
                report_path=path, effect_value=value, systems=(system_id,),
                family=f"{module_id}:within_system_temporal_change",
            ))
    for left, right in itertools.combinations(sorted(means), 2):
        metrics = set(means[left]).intersection(means[right])
        candidates = []
        for metric in metrics:
            raw, standardized = _standardized_difference(
                means[left][metric], means[right][metric]
            )
            rank_value = standardized if standardized is not None else raw
            candidates.append((abs(rank_value), metric, raw, standardized))
        if candidates:
            _, metric, raw, standardized = max(
                candidates, key=lambda row: (row[0], row[1])
            )
            qualifier = (
                f"; standardized replica-mean difference {standardized:+.3f}"
                if standardized is not None else ""
            )
            findings.append(_candidate(
                module_id=module_id, category="other_physical",
                statement=(
                    f"Largest descriptive {module_id} metric difference between {left} "
                    f"and {right} is {metric}: {raw:+.4g} ({left} minus {right})"
                    f"{qualifier}."
                ),
                report_path=path,
                effect_value=standardized if standardized is not None else raw,
                systems=(left, right), family=f"{module_id}:pairwise_metric_difference",
            ))
    return findings


def _residue_label(key: tuple[str, int, str, str]) -> str:
    chain, number, insertion, name = key
    return f"{chain or '_'}:{name}{number}{insertion}"


def _sasa_candidates(report: Mapping[str, object], path: Path) -> List[Dict[str, object]]:
    replicas = report.get("replicas")
    if not isinstance(replicas, list):
        return []
    values: Dict[str, Dict[tuple[str, int, str, str], List[float]]] = {}
    for replica in replicas:
        if not isinstance(replica, dict) or not isinstance(replica.get("per_residue_summaries"), list):
            continue
        system_id = str(replica.get("system_id"))
        for row in replica["per_residue_summaries"]:
            if not isinstance(row, dict) or not isinstance(row.get("summary_angstrom2"), dict):
                continue
            mean = row["summary_angstrom2"].get("mean")
            if not isinstance(mean, (int, float)) or isinstance(mean, bool):
                continue
            key = (
                str(row.get("chain_id", "")), int(row.get("residue_number", 0)),
                str(row.get("insertion_code", "")), str(row.get("residue_name", "UNK")),
            )
            values.setdefault(system_id, {}).setdefault(key, []).append(float(mean))
    findings = []
    for system_id, residue_rows in values.items():
        if residue_rows:
            key = max(residue_rows, key=lambda item: _mean(residue_rows[item]))
            value = _mean(residue_rows[key])
            findings.append(_candidate(
                module_id="solvent_accessible_surface_area", category="other_physical",
                statement=(
                    f"Most solvent-exposed residue by equal-replica mean SASA in {system_id} "
                    f"is {_residue_label(key)} at {value:.3f} angstrom squared."
                ),
                report_path=path, effect_value=value, systems=(system_id,),
                family="solvent_accessible_surface_area:within_system_residue_maximum",
            ))
    for left, right in itertools.combinations(sorted(values), 2):
        common = set(values[left]).intersection(values[right])
        if not common:
            continue
        key = max(
            common,
            key=lambda item: abs(_mean(values[left][item]) - _mean(values[right][item])),
        )
        effect = _mean(values[left][key]) - _mean(values[right][key])
        findings.append(_candidate(
            module_id="solvent_accessible_surface_area", category="other_physical",
            statement=(
                f"Largest descriptive residue SASA difference between {left} and {right} "
                f"is {_residue_label(key)}: {effect:+.3f} angstrom squared "
                f"({left} minus {right})."
            ),
            report_path=path, effect_value=effect, systems=(left, right),
            family="solvent_accessible_surface_area:pairwise_residue_difference",
        ))
    return findings


def _observable_candidates(report: Mapping[str, object], path: Path) -> List[Dict[str, object]]:
    rows = report.get("feature_reports")
    if not isinstance(rows, list):
        return []
    values: Dict[str, Dict[str, List[float]]] = {}
    metadata: Dict[str, tuple[str, str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        system_id = str(row.get("system_id"))
        feature_id = str(row.get("feature_id"))
        kind = str(row.get("kind"))
        units = "fraction"
        value = row.get("contact_occupancy_fraction")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            summary = row.get("native_contact_fraction_summary")
            value = summary.get("mean") if isinstance(summary, dict) else None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            summary = row.get("distance_summary_angstrom")
            value = summary.get("mean") if isinstance(summary, dict) else None
            units = "angstrom"
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        values.setdefault(system_id, {}).setdefault(feature_id, []).append(float(value))
        metadata[feature_id] = (str(row.get("question", feature_id)), kind, units)
    findings = []
    for left, right in itertools.combinations(sorted(values), 2):
        common = set(values[left]).intersection(values[right])
        candidates = []
        for feature_id in common:
            raw, standardized = _standardized_difference(
                values[left][feature_id], values[right][feature_id]
            )
            candidates.append((
                abs(standardized if standardized is not None else raw),
                feature_id, raw, standardized,
            ))
        if not candidates:
            continue
        _, feature_id, raw, standardized = max(
            candidates, key=lambda row: (row[0], row[1])
        )
        question, kind, units = metadata[feature_id]
        findings.append(_candidate(
            module_id="optional_observables", category="other_physical",
            statement=(
                f"Largest descriptive declared-observable difference between {left} and "
                f"{right} is {feature_id} ({question}; {kind}): {raw:+.4g} {units} "
                f"({left} minus {right})."
            ),
            report_path=path,
            effect_value=standardized if standardized is not None else raw,
            systems=(left, right), family="optional_observables:pairwise_feature_difference",
        ))
    return findings


def _rdf_candidates(report: Mapping[str, object], path: Path) -> List[Dict[str, object]]:
    rows = report.get("feature_reports")
    if not isinstance(rows, list):
        return []
    values: Dict[str, Dict[tuple[str, int], List[float]]] = {}
    centers: Dict[tuple[str, int], float] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("bins"), list):
            continue
        system_id = str(row.get("system_id"))
        feature_id = str(row.get("feature_id"))
        for bin_row in row["bins"]:
            if not isinstance(bin_row, dict):
                continue
            value = bin_row.get("g_r")
            index = bin_row.get("bin_index")
            center = bin_row.get("center_radius_angstrom")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or not isinstance(index, int):
                continue
            key = (feature_id, index)
            values.setdefault(system_id, {}).setdefault(key, []).append(float(value))
            if isinstance(center, (int, float)):
                centers[key] = float(center)
    findings = []
    for system_id, bins in values.items():
        if bins:
            key = max(bins, key=lambda item: _mean(bins[item]))
            value = _mean(bins[key])
            findings.append(_candidate(
                module_id="radial_distribution_functions", category="other_physical",
                statement=(
                    f"Strongest equal-replica mean RDF peak in {system_id} is "
                    f"{key[0]} at {centers.get(key, float('nan')):.3f} angstrom with "
                    f"g(r)={value:.3f}."
                ),
                report_path=path, effect_value=value, systems=(system_id,),
                family="radial_distribution_functions:within_system_peak",
            ))
    for left, right in itertools.combinations(sorted(values), 2):
        common = set(values[left]).intersection(values[right])
        if not common:
            continue
        key = max(
            common,
            key=lambda item: abs(_mean(values[left][item]) - _mean(values[right][item])),
        )
        effect = _mean(values[left][key]) - _mean(values[right][key])
        findings.append(_candidate(
            module_id="radial_distribution_functions", category="other_physical",
            statement=(
                f"Largest descriptive RDF difference between {left} and {right} is "
                f"{key[0]} at {centers.get(key, float('nan')):.3f} angstrom: "
                f"delta g(r)={effect:+.3f} ({left} minus {right})."
            ),
            report_path=path, effect_value=effect, systems=(left, right),
            family="radial_distribution_functions:pairwise_bin_difference",
        ))
    return findings


def _dihedral_candidates(report: Mapping[str, object], path: Path) -> List[Dict[str, object]]:
    rows = report.get("circular_summaries")
    if not isinstance(rows, list):
        return []
    values: Dict[str, Dict[tuple[str, int, str, str], List[float]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get("mean_angle_degrees")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        key = (
            str(row.get("chain_id", "")), int(row.get("residue_number", 0)),
            str(row.get("insertion_code", "")), str(row.get("angle_type", "")),
        )
        values.setdefault(str(row.get("system_id")), {}).setdefault(key, []).append(float(value))
    findings = []
    for left, right in itertools.combinations(sorted(values), 2):
        common = set(values[left]).intersection(values[right])
        candidates = []
        for key in common:
            left_angle = math.degrees(math.atan2(
                _mean([math.sin(math.radians(value)) for value in values[left][key]]),
                _mean([math.cos(math.radians(value)) for value in values[left][key]]),
            ))
            right_angle = math.degrees(math.atan2(
                _mean([math.sin(math.radians(value)) for value in values[right][key]]),
                _mean([math.cos(math.radians(value)) for value in values[right][key]]),
            ))
            difference = (left_angle - right_angle + 180.0) % 360.0 - 180.0
            candidates.append((abs(difference), key, difference))
        if candidates:
            _, key, effect = max(candidates)
            findings.append(_candidate(
                module_id="dihedral_distributions", category="other_physical",
                statement=(
                    f"Largest descriptive circular-mean dihedral difference between "
                    f"{left} and {right} is {key[0] or '_'}:{key[1]}{key[2]} {key[3]}: "
                    f"{effect:+.2f} degrees ({left} minus {right}, wrapped to +/-180)."
                ),
                report_path=path, effect_value=effect, systems=(left, right),
                family="dihedral_distributions:pairwise_circular_mean_difference",
            ))
    return findings


def _water_network_candidates(report: Mapping[str, object], path: Path) -> List[Dict[str, object]]:
    rows = report.get("bridge_occupancies")
    dictionaries = report.get("observed_bridge_dictionary")
    endpoints = report.get("endpoint_dictionary")
    if not isinstance(rows, list) or not isinstance(dictionaries, list) or not isinstance(endpoints, list):
        return []
    atom_labels = {
        (str(row.get("system_id")), str(row.get("replica_id")), int(row["atom_index"])):
        _atom_label(row.get("identity", {}))
        for row in endpoints
        if isinstance(row, dict) and isinstance(row.get("atom_index"), int)
        and isinstance(row.get("identity"), dict)
    }
    bridge_indices = {}
    for row in dictionaries:
        if not isinstance(row, dict):
            continue
        bridge_id = str(row.get("bridge_id"))
        first = row.get("first_endpoint_atom_index")
        second = row.get("second_endpoint_atom_index")
        if isinstance(first, int) and isinstance(second, int):
            bridge_indices[(
                str(row.get("system_id", "")), str(row.get("replica_id", "")), bridge_id,
            )] = (first, second)
    values: Dict[str, Dict[tuple[str, str], List[float]]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("cutoff_id") != "primary":
            continue
        system_id = str(row.get("system_id"))
        replica_id = str(row.get("replica_id"))
        bridge_id = str(row.get("bridge_id"))
        indices = bridge_indices.get((system_id, replica_id, bridge_id))
        if indices is None:
            matching = [
                value for (candidate_system, candidate_replica, candidate_id), value
                in bridge_indices.items()
                if candidate_id == bridge_id
                and candidate_system in {"", system_id}
                and candidate_replica in {"", replica_id}
            ]
            if len(set(matching)) == 1:
                indices = matching[0]
        key = None
        if indices:
            first = atom_labels.get((system_id, replica_id, indices[0]))
            second = atom_labels.get((system_id, replica_id, indices[1]))
            if first and second:
                key = tuple(sorted((first, second)))
        value = row.get("occupancy_fraction")
        if key and isinstance(value, (int, float)) and not isinstance(value, bool):
            values.setdefault(system_id, {}).setdefault(key, []).append(float(value))
    findings = []
    for system_id, bridges in values.items():
        if bridges:
            key = max(bridges, key=lambda item: _mean(bridges[item]))
            value = _mean(bridges[key])
            findings.append(_candidate(
                module_id="water_mediated_hydrogen_bond_networks",
                category="coupled_interaction",
                statement=(
                    f"Most occupied primary one-water bridge in {system_id} connects "
                    f"{key[0]} and {key[1]} at equal-replica mean occupancy {value:.1%}."
                ),
                report_path=path, effect_value=value, systems=(system_id,),
                family="water_mediated_hydrogen_bond_networks:within_system_maximum",
            ))
    for left, right in itertools.combinations(sorted(values), 2):
        common = set(values[left]).intersection(values[right])
        if not common:
            continue
        key = max(
            common,
            key=lambda item: abs(_mean(values[left][item]) - _mean(values[right][item])),
        )
        effect = _mean(values[left][key]) - _mean(values[right][key])
        findings.append(_candidate(
            module_id="water_mediated_hydrogen_bond_networks",
            category="coupled_interaction",
            statement=(
                f"Largest descriptive shared one-water-bridge occupancy difference between "
                f"{left} and {right} connects {key[0]} and {key[1]}: {effect:+.1%} "
                f"({left} minus {right})."
            ),
            report_path=path, effect_value=effect, systems=(left, right),
            family="water_mediated_hydrogen_bond_networks:pairwise_occupancy_difference",
        ))
    return findings


def _pald_community_candidates(
    report: Mapping[str, object], path: Path,
) -> List[Dict[str, object]]:
    communities = report.get("communities")
    if not isinstance(communities, list) or not communities:
        return []
    sampled = report.get("sampled_observation_count")
    sample_label = str(sampled) if isinstance(sampled, int) else "the bounded"
    findings = [_candidate(
        module_id="pald_community_analysis", category="other_physical",
        statement=(
            f"PaLD strong ties define {len(communities)} connected communities "
            f"among {sample_label} regularly sampled observations."
        ),
        report_path=path,
        family="pald_community_analysis:sampled_community_count",
    )]
    cores = [
        row for row in communities
        if isinstance(row, dict)
        and isinstance(row.get("core_observation"), dict)
        and isinstance(row["core_observation"].get("local_depth"), (int, float))
        and not isinstance(row["core_observation"].get("local_depth"), bool)
    ]
    if cores:
        core = max(
            cores,
            key=lambda row: float(row["core_observation"]["local_depth"]),
        )
        observation = core["core_observation"]
        depth = float(observation["local_depth"])
        findings.append(_candidate(
            module_id="pald_community_analysis", category="other_physical",
            statement=(
                f"The deepest sampled PaLD community core is community "
                f"{core.get('community_id')} at {observation.get('system_id')}/"
                f"{observation.get('replica_id')}/frame "
                f"{observation.get('source_frame_index')} with local depth "
                f"{depth:.4f}."
            ),
            report_path=path, effect_value=depth,
            systems=(str(observation.get("system_id")),),
            family="pald_community_analysis:sampled_core_depth",
        ))
    ties = report.get("strongest_intercommunity_ties")
    if isinstance(ties, list) and ties and isinstance(ties[0], dict):
        tie = ties[0]
        value = tie.get("mutual_cohesion")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            findings.append(_candidate(
                module_id="pald_community_analysis", category="other_physical",
                statement=(
                    f"The strongest sampled PaLD tie crossing connected communities "
                    f"links communities {tie.get('left_community_id')} and "
                    f"{tie.get('right_community_id')} with mutual cohesion "
                    f"{float(value):.4f}."
                ),
                report_path=path, effect_value=float(value),
                family="pald_community_analysis:sampled_intercommunity_tie",
            ))
    return findings


_CROSS_REPORT_ROWS = {
    "nucleic_acid_geometry": "replica_reports",
    "ion_coordination_geometry": "replica_reports",
    "solvent_accessible_surface_area": "replicas",
    "optional_observables": "feature_reports",
    "radial_distribution_functions": "feature_reports",
    "dihedral_distributions": "circular_summaries",
    "water_mediated_hydrogen_bond_networks": "bridge_occupancies",
}


def _compact_cross_report(
    report: Mapping[str, object], module_id: str
) -> Dict[str, object] | None:
    row_key = _CROSS_REPORT_ROWS.get(module_id)
    rows = report.get(row_key) if row_key else None
    if not row_key or not isinstance(rows, list):
        return None
    compact: Dict[str, object] = {"module_id": module_id, row_key: []}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if module_id in {"nucleic_acid_geometry", "ion_coordination_geometry"}:
            kept = {
                key: row.get(key) for key in (
                    "system_id", "replica_id", "evaluated_frame_count",
                    "metric_summaries", "late_minus_early_metric_means",
                )
            }
        elif module_id == "solvent_accessible_surface_area":
            kept = {
                key: row.get(key) for key in (
                    "system_id", "replica_id", "evaluated_frame_count",
                    "per_residue_summaries",
                )
            }
        elif module_id == "optional_observables":
            kept = {
                key: row.get(key) for key in (
                    "system_id", "replica_id", "feature_id", "kind", "question",
                    "evaluated_frame_count", "contact_occupancy_fraction",
                    "native_contact_fraction_summary", "distance_summary_angstrom",
                )
            }
        elif module_id == "radial_distribution_functions":
            kept = {
                key: row.get(key) for key in (
                    "system_id", "replica_id", "feature_id", "question",
                    "evaluated_frame_count", "bins",
                )
            }
        elif module_id == "dihedral_distributions":
            kept = {
                key: row.get(key) for key in (
                    "system_id", "replica_id", "chain_id", "residue_number",
                    "insertion_code", "residue_name", "angle_type", "count",
                    "mean_angle_degrees",
                )
            }
        else:
            kept = {
                key: row.get(key) for key in (
                    "system_id", "replica_id", "bridge_id", "cutoff_id",
                    "evaluated_frame_count", "occupancy_fraction",
                )
            }
        compact[row_key].append(kept)
    if module_id == "water_mediated_hydrogen_bond_networks":
        for key in ("observed_bridge_dictionary", "endpoint_dictionary"):
            source = report.get(key)
            if isinstance(source, list):
                if key == "observed_bridge_dictionary":
                    compact[key] = [
                        {
                            field: row.get(field) for field in (
                                "system_id", "replica_id", "bridge_id",
                                "first_endpoint_atom_index", "second_endpoint_atom_index",
                            )
                        }
                        for row in source if isinstance(row, dict)
                    ]
                else:
                    compact[key] = [
                        {
                            field: row.get(field) for field in (
                                "system_id", "replica_id", "atom_index", "identity",
                            )
                        }
                        for row in source if isinstance(row, dict)
                    ]
    return compact


def finding_sidecar_evidence(
    report: Mapping[str, object], report_path: Path
) -> Dict[str, object]:
    """Create compact finding evidence while the full report is already in memory."""

    path = Path(report_path).expanduser().resolve(strict=False)
    module_id = str(report.get("module_id", path.parent.name))
    return {
        "finding_evidence_schema": "salsbury-finding-evidence-v1",
        "module_id": module_id,
        "report_path": str(path),
        "candidates": _report_candidates(path, report),
        "cross_report_summary": _compact_cross_report(report, module_id),
    }


def _row_coverage(row: Mapping[str, object]) -> int:
    for key in ("evaluated_frame_count", "count"):
        value = row.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    for key in (
        "distance_summary_angstrom", "native_contact_fraction_summary",
        "summary_angstrom2",
    ):
        summary = row.get(key)
        value = summary.get("count") if isinstance(summary, dict) else None
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    timeseries = row.get("timeseries")
    return len(timeseries) if isinstance(timeseries, list) else 0


def _report_system_coverage(
    report: Mapping[str, object], module_id: str
) -> Dict[str, int]:
    row_key = _CROSS_REPORT_ROWS[module_id]
    rows = report.get(row_key)
    if not isinstance(rows, list):
        return {}
    per_replica: Dict[str, Dict[str, int]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        system_id = str(row.get("system_id"))
        replica_id = str(row.get("replica_id", "all"))
        coverage = _row_coverage(row)
        per_replica.setdefault(system_id, {})[replica_id] = max(
            coverage, per_replica.get(system_id, {}).get(replica_id, 0)
        )
    return {
        system_id: sum(replica_rows.values())
        for system_id, replica_rows in per_replica.items()
    }


def _cross_report_candidates(
    records: Sequence[tuple[Path, Mapping[str, object]]]
) -> List[Dict[str, object]]:
    by_module: Dict[str, List[tuple[Path, Mapping[str, object]]]] = {}
    for path, report in records:
        module_id = str(report.get("module_id", path.parent.name))
        if module_id in _CROSS_REPORT_ROWS:
            by_module.setdefault(module_id, []).append((path, report))
    findings = []
    for module_id, module_records in by_module.items():
        selected: Dict[str, tuple[int, str, Path, Mapping[str, object]]] = {}
        for path, report in module_records:
            for system_id, coverage in _report_system_coverage(report, module_id).items():
                candidate = (coverage, str(path), path, report)
                current = selected.get(system_id)
                if current is None or (coverage, str(path)) > current[:2]:
                    selected[system_id] = candidate
        source_paths = sorted({value[2] for value in selected.values()})
        if len(selected) < 2 or len(source_paths) < 2:
            continue
        row_key = _CROSS_REPORT_ROWS[module_id]
        merged: Dict[str, object] = {"module_id": module_id, row_key: []}
        for system_id in sorted(selected):
            report = selected[system_id][3]
            rows = report.get(row_key)
            if isinstance(rows, list):
                merged[row_key].extend(
                    row for row in rows
                    if isinstance(row, dict) and str(row.get("system_id")) == system_id
                )
            if module_id == "water_mediated_hydrogen_bond_networks":
                for key in ("observed_bridge_dictionary", "endpoint_dictionary"):
                    source = report.get(key)
                    merged.setdefault(key, [])
                    if isinstance(source, list):
                        merged[key].extend(
                            row for row in source
                            if isinstance(row, dict)
                            and str(row.get("system_id")) == system_id
                        )
        primary = source_paths[0]
        if module_id in {"nucleic_acid_geometry", "ion_coordination_geometry"}:
            candidates = _replica_metric_candidates(merged, module_id, primary)
        elif module_id == "solvent_accessible_surface_area":
            candidates = _sasa_candidates(merged, primary)
        elif module_id == "optional_observables":
            candidates = _observable_candidates(merged, primary)
        elif module_id == "radial_distribution_functions":
            candidates = _rdf_candidates(merged, primary)
        elif module_id == "dihedral_distributions":
            candidates = _dihedral_candidates(merged, primary)
        else:
            candidates = _water_network_candidates(merged, primary)
        for candidate in candidates:
            if len(candidate["system_ids"]) >= 2:
                candidate["report_paths"] = [str(value) for value in source_paths]
                findings.append(candidate)
    return findings


def _experimental_method_candidates(
    report: Mapping[str, object], module_id: str, path: Path
) -> List[Dict[str, object]]:
    """Surface bounded descriptive evidence from default-off method reports."""

    findings: List[Dict[str, object]] = []
    systems = report.get("systems")
    if module_id == "perturbation_response_dynamics" and isinstance(systems, list):
        for system in systems:
            if not isinstance(system, dict):
                continue
            values = system.get("dci") or system.get("dfi")
            metric = "DCI" if isinstance(system.get("dci"), list) and system.get("dci") else "DFI"
            if not isinstance(values, list):
                continue
            numeric = [
                (index, float(value)) for index, value in enumerate(values)
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value))
            ]
            if numeric:
                index, value = max(numeric, key=lambda row: (row[1], -row[0]))
                system_id = str(system.get("system_id"))
                findings.append(_candidate(
                    module_id=module_id, category="coupled_interaction",
                    statement=(
                        f"Largest descriptive {metric} value in {system_id} is node "
                        f"{index}: {value:.4g}."
                    ), report_path=path, effect_value=value, systems=(system_id,),
                    family=f"{module_id}:within_system_{metric.lower()}_maximum",
                ))
    elif module_id == "trajectory_reweighting" and isinstance(systems, list):
        for system in systems:
            diagnostics = system.get("diagnostics") if isinstance(system, dict) else None
            if not isinstance(diagnostics, dict):
                continue
            ratio = diagnostics.get("kish_effective_sample_size_ratio")
            if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
                ratio = diagnostics.get("kish_ratio")
            if isinstance(ratio, (int, float)) and not isinstance(ratio, bool):
                system_id = str(system.get("system_id"))
                status = str(diagnostics.get("reweighting_validity_status", "not evaluated"))
                findings.append(_candidate(
                    module_id=module_id, category="other_physical",
                    statement=(
                        f"Frame-weight concentration gate for {system_id} is {status}; "
                        f"the Kish effective-sample ratio is {float(ratio):.1%}."
                    ), report_path=path, effect_value=float(ratio), systems=(system_id,),
                    family=f"{module_id}:weight_reliability",
                ))
    elif module_id == "allosteric_pathways" and isinstance(systems, list):
        nodes = report.get("nodes")
        for system in systems:
            network = system.get("network") if isinstance(system, dict) else None
            if not isinstance(network, dict):
                continue
            values = network.get("combined_allosteric_score")
            metric = "combined prioritization score"
            if not isinstance(values, list):
                values = network.get("weighted_betweenness_centrality")
                metric = "weighted betweenness"
            numeric = [
                (index, float(value)) for index, value in enumerate(values or [])
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if numeric:
                index, value = max(numeric, key=lambda row: (row[1], -row[0]))
                label = str(index)
                if isinstance(nodes, list) and index < len(nodes) and isinstance(nodes[index], dict):
                    label = str(nodes[index].get("node_id", index))
                system_id = str(system.get("system_id"))
                findings.append(_candidate(
                    module_id=module_id, category="coupled_interaction",
                    statement=(
                        f"Highest descriptive pathway {metric} in {system_id} is "
                        f"{label}: {value:.4g}."
                    ), report_path=path, effect_value=value, systems=(system_id,),
                    family=f"{module_id}:within_system_node_priority",
                ))
    elif (
        module_id == "energetic_network_embeddings"
        and report.get("availability_status") == "available"
    ):
        comparisons = report.get("pairwise_system_comparisons")
        if isinstance(comparisons, list):
            for comparison in comparisons:
                rows = comparison.get("residue_distances") if isinstance(comparison, dict) else None
                choices = [
                    row for row in rows or [] if isinstance(row, dict)
                    and isinstance(row.get("summed_wasserstein_distance"), (int, float))
                    and not isinstance(row.get("summed_wasserstein_distance"), bool)
                ] if isinstance(rows, list) else []
                if not choices:
                    continue
                row = max(
                    choices,
                    key=lambda value: float(value["summed_wasserstein_distance"]),
                )
                value = float(row["summed_wasserstein_distance"])
                system_ids = (
                    str(comparison.get("system_i")),
                    str(comparison.get("system_j")),
                )
                findings.append(_candidate(
                    module_id=module_id, category="coupled_interaction",
                    statement=(
                        "Largest descriptive protein-only energetic-network "
                        f"embedding shift between {system_ids[0]} and {system_ids[1]} "
                        f"is {row.get('node_id')}: summed marginal Wasserstein "
                        f"distance {value:.4g}."
                    ), report_path=path, effect_value=value, systems=system_ids,
                    family=f"{module_id}:residue_wasserstein",
                ))
    elif module_id == "multivalent_molecular_bridges":
        rows = report.get("mediator_type_summaries")
        if isinstance(rows, list):
            for system_id in sorted({str(row.get("system_id")) for row in rows if isinstance(row, dict)}):
                choices = [
                    row for row in rows if isinstance(row, dict)
                    and str(row.get("system_id")) == system_id
                    and isinstance(row.get("bridge_occupancy"), (int, float))
                ]
                if choices:
                    row = max(choices, key=lambda value: float(value["bridge_occupancy"]))
                    occupancy = float(row["bridge_occupancy"])
                    findings.append(_candidate(
                        module_id=module_id, category="other_physical",
                        statement=(
                            f"Most occupied multivalent mediator type in {system_id} is "
                            f"{row.get('mediator_type')} at {occupancy:.1%} of evaluated mediator-frames."
                        ), report_path=path, effect_value=occupancy, systems=(system_id,),
                        family=f"{module_id}:within_system_mediator_occupancy",
                    ))
    elif module_id == "reactive_path_ensembles":
        count = report.get("complete_path_count")
        if isinstance(count, int) and not isinstance(count, bool):
            status = str(report.get("transition_sufficiency_status", "not evaluated"))
            findings.append(_candidate(
                module_id=module_id, category="other_physical",
                statement=(
                    f"Reactive-path extraction found {count} complete paths; the "
                    f"transition-sufficiency gate is {status}."
                ), report_path=path, effect_value=float(count),
                family=f"{module_id}:path_sufficiency",
            ))
    elif module_id == "interaction_fingerprints":
        occupancies = report.get("feature_occupancies")
        if isinstance(occupancies, list):
            choices = [
                row for row in occupancies if isinstance(row, dict)
                and isinstance(row.get("occupancy_fraction"), (int, float))
            ]
            if choices:
                row = max(choices, key=lambda value: float(value["occupancy_fraction"]))
                value = float(row["occupancy_fraction"])
                findings.append(_candidate(
                    module_id=module_id, category="coupled_interaction",
                    statement=(
                        f"Most occupied retained interaction-fingerprint feature is "
                        f"{row.get('feature_id')} at {value:.1%} of its source-observed frames."
                    ), report_path=path, effect_value=value,
                    family=f"{module_id}:feature_occupancy",
                ))
    elif module_id == "spatial_interaction_ensembles":
        comparisons = report.get("pairwise_system_spatial_differences")
        choices = [
            row for row in comparisons if isinstance(row, dict)
            and isinstance(row.get("centroid_displacement_angstrom"), (int, float))
            and not isinstance(row.get("centroid_displacement_angstrom"), bool)
        ] if isinstance(comparisons, list) else []
        if choices:
            row = max(
                choices,
                key=lambda value: float(value["centroid_displacement_angstrom"]),
            )
            displacement = float(row["centroid_displacement_angstrom"])
            systems = (str(row.get("system_i")), str(row.get("system_j")))
            findings.append(_candidate(
                module_id=module_id, category="coupled_interaction",
                statement=(
                    "Largest gated descriptive interaction-cloud centroid shift is "
                    f"{row.get('superfeature_id')} between {systems[0]} and "
                    f"{systems[1]}: {displacement:.4g} Å."
                ),
                report_path=path, effect_value=displacement, systems=systems,
                family=f"{module_id}:centroid_displacement",
            ))
        else:
            selected = report.get("selected_spatial_mode_candidates")
            mode_rows = [
                row for row in selected if isinstance(row, dict)
                and isinstance(row.get("silhouette"), (int, float))
                and not isinstance(row.get("silhouette"), bool)
            ] if isinstance(selected, list) else []
            if mode_rows:
                row = max(mode_rows, key=lambda value: float(value["silhouette"]))
                score = float(row["silhouette"])
                system_id = str(row.get("system_id"))
                findings.append(_candidate(
                    module_id=module_id, category="coupled_interaction",
                    statement=(
                        f"Gated spatial mode candidate for {row.get('superfeature_id')} "
                        f"in {system_id} has k={row.get('k')} and exact silhouette "
                        f"{score:.4g}; it is not a binding-state assignment."
                    ),
                    report_path=path, effect_value=score, systems=(system_id,),
                    family=f"{module_id}:gated_spatial_mode",
                ))
    elif module_id == "interaction_persistence":
        summaries = report.get("feature_persistence_summaries")
        choices = [
            row for row in summaries if isinstance(row, dict)
            and row.get("gap_tolerance_observations") == 0
            and row.get("persistence_summary_gate") == "passed"
            and isinstance(row.get("complete_event_duration_summary"), dict)
            and isinstance(
                row["complete_event_duration_summary"].get("median"),
                (int, float),
            )
        ] if isinstance(summaries, list) else []
        if choices:
            row = max(
                choices,
                key=lambda value: float(
                    value["complete_event_duration_summary"]["median"]
                ),
            )
            duration = float(
                row["complete_event_duration_summary"]["median"]
            )
            findings.append(_candidate(
                module_id=module_id, category="coupled_interaction",
                statement=(
                    "Longest gated primary zero-gap fingerprint persistence is "
                    f"{row.get('feature_id')} in {row.get('system_id')}: median "
                    f"complete-event duration {duration:.4g} {row.get('time_unit')} "
                    f"across {row.get('complete_event_count')} complete events."
                ),
                report_path=path, effect_value=duration,
                systems=(str(row.get("system_id")),),
                family=f"{module_id}:complete_event_duration",
            ))
        elif report.get("persistence_readiness_status") == "insufficient_complete_events":
            findings.append(_candidate(
                module_id=module_id, category="coupled_interaction",
                statement=(
                    "Interaction-persistence analysis withheld duration ranking: "
                    "no zero-gap feature/system series passed the configured "
                    "complete-event gate."
                ),
                report_path=path, effect_value=0.0,
                family=f"{module_id}:complete_event_gate",
            ))
    elif module_id == "random_feature_koopman":
        selected = report.get("selected_hyperparameters")
        if isinstance(selected, dict) and isinstance(
            selected.get("selection_score"), (int, float)
        ):
            score = float(selected["selection_score"])
            findings.append(_candidate(
                module_id=module_id, category="other_physical",
                statement=(
                    "Seed-stable nonlinear kinetic sensitivity selected "
                    f"{selected.get('random_feature_count')} random features at "
                    f"bandwidth scale {selected.get('bandwidth_scale')}, with "
                    f"mean held-out VAMP-E {score:.4g} across the prespecified seeds."
                ),
                report_path=path, effect_value=score,
                family=f"{module_id}:stable_candidate",
            ))
        elif report.get("selection_status") == "no_stable_candidate":
            findings.append(_candidate(
                module_id=module_id, category="other_physical",
                statement=(
                    "Random-feature nonlinear kinetics withheld model selection: "
                    "no feature-count/bandwidth candidate passed both prespecified "
                    "feature-map-seed stability gates."
                ),
                report_path=path, effect_value=0.0,
                family=f"{module_id}:seed_stability_gate",
            ))
    elif module_id == "helical_mechanics" and report.get("availability_status") == "available":
        couplings = report.get("neighbor_step_couplings")
        if isinstance(couplings, list):
            choices = [
                row for row in couplings if isinstance(row, dict)
                and isinstance(row.get("mutual_information_bits"), (int, float))
            ]
            if choices:
                row = max(choices, key=lambda value: float(value["mutual_information_bits"]))
                value = float(row["mutual_information_bits"])
                system_id = str(row.get("system_id"))
                findings.append(_candidate(
                    module_id=module_id, category="coupled_interaction",
                    statement=(
                        f"Largest descriptive adjacent-step state mutual information in "
                        f"{system_id} is steps {row.get('step_i')}–{row.get('step_j')}: "
                        f"{value:.4g} bits."
                    ), report_path=path, effect_value=value, systems=(system_id,),
                    family=f"{module_id}:adjacent_step_mutual_information",
                ))
    return findings


def _report_candidates(path: Path, report: Mapping[str, object]) -> List[Dict[str, object]]:
    module_id = str(report.get("module_id", path.parent.name))
    findings = _state_differences(report, module_id, path)
    findings.extend(_score_correlations(report, module_id, path))
    if module_id == "pooled_rmsf":
        findings.extend(_rmsf_candidates(report, path))
    elif module_id == "dccm":
        findings.extend(_dccm_candidates(report, path))
    elif module_id == "hydrogen_bond_discovery":
        findings.extend(_hydrogen_bond_candidates(report, path))
    elif module_id in {"nucleic_acid_geometry", "ion_coordination_geometry"}:
        findings.extend(_replica_metric_candidates(report, module_id, path))
    elif module_id == "solvent_accessible_surface_area":
        findings.extend(_sasa_candidates(report, path))
    elif module_id == "optional_observables":
        findings.extend(_observable_candidates(report, path))
    elif module_id == "radial_distribution_functions":
        findings.extend(_rdf_candidates(report, path))
    elif module_id == "dihedral_distributions":
        findings.extend(_dihedral_candidates(report, path))
    elif module_id == "water_mediated_hydrogen_bond_networks":
        findings.extend(_water_network_candidates(report, path))
    elif module_id == "pald_community_analysis":
        findings.extend(_pald_community_candidates(report, path))
    if module_id in DEFAULT_DISABLED_MODULES:
        findings.extend(_experimental_method_candidates(report, module_id, path))
    if module_id == "pca_fes_basins":
        landscape = report.get("landscape")
        basins = landscape.get("basins") if isinstance(landscape, dict) else None
        if isinstance(basins, list):
            for basin in basins:
                if not isinstance(basin, dict) or not isinstance(basin.get("assigned_fraction"), (int, float)):
                    continue
                fraction = float(basin["assigned_fraction"])
                findings.append(_candidate(
                    module_id=module_id, category="free_energy_surface",
                    statement=(
                        f"Pooled PCA-FES basin {basin.get('basin_id')} contains "
                        f"{fraction:.1%} of evaluated observations at smoothing sigma "
                        f"{report.get('primary_smoothing_sigma_bins')} bins."
                    ),
                    report_path=path, effect_value=fraction,
                    family="pca_fes_basins:basin_population",
                ))
    selected = report.get("selected_model")
    if module_id.startswith("clustering_") and isinstance(selected, dict):
        silhouette = selected.get("silhouette")
        if isinstance(silhouette, (int, float)) and not isinstance(silhouette, bool):
            evaluation = selected.get("silhouette_evaluation")
            stability = report.get("silhouette_selection_stability")
            if isinstance(evaluation, dict) and evaluation.get("estimated") is True:
                silhouette_label = "mean sampled silhouette"
                stability_text = (
                    f" Winner stability: {stability.get('status')}."
                    if isinstance(stability, dict) else ""
                )
            elif isinstance(evaluation, dict) and evaluation.get("estimated") is False:
                silhouette_label = "exact silhouette"
                stability_text = ""
            else:
                silhouette_label = "silhouette"
                stability_text = ""
            findings.append(_candidate(
                module_id=module_id, category="clustering",
                statement=(
                    f"Selected {module_id} partition has {selected.get('k', selected.get('cluster_count'))} "
                    f"states and {silhouette_label} {float(silhouette):.3f}."
                    f"{stability_text}"
                ),
                report_path=path, effect_value=float(silhouette),
                family=f"{module_id}:model_selection",
            ))
    if module_id == "markov_state_models":
        for field, category, label in (
            ("best_clustering_state_model", "clustering", "Best clustering MSM"),
            ("fes_state_model", "free_energy_surface", "FES-basin state model"),
        ):
            model = report.get(field)
            if not isinstance(model, dict):
                continue
            score = model.get("geometric_score")
            findings.append(_candidate(
                module_id=module_id,
                category=category,
                statement=(
                    f"{label} uses {model.get('candidate_id')} with "
                    f"{model.get('state_count')} states; kinetic validation is "
                    f"{model.get('kinetic_validation_status')}."
                ),
                report_path=path,
                effect_value=(
                    float(score) if isinstance(score, (int, float))
                    and not isinstance(score, bool) else None
                ),
                family=f"markov_state_models:{field}",
            ))
    if module_id == "state_coordinate_exports":
        findings.append(_candidate(
            module_id=module_id, category="free_energy_conformation",
            statement=(
                f"Exported {report.get('representative_count')} observed representatives and "
                f"{report.get('exported_frame_count')} state-assigned observations."
            ),
            report_path=path,
            effect_value=float(report.get("exported_frame_count", 0)),
            family="state_coordinate_exports",
        ))
    raw_candidates = report.get("finding_candidates")
    if isinstance(raw_candidates, list):
        for raw in raw_candidates:
            if not isinstance(raw, dict) or not isinstance(raw.get("statement"), str):
                continue
            findings.append(_candidate(
                module_id=module_id,
                category=str(raw.get("category", "other_physical")),
                statement=str(raw["statement"]), report_path=path,
                effect_value=float(raw["effect_value"]) if isinstance(raw.get("effect_value"), (int, float)) else None,
                p_value=float(raw["p_value"]) if isinstance(raw.get("p_value"), (int, float)) else None,
                evidence_level=str(raw.get("evidence_level", "descriptive")),
                systems=tuple(str(value) for value in raw.get("system_ids", [])),
                family=str(raw.get("comparison_family", module_id)),
            ))
    return findings


def _benjamini_hochberg(findings: List[Dict[str, object]]) -> None:
    by_family: Dict[str, List[Dict[str, object]]] = {}
    for row in findings:
        if isinstance(row.get("p_value"), (int, float)):
            by_family.setdefault(str(row["comparison_family"]), []).append(row)
    for rows in by_family.values():
        ordered = sorted(rows, key=lambda row: float(row["p_value"]))
        running = 1.0
        for reverse_rank, row in enumerate(reversed(ordered), start=1):
            rank = len(ordered) - reverse_rank + 1
            running = min(running, float(row["p_value"]) * len(ordered) / rank)
            row["adjusted_p_value"] = min(1.0, running)


def prioritize_findings(root: Path, *, maximum_findings: int | None = None) -> Dict[str, object]:
    analysis_root = Path(root).expanduser().resolve(strict=True)
    config_path = analysis_root / "analysis-config.json"
    config = load_json(config_path) if config_path.is_file() else {}
    comparison_config = config.get("comparisons", {}) if isinstance(config, dict) else {}
    reporting_config = config.get("reporting", {}) if isinstance(config, dict) else {}
    if not isinstance(comparison_config, dict):
        comparison_config = {}
    if maximum_findings is None:
        maximum_findings = int(
            reporting_config.get("maximum_findings", 50)
            if isinstance(reporting_config, dict) else 50
        )
    alpha = float(comparison_config.get("alpha", 0.05))
    mode = str(comparison_config.get("mode", "all_pairs"))
    reference = comparison_config.get("reference_system_id")
    findings = []
    complete_records = []
    method_evidence_coverage = []
    report_paths = {
        *(
            (analysis_root / "results").glob("**/report.json")
            if (analysis_root / "results").is_dir() else ()
        ),
        *analysis_root.glob("*-availability.json"),
    }
    for path in sorted(report_paths):
        candidate_count_before = len(findings)
        sidecar_path = Path(str(path) + ".summary.json")
        if sidecar_path.is_file():
            sidecar = load_json(sidecar_path)
            if sidecar.get("technical_status") != "complete":
                raise FindingPickerError(f"analysis sidecar is not complete: {sidecar_path}")
            if sidecar.get("report_path") != str(path.resolve()):
                raise FindingPickerError(f"analysis sidecar report path mismatch: {sidecar_path}")
            if sidecar.get("report_size_bytes") != path.stat().st_size:
                raise FindingPickerError(f"analysis sidecar report size mismatch: {sidecar_path}")
            if sidecar.get("report_sha256") != _sha256_file(path):
                raise FindingPickerError(f"analysis sidecar report hash mismatch: {sidecar_path}")
            evidence = sidecar.get("finding_evidence")
            if not isinstance(evidence, dict) or not isinstance(evidence.get("candidates"), list):
                raise FindingPickerError(f"analysis sidecar lacks finding evidence: {sidecar_path}")
            findings.extend(
                row for row in evidence["candidates"] if isinstance(row, dict)
            )
            compact = evidence.get("cross_report_summary")
            if isinstance(compact, dict):
                complete_records.append((path, compact))
            method_evidence_coverage.append({
                "module_id": str(sidecar.get("module_id", path.parent.name)),
                "technical_status": str(sidecar.get("technical_status", "unknown")),
                "availability_status": str(
                    sidecar.get("availability_status", "available")
                ),
                "candidate_count": len(findings) - candidate_count_before,
                "report_path": str(path),
            })
            continue
        report = load_json(path)
        if report.get("technical_status") == "complete":
            module_id = str(report.get("module_id", path.parent.name))
            if module_id in _CROSS_REPORT_ROWS:
                compact = _compact_cross_report(report, module_id)
                if isinstance(compact, dict):
                    complete_records.append((path, compact))
            findings.extend(_report_candidates(path, report))
        method_evidence_coverage.append({
            "module_id": str(report.get("module_id", path.parent.name)),
            "technical_status": str(report.get("technical_status", "unknown")),
            "availability_status": str(
                report.get("availability_status", "available")
            ),
            "candidate_count": len(findings) - candidate_count_before,
            "report_path": str(path),
        })
    findings.extend(_cross_report_candidates(complete_records))
    if mode == "reference_vs_all" and reference:
        findings = [
            row for row in findings
            if len(row["system_ids"]) < 2 or str(reference) in row["system_ids"]
        ]
    _benjamini_hochberg(findings)
    for row in findings:
        adjusted = row.get("adjusted_p_value")
        row["statistically_significant"] = (
            bool(adjusted <= alpha) if isinstance(adjusted, (int, float)) else None
        )
    findings.sort(key=lambda row: (
        _PRIORITY.get(str(row["category"]), 99),
        0 if row.get("statistically_significant") is True else 1,
        float(row["adjusted_p_value"]) if isinstance(row.get("adjusted_p_value"), (int, float)) else 2.0,
        -float(row["absolute_effect_value"]) if isinstance(row.get("absolute_effect_value"), (int, float)) else 0.0,
        str(row["module_id"]), str(row["statement"]),
    ))
    for index, row in enumerate(findings, start=1):
        row["finding_id"] = f"finding-{index:06d}"
    selected = findings[:maximum_findings]
    output = {
        "finding_schema": "salsbury-prioritized-findings-v1",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "candidate_count": len(findings),
        "reported_count": len(selected),
        "comparison_mode": mode,
        "multiple_testing": "benjamini_hochberg",
        "alpha": alpha,
        "findings": selected,
        "method_evidence_coverage": method_evidence_coverage,
        "ranking_contract": (
            "scientific presentation category, then declared inferential significance, "
            "then adjusted p value, then absolute effect; no opaque composite score"
        ),
        "cross_report_selection_contract": (
            "for modules stored separately by system, select the technically complete "
            "report with the greatest evaluated-frame coverage per system; break exact "
            "coverage ties deterministically by report path and retain all selected paths"
        ),
        "interpretation": (
            "Only findings with adjusted p values are labeled statistically significant. "
            "All other ranked differences and correlations remain descriptive or exploratory."
        ),
    }
    json_path = analysis_root / "prioritized_findings.json"
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = analysis_root / "prioritized_findings.csv"
    fields = [
        "finding_id", "category", "module_id", "evidence_level", "statement",
        "system_ids", "effect_value", "p_value", "adjusted_p_value",
        "statistically_significant", "report_path",
        "report_paths",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in selected:
            writer.writerow({
                **row,
                "system_ids": ";".join(row["system_ids"]),
                "report_paths": ";".join(row["report_paths"]),
            })
    markdown_path = analysis_root / "prioritized_findings.md"
    lines = [
        "# Prioritized findings", "",
        "Technical status is complete; scientific status is not evaluated.", "",
    ]
    for rank, row in enumerate(selected, start=1):
        qualifier = (
            "statistically significant after BH correction"
            if row["statistically_significant"] is True else
            str(row["evidence_level"])
        )
        lines.append(f"{rank}. {row['statement']} ({qualifier}; `{row['module_id']}`)")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        **output,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "markdown_path": str(markdown_path),
    }
