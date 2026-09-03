"""Transparent, deterministic prioritization of completed analysis findings."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .analysis_config import (
    DEFAULT_DISABLED_MODULES,
    HIGHLIGHTED_FINDINGS_TOTAL,
    MAXIMUM_HEADLINE_FINDINGS,
    MINIMUM_HEADLINE_FINDINGS,
)
from .manifests import load_json
from .presentation_artifacts import finding_target, validate_manifest


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


_TECHNICAL_SUPPORT_MODULES = {
    "provenance_manifest",
    "preflight_inventory",
    "common_atom_mapping",
    "coordinate_cache",
}

_QUALITY_CONTROL_MODULES = {
    "structural_integrity_qc",
    "convergence_uncertainty",
}

_INTERPRETIVE_CONTEXT_MODULES = {
    "individual_pca",
    "common_pca",
    "time_lagged_independent_component_analysis",
    "trajectory_features",
    "representative_frames",
    "representative_structures",
    "state_coordinate_exports",
    "integrated_comparison",
    "grouped_ml",
}


_PRESENTATION_CATEGORY_CYCLE = (
    "free_energy_surface",
    "free_energy_surface",
    "free_energy_conformation",
    "clustering",
    "clustering_conformation",
    "coupled_interaction",
    "rmsf",
    "other_physical",
    "coupled_interaction",
    "other_physical",
)


_FAMILY_PRESENTATION_PRIORITY = {
    "pca_fes_basins:state_population": 0,
    "pca_fes_basins:basin_population": 1,
    "clustering_kmeans:model_selection": 10,
    "alternative_clustering:model_selection": 11,
    "clustering_imwkmeans:model_selection": 12,
    "clustering_kmeans:state_population": 13,
    "clustering_imwkmeans:state_population": 14,
    "pooled_rmsf:pairwise_atom_difference": 20,
    "pooled_rmsf:within_system_maximum": 21,
    "dccm:pairwise_extreme_difference": 30,
    "water_mediated_hydrogen_bond_networks:pairwise_occupancy_difference": 31,
    "correlation_networks:difference_from_reference_dccm": 32,
    "correlation_networks:frame_pooled_dccm": 33,
    "generalized_correlation_and_information:generalized_correlation": 34,
    "generalized_correlation_and_information:normalized_mutual_information": 35,
    "information_dynamics:transfer_entropy": 36,
    "information_dynamics:lagged_cross_correlation": 37,
    "hydrogen_bond_discovery:pairwise_occupancy_difference": 40,
    "hydrogen_bond_discovery:chemical_identity_pairwise_difference": 41,
    "ion_coordination_geometry:pairwise_metric_difference": 42,
    "radial_distribution_functions:pairwise_bin_difference": 43,
    "nucleic_acid_geometry:pairwise_metric_difference": 44,
    "solvent_accessible_surface_area:pairwise_residue_difference": 45,
    "dihedral_distributions:pairwise_circular_mean_difference": 46,
    "ion_atmosphere:pairwise_species_maximum_difference": 47,
}


def _module_review_role(module_id: str) -> str:
    if module_id in _TECHNICAL_SUPPORT_MODULES:
        return "technical_support"
    if module_id in _QUALITY_CONTROL_MODULES:
        return "quality_control"
    if module_id in _INTERPRETIVE_CONTEXT_MODULES:
        return "interpretive_context"
    return "scientific_result"


def _candidate(
    *, module_id: str, category: str, statement: str, report_path: Path,
    effect_value: float | None = None, p_value: float | None = None,
    evidence_level: str = "descriptive", systems: Sequence[str] = (),
    family: str = "single_system",
    report_paths: Sequence[Path] = (),
    ranking_role: str = "scientific_finding",
    validation_status: str = "not_applicable",
    presentation_target: Mapping[str, object] | None = None,
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
        "ranking_role": ranking_role,
        "validation_status": validation_status,
        "presentation_eligible": ranking_role == "scientific_finding",
        "presentation_target": (
            dict(presentation_target) if isinstance(presentation_target, Mapping)
            else None
        ),
    }


def _target_context(path: Path, **extra: object) -> Dict[str, object]:
    parts = path.parts
    context: Dict[str, object] = {}
    if "per-system" in parts:
        index = parts.index("per-system")
        if index + 1 < len(parts):
            context.update({
                "system_id": parts[index + 1],
                "analysis_scope": "per_system",
            })
    if "conformational-views" in parts:
        index = parts.index("conformational-views")
        if index + 1 < len(parts):
            context["view_id"] = parts[index + 1]
            context.setdefault("analysis_scope", "pooled_system_comparison")
    context.update(extra)
    return context


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
                        f"State {state_id} frame fraction differs between "
                        f"{left} and {right} by {float(effect):+.4f} ({left} minus {right})."
                    ),
                    report_path=path, effect_value=float(effect), systems=(left, right),
                    family=f"{module_id}:state_population",
                    presentation_target=finding_target(
                        module_id=module_id, purpose="state_populations",
                        context=_target_context(
                            path, highlight_state_id=state_id,
                            highlight_system_ids=[left, right],
                        ),
                    ),
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


def _numeric(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _matrix_extreme(
    matrix: object, *, directed: bool = False
) -> tuple[int, int, float] | None:
    if not isinstance(matrix, list):
        return None
    best: tuple[int, int, float] | None = None
    for row_index, row in enumerate(matrix):
        if not isinstance(row, list):
            continue
        for column_index, raw in enumerate(row):
            if row_index == column_index:
                continue
            if not directed and column_index < row_index:
                continue
            value = _numeric(raw)
            if value is None:
                continue
            if best is None or abs(value) > abs(best[2]):
                best = (row_index, column_index, value)
    return best


def _alternative_clustering_candidates(
    report: Mapping[str, object], path: Path
) -> List[Dict[str, object]]:
    rows = report.get("algorithm_results")
    if not isinstance(rows, list):
        return []
    scored = [
        (float(row["silhouette"]), row)
        for row in rows if isinstance(row, dict)
        and _numeric(row.get("silhouette")) is not None
    ]
    if not scored:
        return []
    silhouette, selected = max(scored, key=lambda item: (item[0], str(item[1].get("algorithm"))))
    cluster_sizes = selected.get("full_cluster_sizes", selected.get("cluster_sizes"))
    cluster_count = len(cluster_sizes) if isinstance(cluster_sizes, list) else None
    return [_candidate(
        module_id="alternative_clustering", category="clustering",
        statement=(
            f"Best alternative-clustering result is {selected.get('algorithm')}"
            f" with silhouette {silhouette:.3f}"
            f"{f' and {cluster_count} clusters' if cluster_count is not None else ''}."
        ),
        report_path=path, effect_value=silhouette,
        evidence_level="geometric validation",
        family="alternative_clustering:model_selection",
        presentation_target=finding_target(
            module_id="alternative_clustering", purpose="model_selection",
            context=_target_context(
                path, highlight_algorithm=selected.get("algorithm")
            ),
        ),
    )]


def _pca_context_candidates(
    report: Mapping[str, object], module_id: str, path: Path
) -> List[Dict[str, object]]:
    component_sets: List[tuple[str, Mapping[str, object]]] = []
    if module_id == "common_pca":
        basis = report.get("basis")
        pca = basis.get("pca") if isinstance(basis, dict) else None
        components = pca.get("components") if isinstance(pca, dict) else None
        if isinstance(components, list) and components and isinstance(components[0], dict):
            component_sets.append(("shared basis", components[0]))
    else:
        systems = report.get("systems")
        if isinstance(systems, list):
            for system in systems:
                if not isinstance(system, dict):
                    continue
                for replica in system.get("replicas", []):
                    if not isinstance(replica, dict):
                        continue
                    pca = replica.get("pca")
                    components = pca.get("components") if isinstance(pca, dict) else None
                    if isinstance(components, list) and components and isinstance(components[0], dict):
                        label = f"{system.get('system_id')}/{replica.get('replica_id')}"
                        component_sets.append((label, components[0]))
    findings = []
    for label, component in component_sets:
        variance = _numeric(component.get("explained_variance_fraction"))
        if variance is None:
            continue
        findings.append(_candidate(
            module_id=module_id, category="other_physical",
            statement=f"PC1 explains {variance:.1%} of coordinate variance for {label}.",
            report_path=path, effect_value=variance,
            evidence_level="interpretive context",
            family=f"{module_id}:variance_accounting",
        ))
    return findings


def _tica_context_candidates(
    report: Mapping[str, object], path: Path
) -> List[Dict[str, object]]:
    components = report.get("components")
    if not isinstance(components, list) or not components or not isinstance(components[0], dict):
        return []
    first = components[0]
    eigenvalue = _numeric(first.get("eigenvalue"))
    timescale = _numeric(first.get("implied_timescale"))
    if eigenvalue is None:
        return []
    suffix = (
        f" and an implied timescale of {timescale:.3g} {first.get('time_unit', report.get('time_unit', ''))}"
        if timescale is not None else ""
    )
    return [_candidate(
        module_id="time_lagged_independent_component_analysis",
        category="other_physical",
        statement=f"Leading tICA component has eigenvalue {eigenvalue:.3f}{suffix}.",
        report_path=path, effect_value=eigenvalue,
        evidence_level="interpretive context",
        family="time_lagged_independent_component_analysis:leading_component",
    )]


def _information_correlation_candidates(
    report: Mapping[str, object], path: Path
) -> List[Dict[str, object]]:
    findings = []
    systems = report.get("systems")
    if not isinstance(systems, list):
        return findings
    for system in systems:
        if not isinstance(system, dict):
            continue
        system_id = str(system.get("system_id"))
        for field, label in (
            ("generalized_correlation", "generalized correlation"),
            ("normalized_mutual_information", "normalized mutual information"),
        ):
            extreme = _matrix_extreme(system.get(field))
            if extreme is None:
                continue
            left, right, value = extreme
            findings.append(_candidate(
                module_id="generalized_correlation_and_information",
                category="coupled_interaction",
                statement=(
                    f"Strongest {label} in {system_id} is feature {left + 1} with "
                    f"feature {right + 1}: {value:.3f}."
                ),
                report_path=path, effect_value=value, systems=(system_id,),
                evidence_level="descriptive nonlinear dependence",
                family=f"generalized_correlation_and_information:{field}",
            ))
    return findings


def _information_dynamics_candidates(
    report: Mapping[str, object], path: Path
) -> List[Dict[str, object]]:
    analyses = report.get("analyses")
    if not isinstance(analyses, dict):
        return []
    findings = []
    for analysis_id, matrix_key, label, directed in (
        ("transfer_entropy", "transfer_entropy_nats", "transfer entropy", True),
        ("lagged_cross_correlation", "lagged_cross_correlation", "lagged cross-correlation", True),
    ):
        analysis = analyses.get(analysis_id)
        matrix = analysis.get(matrix_key) if isinstance(analysis, dict) else None
        extreme = _matrix_extreme(matrix, directed=directed)
        if extreme is None:
            continue
        source, target, value = extreme
        findings.append(_candidate(
            module_id="information_dynamics", category="coupled_interaction",
            statement=(
                f"Largest absolute {label} is feature {source + 1} at t to feature "
                f"{target + 1} at the declared lag: {value:.3g}."
            ),
            report_path=path, effect_value=value,
            evidence_level="exploratory directional dependence",
            family=f"information_dynamics:{analysis_id}",
        ))
    return findings


def _correlation_network_candidates(
    report: Mapping[str, object], path: Path
) -> List[Dict[str, object]]:
    findings = []
    systems = report.get("systems")
    if not isinstance(systems, list):
        return findings
    for system in systems:
        if not isinstance(system, dict):
            continue
        system_id = str(system.get("system_id"))
        matrices = system.get("matrices")
        if not isinstance(matrices, list):
            continue
        for matrix in matrices:
            network = matrix.get("network") if isinstance(matrix, dict) else None
            strengths = network.get("node_absolute_strengths") if isinstance(network, dict) else None
            if not isinstance(strengths, list):
                continue
            finite = [(float(value), index) for index, value in enumerate(strengths) if _numeric(value) is not None]
            if not finite:
                continue
            strength, index = max(finite)
            findings.append(_candidate(
                module_id="correlation_networks", category="coupled_interaction",
                statement=(
                    f"Highest absolute network strength in {system_id}/"
                    f"{matrix.get('matrix_kind')} is node {index}: {strength:.3f}."
                ),
                report_path=path, effect_value=strength, systems=(system_id,),
                evidence_level="descriptive thresholded network",
                family=f"correlation_networks:{matrix.get('matrix_kind')}",
            ))
    return findings


def _grouped_ml_candidates(
    report: Mapping[str, object], path: Path
) -> List[Dict[str, object]]:
    metrics = report.get("pooled_held_out_metrics")
    if not isinstance(metrics, dict):
        return []
    macro_f1 = _numeric(metrics.get("macro_f1"))
    if macro_f1 is None:
        return []
    return [_candidate(
        module_id="grouped_ml", category="other_physical",
        statement=f"Grouped held-out classification has macro-F1 {macro_f1:.3f}.",
        report_path=path, effect_value=macro_f1,
        evidence_level="held-out predictive diagnostic",
        family="grouped_ml:held_out_macro_f1",
    )]


def _ion_atmosphere_candidates(
    report: Mapping[str, object], path: Path
) -> List[Dict[str, object]]:
    rows = report.get("per_ion_inner_shell_persistence")
    if not isinstance(rows, list):
        return []
    by_system_species: Dict[str, Dict[str, List[float]]] = {}
    row_by_system_species: Dict[tuple[str, str], Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        occupancy = _numeric(row.get("inner_shell_occupancy"))
        if occupancy is None:
            continue
        system_id = str(row.get("system_id"))
        species = str(row.get("species"))
        by_system_species.setdefault(system_id, {}).setdefault(species, []).append(
            occupancy
        )
        key = (system_id, species)
        current = row_by_system_species.get(key)
        if current is None or occupancy > float(current["inner_shell_occupancy"]):
            row_by_system_species[key] = row
    if not by_system_species:
        return []
    findings = []
    maxima: Dict[str, Dict[str, float]] = {}
    for system_id, species_rows in sorted(by_system_species.items()):
        maxima[system_id] = {
            species: max(values) for species, values in species_rows.items()
        }
        species, occupancy = max(
            maxima[system_id].items(), key=lambda item: (item[1], item[0])
        )
        row = row_by_system_species[(system_id, species)]
        findings.append(_candidate(
            module_id="ion_atmosphere", category="other_physical",
            statement=(
                f"Highest observed inner-shell occupancy in {system_id} is "
                f"{occupancy:.1%} for {species} ion {row.get('ion_atom_index')}."
            ),
            report_path=path, effect_value=occupancy, systems=(system_id,),
            evidence_level="descriptive ion-shell persistence",
            family="ion_atmosphere:inner_shell_occupancy",
        ))
    for left, right in itertools.combinations(sorted(maxima), 2):
        shared_species = set(maxima[left]).intersection(maxima[right])
        if not shared_species:
            continue
        species = max(
            shared_species,
            key=lambda value: (
                abs(maxima[left][value] - maxima[right][value]), value
            ),
        )
        effect = maxima[left][species] - maxima[right][species]
        findings.append(_candidate(
            module_id="ion_atmosphere", category="other_physical",
            statement=(
                f"Largest descriptive matched-species ion inner-shell difference "
                f"between {left} and {right} is {species}: {effect:+.1%} "
                f"({left} minus {right}, comparing each system's most persistent ion)."
            ),
            report_path=path, effect_value=effect, systems=(left, right),
            evidence_level="descriptive matched-species ion-shell persistence",
            family="ion_atmosphere:pairwise_species_maximum_difference",
        ))
    return findings


def _rmsd_rg_candidates(
    report: Mapping[str, object], path: Path
) -> List[Dict[str, object]]:
    findings = []
    systems = report.get("systems")
    if not isinstance(systems, list):
        return findings
    for system in systems:
        if not isinstance(system, dict):
            continue
        system_id = str(system.get("system_id"))
        for replica in system.get("replicas", []):
            if not isinstance(replica, dict):
                continue
            for segment in replica.get("segments", []):
                summary = segment.get("summary") if isinstance(segment, dict) else None
                if not isinstance(summary, dict):
                    continue
                for metric, label in (
                    ("rmsd_angstrom", "mean RMSD"),
                    ("radius_of_gyration_angstrom", "mean radius of gyration"),
                ):
                    metric_summary = summary.get(metric)
                    mean = _numeric(metric_summary.get("mean")) if isinstance(metric_summary, dict) else None
                    if mean is None:
                        continue
                    findings.append(_candidate(
                        module_id="replica_rmsd_rg", category="other_physical",
                        statement=(
                            f"{label} for {system_id}/{replica.get('replica_id')}/"
                            f"{segment.get('segment_id')} is {mean:.3f} angstrom."
                        ),
                        report_path=path, effect_value=mean, systems=(system_id,),
                        evidence_level="descriptive stability metric",
                        family=f"replica_rmsd_rg:{metric}",
                    ))
    return findings


def _scalar_distribution_candidates(
    report: Mapping[str, object], path: Path
) -> List[Dict[str, object]]:
    findings = []
    rows = report.get("distribution_reports")
    if not isinstance(rows, list):
        return findings
    for row in rows:
        histogram = row.get("histogram") if isinstance(row, dict) else None
        bins = [value for value in histogram or [] if isinstance(value, dict) and _numeric(value.get("fraction")) is not None]
        if not bins:
            continue
        modal = max(bins, key=lambda value: float(value["fraction"]))
        fraction = float(modal["fraction"])
        findings.append(_candidate(
            module_id="scalar_feature_distributions", category="other_physical",
            statement=(
                f"Modal bin for {row.get('feature_id')} is centered at "
                f"{modal.get('center')} with frame fraction {fraction:.1%}."
            ),
            report_path=path, effect_value=fraction,
            evidence_level="descriptive histogram",
            family=f"scalar_feature_distributions:{row.get('distribution_id')}",
        ))
    return findings


def _scalar_threshold_candidates(
    report: Mapping[str, object], path: Path
) -> List[Dict[str, object]]:
    findings = []
    rows = report.get("state_reports")
    if not isinstance(rows, list):
        return findings
    for row in rows:
        comparison = row.get("state_population_comparison") if isinstance(row, dict) else None
        systems = comparison.get("system_populations") if isinstance(comparison, dict) else None
        dictionary = row.get("state_dictionary") if isinstance(row, dict) else None
        labels = {
            value.get("state_id"): value.get("state_label")
            for value in dictionary or [] if isinstance(value, dict)
        }
        for system in systems or []:
            populations = system.get("state_populations") if isinstance(system, dict) else None
            valid = [value for value in populations or [] if isinstance(value, dict) and _numeric(value.get("fraction_of_all_evaluated")) is not None]
            if not valid:
                continue
            dominant = max(valid, key=lambda value: float(value["fraction_of_all_evaluated"]))
            fraction = float(dominant["fraction_of_all_evaluated"])
            system_id = str(system.get("system_id"))
            state_id = dominant.get("state_id")
            findings.append(_candidate(
                module_id="scalar_threshold_states", category="other_physical",
                statement=(
                    f"Dominant {row.get('feature_id')} threshold state in {system_id} is "
                    f"{labels.get(state_id, state_id)} at {fraction:.1%} of evaluated frames."
                ),
                report_path=path, effect_value=fraction, systems=(system_id,),
                evidence_level="descriptive threshold-state population",
                family=f"scalar_threshold_states:{row.get('state_analysis_id')}",
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
                        presentation_target=finding_target(
                            module_id="dccm", purpose="system_matrix",
                            context={
                                "system_id": system_id,
                                "highlight_atom_i": left_index,
                                "highlight_atom_j": right_index,
                            },
                        ),
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
                    f"Largest DCCM difference between {left} and {right} is "
                    f"{_atom_label(left_atom)} with {_atom_label(right_atom)}: "
                    f"{effect:+.3f} ({left} minus {right})."
                ),
                report_path=path, effect_value=effect, systems=(left, right),
                family="dccm:pairwise_extreme_difference",
                presentation_target=finding_target(
                    module_id="dccm", purpose="pairwise_difference",
                    context={
                        "left_system_id": left, "right_system_id": right,
                        "atom_i": pair[0], "atom_j": pair[1],
                    },
                ),
            ))
    return findings


def _hydrogen_bond_candidates(
    report: Mapping[str, object], path: Path
) -> List[Dict[str, object]]:
    system_views = report.get("system_feature_spaces")
    if isinstance(system_views, list):
        findings: List[Dict[str, object]] = []
        for view in system_views:
            if not isinstance(view, dict):
                continue
            scoped = dict(report)
            scoped.pop("system_feature_spaces", None)
            scoped.update(view)
            findings.extend(_hydrogen_bond_candidates(scoped, path))
        comparative = dict(report)
        comparative.pop("system_feature_spaces", None)
        findings.extend(
            candidate for candidate in _hydrogen_bond_candidates(comparative, path)
            if candidate.get("family")
            == "hydrogen_bond_discovery:pairwise_occupancy_difference"
        )
        return findings
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
        donor = candidate.get("donor_identity")
        acceptor = candidate.get("acceptor_identity")
        if not isinstance(donor, dict):
            donor = atom_by_index.get(int(candidate.get("donor_atom_index", -1)), {})
        if not isinstance(acceptor, dict):
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


def _chemical_atom_key(identity: object) -> tuple[str, str, int, str, str] | None:
    if not isinstance(identity, dict):
        return None
    residue_number = identity.get("residue_number")
    if isinstance(residue_number, bool) or not isinstance(residue_number, int):
        return None
    return (
        str(identity.get("chain_id", "")),
        str(identity.get("residue_name", "")),
        residue_number,
        str(identity.get("insertion_code", "")),
        str(identity.get("atom_name", "")),
    )


def _chemical_atom_key_label(key: tuple[str, str, int, str, str]) -> str:
    chain, residue_name, residue_number, insertion_code, atom_name = key
    insertion = insertion_code if insertion_code else ""
    return f"{chain}:{residue_name}{residue_number}{insertion}:{atom_name}"


def _hydrogen_bond_chemical_summary(
    report: Mapping[str, object], system_id: str
) -> tuple[
    set[tuple[str, str, int, str, str]],
    Dict[tuple[tuple[str, str, int, str, str], tuple[str, str, int, str, str]], float],
]:
    views = report.get("system_feature_spaces")
    if isinstance(views, list):
        selected = [
            row for row in views
            if isinstance(row, dict) and row.get("system_id") == system_id
        ]
        if len(selected) != 1:
            return set(), {}
        scoped = dict(report)
        scoped.pop("system_feature_spaces", None)
        scoped.update(selected[0])
        report = scoped
    atoms = report.get("atom_dictionary")
    candidates = report.get("candidate_dictionary")
    occupancies = report.get("occupancies")
    if not all(isinstance(value, list) for value in (atoms, candidates, occupancies)):
        return set(), {}
    atom_keys = {
        int(row["atom_index"]): _chemical_atom_key(row.get("identity"))
        for row in atoms
        if isinstance(row, dict) and isinstance(row.get("atom_index"), int)
    }
    present_atoms = {value for value in atom_keys.values() if value is not None}
    endpoints_by_bond = {}
    for row in candidates:
        if not isinstance(row, dict) or not isinstance(row.get("bond_id"), str):
            continue
        donor = atom_keys.get(row.get("donor_atom_index"))
        acceptor = atom_keys.get(row.get("acceptor_atom_index"))
        if donor is not None and acceptor is not None:
            endpoints_by_bond[str(row["bond_id"])] = (donor, acceptor)
    values: Dict[
        tuple[tuple[str, str, int, str, str], tuple[str, str, int, str, str]],
        List[float],
    ] = {}
    for row in occupancies:
        if not isinstance(row, dict) or str(row.get("system_id")) != system_id:
            continue
        key = endpoints_by_bond.get(str(row.get("bond_id")))
        if key is None:
            continue
        occupancy = _numeric(row.get("occupancy_fraction"))
        if occupancy is None:
            evaluated = row.get("evaluated_frame_count")
            present = row.get("present_frame_count")
            if (
                isinstance(evaluated, int) and not isinstance(evaluated, bool)
                and evaluated > 0 and isinstance(present, int)
                and not isinstance(present, bool)
            ):
                occupancy = present / evaluated
        if occupancy is not None:
            values.setdefault(key, []).append(occupancy)
    # Equivalent donor hydrogens map to one donor-heavy/acceptor-heavy event.
    # The maximum occupancy is a conservative bounded summary when only compact
    # per-hydrogen occupancy evidence, rather than the frame matrix, is present.
    return present_atoms, {key: max(rows) for key, rows in values.items()}


def _cross_report_hydrogen_bond_candidates(
    selected: Mapping[str, tuple[int, str, Path, Mapping[str, object]]]
) -> List[Dict[str, object]]:
    summaries = {
        system_id: _hydrogen_bond_chemical_summary(value[3], system_id)
        for system_id, value in selected.items()
    }
    source_paths = sorted({value[2] for value in selected.values()})
    if not source_paths:
        return []
    findings = []
    for left, right in itertools.combinations(sorted(summaries), 2):
        left_atoms, left_values = summaries[left]
        right_atoms, right_values = summaries[right]
        eligible = {
            key for key in set(left_values).union(right_values)
            if key[0] in left_atoms and key[1] in left_atoms
            and key[0] in right_atoms and key[1] in right_atoms
        }
        if not eligible:
            continue
        key = max(
            eligible,
            key=lambda value: (
                abs(left_values.get(value, 0.0) - right_values.get(value, 0.0)),
                value,
            ),
        )
        effect = left_values.get(key, 0.0) - right_values.get(key, 0.0)
        candidate = _candidate(
            module_id="hydrogen_bond_discovery", category="other_physical",
            statement=(
                "Largest descriptive chemistry-matched direct-hydrogen-bond "
                f"occupancy difference between {left} and {right} is "
                f"{_chemical_atom_key_label(key[0])} to "
                f"{_chemical_atom_key_label(key[1])}: {effect:+.1%} "
                f"({left} minus {right})."
            ),
            report_path=source_paths[0], effect_value=effect,
            systems=(left, right),
            family="hydrogen_bond_discovery:chemical_identity_pairwise_difference",
        )
        candidate["report_paths"] = [str(value) for value in source_paths]
        findings.append(candidate)
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
    "ion_atmosphere": "per_ion_inner_shell_persistence",
    "hydrogen_bond_discovery": "occupancies",
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
        elif module_id == "ion_atmosphere":
            kept = {
                key: row.get(key) for key in (
                    "system_id", "replica_id", "species", "ion_atom_index",
                    "evaluated_frame_count", "inner_shell_occupancy",
                )
            }
        elif module_id == "hydrogen_bond_discovery":
            kept = {
                key: row.get(key) for key in (
                    "system_id", "replica_id", "bond_id",
                    "evaluated_frame_count", "present_frame_count",
                    "occupancy_fraction",
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
    if module_id == "hydrogen_bond_discovery":
        for key, fields in (
            ("candidate_dictionary", (
                "bond_id", "donor_atom_index", "hydrogen_atom_index",
                "acceptor_atom_index",
            )),
            ("atom_dictionary", ("atom_index", "identity")),
        ):
            source = report.get(key)
            if isinstance(source, list):
                compact[key] = [
                    {field: row.get(field) for field in fields}
                    for row in source if isinstance(row, dict)
                ]
    return compact


def finding_sidecar_evidence(
    report: Mapping[str, object], report_path: Path
) -> Dict[str, object]:
    """Create compact finding evidence while the full report is already in memory."""

    path = Path(report_path).expanduser().resolve(strict=False)
    module_id = str(report.get("module_id", path.parent.name))
    candidates = _report_candidates(path, report)
    quality_control = _quality_control_records(report, path)
    return {
        "finding_evidence_schema": "salsbury-finding-evidence-v1",
        "module_id": module_id,
        "report_path": str(path),
        "candidates": candidates,
        "cross_report_summary": _compact_cross_report(report, module_id),
        "module_review": _module_review_record(
            module_id, path, len(candidates), len(quality_control)
        ),
        "quality_control_records": quality_control,
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
        if module_id == "hydrogen_bond_discovery":
            findings.extend(_cross_report_hydrogen_bond_candidates(selected))
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
        elif module_id == "ion_atmosphere":
            candidates = _ion_atmosphere_candidates(merged, primary)
        else:
            candidates = _water_network_candidates(merged, primary)
        for candidate in candidates:
            if len(candidate["system_ids"]) >= 2:
                candidate["report_paths"] = [str(value) for value in source_paths]
                findings.append(candidate)
    return findings


def _integrated_comparison_candidates(
    report: Mapping[str, object], path: Path
) -> List[Dict[str, object]]:
    raw = report.get("comparison_findings")
    if not isinstance(raw, list):
        raise FindingPickerError(
            "integrated comparison report lacks comparison_findings"
        )
    findings = []
    for index, value in enumerate(raw):
        if not isinstance(value, dict):
            raise FindingPickerError(
                f"integrated comparison finding {index} is not an object"
            )
        systems = value.get("system_ids")
        if not isinstance(systems, list) or len(set(map(str, systems))) < 2:
            raise FindingPickerError(
                f"integrated comparison finding {index} lacks two systems"
            )
        required = {
            "module_id", "category", "evidence_level", "statement",
            "comparison_family", "effect_value", "p_value",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise FindingPickerError(
                f"integrated comparison finding {index} is missing: "
                + ", ".join(missing)
            )
        row = dict(value)
        row.pop("finding_id", None)
        row["adjusted_p_value"] = None
        row["statistically_significant"] = None
        source_paths = row.get("report_paths")
        if not isinstance(source_paths, list):
            source_paths = [row.get("report_path")]
        row["source_report_paths"] = [
            str(value) for value in source_paths if value
        ]
        row["integration_report_path"] = str(path)
        row["report_path"] = str(path)
        row["report_paths"] = list(dict.fromkeys([
            str(path), *row["source_report_paths"],
        ]))
        findings.append(row)
    return findings


def _experimental_method_candidates(
    report: Mapping[str, object], module_id: str, path: Path
) -> List[Dict[str, object]]:
    """Extract bounded, method-aware highlights from default-off methods."""

    findings: List[Dict[str, object]] = []
    systems = report.get("systems")
    if module_id == "perturbation_response_dynamics" and isinstance(systems, list):
        for system in systems:
            if not isinstance(system, dict):
                continue
            values = system.get("dci") or system.get("dfi")
            metric = "DCI" if system.get("dci") else "DFI"
            numeric = [
                (index, value) for index, raw in enumerate(values or [])
                if (value := _numeric(raw)) is not None
            ] if isinstance(values, list) else []
            if numeric:
                index, value = max(numeric, key=lambda row: (row[1], -row[0]))
                system_id = str(system.get("system_id"))
                findings.append(_candidate(
                    module_id=module_id, category="coupled_interaction",
                    statement=(
                        f"Largest descriptive {metric} value in {system_id} is "
                        f"node {index}: {value:.4g}."
                    ),
                    report_path=path, effect_value=value, systems=(system_id,),
                    family=f"{module_id}:within_system_{metric.lower()}_maximum",
                ))
    elif module_id == "trajectory_reweighting" and isinstance(systems, list):
        for system in systems:
            diagnostics = system.get("diagnostics") if isinstance(system, dict) else None
            if not isinstance(diagnostics, dict):
                continue
            ratio = _numeric(diagnostics.get("kish_effective_sample_size_ratio"))
            if ratio is None:
                ratio = _numeric(diagnostics.get("kish_ratio"))
            if ratio is not None:
                system_id = str(system.get("system_id"))
                status = str(diagnostics.get(
                    "reweighting_validity_status", "not evaluated"
                ))
                findings.append(_candidate(
                    module_id=module_id, category="other_physical",
                    statement=(
                        f"Frame-weight concentration gate for {system_id} is "
                        f"{status}; the Kish effective-sample ratio is {ratio:.1%}."
                    ),
                    report_path=path, effect_value=ratio, systems=(system_id,),
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
                (index, value) for index, raw in enumerate(values or [])
                if (value := _numeric(raw)) is not None
            ] if isinstance(values, list) else []
            if numeric:
                index, value = max(numeric, key=lambda row: (row[1], -row[0]))
                label = str(index)
                if (
                    isinstance(nodes, list) and index < len(nodes)
                    and isinstance(nodes[index], dict)
                ):
                    label = str(nodes[index].get("node_id", index))
                system_id = str(system.get("system_id"))
                findings.append(_candidate(
                    module_id=module_id, category="coupled_interaction",
                    statement=(
                        f"Highest descriptive pathway {metric} in {system_id} is "
                        f"{label}: {value:.4g}."
                    ),
                    report_path=path, effect_value=value, systems=(system_id,),
                    family=f"{module_id}:within_system_node_priority",
                ))
    elif module_id == "energetic_network_embeddings":
        comparisons = report.get("pairwise_system_comparisons")
        if isinstance(comparisons, list):
            for comparison in comparisons:
                rows = comparison.get("residue_distances") if isinstance(comparison, dict) else None
                choices = [
                    row for row in rows or [] if isinstance(row, dict)
                    and _numeric(row.get("summed_wasserstein_distance")) is not None
                ] if isinstance(rows, list) else []
                if not choices:
                    continue
                row = max(
                    choices,
                    key=lambda value: float(value["summed_wasserstein_distance"]),
                )
                effect = float(row["summed_wasserstein_distance"])
                pair = (
                    str(comparison.get("system_i")),
                    str(comparison.get("system_j")),
                )
                findings.append(_candidate(
                    module_id=module_id, category="coupled_interaction",
                    statement=(
                        f"Largest descriptive energetic-network shift between "
                        f"{pair[0]} and {pair[1]} is {row.get('node_id')}: "
                        f"summed marginal Wasserstein distance {effect:.4g}."
                    ),
                    report_path=path, effect_value=effect, systems=pair,
                    family=f"{module_id}:residue_wasserstein",
                ))
    elif module_id == "multivalent_molecular_bridges":
        rows = report.get("mediator_type_summaries")
        if isinstance(rows, list):
            system_ids = sorted({
                str(row.get("system_id")) for row in rows if isinstance(row, dict)
            })
            for system_id in system_ids:
                choices = [
                    row for row in rows if isinstance(row, dict)
                    and str(row.get("system_id")) == system_id
                    and _numeric(row.get("bridge_occupancy")) is not None
                ]
                if choices:
                    row = max(choices, key=lambda value: float(value["bridge_occupancy"]))
                    effect = float(row["bridge_occupancy"])
                    findings.append(_candidate(
                        module_id=module_id, category="other_physical",
                        statement=(
                            f"Most occupied multivalent mediator type in {system_id} "
                            f"is {row.get('mediator_type')} at {effect:.1%}."
                        ),
                        report_path=path, effect_value=effect, systems=(system_id,),
                        family=f"{module_id}:mediator_occupancy",
                    ))
    elif module_id == "interaction_fingerprints":
        rows = report.get("feature_occupancies")
        choices = [
            row for row in rows or [] if isinstance(row, dict)
            and _numeric(row.get("occupancy_fraction")) is not None
        ] if isinstance(rows, list) else []
        if choices:
            row = max(choices, key=lambda value: float(value["occupancy_fraction"]))
            effect = float(row["occupancy_fraction"])
            findings.append(_candidate(
                module_id=module_id, category="coupled_interaction",
                statement=(
                    f"Most occupied typed interaction is {row.get('feature_id')} "
                    f"at {effect:.1%} of source-observed frames."
                ),
                report_path=path, effect_value=effect,
                family=f"{module_id}:feature_occupancy",
            ))
    elif module_id == "spatial_interaction_ensembles":
        rows = report.get("pairwise_system_spatial_differences")
        choices = [
            row for row in rows or [] if isinstance(row, dict)
            and _numeric(row.get("centroid_displacement_angstrom")) is not None
        ] if isinstance(rows, list) else []
        if choices:
            row = max(
                choices,
                key=lambda value: float(value["centroid_displacement_angstrom"]),
            )
            effect = float(row["centroid_displacement_angstrom"])
            pair = (str(row.get("system_i")), str(row.get("system_j")))
            findings.append(_candidate(
                module_id=module_id, category="coupled_interaction",
                statement=(
                    f"Largest gated interaction-cloud centroid shift is "
                    f"{row.get('superfeature_id')} between {pair[0]} and {pair[1]}: "
                    f"{effect:.4g} Å."
                ),
                report_path=path, effect_value=effect, systems=pair,
                family=f"{module_id}:centroid_displacement",
            ))
    elif module_id == "interaction_persistence":
        rows = report.get("feature_persistence_summaries")
        choices = [
            row for row in rows or [] if isinstance(row, dict)
            and row.get("gap_tolerance_observations") == 0
            and row.get("persistence_summary_gate") == "passed"
            and isinstance(row.get("complete_event_duration_summary"), dict)
            and _numeric(row["complete_event_duration_summary"].get("median"))
            is not None
        ] if isinstance(rows, list) else []
        if choices:
            row = max(
                choices,
                key=lambda value: float(
                    value["complete_event_duration_summary"]["median"]
                ),
            )
            effect = float(row["complete_event_duration_summary"]["median"])
            system_id = str(row.get("system_id"))
            findings.append(_candidate(
                module_id=module_id, category="coupled_interaction",
                statement=(
                    f"Longest gated zero-gap interaction persistence is "
                    f"{row.get('feature_id')} in {system_id}: median complete-event "
                    f"duration {effect:.4g} {row.get('time_unit')}."
                ),
                report_path=path, effect_value=effect, systems=(system_id,),
                family=f"{module_id}:complete_event_duration",
            ))
    elif module_id == "helical_mechanics":
        rows = report.get("neighbor_step_couplings")
        choices = [
            row for row in rows or [] if isinstance(row, dict)
            and _numeric(row.get("mutual_information_bits")) is not None
        ] if isinstance(rows, list) else []
        if choices:
            row = max(choices, key=lambda value: float(value["mutual_information_bits"]))
            effect = float(row["mutual_information_bits"])
            system_id = str(row.get("system_id"))
            findings.append(_candidate(
                module_id=module_id, category="coupled_interaction",
                statement=(
                    f"Largest adjacent-step mutual information in {system_id} is "
                    f"steps {row.get('step_i')}–{row.get('step_j')}: {effect:.4g} bits."
                ),
                report_path=path, effect_value=effect, systems=(system_id,),
                family=f"{module_id}:adjacent_step_mutual_information",
            ))
    elif module_id == "random_feature_koopman":
        selected = report.get("selected_hyperparameters")
        score = _numeric(selected.get("selection_score")) if isinstance(selected, dict) else None
        if score is not None:
            findings.append(_candidate(
                module_id=module_id, category="other_physical",
                statement=(
                    f"Seed-stable nonlinear kinetics selected "
                    f"{selected.get('random_feature_count')} random features at "
                    f"bandwidth scale {selected.get('bandwidth_scale')}; mean held-out "
                    f"VAMP-E is {score:.4g}."
                ),
                report_path=path, effect_value=score,
                family=f"{module_id}:stable_candidate",
            ))
    elif module_id == "reactive_path_ensembles":
        count = report.get("complete_path_count")
        if isinstance(count, int) and not isinstance(count, bool):
            findings.append(_candidate(
                module_id=module_id, category="other_physical",
                statement=(
                    f"Reactive-path extraction found {count} complete paths; "
                    f"transition sufficiency is "
                    f"{report.get('transition_sufficiency_status', 'not evaluated')}."
                ),
                report_path=path, effect_value=float(count),
                family=f"{module_id}:path_sufficiency",
            ))
    return findings


def _report_candidates(path: Path, report: Mapping[str, object]) -> List[Dict[str, object]]:
    module_id = str(report.get("module_id", path.parent.name))
    findings = _state_differences(report, module_id, path)
    findings.extend(_score_correlations(report, module_id, path))
    if module_id == "integrated_comparison":
        findings.extend(_integrated_comparison_candidates(report, path))
    elif module_id == "pooled_rmsf":
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
    elif module_id == "alternative_clustering":
        findings.extend(_alternative_clustering_candidates(report, path))
    elif module_id in {"individual_pca", "common_pca"}:
        findings.extend(_pca_context_candidates(report, module_id, path))
    elif module_id == "time_lagged_independent_component_analysis":
        findings.extend(_tica_context_candidates(report, path))
    elif module_id == "generalized_correlation_and_information":
        findings.extend(_information_correlation_candidates(report, path))
    elif module_id == "information_dynamics":
        findings.extend(_information_dynamics_candidates(report, path))
    elif module_id == "correlation_networks":
        findings.extend(_correlation_network_candidates(report, path))
    elif module_id == "grouped_ml":
        findings.extend(_grouped_ml_candidates(report, path))
    elif module_id == "ion_atmosphere":
        findings.extend(_ion_atmosphere_candidates(report, path))
    elif module_id == "replica_rmsd_rg":
        findings.extend(_rmsd_rg_candidates(report, path))
    elif module_id == "scalar_feature_distributions":
        findings.extend(_scalar_distribution_candidates(report, path))
    elif module_id == "scalar_threshold_states":
        findings.extend(_scalar_threshold_candidates(report, path))
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
                    presentation_target=finding_target(
                        module_id=module_id, purpose="primary_fes",
                        context=_target_context(
                            path,
                            smoothing_sigma_bins=report.get("primary_smoothing_sigma_bins"),
                            highlight_basin_id=basin.get("basin_id"),
                        ),
                    ),
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
                presentation_target=finding_target(
                    module_id=module_id, purpose="model_selection",
                    context=_target_context(path),
                ),
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


def _quality_control_records(
    report: Mapping[str, object], path: Path
) -> List[Dict[str, object]]:
    module_id = str(report.get("module_id", path.parent.name))
    records: List[Dict[str, object]] = []
    if module_id == "structural_integrity_qc":
        status = str(report.get("qc_status", "not reported"))
        records.append({
            "module_id": module_id,
            "severity": "warning" if status not in {"passed", "no_findings_observed"} else "information",
            "status": status,
            "statement": (
                f"Structural-integrity QC status is {status}; "
                f"{report.get('qc_finding_count', 0)} QC findings were recorded."
            ),
            "report_path": str(path),
        })
    if module_id == "convergence_uncertainty":
        status = str(report.get("population_validity_status", "not reported"))
        records.append({
            "module_id": module_id,
            "severity": "information" if status == "passed" else "warning",
            "status": status,
            "statement": f"Population-validity status is {status}.",
            "report_path": str(path),
        })
    issues = report.get("issues")
    if isinstance(issues, list):
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            severity = str(issue.get("severity", "information")).lower()
            if severity not in {"warning", "error"}:
                continue
            records.append({
                "module_id": module_id,
                "severity": severity,
                "status": str(issue.get("code", "reported_issue")),
                "statement": str(issue.get("message", issue.get("code", "Reported issue"))),
                "report_path": str(path),
            })
    return records


def _module_review_record(
    module_id: str, path: Path, candidate_count: int, quality_control_count: int
) -> Dict[str, object]:
    role = _module_review_role(module_id)
    if candidate_count:
        disposition = "ranked_candidates"
        reason = "The report produced automated candidates for deterministic ranking."
    elif quality_control_count or role == "quality_control":
        disposition = "quality_control"
        reason = "The report is presented in the separate QC and interpretation channel."
    elif role == "technical_support":
        disposition = "technical_support"
        reason = "The report establishes execution or provenance context rather than a scientific result."
    elif role == "interpretive_context":
        disposition = "interpretive_context"
        reason = "The report supplies analysis context or source artifacts and produced no additional ranked result."
    else:
        disposition = "reviewed_no_automatic_highlight"
        reason = (
            "The scientific report was reviewed by the picker, but its completed output "
            "did not satisfy a declared automatic highlight rule; review the linked report."
        )
    return {
        "module_id": module_id,
        "report_path": str(path),
        "review_role": role,
        "candidate_count": candidate_count,
        "quality_control_record_count": quality_control_count,
        "disposition": disposition,
        "reason": reason,
    }


def _aggregate_module_accounting(
    reviews: Sequence[Mapping[str, object]], findings: Sequence[Mapping[str, object]],
    selected: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    by_module: Dict[str, Dict[str, object]] = {}
    for review in reviews:
        module_id = str(review.get("module_id"))
        row = by_module.setdefault(module_id, {
            "module_id": module_id,
            "review_role": review.get("review_role", _module_review_role(module_id)),
            "report_count": 0,
            "report_paths": [],
            "quality_control_record_count": 0,
        })
        row["report_count"] = int(row["report_count"]) + 1
        row["report_paths"].append(str(review.get("report_path")))
        row["quality_control_record_count"] = (
            int(row["quality_control_record_count"])
            + int(review.get("quality_control_record_count", 0))
        )
    candidate_counts: Dict[str, int] = {}
    selected_counts: Dict[str, int] = {}
    for finding in findings:
        module_id = str(finding.get("module_id"))
        candidate_counts[module_id] = candidate_counts.get(module_id, 0) + 1
    for finding in selected:
        module_id = str(finding.get("module_id"))
        selected_counts[module_id] = selected_counts.get(module_id, 0) + 1
    for module_id, row in by_module.items():
        row["candidate_count"] = candidate_counts.get(module_id, 0)
        row["reported_finding_count"] = selected_counts.get(module_id, 0)
        role = str(row["review_role"])
        if row["candidate_count"]:
            row["disposition"] = "ranked_candidates"
            row["reason"] = "The module produced automated candidates for deterministic ranking."
        elif int(row["quality_control_record_count"]) or role == "quality_control":
            row["disposition"] = "quality_control"
            row["reason"] = "The module is represented in the separate QC and interpretation channel."
        elif role == "technical_support":
            row["disposition"] = "technical_support"
            row["reason"] = "The module establishes execution or provenance context rather than a scientific result."
        elif role == "interpretive_context":
            row["disposition"] = "interpretive_context"
            row["reason"] = "The module supplies analysis context or source artifacts without an additional ranked result."
        else:
            row["disposition"] = "reviewed_no_automatic_highlight"
            row["reason"] = (
                "The completed scientific report produced no candidate under a declared automatic "
                "highlight rule; the linked raw report remains available for review."
            )
        row["report_paths"] = sorted(set(row["report_paths"]))
    return [by_module[module_id] for module_id in sorted(by_module)]


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


def _path_context(row: Mapping[str, object]) -> tuple[str | None, str | None]:
    raw_paths = [row.get("report_path")]
    for key in ("source_report_paths", "report_paths"):
        values = row.get(key)
        if isinstance(values, list):
            raw_paths.extend(values)
    system_id = None
    view_id = None
    for raw in raw_paths:
        if not isinstance(raw, str):
            continue
        parts = Path(raw).parts
        if system_id is None and "per-system" in parts:
            index = parts.index("per-system")
            if index + 1 < len(parts):
                system_id = parts[index + 1]
        if view_id is None and "conformational-views" in parts:
            index = parts.index("conformational-views")
            if index + 1 < len(parts):
                view_id = parts[index + 1]
    return system_id, view_id


def _normalize_candidate(row: Mapping[str, object]) -> Dict[str, object]:
    normalized = dict(row)
    module_id = str(normalized.get("module_id", ""))
    statement = str(normalized.get("statement", ""))
    systems = normalized.get("system_ids")
    normalized["system_ids"] = (
        list(dict.fromkeys(str(value) for value in systems if str(value)))
        if isinstance(systems, list) else []
    )
    system_id, view_id = _path_context(normalized)
    if system_id and not normalized["system_ids"]:
        normalized["system_ids"] = [system_id]
    normalized["view_ids"] = [view_id] if view_id else []
    context = [*normalized["system_ids"]]
    if view_id:
        context.append(view_id)
    normalized["context_label"] = "/".join(context) if context else None
    statement = re.sub(r"\bdescriptively\s+", "", statement, flags=re.IGNORECASE)
    statement = re.sub(r"\bdescriptive\s+", "", statement, flags=re.IGNORECASE)
    normalized["statement"] = statement
    role = normalized.get("ranking_role")
    if not isinstance(role, str) or not role:
        role = "scientific_finding"
    normalized["ranking_role"] = role
    normalized.setdefault("validation_status", "not_applicable")
    normalized["presentation_eligible"] = role == "scientific_finding"
    if not isinstance(normalized.get("presentation_target"), dict):
        target_context: Dict[str, object] = {}
        if normalized["system_ids"]:
            target_context["system_ids"] = sorted(normalized["system_ids"])
        if view_id:
            target_context["view_id"] = view_id
        normalized["presentation_target"] = finding_target(
            module_id=module_id,
            purpose=(
                "pairwise_comparison"
                if len(normalized["system_ids"]) >= 2 else "summary"
            ),
            context=target_context,
        )
    return normalized


def _deduplicate_candidates(
    rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    unique: Dict[tuple[object, ...], Dict[str, object]] = {}
    for raw in rows:
        row = dict(raw)
        signature = (
            str(row.get("module_id", "")),
            str(row.get("comparison_family", "")),
            tuple(sorted(map(str, row.get("system_ids", [])))),
            str(row.get("statement", "")),
        )
        current = unique.get(signature)
        if current is None:
            unique[signature] = row
            continue
        paths = []
        for source in (current, row):
            values = source.get("report_paths")
            if isinstance(values, list):
                paths.extend(map(str, values))
            elif source.get("report_path"):
                paths.append(str(source["report_path"]))
        current["report_paths"] = list(dict.fromkeys(paths))
    return list(unique.values())


def _presentation_context_matches(
    target: Mapping[str, object], artifact: Mapping[str, object]
) -> bool:
    artifact_context = artifact.get("context")
    if not isinstance(artifact_context, dict):
        artifact_context = {}
    for key, value in target.items():
        if str(key).startswith("highlight_") or value is None:
            continue
        candidate = artifact_context.get(key)
        if isinstance(candidate, list):
            requested = value if isinstance(value, list) else [value]
            if not set(map(str, requested)).issubset(set(map(str, candidate))):
                return False
        elif candidate != value:
            return False
    return True


def _attach_presentation_artifacts(
    analysis_root: Path, findings: Sequence[Dict[str, object]]
) -> None:
    manifest_path = (
        analysis_root / "presentation-artifacts" / "presentation-manifest.json"
    )
    if not manifest_path.is_file():
        for row in findings:
            row["presentation_artifact_resolution"] = "manifest_not_available"
            row["presentation_artifacts"] = []
        return
    manifest = load_json(manifest_path)
    validate_manifest(manifest)
    if manifest.get("technical_status") != "complete":
        raise FindingPickerError("presentation artifact manifest is not complete")
    if int(manifest.get("unadapted_report_count", 0) or 0) != 0:
        raise FindingPickerError(
            "presentation artifact manifest leaves completed reports unadapted"
        )
    raw_artifacts = manifest.get("artifacts")
    artifacts = (
        [row for row in raw_artifacts if isinstance(row, dict)]
        if isinstance(raw_artifacts, list) else []
    )
    for row in findings:
        target = row.get("presentation_target")
        matches = []
        if isinstance(target, dict):
            context = (
                target.get("context")
                if isinstance(target.get("context"), dict) else {}
            )
            matches = [
                artifact for artifact in artifacts
                if artifact.get("module_id") == target.get("module_id")
                and artifact.get("purpose") == target.get("purpose")
                and _presentation_context_matches(context, artifact)
            ]
        if not matches:
            report_paths = set(map(str, row.get("report_paths", [])))
            matches = [
                artifact for artifact in artifacts
                if artifact.get("module_id") == row.get("module_id")
                and (
                    not report_paths
                    or not set(map(str, artifact.get("source_report_paths", []))).isdisjoint(
                        report_paths
                    )
                )
            ]
        matches.sort(key=lambda artifact: (
            0 if artifact.get("primary_human_output") is True else 1,
            str(artifact.get("artifact_type")),
            str(artifact.get("artifact_id")),
        ))
        row["presentation_artifacts"] = [
            {
                "artifact_id": artifact.get("artifact_id"),
                "artifact_type": artifact.get("artifact_type"),
                "analysis_class": artifact.get("analysis_class"),
                "title": artifact.get("title"),
                "relative_path": artifact.get("relative_path"),
            }
            for artifact in matches
        ]
        row["presentation_artifact_resolution"] = (
            "resolved" if matches else "unresolved"
        )


def _within_family_key(row: Mapping[str, object]) -> tuple[object, ...]:
    systems = row.get("system_ids")
    system_count = len(set(map(str, systems))) if isinstance(systems, list) else 0
    adjusted = row.get("adjusted_p_value")
    effect = row.get("absolute_effect_value")
    return (
        0 if row.get("statistically_significant") is True else 1,
        float(adjusted) if isinstance(adjusted, (int, float)) else 2.0,
        0 if system_count >= 2 else 1,
        -float(effect) if isinstance(effect, (int, float)) else 0.0,
        str(row.get("statement", "")),
    )


def _family_priority(family: str) -> tuple[int, str]:
    if family in _FAMILY_PRESENTATION_PRIORITY:
        return _FAMILY_PRESENTATION_PRIORITY[family], family
    if "pairwise" in family or "difference" in family:
        return 60, family
    if "model_selection" in family:
        return 70, family
    if "within_system" in family:
        return 90, family
    return 80, family


def _balanced_scientific_order(
    rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    by_category: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for row in rows:
        category = str(row.get("category", "other_physical"))
        family = str(row.get("comparison_family", "unspecified"))
        by_category.setdefault(category, {}).setdefault(family, []).append(row)
    queues: Dict[str, List[Dict[str, object]]] = {}
    for category, families in by_category.items():
        ordered_families = sorted(families, key=_family_priority)
        for rows_for_family in families.values():
            rows_for_family.sort(key=_within_family_key)
        queue = []
        while any(families[family] for family in ordered_families):
            for family in ordered_families:
                if families[family]:
                    queue.append(families[family].pop(0))
        queues[category] = queue
    positions = {category: 0 for category in queues}
    categories = [
        *dict.fromkeys(_PRESENTATION_CATEGORY_CYCLE),
        *sorted(set(queues).difference(_PRESENTATION_CATEGORY_CYCLE)),
    ]
    ordered = []
    while True:
        advanced = False
        for category in _PRESENTATION_CATEGORY_CYCLE:
            queue = queues.get(category, [])
            position = positions.get(category, 0)
            if position < len(queue):
                ordered.append(queue[position])
                positions[category] = position + 1
                advanced = True
        for category in categories:
            if category in _PRESENTATION_CATEGORY_CYCLE:
                continue
            queue = queues.get(category, [])
            position = positions.get(category, 0)
            if position < len(queue):
                ordered.append(queue[position])
                positions[category] = position + 1
                advanced = True
        if not advanced:
            return ordered


def prioritize_findings(
    root: Path, *, maximum_findings: int | None = None,
    headline_findings: int | None = None,
    write_outputs: bool = True,
) -> Dict[str, object]:
    analysis_root = Path(root).expanduser().resolve(strict=True)
    config_path = analysis_root / "analysis-config.json"
    config = load_json(config_path) if config_path.is_file() else {}
    comparison_config = config.get("comparisons", {}) if isinstance(config, dict) else {}
    reporting_config = config.get("reporting", {}) if isinstance(config, dict) else {}
    if not isinstance(comparison_config, dict):
        comparison_config = {}
    minimum_headline_findings = int(
        reporting_config.get(
            "minimum_headline_findings", MINIMUM_HEADLINE_FINDINGS
        )
        if isinstance(reporting_config, dict)
        else MINIMUM_HEADLINE_FINDINGS
    )
    maximum_override_supplied = maximum_findings is not None
    if maximum_findings is None:
        maximum_findings = int(
            reporting_config.get(
                "maximum_findings", HIGHLIGHTED_FINDINGS_TOTAL
            )
            if isinstance(reporting_config, dict)
            else HIGHLIGHTED_FINDINGS_TOTAL
        )
    headline_override_supplied = headline_findings is not None
    if headline_findings is None:
        headline_findings = min(maximum_findings, int(
            reporting_config.get(
                "headline_findings", MAXIMUM_HEADLINE_FINDINGS
            )
            if isinstance(reporting_config, dict)
            else MAXIMUM_HEADLINE_FINDINGS
        ))
    if maximum_findings < 1:
        raise FindingPickerError("maximum_findings must be positive")
    if headline_findings < 1:
        raise FindingPickerError("headline_findings must be positive")
    if headline_override_supplied and headline_findings > maximum_findings:
        raise FindingPickerError(
            "headline_findings cannot exceed maximum_findings"
        )
    if not maximum_override_supplied and maximum_findings != HIGHLIGHTED_FINDINGS_TOTAL:
        raise FindingPickerError(
            "reporting.maximum_findings must be 50 for the standard "
            "headline/secondary presentation contract"
        )
    if (
        not headline_override_supplied
        and maximum_findings >= MINIMUM_HEADLINE_FINDINGS
        and not MINIMUM_HEADLINE_FINDINGS
        <= headline_findings
        <= MAXIMUM_HEADLINE_FINDINGS
    ):
        raise FindingPickerError(
            "reporting.headline_findings must be an integer from 10 through 12"
        )
    if (
        not headline_override_supplied
        and not maximum_override_supplied
        and (
            minimum_headline_findings < MINIMUM_HEADLINE_FINDINGS
            or minimum_headline_findings > MAXIMUM_HEADLINE_FINDINGS
            or minimum_headline_findings > headline_findings
        )
    ):
        raise FindingPickerError(
            "reporting.minimum_headline_findings must be from 10 through 12 "
            "and cannot exceed reporting.headline_findings"
        )
    alpha = float(comparison_config.get("alpha", 0.05))
    mode = str(comparison_config.get("mode", "all_pairs"))
    reference = comparison_config.get("reference_system_id")
    findings = []
    complete_records = []
    module_reviews = []
    quality_control_records = []
    integrated_path = (
        analysis_root / "results" / "integrated-comparison" / "report.json"
    )
    integrated_present = integrated_path.is_file()
    if integrated_present:
        integrated_report = load_json(integrated_path)
        if integrated_report.get("technical_status") != "complete":
            raise FindingPickerError(
                "integrated comparison exists but is not technically complete"
            )
    for path in sorted((analysis_root / "results").glob("**/report.json")):
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
            report_candidates = [
                row for row in evidence["candidates"] if isinstance(row, dict)
            ]
            if integrated_present:
                report_candidates = [
                    row for row in report_candidates
                    if len(set(map(str, row.get("system_ids", [])))) < 2
                ]
            findings.extend(report_candidates)
            qc_rows = evidence.get("quality_control_records", [])
            qc_rows = [row for row in qc_rows if isinstance(row, dict)] if isinstance(qc_rows, list) else []
            quality_control_records.extend(qc_rows)
            review = evidence.get("module_review")
            if isinstance(review, dict) and not integrated_present:
                module_reviews.append(review)
            else:
                module_reviews.append(_module_review_record(
                    str(sidecar.get("module_id", path.parent.name)), path,
                    len(report_candidates), len(qc_rows),
                ))
            compact = evidence.get("cross_report_summary")
            if isinstance(compact, dict) and not integrated_present:
                complete_records.append((path, compact))
            continue
        report = load_json(path)
        if report.get("technical_status") == "complete":
            module_id = str(report.get("module_id", path.parent.name))
            if module_id in _CROSS_REPORT_ROWS:
                compact = _compact_cross_report(report, module_id)
                if isinstance(compact, dict):
                    complete_records.append((path, compact))
            report_candidates = _report_candidates(path, report)
            if integrated_present and module_id != "integrated_comparison":
                report_candidates = [
                    row for row in report_candidates
                    if len(set(map(str, row.get("system_ids", [])))) < 2
                ]
            qc_rows = _quality_control_records(report, path)
            findings.extend(report_candidates)
            quality_control_records.extend(qc_rows)
            module_reviews.append(_module_review_record(
                module_id, path, len(report_candidates), len(qc_rows)
            ))
    if not integrated_present:
        findings.extend(_cross_report_candidates(complete_records))
    findings = _deduplicate_candidates(
        [_normalize_candidate(row) for row in findings]
    )
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
    _attach_presentation_artifacts(analysis_root, findings)
    presentation_eligible = _balanced_scientific_order([
        row for row in findings if row.get("presentation_eligible") is True
    ])
    supporting_context = sorted(
        (row for row in findings if row.get("presentation_eligible") is not True),
        key=lambda row: (
            str(row.get("ranking_role", "supporting_context")),
            str(row.get("module_id", "")),
            str(row.get("statement", "")),
        ),
    )
    findings = [*presentation_eligible, *supporting_context]
    selected = presentation_eligible[:maximum_findings]
    boundary_promotions = []
    if headline_override_supplied:
        selected_headline_count = min(headline_findings, len(selected))
        headline_selection_reason = (
            "A direct diagnostic override fixed the headline count."
        )
    else:
        selected_headline_count = min(
            minimum_headline_findings, len(selected)
        )
        for rank in range(
            minimum_headline_findings + 1,
            min(headline_findings, len(selected)) + 1,
        ):
            row = selected[rank - 1]
            if row.get("statistically_significant") is True:
                selected_headline_count = rank
                boundary_promotions.append({
                    "rank": rank,
                    "finding_id": f"finding-{rank:06d}",
                    "adjusted_p_value": row.get("adjusted_p_value"),
                    "comparison_family": row.get("comparison_family"),
                })
        if minimum_headline_findings == headline_findings:
            headline_selection_reason = (
                f"The configured presentation range fixes the opening section "
                f"at {minimum_headline_findings} findings."
            )
        else:
            headline_selection_reason = (
                f"The first {minimum_headline_findings} ranked findings are "
                f"always headlines. Ranks {minimum_headline_findings + 1} "
                f"through {headline_findings} extend the opening section only "
                "when a boundary finding is statistically significant after "
                "Benjamini-Hochberg correction."
            )
    for index, row in enumerate(findings, start=1):
        row["finding_id"] = f"finding-{index:06d}"
        row["presentation_tier"] = (
            "headline" if index <= selected_headline_count else
            "secondary" if index <= len(selected) else
            "additional_candidate"
        )
    headlines = selected[:selected_headline_count]
    secondary = selected[len(headlines):]
    standard_contract_requested = (
        maximum_findings == HIGHLIGHTED_FINDINGS_TOTAL
        and not headline_override_supplied
        and MINIMUM_HEADLINE_FINDINGS
        <= headline_findings
        <= MAXIMUM_HEADLINE_FINDINGS
    )
    candidate_limited = len(presentation_eligible) < HIGHLIGHTED_FINDINGS_TOTAL
    presentation_contract_status = (
        "candidate_limited" if standard_contract_requested and candidate_limited
        else "satisfied" if standard_contract_requested
        else "explicit_override"
    )
    module_accounting = _aggregate_module_accounting(
        module_reviews, findings, selected
    )
    output = {
        "finding_schema": "salsbury-prioritized-findings-v2",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "candidate_count": len(findings),
        "presentation_eligible_candidate_count": len(presentation_eligible),
        "supporting_context_candidate_count": len(supporting_context),
        "reported_count": len(selected),
        "headline_count": len(headlines),
        "secondary_count": len(secondary),
        "searchable_candidate_count": len(findings),
        "additional_candidate_count": len(findings) - len(selected),
        "unreported_candidate_count": len(findings) - len(selected),
        "reviewed_report_count": len(module_reviews),
        "reviewed_module_count": len(module_accounting),
        "silent_omission_count": 0,
        "quality_control_record_count": len(quality_control_records),
        "comparison_mode": mode,
        "multiple_testing": "benjamini_hochberg",
        "alpha": alpha,
        "findings": selected,
        "headline_findings": headlines,
        "secondary_findings": secondary,
        "all_candidates": findings,
        "presentation_contract": {
            "contract_id": "headline-secondary-50-v1",
            "headline_count_range": [
                MINIMUM_HEADLINE_FINDINGS, MAXIMUM_HEADLINE_FINDINGS,
            ],
            "highlighted_findings_total": HIGHLIGHTED_FINDINGS_TOTAL,
            "secondary_count_range": [
                HIGHLIGHTED_FINDINGS_TOTAL - MAXIMUM_HEADLINE_FINDINGS,
                HIGHLIGHTED_FINDINGS_TOTAL - MINIMUM_HEADLINE_FINDINGS,
            ],
            "configured_headline_count": headline_findings,
            "configured_minimum_headline_count": minimum_headline_findings,
            "selected_headline_count": selected_headline_count,
            "configured_highlighted_total": maximum_findings,
            "status": presentation_contract_status,
            "candidate_limited": candidate_limited,
            "headline_selection": "bh_significance_at_boundary",
            "boundary_promotions": boundary_promotions,
            "selection_reason": headline_selection_reason,
        },
        "module_accounting": module_accounting,
        "quality_control_records": quality_control_records,
        "ranking_contract": (
            "presentation-eligible findings are interleaved across scientific "
            "categories and method-specific comparison families; inferential "
            "significance and effect magnitude order candidates only within one "
            "method family; no cross-unit composite score is used"
        ),
        "cross_report_selection_contract": (
            "for modules stored separately by system, select the technically complete "
            "report with the greatest evaluated-frame coverage per system; break exact "
            "coverage ties deterministically by report path and retain all selected paths"
        ),
        "interpretation": (
            "Only findings with adjusted p values are labeled statistically significant. "
            "All other ranked differences and correlations remain exploratory. "
            "Headline and secondary tiers control presentation only; every candidate remains "
            "available in the JSON, CSV, and interactive report. Every completed report is accounted "
            "for as a ranked candidate source, quality-control "
            "evidence, interpretive context, technical support, or an explicit no-highlight result."
        ),
    }
    if not write_outputs:
        return output
    json_path = analysis_root / "prioritized_findings.json"
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = analysis_root / "prioritized_findings.csv"
    fields = [
        "finding_id", "category", "module_id", "evidence_level", "statement",
        "system_ids", "view_ids", "context_label", "comparison_family",
        "effect_value", "p_value", "adjusted_p_value",
        "statistically_significant", "report_path",
        "report_paths", "ranking_role", "validation_status",
        "presentation_eligible", "presentation_artifact_resolution",
        "presentation_tier",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in findings:
            writer.writerow({
                **row,
                "system_ids": ";".join(row["system_ids"]),
                "view_ids": ";".join(row.get("view_ids", [])),
                "report_paths": ";".join(row["report_paths"]),
            })
    markdown_path = analysis_root / "prioritized_findings.md"
    lines = [
        "# Prioritized findings", "",
        (
            f"The report presents {len(headlines)} headline findings first and "
            f"{len(secondary)} secondary findings afterward. All {len(findings)} "
            "candidates remain available in the JSON, CSV, and interactive report."
        ), "", "## Headline findings", "",
    ]
    for rank, row in enumerate(headlines, start=1):
        qualifier = (
            "statistically significant after BH correction"
            if row["statistically_significant"] is True else
            str(row["evidence_level"])
        )
        lines.append(f"{rank}. {row['statement']} ({qualifier}; `{row['module_id']}`)")
    lines.extend(["", "## Secondary findings", ""])
    if secondary:
        for rank, row in enumerate(secondary, start=len(headlines) + 1):
            qualifier = (
                "statistically significant after BH correction"
                if row["statistically_significant"] is True else
                str(row["evidence_level"])
            )
            lines.append(
                f"{rank}. {row['statement']} ({qualifier}; `{row['module_id']}`)"
            )
    else:
        lines.append("No secondary findings were selected.")
    lines.extend([
        "", "## Module accounting", "",
        "Every completed report is represented here even when it produced no ranked finding.", "",
        "| Module | Role | Reports | Candidates | Reported | Disposition |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in module_accounting:
        lines.append(
            f"| `{row['module_id']}` | {row['review_role']} | {row['report_count']} | "
            f"{row['candidate_count']} | {row['reported_finding_count']} | "
            f"{row['disposition']} |"
        )
    qc_markdown_path = analysis_root / "prioritized_findings_qc.md"
    qc_lines = [
        "# Quality-control and interpretation records", "",
        "These records are kept separate from scientific finding ranks.", "",
    ]
    if quality_control_records:
        for row in quality_control_records:
            qc_lines.append(
                f"- **{row['severity']}** — {row['statement']} (`{row['module_id']}`)"
            )
    else:
        qc_lines.append("No separate QC or interpretation records were reported.")
    qc_markdown_path.write_text("\n".join(qc_lines) + "\n", encoding="utf-8")
    lines.extend([
        "", "## Quality-control and interpretation records", "",
        f"{len(quality_control_records)} records are retained in "
        "`prioritized_findings_qc.md`, the JSON output, and the interactive report.",
    ])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        **output,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "markdown_path": str(markdown_path),
        "qc_markdown_path": str(qc_markdown_path),
    }
