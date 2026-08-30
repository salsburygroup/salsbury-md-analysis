"""Minimal-input, method-aware frame sampling plans.

The policy in this module is intentionally conservative.  It inspects a system
manifest, assigns each trajectory-consuming method to a documented scaling
tier, and returns an explicit balanced frame-selection contract.  It never
edits a project manifest, launches a calculation, or treats a computational
ceiling as evidence of scientific sufficiency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .frame_sampling import (
    integer_stride_for_budget,
    integer_stride_selected_count,
)
from .hydrogen_bond_discovery import (
    HydrogenBondDiscoveryError,
    _automatic_candidate_intersection,
)
from .manifests import load_json, resolve_manifest_path, validate_system
from .preflight import FileProbeError, probe_topology, probe_trajectory
from .registry import list_modules
from .resource_planning import plan_campaign_resource_budget
from .resource_calibrations import (
    ResourceCalibrationError, load_resource_calibration_catalog,
)
from .scientific_sampling import (
    POLICY_ID,
    assess_raw_sampling,
    profile_from_contract,
    profile_contract,
    required_frames_per_replica,
    scientific_sampling_profile,
)
from .trajectory_contracts import (
    TrajectoryContractError,
    normalize_segment_timing,
)


class AutomaticSamplingError(ValueError):
    """Raised when the system cannot support a deterministic sampling plan."""


REFERENCE_ATOM_COUNT = 85_199
REFERENCE_HYDROGEN_BOND_CANDIDATE_COUNT = 64_640
MINIMUM_HYDROGEN_BOND_WORKLOAD_MULTIPLIER = 0.50
POLICY_ID = "method-time-size-frame-budgets-v5"
DEFAULT_TARGET_WALL_SECONDS = 14_400.0
DEFAULT_TIME_SAFETY_FACTOR = 1.5
DEFAULT_PCA_MAXIMUM_SAMPLE_MATRIX_ELEMENTS = 25_000_000
DEFAULT_PCA_MAXIMUM_BASIS_FRAMES_PER_REPLICA = 500
DEFAULT_PCA_DENSE_MAXIMUM_FEATURES = 1_500
DEFAULT_PCA_FULL_SAMPLE_SUBSPACE_MAXIMUM_FRAMES = 128
SIMULATION_KINDS = {
    "unbiased_md",
    "biased_or_enhanced_md",
    "weighted_ensemble",
    "ai_ensemble",
}


@dataclass(frozen=True)
class SamplingProfile:
    module_id: str
    tier: str
    maximum_total_frames: int
    minimum_frames_per_replica: int
    atom_scaling_exponent: float
    budget_scope: str = "pooled_total"
    frame_contract: str = "direct_trajectory"
    inherited_from: Optional[str] = None
    rationale: str = ""


@dataclass(frozen=True)
class RuntimeCalibration:
    """Public-safe operational timing model for one direct estimator.

    The rates are Apollo/TREX starting points.  They are deliberately retained
    separately from scientific frame ceilings: a completed timing benchmark
    says how long work took, not how many frames are scientifically adequate.
    """

    module_id: str
    seconds_per_frame: float
    fixed_overhead_seconds: float
    calibration_id: str
    evidence_level: str
    rationale: str


def plan_cartesian_pca_basis(
    feature_count: int,
    source_frames_per_replica: Sequence[int],
    *,
    component_count: int = 10,
    maximum_sample_matrix_elements: int = DEFAULT_PCA_MAXIMUM_SAMPLE_MATRIX_ELEMENTS,
    maximum_basis_frames_per_replica: int = DEFAULT_PCA_MAXIMUM_BASIS_FRAMES_PER_REPLICA,
    minimum_basis_frames_per_replica: int = 20,
    dense_maximum_features: int = DEFAULT_PCA_DENSE_MAXIMUM_FEATURES,
    full_sample_subspace_maximum_frames: int = (
        DEFAULT_PCA_FULL_SAMPLE_SUBSPACE_MAXIMUM_FRAMES
    ),
) -> Dict[str, object]:
    """Plan a feature-aware common-PCA basis and all-frame projection.

    This supplements the trajectory/atom timing planner with the dimensions
    that actually govern Cartesian PCA memory.  It never treats a bounded
    basis sample as permission to omit frames from downstream projection.
    """

    integer_fields = {
        "feature_count": feature_count,
        "component_count": component_count,
        "maximum_sample_matrix_elements": maximum_sample_matrix_elements,
        "maximum_basis_frames_per_replica": maximum_basis_frames_per_replica,
        "minimum_basis_frames_per_replica": minimum_basis_frames_per_replica,
        "dense_maximum_features": dense_maximum_features,
        "full_sample_subspace_maximum_frames": full_sample_subspace_maximum_frames,
    }
    for label, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AutomaticSamplingError(f"{label} must be a positive integer")
    counts = tuple(source_frames_per_replica)
    if not counts:
        raise AutomaticSamplingError("source_frames_per_replica must be nonempty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts):
        raise AutomaticSamplingError(
            "source_frames_per_replica must contain positive integers"
        )
    if component_count > min(feature_count, sum(counts) - 1):
        raise AutomaticSamplingError(
            "component_count exceeds the feature or pooled sample-rank bound"
        )

    replica_count = len(counts)
    method = (
        "dense_covariance_v1"
        if feature_count <= dense_maximum_features
        else "randomized_truncated_svd_v1"
    )
    matrix_limited_budget = maximum_basis_frames_per_replica
    if method == "randomized_truncated_svd_v1":
        matrix_limited_budget = min(
            matrix_limited_budget,
            maximum_sample_matrix_elements // (feature_count * replica_count),
        )
    if matrix_limited_budget < min(minimum_basis_frames_per_replica, min(counts)):
        raise AutomaticSamplingError(
            "the PCA sample-matrix envelope cannot retain the required minimum "
            "basis frames from every replica"
        )
    per_replica_budget = min(max(counts), matrix_limited_budget)
    basis_stride = integer_stride_for_budget(
        list(counts), per_replica_budget, error_type=AutomaticSamplingError
    )
    selected_counts = tuple(
        integer_stride_selected_count(count, basis_stride) for count in counts
    )
    selected_total = sum(selected_counts)
    source_total = sum(counts)
    subsampled = selected_total < source_total
    frame_selection: Dict[str, object] = (
        {
            "mode": "integer_stride_per_replica_v1",
            "stride": basis_stride,
        }
        if subsampled else {"mode": "fixed_stride_v1"}
    )
    sample_elements = selected_total * feature_count
    if method == "randomized_truncated_svd_v1" and sample_elements > maximum_sample_matrix_elements:
        raise AutomaticSamplingError("resolved PCA sample matrix exceeds its element gate")
    randomized_oversampling = 12
    randomized_subspace_policy = (
        "fixed_leading_subspace"
        if method == "randomized_truncated_svd_v1" else "not_applicable"
    )
    if (
        method == "randomized_truncated_svd_v1"
        and selected_total <= full_sample_subspace_maximum_frames
    ):
        randomized_oversampling = max(12, selected_total - component_count)
        randomized_subspace_policy = "full_bounded_sample_space"
    randomized_subspace_size = (
        min(component_count + randomized_oversampling, selected_total, feature_count)
        if method == "randomized_truncated_svd_v1" else None
    )
    return {
        "planning_schema": "salsbury-cartesian-pca-resource-plan-v1",
        "solver_method": method,
        "feature_count": feature_count,
        "component_count": component_count,
        "replica_count": replica_count,
        "source_frames_per_replica": list(counts),
        "source_frame_count": source_total,
        "basis_frame_selection": frame_selection,
        "basis_frames_per_replica": list(selected_counts),
        "basis_frame_count": selected_total,
        "basis_coverage_fraction": selected_total / source_total,
        "basis_subsampling_triggered": subsampled,
        "basis_sampling_strategy": (
            f"exact integer stride {basis_stride} over every replica's concatenated "
            "timeline; frame zero retained; no random draw"
            if subsampled else "all source frames; no random draw"
        ),
        "projection_frame_selection": {"mode": "fixed_stride_v1"},
        "projection_frame_count": source_total,
        "projection_policy": "all source frames",
        "maximum_sample_matrix_elements": maximum_sample_matrix_elements,
        "estimated_sample_matrix_elements": sample_elements,
        "estimated_sample_matrix_bytes_float64": sample_elements * 8,
        "estimated_dense_covariance_bytes_float64": feature_count * feature_count * 8,
        "maximum_basis_frames_per_replica": maximum_basis_frames_per_replica,
        "minimum_basis_frames_per_replica": minimum_basis_frames_per_replica,
        "dense_maximum_features": dense_maximum_features,
        "randomized_solver_oversampling": (
            randomized_oversampling
            if method == "randomized_truncated_svd_v1" else None
        ),
        "randomized_solver_subspace_size": randomized_subspace_size,
        "randomized_solver_subspace_policy": randomized_subspace_policy,
        "full_sample_subspace_maximum_frames": full_sample_subspace_maximum_frames,
        "scientific_boundary": (
            "This is a computational resource plan, not evidence that the PCA basis, "
            "trajectory sampling, state populations, or system comparison has converged."
        ),
    }


def _profile(
    module_id: str,
    tier: str,
    ceiling: int,
    minimum: int,
    exponent: float,
    rationale: str,
    budget_scope: str = "pooled_total",
) -> SamplingProfile:
    return SamplingProfile(
        module_id=module_id,
        tier=tier,
        maximum_total_frames=ceiling,
        minimum_frames_per_replica=minimum,
        atom_scaling_exponent=exponent,
        budget_scope=budget_scope,
        rationale=rationale,
    )


# The frame ceilings below are fail-safe upper guards.  The normal v2 decision
# is driven by an explicit wall-time estimate, not by the old 100,000 / 30,000 /
# 10,000 / 1,000 tier alone.  This lets a calibrated expensive method use more
# than 1,000 frames when it fits the time envelope while still bounding
# extrapolation far outside retained evidence.
DIRECT_SAMPLING_PROFILES: Tuple[SamplingProfile, ...] = (
    _profile("structural_integrity_qc", "streaming", 100_000, 50, 1.0,
             "single-pass coordinate and bounded diagnostic work"),
    _profile("replica_rmsd_rg", "streaming", 100_000, 100, 1.0,
             "single-pass fitted scalar observables kept separate by replica",
             budget_scope="per_replica"),
    _profile("pooled_rmsf", "streaming", 100_000, 100, 1.0,
             "streamed coordinate moments"),
    _profile("trajectory_features", "streaming", 100_000, 100, 1.0,
             "streamed declared feature extraction"),
    _profile("dihedral_distributions", "streaming", 100_000, 100, 1.0,
             "bounded local-coordinate calculations"),
    _profile("nucleic_acid_geometry", "streaming", 100_000, 100, 1.0,
             "bounded declared ring and stacking calculations"),
    _profile("ion_coordination_geometry", "streaming", 100_000, 100, 1.0,
             "bounded ion and candidate-ligand calculations"),
    _profile("ion_atmosphere", "moderate", 100_000, 50, 1.25,
             "species-by-target minimum-image ion shell distances"),
    _profile("optional_observables", "streaming", 100_000, 100, 1.0,
             "bounded declared distance/contact calculations"),
    _profile("dccm", "validated_30k", 100_000, 50, 1.25,
             "30,000 pooled frames completed on the 474-selected-atom TREX validation case; the selected-atom matrix remains the controlling size"),
    _profile("individual_pca", "moderate", 100_000, 50, 1.25,
             "coordinate fitting and covariance basis estimation"),
    _profile("common_pca", "validated_30k", 100_000, 50, 1.25,
             "30,000 pooled basis and projection frames completed on the 474-selected-atom TREX validation case"),
    _profile("hydrogen_bonds", "moderate", 100_000, 25, 1.25,
             "declared bond geometry over frames"),
    _profile("hydrogen_bond_discovery", "validated_30k", 100_000, 25, 1.25,
             "30,000 pooled frames completed on the templated TREX protein-DNA system; candidate count remains the controlling size"),
    _profile("radial_distribution_functions", "moderate", 100_000, 50, 1.25,
             "periodic pair-distance histograms"),
    _profile("water_mediated_hydrogen_bond_networks", "expensive", 100_000, 10, 1.5,
             "all-water spatial screening and sparse bridge construction"),
    _profile("solvent_accessible_surface_area", "expensive", 100_000, 10, 1.5,
             "960-point surface sampling and occlusion tests"),
    _profile("secondary_structure", "expensive", 100_000, 10, 1.0,
             "one external DSSP invocation per selected frame"),
    _profile("nucleic_acid_structure", "expensive", 100_000, 10, 1.0,
             "one external DSSR invocation per selected frame"),
)


_BY_MODULE = {profile.module_id: profile for profile in DIRECT_SAMPLING_PROFILES}


# These estimators implement the concatenated-replica frame-selection contract.
# The others retain an exact integer ``frame_stride`` at present.
CONCATENATED_SELECTION_MODULES = {
    "structural_integrity_qc",
    "dccm",
    "individual_pca",
    "common_pca",
    "hydrogen_bond_discovery",
    "radial_distribution_functions",
    "water_mediated_hydrogen_bond_networks",
    "solvent_accessible_surface_area",
    "secondary_structure",
    "nucleic_acid_structure",
}


# One-CPU Apollo/TREX rates at the 85,199-atom reference size.  Four entries
# use retained multi-point pilots; completed 30,000-frame runs anchor most of
# the remaining entries.  Methods without a matched completed benchmark use a
# conspicuously labeled conservative proxy and should be recalibrated from the
# first completed project-local pilot.  A 1.5 safety factor is applied later.
RUNTIME_CALIBRATIONS: Tuple[RuntimeCalibration, ...] = (
    RuntimeCalibration("structural_integrity_qc", 51.47445170275944, 0.0,
                       "apollo-tba-structural-qc-10k-20260829",
                       "completed_10k_runtime_only",
                       "91,519.266 seconds for 10,000 uniformly distributed frames on a 15,148-atom TBA campaign, normalized to the 85,199-atom reference by the structural-QC profile's linear atom scaling; runtime evidence only, not scientific validation"),
    RuntimeCalibration("replica_rmsd_rg", 0.347933, 0.0,
                       "apollo-trex-rmsd-rg-30k-20260812", "completed_30k",
                       "10,437.99 seconds for 30,000 frames"),
    RuntimeCalibration("pooled_rmsf", 0.329444, 0.0,
                       "apollo-trex-rmsf-30k-20260812", "completed_30k",
                       "9,883.31 seconds for 30,000 frames"),
    RuntimeCalibration("trajectory_features", 0.311181, 0.0,
                       "apollo-trex-trajectory-features-30k-20260812", "completed_30k",
                       "9,335.43 seconds for 30,000 frames"),
    RuntimeCalibration("dihedral_distributions", 0.334117, 0.0,
                       "apollo-trex-dihedrals-30k-20260812", "completed_30k",
                       "10,023.52 seconds for 30,000 frames"),
    RuntimeCalibration("nucleic_acid_geometry", 0.107828, 0.0,
                       "apollo-trex-nucleic-geometry-30k-20260812", "completed_30k",
                       "3,234.84 seconds for 30,000 frames"),
    RuntimeCalibration("ion_coordination_geometry", 0.336311, 0.0,
                       "apollo-trex-ion-geometry-30k-20260812", "completed_30k",
                       "10,089.32 seconds for 30,000 frames"),
    RuntimeCalibration("ion_atmosphere", 0.350000, 0.0,
                       "top1-species-atmosphere-provisional-v1", "conservative_proxy",
                       "temporary conservative proxy pending ingestion of the fresh acceptance-matrix measurement"),
    RuntimeCalibration("optional_observables", 0.005758636438877632, 0.0,
                       "apollo-trex-optional-observables-30k-20260813", "completed_30k",
                       "172.759 seconds for all 30,000 frames and 33 declared bounded features"),
    RuntimeCalibration("dccm", 0.30220794027360776, 86.67595853377134,
                       "apollo-trex-dccm474-100-500perrep-20260812", "multi_point_pilot",
                       "474 selected atoms; fixed-overhead plus per-frame fit"),
    RuntimeCalibration("individual_pca", 0.700000, 0.0,
                       "apollo-trex-tier-proxy-individual-pca-v1", "conservative_proxy",
                       "proxy from the completed common-PCA workload"),
    RuntimeCalibration("common_pca", 0.684497, 0.0,
                       "apollo-trex-common-pca-30k-20260812", "completed_30k",
                       "20,534.91 seconds for 30,000 basis and projection frames"),
    RuntimeCalibration("hydrogen_bonds", 0.400000, 0.0,
                       "apollo-trex-tier-proxy-explicit-hbond-v1", "conservative_proxy",
                       "proxy from automatic direct hydrogen-bond discovery"),
    RuntimeCalibration("hydrogen_bond_discovery", 0.0433, 275.7228153187316,
                       "trex-250ns-hbond-597364-candidate-gate-v53", "partial_gate_timing_calibration",
                       "597,364 common candidates; 2.0 billion candidate-frame observations completed before the prior fixed gate, used for timing only with candidate-count scaling and a 0.5 I/O floor"),
    RuntimeCalibration("radial_distribution_functions", 0.316831, 0.0,
                       "apollo-trex-rdf-30k-20260812", "completed_30k",
                       "9,504.94 seconds for 30,000 frames"),
    RuntimeCalibration("water_mediated_hydrogen_bond_networks", 1.3955911282411155,
                       12.203800316201523,
                       "trex-lesion-water-network-apollo-100-500-v1", "multi_point_pilot",
                       "all-water screening; fixed-overhead plus per-frame fit"),
    RuntimeCalibration("solvent_accessible_surface_area", 2.9879936629492376,
                       31.20452450526261,
                       "apollo-trex-sasa960-25-100perrep-20260812", "multi_point_pilot",
                       "3,744 solute-heavy surface atoms and 960 sphere points"),
    RuntimeCalibration("secondary_structure", 4.754613987713431, 0.0,
                       "apollo-trex-mkdssp-300of30k-20260813", "completed_300",
                       "1,426.384 seconds for 300 uniformly distributed frames using mkdssp 4.6.1"),
    RuntimeCalibration("nucleic_acid_structure", 2.000000, 0.0,
                       "external-dssr-tier-proxy-v1", "conservative_proxy",
                       "external executable cost requires a project-local pilot"),
)


_RUNTIME_BY_MODULE = {
    calibration.module_id: calibration for calibration in RUNTIME_CALIBRATIONS
}


_CAMPAIGN_PRIORITY = {
    "common_pca": 10.0,
    "individual_pca": 8.0,
    "pooled_rmsf": 8.0,
    "dccm": 7.0,
    "hydrogen_bond_discovery": 7.0,
    "nucleic_acid_geometry": 7.0,
    "ion_coordination_geometry": 7.0,
    "ion_atmosphere": 7.0,
    "replica_rmsd_rg": 6.0,
    "dihedral_distributions": 6.0,
    "radial_distribution_functions": 6.0,
    "water_mediated_hydrogen_bond_networks": 5.0,
    "solvent_accessible_surface_area": 5.0,
    "secondary_structure": 5.0,
}


def _campaign_direct_resource_plan(
    dimensions: Mapping[str, object],
    module_ids: Sequence[str],
    execution: Mapping[str, object],
    *,
    time_safety_factor: float,
    measured_calibrations: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> Dict[str, object]:
    """Allocate one configured campaign envelope across direct estimators."""

    atom_count = int(dimensions["maximum_atom_count"])
    # The legacy tier values were calibrated on an approximately 85k-atom
    # solvated protein-DNA system.  Applying them unchanged to a few-hundred
    # atom tutorial system produced two-order-of-magnitude overestimates.  A
    # square-root size term keeps substantial headroom for fixed allocations
    # while allowing the estimate to decrease for genuinely smaller systems.
    memory_atom_scale = min(
        4.0, max(0.1, math.sqrt(atom_count / REFERENCE_ATOM_COUNT))
    )
    replica_counts = [
        int(row["source_frame_count"])
        for row in dimensions["replicas"]  # type: ignore[union-attr]
    ]
    system_ids_per_replica = [
        str(row["system_id"])
        for row in dimensions["replicas"]  # type: ignore[union-attr]
    ]
    timing_available = all(
        row.get("maximum_frame_interval_ns") is not None
        and row.get("source_time_span_ns") is not None
        for row in dimensions["replicas"]  # type: ignore[union-attr]
    )
    frame_intervals_ns_per_replica = (
        [
            float(row["maximum_frame_interval_ns"])
            for row in dimensions["replicas"]  # type: ignore[union-attr]
        ]
        if timing_available else None
    )
    source_time_spans_ns_per_replica = (
        [
            float(row["source_time_span_ns"])
            for row in dimensions["replicas"]  # type: ignore[union-attr]
        ]
        if timing_available else None
    )
    replica_count = len(replica_counts)
    tasks = []
    for module_id in module_ids:
        if module_id not in _BY_MODULE:
            continue
        profile = _BY_MODULE[module_id]
        calibration = _RUNTIME_BY_MODULE[module_id]
        measured = (
            measured_calibrations.get(module_id)
            if measured_calibrations is not None else None
        )
        ceiling, _ = _effective_ceiling(profile, atom_count)
        technical_pilot = _technical_pilot_frames_per_replica(
            profile, atom_count
        )
        scientific_profile = scientific_sampling_profile(module_id)
        scientific_minimum = required_frames_per_replica(
            scientific_profile,
            system_ids_per_replica=system_ids_per_replica,
            source_frames_per_replica=(
                replica_counts if timing_available else None
            ),
            frame_intervals_ns_per_replica=frame_intervals_ns_per_replica,
            source_time_spans_ns_per_replica=(
                source_time_spans_ns_per_replica
            ),
        )
        maximum_per_replica = (
            ceiling
            if profile.budget_scope == "per_replica"
            else max(1, ceiling // replica_count)
        )
        attainable_scientific_minimum = min(
            scientific_minimum, max(replica_counts)
        )
        minimum_per_replica = min(
            max(replica_counts),
            max(technical_pilot, attainable_scientific_minimum),
        )
        maximum_per_replica = min(
            max(replica_counts),
            max(maximum_per_replica, minimum_per_replica),
        )
        workload_multiplier, workload_basis = _runtime_workload_multiplier(
            profile, dimensions
        )
        reference_memory_gib = {
            "streaming": 4.0,
            "validated_30k": 12.0,
            "moderate": 12.0,
            "expensive": 24.0,
        }.get(profile.tier, 12.0)
        if measured is not None:
            reference_memory_gib = max(
                reference_memory_gib,
                float(measured["maximum_resident_memory_mib"])
                * float(execution.get("memory_safety_factor", 1.25)) / 1024.0,
            )
        memory_gib = max(1.0, reference_memory_gib * memory_atom_scale)
        seconds_per_frame = calibration.seconds_per_frame
        calibration_source_policy = "built_in_completed_calibration"
        if measured is not None:
            measured_rate = float(
                measured["conservative_cpu_seconds_per_frame"]
            )
            # A right-censored timeout is a lower bound, not a reason to
            # discard a later completed calibration.  Completed catalog data
            # remains authoritative when available; otherwise retain the
            # larger of the completed built-in rate and the censored bound.
            if int(measured["complete_measurement_count"]) > 0:
                seconds_per_frame = measured_rate
                calibration_source_policy = "completed_catalog_measurement"
            else:
                seconds_per_frame = max(
                    calibration.seconds_per_frame, measured_rate
                )
                calibration_source_policy = (
                    "completed_builtin_over_censored_catalog_lower_bound"
                )
        calibration_id = (
            "measured-catalog:" + str(measured["catalog_sha256"]) + f":{module_id}"
            if measured is not None else calibration.calibration_id
        )
        calibration_status = (
            str(measured["calibration_evidence_status"])
            if measured is not None else calibration.evidence_level
        )
        if calibration_source_policy == (
            "completed_builtin_over_censored_catalog_lower_bound"
        ):
            calibration_id = (
                f"{calibration.calibration_id}+censored-catalog:"
                f"{measured['catalog_sha256']}:{module_id}"
            )
            calibration_status = (
                "completed_builtin_with_censored_catalog_lower_bound"
            )
        parallel_qc_enabled = (
            module_id == "structural_integrity_qc"
            and str(execution.get("coordinate_cache", "auto")) != "off"
        )
        memory_limited_workers = max(
            1,
            math.floor(
                max(0.0, float(execution["maximum_memory_gib"]) - 1.0)
                / (1.5 * memory_gib)
            ),
        )
        parallel_workers = (
            min(
                replica_count,
                int(execution["maximum_parallel_cpus"]),
                memory_limited_workers,
            )
            if parallel_qc_enabled else 1
        )
        aggregate_memory_gib = memory_gib * parallel_workers
        tasks.append({
            "task_id": f"direct:{module_id}",
            "module_id": module_id,
            "task_scope": "direct_trajectory_estimator",
            "dependency_stage": 1,
            "effective_cpu_cap": parallel_workers,
            "intrinsic_cpu_cap": (
                replica_count if parallel_qc_enabled else 1
            ),
            **({
                "parallel_execution_model": "one_process_per_replica_v1",
                "parallel_worker_count": parallel_workers,
                "estimated_peak_memory_gib_per_parallel_worker": memory_gib,
            } if parallel_qc_enabled else {}),
            "source_frames_per_replica": replica_counts,
            "system_ids_per_replica": system_ids_per_replica,
            **({
                "frame_intervals_ns_per_replica": (
                    frame_intervals_ns_per_replica
                ),
                "source_time_spans_ns_per_replica": (
                    source_time_spans_ns_per_replica
                ),
            } if timing_available else {}),
            "minimum_frames_per_replica": minimum_per_replica,
            "minimum_frame_role": "standard_scientific_raw_coverage",
            "minimum_frame_interpretation": (
                "Fixed method-specific sample-count and maximum-temporal-"
                "separation floor. Runtime pilots calibrate cost only; the planner "
                "does not estimate autocorrelation times or event rates."
            ),
            "technical_pilot_frames_per_replica": technical_pilot,
            "scientific_sampling_requirements": profile_contract(
                scientific_profile
            ),
            "scientific_minimum_frames_per_replica": scientific_minimum,
            "attainable_scientific_minimum_frames_per_replica": (
                attainable_scientific_minimum
            ),
            "maximum_frames_per_replica": maximum_per_replica,
            "cpu_seconds_per_physical_frame": (
                seconds_per_frame
                * workload_multiplier
                * time_safety_factor
            ),
            "fixed_cpu_hours": (
                calibration.fixed_overhead_seconds
                * time_safety_factor / 3600.0
            ),
            "estimated_peak_memory_gib": aggregate_memory_gib,
            "reference_peak_memory_gib": reference_memory_gib,
            "memory_atom_scale": memory_atom_scale,
            "memory_reference_atom_count": REFERENCE_ATOM_COUNT,
            "memory_size_scaling_applied": True,
            "measured_memory_multiplier": memory_atom_scale,
            **({
                "measured_memory_cost_model": {
                    "calibration_observations": int(
                        measured["maximum_measured_observation_count"]
                    ),
                    "calibration_memory_gib": aggregate_memory_gib,
                    "memory_exponent": 0.5,
                    "minimum_observation_scale": 0.1,
                    "workload_scaling_applied": True,
                },
            } if (
                measured is not None
                and int(measured["complete_measurement_count"]) > 0
                and int(measured["maximum_measured_observation_count"]) > 0
            ) else {}),
            "priority_weight": _CAMPAIGN_PRIORITY.get(module_id, 4.0),
            "calibration_status": calibration_status,
            "calibration_id": calibration_id,
            "calibration_source_policy": calibration_source_policy,
            "calibration_selected_frame_coverage": (
                int(measured["maximum_measured_selected_frame_count"])
                if measured is not None else None
            ),
            "calibration_observation_coverage": (
                int(measured["maximum_measured_observation_count"])
                if measured is not None else None
            ),
            "calibration_complete_measurement_count": (
                int(measured["complete_measurement_count"])
                if measured is not None else 0
            ),
            "calibration_censored_timeout_count": (
                int(measured["censored_timeout_count"])
                if measured is not None else 0
            ),
            "runtime_workload_scaling": workload_basis,
            "balance_group": f"direct:{module_id}",
            "replica_sampling_mode": (
                "independent_all_available"
                if module_id == "replica_rmsd_rg" else "balanced_pooled"
            ),
        })
    if not tasks:
        raise AutomaticSamplingError(
            "campaign planning requires at least one direct trajectory estimator"
        )
    plan = plan_campaign_resource_budget(
        tasks,
        maximum_parallel_cpus=int(execution["maximum_parallel_cpus"]),
        maximum_wall_hours=float(execution["maximum_hours_per_cpu"]),
        maximum_memory_gib=float(execution["maximum_memory_gib"]),
        planning_utilization=float(execution["planning_utilization"]),
        pilot_budget_fraction=float(execution["pilot_budget_fraction"]),
        finalization_headroom_fraction=float(
            execution.get("finalization_headroom_fraction", 0.0)
        ),
    )
    plan["planning_scope"] = "enabled direct trajectory estimators"
    plan["scope_limit"] = (
        "Derived-data tasks and repeated conformational-view workflows require "
        "the campaign-DAG expansion layer before this can be treated as the final "
        "whole-campaign estimate."
    )
    return plan


def _apply_campaign_direct_allocations(
    method_plans: Sequence[Dict[str, object]],
    dimensions: Mapping[str, object],
    campaign: Mapping[str, object],
    *,
    time_safety_factor: float,
) -> None:
    allocations = {
        str(row["module_id"]): row
        for row in campaign["tasks"]  # type: ignore[union-attr]
        if (
            isinstance(row, dict)
            and "module_id" in row
            and row.get("task_scope") == "direct_trajectory_estimator"
        )
    }
    for row in method_plans:
        module_id = str(row["module_id"])
        allocation = allocations.get(module_id)
        if allocation is None or module_id not in _BY_MODULE:
            continue
        selected_per_replica = [
            int(value)
            for value in allocation["selected_physical_frames_per_replica"]
        ]
        requested_budget = max(selected_per_replica)
        selection, stride, selected_count, selected_maximum = _execution_selection(
            _BY_MODULE[module_id], dimensions, requested_budget
        )
        embedded_contract = allocation.get("scientific_sampling_requirements")
        assessment_profile = (
            profile_from_contract(embedded_contract)
            if isinstance(embedded_contract, Mapping)
            else scientific_sampling_profile(module_id)
        )
        assessment_policy_id = (
            str(embedded_contract.get("policy_id", POLICY_ID))
            if isinstance(embedded_contract, Mapping) else POLICY_ID
        )
        scientific_assessment = assess_raw_sampling(
            assessment_profile,
            selected_frames_per_replica=selected_per_replica,
            source_frames_per_replica=[
                int(value)
                for value in allocation["source_frames_per_replica"]
            ],
            system_ids_per_replica=[
                str(row["system_id"])
                for row in dimensions["replicas"]  # type: ignore[union-attr]
            ],
            integer_stride=int(allocation["integer_stride"]),
            frame_intervals_ns_per_replica=(
                [
                    float(value) for value in allocation[
                        "frame_intervals_ns_per_replica"
                    ]
                ]
                if isinstance(
                    allocation.get("frame_intervals_ns_per_replica"), list
                ) else None
            ),
            source_time_spans_ns_per_replica=(
                [
                    float(value) for value in allocation[
                        "source_time_spans_ns_per_replica"
                    ]
                ]
                if isinstance(
                    allocation.get("source_time_spans_ns_per_replica"), list
                ) else None
            ),
            policy_id=assessment_policy_id,
        )
        source_count = int(dimensions["total_source_frame_count"])
        row.update({
            "requested_maximum_total_frame_capacity": sum(selected_per_replica),
            "requested_maximum_frames_per_replica": requested_budget,
            "resolved_maximum_total_frames": selected_count,
            "resolved_maximum_frames_per_replica": selected_maximum,
            "selected_frame_count": selected_count,
            "coverage_fraction": selected_count / source_count,
            "subsampling_triggered": selected_count < source_count,
            "subsampling_reason": (
                "configured whole-campaign CPU/wall envelope"
                if selected_count < source_count else None
            ),
            "sampling_strategy": (
                "all source frames; no random draw"
                if selected_count == source_count else
                "deterministic full-timespan sampling within every replica; no random draw"
            ),
            "frame_selection": selection,
            "frame_stride": stride,
            "scientific_sampling_assessment": scientific_assessment,
            "scientific_sampling_requirements": scientific_assessment[
                "requirements"
            ],
            "scientific_minimums_source_path": allocation.get(
                "scientific_minimums_source_path"
            ),
            "scientific_minimums_source_sha256": allocation.get(
                "scientific_minimums_source_sha256"
            ),
            "recommended_execution": bool(
                scientific_assessment["keep_enabled"]
            ),
            "campaign_resource_allocation": {
                "task_id": allocation["task_id"],
                "selected_physical_frames_per_replica": selected_per_replica,
                "estimated_cpu_hours": allocation["estimated_cpu_hours"],
                "estimated_wall_hours_at_effective_cpu_cap": allocation[
                    "estimated_wall_hours_at_effective_cpu_cap"
                ],
                "priority_weight": allocation["priority_weight"],
            },
        })
        resource = row.get("resource_time_budget")
        if isinstance(resource, dict):
            estimated_seconds = (
                float(resource["estimated_fixed_overhead_seconds"])
                + float(resource["estimated_seconds_per_frame"]) * selected_count
            ) * time_safety_factor
            resource.update({
                "budget_scope": "whole configured direct-estimator campaign",
                "estimated_selected_wall_seconds": estimated_seconds,
                "estimated_selected_wall_hours": estimated_seconds / 3600.0,
            })


# These modules operate on already-derived observations.  They must retain the
# upstream frame identities and must not silently draw an independent sample.
INHERITED_FRAME_SOURCES: Mapping[str, str] = {
    "generalized_correlation_and_information": "common_pca",
    "information_dynamics": "common_pca",
    "correlation_networks": "dccm",
    "time_lagged_independent_component_analysis": "common_pca",
    "pca_fes_basins": "common_pca",
    "clustering_kmeans": "common_pca",
    "clustering_hdbscan": "common_pca",
    "clustering_imwkmeans": "common_pca",
    "alternative_clustering": "common_pca",
    "pald_community_analysis": "common_pca",
    "representative_frames": "common_pca",
    "state_coordinate_exports": "common_pca",
    "representative_structures": "common_pca",
    "markov_state_models": "common_pca",
    "scalar_feature_distributions": "trajectory_features",
    "scalar_threshold_states": "trajectory_features",
    "hydrogen_bond_patterns": "hydrogen_bond_discovery",
    "hydrogen_bond_comparison": "hydrogen_bond_discovery",
    "grouped_ml": "common_pca",
    "grouped_regularized_classification": "hydrogen_bond_discovery",
    "convergence_uncertainty": "replica_rmsd_rg",
    "rmsf_permutation_inference": "pooled_rmsf",
    "integrated_comparison": "upstream_reports",
}


NO_FRAME_SAMPLING = {
    "provenance_manifest",
    "preflight_inventory",
    "common_atom_mapping",
}


INHERITED_POSTPROCESSING_LIMITS: Mapping[str, Mapping[str, object]] = {
    "pca_fes_basins": {
        "silhouette_focal_observation_ceiling": 1_000,
        "silhouette_reference_partition": "all assigned pooled observations",
        "scaling": "O(B*N*d) with B focal observations, N assigned frames, and d=2 PCA coordinates",
    },
    "clustering_kmeans": {"silhouette_focal_observation_ceiling": 1_000},
    "clustering_hdbscan": {"silhouette_focal_observation_ceiling": 1_000},
    "clustering_imwkmeans": {"silhouette_focal_observation_ceiling": 1_000},
    "alternative_clustering": {"silhouette_focal_observation_ceiling": 1_000},
    "pald_community_analysis": {
        "maximum_community_observations": 500,
        "scaling": "O(B^3) triplet comparisons and O(B^2) matrix storage",
        "sampling": "one common regular stride across replica-member segments",
    },
    "hydrogen_bond_patterns": {
        "maximum_pattern_frames": 1_000,
        "scaling": "O(N^2) frame-to-frame Jaccard distance memory and time",
        "sampling": "balanced across systems and replicas when the upstream report exceeds the ceiling",
    },
}


def sampling_profile(module_id: str) -> SamplingProfile:
    try:
        return _BY_MODULE[module_id]
    except KeyError as exc:
        raise AutomaticSamplingError(
            f"module {module_id!r} does not consume trajectory frames directly"
        ) from exc


def _frame_count(probe: Mapping[str, object]) -> int:
    value = probe.get("observed_frame_count", probe.get("declared_frame_count"))
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise AutomaticSamplingError("trajectory probe did not return a positive frame count")


def inspect_sampling_dimensions(
    system_manifest: Mapping[str, object], system_path: Path
) -> Dict[str, object]:
    """Inspect atom and per-replica frame counts without reading coordinates."""

    source = Path(system_path).expanduser().resolve(strict=False)
    validate_system(system_manifest, source_path=source, check_paths=True)
    replica_rows = []
    atom_counts = []
    for raw_system in system_manifest["systems"]:
        if not isinstance(raw_system, dict):
            raise AutomaticSamplingError("system entries must be objects")
        for raw_replica in raw_system["replicas"]:
            if not isinstance(raw_replica, dict):
                raise AutomaticSamplingError("replica entries must be objects")
            topology = resolve_manifest_path(str(raw_replica["topology"]), source)
            try:
                topology_probe = probe_topology(topology)
            except (FileProbeError, OSError) as exc:
                raise AutomaticSamplingError(str(exc)) from exc
            atom_count = topology_probe.get("atom_count")
            if isinstance(atom_count, bool) or not isinstance(atom_count, int) or atom_count <= 0:
                raise AutomaticSamplingError("topology probe did not return an atom count")
            atom_counts.append(atom_count)
            segments = []
            replica_frames = 0
            replica_time_span_ns = 0.0
            replica_frame_intervals_ns = []
            timing_available = True
            for raw_segment in raw_replica["segments"]:
                if not isinstance(raw_segment, dict):
                    raise AutomaticSamplingError("segment entries must be objects")
                trajectory = resolve_manifest_path(str(raw_segment["trajectory"]), source)
                try:
                    trajectory_probe = probe_trajectory(trajectory)
                except (FileProbeError, OSError) as exc:
                    raise AutomaticSamplingError(str(exc)) from exc
                trajectory_atoms = trajectory_probe.get("atom_count")
                if trajectory_atoms != atom_count:
                    raise AutomaticSamplingError(
                        f"{raw_system['system_id']}/{raw_replica['replica_id']}/"
                        f"{raw_segment['segment_id']} has {trajectory_atoms} trajectory atoms "
                        f"but {atom_count} topology atoms"
                    )
                count = _frame_count(trajectory_probe)
                replica_frames += count
                segment_row = {
                    "segment_id": str(raw_segment["segment_id"]),
                    "source_frame_count": count,
                }
                if raw_segment.get("timing") is not None:
                    try:
                        normalized_timing = normalize_segment_timing(
                            raw_segment, "ns"
                        )
                    except TrajectoryContractError as exc:
                        raise AutomaticSamplingError(str(exc)) from exc
                    interval_ns = float(normalized_timing["frame_interval"])
                    span_ns = max(0, count - 1) * interval_ns
                    replica_frame_intervals_ns.append(interval_ns)
                    replica_time_span_ns += span_ns
                    segment_row.update({
                        "frame_interval_ns": interval_ns,
                        "source_time_span_ns": span_ns,
                    })
                else:
                    timing_available = False
                segments.append(segment_row)
            replica_rows.append({
                "system_id": str(raw_system["system_id"]),
                "replica_id": str(raw_replica["replica_id"]),
                "atom_count": atom_count,
                "source_frame_count": replica_frames,
                "segments": segments,
                "maximum_frame_interval_ns": (
                    max(replica_frame_intervals_ns)
                    if timing_available and replica_frame_intervals_ns else None
                ),
                "source_time_span_ns": (
                    replica_time_span_ns
                    if timing_available and replica_frame_intervals_ns else None
                ),
            })
    if not replica_rows:
        raise AutomaticSamplingError("system manifest contains no replicas")
    return {
        "system_ids": sorted({str(row["system_id"]) for row in replica_rows}),
        "replica_count": len(replica_rows),
        "maximum_atom_count": max(atom_counts),
        "minimum_atom_count": min(atom_counts),
        "total_source_frame_count": sum(int(row["source_frame_count"]) for row in replica_rows),
        "minimum_source_frames_per_replica": min(
            int(row["source_frame_count"]) for row in replica_rows
        ),
        "maximum_source_frames_per_replica": max(
            int(row["source_frame_count"]) for row in replica_rows
        ),
        "replicas": replica_rows,
        "count_provenance": "topology records and trajectory header/record metadata",
    }


def _hydrogen_bond_candidate_dimensions(
    system_manifest: Mapping[str, object], system_path: Path,
) -> Dict[str, object]:
    """Enumerate the default automatic common candidate universe once.

    This is an outcome-independent topology/connectivity calculation. It does
    not read trajectory coordinates or use hydrogen-bond occupancies.
    """

    try:
        common, report = _automatic_candidate_intersection(
            system_manifest,
            Path(system_path).expanduser().resolve(strict=False),
            {
                "interaction_scope": "all_solute",
                "exclude_same_residue": True,
            },
        )
    except (HydrogenBondDiscoveryError, OSError, ValueError) as exc:
        return {
            "status": "unavailable",
            "reason": str(exc),
            "planning_fallback": (
                "total topology atom count; generated projects retain an explicit "
                "feature-observation gate"
            ),
        }
    return {
        "status": "complete",
        "chemistry_policy": "automatic_topology_templates_v1",
        "interaction_scope": "all_solute",
        "exclude_same_residue": True,
        "candidate_harmonization": "intersection_by_atom_index_v1",
        "common_candidate_count": len(common),
        "union_candidate_count": int(report["union_candidate_count"]),
        "excluded_from_common_union_count": int(
            report["excluded_from_common_union_count"]
        ),
        "replica_dictionaries": report["replica_dictionaries"],
        "selection_basis": report["selection_basis"],
        "coordinate_data_used": False,
    }


def _runtime_workload_multiplier(
    profile: SamplingProfile, dimensions: Mapping[str, object],
) -> Tuple[float, Dict[str, object]]:
    """Return the method-specific runtime dimension and provenance."""

    atom_count = int(dimensions["maximum_atom_count"])
    atom_multiplier = max(
        0.01,
        (atom_count / REFERENCE_ATOM_COUNT) ** profile.atom_scaling_exponent,
    )
    candidate_plan = dimensions.get("hydrogen_bond_candidate_planning")
    if (
        profile.module_id == "hydrogen_bond_discovery"
        and isinstance(candidate_plan, dict)
        and candidate_plan.get("status") == "complete"
    ):
        candidate_count = int(candidate_plan["common_candidate_count"])
        candidate_multiplier = max(
            MINIMUM_HYDROGEN_BOND_WORKLOAD_MULTIPLIER,
            candidate_count / REFERENCE_HYDROGEN_BOND_CANDIDATE_COUNT,
        )
        return candidate_multiplier, {
            "dimension": "common automatic donor-hydrogen-acceptor candidates",
            "observed_candidate_count": candidate_count,
            "reference_candidate_count": REFERENCE_HYDROGEN_BOND_CANDIDATE_COUNT,
            "minimum_multiplier": MINIMUM_HYDROGEN_BOND_WORKLOAD_MULTIPLIER,
            "resolved_multiplier": candidate_multiplier,
            "coordinate_data_used": False,
        }
    return atom_multiplier, {
        "dimension": "maximum topology atom count proxy",
        "observed_atom_count": atom_count,
        "reference_atom_count": REFERENCE_ATOM_COUNT,
        "atom_scaling_exponent": profile.atom_scaling_exponent,
        "resolved_multiplier": atom_multiplier,
        "limitation": (
            candidate_plan.get("reason")
            if profile.module_id == "hydrogen_bond_discovery"
            and isinstance(candidate_plan, dict)
            else None
        ),
    }


def _effective_ceiling(profile: SamplingProfile, atom_count: int) -> Tuple[int, float]:
    size_ratio = REFERENCE_ATOM_COUNT / atom_count
    size_factor = min(1.0, size_ratio ** profile.atom_scaling_exponent)
    ceiling = math.floor(profile.maximum_total_frames * size_factor)
    return max(1, ceiling), size_factor


def _technical_pilot_frames_per_replica(
    profile: SamplingProfile, atom_count: int
) -> int:
    """Scale a small runtime pilot without claiming scientific sufficiency."""

    size_factor = min(
        1.0,
        (REFERENCE_ATOM_COUNT / atom_count) ** profile.atom_scaling_exponent,
    )
    return max(5, min(profile.minimum_frames_per_replica, math.ceil(
        profile.minimum_frames_per_replica * size_factor
    )))


def _runtime_budget(
    profile: SamplingProfile,
    *,
    dimensions: Mapping[str, object],
    source_frame_count: int,
    target_wall_seconds: float,
    time_safety_factor: float,
    measured_calibration: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Estimate full cost and a target-fitting frame capacity.

    Cost per frame scales conservatively from the reference system with the
    method's declared atom exponent.  This is still only a proxy for methods
    governed by selected atoms, candidates, waters, or sphere points; that
    limitation remains explicit in the returned record.
    """

    if target_wall_seconds <= 0.0 or not math.isfinite(target_wall_seconds):
        raise AutomaticSamplingError("target_wall_seconds must be finite and positive")
    if time_safety_factor <= 0.0 or not math.isfinite(time_safety_factor):
        raise AutomaticSamplingError("time_safety_factor must be finite and positive")
    calibration = _RUNTIME_BY_MODULE[profile.module_id]
    seconds_per_frame = calibration.seconds_per_frame
    fixed_overhead_seconds = calibration.fixed_overhead_seconds
    calibration_id = calibration.calibration_id
    evidence_level = calibration.evidence_level
    rationale = calibration.rationale
    if measured_calibration is not None:
        seconds_per_frame = float(
            measured_calibration["conservative_cpu_seconds_per_frame"]
        )
        fixed_overhead_seconds = 0.0
        calibration_id = (
            "measured-catalog:"
            + str(measured_calibration["catalog_sha256"])
            + f":{profile.module_id}"
        )
        evidence_level = str(
            measured_calibration["calibration_evidence_status"]
        )
        rationale = (
            f"conservative planner rate from "
            f"{measured_calibration['complete_measurement_count']} complete "
            "report-bound measurement(s) and "
            f"{measured_calibration['censored_timeout_count']} separately "
            "labeled right-censored timeout(s)"
        )
    atom_count = int(dimensions["maximum_atom_count"])
    atom_multiplier = max(
        0.01,
        (atom_count / REFERENCE_ATOM_COUNT) ** profile.atom_scaling_exponent,
    )
    workload_multiplier, workload_basis = _runtime_workload_multiplier(
        profile, dimensions
    )
    estimated_seconds_per_frame = (
        seconds_per_frame * workload_multiplier
    )
    # Fixed startup/setup work is not a per-atom term.  Scaling it by the
    # trajectory atom count made small/no-water systems look implausibly cheap
    # and distorted the intercept retained by multi-point pilots.
    estimated_fixed_overhead = fixed_overhead_seconds
    estimated_full_wall = (
        estimated_fixed_overhead
        + estimated_seconds_per_frame * source_frame_count
    ) * time_safety_factor
    usable_seconds = target_wall_seconds / time_safety_factor - estimated_fixed_overhead
    capacity = (
        math.floor(usable_seconds / estimated_seconds_per_frame)
        if usable_seconds > 0.0 else 0
    )
    return {
        "target_wall_seconds": target_wall_seconds,
        "target_wall_hours": target_wall_seconds / 3600.0,
        "time_safety_factor": time_safety_factor,
        "calibration_id": calibration_id,
        "calibration_evidence_level": evidence_level,
        "reference_seconds_per_frame": seconds_per_frame,
        "reference_fixed_overhead_seconds": fixed_overhead_seconds,
        "measured_frame_coverage": (
            int(measured_calibration["maximum_measured_selected_frame_count"])
            if measured_calibration is not None else None
        ),
        "measured_observation_coverage": (
            int(measured_calibration["maximum_measured_observation_count"])
            if measured_calibration is not None else None
        ),
        "complete_measurement_count": (
            int(measured_calibration["complete_measurement_count"])
            if measured_calibration is not None else 0
        ),
        "censored_timeout_count": (
            int(measured_calibration["censored_timeout_count"])
            if measured_calibration is not None else 0
        ),
        "censored_timeout_rate_lower_bound": (
            measured_calibration.get(
                "maximum_censored_cpu_seconds_per_frame_lower_bound"
            )
            if measured_calibration is not None else None
        ),
        "measured_maximum_resident_memory_mib": (
            float(measured_calibration["maximum_resident_memory_mib"])
            if measured_calibration is not None else None
        ),
        "atom_runtime_multiplier": atom_multiplier,
        "workload_runtime_multiplier": workload_multiplier,
        "runtime_workload_scaling": workload_basis,
        "estimated_seconds_per_frame": estimated_seconds_per_frame,
        "estimated_fixed_overhead_seconds": estimated_fixed_overhead,
        "estimated_full_wall_seconds": estimated_full_wall,
        "estimated_full_wall_hours": estimated_full_wall / 3600.0,
        "time_limited_total_frame_capacity": max(0, capacity),
        "full_frames_fit_time_budget": estimated_full_wall <= target_wall_seconds,
        "rationale": rationale,
        "limitation": (
            "The atom-count adjustment is a conservative proxy. Selected atoms, "
            "candidate bonds, waters, surface atoms, sphere points, external-program "
            "cost, node generation, and filesystem state can require a project-local pilot."
        ),
    }


def _fixed_stride_selected_count(
    dimensions: Mapping[str, object], stride: int,
) -> Tuple[int, int]:
    replica_counts = []
    for row in dimensions["replicas"]:  # type: ignore[union-attr]
        replica_counts.append(sum(
            integer_stride_selected_count(
                int(segment["source_frame_count"]), stride
            )
            for segment in row["segments"]
        ))
    return sum(replica_counts), max(replica_counts)


def _execution_selection(
    profile: SamplingProfile, dimensions: Mapping[str, object], budget: int,
) -> Tuple[Dict[str, object], int, int, int]:
    source_max = int(dimensions["maximum_source_frames_per_replica"])
    if source_max <= budget:
        source_count = int(dimensions["total_source_frame_count"])
        return {"mode": "fixed_stride_v1"}, 1, source_count, source_max
    source_counts = [
        int(row["source_frame_count"])
        for row in dimensions["replicas"]  # type: ignore[union-attr]
    ]
    stride = integer_stride_for_budget(
        source_counts, budget, error_type=AutomaticSamplingError
    )
    if profile.module_id in CONCATENATED_SELECTION_MODULES:
        replica_counts = [
            integer_stride_selected_count(source_count, stride)
            for source_count in source_counts
        ]
        return (
            {"mode": "integer_stride_per_replica_v1", "stride": stride},
            1,
            sum(replica_counts),
            max(replica_counts),
        )
    selected_count, maximum_per_replica = _fixed_stride_selected_count(
        dimensions, stride
    )
    while maximum_per_replica > budget:
        stride += 1
        selected_count, maximum_per_replica = _fixed_stride_selected_count(
            dimensions, stride
        )
    while stride > 1:
        finer_count, finer_maximum = _fixed_stride_selected_count(
            dimensions, stride - 1
        )
        if finer_maximum > budget:
            break
        stride -= 1
        selected_count, maximum_per_replica = finer_count, finer_maximum
    return {"mode": "fixed_stride_v1"}, stride, selected_count, maximum_per_replica


def _module_plan(
    profile: SamplingProfile,
    dimensions: Mapping[str, object],
    *,
    b_vs_2b: bool,
    replica_diagnostics: bool,
    target_wall_seconds: float,
    time_safety_factor: float,
    measured_calibration: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    ceiling, size_factor = _effective_ceiling(
        profile, int(dimensions["maximum_atom_count"])
    )
    source_count = int(dimensions["total_source_frame_count"])
    replica_count = int(dimensions["replica_count"])
    runtime = _runtime_budget(
        profile,
        dimensions=dimensions,
        source_frame_count=source_count,
        target_wall_seconds=target_wall_seconds,
        time_safety_factor=time_safety_factor,
        measured_calibration=measured_calibration,
    )
    hard_total_ceiling = (
        ceiling * replica_count
        if profile.budget_scope == "per_replica" else ceiling
    )
    resolved_total_ceiling = min(
        hard_total_ceiling,
        int(runtime["time_limited_total_frame_capacity"]),
    )
    per_replica_budget = max(1, resolved_total_ceiling // replica_count)
    resolved_total_ceiling = per_replica_budget * replica_count
    technical_pilot = _technical_pilot_frames_per_replica(
        profile, int(dimensions["maximum_atom_count"])
    )
    selection, frame_stride, selected_count, maximum_selected_per_replica = (
        _execution_selection(profile, dimensions, per_replica_budget)
    )
    stride = int(selection.get("stride", frame_stride))
    selected_per_replica = [
        integer_stride_selected_count(int(row["source_frame_count"]), stride)
        for row in dimensions["replicas"]  # type: ignore[union-attr]
    ]
    minimum_pilot_met = all(
        count >= technical_pilot for count in selected_per_replica
    )
    scientific_profile = scientific_sampling_profile(profile.module_id)
    system_ids_per_replica = [
        str(row["system_id"])
        for row in dimensions["replicas"]  # type: ignore[union-attr]
    ]
    scientific_assessment = assess_raw_sampling(
        scientific_profile,
        selected_frames_per_replica=selected_per_replica,
        source_frames_per_replica=[
            int(row["source_frame_count"])
            for row in dimensions["replicas"]  # type: ignore[union-attr]
        ],
        system_ids_per_replica=system_ids_per_replica,
        integer_stride=stride,
    )
    subsampled = selected_count < source_count
    selected_wall_seconds = (
        float(runtime["estimated_fixed_overhead_seconds"])
        + float(runtime["estimated_seconds_per_frame"]) * selected_count
    ) * time_safety_factor
    reasons = []
    if source_count > int(runtime["time_limited_total_frame_capacity"]):
        reasons.append(
            f"estimated all-frame wall time exceeds the {target_wall_seconds / 3600.0:g}-hour per-method target"
        )
    if source_count > hard_total_ceiling:
        reasons.append("source trajectory exceeds the fail-safe extrapolation ceiling")
    if b_vs_2b and subsampled:
        doubled_total = min(2 * resolved_total_ceiling, source_count)
        doubled = (
            min(2 * per_replica_budget, int(dimensions["maximum_source_frames_per_replica"]))
            if profile.budget_scope == "per_replica"
            else max(technical_pilot, doubled_total // replica_count)
        )
        doubled_selection, doubled_stride, doubled_selected, doubled_maximum = (
            _execution_selection(profile, dimensions, doubled)
        )
        sensitivity = {
            "enabled": True,
            "status": "planned",
            "base_maximum_total_frames": selected_count,
            "base_maximum_frames_per_replica": maximum_selected_per_replica,
            "doubled_maximum_total_frames": doubled_selected,
            "doubled_maximum_frames_per_replica": doubled_maximum,
            "doubled_frame_selection": doubled_selection,
            "doubled_frame_stride": doubled_stride,
            "interpretation": "optional computational sensitivity comparison; not a convergence proof",
        }
    else:
        sensitivity = {
            "enabled": b_vs_2b,
            "status": "not_applicable" if b_vs_2b else "off",
            "rationale": (
                "base plan already evaluates all source frames"
                if b_vs_2b else "not requested by the project owner"
            ),
        }
    return {
        "module_id": profile.module_id,
        "frame_contract": profile.frame_contract,
        "budget_scope": profile.budget_scope,
        "scaling_tier": profile.tier,
        "tier_maximum_total_frames": profile.maximum_total_frames,
        "technical_pilot_frames_per_replica": technical_pilot,
        "technical_pilot_role": (
            "runtime and memory calibration only; not a scientific minimum, "
            "convergence threshold, or production recommendation"
        ),
        "minimum_pilot_frames_per_replica_met": minimum_pilot_met,
        "scientific_sampling_requirements": profile_contract(
            scientific_profile
        ),
        "scientific_sampling_assessment": scientific_assessment,
        "recommended_execution": bool(
            scientific_assessment["keep_enabled"]
        ),
        "planned_selected_frames_per_replica": selected_per_replica,
        "reference_atom_count": REFERENCE_ATOM_COUNT,
        "observed_maximum_atom_count": int(dimensions["maximum_atom_count"]),
        "atom_scaling_exponent": profile.atom_scaling_exponent,
        "atom_size_factor": size_factor,
        "resource_time_budget": {
            **runtime,
            "estimated_selected_wall_seconds": selected_wall_seconds,
            "estimated_selected_wall_hours": selected_wall_seconds / 3600.0,
        },
        "requested_maximum_total_frame_capacity": resolved_total_ceiling,
        "requested_maximum_frames_per_replica": per_replica_budget,
        "resolved_maximum_total_frames": selected_count,
        "resolved_maximum_frames_per_replica": maximum_selected_per_replica,
        "source_frame_count": source_count,
        "selected_frame_count": selected_count,
        "coverage_fraction": selected_count / source_count,
        "subsampling_triggered": subsampled,
        "subsampling_reason": (
            "; ".join(reasons)
            if subsampled else None
        ),
        "sampling_strategy": (
            f"exact integer stride {selection.get('stride', frame_stride)} over every "
            + (
                "replica's concatenated timeline; frame zero retained; no random draw"
                if selection["mode"] == "integer_stride_per_replica_v1" else
                "trajectory segment; segment frame zero retained; no random draw"
            )
            if subsampled else "all source frames; no random draw"
        ),
        "frame_selection": selection,
        "frame_stride": frame_stride,
        "b_vs_2b": sensitivity,
        "replica_diagnostics": {
            "enabled": replica_diagnostics,
            "status": "optional_exploratory" if replica_diagnostics else "off",
            "recommended": False,
            "methods": (
                ["autocorrelation-adjusted effective sample size", "time-block uncertainty"]
                if replica_diagnostics else []
            ),
            "interpretation": (
                "May identify analyses for which additional independent simulations could be useful; "
                "replica agreement and leave-one-replica-out are not acceptance measures."
            ),
        },
        "rationale": profile.rationale,
        "planning_warning": (
            "selected coverage is below the method's sample-count or applicable "
            "ordered-method temporal-resolution floor; increase resources or "
            "reduce the enabled-method set"
            if not scientific_assessment["keep_enabled"] else
            None if minimum_pilot_met else
            "the pooled ceiling cannot provide even the small technical runtime pilot to every replica"
        ),
    }


def automatic_sampling_plan(
    system_path: Path,
    *,
    simulation_kind: str,
    module_ids: Optional[Sequence[str]] = None,
    b_vs_2b: bool = False,
    replica_diagnostics: bool = False,
    target_wall_seconds: float = DEFAULT_TARGET_WALL_SECONDS,
    time_safety_factor: float = DEFAULT_TIME_SAFETY_FACTOR,
    campaign_execution: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Create a complete read-only sampling plan from minimal project input."""

    if simulation_kind not in SIMULATION_KINDS:
        raise AutomaticSamplingError(
            "simulation_kind must be one of " + ", ".join(sorted(SIMULATION_KINDS))
        )
    source = Path(system_path).expanduser().resolve(strict=False)
    try:
        manifest = load_json(source)
        dimensions = inspect_sampling_dimensions(manifest, source)
    except (FileProbeError, OSError, ValueError) as exc:
        if isinstance(exc, AutomaticSamplingError):
            raise
        raise AutomaticSamplingError(str(exc)) from exc
    requested = (
        list(module_ids) if module_ids is not None
        else [module.module_id for module in list_modules()]
    )
    if not requested or len(set(requested)) != len(requested):
        raise AutomaticSamplingError("module_ids must be nonempty and unique")
    unknown = sorted(
        set(requested)
        .difference(_BY_MODULE)
        .difference(INHERITED_FRAME_SOURCES)
        .difference(NO_FRAME_SAMPLING)
    )
    if unknown:
        raise AutomaticSamplingError("unknown sampling modules: " + ", ".join(unknown))
    if "hydrogen_bond_discovery" in requested:
        dimensions["hydrogen_bond_candidate_planning"] = (
            _hydrogen_bond_candidate_dimensions(manifest, source)
        )
    measured_calibrations: Dict[str, Dict[str, object]] = {}
    if campaign_execution is not None:
        try:
            measured_calibrations = load_resource_calibration_catalog(
                campaign_execution.get("resource_calibration_catalog"),
                censored_timeout_safety_factor=float(
                    campaign_execution.get(
                        "censored_timeout_safety_factor", 1.5
                    )
                ),
            )
        except (ResourceCalibrationError, OSError, ValueError) as exc:
            raise AutomaticSamplingError(str(exc)) from exc
    direct = [
        _module_plan(
            _BY_MODULE[module_id], dimensions,
            b_vs_2b=b_vs_2b,
            replica_diagnostics=replica_diagnostics,
            target_wall_seconds=target_wall_seconds,
            time_safety_factor=time_safety_factor,
            measured_calibration=measured_calibrations.get(module_id),
        )
        for module_id in requested if module_id in _BY_MODULE
    ]
    campaign_resource_plan = None
    if campaign_execution is not None:
        required_execution = {
            "maximum_parallel_cpus", "maximum_hours_per_cpu",
            "maximum_memory_gib", "planning_utilization",
            "pilot_budget_fraction",
        }
        missing_execution = sorted(required_execution.difference(campaign_execution))
        if missing_execution:
            raise AutomaticSamplingError(
                "campaign_execution is missing: " + ", ".join(missing_execution)
            )
        if direct:
            campaign_resource_plan = _campaign_direct_resource_plan(
                dimensions,
                requested,
                campaign_execution,
                time_safety_factor=time_safety_factor,
                measured_calibrations=measured_calibrations,
            )
            _apply_campaign_direct_allocations(
                direct,
                dimensions,
                campaign_resource_plan,
                time_safety_factor=time_safety_factor,
            )
        else:
            campaign_resource_plan = {
                "planning_schema": "salsbury-campaign-resource-plan-v1",
                "technical_status": "complete",
                "scientific_status": "planning only",
                "feasibility_status": "feasible",
                "execution_authorized": True,
                "planning_scope": "no enabled direct trajectory estimators",
                "raw_capacity_cpu_hours": (
                    int(campaign_execution["maximum_parallel_cpus"])
                    * float(campaign_execution["maximum_hours_per_cpu"])
                ),
                "estimated_selected_cpu_hours": 0.0,
                "tasks": [],
            }
    inherited = [
        {
            "module_id": module_id,
            "frame_contract": "inherit_upstream_frame_identities",
            "inherited_from": INHERITED_FRAME_SOURCES[module_id],
            "independent_resampling_allowed": False,
            "b_vs_2b": "inherited from the upstream estimator when requested",
            "replica_diagnostics": {
                "enabled": replica_diagnostics,
                "status": "optional_exploratory" if replica_diagnostics else "off",
                "recommended": False,
            },
            "postprocessing_limits": dict(
                INHERITED_POSTPROCESSING_LIMITS.get(module_id, {})
            ),
            "target_wall_seconds": target_wall_seconds,
            "target_wall_hours": target_wall_seconds / 3600.0,
            "time_estimate_status": "requires upstream result dimensions or a project-local pilot",
            "scientific_sampling_requirements": profile_contract(
                scientific_sampling_profile(module_id)
            ),
            "scientific_sampling_assessment": {
                "raw_coverage_status": "inherited_from_upstream",
                "upstream_module_id": INHERITED_FRAME_SOURCES[module_id],
                "postrun_diagnostics_and_temporal_validation_remain": True,
            },
        }
        for module_id in requested if module_id in INHERITED_FRAME_SOURCES
    ]
    not_applicable = [
        {
            "module_id": module_id,
            "frame_contract": "not_applicable",
            "reason": "metadata, inventory, or atom-identity operation does not evaluate trajectory frames",
            "scientific_sampling_requirements": profile_contract(
                scientific_sampling_profile(module_id)
            ),
            "scientific_sampling_assessment": {
                "raw_coverage_status": "not_applicable",
                "keep_enabled": True,
                "scientific_interpretation_ready": True,
            },
        }
        for module_id in requested if module_id in NO_FRAME_SAMPLING
    ]
    limitations = [
        "Standard raw-frame floors prevent a runtime pilot from being presented as production coverage; they do not establish convergence.",
        "Total topology atom count is a conservative proxy; selected atoms, candidate bonds, waters, surface atoms, sphere points, and external executables can require a method-specific pilot.",
        "Biased, enhanced, weighted, or AI ensembles may require weight-aware or state-aware sampling beyond uniform frame selection.",
        "B-versus-2B and replica diagnostics are performed only when explicitly requested.",
        "Except for replica-resolved RMSD/Rg, ceilings apply to the pooled estimator total and are allocated equally across replicas.",
    ]
    method_plans = direct + inherited + not_applicable
    below_standard = sorted(
        str(row["module_id"])
        for row in direct
        if not bool(row.get("recommended_execution", True))
    )
    return {
        "planning_schema": "salsbury-automatic-sampling-plan-v1",
        "policy_id": POLICY_ID,
        "technical_status": "complete",
        "scientific_status": "planning only",
        "system_manifest_path": str(source),
        "simulation_kind": simulation_kind,
        "trajectory_authority_status": (
            "owner-supplied manifest; authority is a project decision and is not "
            "inferred from successful inspection"
        ),
        "dimensions": dimensions,
        "owner_choices": {
            "b_vs_2b": b_vs_2b,
            "replica_diagnostics": replica_diagnostics,
        },
        "default_resource_envelope": {
            "target_wall_seconds_per_method": target_wall_seconds,
            "target_wall_hours_per_method": target_wall_seconds / 3600.0,
            "time_safety_factor": time_safety_factor,
            "policy": "estimate every direct method, use all frames when they fit, otherwise sample uniformly across replicas",
        },
        "method_plans": method_plans,
        "scientific_sampling_summary": {
            "policy_id": POLICY_ID,
            "profile_count": len(method_plans),
            "direct_modules_below_standard_raw_floor": below_standard,
            "all_requested_modules_have_explicit_requirements": True,
            "postrun_policy": (
                "Event or transition counts, temporal validation, and "
                "independent-unit requirements remain explicit post-run "
                "diagnostics and are never inferred from raw frame count."
            ),
        },
        "campaign_resource_plan": campaign_resource_plan,
        "measured_resource_calibration": {
            "configured": bool(measured_calibrations),
            "modules_with_measurements": sorted(measured_calibrations),
            "policy": (
                "maximum measured CPU rate and RSS with 1.25 memory safety; "
                "measured frame coverage is evidence, not a scientific ceiling"
            ),
        },
        "limitations": limitations,
    }
