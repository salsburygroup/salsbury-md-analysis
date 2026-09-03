"""Method-specific scientific sampling contracts for campaign planning.

The resource planner historically used small runtime-pilot minima.  Those are
useful for calibration, but they are not enough to justify keeping an analysis
enabled in a production campaign.  This module supplies a separate, exhaustive
standard-coverage policy for every registered analysis method.

The numerical floors are deliberately permissive feasibility minima, not
publication targets or universal convergence claims.  They are intended to
prevent scientifically empty calculations without disabling useful methods
merely because a campaign is resource constrained. Raw-frame coverage is
enforceable before execution. Event and transition counts remain
observable-specific post-run diagnostics. Short pilots are execution and
throughput checks only. The planner does not estimate autocorrelation times or
event rates. It uses a method-specific minimum sample count and, only for
ordered methods, an explicit temporal-resolution or configured lag-pair
requirement.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence


POLICY_ID = "scientific-sampling-standard-v3"
MINIMUMS_SCHEMA = "salsbury-scientific-minimums-v1"


class ScientificSamplingError(ValueError):
    """Raised when a scientific sampling contract is invalid or missing."""


@dataclass(frozen=True)
class ScientificSamplingProfile:
    module_id: str
    sampling_class: str
    minimum_frames_per_replica: int
    minimum_frames_per_system: int
    maximum_uniform_spacing_ns: float
    requires_contiguous_frames: bool
    temporal_resolution_rule: str
    minimum_reported_events_or_transitions: int = 0
    minimum_independent_units_per_group: int = 0
    inherited_from: Optional[str] = None
    insufficiency_action: str = "disable_or_increase_resources"
    rationale: str = ""


def _profile(
    module_id: str,
    sampling_class: str,
    per_replica: int,
    per_system: int,
    *,
    maximum_spacing_ns: float = 0.0,
    contiguous: bool = False,
    temporal_rule: str = "uniform_static_ensemble",
    events: int = 0,
    independent_units: int = 0,
    inherited_from: Optional[str] = None,
    action: str = "disable_or_increase_resources",
    rationale: str,
) -> ScientificSamplingProfile:
    return ScientificSamplingProfile(
        module_id=module_id,
        sampling_class=sampling_class,
        minimum_frames_per_replica=per_replica,
        minimum_frames_per_system=per_system,
        maximum_uniform_spacing_ns=maximum_spacing_ns,
        requires_contiguous_frames=contiguous,
        temporal_resolution_rule=temporal_rule,
        minimum_reported_events_or_transitions=events,
        minimum_independent_units_per_group=independent_units,
        inherited_from=inherited_from,
        insufficiency_action=action,
        rationale=rationale,
    )


_NO_FRAME = {
    "provenance_manifest": "provenance and file identity do not sample frames",
    "preflight_inventory": "input inventory and compatibility checks do not estimate an ensemble",
    "common_atom_mapping": "atom-identity mapping does not estimate an ensemble",
    "integrated_comparison": "report integration inherits every upstream scientific gate",
}


_PROFILES = [
    *(
        _profile(
            module_id, "no_frame_estimator", 0, 0,
            maximum_spacing_ns=0.0,
            temporal_rule="not_applicable", action="not_applicable",
            rationale=rationale,
        )
        for module_id, rationale in _NO_FRAME.items()
    ),
    _profile("structural_integrity_qc", "trajectory_quality_control", 100, 500,
             rationale="distributed structural checks must cover the production span; lightweight continuity checks should still scan every raw frame"),
    _profile("replica_rmsd_rg", "replica_time_series", 100, 100,
             temporal_rule="uniform_ensemble_with_ordered_series_output", rationale="each replica requires its own fitted trajectory profile; time-dependent convergence is evaluated separately"),
    _profile("pooled_rmsf", "ensemble_fluctuation", 200, 1_000,
             rationale="per-position fluctuations require broad per-system ensemble coverage"),
    _profile("dccm", "pairwise_correlation", 250, 1_000,
             rationale="a dense correlation matrix is unstable with a small pooled configuration sample"),
    _profile("individual_pca", "conformational_basis", 250, 1_000,
             rationale="a per-system covariance basis requires broad configuration-space coverage"),
    _profile("common_pca", "shared_conformational_basis", 250, 1_000,
             rationale="shared bases and projections require balanced coverage from every compared system"),
    _profile("dihedral_distributions", "static_distribution", 200, 1_000,
             rationale="angular populations and Scott-rule histograms require more than a sparse frame screen"),
    _profile("hydrogen_bonds", "contact_occupancy", 200, 1_000, events=20,
             rationale="bond occupancies require enough frames and occurrences to distinguish rare contacts from noise"),
    _profile("hydrogen_bond_discovery", "high_dimensional_contact_occupancy", 200, 1_000, events=20,
             rationale="automatic candidate discovery and occupancy ranking are multiple high-dimensional estimates"),
    _profile("water_mediated_hydrogen_bond_networks", "high_dimensional_network_occupancy", 100, 500, events=20,
             rationale="water-bridge edges and network populations cannot be supported by a few disconnected snapshots"),
    _profile("secondary_structure", "categorical_residue_occupancy", 100, 500, events=20,
             rationale="per-residue DSSP populations require distributed observations despite external-program cost"),
    _profile("nucleic_acid_structure", "categorical_structural_occupancy", 100, 500, events=20,
             rationale="motif and descriptor populations require distributed observations despite external-program cost"),
    _profile("nucleic_acid_geometry", "static_distribution", 200, 1_000,
             rationale="ring, stacking, and helical geometry distributions require broad temporal coverage"),
    _profile("ion_coordination_geometry", "contact_and_geometry_occupancy", 200, 1_000, events=20,
             rationale="coordination identities and geometry populations require repeated observations"),
    _profile("ion_atmosphere", "species_resolved_shell_occupancy", 200, 1_000, events=20,
             rationale="species-resolved shell populations require enough configurations per system and ion species"),
    _profile("solvent_accessible_surface_area", "static_distribution", 100, 500,
             rationale="SASA distributions and residue summaries require more than a small surface-calculation pilot"),
    _profile("radial_distribution_functions", "normalized_pair_distribution", 200, 1_000,
             rationale="normalized shell counts require broad cell-volume and pair-distance sampling"),
    _profile("optional_observables", "question_defined_distribution", 200, 1_000, events=20,
             rationale="distance, contact, and native-contact questions require explicit standard coverage"),
    _profile("trajectory_features", "feature_time_series", 200, 1_000,
             temporal_rule="uniform_ensemble_features; temporal consumers impose their own spacing", rationale="downstream distributions and states inherit this feature coverage"),
]


def _inherited(
    module_id: str,
    source: str,
    sampling_class: str,
    *,
    per_replica: int = 200,
    per_system: int = 1_000,
    maximum_spacing_ns: float = 0.0,
    contiguous: bool = False,
    temporal_rule: str = "inherit_uniform_upstream_frames",
    events: int = 0,
    independent_units: int = 0,
    rationale: str,
) -> ScientificSamplingProfile:
    return _profile(
        module_id, sampling_class, per_replica, per_system,
        maximum_spacing_ns=maximum_spacing_ns,
        contiguous=contiguous, temporal_rule=temporal_rule,
        events=events,
        independent_units=independent_units, inherited_from=source,
        rationale=rationale,
    )


_PROFILES.extend([
    _inherited("generalized_correlation_and_information", "common_pca", "nonlinear_dependence", per_replica=250, per_system=1_000, rationale="nonlinear dependence estimates inherit the shared feature sample"),
    _inherited("information_dynamics", "common_pca", "lagged_information_dynamics", per_replica=500, per_system=2_000, maximum_spacing_ns=0.5, contiguous=True, temporal_rule="segment_contiguous_lag_pairs_with_stride_sensitivity", events=100, rationale="lagged information estimates require contiguous, segment-safe pairs and sensitivity to bins and lag"),
    _inherited("correlation_networks", "dccm", "derived_correlation_network", per_replica=250, per_system=1_000, rationale="network edges inherit the complete DCCM sampling gate"),
    _inherited("time_lagged_independent_component_analysis", "common_pca", "time_lagged_basis", per_replica=500, per_system=2_000, maximum_spacing_ns=0.5, contiguous=True, temporal_rule="segment_contiguous_lag_pairs_with_lag_sensitivity", events=100, rationale="tICA requires ordered lag pairs and cannot use disconnected sparse observations"),
    _inherited("pca_fes_basins", "common_pca", "free_energy_surface", per_replica=250, per_system=1_000, rationale="density surfaces and basin populations require broad balanced projected coverage"),
    _inherited("clustering_kmeans", "common_pca", "partition_clustering", per_replica=250, per_system=1_000, events=20, rationale="cluster selection and populations require enough observations and members per reported cluster"),
    _inherited("clustering_hdbscan", "common_pca", "density_clustering", per_replica=250, per_system=1_000, events=20, rationale="density clusters and noise fractions require a sufficiently populated feature sample"),
    _inherited("clustering_imwkmeans", "common_pca", "partition_clustering", per_replica=250, per_system=1_000, events=20, rationale="weighted partition fitting and populations require a populated feature sample"),
    _inherited("alternative_clustering", "common_pca", "algorithm_specific_clustering", per_replica=250, per_system=1_000, events=20, rationale="each clustering family retains its separate fit floor and complete-assignment contract"),
    _inherited("pald_community_analysis", "common_pca", "bounded_community_sample", per_replica=20, per_system=100, events=10, rationale="the cubic bounded sample still needs enough observations to define communities"),
    _inherited("representative_frames", "common_pca", "state_representatives", per_replica=250, per_system=1_000, events=20, rationale="representatives inherit state definitions and require adequate observations in every exported state"),
    _inherited("state_coordinate_exports", "common_pca", "state_export", per_replica=250, per_system=1_000, events=1, rationale="exports inherit accepted state assignments and always retain representative structures"),
    _inherited("representative_structures", "common_pca", "state_representatives", per_replica=250, per_system=1_000, events=20, rationale="means, medoids, and central structures inherit adequately populated aligned states"),
    _inherited("markov_state_models", "common_pca", "transition_model", per_replica=500, per_system=2_000, maximum_spacing_ns=0.5, contiguous=True, temporal_rule="segment_contiguous_transition_counts_with_lag_sensitivity", events=100, rationale="MSMs require ordered state sequences, connected counts, and lag-time validation"),
    _inherited("scalar_feature_distributions", "trajectory_features", "static_distribution", per_replica=200, per_system=1_000, rationale="automatic histograms require an adequately sampled upstream scalar series"),
    _inherited("scalar_threshold_states", "trajectory_features", "threshold_state_series", per_replica=250, per_system=1_000, contiguous=True, temporal_rule="segment_contiguous_state_runs_with_stride_sensitivity", events=50, rationale="state populations, transitions, and residence runs require ordered segment-safe series; temporal resolution is reported from the configured stride"),
    _inherited("hydrogen_bond_patterns", "hydrogen_bond_discovery", "contact_pattern_clustering", per_replica=200, per_system=1_000, events=20, rationale="pattern clusters inherit hydrogen-bond coverage and need populated patterns"),
    _inherited("hydrogen_bond_comparison", "hydrogen_bond_discovery", "matched_contact_comparison", per_replica=200, per_system=1_000, events=20, rationale="each compared system must independently meet the upstream occupancy floor"),
    _inherited("grouped_ml", "common_pca", "grouped_predictive_validation", per_replica=250, per_system=1_000, independent_units=5, rationale="held-out validation requires complete independent groups rather than random frame splits"),
    _inherited("grouped_regularized_classification", "hydrogen_bond_discovery", "grouped_predictive_validation", per_replica=200, per_system=1_000, independent_units=2, rationale="each class requires multiple independent held-out groups"),
    _inherited("convergence_uncertainty", "replica_rmsd_rg", "autocorrelation_and_uncertainty", per_replica=250, per_system=250, contiguous=True, temporal_rule="ordered_series_for_uncertainty_blocks", rationale="uncertainty diagnostics require ordered per-replica series; their selected spacing and physical span are reported rather than compared with a universal duration gate"),
    _inherited("rmsf_permutation_inference", "pooled_rmsf", "independent_unit_inference", per_replica=200, per_system=1_000, independent_units=2, rationale="permutation units are independent replicas or justified blocks, never individual frames"),
])


_BY_MODULE = {profile.module_id: profile for profile in _PROFILES}
if len(_BY_MODULE) != len(_PROFILES):
    raise RuntimeError("duplicate scientific sampling module IDs")


def list_scientific_sampling_profiles() -> list[ScientificSamplingProfile]:
    return list(_PROFILES)


def scientific_sampling_profile(module_id: str) -> ScientificSamplingProfile:
    try:
        return _BY_MODULE[module_id]
    except KeyError as exc:
        raise ScientificSamplingError(
            f"no scientific sampling profile for module {module_id!r}"
        ) from exc


def profile_contract(
    profile: ScientificSamplingProfile, *, policy_id: str = POLICY_ID
) -> Dict[str, object]:
    return {"policy_id": policy_id, **asdict(profile)}


def scientific_minimums_document() -> Dict[str, object]:
    """Return the complete editable sampling-minimums document."""

    return {
        "minimums_schema": MINIMUMS_SCHEMA,
        "base_policy_id": POLICY_ID,
        "interpretation": {
            "minimum_frames_per_replica": (
                "Minimum retained physical frames in each simulation replica."
            ),
            "minimum_frames_overall_per_system": (
                "Minimum retained physical frames pooled across replicas of one "
                "system; this is not a multi-system campaign total."
            ),
            "maximum_time_gap_between_retained_frames_ns": (
                "Largest allowed time gap between retained frames for an ordered "
                "method. Zero means the method is treated as a static ensemble "
                "estimator and has no time-gap gate."
            ),
        },
        "override_policy": (
            "Values may keep or strengthen the packaged standard. Frame minima "
            "may increase. A positive time-gap maximum may decrease; a method "
            "with no packaged time-gap gate may be given a positive one."
        ),
        "methods": {
            profile.module_id: {
                "minimum_frames_per_replica": profile.minimum_frames_per_replica,
                "minimum_frames_overall_per_system": (
                    profile.minimum_frames_per_system
                ),
                "maximum_time_gap_between_retained_frames_ns": (
                    profile.maximum_uniform_spacing_ns
                ),
            }
            for profile in sorted(_PROFILES, key=lambda value: value.module_id)
        },
    }


def load_scientific_minimums(
    path: Optional[Path],
) -> Dict[str, object]:
    """Load a standard or stricter user sampling policy with provenance."""

    if path is None:
        return {
            "policy_id": POLICY_ID,
            "source_path": None,
            "source_sha256": None,
            "profiles": dict(_BY_MODULE),
        }
    source = Path(path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScientificSamplingError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise ScientificSamplingError("scientific minimums file must be an object")
    allowed_top = {
        "minimums_schema", "base_policy_id", "interpretation",
        "override_policy", "methods",
    }
    unknown_top = sorted(set(payload).difference(allowed_top))
    if unknown_top:
        raise ScientificSamplingError(
            "scientific minimums file has unknown fields: "
            + ", ".join(unknown_top)
        )
    if payload.get("minimums_schema") != MINIMUMS_SCHEMA:
        raise ScientificSamplingError(
            f"minimums_schema must be {MINIMUMS_SCHEMA}"
        )
    if payload.get("base_policy_id") != POLICY_ID:
        raise ScientificSamplingError(
            f"base_policy_id must be the current {POLICY_ID}"
        )
    methods = payload.get("methods")
    if not isinstance(methods, dict):
        raise ScientificSamplingError("scientific minimums methods must be an object")
    unknown_methods = sorted(set(methods).difference(_BY_MODULE))
    if unknown_methods:
        raise ScientificSamplingError(
            "scientific minimums file names unknown methods: "
            + ", ".join(unknown_methods)
        )
    profiles = dict(_BY_MODULE)
    allowed_fields = {
        "minimum_frames_per_replica", "minimum_frames_overall_per_system",
        "maximum_time_gap_between_retained_frames_ns",
    }
    for module_id, raw in methods.items():
        if not isinstance(raw, dict) or set(raw).difference(allowed_fields):
            raise ScientificSamplingError(
                f"scientific minimums for {module_id} are invalid"
            )
        base = _BY_MODULE[module_id]
        per_replica = raw.get(
            "minimum_frames_per_replica", base.minimum_frames_per_replica
        )
        per_system = raw.get(
            "minimum_frames_overall_per_system", base.minimum_frames_per_system
        )
        gap = raw.get(
            "maximum_time_gap_between_retained_frames_ns",
            base.maximum_uniform_spacing_ns,
        )
        for value, label, floor in (
            (per_replica, "minimum_frames_per_replica", base.minimum_frames_per_replica),
            (
                per_system, "minimum_frames_overall_per_system",
                base.minimum_frames_per_system,
            ),
        ):
            if (
                isinstance(value, bool) or not isinstance(value, int)
                or value < floor
            ):
                raise ScientificSamplingError(
                    f"{module_id}.{label} must be an integer at least {floor}"
                )
        if (
            isinstance(gap, bool) or not isinstance(gap, (int, float))
            or not math.isfinite(float(gap)) or float(gap) < 0.0
        ):
            raise ScientificSamplingError(
                f"{module_id}.maximum_time_gap_between_retained_frames_ns must "
                "be finite and nonnegative"
            )
        if (
            base.maximum_uniform_spacing_ns > 0.0
            and (
                float(gap) <= 0.0
                or float(gap) > base.maximum_uniform_spacing_ns + 1.0e-12
            )
        ):
            raise ScientificSamplingError(
                f"{module_id}.maximum_time_gap_between_retained_frames_ns may not "
                "exceed the "
                f"packaged {base.maximum_uniform_spacing_ns:g} ns gate"
            )
        profiles[module_id] = replace(
            base,
            minimum_frames_per_replica=int(per_replica),
            minimum_frames_per_system=int(per_system),
            maximum_uniform_spacing_ns=float(gap),
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "policy_id": f"{POLICY_ID}+strict-{digest[:12]}",
        "source_path": str(source),
        "source_sha256": digest,
        "profiles": profiles,
    }


def profile_from_contract(contract: Mapping[str, object]) -> ScientificSamplingProfile:
    """Reconstruct a profile embedded in a planner task or report."""

    fields = set(ScientificSamplingProfile.__dataclass_fields__)
    missing = sorted(fields.difference(contract))
    if missing:
        raise ScientificSamplingError(
            "scientific sampling contract is missing: " + ", ".join(missing)
        )
    return ScientificSamplingProfile(**{
        field: contract[field] for field in fields
    })


def apply_scientific_minimums_to_tasks(
    tasks: Sequence[Mapping[str, object]], policy: Mapping[str, object]
) -> list[Dict[str, object]]:
    """Raise task floors to a resolved user policy without mutating inputs."""

    profiles = policy.get("profiles")
    policy_id = policy.get("policy_id")
    if not isinstance(profiles, Mapping) or not isinstance(policy_id, str):
        raise ScientificSamplingError("resolved scientific minimums policy is invalid")
    output: list[Dict[str, object]] = []
    for raw in tasks:
        row = dict(raw)
        module_id = str(row.get("module_id", ""))
        profile = profiles.get(module_id)
        if not isinstance(profile, ScientificSamplingProfile):
            output.append(row)
            continue
        source = row.get("source_frames_per_replica")
        if not isinstance(source, (list, tuple)) or not source:
            raise ScientificSamplingError(
                f"task {row.get('task_id')} has no source frame counts"
            )
        counts = [int(value) for value in source]
        system_ids = row.get("system_ids_per_replica")
        ids = (
            [str(value) for value in system_ids]
            if isinstance(system_ids, list) else None
        )
        intervals = row.get("frame_intervals_ns_per_replica")
        spans = row.get("source_time_spans_ns_per_replica")
        timing_available = isinstance(intervals, list) and isinstance(spans, list)
        required = required_frames_per_replica(
            profile,
            system_ids_per_replica=ids,
            replica_count=None if ids is not None else len(counts),
            source_frames_per_replica=counts if timing_available else None,
            frame_intervals_ns_per_replica=(
                [float(value) for value in intervals]
                if timing_available else None
            ),
            source_time_spans_ns_per_replica=(
                [float(value) for value in spans]
                if timing_available else None
            ),
        )
        existing_scientific = row.get("scientific_minimum_frames_per_replica")
        if (
            isinstance(existing_scientific, int)
            and not isinstance(existing_scientific, bool)
            and existing_scientific > 0
        ):
            required = max(required, existing_scientific)
        attainable = min(required, min(counts))
        existing = int(row.get("minimum_frames_per_replica", 1))
        row["minimum_frames_per_replica"] = max(existing, attainable)
        row["maximum_frames_per_replica"] = max(
            int(row.get("maximum_frames_per_replica", max(counts))),
            int(row["minimum_frames_per_replica"]),
        )
        row["scientific_minimum_frames_per_replica"] = required
        row["attainable_scientific_minimum_frames_per_replica"] = attainable
        row["scientific_sampling_requirements"] = profile_contract(
            profile, policy_id=policy_id
        )
        row["scientific_minimums_source_path"] = policy.get("source_path")
        row["scientific_minimums_source_sha256"] = policy.get("source_sha256")
        output.append(row)
    return output


def required_frames_per_replica(
    profile: ScientificSamplingProfile,
    *,
    system_ids_per_replica: Optional[Sequence[str]] = None,
    replica_count: Optional[int] = None,
    source_frames_per_replica: Optional[Sequence[int]] = None,
    frame_intervals_ns_per_replica: Optional[Sequence[float]] = None,
    source_time_spans_ns_per_replica: Optional[Sequence[float]] = None,
) -> int:
    """Return the per-replica count and applicable temporal-spacing floor.

    For an ordered method with a maximum allowed spacing, timing-aware planning
    converts that spacing to an exact integer-stride frame floor. Supplied
    trajectory duration is provenance, not an acceptance gate. No
    autocorrelation or event-rate estimate is used.
    """

    if profile.minimum_frames_per_replica == 0:
        return 0
    if system_ids_per_replica is not None:
        if not system_ids_per_replica:
            raise ScientificSamplingError("system_ids_per_replica must not be empty")
        counts: Dict[str, int] = {}
        for value in system_ids_per_replica:
            if not isinstance(value, str) or not value:
                raise ScientificSamplingError(
                    "system_ids_per_replica must contain nonempty strings"
                )
            counts[value] = counts.get(value, 0) + 1
        smallest_system_replica_count = min(counts.values())
    else:
        if (
            isinstance(replica_count, bool)
            or not isinstance(replica_count, int)
            or replica_count <= 0
        ):
            raise ScientificSamplingError(
                "a positive replica_count is required without system IDs"
            )
        smallest_system_replica_count = replica_count
    system_floor = math.ceil(
        profile.minimum_frames_per_system / smallest_system_replica_count
    )
    required = max(profile.minimum_frames_per_replica, system_floor)
    timing_values = (
        source_frames_per_replica,
        frame_intervals_ns_per_replica,
        source_time_spans_ns_per_replica,
    )
    if all(value is None for value in timing_values):
        return required
    if any(value is None for value in timing_values):
        raise ScientificSamplingError(
            "source counts, frame intervals, and time spans must be supplied together"
        )
    source_counts = [int(value) for value in source_frames_per_replica or ()]
    intervals = [float(value) for value in frame_intervals_ns_per_replica or ()]
    spans = [float(value) for value in source_time_spans_ns_per_replica or ()]
    if not source_counts or not (
        len(source_counts) == len(intervals) == len(spans)
    ):
        raise ScientificSamplingError(
            "timing arrays must have equal nonzero lengths"
        )
    if any(value <= 0 for value in source_counts + intervals) or any(
        value < 0 for value in spans
    ):
        raise ScientificSamplingError("timing arrays contain invalid values")
    if profile.maximum_uniform_spacing_ns > 0.0:
        for count, interval in zip(source_counts, intervals):
            maximum_integer_stride = math.floor(
                (profile.maximum_uniform_spacing_ns + 1.0e-12) / interval
            )
            spacing_floor = (
                count + 1 if maximum_integer_stride <= 0 else
                math.ceil(count / maximum_integer_stride)
            )
            required = max(required, spacing_floor)
    return required


def assess_raw_sampling(
    profile: ScientificSamplingProfile,
    *,
    selected_frames_per_replica: Sequence[int],
    source_frames_per_replica: Sequence[int],
    system_ids_per_replica: Optional[Sequence[str]] = None,
    integer_stride: int = 1,
    frame_intervals_ns_per_replica: Optional[Sequence[float]] = None,
    source_time_spans_ns_per_replica: Optional[Sequence[float]] = None,
    policy_id: str = POLICY_ID,
) -> Dict[str, object]:
    """Assess count floors and applicable ordered-method spacing gates.

    Selected coverage and physical duration are reported as provenance and do
    not determine whether a short supplied trajectory may run. Event and
    transition counts require the resulting observable and are returned as
    post-run diagnostics, not estimated by the planner.
    """

    selected = [int(value) for value in selected_frames_per_replica]
    source = [int(value) for value in source_frames_per_replica]
    if not selected or len(selected) != len(source):
        raise ScientificSamplingError(
            "selected and source frame counts must have equal nonzero lengths"
        )
    if any(value <= 0 for value in selected + source) or any(
        chosen > available for chosen, available in zip(selected, source)
    ):
        raise ScientificSamplingError("frame counts are invalid")
    if integer_stride <= 0:
        raise ScientificSamplingError("integer_stride must be positive")
    if profile.minimum_frames_per_replica == 0:
        return {
            "policy_id": policy_id,
            "raw_coverage_status": "not_applicable",
            "keep_enabled": True,
            "scientific_interpretation_ready": True,
            "requirements": profile_contract(profile, policy_id=policy_id),
        }
    ids = (
        list(system_ids_per_replica)
        if system_ids_per_replica is not None
        else ["pooled_system"] * len(selected)
    )
    if len(ids) != len(selected):
        raise ScientificSamplingError(
            "system_ids_per_replica must match the replica counts"
        )
    intervals = None
    source_spans = None
    if (
        frame_intervals_ns_per_replica is not None
        or source_time_spans_ns_per_replica is not None
    ):
        if (
            frame_intervals_ns_per_replica is None
            or source_time_spans_ns_per_replica is None
        ):
            raise ScientificSamplingError(
                "frame intervals and source time spans must be supplied together"
            )
        intervals = [float(value) for value in frame_intervals_ns_per_replica]
        source_spans = [float(value) for value in source_time_spans_ns_per_replica]
        if len(intervals) != len(selected) or len(source_spans) != len(selected):
            raise ScientificSamplingError(
                "timing arrays must match the replica counts"
            )
        if any(value <= 0.0 for value in intervals) or any(
            value < 0.0 for value in source_spans
        ):
            raise ScientificSamplingError("timing arrays contain invalid values")
    required_per_replica = required_frames_per_replica(
        profile,
        system_ids_per_replica=ids,
        source_frames_per_replica=source if intervals is not None else None,
        frame_intervals_ns_per_replica=intervals,
        source_time_spans_ns_per_replica=source_spans,
    )
    per_system: Dict[str, int] = {}
    for system_id, count in zip(ids, selected):
        per_system[system_id] = per_system.get(system_id, 0) + count
    replica_failures = [
        index for index, count in enumerate(selected)
        if count < min(required_per_replica, source[index])
    ]
    system_failures = [
        system_id for system_id, count in per_system.items()
        if count < min(
            profile.minimum_frames_per_system,
            sum(
                available for current, available in zip(ids, source)
                if current == system_id
            ),
        )
    ]
    span_fractions = []
    for chosen, available in zip(selected, source):
        if available <= 1 or chosen == available:
            span_fractions.append(1.0)
        else:
            last_index = min(available - 1, (chosen - 1) * integer_stride)
            span_fractions.append(last_index / (available - 1))
    selected_spans_ns = None
    selected_spacings_ns = None
    temporal_spacing_failures = []
    source_temporal_resolution_failures = []
    if intervals is not None and source_spans is not None:
        selected_spans_ns = []
        selected_spacings_ns = []
        for index, (chosen, available, interval, source_span) in enumerate(
            zip(selected, source, intervals, source_spans)
        ):
            last_index = min(available - 1, max(0, (chosen - 1) * integer_stride))
            observed_span = min(source_span, last_index * interval)
            observed_spacing = interval if chosen == available else interval * integer_stride
            selected_spans_ns.append(observed_span)
            selected_spacings_ns.append(observed_spacing)
            if (
                profile.maximum_uniform_spacing_ns > 0.0
                and observed_spacing - 1.0e-12
                > profile.maximum_uniform_spacing_ns
            ):
                temporal_spacing_failures.append(index)
            if (
                profile.maximum_uniform_spacing_ns > 0.0
                and interval - 1.0e-12 > profile.maximum_uniform_spacing_ns
            ):
                source_temporal_resolution_failures.append(index)
    source_limited = any(
        available < required_per_replica
        for available in source
    ) or any(
        sum(
            available for current, available in zip(ids, source)
            if current == system_id
        ) < profile.minimum_frames_per_system
        for system_id in set(ids)
    ) or bool(source_temporal_resolution_failures)
    keep = not (
        replica_failures or system_failures or temporal_spacing_failures
    )
    return {
        "policy_id": policy_id,
        "raw_coverage_status": (
            "source_limited_below_standard" if source_limited else
            "meets_standard_raw_floor" if keep else
            "resource_limited_below_standard"
        ),
        "keep_enabled": keep,
        "scientific_interpretation_ready": (
            keep
            and not source_limited
            and not profile.requires_contiguous_frames
        ),
        "required_frames_per_replica": required_per_replica,
        "sampling_floor_basis": (
            "minimum_samples_and_maximum_temporal_separation"
            if profile.minimum_frames_per_replica > 0
            and profile.maximum_uniform_spacing_ns > 0.0 else
            "minimum_samples"
            if profile.minimum_frames_per_replica > 0 else
            "maximum_temporal_separation"
        ),
        "selected_frames_per_system": per_system,
        "minimum_observed_time_span_fraction": min(span_fractions),
        "time_span_fraction_is_acceptance_gate": False,
        "physical_time_span_is_acceptance_gate": False,
        "replica_floor_failures": replica_failures,
        "system_floor_failures": sorted(system_failures),
        "timing_metadata_status": (
            "available" if intervals is not None else "not_supplied"
        ),
        "source_time_spans_ns_per_replica": source_spans,
        "selected_time_spans_ns_per_replica": selected_spans_ns,
        "selected_temporal_spacings_ns_per_replica": selected_spacings_ns,
        "temporal_spacing_failures": temporal_spacing_failures,
        "source_temporal_resolution_failures": (
            source_temporal_resolution_failures
        ),
        "planner_estimates_autocorrelation_or_event_rates": False,
        "postrun_event_or_transition_diagnostic": (
            profile.minimum_reported_events_or_transitions
            if profile.minimum_reported_events_or_transitions else None
        ),
        "temporal_resolution_validation_required": (
            profile.requires_contiguous_frames
            or profile.maximum_uniform_spacing_ns > 0.0
        ),
        "requirements": profile_contract(profile, policy_id=policy_id),
    }
