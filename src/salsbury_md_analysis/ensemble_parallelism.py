"""Scientific contracts for parallel work across trajectory replicas.

Replica identity is an execution boundary, not automatically an estimator
boundary.  The contracts in this module state what a replica worker may emit
and where the primary estimator must be finalized.  Schedulers can therefore
parallelize coordinate work without silently fitting incompatible per-replica
means, bases, state definitions, or comparison models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Tuple


class EnsembleParallelismError(ValueError):
    """Raised when a proposed task partition changes estimator semantics."""


@dataclass(frozen=True)
class EnsembleParallelismContract:
    module_id: str
    worker_output: str
    primary_estimator_scope: str
    replica_shard_may_finalize_primary_result: bool
    reduction_rule: str
    ordered_boundary_rule: str
    allowed_parallel_phases: Tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _contract(
    module_id: str,
    worker_output: str,
    primary_estimator_scope: str,
    replica_shard_may_finalize_primary_result: bool,
    reduction_rule: str,
    ordered_boundary_rule: str = "not_ordered",
    allowed_parallel_phases: Tuple[str, ...] = (),
) -> EnsembleParallelismContract:
    return EnsembleParallelismContract(
        module_id=module_id,
        worker_output=worker_output,
        primary_estimator_scope=primary_estimator_scope,
        replica_shard_may_finalize_primary_result=(
            replica_shard_may_finalize_primary_result
        ),
        reduction_rule=reduction_rule,
        ordered_boundary_rule=ordered_boundary_rule,
        allowed_parallel_phases=allowed_parallel_phases,
    )


_CONTEXT = {
    "provenance_manifest", "preflight_inventory", "common_atom_mapping",
}
_REPLICA_FINAL = {
    "structural_integrity_qc", "replica_rmsd_rg", "individual_pca",
    "convergence_uncertainty",
}
_SYSTEM_MOMENTS = {
    "pooled_rmsf", "dccm",
}
_FRAME_MAP_SYSTEM_REDUCE = {
    "dihedral_distributions", "hydrogen_bonds", "hydrogen_bond_discovery",
    "water_mediated_hydrogen_bond_networks", "secondary_structure",
    "nucleic_acid_structure", "nucleic_acid_geometry",
    "ion_coordination_geometry", "ion_atmosphere",
    "solvent_accessible_surface_area", "radial_distribution_functions",
    "optional_observables", "trajectory_features",
}
_POOLED_VIEW_FIT = {
    "common_pca", "generalized_correlation_and_information",
    "pca_fes_basins", "clustering_kmeans", "clustering_hdbscan",
    "clustering_imwkmeans", "alternative_clustering",
    "pald_community_analysis", "correlation_networks",
    "hydrogen_bond_patterns", "representative_structures",
}
_SEGMENT_SAFE_POOLED_FIT = {
    "time_lagged_independent_component_analysis", "markov_state_models",
    "information_dynamics", "scalar_feature_distributions",
    "scalar_threshold_states",
}
_COMPARISON_FIT = {
    "hydrogen_bond_comparison", "grouped_ml",
    "grouped_regularized_classification", "integrated_comparison",
    "rmsf_permutation_inference",
}
_STATE_EXPORT = {
    "representative_frames", "state_coordinate_exports",
}


def _build_contracts() -> Dict[str, EnsembleParallelismContract]:
    contracts: Dict[str, EnsembleParallelismContract] = {}
    for module_id in _CONTEXT:
        contracts[module_id] = _contract(
            module_id, "metadata_or_mapping_shard", "declared_project_context",
            False,
            "validate every shard, then finalize one project-level context",
            allowed_parallel_phases=("metadata_validation_shards",),
        )
    for module_id in _REPLICA_FINAL:
        contracts[module_id] = _contract(
            module_id, "complete_replica_result", "replica",
            True,
            "replica results remain distinct; any campaign summary is assembled "
            "without refitting a pooled primary estimator",
            (
                "preserve segment order within each replica"
                if module_id in {"replica_rmsd_rg", "convergence_uncertainty"}
                else "not_ordered"
            ),
            allowed_parallel_phases=("complete_replica_analysis",),
        )
    for module_id in _SYSTEM_MOMENTS:
        contracts[module_id] = _contract(
            module_id, "mergeable_sufficient_statistics",
            "system_all_declared_replicas", False,
            (
                "merge count, global mean, and centered second moments across "
                "every replica before calculating the frame-pooled result"
            ),
            allowed_parallel_phases=(
                "replica_sufficient_statistics",
                "replica_diagnostics_after_pooled_finalization",
            ),
        )
    for module_id in _FRAME_MAP_SYSTEM_REDUCE:
        contracts[module_id] = _contract(
            module_id, "frame_records_or_additive_statistics",
            "system_all_declared_replicas", False,
            (
                "concatenate identity-preserving frame records or merge additive "
                "statistics, then calculate system summaries from the combined data"
            ),
            (
                "preserve segment boundaries for residence or continuity quantities"
                if module_id in {"optional_observables", "trajectory_features"}
                else "not_ordered"
            ),
            allowed_parallel_phases=(
                "replica_frame_extraction", "replica_additive_statistics",
                "replica_reporting_after_system_reduction",
            ),
        )
    for module_id in _POOLED_VIEW_FIT:
        phases = (
            (
                "replica_sufficient_statistics_or_basis_samples",
                "replica_projection_after_global_basis",
            )
            if module_id == "common_pca" else
            (
                "replica_feature_extraction_before_global_regular_sample",
                "global_community_analysis_on_declared_pooled_sample",
            )
            if module_id == "pald_community_analysis" else
            (
                "replica_feature_extraction",
                "replica_assignment_after_global_fit_when_supported",
                "pooled_full_observation_fit_when_assignment_is_unsupported",
                "focal_validation_chunks_against_global_partition",
            )
            if module_id in {
                "clustering_kmeans", "clustering_hdbscan",
                "clustering_imwkmeans", "alternative_clustering",
                "hydrogen_bond_patterns",
            }
            else ("replica_feature_extraction", "global_model_postprocessing")
        )
        contracts[module_id] = _contract(
            module_id, "identity_preserving_features_or_sufficient_statistics",
            "declared_view_all_systems_and_replicas", False,
            (
                "assemble the declared pooled observation set before centering, "
                "standardization, basis construction, grid construction, or model fit"
            ),
            allowed_parallel_phases=phases,
        )
    for module_id in _SEGMENT_SAFE_POOLED_FIT:
        contracts[module_id] = _contract(
            module_id, "ordered_segment_statistics_or_assignments",
            "declared_view_all_systems_and_replicas", False,
            (
                "form lag pairs, transitions, or residence runs only within each "
                "declared segment, then pool their sufficient statistics for one fit"
            ),
            "never join replica, member, or discontinuous segment boundaries",
            allowed_parallel_phases=(
                "ordered_replica_or_segment_statistics",
                "global_reduce_after_boundary_safe_partials",
            ),
        )
    for module_id in _COMPARISON_FIT:
        contracts[module_id] = _contract(
            module_id, "system_or_independent_unit_records",
            "declared_comparison_all_systems", False,
            (
                "assemble every declared comparison system and preserve the declared "
                "independent-unit identity before inference or integrated ranking"
            ),
            allowed_parallel_phases=(
                "system_report_extraction", "independent_unit_statistics",
            ),
        )
    for module_id in _STATE_EXPORT:
        contracts[module_id] = _contract(
            module_id, "state_assignment_partition", "declared_state_and_system",
            False,
            (
                "consume one already fitted pooled state definition; parallel state "
                "or replica reads must not refit centers, basins, or representatives"
            ),
            allowed_parallel_phases=(
                "state_or_replica_coordinate_reads_after_global_state_fit",
            ),
        )
    contracts["random_feature_koopman"] = _contract(
        "random_feature_koopman", "ordered_replica_feature_statistics",
        "declared_view_all_systems_and_replicas", False,
        (
            "construct random features and lagged validation records within each "
            "continuous segment, then select and fit one pooled Koopman model"
        ),
        "never form lag pairs across replicas, members, or segment boundaries",
        allowed_parallel_phases=(
            "replica_random_feature_extraction",
            "ordered_replica_validation_statistics",
        ),
    )
    contracts["reactive_path_ensembles"] = _contract(
        "reactive_path_ensembles", "segment_safe_path_records",
        "declared_view_global_state_partition", False,
        (
            "consume one pooled state partition, extract paths within continuous "
            "segments, and compare or cluster route families only after pooling"
        ),
        "never join paths across replicas, members, or segment boundaries",
        allowed_parallel_phases=(
            "replica_path_extraction_after_global_state_fit",
        ),
    )
    contracts["helical_mechanics"] = _contract(
        "helical_mechanics", "identity_preserving_frame_geometry",
        "system_all_declared_replicas", False,
        (
            "pool mapped frame geometry across the system while preserving replica "
            "identity for summaries and uncertainty diagnostics"
        ),
        allowed_parallel_phases=("replica_geometry_extraction",),
    )
    contracts["perturbation_response_dynamics"] = _contract(
        "perturbation_response_dynamics", "replica_feature_covariance_statistics",
        "system_all_declared_replicas", False,
        (
            "merge common-basis score covariance statistics across every declared "
            "replica before constructing the response, DFI, or DCI estimator"
        ),
        allowed_parallel_phases=("replica_feature_covariance_statistics",),
    )
    contracts["trajectory_reweighting"] = _contract(
        "trajectory_reweighting", "weighted_replica_moment_statistics",
        "system_all_declared_replicas", False,
        (
            "normalize weights over the declared system and merge weighted common-"
            "basis moments before calculating the system estimator"
        ),
        allowed_parallel_phases=(
            "replica_identity_join", "replica_weighted_moment_statistics",
        ),
    )
    contracts["allosteric_pathways"] = _contract(
        "allosteric_pathways", "replica_contact_counts",
        "system_all_declared_replicas", False,
        (
            "sum contact counts and denominators across replicas before thresholding "
            "occupancy edges or calculating system pathways"
        ),
        allowed_parallel_phases=("replica_contact_counting",),
    )
    contracts["energetic_network_embeddings"] = _contract(
        "energetic_network_embeddings", "identity_preserving_frame_network",
        "declared_comparison_all_systems", False,
        (
            "retain frame and system identity through energy-network construction, "
            "then finalize embeddings and cross-system distances at declared scope"
        ),
        allowed_parallel_phases=("replica_frame_energy_networks",),
    )
    contracts["multivalent_molecular_bridges"] = _contract(
        "multivalent_molecular_bridges", "frame_hyperedges_and_ordered_runs",
        "system_all_declared_replicas", False,
        (
            "pool identity-preserving hyperedges for occupancy while forming "
            "residence events only within each continuous segment"
        ),
        "never join residence runs across replicas, members, or segment boundaries",
        allowed_parallel_phases=("replica_bridge_extraction",),
    )
    contracts["hydration_density_channels"] = _contract(
        "hydration_density_channels", "replica_voxel_counts",
        "system_all_declared_replicas", False,
        (
            "sum aligned voxel counts and frame denominators across replicas before "
            "defining system density components or comparison grids"
        ),
        allowed_parallel_phases=("replica_aligned_voxel_counting",),
    )
    contracts["ensemble_pocket_dynamics"] = _contract(
        "ensemble_pocket_dynamics", "replica_voxel_counts_and_frame_records",
        "system_all_declared_replicas", False,
        (
            "pool aligned voxel frequencies across replicas before defining recurrent "
            "system pockets and assigning frames to those shared regions"
        ),
        allowed_parallel_phases=("replica_pocket_voxel_counting",),
    )
    contracts["interaction_fingerprints"] = _contract(
        "interaction_fingerprints", "identity_preserving_sparse_frame_features",
        "system_all_declared_replicas", False,
        (
            "join upstream sources by exact frame identity, then pool source-aware "
            "counts across replicas for system fingerprints and co-occurrence"
        ),
        allowed_parallel_phases=("replica_exact_frame_joins",),
    )
    contracts["spatial_interaction_ensembles"] = _contract(
        "spatial_interaction_ensembles", "aligned_partner_coordinate_records",
        "declared_view_all_systems_and_replicas", False,
        (
            "assemble aligned partner-coordinate observations across the declared "
            "view before spatial partitioning or system comparison"
        ),
        allowed_parallel_phases=("replica_partner_coordinate_extraction",),
    )
    contracts["interaction_persistence"] = _contract(
        "interaction_persistence", "ordered_segment_event_records",
        "system_all_declared_replicas", False,
        (
            "form continuous and gap-tolerant events within each segment, then pool "
            "labeled event records for system summaries"
        ),
        "never join persistence events across replicas, members, or segments",
        allowed_parallel_phases=("ordered_replica_event_extraction",),
    )
    contracts["coordinate_cache"] = _contract(
        "coordinate_cache", "complete_replica_coordinate_cache", "replica",
        True,
        (
            "scan frames sequentially within each replica for continuous unwrapping; "
            "different replicas may be processed concurrently"
        ),
        "never carry unwrapping state across replicas or discontinuous segments",
        allowed_parallel_phases=("sequential_scan_within_each_replica",),
    )
    return contracts


CONTRACTS = _build_contracts()


def get_ensemble_parallelism_contract(
    module_id: str,
) -> EnsembleParallelismContract:
    try:
        return CONTRACTS[module_id]
    except KeyError as exc:
        raise EnsembleParallelismError(
            f"no ensemble-parallelism contract is registered for {module_id}"
        ) from exc


def annotate_task_parallelism(task: Mapping[str, object]) -> Dict[str, object]:
    """Return a task copy carrying its scientific parallelism contract.

    A task may declare ``replica_partition_id`` only when a replica shard is
    scientifically allowed to finalize the module's primary result.  Global
    estimators may still use internal replica workers, but those workers must
    return the contract's partial output to a single pooled finalizer.
    """

    result = dict(task)
    module_id = result.get("module_id")
    if not isinstance(module_id, str) or module_id not in CONTRACTS:
        return result
    contract = CONTRACTS[module_id]
    if (
        result.get("replica_partition_id") is not None
        and not contract.replica_shard_may_finalize_primary_result
    ):
        raise EnsembleParallelismError(
            f"{module_id} cannot be finalized as an independent replica task; "
            f"its primary estimator scope is {contract.primary_estimator_scope}"
        )
    result["ensemble_parallelism_contract"] = contract.as_dict()
    return result


def validate_registry_coverage(module_ids: set[str]) -> None:
    missing = sorted(module_ids.difference(CONTRACTS))
    unexpected = sorted(set(CONTRACTS).difference(module_ids | {"coordinate_cache"}))
    if missing or unexpected:
        parts = []
        if missing:
            parts.append("missing: " + ", ".join(missing))
        if unexpected:
            parts.append("unexpected: " + ", ".join(unexpected))
        raise EnsembleParallelismError("parallelism contract coverage mismatch; " + "; ".join(parts))
