"""Replica/block-level permutation inference for RMSF differences."""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


class RMSFInferenceError(ValueError):
    """Raised when RMSF inference units are invalid."""


def rmsf_permutation_test(
    group_a_profiles: Sequence[Sequence[float]],
    group_b_profiles: Sequence[Sequence[float]],
    permutations: int = 9999,
    random_seed: int = 0,
    exact_partition_limit: int = 100000,
) -> Dict[str, object]:
    """Compare independent replica/block RMSF profiles with pointwise and max-T p-values.

    Rows, not trajectory frames, are the exchangeable units.  Exact enumeration
    is used when the number of unique label partitions is within the declared
    resource limit; otherwise a seeded Monte Carlo test is used.
    """

    first = np.asarray(group_a_profiles, dtype=float)
    second = np.asarray(group_b_profiles, dtype=float)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1:] != second.shape[1:]:
        raise RMSFInferenceError("groups must be two-dimensional with the same feature count")
    if first.shape[0] < 2 or second.shape[0] < 2:
        raise RMSFInferenceError("each group requires at least two independent profiles")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise RMSFInferenceError("RMSF profiles must be finite")
    if permutations < 1 or exact_partition_limit < 1:
        raise RMSFInferenceError("permutation and exact-partition limits must be positive")
    combined = np.vstack([first, second])
    first_count = first.shape[0]
    total_count = combined.shape[0]
    observed = first.mean(axis=0) - second.mean(axis=0)
    partition_count = math.comb(total_count, first_count)
    if partition_count <= exact_partition_limit:
        partitions = itertools.combinations(range(total_count), first_count)
        method = "exact"
    else:
        rng = np.random.default_rng(int(random_seed))
        partitions = (
            tuple(sorted(rng.choice(total_count, size=first_count, replace=False).tolist()))
            for _ in range(int(permutations))
        )
        method = "monte_carlo"
    exceed = np.zeros(observed.shape, dtype=int)
    exceed_max = np.zeros(observed.shape, dtype=int)
    evaluated = 0
    all_indices = np.arange(total_count)
    for group_indices in partitions:
        mask = np.zeros(total_count, dtype=bool)
        mask[list(group_indices)] = True
        difference = combined[mask].mean(axis=0) - combined[~mask].mean(axis=0)
        absolute = np.abs(difference)
        exceed += absolute >= np.abs(observed) - 1.0e-15
        exceed_max += np.max(absolute) >= np.abs(observed) - 1.0e-15
        evaluated += 1
    correction = 0 if method == "exact" else 1
    denominator = evaluated + correction
    return {
        "method": method,
        "exchangeable_unit": "profile_row",
        "group_a_unit_count": int(first.shape[0]),
        "group_b_unit_count": int(second.shape[0]),
        "feature_count": int(first.shape[1]),
        "observed_mean_difference": observed.tolist(),
        "two_sided_pointwise_p_values": ((exceed + correction) / denominator).tolist(),
        "max_t_familywise_p_values": ((exceed_max + correction) / denominator).tolist(),
        "evaluated_partition_count": evaluated,
        "possible_partition_count": partition_count,
        "random_seed": int(random_seed) if method == "monte_carlo" else None,
    }


def _replica_profiles(
    system: Mapping[str, object],
) -> Tuple[List[str], List[List[float]], List[Tuple[object, ...]]]:
    replicas = system.get("replicas")
    if not isinstance(replicas, list):
        raise RMSFInferenceError("RMSF system report has no replica list")
    replica_ids: List[str] = []
    profiles: List[List[float]] = []
    identity: List[Tuple[object, ...]] = []
    for replica in replicas:
        if not isinstance(replica, dict) or replica.get("technical_status") != "complete":
            continue
        rows = replica.get("atom_statistics")
        if not isinstance(rows, list) or not rows:
            continue
        ordered = sorted(
            (row for row in rows if isinstance(row, dict)),
            key=lambda row: int(row.get("common_atom_index", -1)),
        )
        current_identity = [
            (
                row.get("common_atom_index"), row.get("chain_id"),
                row.get("residue_id"), row.get("insertion_code"),
                row.get("residue_name"), row.get("atom_name"),
            )
            for row in ordered
        ]
        try:
            profile = [float(row["rmsf_angstrom"]) for row in ordered]
        except (KeyError, TypeError, ValueError) as exc:
            raise RMSFInferenceError(
                "replica RMSF rows lack finite rmsf_angstrom values"
            ) from exc
        if not all(math.isfinite(value) for value in profile):
            raise RMSFInferenceError("replica RMSF profiles contain nonfinite values")
        if identity and current_identity != identity:
            raise RMSFInferenceError(
                "replica RMSF atom identities are not identical across systems"
            )
        identity = current_identity
        replica_ids.append(str(replica.get("replica_id")))
        profiles.append(profile)
    return replica_ids, profiles, identity


def rmsf_replica_permutation_comparisons(
    rmsf_report: Mapping[str, object],
    comparison_policy: Mapping[str, object],
    permutations: int = 9999,
    random_seed: int = 0,
    exact_partition_limit: int = 100000,
) -> Dict[str, object]:
    """Infer safe system comparisons from independently declared replicas.

    Each complete simulation replica contributes one RMSF profile. Trajectory
    frames and symmetry-related oligomer members never become exchangeable units.
    """

    if rmsf_report.get("technical_status") != "complete":
        raise RMSFInferenceError("pooled RMSF report is not technically complete")
    systems = rmsf_report.get("systems")
    if not isinstance(systems, list) or len(systems) < 2:
        raise RMSFInferenceError("RMSF comparison requires at least two systems")
    profiles: Dict[str, List[List[float]]] = {}
    replica_ids: Dict[str, List[str]] = {}
    common_identity: List[Tuple[object, ...]] = []
    for raw_system in systems:
        if not isinstance(raw_system, dict):
            raise RMSFInferenceError("RMSF system entries must be objects")
        system_id = str(raw_system.get("system_id"))
        ids, rows, identity = _replica_profiles(raw_system)
        if common_identity and identity != common_identity:
            raise RMSFInferenceError(
                "system RMSF reports do not share one atom-identity mapping"
            )
        if identity:
            common_identity = identity
        profiles[system_id] = rows
        replica_ids[system_id] = ids
    mode = str(comparison_policy.get("mode", "all_pairs"))
    system_ids = list(profiles)
    if mode == "all_pairs":
        pairs = list(itertools.combinations(system_ids, 2))
    elif mode == "reference_vs_all":
        reference = str(comparison_policy.get("reference_system_id"))
        if reference not in profiles:
            raise RMSFInferenceError(
                "reference_vs_all RMSF comparison names an unknown reference system"
            )
        pairs = [(reference, candidate) for candidate in system_ids if candidate != reference]
    else:
        raise RMSFInferenceError(
            "RMSF comparison mode must be all_pairs or reference_vs_all"
        )
    comparisons = []
    issues = []
    for group_a, group_b in pairs:
        if len(profiles[group_a]) < 2 or len(profiles[group_b]) < 2:
            comparisons.append({
                "system_a": group_a,
                "system_b": group_b,
                "comparison_status": "insufficient_independent_replicas",
                "system_a_replica_count": len(profiles[group_a]),
                "system_b_replica_count": len(profiles[group_b]),
                "minimum_required_per_system": 2,
            })
            issues.append({
                "severity": "warning",
                "code": "RMSF_PERMUTATION_INSUFFICIENT_REPLICAS",
                "location": f"{group_a} versus {group_b}",
                "message": "each system needs at least two independent simulation replicas",
            })
            continue
        result = rmsf_permutation_test(
            profiles[group_a], profiles[group_b], permutations=permutations,
            random_seed=random_seed, exact_partition_limit=exact_partition_limit,
        )
        comparisons.append({
            "system_a": group_a,
            "system_b": group_b,
            "comparison_status": "complete",
            "system_a_replica_ids": replica_ids[group_a],
            "system_b_replica_ids": replica_ids[group_b],
            "result": result,
        })
    return {
        "module_id": "rmsf_permutation_inference",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "exchangeable_unit": "independently_declared_simulation_replica",
        "exchangeability_inference": (
            "one unit per declared simulation replica; trajectory frames, time "
            "blocks, and oligomer members are not counted as independent units"
        ),
        "comparison_mode": mode,
        "atom_count": len(common_identity),
        "comparisons": comparisons,
        "error_count": 0,
        "warning_count": len(issues),
        "issues": issues,
    }
