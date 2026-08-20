"""Replica/block-level permutation inference for RMSF differences."""

from __future__ import annotations

import itertools
import math
from typing import Dict, Sequence

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
