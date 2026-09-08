"""Outer replica holdouts for caller-supplied complete fit/score pipelines."""

from __future__ import annotations

from copy import deepcopy
import math
from collections.abc import Mapping


def replica_pipeline_holdout(replica_data, fit_pipeline, score_pipeline):
    """Fit preprocessing, representation and estimator anew for every holdout.

    Keys must identify independent replicas (including system/condition where
    needed), not segments or oligomer members. The fit callback receives only
    training replicas. The score callback receives the fitted pipeline and test
    replicas. Callback closures must not import a globally fitted representation.
    """
    if not isinstance(replica_data, Mapping) or len(replica_data) < 2:
        raise ValueError(
            "at least two explicitly identified independent replicas are required"
        )
    if any(not isinstance(key, str) or not key.strip() for key in replica_data):
        raise ValueError("replica identities must be nonempty strings")
    reports = []
    keys = sorted(replica_data)
    for heldout in keys:
        training = {key: deepcopy(replica_data[key]) for key in keys if key != heldout}
        testing = {heldout: deepcopy(replica_data[heldout])}
        pipeline = fit_pipeline(training)
        score = score_pipeline(pipeline, testing)
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError("score_pipeline must return a finite scalar score")
        reports.append(
            {
                "training_replica_ids": list(training),
                "testing_replica_ids": [heldout],
                "score": float(score),
            }
        )
    return {
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "scope": "outer replica holdout of the caller-supplied complete fitting pipeline",
        "folds": reports,
        "mean_score": sum(row["score"] for row in reports) / len(reports),
        "acceptance_gate": False,
        "limitations": [
            "Training-only fitting is enforced at the callback boundary; caller-supplied closures and preprocessing must respect that boundary.",
            "Independent replica identity is declared by the caller; this result does not establish physical validity.",
        ],
    }
