"""Execution policy for explicitly configured discontinuous ensembles."""
import os


STATIC_DISABLED_MODULES = frozenset({
    "convergence_uncertainty", "information_dynamics", "markov_state_models",
    "scalar_threshold_states", "time_lagged_independent_component_analysis",
    "grouped_ml", "random_feature_koopman", "reactive_path_ensembles",
    "interaction_persistence", "spatial_interaction_ensembles",
})


def validate_trajectory_mode(value: object) -> str:
    if not isinstance(value, str) or value not in {"continuous", "static_ensemble"}:
        raise ValueError("execution.trajectory_mode must be continuous or static_ensemble")
    return value


def static_ensemble_enabled() -> bool:
    value = os.environ.get("SALSBURY_STATIC_ENSEMBLE", "0")
    if value not in {"0", "1"}:
        raise ValueError("SALSBURY_STATIC_ENSEMBLE must be 0 or 1")
    return value == "1"


def temporal_output_policy() -> str:
    return (
        "disabled: discontinuous static ensemble"
        if static_ensemble_enabled() else "ordered segment-local summaries"
    )
