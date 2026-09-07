"""Explicit campaign-local static-ensemble execution policy.

The frozen site profile exports this flag only for discontinuous retained-frame
inputs. It changes neither frame selection nor static estimators.
"""
import os


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
