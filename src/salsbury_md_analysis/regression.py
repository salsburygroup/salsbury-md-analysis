"""Hash-pinned analysis regression runner with an explicit approval boundary."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence, Union

from .dccm import dccm_project_safe
from .manifests import (
    ManifestValidationError,
    load_json,
    resolve_manifest_path,
    validate_regression,
)
from .pca import common_pca_project_safe, individual_pca_project_safe
from .provenance import stable_json_sha256
from .rmsd_rg import replica_rmsd_rg_project_safe
from .rmsf import pooled_rmsf_project_safe
from .structural_qc import structural_qc_project_safe


PathComponent = Union[str, int]
Runner = Callable[[Path, bool], Dict[str, object]]
_RUNNERS: Dict[str, Runner] = {
    "structural_integrity_qc": structural_qc_project_safe,
    "replica_rmsd_rg": replica_rmsd_rg_project_safe,
    "pooled_rmsf": pooled_rmsf_project_safe,
    "dccm": dccm_project_safe,
    "individual_pca": individual_pca_project_safe,
    "common_pca": common_pca_project_safe,
}


class RegressionError(ValueError):
    """Raised when a regression case cannot be evaluated safely."""


def _value_at(payload: object, path: Sequence[PathComponent]) -> object:
    value = payload
    for component in path:
        if isinstance(component, int) and not isinstance(component, bool):
            if not isinstance(value, list):
                raise RegressionError(
                    f"path component {component} requires an array, observed {type(value).__name__}"
                )
            try:
                value = value[component]
            except IndexError as exc:
                raise RegressionError(f"array index is out of range: {component}") from exc
        else:
            if not isinstance(component, str) or not isinstance(value, dict):
                raise RegressionError(
                    f"path component {component!r} requires an object"
                )
            if component not in value:
                raise RegressionError(f"report path is absent: {component!r}")
            value = value[component]
    return value


def _assertion_result(
    assertion: Mapping[str, object], report: Mapping[str, object]
) -> Dict[str, object]:
    raw_path = assertion["path"]
    assert isinstance(raw_path, list)
    path = tuple(raw_path)
    operator = str(assertion["operator"])
    expected = assertion["expected"]
    try:
        observed = _value_at(report, path)
        if operator == "equal":
            passed = observed == expected
        elif operator == "is_null":
            passed = observed is None and expected is None
        elif operator == "contains":
            if not isinstance(observed, (str, list, dict)):
                raise RegressionError("contains requires a string, array, or object")
            passed = expected in observed
        else:
            if (
                isinstance(observed, bool)
                or isinstance(expected, bool)
                or not isinstance(observed, (int, float))
                or not isinstance(expected, (int, float))
                or not math.isfinite(float(observed))
                or not math.isfinite(float(expected))
            ):
                raise RegressionError("close requires two finite numeric values")
            tolerance = float(assertion["absolute_tolerance"])
            passed = math.isclose(
                float(observed), float(expected), rel_tol=0.0, abs_tol=tolerance
            )
        message = None if passed else (
            f"observed {observed!r}; expected {operator} {expected!r}"
        )
    except RegressionError as exc:
        observed = None
        passed = False
        message = str(exc)
    result: Dict[str, object] = {
        "path": list(path),
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }
    if "absolute_tolerance" in assertion:
        result["absolute_tolerance"] = assertion["absolute_tolerance"]
    if message is not None:
        result["message"] = message
    return result


def run_regression_case(case_path: Path) -> Dict[str, object]:
    """Execute one declared module and compare hash-pinned reference assertions."""

    source = Path(case_path).expanduser().resolve(strict=False)
    case = load_json(source)
    validate_regression(case, source_path=source, check_paths=True)
    module_id = str(case["module_id"])
    project_path = resolve_manifest_path(str(case["project_manifest"]), source)
    report = _RUNNERS[module_id](project_path, True)

    expected_identity = case["expected_identity"]
    assert isinstance(expected_identity, dict)
    identity_fields = (
        "project_manifest_sha256",
        "system_manifest_sha256",
        "input_content_signature_sha256",
    )
    identity_results = []
    for field in identity_fields:
        expected = expected_identity[field]
        observed = report.get(field)
        identity_results.append({
            "field": field,
            "expected": expected,
            "observed": observed,
            "passed": observed == expected,
        })

    raw_assertions = case["assertions"]
    assert isinstance(raw_assertions, list)
    assertion_results = [
        _assertion_result(assertion, report)
        for assertion in raw_assertions
        if isinstance(assertion, dict)
    ]
    passed = all(result["passed"] for result in identity_results + assertion_results)
    approval = case["approval"]
    assert isinstance(approval, dict)
    return {
        "regression_id": str(case["regression_id"]),
        "module_id": module_id,
        "technical_status": "complete" if passed else "failed",
        "scientific_status": "not evaluated",
        "regression_approval_status": str(approval["status"]),
        "approval_owner": str(approval["owner"]),
        "approval_reviewers": list(approval["reviewers"]),
        "approval_decision_utc": approval.get("decision_utc"),
        "approval_notes": list(approval["notes"]),
        "case_path": str(source),
        "project_manifest_path": str(project_path),
        "module_technical_status": report.get("technical_status"),
        "module_scientific_status": report.get("scientific_status"),
        "module_report_signature_sha256": stable_json_sha256(report),
        "identity_checks": identity_results,
        "assertion_checks": assertion_results,
        "passed_check_count": sum(
            bool(result["passed"])
            for result in identity_results + assertion_results
        ),
        "total_check_count": len(identity_results) + len(assertion_results),
        "limitations": [
            "A passing regression establishes repeatability only for the pinned inputs and declared assertions.",
            "Candidate status is not approval; approval requires a named reviewer and decision timestamp in the case.",
            "Regression approval does not establish equilibration, convergence, adequate sampling, mechanism, or scientific validity.",
        ],
    }


def run_regression_case_safe(case_path: Path) -> Dict[str, object]:
    """Convert invalid cases and execution failures into a machine report."""

    try:
        return run_regression_case(case_path)
    except (ManifestValidationError, RegressionError, OSError, KeyError) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "regression_approval_status": "unknown",
            "case_path": str(Path(case_path).expanduser().resolve(strict=False)),
            "error_count": len(messages),
            "issues": [
                {"severity": "error", "code": "REGRESSION_INVALID", "message": message}
                for message in messages
            ],
        }
