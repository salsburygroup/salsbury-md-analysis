"""Nested group-held-out logistic and elastic-net classification."""

from __future__ import annotations

import importlib
import math
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .grouped_ml import _metrics
from .hydrogen_bond_discovery import (
    HydrogenBondDiscoveryError, hydrogen_bond_discovery_project,
)
from .hydrogen_bond_sparse import dense_primary_values
from .manifests import ManifestValidationError, load_json
from .validation import positive_integer


class GroupedRegularizedClassificationError(ValueError):
    """Raised when grouped classification would leak or is underdetermined."""


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("grouped_regularized_classification") if isinstance(definitions, dict) else None
    required = {
        "feature_source", "target_source", "group_strategy",
        "group_block_size_frames", "estimators",
        "inverse_regularization_strengths", "elastic_net_l1_ratios",
        "class_weight", "maximum_iterations", "minimum_outer_groups",
        "minimum_inner_folds", "maximum_observations", "random_seed",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise GroupedRegularizedClassificationError(
            "definitions.grouped_regularized_classification fields do not match the contract"
        )
    if raw["feature_source"] != "hydrogen_bond_discovery":
        raise GroupedRegularizedClassificationError(
            "feature_source must be hydrogen_bond_discovery"
        )
    if raw["target_source"] != "system_id":
        raise GroupedRegularizedClassificationError("target_source must be system_id")
    if raw["group_strategy"] not in {"replica", "segment", "segment_time_blocks"}:
        raise GroupedRegularizedClassificationError(
            "group_strategy must be replica, segment, or segment_time_blocks"
        )
    estimators = raw["estimators"]
    if (
        not isinstance(estimators, list) or not estimators
        or len(set(estimators)) != len(estimators)
        or any(value not in {"logistic_l2", "elastic_net"} for value in estimators)
    ):
        raise GroupedRegularizedClassificationError(
            "estimators must be a unique subset of logistic_l2 and elastic_net"
        )

    def finite_grid(value: object, label: str, lower: float, upper: float | None = None) -> List[float]:
        if not isinstance(value, list) or not value:
            raise GroupedRegularizedClassificationError(f"{label} must be a nonempty array")
        result = []
        for item in value:
            if (
                isinstance(item, bool) or not isinstance(item, (int, float))
                or not math.isfinite(float(item)) or float(item) <= lower
                or (upper is not None and float(item) >= upper)
            ):
                raise GroupedRegularizedClassificationError(f"{label} contains an invalid value")
            result.append(float(item))
        if len(set(result)) != len(result):
            raise GroupedRegularizedClassificationError(f"{label} values must be unique")
        return sorted(result)

    strengths = finite_grid(
        raw["inverse_regularization_strengths"],
        "inverse_regularization_strengths", 0.0,
    )
    l1_ratios = finite_grid(
        raw["elastic_net_l1_ratios"], "elastic_net_l1_ratios", -1.0, 1.0000001
    )
    if any(not 0.0 <= value <= 1.0 for value in l1_ratios):
        raise GroupedRegularizedClassificationError(
            "elastic_net_l1_ratios must be in [0,1]"
        )
    if raw["class_weight"] not in {"none", "balanced"}:
        raise GroupedRegularizedClassificationError("class_weight must be none or balanced")
    integers = {}
    for label in (
        "group_block_size_frames", "maximum_iterations", "minimum_outer_groups",
        "minimum_inner_folds", "maximum_observations",
    ):
        integers[label] = positive_integer(
            raw[label], label, error_type=GroupedRegularizedClassificationError
        )
    seed = raw["random_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise GroupedRegularizedClassificationError("random_seed must be nonnegative")
    return {
        **raw,
        **integers,
        "estimators": list(estimators),
        "inverse_regularization_strengths": strengths,
        "elastic_net_l1_ratios": l1_ratios,
    }


def _parameter_grid(settings: Mapping[str, object]) -> List[Dict[str, object]]:
    rows = []
    for estimator in settings["estimators"]:  # type: ignore[union-attr]
        for strength in settings["inverse_regularization_strengths"]:  # type: ignore[union-attr]
            if estimator == "logistic_l2":
                rows.append({
                    "estimator": estimator,
                    "inverse_regularization_strength": float(strength),
                    "l1_ratio": None,
                })
            else:
                for ratio in settings["elastic_net_l1_ratios"]:  # type: ignore[union-attr]
                    rows.append({
                        "estimator": estimator,
                        "inverse_regularization_strength": float(strength),
                        "l1_ratio": float(ratio),
                    })
    return rows


def _fit(
    vectors: np.ndarray, labels: np.ndarray, parameters: Mapping[str, object],
    settings: Mapping[str, object], seed_offset: int = 0,
):
    try:
        linear_model = importlib.import_module("sklearn.linear_model")
    except ImportError as exc:
        raise GroupedRegularizedClassificationError(
            "grouped regularized classification requires the clustering extra (scikit-learn)"
        ) from exc
    estimator = str(parameters["estimator"])
    version_text = importlib_metadata.version("scikit-learn")
    version_parts = tuple(
        int(part) for part in version_text.split(".")[:2]
    )
    regularization = (
        {"l1_ratio": 0.0 if estimator == "logistic_l2" else float(parameters["l1_ratio"])}
        if version_parts >= (1, 8)
        else {
            "penalty": "l2" if estimator == "logistic_l2" else "elasticnet",
            "l1_ratio": (
                None if estimator == "logistic_l2" else float(parameters["l1_ratio"])
            ),
        }
    )
    model = linear_model.LogisticRegression(
        solver="lbfgs" if estimator == "logistic_l2" else "saga",
        C=float(parameters["inverse_regularization_strength"]),
        class_weight=(None if settings["class_weight"] == "none" else "balanced"),
        max_iter=int(settings["maximum_iterations"]),
        random_state=int(settings["random_seed"]) + seed_offset,
        tol=1.0e-6,
        **regularization,
    )
    model.fit(vectors, labels)
    if int(np.max(model.n_iter_)) >= int(settings["maximum_iterations"]):
        raise GroupedRegularizedClassificationError(
            "regularized classifier reached maximum_iterations without convergence"
        )
    return model


def _valid_fold(
    labels: Sequence[int], train_indices: Sequence[int], test_indices: Sequence[int]
) -> bool:
    training = {labels[index] for index in train_indices}
    testing = {labels[index] for index in test_indices}
    return len(training) >= 2 and testing.issubset(training)


def _select_parameters(
    vectors: np.ndarray, labels: Sequence[int], groups: Sequence[Tuple[object, ...]],
    settings: Mapping[str, object], feature_indices: Sequence[int] | None = None,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    columns = list(range(vectors.shape[1])) if feature_indices is None else list(feature_indices)
    diagnostics = []
    candidates = []
    unique_groups = sorted(set(groups))
    for parameter_index, parameters in enumerate(_parameter_grid(settings)):
        true_all: List[int] = []
        predicted_all: List[int] = []
        fold_count = 0
        for fold_index, held_out in enumerate(unique_groups):
            train = [index for index, group in enumerate(groups) if group != held_out]
            test = [index for index, group in enumerate(groups) if group == held_out]
            if not _valid_fold(labels, train, test):
                continue
            try:
                model = _fit(
                    vectors[np.ix_(train, columns)], np.asarray([labels[index] for index in train]),
                    parameters, settings, seed_offset=parameter_index * 1009 + fold_index,
                )
            except GroupedRegularizedClassificationError:
                continue
            predictions = [int(value) for value in model.predict(vectors[np.ix_(test, columns)])]
            true_all.extend(labels[index] for index in test)
            predicted_all.extend(predictions)
            fold_count += 1
        eligible = fold_count >= int(settings["minimum_inner_folds"])
        metrics = (
            _metrics(true_all, predicted_all, sorted(set(labels)))
            if eligible else None
        )
        row = {
            "parameters": parameters,
            "eligible": eligible,
            "valid_group_fold_count": fold_count,
            "pooled_group_held_out_metrics": metrics,
        }
        diagnostics.append(row)
        if eligible:
            candidates.append(row)
    if not candidates:
        raise GroupedRegularizedClassificationError(
            "no regularized parameter candidate passed grouped inner-validation gates"
        )
    selected = sorted(
        candidates,
        key=lambda row: (
            -float(row["pooled_group_held_out_metrics"]["macro_f1"]),  # type: ignore[index]
            -float(row["pooled_group_held_out_metrics"]["accuracy"]),  # type: ignore[index]
            str(row["parameters"]["estimator"]),  # type: ignore[index]
            float(row["parameters"]["inverse_regularization_strength"]),  # type: ignore[index]
            -1.0 if row["parameters"]["l1_ratio"] is None else float(row["parameters"]["l1_ratio"]),  # type: ignore[index]
        ),
    )[0]
    return dict(selected["parameters"]), diagnostics  # type: ignore[arg-type]


def nested_grouped_classification(
    vectors: Sequence[Sequence[float]], labels: Sequence[int],
    groups: Sequence[Tuple[object, ...]], feature_names: Sequence[str],
    settings: Mapping[str, object],
) -> Dict[str, object]:
    """Run nested leave-one-group-out tuning and outer evaluation."""

    values = np.asarray(vectors, dtype=float)
    if (
        values.ndim != 2 or len(values) != len(labels) or len(values) != len(groups)
        or values.shape[1] != len(feature_names) or not np.isfinite(values).all()
    ):
        raise GroupedRegularizedClassificationError("classification arrays are invalid")
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise GroupedRegularizedClassificationError("classification requires at least two classes")
    unique_groups = sorted(set(groups))
    if len(unique_groups) < int(settings["minimum_outer_groups"]):
        raise GroupedRegularizedClassificationError("group count is below minimum_outer_groups")
    groups_by_class = {
        label: {groups[index] for index, value in enumerate(labels) if value == label}
        for label in classes
    }
    if any(len(class_groups) < 2 for class_groups in groups_by_class.values()):
        raise GroupedRegularizedClassificationError(
            "each class requires at least two independent groups"
        )
    all_true: List[int] = []
    all_predictions: List[int] = []
    top_only_predictions: List[int] = []
    without_top_predictions: List[int] = []
    coefficient_rows = []
    folds = []
    for outer_index, held_out in enumerate(unique_groups):
        train = [index for index, group in enumerate(groups) if group != held_out]
        test = [index for index, group in enumerate(groups) if group == held_out]
        if not _valid_fold(labels, train, test):
            raise GroupedRegularizedClassificationError(
                "an outer held-out group removes a target class from training"
            )
        inner_groups = [groups[index] for index in train]
        inner_labels = [labels[index] for index in train]
        selected, tuning = _select_parameters(
            values[train], inner_labels, inner_groups, settings
        )
        model = _fit(
            values[train], np.asarray(inner_labels), selected, settings,
            seed_offset=outer_index * 100003,
        )
        predictions = [int(value) for value in model.predict(values[test])]
        true = [labels[index] for index in test]
        importance = np.mean(np.abs(np.asarray(model.coef_, dtype=float)), axis=0)
        top_feature = int(np.argmax(importance))
        top_model = _fit(
            values[np.ix_(train, [top_feature])], np.asarray(inner_labels),
            selected, settings, seed_offset=outer_index * 100003 + 1,
        )
        top_predictions = [
            int(value) for value in top_model.predict(values[np.ix_(test, [top_feature])])
        ]
        remaining = [index for index in range(values.shape[1]) if index != top_feature]
        no_top_predictions: List[int] = []
        if remaining:
            no_top_model = _fit(
                values[np.ix_(train, remaining)], np.asarray(inner_labels),
                selected, settings, seed_offset=outer_index * 100003 + 2,
            )
            no_top_predictions = [
                int(value) for value in no_top_model.predict(values[np.ix_(test, remaining)])
            ]
            without_top_predictions.extend(no_top_predictions)
        all_true.extend(true)
        all_predictions.extend(predictions)
        top_only_predictions.extend(top_predictions)
        coefficient_rows.append({
            "held_out_group": list(held_out),
            "classes": [int(value) for value in model.classes_],
            "intercepts": np.asarray(model.intercept_, dtype=float).tolist(),
            "coefficients": np.asarray(model.coef_, dtype=float).tolist(),
            "top_feature_index": top_feature,
            "top_feature_name": feature_names[top_feature],
        })
        folds.append({
            "held_out_group": list(held_out),
            "training_observation_count": len(train),
            "held_out_observation_count": len(test),
            "selected_parameters": selected,
            "inner_tuning": tuning,
            "held_out_metrics": _metrics(true, predictions, classes),
            "top_feature_only_held_out_metrics": _metrics(true, top_predictions, classes),
            "without_top_feature_held_out_metrics": (
                _metrics(true, no_top_predictions, classes) if remaining else None
            ),
        })
    final_parameters, final_tuning = _select_parameters(
        values, labels, groups, settings
    )
    final_model = _fit(values, np.asarray(labels), final_parameters, settings, seed_offset=999983)
    return {
        "outer_fold_count": len(folds),
        "fold_reports": folds,
        "pooled_outer_held_out_metrics": _metrics(all_true, all_predictions, classes),
        "pooled_top_feature_only_metrics": _metrics(
            all_true, top_only_predictions, classes
        ),
        "pooled_without_top_feature_metrics": (
            _metrics(all_true, without_top_predictions, classes)
            if values.shape[1] > 1 else None
        ),
        "outer_fold_coefficients": coefficient_rows,
        "final_grouped_tuning": final_tuning,
        "final_model": {
            "selected_parameters": final_parameters,
            "classes": [int(value) for value in final_model.classes_],
            "intercepts": np.asarray(final_model.intercept_, dtype=float).tolist(),
            "coefficients": np.asarray(final_model.coef_, dtype=float).tolist(),
            "feature_names": list(feature_names),
        },
    }


def grouped_regularized_classification_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    discovery = hydrogen_bond_discovery_project(source, hash_content=hash_content)
    rows = discovery["frame_bond_matrix"]
    if len(rows) > int(settings["maximum_observations"]):
        raise GroupedRegularizedClassificationError("maximum_observations gate exceeded")
    candidate_count = len(discovery["candidate_dictionary"])
    if len(rows) * candidate_count > 50_000_000:
        raise GroupedRegularizedClassificationError(
            "dense classifier materialization exceeds the 50,000,000-element resource gate"
        )
    systems = sorted({str(row["system_id"]) for row in rows})
    class_by_system = {system_id: index for index, system_id in enumerate(systems)}
    labels = [class_by_system[str(row["system_id"])] for row in rows]
    vectors = [dense_primary_values(row, candidate_count) for row in rows]
    strategy = settings["group_strategy"]
    groups = []
    for row in rows:
        base = (str(row["system_id"]), str(row["replica_id"]))
        if strategy == "replica":
            group = base
        elif strategy == "segment":
            group = (*base, str(row["segment_id"]))
        else:
            group = (
                *base, str(row["segment_id"]),
                int(row["source_frame_index"]) // int(settings["group_block_size_frames"]),
            )
        groups.append(group)
    feature_names = [str(row["bond_id"]) for row in discovery["candidate_dictionary"]]
    analysis = nested_grouped_classification(
        vectors, labels, groups, feature_names, settings
    )
    issues = [issue for issue in discovery.get("issues", []) if isinstance(issue, dict)]
    try:
        sklearn_version = importlib_metadata.version("scikit-learn")
    except importlib_metadata.PackageNotFoundError:
        sklearn_version = "unknown"
    return {
        "module_id": "grouped_regularized_classification",
        "technical_status": "complete", "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": discovery["project_manifest_sha256"],
        "system_manifest_path": discovery["system_manifest_path"],
        "system_manifest_sha256": discovery["system_manifest_sha256"],
        "input_content_signature_sha256": discovery["input_content_signature_sha256"],
        "content_hashes_included": hash_content, "settings": settings,
        "implementation": {"package": "scikit-learn", "version": sklearn_version},
        "target_dictionary": [
            {"class_id": class_id, "system_id": system_id}
            for system_id, class_id in class_by_system.items()
        ],
        "group_contract": (
            f"nested leave-one-{strategy}-out; every held-out group is excluded from "
            "both fitting and hyperparameter selection"
        ),
        "observation_count": len(rows), "group_count": len(set(groups)),
        "feature_count": len(feature_names),
        **analysis,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "System labels are predictive targets; classification performance and coefficients do not establish causality or mechanism.",
            "At least two independent groups per class are required; replica-level grouping is preferred when biological/technical replicas exist.",
            "Time-block grouping reduces adjacent-frame leakage but cannot manufacture independent replicas.",
            "Feature dictionaries are declared without occupancy-based prefiltering; any subsequent feature selection must remain nested inside training folds.",
            "Coefficients, regularization grids, class weighting, and ablations require stability and domain review before interpretation.",
        ],
    }


def grouped_regularized_classification_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return grouped_regularized_classification_project(
            project_path, hash_content=hash_content
        )
    except (
        ManifestValidationError, HydrogenBondDiscoveryError,
        GroupedRegularizedClassificationError, ImportError,
        OSError, KeyError, ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "grouped_regularized_classification",
            "technical_status": "failed", "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "GROUPED_REGULARIZED_CLASSIFICATION_INVALID", "message": message}
                for message in messages
            ],
        }
