"""Leakage-resistant grouped decision-tree classification diagnostics."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .clustering import ClusteringAnalysisError, clustering_kmeans_project
from .manifests import ManifestValidationError, load_json
from .pca import PCAAnalysisError


class GroupedMLAnalysisError(ValueError):
    """Raised when a grouped-learning contract is invalid."""


def _gini(labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    counts: Dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((count / len(labels)) ** 2 for count in counts.values())


def _majority(labels: Sequence[int]) -> int:
    counts: Dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return min(counts, key=lambda label: (-counts[label], label))


def _thresholds(values: Sequence[float], maximum: int) -> List[float]:
    unique = sorted(set(values))
    candidates = [(left + right) / 2.0 for left, right in zip(unique, unique[1:])]
    if len(candidates) <= maximum:
        return candidates
    if maximum == 1:
        return [candidates[len(candidates) // 2]]
    indices = sorted({round(index * (len(candidates) - 1) / (maximum - 1)) for index in range(maximum)})
    return [candidates[index] for index in indices]


def fit_decision_tree(
    vectors: Sequence[Sequence[float]], labels: Sequence[int],
    maximum_depth: int, minimum_leaf_size: int,
    maximum_thresholds_per_feature: int,
) -> Tuple[Dict[str, object], List[float]]:
    """Fit a deterministic axis-aligned CART classification tree."""

    if not vectors or len(vectors) != len(labels):
        raise GroupedMLAnalysisError("decision-tree vectors and labels must be nonempty and aligned")
    feature_count = len(vectors[0])
    if feature_count == 0 or any(len(vector) != feature_count for vector in vectors):
        raise GroupedMLAnalysisError("decision-tree feature vectors must have one fixed positive width")
    importances = [0.0 for _ in range(feature_count)]

    def build(indices: List[int], depth: int) -> Dict[str, object]:
        node_labels = [labels[index] for index in indices]
        prediction = _majority(node_labels)
        impurity = _gini(node_labels)
        node: Dict[str, object] = {
            "prediction": prediction, "sample_count": len(indices),
            "gini": impurity, "class_counts": {
                str(label): node_labels.count(label) for label in sorted(set(node_labels))
            },
        }
        if depth >= maximum_depth or impurity <= 0.0 or len(indices) < 2 * minimum_leaf_size:
            node["leaf"] = True
            return node
        best: Optional[Tuple[float, int, float, List[int], List[int]]] = None
        for feature in range(feature_count):
            values = [float(vectors[index][feature]) for index in indices]
            for threshold in _thresholds(values, maximum_thresholds_per_feature):
                left = [index for index in indices if vectors[index][feature] <= threshold]
                right = [index for index in indices if vectors[index][feature] > threshold]
                if min(len(left), len(right)) < minimum_leaf_size:
                    continue
                weighted = (
                    len(left) * _gini([labels[index] for index in left])
                    + len(right) * _gini([labels[index] for index in right])
                ) / len(indices)
                gain = impurity - weighted
                candidate = (gain, feature, threshold, left, right)
                if best is None or (gain, -feature, -threshold) > (best[0], -best[1], -best[2]):
                    best = candidate
        if best is None or best[0] <= 0.0:
            node["leaf"] = True
            return node
        gain, feature, threshold, left, right = best
        importances[feature] += gain * len(indices)
        node.update({
            "leaf": False, "feature_index": feature, "threshold": threshold,
            "impurity_decrease": gain,
            "left": build(left, depth + 1), "right": build(right, depth + 1),
        })
        return node

    tree = build(list(range(len(vectors))), 0)
    total = sum(importances)
    normalized = [value / total if total > 0.0 else 0.0 for value in importances]
    return tree, normalized


def predict_tree(tree: Mapping[str, object], vector: Sequence[float]) -> int:
    node = tree
    while not bool(node["leaf"]):
        feature = int(node["feature_index"])
        node = node["left"] if vector[feature] <= float(node["threshold"]) else node["right"]
    return int(node["prediction"])


def _metrics(labels: Sequence[int], predictions: Sequence[int], classes: Sequence[int]) -> Dict[str, object]:
    matrix = [[0 for _ in classes] for _ in classes]
    class_index = {label: index for index, label in enumerate(classes)}
    for label, prediction in zip(labels, predictions):
        matrix[class_index[label]][class_index[prediction]] += 1
    accuracy = sum(label == prediction for label, prediction in zip(labels, predictions)) / len(labels)
    scores = []
    for index, label in enumerate(classes):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(len(classes)) if row != index)
        false_negative = sum(matrix[index][column] for column in range(len(classes)) if column != index)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append({"class": label, "precision": precision, "recall": recall, "f1": f1})
    return {
        "accuracy": accuracy,
        "macro_precision": sum(row["precision"] for row in scores) / len(scores),
        "macro_recall": sum(row["recall"] for row in scores) / len(scores),
        "macro_f1": sum(row["f1"] for row in scores) / len(scores),
        "classes": list(classes), "confusion_matrix": matrix, "per_class": scores,
    }


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("grouped_ml") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict):
        raise GroupedMLAnalysisError("definitions.grouped_ml must be an object")
    required = {
        "feature_source", "target_source", "group_strategy", "group_block_size_frames",
        "estimator", "maximum_depth", "minimum_leaf_size",
        "maximum_thresholds_per_feature", "permutation_repeats", "random_seed",
        "minimum_groups", "maximum_observations",
    }
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required))
    if missing:
        raise GroupedMLAnalysisError("grouped-ML settings missing: " + ", ".join(missing))
    if unknown:
        raise GroupedMLAnalysisError("grouped-ML settings contain unknown fields: " + ", ".join(unknown))
    if raw["feature_source"] != "clustering_kmeans_features" or raw["target_source"] != "clustering_kmeans_assignments":
        raise GroupedMLAnalysisError("current grouped ML requires KMeans PCA features and assignments")
    if raw["group_strategy"] != "segment_time_blocks":
        raise GroupedMLAnalysisError("group_strategy currently supports only segment_time_blocks")
    if raw["estimator"] != "decision_tree":
        raise GroupedMLAnalysisError("estimator currently supports only decision_tree")
    for label in (
        "group_block_size_frames", "maximum_depth", "minimum_leaf_size",
        "maximum_thresholds_per_feature", "permutation_repeats", "minimum_groups",
        "maximum_observations",
    ):
        value = raw[label]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise GroupedMLAnalysisError(f"{label} must be a positive integer")
    seed = raw["random_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise GroupedMLAnalysisError("random_seed must be a nonnegative integer")
    return dict(raw)


def grouped_ml_project(project_path: Path, hash_content: bool = False) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    clustering = clustering_kmeans_project(source, hash_content=hash_content)
    rows = clustering["assignments"]
    if len(rows) > int(settings["maximum_observations"]):
        raise GroupedMLAnalysisError("maximum_observations gate exceeded")
    vectors = []
    feature_width = None
    for row_index, row in enumerate(rows):
        values = row.get("feature_values") if isinstance(row, dict) else None
        if not isinstance(values, list) or not values:
            raise GroupedMLAnalysisError(
                f"KMeans assignment {row_index} has no nonempty feature_values vector"
            )
        try:
            vector = tuple(float(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise GroupedMLAnalysisError(
                f"KMeans assignment {row_index} contains a nonnumeric feature value"
            ) from exc
        if not all(math.isfinite(value) for value in vector):
            raise GroupedMLAnalysisError(
                f"KMeans assignment {row_index} contains a nonfinite feature value"
            )
        if feature_width is None:
            feature_width = len(vector)
        elif len(vector) != feature_width:
            raise GroupedMLAnalysisError(
                "KMeans assignment feature_values vectors do not have one fixed width"
            )
        vectors.append(vector)
    labels = [int(row["cluster_id"]) for row in rows]
    groups = [
        (
            str(row["system_id"]), str(row["replica_id"]), str(row["segment_id"]),
            int(row["source_frame_index"]) // int(settings["group_block_size_frames"]),
        )
        for row in rows
    ]
    unique_groups = sorted(set(groups))
    if len(unique_groups) < int(settings["minimum_groups"]):
        raise GroupedMLAnalysisError("group count is below minimum_groups")
    classes = sorted(set(labels))
    fold_reports = []
    all_labels = []
    all_predictions = []
    permutation_drops = [[] for _ in vectors[0]]
    for fold_index, held_out in enumerate(unique_groups):
        train_indices = [index for index, group in enumerate(groups) if group != held_out]
        test_indices = [index for index, group in enumerate(groups) if group == held_out]
        tree, _ = fit_decision_tree(
            [vectors[index] for index in train_indices], [labels[index] for index in train_indices],
            int(settings["maximum_depth"]), int(settings["minimum_leaf_size"]),
            int(settings["maximum_thresholds_per_feature"]),
        )
        predictions = [predict_tree(tree, vectors[index]) for index in test_indices]
        true = [labels[index] for index in test_indices]
        baseline = sum(left == right for left, right in zip(true, predictions)) / len(true)
        for feature in range(len(vectors[0])):
            for repeat in range(int(settings["permutation_repeats"])):
                rng = random.Random(int(settings["random_seed"]) + fold_index * 100003 + feature * 1009 + repeat)
                permuted_values = [vectors[index][feature] for index in test_indices]
                rng.shuffle(permuted_values)
                permuted_predictions = []
                for local_index, source_index in enumerate(test_indices):
                    vector = list(vectors[source_index])
                    vector[feature] = permuted_values[local_index]
                    permuted_predictions.append(predict_tree(tree, vector))
                permuted_accuracy = sum(left == right for left, right in zip(true, permuted_predictions)) / len(true)
                permutation_drops[feature].append(baseline - permuted_accuracy)
        metrics = _metrics(true, predictions, classes)
        fold_reports.append({
            "held_out_group": list(held_out), "training_observation_count": len(train_indices),
            "held_out_observation_count": len(test_indices), "metrics": metrics,
        })
        all_labels.extend(true)
        all_predictions.extend(predictions)
    final_tree, impurity_importance = fit_decision_tree(
        vectors, labels, int(settings["maximum_depth"]), int(settings["minimum_leaf_size"]),
        int(settings["maximum_thresholds_per_feature"]),
    )
    permutation_importance = [
        sum(values) / len(values) if values else 0.0 for values in permutation_drops
    ]
    issues = [issue for issue in clustering.get("issues", []) if isinstance(issue, dict)]
    return {
        "module_id": "grouped_ml", "technical_status": "complete",
        "scientific_status": "not evaluated", "project_manifest_path": str(source),
        "project_manifest_sha256": clustering["project_manifest_sha256"],
        "system_manifest_path": clustering["system_manifest_path"],
        "system_manifest_sha256": clustering["system_manifest_sha256"],
        "input_content_signature_sha256": clustering["input_content_signature_sha256"],
        "content_hashes_included": hash_content, "settings": settings,
        "feature_contract": clustering.get("feature_contract"),
        "target_contract": "KMeans cluster ID used only as a technical classification target",
        "group_contract": "leave-one-segment-time-block-out; no frame from a held-out block enters training",
        "observation_count": len(rows), "group_count": len(unique_groups),
        "class_count": len(classes), "fold_reports": fold_reports,
        "pooled_held_out_metrics": _metrics(all_labels, all_predictions, classes),
        "final_model": final_tree,
        "feature_importance": [{
            "feature_index": index,
            "impurity_importance": impurity_importance[index],
            "held_out_permutation_accuracy_drop": permutation_importance[index],
        } for index in range(len(vectors[0]))],
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "The KMeans label is a technical target derived from the same PCA features; performance is not a biological result.",
            "Groups are held out as complete time blocks to reduce direct frame leakage.",
            "Impurity importance and permutation accuracy drops are predictive diagnostics, not mechanistic importance.",
            "The base implementation currently supports a deterministic decision tree; random forests require a separately validated estimator extension.",
        ],
    }


def grouped_ml_project_safe(project_path: Path, hash_content: bool = False) -> Dict[str, object]:
    try:
        return grouped_ml_project(project_path, hash_content=hash_content)
    except (
        ManifestValidationError, GroupedMLAnalysisError, ClusteringAnalysisError,
        PCAAnalysisError, OSError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "grouped_ml", "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [{"severity": "error", "code": "GROUPED_ML_INVALID", "message": message} for message in messages],
        }
