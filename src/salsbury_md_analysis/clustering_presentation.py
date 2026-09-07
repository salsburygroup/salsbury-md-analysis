"""Comparable geometric evaluation and one primary partition per report scope.

This contract is for presentation, not kinetic model selection. It never changes
the fitted labels, scientific sampling, or which results are retained.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

MODULES = {"clustering_kmeans", "clustering_imwkmeans", "clustering_hdbscan", "alternative_clustering"}
MAXIMUM_EVALUATION_OBSERVATIONS = 1024


def _identity(row):
    # Exclude method-specific fields, source paths, and fit-sample indices.
    return {key: row[key] for key in (
        "system_id", "replica_id", "member_id", "exchangeable_unit_id", "segment_id",
        "trajectory_id", "frame_index", "source_frame_index", "time_ps",
    ) if key in row}


def evaluate_partition(vectors, metadata, labels, *, feature_definition=None, maximum_observations=MAXIMUM_EVALUATION_OBSERVATIONS):
    """Evaluate a fixed hash sample in declared (already transformed) geometry.

    Peak pair-distance memory is O(B squared), B <= 1024 by default. Selection
    uses observation identity alone, never labels or a method-specific seed.
    Sampling uncertainty concerns this geometric score, not ensemble uncertainty.
    """
    if isinstance(maximum_observations, bool) or not isinstance(maximum_observations, int) or maximum_observations < 3:
        raise ValueError("clustering presentation evaluation needs at least three observations")
    if len(vectors) != len(metadata) or len(labels) != len(metadata) or len(labels) < 3:
        return {"status": "ineligible", "reason": "missing or incomplete observation identities and labels"}
    if any(label is None or isinstance(label, bool) or int(label) < 0 for label in labels):
        return {"status": "ineligible", "reason": "noise or unassigned observations; not a complete partition"}
    digest = hashlib.sha256()
    semantic = {key: feature_definition[key] for key in (
        "feature_source", "component_indices", "trajectory_feature_columns", "standardize_features"
    ) if feature_definition is not None and key in feature_definition}
    digest.update(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode())
    heap = []
    width = None
    for index, (vector, record) in enumerate(zip(vectors, metadata)):
        identity = _identity(record)
        if not identity or not any(key in identity for key in ("frame_index", "source_frame_index", "time_ps")):
            return {"status": "ineligible", "reason": "frame identity is missing"}
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        array = np.asarray(vector, dtype="<f8")
        if array.ndim != 1 or not np.isfinite(array).all() or not len(array):
            raise ValueError("clustering evaluation requires finite feature vectors")
        if width is None:
            width = len(array)
        elif width != len(array):
            raise ValueError("inconsistent clustering feature width")
        digest.update(len(encoded).to_bytes(8, "little")); digest.update(encoded)
        digest.update(array.tobytes())
        priority = int.from_bytes(hashlib.sha256(encoded).digest(), "big")
        entry = (-priority, -index, index)
        if len(heap) < maximum_observations:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    indices = sorted(entry[2] for entry in heap)
    sample_labels = np.asarray([int(labels[i]) for i in indices])
    unique = np.unique(sample_labels)
    full_unique = {int(value) for value in labels}
    if len(unique) < 2 or len(unique) == len(indices) or set(unique) != full_unique:
        return {"status": "ineligible", "reason": "evaluation sample does not cover at least two nonsingleton partition states"}
    sample = np.asarray([vectors[i] for i in indices], dtype=float)
    from sklearn.metrics import silhouette_samples
    scores = silhouette_samples(sample, sample_labels, metric="euclidean")
    # Heuristic score-resolution band, not a calibrated CI: the reference
    # partition is sampled too, so an independent-score standard error is not
    # a complete uncertainty estimate.
    half_width = (0.0 if len(indices) == len(labels) else
                  min(2.0, 1.96 * float(np.std(scores, ddof=1)) / math.sqrt(len(indices))))
    return {
        "schema": "salsbury-clustering-presentation-evaluation-v1", "status": "eligible",
        "geometry_and_identity_sha256": digest.hexdigest(),
        "evaluation_indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
        "metric": "euclidean_in_declared_transformed_feature_space",
        "noise_policy": "require_all_observations_assigned",
        "sampling": "lowest_identity_sha256_without_replacement",
        "observation_count": len(labels), "evaluated_observation_count": len(indices),
        "feature_count": width, "state_count": len(full_unique),
        "score": float(np.mean(scores)), "sampling_half_width": half_width,
        "uncertainty_scope": "heuristic geometric score-resolution band; not a calibrated confidence interval or ensemble uncertainty",
    }


def report_models(report, path):
    """Compact records for sidecars and the shared picker/viewer contract."""
    module = str(report.get("module_id", ""))
    if module not in MODULES:
        return []
    rows = report.get("algorithm_results", []) if module == "alternative_clustering" else [report.get("selected_model", {})]
    models = []
    for model in rows:
        if not isinstance(model, dict) or not model:
            continue
        algorithm = str(model.get("requested_algorithm", model.get("algorithm", module)))
        if algorithm == "pald":
            continue
        models.append({"report_path": str(path), "module_id": module, "algorithm": algorithm,
            "evaluation": model.get("presentation_evaluation"),
            "reported_silhouette": model.get("silhouette", model.get("retained_only_silhouette")),
            "candidate_id": hashlib.sha256((str(path)+"|"+algorithm).encode()).hexdigest()[:20]})
    return models


def scope_key(path):
    parts = Path(path).parts
    # The parent before the module directory includes per-system/member/view scope.
    if "results" in parts:
        return "/".join(parts[parts.index("results")+1:-2]) or "pooled"
    return str(Path(path).parent.parent)


def load_report_models(path, module_id):
    """Stream compact method records from legacy reports, skipping assignments."""
    if module_id not in MODULES:
        return []
    import ijson
    compact = {"module_id": module_id, "selected_model": {}, "algorithm_results": []}
    current = None
    with Path(path).open("rb") as handle:
        for prefix, event, value in ijson.parse(handle, use_float=True):
            if prefix == "algorithm_results.item" and event == "start_map":
                current = {}
            elif prefix == "algorithm_results.item" and event == "end_map":
                compact["algorithm_results"].append(current)
                current = None
            if event not in {"string", "number", "boolean", "null"}:
                continue
            owner = compact["selected_model"] if prefix.startswith("selected_model.") else current
            base = "selected_model." if prefix.startswith("selected_model.") else "algorithm_results.item."
            if owner is None or not prefix.startswith(base):
                continue
            field = prefix[len(base):]
            if field in {"algorithm", "requested_algorithm", "silhouette", "retained_only_silhouette"}:
                owner[field] = value
            elif field.startswith("presentation_evaluation.") and field.count(".") == 1:
                owner.setdefault("presentation_evaluation", {})[field.split(".")[1]] = value
    return report_models(compact, path)


def select_primary_partitions(models):
    groups = defaultdict(list)
    for model in models:
        groups[scope_key(model["report_path"])].append(dict(model))
    results = []
    keys = ("geometry_and_identity_sha256", "evaluation_indices_sha256", "metric", "noise_policy", "observation_count")
    for scope, candidates in sorted(groups.items()):
        eligible = []
        for candidate in candidates:
            evaluation = candidate.get("evaluation")
            if isinstance(evaluation, dict) and evaluation.get("status") == "eligible" and all(evaluation.get(key) is not None for key in keys):
                if isinstance(evaluation.get("score"), (int, float)) and math.isfinite(evaluation["score"]):
                    eligible.append(candidate)
        contracts = {tuple(row["evaluation"][key] for key in keys) for row in eligible}
        selected = None
        tied = []
        if len(contracts) > 1:
            status, reason = "abstained", "Feature geometry or evaluation observations differ; no cross-contract winner."
        elif not eligible:
            status, reason = "abstained", "No complete partition has a comparable geometric evaluation. Existing scores remain available."
        else:
            best = max(eligible, key=lambda row: row["evaluation"]["score"])
            best_eval = best["evaluation"]
            tied = [row for row in eligible if best_eval["score"] - row["evaluation"]["score"] <=
                    max(1e-9, best_eval.get("sampling_half_width", 0) + row["evaluation"].get("sampling_half_width", 0))]
            selected = min(tied, key=lambda row: (row["evaluation"].get("state_count", math.inf), row["algorithm"], row["candidate_id"]))
            status = "tied" if len(tied) > 1 else "selected"
            reason = ("Scores fall within a heuristic sampling-resolution band; show the fewer-state partition, then algorithm name as a deterministic tie-break. This is not an equivalence test."
                      if len(tied) > 1 else "Highest silhouette on the same evaluation observations and feature geometry.")
        for candidate in candidates:
            candidate["presentation_role"] = "primary" if selected and candidate["candidate_id"] == selected["candidate_id"] else "alternative"
        results.append({"scope": scope, "status": status, "reason": reason,
                        "primary_candidate_id": selected["candidate_id"] if selected else None,
                        "tied_candidate_ids": [row["candidate_id"] for row in tied],
                        "unevaluated_candidate_count": len(candidates)-len(eligible), "candidates": candidates})
    return {"schema": "salsbury-primary-clustering-presentation-v1", "groups": results,
            "kinetic_model_selection_changed": False, "fes_selection_changed": False}


def apply_primary_findings(findings, selection):
    """Do not let method scores or alternative partitions fill physical headlines."""
    primaries = {(row["report_path"], row["algorithm"]) for group in selection["groups"] for row in group["candidates"] if row["presentation_role"] == "primary"}
    known_paths = {row["report_path"] for group in selection["groups"] for row in group["candidates"]}
    for row in findings:
        module = str(row.get("module_id", ""))
        family = str(row.get("comparison_family", ""))
        if "model_selection" in family or (module == "markov_state_models" and row.get("ranking_role") != "validation_context" and family.endswith(("best_clustering_state_model", "fes_state_model"))):
            row.update(ranking_role="method_diagnostic", presentation_eligible=False)
        elif module in MODULES:
            sources = {str(row.get("report_path")), *map(str, row.get("source_report_paths", []))}
            algorithm = row.get("clustering_algorithm", module)
            if not any((source, algorithm) in primaries for source in sources):
                row.update(ranking_role="alternative_partition" if sources & known_paths else "unranked_partition", presentation_eligible=False)
    return findings
