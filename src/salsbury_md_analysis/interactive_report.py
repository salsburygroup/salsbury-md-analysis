"""Offline-first interactive browser for completed analysis campaigns.

The generated report is one self-contained HTML document plus a small,
hash-bound manifest.  It does not require a Python server, a JavaScript package
manager, or an internet connection.  Raw analysis artifacts remain the source
of record and are linked rather than rewritten.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .analysis_config import DEFAULT_DISABLED_MODULES
from .manifests import load_json


class InteractiveReportError(ValueError):
    """Raised when an interactive result cannot preserve source evidence."""


_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}
_MAXIMUM_REPORT_PARSE_BYTES = 128_000_000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(root))
    except ValueError as exc:
        raise InteractiveReportError(
            f"interactive-report asset escapes the analysis root: {path}"
        ) from exc


def _raw_link(relative_path: str) -> str:
    return "../" + relative_path.replace(os.sep, "/")


def _optional_json(path: Path) -> object | None:
    if not path.is_file():
        return None
    try:
        return load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise InteractiveReportError(f"cannot read JSON evidence {path}: {exc}") from exc


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _json_safe(value: object) -> object:
    """Return strict JSON data, replacing non-finite scientific sentinels.

    Python otherwise emits ``NaN`` and ``Infinity`` tokens. Browsers reject
    those tokens in ``JSON.parse``. Raw reports remain linked and hash-bound;
    the interactive preview represents an undefined numeric value as ``null``.
    """

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _preview(value: object, *, depth: int = 0, maximum_items: int = 80) -> object:
    """Return a deterministic bounded preview while leaving raw JSON linked."""

    if depth >= 6:
        if isinstance(value, (dict, list)):
            return {"preview_truncated": True, "value_type": type(value).__name__}
        return value
    if isinstance(value, dict):
        result: Dict[str, object] = {}
        for index, key in enumerate(sorted(value, key=str)):
            if index >= maximum_items:
                result["preview_truncated_keys"] = len(value) - maximum_items
                break
            result[str(key)] = _preview(
                value[key], depth=depth + 1, maximum_items=maximum_items
            )
        return result
    if isinstance(value, list):
        if len(value) <= maximum_items:
            return [
                _preview(row, depth=depth + 1, maximum_items=maximum_items)
                for row in value
            ]
        step = max(1, math.ceil(len(value) / maximum_items))
        sampled = value[::step][:maximum_items]
        return {
            "preview_truncated": True,
            "source_item_count": len(value),
            "deterministic_preview_stride": step,
            "items": [
                _preview(row, depth=depth + 1, maximum_items=maximum_items)
                for row in sampled
            ],
        }
    return value


def _key_metrics(report: Mapping[str, object]) -> List[Dict[str, object]]:
    preferred = re.compile(
        r"(count|fraction|occupancy|silhouette|score|rms[fd]|sasa|memory|cpu|"
        r"wall|frame|observation|cluster|basin|state|ion|coordination)",
        flags=re.IGNORECASE,
    )
    metrics: List[Dict[str, object]] = []

    def visit(value: object, path: Sequence[str], depth: int) -> None:
        if len(metrics) >= 24 or depth > 3:
            return
        if _finite_number(value) and path and preferred.search(path[-1]):
            metrics.append({"label": ".".join(path), "value": value})
            return
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                if str(key) in {
                    "assignments", "frame_assignments", "grid", "matrix",
                    "projections", "coordinates", "atom_statistics",
                }:
                    continue
                visit(value[key], [*path, str(key)], depth + 1)

    visit(report, [], 0)
    return metrics


def _fes_visuals(report: Mapping[str, object]) -> List[Dict[str, object]]:
    rows = report.get("smoothing_landscapes")
    visuals: List[Dict[str, object]] = []
    if not isinstance(rows, list):
        landscape = report.get("landscape")
        if isinstance(landscape, dict):
            rows = [{
                "smoothing_sigma_bins": report.get("primary_smoothing_sigma_bins"),
                "landscape": landscape,
                "per_system_landscapes": report.get("per_system_landscapes", []),
            }]
        else:
            return visuals
    for smoothing in rows:
        if not isinstance(smoothing, dict):
            continue
        sigma = smoothing.get("smoothing_sigma_bins")
        pooled = smoothing.get("landscape")
        if isinstance(pooled, dict) and isinstance(pooled.get("grid"), list):
            visuals.append({
                "kind": "fes",
                "title": f"Pooled FES / occupancy surface; smoothing sigma {sigma} bins",
                "system_id": "pooled",
                "smoothing_sigma_bins": sigma,
                "landscape": pooled,
            })
        systems = smoothing.get("per_system_landscapes")
        complete_systems = [
            system for system in systems
            if (
                isinstance(system, dict)
                and system.get("technical_status") == "complete"
                and isinstance(system.get("landscape"), dict)
            )
        ] if isinstance(systems, list) else []
        # For one system, the pooled and per-system surfaces are the same
        # observation set. Keep the pooled view only; comparisons retain every
        # system-specific surface alongside the common pooled surface.
        if len(complete_systems) > 1:
            for system in complete_systems:
                if (
                    isinstance(system, dict)
                    and isinstance(system.get("landscape"), dict)
                ):
                    visuals.append({
                        "kind": "fes",
                        "title": (
                            f"{system.get('system_id')} FES / occupancy surface; "
                            f"smoothing sigma {sigma} bins"
                        ),
                        "system_id": str(system.get("system_id")),
                        "smoothing_sigma_bins": sigma,
                        "landscape": system["landscape"],
                    })
    return visuals


def _clustering_visuals(report: Mapping[str, object]) -> List[Dict[str, object]]:
    model = report.get("selected_model")
    if not isinstance(model, dict):
        return []
    sizes = model.get("cluster_sizes")
    if not isinstance(sizes, list) or not all(_finite_number(value) for value in sizes):
        return []
    evaluation = model.get("silhouette_evaluation")
    compact_evaluation = (
        {
            key: evaluation.get(key)
            for key in (
                "method", "estimated", "evaluated_observation_count",
                "total_observation_count", "sampling_replicate_count",
                "score_minimum", "score_maximum",
                "score_population_standard_deviation",
                "configured_random_seeds", "evaluated_random_seeds",
            )
            if key in evaluation
        }
        if isinstance(evaluation, dict) else {}
    )
    stability = report.get("silhouette_selection_stability")
    compact_stability = (
        {
            key: stability.get(key)
            for key in (
                "gate_applied", "status", "observed_winner_agreement_fraction",
                "configured_random_seeds", "evaluated_random_seeds",
                "winning_k_counts", "selected_k",
            )
            if key in stability
        }
        if isinstance(stability, dict) else {}
    )
    return [{
        "kind": "cluster_populations",
        "title": "Selected clustering populations",
        "algorithm": str(report.get("module_id", "clustering")),
        "cluster_sizes": sizes,
        "silhouette": model.get("silhouette"),
        "silhouette_evidence": compact_evaluation,
        "silhouette_selection_stability": compact_stability,
        "model": {
            key: model.get(key)
            for key in (
                "k", "seed", "iteration_count", "inertia", "silhouette",
                "mean_adjusted_rand_to_best",
            )
            if key in model
        },
    }]


def _rmsf_visuals(report: Mapping[str, object]) -> List[Dict[str, object]]:
    systems = report.get("systems")
    if not isinstance(systems, list):
        return []
    visuals = []
    for system in systems:
        if not isinstance(system, dict) or not isinstance(system.get("atom_statistics"), list):
            continue
        residues: Dict[tuple[str, int, str, str], List[float]] = {}
        for atom in system["atom_statistics"]:
            if not isinstance(atom, dict):
                continue
            value = atom.get("frame_pooled_rmsf_angstrom", atom.get("rmsf_angstrom"))
            number = atom.get("residue_number")
            if not _finite_number(value) or isinstance(number, bool) or not isinstance(number, int):
                continue
            key = (
                str(atom.get("chain_id", "")), number,
                str(atom.get("insertion_code", "")),
                str(atom.get("residue_name", "UNK")),
            )
            residues.setdefault(key, []).append(float(value))
        rows = [
            {
                "chain_id": key[0], "residue_number": key[1],
                "insertion_code": key[2], "residue_name": key[3],
                "mean_rmsf_angstrom": sum(values) / len(values),
            }
            for key, values in sorted(residues.items())
        ]
        if rows:
            visuals.append({
                "kind": "rmsf",
                "title": f"Residue-mean RMSF: {system.get('system_id')}",
                "system_id": str(system.get("system_id")),
                "residues": rows,
            })
    return visuals


def _dccm_visuals(report: Mapping[str, object], maximum_atoms: int = 180) -> List[Dict[str, object]]:
    atoms = report.get("analysis_atoms")
    systems = report.get("systems")
    if not isinstance(atoms, list) or not isinstance(systems, list) or not atoms:
        return []
    stride = max(1, math.ceil(len(atoms) / maximum_atoms))
    indices = list(range(0, len(atoms), stride))[:maximum_atoms]
    labels = []
    for index in indices:
        atom = atoms[index]
        if isinstance(atom, dict):
            labels.append(
                f"{atom.get('chain_id', '_')}:{atom.get('residue_name', 'UNK')}"
                f"{atom.get('residue_number', '?')}:{atom.get('atom_name', '?')}"
            )
        else:
            labels.append(str(index))
    visuals = []
    for system in systems:
        if not isinstance(system, dict):
            continue
        payload = system.get("frame_pooled_dccm")
        matrix = payload.get("matrix") if isinstance(payload, dict) else None
        if not isinstance(matrix, list) or len(matrix) < len(atoms):
            continue
        reduced = []
        valid = True
        for left in indices:
            row = matrix[left]
            if not isinstance(row, list) or len(row) < len(atoms):
                valid = False
                break
            reduced.append([
                float(row[right]) if _finite_number(row[right]) else None
                for right in indices
            ])
        if valid:
            visuals.append({
                "kind": "dccm",
                "title": f"DCCM: {system.get('system_id')}",
                "system_id": str(system.get("system_id")),
                "matrix": reduced,
                "labels": labels,
                "source_atom_count": len(atoms),
                "display_stride": stride,
            })
    return visuals


_DEFAULT_OFF_EXPERIMENTAL_MODULES = frozenset(DEFAULT_DISABLED_MODULES)


def _experimental_visuals(
    module_id: str, report: Mapping[str, object]
) -> List[Dict[str, object]]:
    metrics = _key_metrics(report)[:16]
    visuals: List[Dict[str, object]] = [{
        "kind": "experimental_method_summary",
        "title": module_id.replace("_", " ").title(),
        "module_id": module_id,
        "technical_status": str(report.get("technical_status", "unknown")),
        "availability_status": str(report.get("availability_status", "available")),
        "scientific_status": str(report.get("scientific_status", "not evaluated")),
        "metrics": metrics,
        "interpretation": "Default-off experimental method; descriptive preview only.",
    }]
    if module_id == "hydration_density_channels":
        rows = report.get("density_projections_xy")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get("matrix"), list):
                    visuals.append({
                        "kind": "spatial_density",
                        "title": (
                            f"Aligned {row.get('species')} density: "
                            f"{row.get('system_id')}"
                        ),
                        "system_id": str(row.get("system_id")),
                        "species": str(row.get("species")),
                        "matrix": row["matrix"],
                        "normalization": row.get("normalization"),
                    })
    elif module_id == "ensemble_pocket_dynamics":
        rows = report.get("pocket_clusters")
        if isinstance(rows, list):
            labels, values = [], []
            for row in rows[:40]:
                if not isinstance(row, dict):
                    continue
                occupancies = row.get("per_system_occupancy")
                if not isinstance(occupancies, list):
                    continue
                for occupancy in occupancies:
                    value = occupancy.get("occupancy_fraction") if isinstance(occupancy, dict) else None
                    if _finite_number(value):
                        labels.append(
                            f"{row.get('pocket_cluster_id')} · {occupancy.get('system_id')}"
                        )
                        values.append(float(value))
            if values:
                visuals.append({
                    "kind": "experimental_metric_bars",
                    "title": "Geometric pocket persistence by system",
                    "labels": labels, "values": values,
                    "value_label": "evaluated-frame occupancy",
                })
    elif module_id == "interaction_fingerprints":
        rows = report.get("feature_occupancies")
        choices = [
            row for row in rows or [] if isinstance(row, dict)
            and _finite_number(row.get("occupancy_fraction"))
        ] if isinstance(rows, list) else []
        choices = sorted(
            choices, key=lambda row: (-float(row["occupancy_fraction"]), str(row.get("feature_id")))
        )[:30]
        if choices:
            visuals.append({
                "kind": "experimental_metric_bars",
                "title": "Most occupied interaction-fingerprint features",
                "labels": [str(row.get("feature_id")) for row in choices],
                "values": [float(row["occupancy_fraction"]) for row in choices],
                "value_label": "source-observed-frame occupancy",
            })
    elif module_id == "energetic_network_embeddings":
        comparisons = report.get("pairwise_system_comparisons")
        rows = []
        if isinstance(comparisons, list):
            for comparison in comparisons:
                if not isinstance(comparison, dict):
                    continue
                for row in comparison.get("residue_distances", []):
                    if isinstance(row, dict) and _finite_number(
                        row.get("summed_wasserstein_distance")
                    ):
                        rows.append({
                            **row,
                            "system_i": comparison.get("system_i"),
                            "system_j": comparison.get("system_j"),
                        })
        choices = sorted(
            rows,
            key=lambda row: -float(row["summed_wasserstein_distance"]),
        )[:30]
        if choices:
            visuals.append({
                "kind": "experimental_metric_bars",
                "title": "Largest residue energetic-network embedding shifts",
                "labels": [
                    f"{row.get('node_id')} · {row.get('system_i')} vs {row.get('system_j')}"
                    for row in choices
                ],
                "values": [
                    float(row["summed_wasserstein_distance"])
                    for row in choices
                ],
                "value_label": "summed marginal Wasserstein distance",
            })
    elif module_id == "spatial_interaction_ensembles":
        summaries = report.get("spatial_ensemble_summaries")
        observations = report.get("point_observations")
        summary_rows = [
            row for row in summaries if isinstance(row, dict)
            and row.get("spatial_summary_gate") == "passed"
            and _finite_number(row.get("rms_radius_angstrom"))
        ] if isinstance(summaries, list) else []
        choices = sorted(
            summary_rows,
            key=lambda row: (
                -int(row.get("point_observation_count", 0)),
                str(row.get("system_id")), str(row.get("superfeature_id")),
            ),
        )[:20]
        if choices:
            visuals.append({
                "kind": "experimental_metric_bars",
                "title": "Spatial interaction-cloud spread",
                "labels": [
                    f"{row.get('superfeature_id')} · {row.get('system_id')}"
                    for row in choices
                ],
                "values": [float(row["rms_radius_angstrom"]) for row in choices],
                "value_label": "receptor-aligned RMS radius (Å)",
            })
        if isinstance(observations, list):
            for summary in choices[:6]:
                values = [
                    row for row in observations if isinstance(row, dict)
                    and str(row.get("system_id")) == str(summary.get("system_id"))
                    and str(row.get("superfeature_id"))
                    == str(summary.get("superfeature_id"))
                    and isinstance(row.get("coordinate_angstrom"), list)
                    and len(row["coordinate_angstrom"]) == 3
                ]
                stride = max(1, math.ceil(len(values) / 600))
                shown = values[::stride][:600]
                if shown:
                    visuals.append({
                        "kind": "spatial_interaction_cloud",
                        "title": (
                            f"Interaction cloud: {summary.get('system_id')} · "
                            f"{summary.get('superfeature_id')}"
                        ),
                        "points_xy_angstrom": [
                            [float(row["coordinate_angstrom"][0]),
                             float(row["coordinate_angstrom"][1])]
                            for row in shown
                        ],
                        "centroid_xy_angstrom": [
                            float(summary["centroid_angstrom"][0]),
                            float(summary["centroid_angstrom"][1]),
                        ],
                        "display_stride": stride,
                        "source_point_count": len(values),
                        "mode_inference_status": summary.get(
                            "mode_inference_status"
                        ),
                    })
    elif module_id == "interaction_persistence":
        rows = report.get("feature_persistence_summaries")
        choices = [
            row for row in rows or [] if isinstance(row, dict)
            and row.get("gap_tolerance_observations") == 0
            and row.get("persistence_summary_gate") == "passed"
            and isinstance(row.get("complete_event_duration_summary"), dict)
            and _finite_number(
                row["complete_event_duration_summary"].get("median")
            )
        ] if isinstance(rows, list) else []
        choices = sorted(
            choices,
            key=lambda row: (
                -float(row["complete_event_duration_summary"]["median"]),
                str(row.get("feature_id")), str(row.get("system_id")),
            ),
        )[:30]
        if choices:
            units = sorted({str(row.get("time_unit")) for row in choices})
            unit = units[0] if len(units) == 1 else "physical-time units"
            visuals.append({
                "kind": "experimental_metric_bars",
                "title": "Primary zero-gap interaction persistence",
                "labels": [
                    f"{row.get('feature_id')} · {row.get('system_id')}"
                    for row in choices
                ],
                "values": [
                    float(row["complete_event_duration_summary"]["median"])
                    for row in choices
                ],
                "value_label": f"median complete-event duration ({unit})",
            })
    elif module_id == "random_feature_koopman":
        rows = report.get("hyperparameter_candidates")
        choices = [
            row for row in rows or [] if isinstance(row, dict)
            and _finite_number(row.get("mean_seed_heldout_vamp_e"))
        ] if isinstance(rows, list) else []
        choices = sorted(
            choices,
            key=lambda row: (
                -float(row["mean_seed_heldout_vamp_e"]),
                int(row.get("random_feature_count", 0)),
                float(row.get("bandwidth_scale", 0.0)),
            ),
        )[:30]
        if choices:
            visuals.append({
                "kind": "experimental_metric_bars",
                "title": "Random-feature Koopman candidate validation",
                "labels": [
                    f"p={row.get('random_feature_count')} · "
                    f"bandwidth×{row.get('bandwidth_scale')}"
                    + (" · gate passed" if row.get("eligible") else " · gate failed")
                    for row in choices
                ],
                "values": [
                    float(row["mean_seed_heldout_vamp_e"])
                    for row in choices
                ],
                "value_label": "mean held-out VAMP-E across prespecified seeds",
            })
    return visuals


def _module_visuals(module_id: str, report: Mapping[str, object]) -> List[Dict[str, object]]:
    if module_id == "pca_fes_basins":
        return _fes_visuals(report)
    if module_id in {"clustering_kmeans", "clustering_imwkmeans"}:
        return _clustering_visuals(report)
    if module_id == "pooled_rmsf":
        return _rmsf_visuals(report)
    if module_id == "dccm":
        return _dccm_visuals(report)
    if module_id in _DEFAULT_OFF_EXPERIMENTAL_MODULES:
        return _experimental_visuals(module_id, report)
    return []


def _report_record(path: Path, root: Path) -> Dict[str, object]:
    size_bytes = path.stat().st_size
    relative = _relative(path, root)
    if size_bytes > _MAXIMUM_REPORT_PARSE_BYTES:
        sidecar_path = Path(str(path) + ".summary.json")
        sidecar = _optional_json(sidecar_path)
        if not isinstance(sidecar, dict):
            sidecar = {}
        module_id = str(sidecar.get("module_id", path.parent.name))
        return {
            "module_id": module_id,
            "title": module_id.replace("_", " ").title(),
            "technical_status": str(sidecar.get("technical_status", "unknown")),
            "scientific_status": str(
                sidecar.get("scientific_status", "not evaluated")
            ),
            "path": relative,
            "href": _raw_link(relative),
            "size_bytes": size_bytes,
            "sha256": _sha256_file(path),
            "issues": [{
                "severity": "info",
                "code": "INTERACTIVE_REPORT_PREVIEW_OMITTED",
                "message": (
                    "The raw report exceeds the bounded interactive-parser size; "
                    "its hash, compact sidecar, and raw link are retained."
                ),
            }],
            "limitations": [],
            "key_metrics": _key_metrics(sidecar),
            "visuals": [],
            "preview": _preview(sidecar),
        }
    report = load_json(path)
    if not isinstance(report, dict):
        raise InteractiveReportError(f"module report is not a JSON object: {path}")
    module_id = str(report.get("module_id", path.parent.name))
    issues = [row for row in report.get("issues", []) if isinstance(row, dict)]
    limitations = [
        str(row) for row in report.get("limitations", []) if isinstance(row, str)
    ]
    return {
        "module_id": module_id,
        "title": module_id.replace("_", " ").title(),
        "technical_status": str(report.get("technical_status", "unknown")),
        "scientific_status": str(report.get("scientific_status", "not evaluated")),
        "path": relative,
        "href": _raw_link(relative),
        "size_bytes": size_bytes,
        "sha256": _sha256_file(path),
        "issues": issues,
        "limitations": limitations,
        "key_metrics": _key_metrics(report),
        "visuals": _module_visuals(module_id, report),
        "preview": _preview(report),
    }


def _structure_records(
    root: Path,
    *,
    maximum_structures: int,
    maximum_total_bytes: int,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    candidates = sorted({
        *root.glob("results/**/*.pdb"),
        *root.glob("results/**/*.ent"),
        *root.glob("*.rmsf_bfactor.pdb"),
    })
    included: List[Dict[str, object]] = []
    omitted: List[Dict[str, object]] = []
    total_bytes = 0
    for path in candidates:
        relative = _relative(path, root)
        size = path.stat().st_size
        relative_parts = Path(relative).parts
        state_part = next(
            (part for part in relative_parts if re.fullmatch(r"state-\d+", part)),
            None,
        )
        state_index = relative_parts.index(state_part) if state_part else -1
        method_part = relative_parts[state_index - 1] if state_index > 0 else None
        system_part = next(
            (part[7:] for part in relative_parts if part.startswith("system-")),
            None,
        )
        display_name = path.stem
        if state_part:
            state_label = state_part.replace("-", " ").title()
            method_label = (
                method_part.replace("_", " ").replace("-", " ").title()
                if method_part else "State"
            )
            display_name = f"{method_label} · {state_label}"
        record = {
            "structure_id": f"structure-{len(included) + len(omitted) + 1:05d}",
            "name": display_name,
            "path": relative,
            "href": _raw_link(relative),
            "size_bytes": size,
            "sha256": _sha256_file(path),
            "state_id": state_part,
            "system_id": system_part,
            "method_id": method_part,
            "module_id": relative_parts[1]
            if len(relative_parts) > 1 and relative_parts[0] == "results"
            else None,
        }
        if len(included) >= maximum_structures or total_bytes + size > maximum_total_bytes:
            omitted.append({**record, "reason": "interactive inline-asset limit"})
            continue
        record["pdb_text"] = path.read_text(encoding="utf-8", errors="replace")
        included.append(record)
        total_bytes += size
    return included, omitted


def _figure_records(
    root: Path, *, maximum_total_bytes: int
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    candidates = sorted({
        path
        for extension in _IMAGE_MEDIA_TYPES
        for path in root.glob(f"results/**/*{extension}")
    })
    included: List[Dict[str, object]] = []
    omitted: List[Dict[str, object]] = []
    total_bytes = 0
    for path in candidates:
        relative = _relative(path, root)
        size = path.stat().st_size
        record = {
            "name": path.stem,
            "path": relative,
            "href": _raw_link(relative),
            "size_bytes": size,
            "sha256": _sha256_file(path),
            "media_type": _IMAGE_MEDIA_TYPES[path.suffix.lower()],
        }
        if total_bytes + size > maximum_total_bytes:
            omitted.append({**record, "reason": "interactive inline-asset limit"})
            continue
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        record["data_uri"] = f"{record['media_type']};base64,{payload}"
        included.append(record)
        total_bytes += size
    return included, omitted


def _project_title(root: Path, project: object, system: object) -> str:
    for payload in (project, system):
        if isinstance(payload, dict):
            for key in ("project_id", "campaign_id", "system_id", "title"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return root.name


def _system_ids(system: object, project: object) -> List[str]:
    values = []
    for payload in (system, project):
        if not isinstance(payload, dict):
            continue
        systems = payload.get("systems")
        if isinstance(systems, list):
            for row in systems:
                if isinstance(row, dict) and isinstance(row.get("system_id"), str):
                    values.append(str(row["system_id"]))
        if isinstance(payload.get("system_id"), str):
            values.append(str(payload["system_id"]))
    return sorted(set(values))


def _resource_rows_with_sampling(
    rows: Sequence[Mapping[str, object]],
    sampling: object,
    views: object,
) -> List[Dict[str, object]]:
    """Attach explicit integer-stride context to measured resource rows."""

    direct_strides: Dict[str, int] = {}
    if isinstance(sampling, dict) and isinstance(sampling.get("method_plans"), list):
        for plan in sampling["method_plans"]:
            if not isinstance(plan, dict):
                continue
            module_id = plan.get("module_id")
            stride = plan.get("frame_stride")
            if isinstance(module_id, str) and isinstance(stride, int) and stride > 0:
                direct_strides[module_id] = stride

    view_strides: Dict[str, int] = {}
    if isinstance(views, dict) and isinstance(views.get("views"), list):
        for view in views["views"]:
            if not isinstance(view, dict):
                continue
            output_root = view.get("analysis_output_root")
            plan = view.get("resource_plan")
            selection = plan.get("basis_frame_selection") if isinstance(plan, dict) else None
            stride = selection.get("stride") if isinstance(selection, dict) else None
            if isinstance(output_root, str) and isinstance(stride, int) and stride > 0:
                view_strides[output_root.rstrip("/") + "/"] = stride

    annotated: List[Dict[str, object]] = []
    for source in rows:
        row = dict(source)
        module_id = str(row.get("module_id", ""))
        report_path = str(row.get("report_path", "")).replace(os.sep, "/")
        basis_stride = next(
            (
                stride for prefix, stride in view_strides.items()
                if report_path.startswith(prefix) or f"/{prefix}" in report_path
            ),
            None,
        )
        source_frames = row.get("source_physical_frames_available")
        selected_frames = row.get("selected_source_physical_frames")
        analysis_stride = direct_strides.get(module_id)
        if basis_stride is not None:
            row["basis_frame_stride"] = basis_stride
            if module_id == "common_pca":
                analysis_stride = basis_stride
        if analysis_stride is None and (
            isinstance(source_frames, int)
            and isinstance(selected_frames, int)
            and source_frames == selected_frames
        ):
            analysis_stride = 1
        if analysis_stride is not None:
            row["analysis_frame_stride"] = analysis_stride
        annotated.append(row)
    return annotated


def _collect_data(
    root: Path,
    *,
    title: str | None,
    maximum_inline_structures: int,
    maximum_inline_structure_bytes: int,
    maximum_inline_figure_bytes: int,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    project = _optional_json(root / "project.json")
    system = _optional_json(root / "system.json")
    config = _optional_json(root / "analysis-config.json")
    preflight = _optional_json(root / "preflight.report.json")
    findings = _optional_json(root / "prioritized_findings.json")
    resources = _optional_json(root / "analysis_resource_and_frame_table.json")
    coverage = _optional_json(root / "module-coverage.json")
    chemistry = _optional_json(root / "automatic-chemical-context.json")
    views = _optional_json(root / "conformational-views.json")
    sampling = _optional_json(root / "sampling-plan.json")

    report_paths = {
        *(
            (root / "results").glob("**/report.json")
            if (root / "results").is_dir() else ()
        ),
        *root.glob("*-availability.json"),
    }
    reports = [
        _report_record(path, root) for path in sorted(report_paths)
    ]
    structures, omitted_structures = _structure_records(
        root,
        maximum_structures=maximum_inline_structures,
        maximum_total_bytes=maximum_inline_structure_bytes,
    )
    figures, omitted_figures = _figure_records(
        root, maximum_total_bytes=maximum_inline_figure_bytes
    )
    qc_issues: List[Dict[str, object]] = []
    if isinstance(preflight, dict):
        qc_issues.extend(
            {**row, "source": "preflight.report.json"}
            for row in preflight.get("issues", []) if isinstance(row, dict)
        )
    for report in reports:
        qc_issues.extend(
            {**row, "source": report["path"], "module_id": report["module_id"]}
            for row in report["issues"] if isinstance(row, dict)
        )
    severity_order = {"error": 0, "warning": 1, "info": 2}
    qc_issues.sort(key=lambda row: (
        severity_order.get(str(row.get("severity", "info")), 9),
        str(row.get("module_id", "")), str(row.get("code", "")),
    ))
    finding_rows = (
        [row for row in findings.get("findings", []) if isinstance(row, dict)]
        if isinstance(findings, dict) else []
    )
    raw_resource_rows = (
        [row for row in resources.get("rows", []) if isinstance(row, dict)]
        if isinstance(resources, dict) else []
    )
    resource_rows = _resource_rows_with_sampling(raw_resource_rows, sampling, views)
    status_counts: Dict[str, int] = {}
    for report in reports:
        key = str(report["technical_status"])
        status_counts[key] = status_counts.get(key, 0) + 1

    source_records = [
        {
            "path": report["path"],
            "size_bytes": report["size_bytes"],
            "sha256": report["sha256"],
        }
        for report in reports
    ]
    data = {
        "interactive_report_schema": "salsbury-interactive-results-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": title or _project_title(root, project, system),
        "technical_status": (
            "attention_required" if status_counts.get("failed", 0) else "complete"
        ),
        "scientific_status": "not evaluated",
        "scientific_boundary": (
            "Technical completion and automated ranking do not establish convergence, "
            "mechanism, causality, biological importance, or scientific validity."
        ),
        "system_ids": _system_ids(system, project),
        "status_counts": status_counts,
        "findings": finding_rows,
        "finding_metadata": _preview(findings) if isinstance(findings, dict) else None,
        "reports": reports,
        "resources": resource_rows,
        "resource_metadata": _preview(resources) if isinstance(resources, dict) else None,
        "qc_issues": qc_issues,
        "structures": structures,
        "omitted_structures": omitted_structures,
        "figures": figures,
        "omitted_figures": omitted_figures,
        "module_coverage": _preview(coverage),
        "chemical_context": _preview(chemistry),
        "conformational_views": _preview(views),
        "sampling_plan": _preview(sampling),
        "configuration": _preview(config),
        "project_manifest": _preview(project),
        "system_manifest": _preview(system),
        "preflight": _preview(preflight),
        "raw_links": {
            "findings_json": "../prioritized_findings.json",
            "resources_json": "../analysis_resource_and_frame_table.json",
            "module_coverage": "../module-coverage.json",
            "analysis_config": "../analysis-config.json",
            "preflight": "../preflight.report.json",
            "sampling_plan": "../sampling-plan.json",
        },
    }
    return data, source_records


_CSS = r"""
:root{--ink:#18211d;--muted:#617069;--paper:#f7f5ef;--card:#fff;--line:#d9ddd7;--forest:#173f35;--teal:#1c7166;--gold:#d69b35;--red:#a43d32;--blue:#3767a6;--shadow:0 10px 35px rgba(25,44,36,.08)}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,input,select{font:inherit}a{color:var(--teal)}
.shell{display:grid;grid-template-columns:250px minmax(0,1fr);min-height:100vh}.sidebar{background:var(--forest);color:#eef5f1;padding:25px 18px;position:sticky;top:0;height:100vh;overflow:auto}.brand{font:700 22px/1.1 ui-serif,Georgia,serif;margin:0 8px 8px}.subtitle{color:#b9cbc4;font-size:12px;margin:0 8px 25px}.nav button{display:block;width:100%;border:0;background:transparent;color:#dceae4;text-align:left;padding:10px 12px;border-radius:10px;margin:2px 0;cursor:pointer}.nav button.active,.nav button:hover{background:#285b4e;color:#fff}.sidebar .boundary{font-size:11px;color:#c7d7d0;border-top:1px solid #3c655a;margin-top:24px;padding:18px 8px}
.main{padding:32px clamp(20px,4vw,64px);max-width:1600px}.topline{display:flex;gap:15px;align-items:flex-start;justify-content:space-between;margin-bottom:22px}.topline h1{font:700 clamp(28px,4vw,46px)/1.05 ui-serif,Georgia,serif;margin:0}.eyebrow{text-transform:uppercase;letter-spacing:.12em;color:var(--teal);font-size:12px;font-weight:800}.status{border-radius:999px;padding:7px 12px;background:#e4efe9;color:var(--forest);font-weight:700;white-space:nowrap}.status.failed,.severity-error{background:#fae5e2;color:var(--red)}.status.warning,.severity-warning{background:#fff1d7;color:#875c0b}
.view{display:none}.view.active{display:block}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0 24px}.stat,.card{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}.stat{padding:16px}.stat strong{font-size:26px;display:block}.stat span{font-size:12px;color:var(--muted)}.card{padding:20px;margin-bottom:18px}.card h2,.card h3{font-family:ui-serif,Georgia,serif;margin-top:0}.card h2{font-size:25px}.card h3{font-size:19px}.muted{color:var(--muted)}.boundary-banner{border-left:5px solid var(--gold);background:#fff8e9;padding:14px 18px;border-radius:10px;margin:18px 0}
.finding{display:grid;grid-template-columns:42px minmax(0,1fr);gap:12px;padding:15px 0;border-bottom:1px solid var(--line)}.finding:last-child{border:0}.rank{width:36px;height:36px;border-radius:50%;background:#e6f0ec;color:var(--forest);display:grid;place-items:center;font-weight:800}.badges{display:flex;gap:6px;flex-wrap:wrap;margin-top:7px}.badge{font-size:11px;border-radius:999px;padding:3px 8px;background:#edf0ee;color:#46534d}.badge.sig{background:#dcebd8;color:#2f642a}.filters{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0 20px}.filters input,.filters select{border:1px solid var(--line);background:#fff;border-radius:9px;padding:9px 11px;min-width:180px}
.issue{padding:12px 14px;border-left:4px solid var(--line);background:#fafafa;margin:9px 0;border-radius:7px}.issue.error{border-color:var(--red)}.issue.warning{border-color:var(--gold)}.issue code{font-size:11px}.module-row{border-bottom:1px solid var(--line);padding:12px 0}.module-row summary{cursor:pointer;font-weight:700;display:flex;justify-content:space-between;gap:12px}.metric-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:7px;margin:12px 0}.metric{background:#f4f6f3;padding:8px 10px;border-radius:7px;font-size:12px}.json{background:#17231e;color:#dce9e2;padding:14px;border-radius:9px;overflow:auto;max-height:460px;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
.chart{width:100%;min-height:300px;border:1px solid var(--line);border-radius:12px;background:#fff}.chart svg,.chart canvas{display:block;width:100%;height:auto}.chart-note{font-size:12px;color:var(--muted);margin-top:8px}.visual-controls{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}.visual-controls select{padding:7px;border:1px solid var(--line);border-radius:8px;background:white}
.molecule-layout{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(250px,.7fr);gap:16px}.viewer{background:#0e1714;border-radius:15px;overflow:hidden;position:relative;min-height:520px}.viewer canvas{width:100%;height:520px;display:block;cursor:grab}.viewer canvas:active{cursor:grabbing}.viewer-tools{display:flex;gap:7px;flex-wrap:wrap;padding:10px;background:#17231e}.viewer-tools select,.viewer-tools input,.viewer-tools button{background:#f5f8f6;border:0;border-radius:7px;padding:7px}.viewer-info{position:absolute;left:12px;bottom:12px;background:rgba(0,0,0,.68);color:#fff;padding:8px 10px;border-radius:7px;font-size:12px;max-width:85%}.structure-list{max-height:585px;overflow:auto}.structure-item{padding:9px;border-bottom:1px solid var(--line);cursor:pointer}.structure-item.active{background:#e7f1ed}.structure-item small{display:block;color:var(--muted)}
.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}th{position:sticky;top:0;background:#f0f3ef}.figure img{max-width:100%;height:auto;border-radius:10px}.empty{padding:30px;text-align:center;border:1px dashed #b8c0ba;border-radius:12px;color:var(--muted)}
@media(max-width:850px){.shell{display:block}.sidebar{position:static;height:auto}.nav{display:flex;overflow:auto}.nav button{width:auto;white-space:nowrap}.sidebar .boundary{display:none}.main{padding:22px 15px}.molecule-layout{grid-template-columns:1fr}.topline{display:block}.status{display:inline-block;margin-top:12px}}
@media print{.sidebar,.filters,.viewer-tools{display:none!important}.shell{display:block}.main{padding:0}.view{display:block!important;break-before:page}.card{box-shadow:none;break-inside:avoid}}
"""


_JS = r"""
const DATA=JSON.parse(document.getElementById('report-data').textContent);
const $=(s,e=document)=>e.querySelector(s);const $$=(s,e=document)=>[...e.querySelectorAll(s)];
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=v=>typeof v==='number'?(Math.abs(v)>=1000?v.toLocaleString():Number(v.toPrecision(5)).toString()):String(v??'—');
function go(name){$$('.view').forEach(v=>v.classList.toggle('active',v.id===`view-${name}`));$$('.nav button').forEach(b=>b.classList.toggle('active',b.dataset.view===name));location.hash=name;window.scrollTo(0,0)}
$$('.nav button').forEach(b=>b.onclick=()=>go(b.dataset.view));
function badge(text,cls=''){return `<span class="badge ${cls}">${esc(text)}</span>`}
function findingHTML(f,i){const q=f.statistically_significant===true?'statistically significant after correction':(f.evidence_level||'descriptive');return `<div class="finding" data-category="${esc(f.category)}" data-systems="${esc((f.system_ids||[]).join(' '))}" data-search="${esc(JSON.stringify(f).toLowerCase())}"><div class="rank">${i+1}</div><div><div>${esc(f.statement)}</div><div class="badges">${badge(f.category)}${badge(f.module_id)}${badge(q,f.statistically_significant===true?'sig':'')}${(f.system_ids||[]).map(s=>badge(s)).join('')}</div><div class="muted" style="font-size:11px;margin-top:5px">Effect: ${fmt(f.effect_value)} · <a href="${esc(f.report_path?('../'+String(f.report_path).replace(/^.*?results\//,'results/')):'#')}" target="_blank">raw evidence</a></div></div></div>`}
function renderFindings(){const host=$('#findings-list');host.innerHTML=DATA.findings.length?DATA.findings.map(findingHTML).join(''):'<div class="empty">No ranked findings were generated.</div>';const cats=[...new Set(DATA.findings.map(f=>f.category))].sort();const systems=[...new Set(DATA.findings.flatMap(f=>f.system_ids||[]))].sort();$('#finding-category').innerHTML='<option value="">All categories</option>'+cats.map(v=>`<option>${esc(v)}</option>`).join('');$('#finding-system').innerHTML='<option value="">All systems</option>'+systems.map(v=>`<option>${esc(v)}</option>`).join('');const filter=()=>{const q=$('#finding-search').value.toLowerCase(),c=$('#finding-category').value,s=$('#finding-system').value;$$('.finding',host).forEach(row=>row.hidden=!!((q&&!row.dataset.search.includes(q))||(c&&row.dataset.category!==c)||(s&&!row.dataset.systems.split(' ').includes(s))))};['finding-search','finding-category','finding-system'].forEach(id=>$(`#${id}`).oninput=filter)}
function issueHTML(i){const sev=String(i.severity||'info').toLowerCase();return `<div class="issue ${esc(sev)}"><strong>${esc(sev.toUpperCase())}</strong> ${i.code?`<code>${esc(i.code)}</code>`:''}<div>${esc(i.message||i.reason||JSON.stringify(i))}</div><small class="muted">${esc(i.module_id||'')} ${esc(i.source||'')} ${esc(i.location||'')}</small></div>`}
function renderOverview(){const complete=DATA.status_counts.complete||0,failed=DATA.status_counts.failed||0;$('#stats').innerHTML=[['Module reports',DATA.reports.length],['Complete',complete],['Need attention',failed],['Ranked findings',DATA.findings.length],['Structures',DATA.structures.length],['Systems',DATA.system_ids.length||1]].map(([a,b])=>`<div class="stat"><strong>${fmt(b)}</strong><span>${esc(a)}</span></div>`).join('');$('#overview-findings').innerHTML=DATA.findings.slice(0,8).map(findingHTML).join('')||'<div class="empty">No findings available.</div>';const urgent=DATA.qc_issues.filter(i=>['error','warning'].includes(String(i.severity).toLowerCase())).slice(0,8);$('#overview-qc').innerHTML=urgent.map(issueHTML).join('')||'<div class="empty">No error or warning issues were indexed.</div>'}
function moduleHTML(r){const metrics=(r.key_metrics||[]).map(m=>`<div class="metric"><strong>${esc(m.label)}</strong><br>${esc(fmt(m.value))}</div>`).join('');return `<details class="module-row" data-search="${esc((r.module_id+' '+JSON.stringify(r.issues)).toLowerCase())}"><summary><span>${esc(r.title)}</span><span>${badge(r.technical_status,r.technical_status==='failed'?'severity-error':'')}</span></summary><p class="muted">Scientific status: ${esc(r.scientific_status)} · ${(r.size_bytes/1024).toFixed(1)} KiB · <a href="${esc(r.href)}" target="_blank">open raw JSON</a></p><div class="metric-list">${metrics||'<span class="muted">No compact numeric metrics indexed.</span>'}</div>${(r.issues||[]).map(issueHTML).join('')}<details><summary>Bounded JSON preview</summary><pre class="json">${esc(JSON.stringify(r.preview,null,2))}</pre></details>${(r.limitations||[]).length?`<details><summary>Scientific limitations</summary><ul>${r.limitations.map(x=>`<li>${esc(x)}</li>`).join('')}</ul></details>`:''}</details>`}
function renderModules(){const host=$('#module-list');host.innerHTML=DATA.reports.map(moduleHTML).join('')||'<div class="empty">No module reports found.</div>';$('#module-search').oninput=()=>{const q=$('#module-search').value.toLowerCase();$$('.module-row',host).forEach(r=>r.hidden=q&&!r.dataset.search.includes(q))}}
function colorScale(t){t=Math.max(0,Math.min(1,t));const stops=[[42,30,73],[40,98,130],[35,151,138],[132,196,103],[244,222,55]];const p=t*(stops.length-1),i=Math.min(stops.length-2,Math.floor(p)),f=p-i;return `rgb(${stops[i].map((v,j)=>Math.round(v+(stops[i+1][j]-v)*f)).join(',')})`}
function renderFES(v,host){const land=v.landscape||{},grid=land.grid||[];if(!grid.length){host.innerHTML='<div class="empty">No FES grid.</div>';return}const nx=1+Math.max(...grid.map(r=>r.x_bin)),ny=1+Math.max(...grid.map(r=>r.y_bin));const field=grid.some(r=>typeof r.relative_free_energy_kcal_per_mol==='number')?'relative_free_energy_kcal_per_mol':'relative_occupancy_score';const vals=grid.map(r=>r[field]).filter(Number.isFinite),max=Math.max(...vals,1e-9);const c=document.createElement('canvas');c.width=Math.max(500,nx*12);c.height=Math.max(420,ny*12);c.className='chart';const x=c.getContext('2d'),pad=48,w=c.width-2*pad,h=c.height-2*pad;x.fillStyle='#fff';x.fillRect(0,0,c.width,c.height);grid.forEach(r=>{const val=r[field];x.fillStyle=Number.isFinite(val)?colorScale(1-val/max):'#eceeea';const cw=w/nx,ch=h/ny;x.fillRect(pad+r.x_bin*cw,pad+(ny-1-r.y_bin)*ch,cw+.4,ch+.4)});x.strokeStyle='#26362f';x.strokeRect(pad,pad,w,h);x.fillStyle='#18211d';x.font='13px sans-serif';x.fillText('PC '+(DATA.reports.find(r=>r.module_id==='pca_fes_basins')?.preview?.pca_basis?.x_component??1),pad+w/2-20,c.height-10);x.save();x.translate(13,pad+h/2+20);x.rotate(-Math.PI/2);x.fillText('PC '+(DATA.reports.find(r=>r.module_id==='pca_fes_basins')?.preview?.pca_basis?.y_component??2),0,0);x.restore();(land.basins||[]).forEach(b=>{const cw=w/nx,ch=h/ny,px=pad+(b.root_x_bin+.5)*cw,py=pad+(ny-1-b.root_y_bin+.5)*ch;x.beginPath();x.arc(px,py,8,0,Math.PI*2);x.fillStyle='rgba(255,255,255,.9)';x.fill();x.strokeStyle='#111';x.stroke();x.fillStyle='#111';x.font='bold 10px sans-serif';x.textAlign='center';x.fillText(String(b.basin_id),px,py+3)});c.title=`${field}; click raw report for cell values`;host.appendChild(c);host.insertAdjacentHTML('beforeend',`<div class="chart-note">${esc(v.title)}. Color shows ${esc(field.replaceAll('_',' '))}; labeled circles mark basin roots. Basin identities and smoothing are descriptive analysis choices.</div>`)}
function renderClusters(v,host){const sizes=v.cluster_sizes||[],total=sizes.reduce((a,b)=>a+b,0)||1,w=700,h=340,p=48;let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Cluster populations">`;const bw=(w-2*p)/Math.max(1,sizes.length);sizes.forEach((n,i)=>{const bh=(h-2*p)*n/Math.max(...sizes,1),x=p+i*bw+5,y=h-p-bh;svg+=`<rect x="${x}" y="${y}" width="${Math.max(4,bw-10)}" height="${bh}" fill="#1c7166"><title>Cluster ${i+1}: ${n} (${(100*n/total).toFixed(1)}%)</title></rect><text x="${x+(bw-10)/2}" y="${h-p+17}" text-anchor="middle" font-size="11">${i+1}</text>`});svg+=`<text x="${w/2}" y="${h-8}" text-anchor="middle" font-size="12">Cluster</text></svg>`;const ev=v.silhouette_evidence||{},gate=v.silhouette_selection_stability||{},sampling=ev.estimated?`Estimated over ${fmt(ev.sampling_replicate_count)} recorded random samples; range ${fmt(ev.score_minimum)}–${fmt(ev.score_maximum)}.`:`Exact over ${fmt(ev.evaluated_observation_count)} observations; configured sampling seeds were unused.`,gateText=gate.status?` Selection stability: ${esc(gate.status.replaceAll('_',' '))}.`:'';host.innerHTML=svg+`<div class="chart-note">Silhouette: ${fmt(v.silhouette)}. ${sampling}${gateText} Populations use complete assignments where the source method supports them.</div>`}
function renderRMSF(v,host){const rows=v.residues||[],w=900,h=360,p=50;if(!rows.length){host.innerHTML='<div class="empty">No RMSF rows.</div>';return}const max=Math.max(...rows.map(r=>r.mean_rmsf_angstrom),1e-9),step=Math.max(1,Math.ceil(rows.length/1000)),shown=rows.filter((_,i)=>i%step===0);const pts=shown.map((r,i)=>`${p+i*(w-2*p)/Math.max(1,shown.length-1)},${h-p-r.mean_rmsf_angstrom*(h-2*p)/max}`).join(' ');host.innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Residue RMSF"><line x1="${p}" y1="${p}" x2="${p}" y2="${h-p}" stroke="#777"/><line x1="${p}" y1="${h-p}" x2="${w-p}" y2="${h-p}" stroke="#777"/><polyline points="${pts}" fill="none" stroke="#3767a6" stroke-width="2"/><text x="15" y="${h/2}" transform="rotate(-90 15 ${h/2})" font-size="12">RMSF (Å)</text><text x="${w/2}" y="${h-10}" text-anchor="middle" font-size="12">Residue order</text></svg><div class="chart-note">${esc(v.title)}; residue values are means over mapped atom RMSFs. Display stride ${step}.</div>`}
function renderDCCM(v,host){const m=v.matrix||[],n=m.length;if(!n){host.innerHTML='<div class="empty">No DCCM matrix.</div>';return}const c=document.createElement('canvas');c.width=Math.max(480,n*3);c.height=c.width;const x=c.getContext('2d'),cell=c.width/n;m.forEach((row,i)=>row.forEach((z,j)=>{const q=Number.isFinite(z)?z:0;x.fillStyle=q<0?`rgb(${Math.round(255*(1+q))},${Math.round(255*(1+q))},255)`:`rgb(255,${Math.round(255*(1-q))},${Math.round(255*(1-q))})`;x.fillRect(i*cell,(n-1-j)*cell,cell+.5,cell+.5)}));c.className='chart';host.appendChild(c);host.insertAdjacentHTML('beforeend',`<div class="chart-note">${esc(v.title)}; blue is anticorrelation and red is positive correlation. ${v.source_atom_count} source atoms; display stride ${v.display_stride}.</div>`)}
function renderExperimentalSummary(v,host){const metrics=v.metrics||[];host.innerHTML=`<div class="badges">${badge(v.technical_status)}${badge(v.availability_status)}${badge(v.scientific_status)}</div><p>${esc(v.interpretation)}</p><div class="metric-list">${metrics.map(m=>`<div class="metric"><strong>${esc(m.label)}</strong><br>${esc(fmt(m.value))}</div>`).join('')||'<span class="muted">No compact numeric result is available; inspect availability and issues in All analyses.</span>'}</div>`}
function renderMetricBars(v,host){const values=v.values||[],labels=v.labels||[];if(!values.length){host.innerHTML='<div class="empty">No finite values.</div>';return}const w=900,h=Math.max(300,values.length*25+70),p=230,max=Math.max(...values.map(Math.abs),1e-12);let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(v.title)}">`;values.forEach((value,i)=>{const y=20+i*25,width=Math.abs(value)*(w-p-45)/max;svg+=`<text x="${p-8}" y="${y+13}" text-anchor="end" font-size="10">${esc(String(labels[i]??i).slice(0,42))}</text><rect x="${p}" y="${y}" width="${width}" height="17" fill="#1c7166"><title>${esc(labels[i])}: ${fmt(value)}</title></rect>`});svg+=`</svg>`;host.innerHTML=svg+`<div class="chart-note">${esc(v.value_label||'descriptive value')}. Review the raw report for definitions and uncertainty limits.</div>`}
function renderSpatialDensity(v,host){const m=v.matrix||[],nx=m.length,ny=nx&&Array.isArray(m[0])?m[0].length:0;if(!nx||!ny){host.innerHTML='<div class="empty">No density projection.</div>';return}const vals=m.flat().filter(Number.isFinite),max=Math.max(...vals,1e-12),c=document.createElement('canvas');c.width=Math.max(480,nx*5);c.height=Math.max(420,ny*5);const x=c.getContext('2d'),cw=c.width/nx,ch=c.height/ny;m.forEach((row,i)=>row.forEach((value,j)=>{x.fillStyle=Number.isFinite(value)?colorScale(value/max):'#eceeea';x.fillRect(i*cw,(ny-1-j)*ch,cw+.5,ch+.5)}));c.className='chart';host.appendChild(c);host.insertAdjacentHTML('beforeend',`<div class="chart-note">${esc(v.normalization)}. This aligned projection is descriptive density, not transport or free energy.</div>`)}
function renderSpatialInteractionCloud(v,host){const pts=v.points_xy_angstrom||[];if(!pts.length){host.innerHTML='<div class="empty">No spatial points.</div>';return}const w=700,h=520,p=45,xv=pts.map(q=>q[0]),yv=pts.map(q=>q[1]),cx=v.centroid_xy_angstrom?.[0],cy=v.centroid_xy_angstrom?.[1],xmin=Math.min(...xv,cx),xmax=Math.max(...xv,cx),ymin=Math.min(...yv,cy),ymax=Math.max(...yv,cy),dx=Math.max(xmax-xmin,1e-9),dy=Math.max(ymax-ymin,1e-9),px=x=>p+(x-xmin)*(w-2*p)/dx,py=y=>h-p-(y-ymin)*(h-2*p)/dy;let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Aligned interaction cloud">`;pts.forEach(q=>{svg+=`<circle cx="${px(q[0])}" cy="${py(q[1])}" r="3" fill="#1c7166" fill-opacity=".35"/>`});svg+=`<circle cx="${px(cx)}" cy="${py(cy)}" r="7" fill="#d69b35" stroke="#18211d"><title>Centroid</title></circle><line x1="${p}" y1="${h-p}" x2="${w-p}" y2="${h-p}" stroke="#777"/><line x1="${p}" y1="${p}" x2="${p}" y2="${h-p}" stroke="#777"/><text x="${w/2}" y="${h-8}" text-anchor="middle" font-size="12">aligned x (Å)</text><text x="14" y="${h/2}" transform="rotate(-90 14 ${h/2})" text-anchor="middle" font-size="12">aligned y (Å)</text></svg>`;host.innerHTML=svg+`<div class="chart-note">XY display of ${fmt(pts.length)} of ${fmt(v.source_point_count)} exact aligned points (display stride ${fmt(v.display_stride)}); gold marks the centroid. Mode status: ${esc(String(v.mode_inference_status||'not evaluated').replaceAll('_',' '))}. This projection is not a binding-state or free-energy assignment.</div>`}
function renderVisuals(){const visuals=DATA.reports.flatMap(r=>(r.visuals||[]).map(v=>({...v,module_id:r.module_id})));const select=$('#visual-kind');const kinds=[...new Set(visuals.map(v=>v.kind))];select.innerHTML='<option value="">All generated views</option>'+kinds.map(k=>`<option value="${esc(k)}">${esc(k.replaceAll('_',' '))}</option>`).join('');function draw(){const host=$('#visual-list'),kind=select.value;host.innerHTML='';visuals.filter(v=>!kind||v.kind===kind).forEach(v=>{const card=document.createElement('section');card.className='card';card.innerHTML=`<h3>${esc(v.title)}</h3><div class="chart"></div>`;host.appendChild(card);const target=$('.chart',card);target.className='';if(v.kind==='fes')renderFES(v,target);else if(v.kind==='cluster_populations')renderClusters(v,target);else if(v.kind==='rmsf')renderRMSF(v,target);else if(v.kind==='dccm')renderDCCM(v,target);else if(v.kind==='experimental_method_summary')renderExperimentalSummary(v,target);else if(v.kind==='experimental_metric_bars')renderMetricBars(v,target);else if(v.kind==='spatial_density')renderSpatialDensity(v,target);else if(v.kind==='spatial_interaction_cloud')renderSpatialInteractionCloud(v,target)});if(!host.children.length)host.innerHTML='<div class="empty">No structured visualization is available for this filter. The complete raw reports remain in All analyses.</div>'}select.oninput=draw;draw()}
function renderFigures(){const host=$('#figure-list');host.innerHTML=DATA.figures.map(f=>`<section class="card figure"><h3>${esc(f.name)}</h3><img src="data:${f.data_uri}" alt="${esc(f.name)}"><p><a href="${esc(f.href)}" target="_blank">Open source figure</a></p></section>`).join('')||'<div class="empty">No pre-rendered image files were found. Structured plots appear under Molecular states & figures.</div>'}
function renderResources(){const rows=DATA.resources||[],host=$('#resource-table');if(!rows.length){host.innerHTML='<div class="empty">No consolidated resource/frame table was found.</div>';return}const wanted=['module_id','technical_status','total_cpu_seconds','wall_seconds','maximum_resident_memory_mib','selected_source_physical_frames','analysis_frame_stride','basis_frame_stride','symmetry_expanded_observations','model_fit_observations','full_assignment_observations'];const cols=wanted.filter(k=>rows.some(r=>k in r));host.innerHTML=`<table><thead><tr>${cols.map(c=>`<th>${esc(c.replaceAll('_',' '))}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${cols.map(c=>`<td>${esc(fmt(r[c]))}</td>`).join('')}</tr>`).join('')}</tbody></table>`}
function renderQC(){const host=$('#qc-list');host.innerHTML=DATA.qc_issues.map(issueHTML).join('')||'<div class="empty">No issues were indexed.</div>';$('#provenance-json').textContent=JSON.stringify({module_coverage:DATA.module_coverage,chemical_context:DATA.chemical_context,conformational_views:DATA.conformational_views,sampling_plan:DATA.sampling_plan,configuration:DATA.configuration,project_manifest:DATA.project_manifest,system_manifest:DATA.system_manifest,preflight:DATA.preflight,omitted_structures:DATA.omitted_structures,omitted_figures:DATA.omitted_figures},null,2)}
const viewer={atoms:[],yaw:.4,pitch:-.2,zoom:1,panX:0,panY:0,drag:false,lastX:0,lastY:0,projected:[]};
const elemColors={H:'#e7ecea',C:'#a9b3ae',N:'#4b77d1',O:'#d94a45',S:'#e3b63b',P:'#e27731',ZN:'#8b6ab8',MG:'#44b78b',K:'#b55ec5',NA:'#6c77d8',CA:'#3ba86a',CL:'#4db04f',FE:'#c66a35'};
const polymerResidues=new Set('ALA ARG ASN ASP CYS GLN GLU GLY HIS HSD HSE HSP ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL ACE NME A C G U T DA DC DG DT DU ADE CYT GUA THY URA RA RC RG RU'.split(' '));
const waterResidues=new Set('HOH WAT TIP3 TIP3P SOL H2O'.split(' '));
function parsePDB(text){const atoms=[];text.split(/\r?\n/).forEach((line,i)=>{const rec=line.slice(0,6).trim();if(!['ATOM','HETATM'].includes(rec))return;const x=+line.slice(30,38),y=+line.slice(38,46),z=+line.slice(46,54);if(![x,y,z].every(Number.isFinite))return;let el=line.slice(76,78).trim().toUpperCase()||line.slice(12,16).trim().replace(/[0-9]/g,'').slice(0,2).toUpperCase();if(el.length>1&&!['ZN','MG','NA','CL','FE','CA'].includes(el))el=el[0];atoms.push({i,rec,name:line.slice(12,16).trim(),res:line.slice(17,20).trim(),chain:line.slice(21,22).trim()||'_',resid:+line.slice(22,26),x,y,z,b:+line.slice(60,66)||0,el})});if(!atoms.length)return atoms;const cx=atoms.reduce((s,a)=>s+a.x,0)/atoms.length,cy=atoms.reduce((s,a)=>s+a.y,0)/atoms.length,cz=atoms.reduce((s,a)=>s+a.z,0)/atoms.length;atoms.forEach(a=>{a.x-=cx;a.y-=cy;a.z-=cz});return atoms}
function isHetero(a){return !waterResidues.has(a.res)&&(a.rec==='HETATM'||['ZN','MG','K','NA','CA','CL','FE'].includes(a.el)||!polymerResidues.has(a.res))}
function atomFilter(a){const rep=$('#viewer-representation').value,showH=$('#viewer-h').checked,q=$('#viewer-search').value.trim().toUpperCase(),hetero=isHetero(a);if(!showH&&a.el==='H')return false;if(rep==='overview'&&!['CA','P'].includes(a.name)&&!hetero)return false;if(rep==='backbone'&&!['CA','P'].includes(a.name))return false;if(rep==='hetero'&&!hetero)return false;if(q&&!`${a.chain}:${a.res}${a.resid}:${a.name} ${a.el}`.toUpperCase().includes(q))return false;return true}
function rotate(a){const cy=Math.cos(viewer.yaw),sy=Math.sin(viewer.yaw),cp=Math.cos(viewer.pitch),sp=Math.sin(viewer.pitch),x=cy*a.x+sy*a.z,z=-sy*a.x+cy*a.z,y=cp*a.y-sp*z;return{x,y,z:sp*a.y+cp*z}}
function atomColor(a,minB,maxB){const mode=$('#viewer-color').value;if(mode==='chain'){let h=0;for(const c of a.chain)h=(h*31+c.charCodeAt(0))%360;return`hsl(${h} 55% 58%)`}if(mode==='bfactor'){const t=(a.b-minB)/Math.max(1e-9,maxB-minB);return`rgb(${Math.round(45+210*t)},${Math.round(90+80*(1-t))},${Math.round(210-150*t)})`}return elemColors[a.el]||'#d0a8c9'}
function drawMolecule(){const c=$('#molecule-canvas'),dpr=window.devicePixelRatio||1,w=c.clientWidth||800,h=c.clientHeight||520;if(c.width!==Math.round(w*dpr)||c.height!==Math.round(h*dpr)){c.width=Math.round(w*dpr);c.height=Math.round(h*dpr)}const x=c.getContext('2d');x.setTransform(dpr,0,0,dpr,0,0);x.fillStyle='#0e1714';x.fillRect(0,0,w,h);const filtered=viewer.atoms.filter(atomFilter),rot=filtered.map(a=>({a,...rotate(a)}));if(!rot.length){viewer.projected=[];x.fillStyle='#dce9e2';x.fillText('No atoms match this view.',20,30);$('#viewer-info').textContent='No atoms match the current representation and search.';return}let extent=1,minB=Infinity,maxB=-Infinity;rot.forEach(p=>extent=Math.max(extent,Math.abs(p.x),Math.abs(p.y)));filtered.forEach(a=>{minB=Math.min(minB,a.b);maxB=Math.max(maxB,a.b)});const scale=.42*Math.min(w,h)/extent*viewer.zoom;viewer.projected=rot.map(p=>({...p,sx:w/2+viewer.panX+p.x*scale,sy:h/2+viewer.panY-p.y*scale})).sort((a,b)=>a.z-b.z);if(['overview','backbone'].includes($('#viewer-representation').value)){const traces={};viewer.projected.filter(p=>['CA','P'].includes(p.a.name)).forEach(p=>(traces[p.a.chain]??=[]).push(p));x.lineWidth=2;x.strokeStyle='#8eb8aa';Object.values(traces).forEach(t=>{x.beginPath();t.forEach((p,i)=>i?x.lineTo(p.sx,p.sy):x.moveTo(p.sx,p.sy));x.stroke()})}viewer.projected.forEach(p=>{const r=Math.max(1.5,(['ZN','MG','K','NA','CA','FE'].includes(p.a.el)?5:3)*(0.75+0.25*(p.z/extent+1)));x.beginPath();x.arc(p.sx,p.sy,r,0,Math.PI*2);x.fillStyle=atomColor(p.a,minB,maxB);x.fill()});$('#viewer-info').textContent=`${filtered.length.toLocaleString()} of ${viewer.atoms.length.toLocaleString()} atoms · drag to rotate · wheel to zoom · click an atom for identity`}
function loadStructure(index){const s=DATA.structures[index];if(!s)return;viewer.atoms=parsePDB(s.pdb_text);viewer.yaw=.4;viewer.pitch=-.2;viewer.zoom=1;viewer.panX=viewer.panY=0;$('#structure-title').textContent=s.name;$$('.structure-item').forEach((e,i)=>e.classList.toggle('active',i===index));drawMolecule()}
function renderMolecules(){const list=$('#structure-list');list.innerHTML=DATA.structures.map((s,i)=>`<div class="structure-item" data-index="${i}"><strong>${esc(s.name)}</strong><small>${esc(s.system_id||s.module_id||'structure')} · ${(s.size_bytes/1024).toFixed(1)} KiB · <a href="${esc(s.href)}" target="_blank">PDB</a></small></div>`).join('')||'<div class="empty">No PDB representative structures were found.</div>';$$('.structure-item',list).forEach(e=>e.addEventListener('click',ev=>{if(ev.target.tagName!=='A')loadStructure(+e.dataset.index)}));const c=$('#molecule-canvas');c.addEventListener('pointerdown',e=>{viewer.drag=true;viewer.lastX=e.clientX;viewer.lastY=e.clientY;c.setPointerCapture(e.pointerId)});c.addEventListener('pointermove',e=>{if(!viewer.drag)return;viewer.yaw+=(e.clientX-viewer.lastX)*.01;viewer.pitch+=(e.clientY-viewer.lastY)*.01;viewer.lastX=e.clientX;viewer.lastY=e.clientY;drawMolecule()});c.addEventListener('pointerup',()=>viewer.drag=false);c.addEventListener('wheel',e=>{e.preventDefault();viewer.zoom*=Math.exp(-e.deltaY*.001);viewer.zoom=Math.max(.15,Math.min(12,viewer.zoom));drawMolecule()},{passive:false});c.addEventListener('click',e=>{if(!viewer.projected.length)return;const r=c.getBoundingClientRect(),px=e.clientX-r.left,py=e.clientY-r.top;let best=null,dist=144;viewer.projected.forEach(p=>{const d=(p.sx-px)**2+(p.sy-py)**2;if(d<dist){dist=d;best=p}});if(best)$('#viewer-info').textContent=`${best.a.chain}:${best.a.res}${best.a.resid}:${best.a.name} · ${best.a.el} · B ${best.a.b.toFixed(2)}`});['viewer-representation','viewer-color','viewer-h','viewer-search'].forEach(id=>{const control=$(`#${id}`);control.addEventListener('input',drawMolecule);control.addEventListener('change',drawMolecule)});$('#viewer-reset').addEventListener('click',()=>loadStructure($$('.structure-item').findIndex(e=>e.classList.contains('active'))));if(DATA.structures.length)loadStructure(0);if('ResizeObserver' in window)new ResizeObserver(drawMolecule).observe(c);else window.addEventListener('resize',drawMolecule)}
renderOverview();renderFindings();renderModules();renderVisuals();renderFigures();renderResources();renderQC();renderMolecules();go((location.hash||'#overview').slice(1));
"""


def _render_html(data: Mapping[str, object]) -> str:
    encoded = json.dumps(
        _json_safe(data),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    encoded = encoded.replace("</script", "<\\/script").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    title = html.escape(str(data["title"]))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data: 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; object-src 'none'; frame-src 'none'; connect-src 'none'; media-src 'self'">
<title>{title} — interactive molecular analysis</title><style>{_CSS}</style></head>
<body><div class="shell"><aside class="sidebar"><div class="brand">Salsbury MD Analysis</div><div class="subtitle">Interactive results · v0.2 candidate</div><nav class="nav">
<button data-view="overview">Overview</button><button data-view="findings">Key findings</button><button data-view="states">Molecular states & figures</button><button data-view="molecules">Molecular structures</button><button data-view="analyses">All analyses</button><button data-view="resources">Resources & sampling</button><button data-view="qc">QC & provenance</button></nav><div class="boundary">Technical success and automated prioritization are not scientific validation. Review the source reports, sampling, convergence, and chemistry.</div></aside>
<main class="main"><div class="topline"><div><div class="eyebrow">Analysis campaign</div><h1>{title}</h1><div class="muted">{html.escape(', '.join(data.get('system_ids', [])) or 'single or inferred system')}</div></div><span class="status">{html.escape(str(data['technical_status']))}</span></div>
<div class="boundary-banner"><strong>Interpretation boundary.</strong> {html.escape(str(data['scientific_boundary']))}</div>
<section id="view-overview" class="view"><div id="stats" class="stats"></div><section class="card"><h2>Highest-priority findings</h2><p class="muted">Ranked by the documented presentation contract. Descriptive rankings are not claims of biological importance.</p><div id="overview-findings"></div><p><button onclick="go('findings')">Review all ranked findings</button></p></section><section class="card"><h2>QC requiring attention</h2><div id="overview-qc"></div></section></section>
<section id="view-findings" class="view"><section class="card"><h2>Key findings</h2><div class="filters"><input id="finding-search" placeholder="Search findings"><select id="finding-category"></select><select id="finding-system"></select></div><div id="findings-list"></div></section></section>
<section id="view-states" class="view"><section class="card"><h2>Molecular states & generated figures</h2><p class="muted">FES surfaces preserve smoothing identity; clustering panels preserve method and silhouette evidence.</p><div class="visual-controls"><select id="visual-kind"></select></div></section><div id="visual-list"></div><h2>Pre-rendered figures</h2><div id="figure-list" class="grid"></div></section>
<section id="view-molecules" class="view"><section class="card"><h2>Representative molecular structures</h2><p class="muted">Offline point/trace viewer. CA/P lines are visual guides, not inferred chemical bonds. Download the source PDB for full molecular-software inspection.</p><div class="molecule-layout"><div class="viewer"><div class="viewer-tools"><select id="viewer-representation"><option value="overview">Macromolecule + ligands/ions</option><option value="all">All atoms</option><option value="backbone">CA/P trace</option><option value="hetero">Ligands/ions/cofactors</option></select><select id="viewer-color"><option value="element">Color by element</option><option value="chain">Color by chain</option><option value="bfactor">Color by B factor</option></select><label style="color:white"><input id="viewer-h" type="checkbox"> H</label><input id="viewer-search" placeholder="A:CYS54:SG"><button id="viewer-reset">Reset</button></div><canvas id="molecule-canvas" width="900" height="520"></canvas><div id="viewer-info" class="viewer-info"></div></div><div><h3 id="structure-title">Structures</h3><div id="structure-list" class="structure-list"></div></div></div></section></section>
<section id="view-analyses" class="view"><section class="card"><h2>All analyses</h2><p class="muted">Every indexed module is listed whether or not it produced a ranked finding.</p><div class="filters"><input id="module-search" placeholder="Search modules and issues"></div><div id="module-list"></div></section></section>
<section id="view-resources" class="view"><section class="card"><h2>Resources, frames, and sampling</h2><div id="resource-table" class="table-wrap"></div></section></section>
<section id="view-qc" class="view"><section class="card"><h2>QC issues</h2><div id="qc-list"></div></section><section class="card"><h2>Configuration and provenance</h2><p><a href="../analysis-config.json" target="_blank">Resolved configuration</a> · <a href="../module-coverage.json" target="_blank">Module coverage</a> · <a href="../preflight.report.json" target="_blank">Preflight report</a></p><pre id="provenance-json" class="json"></pre></section></section>
</main></div><script id="report-data" type="application/json">{encoded}</script><script>{_JS}</script></body></html>"""


def _validate_existing(target: Path) -> Dict[str, object]:
    manifest_path = target / "manifest.json"
    index_path = target / "index.html"
    if not manifest_path.is_file() or not index_path.is_file():
        raise InteractiveReportError(
            f"interactive-report directory exists without complete evidence: {target}"
        )
    manifest = load_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("technical_status") != "complete"
        or manifest.get("index_sha256") != _sha256_file(index_path)
    ):
        raise InteractiveReportError(
            f"existing interactive report is incomplete or hash-mismatched: {target}"
        )
    return {**manifest, "reused": True}


def build_interactive_report(
    root: Path,
    *,
    output_name: str = "interactive-report",
    title: str | None = None,
    maximum_inline_structures: int = 100,
    maximum_inline_structure_bytes: int = 50_000_000,
    maximum_inline_figure_bytes: int = 25_000_000,
) -> Dict[str, object]:
    """Build an immutable, offline interactive result under an analysis root."""

    analysis_root = Path(root).expanduser().resolve(strict=True)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", output_name):
        raise InteractiveReportError("output_name must be one safe path component")
    for label, value in (
        ("maximum_inline_structures", maximum_inline_structures),
        ("maximum_inline_structure_bytes", maximum_inline_structure_bytes),
        ("maximum_inline_figure_bytes", maximum_inline_figure_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InteractiveReportError(f"{label} must be a nonnegative integer")
    target = analysis_root / output_name
    if target.exists():
        return _validate_existing(target)
    temporary = analysis_root / f".{output_name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise InteractiveReportError(f"temporary interactive-report path exists: {temporary}")
    temporary.mkdir()
    try:
        data, source_records = _collect_data(
            analysis_root,
            title=title,
            maximum_inline_structures=maximum_inline_structures,
            maximum_inline_structure_bytes=maximum_inline_structure_bytes,
            maximum_inline_figure_bytes=maximum_inline_figure_bytes,
        )
        html_text = _render_html(data)
        index_path = temporary / "index.html"
        index_path.write_text(html_text, encoding="utf-8")
        manifest = {
            "interactive_report_manifest_schema": "salsbury-interactive-results-manifest-v1",
            "technical_status": "complete",
            "scientific_status": "not evaluated",
            "generated_at_utc": data["generated_at_utc"],
            "index_path": f"{output_name}/index.html",
            "index_sha256": _sha256_file(index_path),
            "index_size_bytes": index_path.stat().st_size,
            "module_report_count": len(data["reports"]),
            "finding_count": len(data["findings"]),
            "inline_structure_count": len(data["structures"]),
            "omitted_structure_count": len(data["omitted_structures"]),
            "inline_figure_count": len(data["figures"]),
            "omitted_figure_count": len(data["omitted_figures"]),
            "source_report_records": source_records,
            "network_dependency": "none",
            "interpretation": data["scientific_boundary"],
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
        return {
            **manifest,
            "output_directory": str(target),
            "index_path": str(target / "index.html"),
            "manifest_path": str(target / "manifest.json"),
            "reused": False,
        }
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
