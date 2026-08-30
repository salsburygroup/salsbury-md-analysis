"""Measured execution-resource provenance and consolidated analysis tables."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

from .manifests import load_json
from .analysis_config import COMMAND_MODULES


class ExecutionResourceError(ValueError):
    """Raised when execution evidence is missing or malformed."""


_REPORTING_ONLY_MODULES = frozenset({
    "integrated_comparison",
    "rmsf_permutation_inference",
})


def _maximum_rss_mib(raw: float) -> float:
    # Linux reports KiB; macOS reports bytes.
    return raw / (1024.0 * 1024.0) if sys.platform == "darwin" else raw / 1024.0


def run_instrumented_project_command(
    command: str, project_path: Path, *, hash_content: bool
) -> Dict[str, object]:
    """Run one normal CLI analysis in a child and attach same-job measurements."""

    source = Path(project_path).expanduser().resolve(strict=False)
    argv = [sys.executable, "-m", "salsbury_md_analysis", command, str(source)]
    if hash_content:
        argv.append("--hash-content")
    started = time.perf_counter()
    completed = subprocess.run(argv, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ExecutionResourceError(
            f"instrumented command did not emit one JSON report; exit={completed.returncode}; "
            f"stderr={completed.stderr[-2000:]}"
        ) from exc
    if not isinstance(report, dict):
        raise ExecutionResourceError("instrumented command JSON must be an object")
    report["execution_resources"] = {
        "measurement_schema": "salsbury-execution-resources-v1",
        "computer_hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_cluster_name": os.environ.get("SLURM_CLUSTER_NAME"),
        "requested_cpu_count": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        "requested_memory": os.environ.get("SLURM_MEM_PER_NODE")
        or os.environ.get("SLURM_MEM_PER_CPU"),
        "requested_time_limit": os.environ.get("SLURM_TIMELIMIT"),
        "wall_seconds": elapsed,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "total_cpu_seconds": usage.ru_utime + usage.ru_stime,
        "maximum_resident_memory_mib": _maximum_rss_mib(float(usage.ru_maxrss)),
        "child_exit_code": completed.returncode,
        "stderr_nonempty": bool(completed.stderr.strip()),
        "stderr_tail": completed.stderr[-4000:] if completed.stderr.strip() else None,
        "measurement_scope": "one fresh child process for one analysis command",
    }
    project = load_json(source) if source.is_file() else None
    physical_count, observation_count = _observation_counts(report, project)
    report["planner_benchmark"] = {
        "technical_status": report.get("technical_status"),
        "module_id": report.get("module_id", command),
        "project_sha256": report.get("project_manifest_sha256"),
        "report_sha256": None,
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_executable": sys.executable,
        },
        "frame_coverage": {
            "estimator_selected_frame_count": physical_count,
            "symmetry_expanded_observation_count": observation_count,
            "work_unit_warning": (
                "physical frames are the frame-planner unit; member-expanded or "
                "quadratic methods also require their observation-specific planner"
            ),
        },
        "resources": {
            "wall_seconds": elapsed,
            "maximum_rss_kib": _maximum_rss_mib(float(usage.ru_maxrss)) * 1024.0,
        },
        "report_size_bytes": len(completed.stdout.encode("utf-8")),
        "workload_signature_sha256": report.get("workload_signature_sha256"),
        "full_assignment_observation_count": report.get(
            "observation_count", observation_count
        ),
    }
    return report


def run_instrumented_coordinate_cache(
    system_path: Path,
    output_directory: Path,
    *,
    maximum_workers: int,
    cache_stride: int = 1,
) -> Dict[str, object]:
    """Build one coordinate cache and attach whole-job resource measurements."""

    source = Path(system_path).expanduser().resolve(strict=True)
    output = Path(output_directory).expanduser().resolve(strict=False)
    if (
        isinstance(maximum_workers, bool)
        or not isinstance(maximum_workers, int)
        or maximum_workers <= 0
    ):
        raise ExecutionResourceError("maximum_workers must be a positive integer")
    if (
        isinstance(cache_stride, bool)
        or not isinstance(cache_stride, int)
        or cache_stride <= 0
    ):
        raise ExecutionResourceError("cache_stride must be a positive integer")
    argv = [
        sys.executable, "-m", "salsbury_md_analysis", "build-coordinate-cache",
        str(source), "--output", str(output), "--workers", str(maximum_workers),
        "--cache-stride", str(cache_stride),
    ]
    started = time.perf_counter()
    completed = subprocess.run(argv, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ExecutionResourceError(
            "instrumented coordinate cache did not emit one JSON report; "
            f"exit={completed.returncode}; stderr={completed.stderr[-2000:]}"
        ) from exc
    if not isinstance(report, dict):
        raise ExecutionResourceError("coordinate cache JSON must be an object")
    rows = report.get("rows")
    decoded_count = 0
    retained_count = 0
    if isinstance(rows, list):
        for row in rows:
            segments = row.get("segments") if isinstance(row, dict) else None
            if isinstance(segments, list):
                decoded_count += sum(
                    int(segment.get("decoded_frame_count", segment["frame_count"]))
                    for segment in segments
                    if isinstance(segment, dict)
                    and isinstance(segment.get("frame_count"), int)
                )
                retained_count += sum(
                    int(segment["frame_count"])
                    for segment in segments
                    if isinstance(segment, dict)
                    and isinstance(segment.get("frame_count"), int)
                )
    report.update({
        "module_id": "coordinate_cache",
        "system_manifest_path": str(source),
        "cache_output_directory": str(output),
        "observation_accounting": {
            "decoded_physical_frame_count": decoded_count,
            "selected_physical_frame_count": retained_count,
            "symmetry_expanded_observation_count": retained_count,
            "subsampling_triggered": cache_stride > 1,
            "frame_selection": {
                "mode": "integer_stride_per_replica_v1",
                "stride": cache_stride,
                "unwrapping_scan": "all source frames",
            },
        },
        "execution_resources": {
            "measurement_schema": "salsbury-execution-resources-v1",
            "computer_hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "slurm_cluster_name": os.environ.get("SLURM_CLUSTER_NAME"),
            "requested_cpu_count": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
            "requested_memory": os.environ.get("SLURM_MEM_PER_NODE")
            or os.environ.get("SLURM_MEM_PER_CPU"),
            "requested_time_limit": os.environ.get("SLURM_TIMELIMIT"),
            "wall_seconds": elapsed,
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
            "total_cpu_seconds": usage.ru_utime + usage.ru_stime,
            "maximum_resident_memory_mib": _maximum_rss_mib(float(usage.ru_maxrss)),
            "child_exit_code": completed.returncode,
            "stderr_nonempty": bool(completed.stderr.strip()),
            "stderr_tail": completed.stderr[-4000:] if completed.stderr.strip() else None,
            "measurement_scope": (
                "one cache builder child including its replica-parallel descendants"
            ),
        },
    })
    report["planner_benchmark"] = {
        "technical_status": report.get("technical_status"),
        "module_id": "coordinate_cache",
        "project_sha256": report.get("source_system_manifest_sha256"),
        "report_sha256": None,
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_executable": sys.executable,
        },
        "frame_coverage": {
            "source_decoded_frame_count": decoded_count,
            "estimator_selected_frame_count": retained_count,
            "symmetry_expanded_observation_count": retained_count,
        },
        "resources": {
            "wall_seconds": elapsed,
            "maximum_rss_kib": _maximum_rss_mib(float(usage.ru_maxrss)) * 1024.0,
        },
        "report_size_bytes": len(completed.stdout.encode("utf-8")),
        "full_assignment_observation_count": retained_count,
    }
    return report


def _recursive_segment_counts(value: object) -> tuple[int, int]:
    physical = 0
    observations = 0
    if isinstance(value, dict):
        if "segment_id" in value:
            physical_value = value.get(
                "physical_evaluated_frame_count", value.get("evaluated_frame_count")
            )
            observation_value = value.get("evaluated_member_observation_count")
            if isinstance(physical_value, int) and not isinstance(physical_value, bool):
                physical += physical_value
            if isinstance(observation_value, int) and not isinstance(observation_value, bool):
                observations += observation_value
        for child in value.values():
            child_physical, child_observations = _recursive_segment_counts(child)
            physical += child_physical
            observations += child_observations
    elif isinstance(value, list):
        for child in value:
            child_physical, child_observations = _recursive_segment_counts(child)
            physical += child_physical
            observations += child_observations
    return physical, observations


def _symmetry_multiplier(
    report: Mapping[str, object], project: Mapping[str, object] | None = None
) -> int:
    candidates = [report.get("symmetry_expansion")]
    settings = report.get("settings")
    feature_contract = report.get("feature_contract")
    feature_lineage = report.get("feature_lineage")
    if isinstance(settings, dict):
        candidates.append(settings.get("symmetry_expansion"))
    if isinstance(feature_contract, dict):
        candidates.append(feature_contract.get("symmetry_expansion"))
    if isinstance(feature_lineage, dict):
        common_settings = feature_lineage.get("common_pca_settings")
        if isinstance(common_settings, dict):
            candidates.append(common_settings.get("symmetry_expansion"))
    if isinstance(project, dict):
        definitions = project.get("definitions")
        common = definitions.get("common_pca") if isinstance(definitions, dict) else None
        if isinstance(common, dict):
            candidates.append(common.get("symmetry_expansion"))
    for candidate in candidates:
        if isinstance(candidate, dict) and isinstance(candidate.get("member_count"), int):
            return max(1, int(candidate["member_count"]))
    return 1


def _top_level_segment_counts(
    report: Mapping[str, object], multiplier: int,
) -> tuple[int | None, int | None]:
    """Read one exact count per top-level physical trajectory segment."""

    rows = report.get("segment_reports")
    if not isinstance(rows, list) or not rows:
        return None, None
    counts: Dict[tuple[str, str, str], tuple[int, int]] = {}
    for row in rows:
        if not isinstance(row, dict) or "segment_id" not in row:
            return None, None
        physical = row.get(
            "physical_evaluated_frame_count", row.get("evaluated_frame_count")
        )
        if not isinstance(physical, int) or isinstance(physical, bool):
            return None, None
        observations = row.get("evaluated_member_observation_count")
        if not isinstance(observations, int) or isinstance(observations, bool):
            observations = physical * multiplier
        identity = (
            str(row.get("system_id", "")),
            str(row.get("replica_id", "")),
            str(row["segment_id"]),
        )
        previous = counts.get(identity)
        current = (physical, observations)
        if previous is not None and previous != current:
            return None, None
        counts[identity] = current
    return (
        sum(value[0] for value in counts.values()),
        sum(value[1] for value in counts.values()),
    )


def _nested_scalar_frame_counts(
    report: Mapping[str, object], multiplier: int,
) -> tuple[int | None, int | None]:
    """Deduplicate physical frames repeated across scalar features or states.

    Scalar-distribution and threshold reports intentionally contain one full
    assignment series per feature/state.  Their generic ``observation_count``
    is therefore feature-frames, not trajectory frames.  Count exact source
    frame identities across all series, retaining an explicit oligomer member
    only in the symmetry-expanded observation identity.
    """

    module_id = report.get("module_id")
    container_key = {
        "scalar_feature_distributions": "distribution_reports",
        "scalar_threshold_states": "state_reports",
    }.get(str(module_id))
    if container_key is None:
        return None, None
    containers = report.get(container_key)
    if not isinstance(containers, list) or not containers:
        return None, None
    physical_identities: set[tuple[str, str, str, int]] = set()
    observation_identities: set[tuple[str, str, str, int, str]] = set()
    explicit_member = False
    for container in containers:
        assignments = container.get("assignments") if isinstance(container, dict) else None
        if not isinstance(assignments, list):
            return None, None
        for row in assignments:
            frame = row.get("source_frame_index") if isinstance(row, dict) else None
            if not isinstance(frame, int) or isinstance(frame, bool):
                return None, None
            identity = (
                str(row.get("system_id", "")),
                str(row.get("replica_id", "")),
                str(row.get("segment_id", "")),
                frame,
            )
            physical_identities.add(identity)
            if "member_id" in row:
                explicit_member = True
                member_id = str(row["member_id"])
            else:
                member_id = ""
            observation_identities.add((*identity, member_id))
    if not physical_identities:
        return None, None
    observations = (
        len(observation_identities)
        if explicit_member
        else len(physical_identities) * multiplier
    )
    return len(physical_identities), observations


def _observation_counts(
    report: Mapping[str, object], project: Mapping[str, object] | None = None
) -> tuple[int | None, int | None]:
    # PCA has two distinct workloads: basis fitting and projection/assignment.
    # The projection stream is the downstream observation set and therefore the
    # primary selected-frame count.  Older generic fallback logic found the
    # basis selection first and silently reported that smaller count instead.
    if report.get("module_id") == "common_pca":
        projection = report.get("projection_frame_selection")
        if isinstance(projection, dict):
            physical = projection.get("selected_frame_count")
            if isinstance(physical, int) and not isinstance(physical, bool):
                multiplier = _symmetry_multiplier(report, project)
                return physical, physical * multiplier
    accounting = report.get("observation_accounting")
    if isinstance(accounting, dict):
        physical = next((
            accounting[key] for key in (
                "selected_physical_frame_count", "source_physical_frame_count",
                "physical_evaluated_frame_count", "projection_physical_frame_count",
            ) if isinstance(accounting.get(key), int)
        ), None)
        observations = next((
            accounting[key] for key in (
                "symmetry_expanded_observation_count", "exported_observation_count",
                "evaluated_member_observation_count", "observation_count",
            ) if isinstance(accounting.get(key), int)
        ), None)
        if physical is not None or observations is not None:
            return physical, observations if observations is not None else physical
    multiplier = _symmetry_multiplier(report, project)
    for key in ("evaluated_frame_count", "exported_frame_count"):
        value = report.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return max(1, value // multiplier), value
    feature_contract = report.get("feature_contract")
    if isinstance(feature_contract, dict) and isinstance(feature_contract.get("observation_accounting"), dict):
        nested = feature_contract["observation_accounting"]
        physical = nested.get("source_physical_frame_count")
        observations = nested.get("symmetry_expanded_observation_count")
        if isinstance(physical, int):
            return physical, observations if isinstance(observations, int) else physical
    frame_selection = report.get("frame_selection")
    if isinstance(frame_selection, dict):
        physical = frame_selection.get("selected_frame_count")
        if isinstance(physical, int) and not isinstance(physical, bool):
            # A module's generic ``observation_count`` may count atom-frames,
            # residue-frames, candidate evaluations, or surface points.  It is
            # not a physical-frame count and is not necessarily the oligomer
            # symmetry-expanded observation count used by the campaign planner.
            # Exact frame_selection provenance therefore precedes both an old
            # planner benchmark and a generic observation_count fallback.
            return physical, physical * multiplier
    segment_physical, segment_observations = _top_level_segment_counts(
        report, multiplier
    )
    if segment_physical is not None:
        return segment_physical, segment_observations
    scalar_physical, scalar_observations = _nested_scalar_frame_counts(
        report, multiplier
    )
    if scalar_physical is not None:
        return scalar_physical, scalar_observations
    planner_benchmark = report.get("planner_benchmark")
    if isinstance(planner_benchmark, dict):
        frame_coverage = planner_benchmark.get("frame_coverage")
        if isinstance(frame_coverage, dict):
            physical = frame_coverage.get("estimator_selected_frame_count")
            observations = frame_coverage.get("symmetry_expanded_observation_count")
            if isinstance(physical, int) and not isinstance(physical, bool):
                return (
                    physical,
                    observations
                    if isinstance(observations, int) and not isinstance(observations, bool)
                    else physical,
                )
    physical, observations = _recursive_segment_counts(report)
    if physical:
        return physical, observations or physical
    for key in ("assignments", "frame_assignments", "frame_identity"):
        rows = report.get(key)
        if isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows):
            identities = {
                (
                    str(row.get("system_id")), str(row.get("replica_id")),
                    str(row.get("segment_id")), int(row["source_frame_index"]),
                )
                for row in rows if isinstance(row.get("source_frame_index"), int)
            }
            if identities:
                return len(identities), len(rows)
    if isinstance(feature_contract, dict) and isinstance(feature_contract.get("observation_count"), int):
        value = int(feature_contract["observation_count"])
        return max(1, value // multiplier), value
    for key in ("replicas", "replica_reports"):
        rows = report.get(key)
        if not isinstance(rows, list) or not rows:
            continue
        row_physical = 0
        row_observations = 0
        complete = True
        for row in rows:
            if not isinstance(row, dict):
                complete = False
                break
            physical_value = row.get(
                "physical_evaluated_frame_count", row.get("evaluated_frame_count")
            )
            observation_value = row.get("observation_count")
            if isinstance(physical_value, int) and not isinstance(physical_value, bool):
                row_physical += physical_value
                row_observations += (
                    observation_value
                    if isinstance(observation_value, int) and not isinstance(observation_value, bool)
                    else physical_value * multiplier
                )
            elif isinstance(observation_value, int) and not isinstance(observation_value, bool):
                row_observations += observation_value
                row_physical += max(1, observation_value // multiplier)
            else:
                complete = False
                break
        if complete and row_physical:
            return row_physical, row_observations
    value = report.get("observation_count")
    if isinstance(value, int) and not isinstance(value, bool):
        return max(1, value // multiplier), value
    return None, None


def _analysis_workload_counts(
    report: Mapping[str, object], physical: int | None, observations: int | None
) -> Dict[str, int | None]:
    accounting = report.get("observation_accounting")
    feature_contract = report.get("feature_contract")
    if not isinstance(accounting, dict) and isinstance(feature_contract, dict):
        candidate = feature_contract.get("observation_accounting")
        accounting = candidate if isinstance(candidate, dict) else None
    basis_physical = (
        accounting.get("basis_selected_physical_frame_count")
        if isinstance(accounting, dict) else None
    )
    basis_observations = (
        accounting.get("basis_member_observation_count")
        if isinstance(accounting, dict) else None
    )
    if report.get("module_id") == "common_pca":
        basis_selection = report.get("basis_frame_selection")
        if isinstance(basis_selection, dict):
            selected = basis_selection.get("selected_frame_count")
            if isinstance(selected, int) and not isinstance(selected, bool):
                basis_physical = selected
                basis_observations = selected * _symmetry_multiplier(report)
    fit_observations = report.get("fit_observation_count")
    module_id = str(report.get("module_id", ""))
    if module_id == "pald_community_analysis":
        fit_observations = report.get("sampled_observation_count")
    assignment_modules = {
        "common_pca", "pca_fes_basins", "clustering_kmeans",
        "clustering_hdbscan", "clustering_imwkmeans", "alternative_clustering",
    }
    if not isinstance(fit_observations, int) and module_id in {
        "clustering_kmeans", "clustering_hdbscan", "clustering_imwkmeans",
    }:
        fit_observations = observations
    full_assignment = report.get("full_assignment_observation_count")
    if not isinstance(full_assignment, int) and module_id in assignment_modules:
        benchmark = report.get("planner_benchmark")
        full_assignment = (
            benchmark.get("full_assignment_observation_count")
            if isinstance(benchmark, dict) else None
        )
    if not isinstance(full_assignment, int) and module_id in assignment_modules:
        full_assignment = observations
    if module_id not in assignment_modules:
        full_assignment = None
    silhouette_counts = []
    selected_model = report.get("selected_model")
    if isinstance(selected_model, dict):
        evaluation = selected_model.get("silhouette_evaluation")
        count = evaluation.get("evaluated_observation_count") if isinstance(evaluation, dict) else None
        if isinstance(count, int) and not isinstance(count, bool):
            silhouette_counts.append(count)
    algorithm_results = report.get("algorithm_results")
    if isinstance(algorithm_results, list):
        for result in algorithm_results:
            evaluation = result.get("silhouette_evaluation") if isinstance(result, dict) else None
            count = evaluation.get("evaluated_observation_count") if isinstance(evaluation, dict) else None
            if isinstance(count, int) and not isinstance(count, bool):
                silhouette_counts.append(count)
    multiplier = _symmetry_multiplier(report)
    return {
        "basis_selected_physical_frames": (
            basis_physical if isinstance(basis_physical, int) else None
        ),
        "basis_member_observations": (
            basis_observations if isinstance(basis_observations, int) else None
        ),
        "model_fit_observations": (
            fit_observations if isinstance(fit_observations, int) else None
        ),
        "model_fit_equivalent_physical_frames": (
            max(1, fit_observations // multiplier)
            if isinstance(fit_observations, int) else None
        ),
        "full_assignment_observations": (
            full_assignment if isinstance(full_assignment, int) else None
        ),
        "silhouette_evaluation_observations": (
            max(silhouette_counts) if silhouette_counts else None
        ),
    }


def analysis_report_sidecar(
    report: Mapping[str, object], report_path: Path, *,
    report_sha256: str, report_size_bytes: int,
) -> Dict[str, object]:
    """Build compact, report-hash-bound evidence before releasing a large report."""

    path = Path(report_path).expanduser().resolve(strict=False)
    if report.get("technical_status") != "complete":
        raise ExecutionResourceError("cannot summarize a technically incomplete report")
    resources = report.get("execution_resources")
    if not isinstance(resources, dict):
        raise ExecutionResourceError("analysis report lacks execution resources")
    project = None
    project_path_value = report.get("project_manifest_path")
    if isinstance(project_path_value, str) and Path(project_path_value).is_file():
        project = load_json(Path(project_path_value))
    physical, observations = _observation_counts(report, project)
    if physical is None or observations is None:
        raise ExecutionResourceError("analysis report lacks exact frame/observation accounting")
    from .finding_picker import finding_sidecar_evidence
    return {
        "sidecar_schema": "salsbury-analysis-report-sidecar-v1",
        "technical_status": "complete",
        "module_id": report.get("module_id"),
        "report_path": str(path),
        "report_sha256": report_sha256,
        "report_size_bytes": report_size_bytes,
        "resource_evidence": {
            "selected_source_physical_frames": physical,
            "symmetry_expanded_observations": observations,
            **_analysis_workload_counts(report, physical, observations),
            "execution_resources": resources,
        },
        "finding_evidence": finding_sidecar_evidence(report, path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_execution_resources(root: Path) -> Dict[str, object]:
    """Create JSON/CSV/Markdown tables from completed immutable report files."""

    analysis_root = Path(root).expanduser().resolve(strict=True)
    results_root = analysis_root / "results"
    if not results_root.is_dir():
        raise ExecutionResourceError(f"results directory is absent: {results_root}")
    sampling_path = analysis_root / "sampling-plan.json"
    total_source = None
    if sampling_path.is_file():
        sampling = load_json(sampling_path)
        dimensions = sampling.get("dimensions", sampling.get("system_dimensions"))
        if isinstance(dimensions, dict) and isinstance(dimensions.get("total_source_frame_count"), int):
            total_source = int(dimensions["total_source_frame_count"])
    expected: set[Path] = set()
    workflow_path = analysis_root / "workflow-stages.json"
    if workflow_path.is_file():
        workflow = load_json(workflow_path)
        stages = workflow.get("stages")
        if isinstance(stages, list):
            for stage in stages:
                if isinstance(stage, dict) and isinstance(stage.get("commands"), list):
                    for command in stage["commands"]:
                        expected.add(results_root / str(command) / "report.json")
    views_path = analysis_root / "conformational-views.json"
    if views_path.is_file():
        view_plan = load_json(views_path)
        views = view_plan.get("views")
        if isinstance(views, list):
            for view in views:
                if not isinstance(view, dict) or not str(view.get("project_manifest", "")).startswith("project-"):
                    continue
                view_id = str(view["view_id"])
                project = load_json(analysis_root / str(view["project_manifest"]))
                requested = set(project.get("requested_modules", []))
                for command, module_id in COMMAND_MODULES.items():
                    if module_id in requested and command in {
                        "common-pca", "information-correlation", "information-dynamics",
                        "tica", "pca-fes-basins", "cluster-kmeans", "cluster-hdbscan",
                        "cluster-imwkmeans", "alternative-clustering",
                        "representative-frames", "state-coordinate-exports",
                        "markov-models", "grouped-ml",
                    }:
                        expected.add(
                            results_root / "conformational-views" / view_id
                            / command / "report.json"
                        )
    missing = sorted(path for path in expected if not path.is_file())
    if missing:
        raise ExecutionResourceError(
            f"{len(missing)} expected analysis reports are absent; first={missing[0]}"
        )
    rows = []
    for path in sorted(results_root.glob("**/report.json")):
        sidecar_path = Path(str(path) + ".summary.json")
        if sidecar_path.is_file():
            sidecar = load_json(sidecar_path)
            if sidecar.get("technical_status") != "complete":
                raise ExecutionResourceError(f"analysis sidecar is not complete: {sidecar_path}")
            if sidecar.get("report_path") != str(path.resolve()):
                raise ExecutionResourceError(f"analysis sidecar report path mismatch: {sidecar_path}")
            if sidecar.get("report_size_bytes") != path.stat().st_size:
                raise ExecutionResourceError(f"analysis sidecar report size mismatch: {sidecar_path}")
            if sidecar.get("report_sha256") != _sha256_file(path):
                raise ExecutionResourceError(f"analysis sidecar report hash mismatch: {sidecar_path}")
            evidence = sidecar.get("resource_evidence")
            if not isinstance(evidence, dict):
                raise ExecutionResourceError(f"analysis sidecar lacks resource evidence: {sidecar_path}")
            resources = evidence.get("execution_resources")
            physical = evidence.get("selected_source_physical_frames")
            observations = evidence.get("symmetry_expanded_observations")
            workload = {
                key: evidence.get(key) for key in (
                    "basis_selected_physical_frames", "basis_member_observations",
                    "model_fit_observations", "model_fit_equivalent_physical_frames",
                    "full_assignment_observations", "silhouette_evaluation_observations",
                )
            }
            module_id = sidecar.get("module_id")
        else:
            report = load_json(path)
            if report.get("technical_status") != "complete":
                raise ExecutionResourceError(
                    f"analysis report is not technically complete: {path}"
                )
            module_id = report.get("module_id")
            # These reports are created by the reporting layer rather than an
            # instrumented analysis worker. Their source analyses remain in the
            # table; counting the derived reporting artifacts as measured jobs
            # would misstate both resources and frame coverage.
            if module_id in _REPORTING_ONLY_MODULES:
                continue
            resources = report.get("execution_resources")
            project = None
            project_path_value = report.get("project_manifest_path")
            if isinstance(project_path_value, str) and Path(project_path_value).is_file():
                project = load_json(Path(project_path_value))
            physical, observations = _observation_counts(report, project)
            workload = _analysis_workload_counts(report, physical, observations)
        if not isinstance(resources, dict):
            raise ExecutionResourceError(
                f"analysis report lacks measured execution resources: {path}"
            )
        if not isinstance(physical, int) or not isinstance(observations, int):
            raise ExecutionResourceError(
                f"analysis report lacks exact physical/observation accounting: {path}"
            )
        rows.append({
            "analysis_id": str(path.parent.relative_to(results_root)),
            "module_id": module_id,
            "technical_status": "complete",
            "source_physical_frames_available": total_source,
            "selected_source_physical_frames": physical,
            "symmetry_expanded_observations": observations,
            **workload,
            "member_observation_multiplier": (
                observations / physical if physical and observations is not None else None
            ),
            "computer_hostname": resources.get("computer_hostname"),
            "slurm_job_id": resources.get("slurm_job_id"),
            "slurm_array_task_id": resources.get("slurm_array_task_id"),
            "requested_cpu_count": resources.get("requested_cpu_count"),
            "requested_memory": resources.get("requested_memory"),
            "requested_time_limit": resources.get("requested_time_limit"),
            "wall_seconds": resources.get("wall_seconds"),
            "total_cpu_seconds": resources.get("total_cpu_seconds"),
            "total_cpu_hours": (
                float(resources["total_cpu_seconds"]) / 3600.0
                if isinstance(resources.get("total_cpu_seconds"), (int, float))
                and not isinstance(resources.get("total_cpu_seconds"), bool)
                else None
            ),
            "maximum_resident_memory_mib": resources.get("maximum_resident_memory_mib"),
            "report_path": str(path),
        })
    if not rows:
        raise ExecutionResourceError("no completed analysis report files were found")
    fields = list(rows[0])
    csv_path = analysis_root / "analysis_resource_and_frame_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path = analysis_root / "analysis_resource_and_frame_table.json"
    total_measured_cpu_hours = sum(
        float(row["total_cpu_hours"])
        for row in rows if isinstance(row.get("total_cpu_hours"), (int, float))
    )
    json_path.write_text(json.dumps({
        "table_schema": "salsbury-analysis-resource-frame-table-v1",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "rows": rows,
        "total_measured_cpu_hours": total_measured_cpu_hours,
        "interpretation": (
            "selected_source_physical_frames count original trajectory frames; symmetry-expanded "
            "observations count aligned oligomer members and are not independent replicas"
        ),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = analysis_root / "analysis_resource_and_frame_table.md"
    columns = [
        "analysis_id", "module_id", "technical_status",
        "source_physical_frames_available",
        "selected_source_physical_frames", "symmetry_expanded_observations",
        "member_observation_multiplier",
        "basis_selected_physical_frames", "basis_member_observations",
        "model_fit_observations", "model_fit_equivalent_physical_frames",
        "full_assignment_observations",
        "silhouette_evaluation_observations",
        "requested_cpu_count", "requested_memory", "requested_time_limit",
        "wall_seconds", "total_cpu_seconds", "total_cpu_hours",
        "maximum_resident_memory_mib",
        "computer_hostname", "slurm_job_id", "slurm_array_task_id",
    ]
    markdown = [
        "# Analysis resource and frame table", "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        markdown.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    markdown.extend([
        "", f"Total measured CPU time: {total_measured_cpu_hours:.6g} CPU-hours.",
        "", "Physical-frame counts refer to original trajectory frames. Symmetry-expanded member ",
        "observations are paired representations and do not increase the independent-replica count.",
        "", "Technical completion records execution and output integrity; scientific status is not evaluated.",
    ])
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return {
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "table_schema": "salsbury-analysis-resource-frame-table-v1",
        "row_count": len(rows),
        "total_measured_cpu_hours": total_measured_cpu_hours,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }
