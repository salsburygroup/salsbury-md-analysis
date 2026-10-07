"""Freeze sampling from a preparation; re-estimate resources without resampling."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .frame_sampling import integer_stride_selected_count

SCHEMA = "salsbury-fixed-sampling-schedule-v1"


class FixedSamplingError(ValueError):
    """A frozen schedule is incomplete, incompatible, or cannot be honored."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sampling_key(key):
    return (
        "stride" in key or "sampling" in key or "sample" in key
        or key in {"frame_selection", "projection_frame_selection", "random_seed", "random_seeds"}
        or key.endswith(("_frames", "_observations", "_candidates", "_feature_values"))
    )


def sampling_fields(value):
    """Retain sampling controls recursively, including feature-local controls."""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if _sampling_key(key):
                result[key] = deepcopy(child)
            else:
                nested = sampling_fields(child)
                if nested not in ({}, [], None):
                    result[key] = nested
        return result
    if isinstance(value, list):
        children = [sampling_fields(child) for child in value]
        return children if any(child not in ({}, [], None) for child in children) else []
    return None


def _restore(target, frozen, location):
    if isinstance(frozen, dict) and isinstance(target, dict):
        # Remove generated sampling defaults absent in the frozen definition.
        for key in list(target):
            if _sampling_key(key) and key not in frozen:
                del target[key]
        for key, child in frozen.items():
            if _sampling_key(key):
                target[key] = deepcopy(child)
            elif key not in target:
                raise FixedSamplingError(f"fixed sampling target is missing: {location}.{key}")
            else:
                _restore(target[key], child, f"{location}.{key}")
    elif isinstance(frozen, list) and isinstance(target, list) and len(target) == len(frozen):
        for index, (current, child) in enumerate(zip(target, frozen)):
            if child is not None:
                _restore(current, child, f"{location}[{index}]")
    else:
        raise FixedSamplingError(f"fixed sampling structure differs at {location}")


def _project_names(plan):
    names = {"project.json"}
    for row in plan.get("tasks", []):
        if row.get("task_scope") in {"conformational_view", "conformational_view_algorithm_fit", "automatic_chemical_context"}:
            names.add(f"project-{row['workflow_id']}.json")
    return sorted(names)


def _source_identity(root):
    """Bind the source manifest, not output-directory or cache manifest paths.

    This is an input declaration check, not trajectory-content validation or
    permission to adopt old results. Runtime provenance checks remain required.
    """
    def normalize(value):
        if isinstance(value, dict):
            return {
                key: str((root / child).resolve())
                if key in {"trajectory", "topology", "connectivity"} and isinstance(child, str)
                else normalize(child)
                for key, child in value.items() if key != "metadata"
            }
        if isinstance(value, list):
            return [normalize(child) for child in value]
        return value
    return normalize(json.loads((root / "system.json").read_text()))


def _validate_document(document):
    for name in ("projects", "tasks", "source_identity"):
        if not isinstance(document.get(name), dict) or not document[name]:
            raise FixedSamplingError(f"fixed sampling {name} must be a nonempty object")
    if not isinstance(document.get("replicas"), list) or not document["replicas"]:
        raise FixedSamplingError("fixed sampling replicas must be a nonempty list")
    coupling = document.get("global_stride_coupling")
    if coupling is not None:
        for name in ("selected_coordinate_cache_integer_stride", "selected_overall_trajectory_integer_stride"):
            value = coupling.get(name) if isinstance(coupling, dict) else None
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise FixedSamplingError(f"fixed sampling coupling lacks a valid {name}")
    for task_id, row in document["tasks"].items():
        if not isinstance(row, dict):
            raise FixedSamplingError(f"invalid fixed sampling task: {task_id}")
        source = row.get("source_frames_per_replica")
        stride = row.get("integer_stride")
        if (not isinstance(source, list) or not source
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in source)
                or isinstance(stride, bool) or not isinstance(stride, int) or stride < 1):
            raise FixedSamplingError(f"invalid fixed sampling source counts or stride: {task_id}")
        if row.get("selected_physical_frames_per_replica") != [integer_stride_selected_count(n, stride) for n in source]:
            raise FixedSamplingError(f"fixed stride/count mismatch: {task_id}")
    for name, project in document["projects"].items():
        if (not isinstance(project, dict)
                or not isinstance(project.get("sampling_fields"), dict)
                or not isinstance(project.get("requested_modules"), list)):
            raise FixedSamplingError(f"invalid fixed sampling project: {name}")
    # A frozen task budget must agree with the worker settings it preserves.
    # Older preparations omitted the ion-atmosphere allocation from the worker
    # project; replaying both records unchanged would repeat a full-frame run
    # while pricing a sparse one. Reject that contradiction before restoration.
    for task_id, row in document["tasks"].items():
        if row.get("module_id") != "ion_atmosphere":
            continue
        name = (
            f"project-{row['workflow_id']}.json"
            if row.get("task_scope") == "automatic_chemical_context"
            else "project.json"
        )
        fields = document["projects"].get(name, {}).get("sampling_fields", {})
        definition = fields.get("ion_atmosphere", {})
        if not isinstance(definition, dict):
            raise FixedSamplingError(f"invalid ion-atmosphere worker settings: {name}")
        selection = definition.get("frame_selection", {"mode": "fixed_stride_v1"})
        if not isinstance(selection, dict):
            raise FixedSamplingError(f"invalid ion-atmosphere frame selection: {name}")
        mode = selection.get("mode")
        stride = (
            selection.get("stride") if mode == "integer_stride_per_replica_v1"
            else definition.get("frame_stride", 1)
        )
        maximum = definition.get("maximum_frames")
        if (not definition or mode not in {"fixed_stride_v1", "integer_stride_per_replica_v1"}
                or isinstance(stride, bool) or not isinstance(stride, int)
                or stride != row["integer_stride"]
                or (mode == "integer_stride_per_replica_v1" and definition.get("frame_stride", 1) != 1)
                or isinstance(maximum, bool) or not isinstance(maximum, int)
                or maximum < sum(row["selected_physical_frames_per_replica"])):
            raise FixedSamplingError(
                f"fixed sampling worker mismatch: {task_id} in {name}; "
                f"planned stride {row['integer_stride']}, worker stride {stride}. "
                "Preserve the original preparation; generate and review a corrected "
                "worker configuration before replay. No sampling was changed."
            )


def schedule_document(root: Path, plan, sampling_plan, project_paths=None):
    paths = list(project_paths) if project_paths is not None else [root/name for name in _project_names(plan)]
    projects = {}
    for path in paths:
        project = json.loads(path.read_text())
        projects[path.name] = {
            "requested_modules": sorted(project["requested_modules"]),
            "sampling_fields": sampling_fields(project["definitions"]),
            "source_project_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    task_fields = (
        "module_id", "task_scope", "workflow_id", "source_frames_per_replica",
        "selected_physical_frames_per_replica", "integer_stride",
        "member_observation_multiplier", "frame_intervals_ns_per_replica",
        "source_time_spans_ns_per_replica", "system_ids_per_replica",
        "overall_trajectory_integer_stride", "coordinate_cache_integer_stride",
        "global_stride_raw_source_frames_per_replica", "effective_raw_integer_stride",
    )
    tasks = {}
    for row in plan["tasks"]:
        task_id = row["task_id"]
        if task_id in tasks:
            raise FixedSamplingError(f"duplicate fixed sampling task: {task_id}")
        tasks[task_id] = {key: deepcopy(row[key]) for key in task_fields if key in row}
    body = {
        "schedule_schema": SCHEMA,
        "replicas": deepcopy(sampling_plan["dimensions"]["replicas"]),
        "source_identity": _source_identity(root),
        "projects": projects, "tasks": tasks,
        "global_stride_coupling": deepcopy(plan.get("global_stride_coupling")),
        "comparison_clustering_consistency_skips": deepcopy(plan.get("comparison_clustering_consistency_skips", {})),
        "source_plan_sha256": _digest(plan),
        "planning_refinement": deepcopy(plan.get("planning_refinement")),
    }
    _validate_document(body)
    return {**body, "content_sha256": _digest(body)}


def export_fixed_sampling_schedule(prepared_directory: Path, output: Path):
    """Export a read-only historical preparation snapshot, never its results."""
    root = prepared_directory.expanduser().resolve(strict=True)
    plan = json.loads((root/"campaign-resource-plan.json").read_text())
    sampling = json.loads((root/"sampling-plan.json").read_text())
    document = schedule_document(root, plan, sampling)
    # Exclusive creation preserves both old preparations and previous exports.
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, indent=2, sort_keys=True)+"\n")
    return {"technical_status": "complete", "schedule_path": str(output.resolve()),
            "content_sha256": document["content_sha256"], "task_count": len(document["tasks"]),
            "execution_authorized": False}


def load_fixed_schedule(path: Path, replicas, project_paths):
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or document.get("schedule_schema") != SCHEMA:
        raise FixedSamplingError(f"fixed schedule must use {SCHEMA}")
    if document.get("content_sha256") != _digest({k:v for k,v in document.items() if k != "content_sha256"}):
        raise FixedSamplingError("fixed sampling schedule content hash mismatch")
    _validate_document(document)
    if document.get("replicas") != replicas:
        raise FixedSamplingError("fixed sampling source replica identities, frame counts, segments or timing differ; regenerate a reviewed schedule for changed inputs")
    paths = {path.name:path for path in project_paths}
    if document.get("source_identity") != _source_identity(paths["project.json"].parent):
        raise FixedSamplingError("fixed sampling source manifest paths, identities or timing differ")
    if set(document.get("projects", {})) != set(paths):
        raise FixedSamplingError("fixed sampling project set differs from the generated campaign")
    restored = []
    for name, path in paths.items():
        project = json.loads(path.read_text())
        frozen = document["projects"][name]
        if sorted(project["requested_modules"]) != frozen["requested_modules"]:
            raise FixedSamplingError(f"fixed sampling requested modules differ: {name}")
        _restore(project["definitions"], frozen["sampling_fields"], name)
        if sampling_fields(project["definitions"]) != frozen["sampling_fields"]:
            raise FixedSamplingError(f"fixed sampling includes unaccounted generated controls: {name}")
        restored.append((path, project))
    # Validate every frozen project before applying any patch to the new copy.
    from .manifests import validate_project
    for path, project in restored:
        validate_project(project, source_path=path, check_paths=True)
    for path, project in restored:
        path.write_text(json.dumps(project, indent=2, sort_keys=True)+"\n")
    return document


def freeze_task_sampling(tasks, schedule, *, coordinate_cache_full_scan_fraction=1.0):
    frozen = schedule["tasks"]
    if {row["task_id"] for row in tasks} != set(frozen):
        missing = sorted(set(frozen)-{row["task_id"] for row in tasks})
        extra = sorted({row["task_id"] for row in tasks}-set(frozen))
        raise FixedSamplingError(f"fixed sampling task set differs: missing={missing}, added={extra}; no task may be silently disabled")
    output = []
    for original in tasks:
        row = deepcopy(original)
        saved = frozen[row["task_id"]]
        for field in ("module_id", "task_scope", "workflow_id", "member_observation_multiplier"):
            # Cache scope changes when the global stride planner materializes it.
            if field == "task_scope" and row.get("module_id") == "coordinate_cache":
                continue
            if row.get(field, 1 if field == "member_observation_multiplier" else None) != saved.get(field, 1 if field == "member_observation_multiplier" else None):
                raise FixedSamplingError(f"fixed sampling {field} differs: {row['task_id']}")
        source = saved["source_frames_per_replica"]
        stride = saved["integer_stride"]
        if (row["module_id"] == "coordinate_cache" or row.get("task_scope") == "conformational_view_algorithm_fit") and row["source_frames_per_replica"] != source:
            raise FixedSamplingError(f"fixed sampling source stream differs: {row['task_id']}")
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise FixedSamplingError(f"invalid fixed integer stride: {row['task_id']}")
        expected = [integer_stride_selected_count(count, stride) for count in source]
        if expected != saved["selected_physical_frames_per_replica"]:
            raise FixedSamplingError(f"fixed stride/count mismatch: {row['task_id']}")
        row.update(deepcopy(saved))
        row["required_integer_stride"] = stride
        # Preserve scientific minima; only the allocation ceiling is fixed.
        row["maximum_frames_per_replica"] = max(max(expected), int(row["minimum_frames_per_replica"]))
        if row["module_id"] == "coordinate_cache":
            row["minimum_frames_per_replica"] = max(expected)
            row["maximum_frames_per_replica"] = max(expected)
            fraction = float(coordinate_cache_full_scan_fraction)
            rate = float(row["cpu_seconds_per_physical_frame"])
            row["coordinate_cache_original_fixed_cpu_hours"] = float(row.get("fixed_cpu_hours", 0.0))
            row["coordinate_cache_raw_source_frames_per_replica"] = list(source)
            row["coordinate_cache_full_scan_fraction"] = fraction
            row["coordinate_cache_candidate_stride"] = stride
            row["fixed_cpu_hours"] = float(row.get("fixed_cpu_hours", 0.0)) + rate*sum(source)*fraction/3600.0
            row["cpu_seconds_per_physical_frame"] = rate*(1.0-fraction)
        output.append(row)
    return output


def verify_fixed_plan(plan, schedule):
    actual = {row["task_id"]:row for row in plan["tasks"]}
    for task_id, expected in schedule["tasks"].items():
        row = actual.get(task_id, {})
        for field in ("integer_stride", "source_frames_per_replica", "selected_physical_frames_per_replica"):
            if row.get(field) != expected[field]:
                raise FixedSamplingError(f"fixed sampling changed {task_id}.{field}")
    return {"schedule_sha256": schedule["content_sha256"], "task_count":len(actual),
            "sampling_preserved":True, "automatic_resampling_permitted":False,
            "automatic_method_reduction_permitted":False}
