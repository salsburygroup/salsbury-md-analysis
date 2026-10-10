"""Explicit, caller-pinned acceptance of immutable qualified dihedral outputs.

This route validates retained output commitments. Raw inputs and coordinate
caches are historical identities, not freshly validated execution inputs.
Approval of a policy hash is external to this module; no eligible receipt is
generated from a saved accepted flag or inferred from a numerical result.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from .manifests import content_hash_session, load_json, sha256_file, stable_json_sha256, _unique_object

MODULE = "dihedral_distributions"
SCHEMA = "salsbury-qualified-dihedral-adoption-v1"
POLICY = "salsbury-qualified-dihedral-policy-v1"
PROOF = "salsbury-qualified-dihedral-proof-v1"
BASIS = "retained-qualified-outputs; historical-raw-and-cache-identities"
ANCHORS = {"historical_lineage", "cache_qualification", "numerical_qualification",
           "result_review", "assembly_review", "scientific_owner"}
SERIES_KEYS = ("system_id", "replica_id", "chain_id", "residue_number", "insertion_code", "angle_type")
SEGMENT_KEYS = ("system_id", "replica_id", "segment_id")


class QualifiedDihedralError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise QualifiedDihedralError(message)


def _absolute(value):
    p = Path(value)
    _require(p.is_absolute() and not p.is_symlink(), "qualified commitments require absolute non-symlink paths")
    _require(p == p.resolve(), "qualified commitment path has a symlink or noncanonical component")
    return p


def _snapshot(path, digest):
    """Parse the exact bytes whose digest was checked, rejecting NaN/duplicates."""
    data = Path(path).read_bytes()
    _require(hashlib.sha256(data).hexdigest() == digest, f"changed qualified artifact: {path}")
    def nonfinite(value):
        raise QualifiedDihedralError(f"nonfinite JSON value: {value}")
    return json.loads(data, object_pairs_hook=_unique_object, parse_constant=nonfinite)


def _refs(value, base):
    """Recognize native artifact commitments, including nested input hash maps."""
    if isinstance(value, list):
        for child in value:
            yield from _refs(child, base)
    elif isinstance(value, dict):
        for key, name in value.items():
            if isinstance(name, str):
                digest = (value.get(key[:-5] + "_sha256") if key.endswith("_path")
                          else value.get("sha256", value.get("artifact_sha256"))
                          if key in {"path", "artifact_path"} else None)
                if isinstance(digest, str):
                    p = Path(name)
                    yield str((p if p.is_absolute() else base / p).resolve()), digest
            if key in {"input_hashes", "output_hashes"} and isinstance(name, dict):
                for filename, digest in name.items():
                    p = Path(filename)
                    yield str((p if p.is_absolute() else base / p).resolve()), digest
            if isinstance(name, (dict, list)):
                yield from _refs(name, base)


def _identity(value, keys):
    return tuple(value[k] for k in keys)


def _population(report, population):
    """Check native scalar structure; do not recompute torsions or statistics."""
    segments, expected = report["segment_reports"], population["segments"]
    _require([_identity(s, SEGMENT_KEYS) for s in segments] ==
             [_identity(s, SEGMENT_KEYS) for s in expected], "segment identity/order differs")
    _require(len({_identity(s, SEGMENT_KEYS) for s in expected}) == len(expected), "duplicate segment")
    total = 0
    for actual, selected in zip(segments, expected):
        frames = selected["selected_frames"]
        _require(all(type(f) is int and f >= 0 for f in frames) and frames == sorted(set(frames)),
                 "selected frames must be unique and ordered")
        _require(actual["evaluated_frame_count"] == len(frames), "selected frame count differs")
        _require(all(f < actual["source_frame_count"] for f in frames), "selected frame outside segment")
        _require(actual["torsion_definition_count"] == selected["torsion_definition_count"], "torsion definitions differ")
        total += len(frames)
    rows, expected_rows = report["circular_summaries"], population["series"]
    _require([_identity(r, SERIES_KEYS) for r in rows] == [_identity(r, SERIES_KEYS) for r in expected_rows],
             "torsion series identity/order differs")
    _require(len({_identity(r, SERIES_KEYS) for r in rows}) == len(rows), "duplicate torsion series")
    observations = 0
    bins = population["histogram_bins"]
    _require(type(bins) is int and bins > 0 and report["settings"]["histogram_bins"] == bins, "histogram bin contract differs")
    for row, expected_row in zip(rows, expected_rows):
        count = row["count"]
        _require(type(count) is int and count > 0 and count == expected_row["count"], "series count differs")
        hist = row["histogram"]
        _require(len(hist) == bins and sum(h["count"] for h in hist) == count, "histogram count/shape differs")
        for i, h in enumerate(hist):
            _require(type(h["count"]) is int and h["count"] >= 0, "invalid histogram count")
            _require(math.isclose(h["lower_degrees"], -180 + 360 * i / bins, abs_tol=1e-9)
                     and math.isclose(h["upper_degrees"], -180 + 360 * (i + 1) / bins, abs_tol=1e-9),
                     "histogram bounds differ")
            _require(math.isclose(h["fraction"], h["count"] / count, abs_tol=1e-12), "histogram denominator differs")
        for key in ("mean_angle_degrees", "mean_resultant_length", "circular_variance"):
            value = row.get(key)
            _require(value is None or (isinstance(value, (int, float)) and math.isfinite(value)), "invalid circular summary")
        observations += count
    _require(total == population["selected_frame_count"], "selected population differs")
    _require(len(rows) == population["series_count"] == report["series_count"], "series population differs")
    _require(observations == population["observation_count"] == report["observation_count"], "observation population differs")


def _population_evidence(population, evidence, execution):
    """Check retained definitions/masks without evaluating molecular coordinates."""
    _require(evidence.get("schema") == "salsbury-qualified-dihedral-population-v1",
             "unknown population evidence schema")
    _require(evidence["population"] == population, "population evidence differs")
    definitions = evidence["series_definitions"]
    masks, bounds = evidence["degeneracy_masks"], evidence["group_bounds"]
    rows = population["series"]
    _require(len(definitions) == len(bounds) == len(rows), "definition/group count differs")
    _require(all(type(v) is bool for v in masks), "invalid degeneracy mask")
    _require([_identity(r, SERIES_KEYS) for r in definitions] ==
             [_identity(r, SERIES_KEYS) for r in rows], "definition identity/order differs")
    cursor = 0
    for row, definition, bound in zip(rows, definitions, bounds):
        atoms = definition["atom_indices"]
        _require(len(atoms) == 4 and len(set(atoms)) == 4 and
                 all(type(a) is int and a >= 0 for a in atoms), "invalid torsion definition")
        _require(len(bound) == 2 and all(type(b) is int for b in bound) and
                 bound[0] == cursor and cursor <= bound[1] <= len(masks), "invalid group bounds")
        available = sum(len(s["selected_frames"]) for s in population["segments"]
                        if (s["system_id"], s["replica_id"]) == (row["system_id"], row["replica_id"]))
        _require(bound[1] - cursor == available, "group frame population differs")
        _require(sum(not v for v in masks[cursor:bound[1]]) == row["count"],
                 "degeneracy mask/count differs")
        cursor = bound[1]
    _require(cursor == len(masks), "extra mask observations")
    _require(execution["current_selected_frames"] == population["selected_frame_count"],
             "qualification selected-frame counter differs")


@dataclass(frozen=True)
class VerifiedQualifiedDihedral:
    report: dict
    summary: dict
    policy: dict
    proof: dict
    report_sha256: str
    policy_sha256: str
    validation_basis: str = BASIS


@content_hash_session()
def validate_qualified_dihedral_report(report_path, *, expected_project, expected_task,
                                      qualification_receipt, policy_path, policy_sha256,
                                      prepared_root):
    """Require an externally approved policy hash; never manufacture approval."""
    root = Path(prepared_root).resolve(strict=True)
    policy = _snapshot(_absolute(str(policy_path)), policy_sha256)
    _require(policy.get("schema") == POLICY and policy.get("validation_basis") == BASIS, "unknown qualified policy/basis")
    _require(policy.get("module_id") == MODULE == expected_task.get("module_id"), "qualified module differs")
    _require(policy["task_id"] == expected_task["task_id"] and policy["task_sha256"] == stable_json_sha256(expected_task), "qualified task differs")
    _require(sha256_file(root / "local-execution-plan.json") == policy["plan_sha256"], "prepared plan changed")
    _require(sha256_file(root / "analysis-config.json") == policy["analysis_config_sha256"], "analysis configuration changed")
    project = _absolute(str(expected_project))
    _require(str(project) == policy["project_manifest"]["path"] and sha256_file(project) == policy["project_manifest"]["sha256"], "project identity differs")
    project_data = _snapshot(project, policy["project_manifest"]["sha256"])
    system = (project.parent / project_data["system_manifest"]).resolve()
    _require(str(system) == policy["system_manifest"]["path"] and sha256_file(system) == policy["system_manifest"]["sha256"], "system identity differs")
    proof = _snapshot(_absolute(str(qualification_receipt)), policy["qualification_receipt_sha256"])
    _require(proof.get("schema") == PROOF and proof.get("module_id") == MODULE, "unknown qualification receipt")
    _require(proof["task_id"] == expected_task["task_id"], "qualification task differs")
    _require(set(proof["acceptance_anchors"]) == ANCHORS and proof["acceptance_anchors"] == policy["acceptance_anchors"], "approval anchors differ or are incomplete")
    _require(isinstance(policy.get("assumptions"), list) and policy["assumptions"] and
             proof.get("assumptions") == policy["assumptions"], "missing or changed lineage/arithmetic assumptions")
    _require(proof.get("historical_inputs") == policy.get("historical_inputs") and isinstance(proof.get("historical_inputs"), list), "historical identity policy differs")
    historical = {}
    for item in proof["historical_inputs"]:
        _require(item["role"] in {"raw_input", "qualified_cache"}, "unknown historical identity role")
        key = str(_absolute(item["path"]))
        _require(key not in historical, "duplicate historical identity")
        historical[key] = item["sha256"]
    records = proof["artifacts"]
    _require(isinstance(records, list) and records, "missing retained artifact closure")
    byte_limit = policy["maximum_retained_artifact_bytes"]
    _require(type(byte_limit) is int and byte_limit > 0, "missing bounded validation byte budget")
    _require(sum(r["size_bytes"] for r in records) <= byte_limit, "retained artifact validation exceeds byte budget")
    artifacts, snapshots = {}, {}
    for record in records:
        path = _absolute(record["path"])
        _require(str(path) not in artifacts and str(path) not in historical, "duplicate or historical retained artifact")
        _require(record["kind"] in {"json", "opaque"} and (path.suffix != ".json" or record["kind"] == "json"), "invalid retained artifact kind")
        _require(path.suffix.lower() not in {".dcd", ".xtc", ".trr", ".nc", ".pdb", ".psf", ".gro", ".prmtop", ".parm7", ".xyz"}, "trajectory/topology bytes cannot be retained proof artifacts")
        _require(type(record["size_bytes"]) is int and record["size_bytes"] >= 0 and path.stat().st_size == record["size_bytes"], "retained artifact size differs")
        artifacts[str(path)] = record["sha256"]
        if record["kind"] == "json":
            snapshots[str(path)] = _snapshot(path, record["sha256"])
        else:
            _require(sha256_file(path) == record["sha256"], f"retained artifact hash differs: {path}")
    # Every recognized recursive native reference must resolve to a checked
    # output or an explicitly approved historical identity; never silently skip.
    for name, payload in snapshots.items():
        for filename, digest in _refs(payload, Path(name).parent):
            _require(artifacts.get(filename, historical.get(filename)) == digest,
                     f"uncommitted or changed recursive artifact: {filename}")
    for anchor in proof["acceptance_anchors"].values():
        _require(artifacts.get(anchor["path"]) == anchor["sha256"], "approval anchor is not retained and validated")
    for field in ("report", "summary", "numerical_supplement", "population_evidence"):
        value = proof[field]
        _require(artifacts.get(value["path"]) == value["sha256"] and value["path"] in snapshots, f"missing {field} commitment")
    source = proof["report"]
    _require(not Path(source["path"] + ".adoption.json").exists(),
             "chained qualified adoption is not supported")
    _require(sha256_file(Path(report_path)) == source["sha256"], "published report bytes differ")
    report, summary = snapshots[source["path"]], snapshots[proof["summary"]["path"]]
    from .accepted_artifacts import _complete
    _complete(report, "qualified report")
    _complete(summary, "qualified summary")
    _require(report.get("module_id") == summary.get("module_id") == MODULE, "report module differs")
    _require(report.get("scientific_status") == "not evaluated", "technical receipt must not assert scientific acceptance")
    _require(summary["report_sha256"] == source["sha256"] and summary["report_path"] == source["path"] and
             summary["report_size_bytes"] == Path(source["path"]).stat().st_size, "summary identity differs")
    for prefix in ("project_manifest", "system_manifest"):
        _require(report[prefix + "_path"] == policy[prefix]["path"] and report[prefix + "_sha256"] == policy[prefix]["sha256"], "historical manifest differs")
    _require(report.get("input_content_signature_sha256") == policy["historical_input_signature_sha256"], "historical signature differs")
    _require(stable_json_sha256(report["settings"]) == policy["settings_sha256"], "settings differ")
    execution = report["derived_execution"]
    _require(execution["historical_context_is_not_current_raw_validation"] is True and
             execution["assembly_coordinate_frames_decoded"] == 0 and
             report["execution_resources"]["coordinate_frames_decoded"] == 0 and
             execution["native_task_adopted"] is False, "false current-raw/assembly execution claim")
    _require(execution == proof["derived_execution"], "source/qualification/assembly provenance differs")
    _population(report, proof["population"])
    _population_evidence(proof["population"], snapshots[proof["population_evidence"]["path"]], execution)
    return VerifiedQualifiedDihedral(report, summary, policy, proof, source["sha256"], policy_sha256)


def _task(root, task_id):
    plan = load_json(root / "local-execution-plan.json")
    matches = [t for p in plan["phases"] for t in p["tasks"] if t["task_id"] == task_id]
    _require(len(matches) == 1 and matches[0].get("module_id") == MODULE, "expected exactly one dihedral task")
    task = matches[0]
    _require(len(task.get("completion_reports", [])) == 1, "qualified publication needs one task report")
    return task


def _entry(root, task_id):
    return root / ".qualified-reports" / (stable_json_sha256(task_id) + ".json")


def effective_report_path(path, root=None):
    """Resolve only an explicit native prepared-task registration."""
    path = Path(path).resolve()
    roots = [Path(root).resolve()] if root is not None else [p for p in path.parents if (p / "local-execution-plan.json").is_file()]
    for prepared in roots:
        if not (prepared / ".qualified-reports").is_dir():
            continue
        plan = load_json(prepared / "local-execution-plan.json")
        for phase in plan["phases"]:
            for task in phase["tasks"]:
                binding = _entry(prepared, task["task_id"])
                if not binding.exists():
                    continue
                originals = [(prepared / p).resolve() for p in task.get("completion_reports", [])]
                if path not in originals and prepared / ".qualified-reports" / ".versions" not in path.parents:
                    continue
                b = load_json(binding)
                destination = Path(b["report_path"])
                if path in originals or path == destination:
                    _require(binding.resolve() == binding and destination.resolve() == destination and
                             prepared / ".qualified-reports" / ".versions" in destination.parents,
                             "qualified registration escaped root")
                    _require(b["task_id"] == task["task_id"], "qualified registration binding differs")
                    return destination
    return path


def active_report_paths(root, paths):
    """Use registered recovered paths without counting historical failures twice."""
    root = Path(root).resolve()
    result = {effective_report_path(p, root) for p in paths}
    if (root / ".qualified-reports").is_dir():
        for phase in load_json(root / "local-execution-plan.json")["phases"]:
            for task in phase["tasks"]:
                if _entry(root, task["task_id"]).is_file():
                    result.update(effective_report_path(root / p, root) for p in task["completion_reports"])
    return sorted(result)


def registered_validation(path, expected_project=None):
    path = effective_report_path(path)
    _require(path.parent.parent.name == ".versions" and path.parent.parent.parent.name == ".qualified-reports", "unregistered qualified receipt")
    root = path.parent.parent.parent.parent
    r = load_json(Path(str(path) + ".adoption.json"))
    binding = load_json(_entry(root, r["task_id"]))
    _require(binding["report_path"] == str(path), "qualified task registration location differs")
    task = _task(root, binding["task_id"])
    project = (root / task["project_filename"]).resolve(strict=True)
    _require(root in project.parents and (expected_project is None or project == Path(expected_project).resolve()), "qualified target project differs")
    receipt = Path(str(path) + ".adoption.json")
    _require(sha256_file(receipt) == binding["receipt_sha256"], "qualified publication receipt changed")
    _require(r.get("adoption_schema") == SCHEMA and r["policy_sha256"] == binding["policy_sha256"] and r["task_id"] == task["task_id"], "qualified publication receipt binding differs")
    result = validate_qualified_dihedral_report(path, expected_project=project, expected_task=task,
        qualification_receipt=path.parent / "proof.json", policy_path=path.parent / "policy.json",
        policy_sha256=binding["policy_sha256"], prepared_root=root)
    _require(sha256_file(Path(str(path) + ".summary.json")) == result.proof["summary"]["sha256"], "published qualified summary changed")
    for name, digest in r["preserved_failure_files"].items():
        _require(sha256_file(Path(name)) == digest, "preserved failure evidence changed")
    return result


def is_qualified_report(path):
    receipt = Path(str(path) + ".adoption.json")
    return receipt.is_file() and load_json(receipt).get("adoption_schema") == SCHEMA


def qualified_summary(path):
    """Return a presentation view without rewriting the source sidecar."""
    result = registered_validation(path)
    summary = deepcopy(result.summary)
    summary["report_path"] = str(Path(path).resolve())
    summary["qualified_validation_basis"] = BASIS
    summary["scientific_acceptance"] = False
    # Preserve every candidate in the immutable source sidecar, but do not
    # release helper-generated scientific findings through technical adoption.
    if isinstance(summary.get("finding_evidence"), dict):
        summary["finding_evidence"] = {"candidates": [], "quality_control_records": [],
            "clustering_models": [], "release_status": "qualified technical adoption only"}
    return summary


def _idle(root):
    from .user_workflow import campaign_activity, _check_submission_history
    activity = campaign_activity(root, _lock_held=True)
    config = load_json(root / "analysis-config.json")
    adapter = config.get("execution", {}).get("submission_adapter", "local")
    _require(adapter in {"local", "slurm"} and not activity["slurm_jobs"] and
             activity["scheduler_query"] in {"complete", "not_applicable"}, "active campaign or unknown scheduler state")
    if adapter == "slurm":
        from .execution_adapters import load_slurm_profile
        ids = {t["task_id"] for p in load_json(root / "local-execution-plan.json")["phases"] for t in p["tasks"]}
        _check_submission_history(root, ids, load_slurm_profile(root / "slurm-profile.json"), allow_completed=True)
    else:
        _require(not list((root / "submission-intents").glob("*.json")) and
                 not list((root / "submission-ledgers").glob("*.tsv")), "local publication has unresolved scheduler history")


def publish_qualified_dihedral(source_report, prepared_root, task_id, *, qualification_receipt,
                               policy_path, policy_sha256, apply=False):
    """Review by default; apply once under the native lock into a fresh directory."""
    from contextlib import nullcontext
    from .user_workflow import campaign_lock
    root = Path(prepared_root).resolve(strict=True)
    with campaign_lock(root) if apply else nullcontext():
        if apply:
            _idle(root)
        task = _task(root, task_id)
        project = (root / task["project_filename"]).resolve(strict=True)
        original = (root / task["completion_reports"][0]).resolve()
        _require(root in project.parents and root in original.parents, "task path escapes prepared root")
        target = _entry(root, task_id)
        _require(target.resolve() == target and not target.exists(), "qualified destination already exists or escapes root")
        validated = validate_qualified_dihedral_report(source_report, expected_project=project, expected_task=task,
            qualification_receipt=qualification_receipt, policy_path=policy_path, policy_sha256=policy_sha256, prepared_root=root)
        result = {"eligible": True, "validation_basis": BASIS, "task_id": task_id,
                  "scientific_acceptance": False,
                  "policy_sha256": policy_sha256, "applied": False}
        if not apply:
            return result
        def failures():
            return {str(p): sha256_file(_absolute(str(p))) for p in sorted(original.parent.rglob("*")) if p.is_file()}
        before = failures()
        versions = target.parent / ".versions"
        versions.mkdir(parents=True, exist_ok=True)
        _require(versions.resolve() == versions, "qualified versions directory escaped root")
        staging = Path(tempfile.mkdtemp(prefix="dihedral-", dir=versions))
        for source, name in ((source_report, "report.json"), (validated.proof["summary"]["path"], "report.json.summary.json"),
                             (qualification_receipt, "proof.json"), (policy_path, "policy.json")):
            shutil.copyfile(source, staging / name)
        receipt = {"adoption_schema": SCHEMA, "task_id": task_id, "policy_sha256": policy_sha256,
                   "validation_basis": BASIS, "numerical_payload_rewritten": False,
                   "scientific_acceptance": False, "preserved_failure_files": before}
        def write(name, payload):
            with (staging / name).open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        write("report.json.adoption.json", receipt)
        write("binding.json", {"task_id": task_id, "policy_sha256": policy_sha256,
              "receipt_sha256": sha256_file(staging / "report.json.adoption.json"),
              "report_path": str(staging / "report.json")})
        # Query activity before the last artifact/CAS pass, not after it.
        _idle(root)
        # A fresh validation pass and file comparison are the precommit CAS.
        validate_qualified_dihedral_report(staging / "report.json", expected_project=project, expected_task=task,
            qualification_receipt=staging / "proof.json", policy_path=staging / "policy.json", policy_sha256=policy_sha256, prepared_root=root)
        _require(sha256_file(staging / "report.json.summary.json") == validated.proof["summary"]["sha256"], "staged summary changed")
        _require(failures() == before, "original report directory changed during publication")
        _require(not target.exists() and target.resolve() == target, "qualified destination appeared during publication")
        # One hard-link creation is the atomic commit point. Unlike replacing a
        # directory with rename, link cannot overwrite even a raced empty target.
        # Unregistered complete/incomplete versions remain evidence, not results.
        for file in staging.iterdir():
            with file.open("rb") as stream:
                os.fsync(stream.fileno())
        def sync_directory(directory):
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        sync_directory(staging)
        sync_directory(versions)
        os.link(staging / "binding.json", target)
        sync_directory(target.parent)
        sync_directory(root)
        registered_validation(staging / "report.json", project)
        return {**result, "report_path": str(staging / "report.json"), "applied": True}
