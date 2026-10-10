"""Admission uses the same cache identities as the coordinate consumer."""
import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.cli import main
from salsbury_md_analysis.execution_adapters import ExecutionAdapterError, validate_worker_projects
from salsbury_md_analysis.preflight import preflight_system
from salsbury_md_analysis.validated_cache_coordinates import (
    discover_cached_replica, validate_cached_manifest,
)


def put(path, data):
    path.write_text(json.dumps(data, sort_keys=True) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle(root, *, static=False):
    root.mkdir(parents=True, exist_ok=True)
    top = root / "topology.pdb"
    top.write_text("ATOM      1  P    DG A   1       0.000   0.000   0.000  1.00  0.00           P\nEND\n")
    connectivity = root / "bonds.json"
    put(connectivity, {"format": "salsbury-bonds-v1", "atom_count": 1, "index_base": 0, "bonds": []})
    coords = root / "two_frames.pdb"
    coords.write_text("MODEL        1\n" + top.read_text() + "ENDMDL\nMODEL        2\n" + top.read_text() + "ENDMDL\n")
    replica = {"replica_id": "r1", "topology": str(top), "connectivity": str(connectivity),
               "segments": [{"segment_id": "full", "trajectory": str(coords),
                             "timing": {"first_frame_time": 100, "frame_interval": 100, "unit": "ps"}}]}
    kind = "independent_make_whole_molecular_payload_v1" if static else "continuous_unwrap_strided_molecular_payload_v2"
    data = {"systems": [{"system_id": name, "metadata": {"coordinate_cache": kind},
                         "replicas": [{**copy.deepcopy(replica), "replica_id": name + "-r1"}]}
                        for name in ("alpha", "beta")]}
    manifest = root / "arbitrary-pooled-name.json"
    put(manifest, data)
    rows = [{"system_id": name, "replica_id": name + "-r1", "cached_atom_count": 1,
             "topology_sha256": digest(top), "connectivity_sha256": digest(connectivity),
             "segments": [{"segment_id": "full", "cache": {"path": str(coords), "sha256": digest(coords)}}]}
            for name in ("alpha", "beta")]
    report = {"technical_status": "complete",
              "coordinate_representation": ("independent_make_whole_unaligned_strided" if static
                                             else "continuous_unwrap_unaligned_strided"),
              "selection": "molecular_payload", "source_frame_scan": "all source frames decoded in order",
              "cached_system_manifest": str(manifest), "cached_system_manifest_sha256": digest(manifest),
              "cached_per_system_manifests": {}, "rows": rows}
    sidecar = root / "coordinate-cache-report.json"
    put(sidecar, report)
    return manifest, data, sidecar, report


def worker_plan(root, manifest, *, project_name="project.json"):
    project = root / project_name
    put(project, {"system_manifest": str(manifest), "definitions": {}})
    script = root / (project.stem + ".sh")
    script.write_text(f'PROJECT="{project.name}"\n')
    return {"phases": [{"tasks": [{"task_id": "consumer", "script": script.name,
                                    "project_filename": project.name}]}]}


class CacheProvenancePreflightTests(unittest.TestCase):
    def assertRejected(self, root, manifest, data, message):
        report = preflight_system(data, manifest, hash_content=False)
        self.assertEqual(report["technical_status"], "failed")
        failures = [r for r in report["issues"] if r["code"] == "CACHE_PROVENANCE_INVALID"]
        self.assertEqual(len(failures), 1)
        self.assertIn(message, failures[0]["message"])
        with self.assertRaisesRegex(ExecutionAdapterError, message):
            validate_worker_projects(root, worker_plan(root, manifest))

    def test_original_continuous_and_independent_bundles(self):
        with tempfile.TemporaryDirectory() as tmp:
            for static in (False, True):
                with self.subTest(static=static):
                    root = Path(tmp) / str(static)
                    manifest, data, _, _ = bundle(root, static=static)
                    before = {p: digest(p) for p in root.iterdir()}
                    for hash_content in (False, True):
                        report = preflight_system(data, manifest, hash_content=hash_content)
                        self.assertEqual(report["technical_status"], "complete")
                        self.assertEqual(len(report["validated_coordinate_caches"]), 2)
                    validate_worker_projects(root, worker_plan(root, manifest))
                    self.assertTrue(all(digest(p) == h for p, h in before.items()))

    def test_derived_pooled_alias_and_per_system_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, data, sidecar, report = bundle(root / "original")
            migrated = root / "derived"
            migrated.mkdir()
            pooled = migrated / "input-01.json"
            alias = migrated / "identical-bytes-different-path.json"
            single = migrated / "one-system-custom-name.json"
            pooled.write_bytes(original.read_bytes())
            alias.write_bytes(pooled.read_bytes())
            single_data = {"systems": [data["systems"][0]]}
            put(single, single_data)
            derived = copy.deepcopy(report)
            derived.update(cached_system_manifest=str(pooled), cached_system_manifest_sha256=digest(pooled),
                           cached_per_system_manifests={"alias": {"path": str(alias), "sha256": digest(alias)},
                                                        "alpha": {"path": str(single), "sha256": digest(single)}})
            put(migrated / sidecar.name, derived)
            for path, expected in ((pooled, 2), (alias, 2), (single, 1)):
                report = preflight_system(json.loads(path.read_text()), path)
                self.assertEqual(report["technical_status"], "complete")
                self.assertEqual(len(report["validated_coordinate_caches"]), expected)
                validate_worker_projects(migrated, worker_plan(migrated, path))

    def test_missing_report_and_stale_manifest_identity_fail_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, data, sidecar, report = bundle(root)
            sidecar.unlink()
            self.assertRejected(root, manifest, data, "coordinate-cache-report.json")
            put(sidecar, report)
            moved = root / "unbound-alias.json"
            moved.write_bytes(manifest.read_bytes())
            self.assertRejected(root, moved, data, "different system manifest")
            sidecar.write_text("{")
            self.assertRejected(root, manifest, data, "Expecting property name")

    def test_wrong_manifest_coordinate_topology_and_connectivity_hashes_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, data, sidecar, original = bundle(root)
            for target, expected in (("manifest", "manifest hash"), ("coordinate", "trajectory"),
                                     ("topology", "topology hash"), ("connectivity", "connectivity hash")):
                with self.subTest(target=target):
                    report = copy.deepcopy(original)
                    if target == "manifest":
                        report["cached_system_manifest_sha256"] = "0" * 64
                    elif target == "coordinate":
                        report["rows"][0]["segments"][0]["cache"]["sha256"] = "0" * 64
                    else:
                        report["rows"][0][target + "_sha256"] = "0" * 64
                    put(sidecar, report)
                    self.assertRejected(root, manifest, data, expected)

    def test_raw_inputs_need_no_cache_report_and_mixed_inputs_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, data, sidecar, _ = bundle(root)
            data["systems"][0].pop("metadata")
            put(manifest, data)
            self.assertRejected(root, manifest, data, "mixed raw/cache")
            data["systems"][1].pop("metadata")
            put(manifest, data)
            sidecar.unlink()
            self.assertEqual(preflight_system(data, manifest)["technical_status"], "complete")
            validate_worker_projects(root, worker_plan(root, manifest))

    def test_explicit_future_project_requires_exact_producer_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = worker_plan(root, root / "coordinate-cache/system.json", project_name="project-cache-base.json")
            (root / "project-cache-base.json").unlink()
            put(root / "project.json", {})
            put(root / "base-cache-routing.json", {"routing_status": "planned_after_coordinate_cache_validation",
                "runtime_cache_project": "project-cache-base.json", "source_project": "project.json"})
            (root / "run_coordinate_cache.slurm").write_text("exit 0\n")
            task = plan["phases"][0]["tasks"][0]
            task["depends_on_task_ids"] = ["cache-producer"]
            plan["phases"][0]["tasks"].append({"task_id": "cache-producer", "script": "run_coordinate_cache.slurm"})
            validate_worker_projects(root, plan)
            with self.assertRaisesRegex(ExecutionAdapterError, "does not exist"):
                validate_worker_projects(root, plan, allow_future_inputs=False)
            task["depends_on_task_ids"] = []
            with self.assertRaisesRegex(ExecutionAdapterError, "does not exist"):
                validate_worker_projects(root, plan)
            task["depends_on_task_ids"] = ["cache-producer"]
            # Once materialized, even a declared producer cannot excuse missing
            # provenance. Runtime validation must also enforce that contract.
            manifest, _, sidecar, _ = bundle(root / "coordinate-cache")
            put(root / "project-cache-base.json", {"system_manifest": str(manifest)})
            sidecar.unlink()
            for allow_future in (False, True):
                with self.assertRaisesRegex(ExecutionAdapterError, "coordinate-cache-report.json"):
                    validate_worker_projects(root, plan, allow_future_inputs=allow_future)

    def test_shared_materialized_manifest_is_hashed_once_per_admission_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _, _, _ = bundle(root)
            first = worker_plan(root, manifest, project_name="first.json")
            second = worker_plan(root, manifest, project_name="second.json")
            first["phases"][0]["tasks"] += second["phases"][0]["tasks"]
            with patch("salsbury_md_analysis.validated_cache_coordinates.validate_cached_manifest",
                       wraps=validate_cached_manifest) as checked:
                validate_worker_projects(root, first)
                self.assertEqual(checked.call_count, 1)

    def test_cli_preflight_fails_and_strict_runtime_remains_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, data, sidecar, _ = bundle(root)
            sidecar.unlink()
            with redirect_stdout(io.StringIO()) as output:
                result = main(["preflight-system", str(manifest)])
            self.assertNotEqual(result, 0)
            self.assertIn("CACHE_PROVENANCE_INVALID", output.getvalue())
            with self.assertRaises(ValueError):
                discover_cached_replica(manifest, data["systems"][0]["replicas"][0], 1, static=False)


if __name__ == "__main__":
    unittest.main()
