import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.analysis_config import DEFAULT_DISABLED_MODULES
from salsbury_md_analysis.comparative_quickstart import prepare_comparative_analysis
from salsbury_md_analysis.experimental_extension import (
    ExperimentalExtensionError,
    inspect_main_campaign,
)
from salsbury_md_analysis.manifests import load_json, resolve_manifest_path, sha256_file
from salsbury_md_analysis.quickstart import prepare_standard_analysis
from salsbury_md_analysis.upstream_cache import project_module_contract_sha256
from tests.test_quickstart import _write_inputs


_REBUILT_REPORTS = {"integrated_comparison", "rmsf_permutation_inference"}


def _write_complete_report(path: Path, project_path: Path, module_id: str) -> None:
    project = load_json(project_path)
    system_path = resolve_manifest_path(project["system_manifest"], project_path)
    payload = {
        "module_id": module_id,
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(project_path),
        "project_manifest_sha256": sha256_file(project_path),
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": sha256_file(system_path),
        "error_count": 0,
        "warning_count": 0,
        "issues": [],
    }
    try:
        payload["module_contract_sha256"] = project_module_contract_sha256(
            module_id, project_path
        )
    except ValueError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")
    summary = {
        "technical_status": "complete",
        "report_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }
    Path(str(path) + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _complete_prepared_campaign(root: Path) -> None:
    cache_root = (root / "coordinate-cache").resolve()
    cache_root.mkdir(exist_ok=True)
    system = load_json(root / "system.json")
    (cache_root / "system-cache.json").write_text(
        json.dumps(system, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for project_path in root.glob("project-*.json"):
        project_payload = load_json(project_path)
        system_value = project_payload.get("system_manifest")
        reference_system = project_payload.get("reference_system")
        if not isinstance(system_value, str) or not isinstance(reference_system, str):
            continue
        cache_manifest = resolve_manifest_path(system_value, project_path)
        if cache_manifest.is_file() or cache_manifest.parent != cache_root:
            continue
        source_manifest = root / f"system-{reference_system}.json"
        if source_manifest.is_file():
            cache_manifest.write_bytes(source_manifest.read_bytes())
        else:
            matching = [
                row for row in system["systems"]
                if row["system_id"] == reference_system
            ]
            if matching:
                cache_manifest.write_text(
                    json.dumps(
                        {"systems": matching}, indent=2, sort_keys=True
                    ) + "\n",
                    encoding="utf-8",
                )
    cache_rows = []
    for system_row in system["systems"]:
        for replica in system_row["replicas"]:
            topology = Path(replica["topology"])
            connectivity = Path(replica["connectivity"])
            identity = lambda path: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "modified_time_ns": path.stat().st_mtime_ns,
                "sha256": None,
            }
            cache_rows.append({
                "system_id": system_row["system_id"],
                "replica_id": replica["replica_id"],
                "source_topology": identity(topology),
                "source_connectivity": identity(connectivity),
                "segments": [{
                    "source": identity(Path(segment["trajectory"])),
                    "decoded_frame_count": 20,
                    "retained_frame_count": 20,
                } for segment in replica["segments"]],
            })
    (cache_root / "coordinate-cache-report.json").write_text(
        json.dumps({
            "technical_status": "complete",
            "cache_stride": 1,
            "source_frame_scan": "all source frames decoded in order",
            "rows": cache_rows,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cache_project = root / "project-cache-base.json"
    cache_project.write_bytes((root / "project.json").read_bytes())
    plan = load_json(root / "local-execution-plan.json")
    for phase in plan["phases"]:
        for task in phase["tasks"]:
            module_id = task.get("module_id")
            if (
                not isinstance(module_id, str)
                or module_id in DEFAULT_DISABLED_MODULES
                or module_id in _REBUILT_REPORTS
            ):
                continue
            project_path = Path(str(task["project_filename"]))
            if not project_path.is_absolute():
                project_path = root / project_path
            project_path = project_path.resolve(strict=True)
            project = load_json(project_path)
            output_root = Path(str(project["analysis_output_root"]))
            if not output_root.is_absolute():
                output_root = root / output_root
            report_path = output_root / str(task["command"]) / "report.json"
            _write_complete_report(report_path, project_path, module_id)


class ExperimentalExtensionTests(unittest.TestCase):
    def test_after_main_reuses_complete_reports_and_plans_only_new_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            pdb, psf, trajectories = _write_inputs(inputs)
            main = root / "main"
            prepare_standard_analysis(
                pdb_path=pdb,
                psf_path=psf,
                trajectories=trajectories,
                output_directory=main,
                project_id="main-campaign",
                frame_interval_ps=10.0,
            )
            _complete_prepared_campaign(main)
            # This test exercises report reuse independently of the existing
            # coordinate-cache validation suite. Omitting the cache report
            # makes the extension plan a fresh cache build while retaining the
            # authenticated main analysis reports.
            (main / "coordinate-cache" / "coordinate-cache-report.json").unlink()
            contract = inspect_main_campaign(main)
            reused_ids = {
                row["module_id"] for row in contract["reusable_reports"]
            }
            extension = root / "experimental-extension"
            report = prepare_standard_analysis(
                pdb_path=pdb,
                psf_path=psf,
                trajectories=trajectories,
                output_directory=extension,
                project_id="experimental-extension",
                frame_interval_ps=10.0,
                experimental_after_main=main,
            )
            extension_plan = load_json(
                extension / "local-execution-plan.json"
            )
            planned_ids = {
                task["module_id"]
                for phase in extension_plan["phases"]
                for task in phase["tasks"]
                if isinstance(task.get("module_id"), str)
            }
            coverage = load_json(extension / "module-coverage.json")
            extension_contract = load_json(
                extension / "experimental-after-main-contract.json"
            )
        self.assertEqual(report["technical_status"], "complete")
        self.assertTrue(reused_ids)
        self.assertTrue(reused_ids.isdisjoint(planned_ids))
        self.assertTrue(
            any(
                row["status"] == "reused_from_main"
                for row in coverage["module_status"].values()
            )
        )
        self.assertTrue(extension_contract["immutable_upstream"])
        self.assertFalse(extension_contract["integrated_comparison_recomputed"])

    def test_incomplete_main_report_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            pdb, psf, trajectories = _write_inputs(inputs)
            main = root / "main"
            prepare_standard_analysis(
                pdb_path=pdb,
                psf_path=psf,
                trajectories=trajectories,
                output_directory=main,
                project_id="incomplete-main",
                frame_interval_ps=10.0,
            )
            with self.assertRaises(ExperimentalExtensionError):
                inspect_main_campaign(main)

    def test_comparative_after_main_reuses_per_system_and_shared_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            pdb, psf, trajectories = _write_inputs(inputs)
            request = root / "comparison.json"
            request.write_text(json.dumps({
                "request_schema": "salsbury-comparative-analysis-input-v1",
                "systems": [{
                    "system_id": system_id,
                    "pdb": str(pdb),
                    "psf": str(psf),
                    "trajectories": [str(path) for path in trajectories],
                    "frame_interval_ps": 10.0,
                } for system_id in ("control", "variant")],
            }), encoding="utf-8")
            main = root / "main-comparison"
            prepare_comparative_analysis(
                request_path=request,
                output_directory=main,
                project_id="main-comparison",
            )
            _complete_prepared_campaign(main)
            (main / "coordinate-cache" / "coordinate-cache-report.json").unlink()
            reusable = inspect_main_campaign(main)
            extension = root / "experimental-comparison"
            report = prepare_comparative_analysis(
                request_path=request,
                output_directory=extension,
                project_id="experimental-comparison",
                experimental_after_main=main,
            )
            local_plan = load_json(extension / "local-execution-plan.json")
            planned = {
                task["module_id"]
                for phase in local_plan["phases"]
                for task in phase["tasks"]
                if isinstance(task.get("module_id"), str)
            }
            reused = {
                row["module_id"] for row in reusable["reusable_reports"]
            }
        self.assertEqual(report["technical_status"], "complete")
        self.assertTrue(reused)
        self.assertTrue(reused.isdisjoint(planned))
