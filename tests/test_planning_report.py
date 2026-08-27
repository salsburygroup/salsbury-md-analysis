import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.execution_adapters import prepare_execution_artifacts
from salsbury_md_analysis.planning_report import (
    build_plan_matrix,
    build_planning_report,
    render_plan_matrix_markdown,
    write_planning_report,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class PlanningReportTests(unittest.TestCase):
    def test_family_report_uses_effective_raw_stride_and_distinct_off_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_json(root / "analysis-config.json", {
                "modules": {
                    "structural_integrity_qc": {"enabled": False},
                    "pooled_rmsf": {"enabled": True},
                },
                "views": {}, "clustering": {"methods": {}},
                "community_analysis": {"pald": {
                    "enabled": False, "community_msm_enabled": False,
                }},
                "sampling": {}, "reporting": {}, "inference": {},
                "comparisons": {}, "exports": {},
            })
            _write_json(root / "campaign-resource-plan.json", {
                "technical_status": "complete", "feasibility_status": "complete",
                "execution_authorized": True,
                "maximum_parallel_cpus_input": 8,
                "maximum_parallel_memory_gib_input": 64,
                "maximum_wall_hours_input": 24,
                "tasks": [{
                    "task_id": "direct:pooled_rmsf", "module_id": "pooled_rmsf",
                    "dependency_stage": 0,
                    "overall_trajectory_integer_stride": 2,
                    "coordinate_cache_integer_stride": 2,
                    "projection_integer_stride": 3, "integer_stride": 5,
                    "effective_raw_integer_stride": 30,
                    "source_frames_per_replica": [1000, 1000],
                    "selected_physical_frames_per_replica": [34, 34],
                    "selected_physical_frame_count": 68,
                    "frame_intervals_ns_per_replica": [0.1, 0.1],
                    "minimum_frames_per_replica": 20,
                    "source_limited_below_declared_minimum": False,
                }],
            })
            _write_json(root / "sampling-plan.json", {})
            _write_json(root / "module-coverage.json", {"module_status": {
                "pooled_rmsf": {"status": "automatic"},
                "structural_integrity_qc": {
                    "status": "deferred", "reason": "disabled by analysis config",
                },
                "ion_atmosphere": {
                    "status": "deferred", "reason": "supported ions are not present",
                },
                "hydrogen_bonds": {
                    "status": "deferred",
                    "reason": "optional manual fixed-feature override",
                },
                "representative_structures": {
                    "status": "deferred",
                    "reason": "optional coordinate-space mean/medoid utility",
                },
                "integrated_comparison": {"status": "automatic"},
            }})
            _write_json(root / "launcher-contract.json", {"phases": []})
            _write_json(root / "execution-adapter.json", {"active_adapter": "local"})
            files = write_planning_report(root)
            report = build_planning_report(root)
            families = {
                row["family_id"]: row for row in report["sampling"]["analysis_families"]
            }
            self.assertEqual(files, ["planning-report.json", "planning-report.md"])
            self.assertEqual(families["rmsf_dccm"]["effective_raw_stride_display"], "30")
            self.assertEqual(families["structural_qc"]["status"], "off")
            self.assertEqual(families["ion_atmosphere"]["status"], "not_applicable")
            self.assertNotIn("explicit_hydrogen_bonds", families)
            self.assertNotIn("representative_structures", families)
            self.assertEqual(
                families["integrated_comparison"]["status"],
                "on_no_trajectory",
            )
            optional = {
                row["id"]: row for row in report["features"]["deferred_or_inapplicable"]
            }
            self.assertEqual(
                optional["hydrogen_bonds"]["category"],
                "optional_manual_utility",
            )
            markdown = (root / "planning-report.md").read_text(encoding="utf-8")
            self.assertIn("| RMSF and DCCM | 30 |", markdown)
            self.assertIn("| Structural-integrity QC | Off |", markdown)

    def test_plan_matrix_preserves_scenario_labels_and_family_cells(self):
        reports = []
        for label, cell in (("8 h reduced", "Off"), ("48 h reduced", "50")):
            reports.append((label, {
                "resource_envelope": {"maximum_wall_hours": 8 if cell == "Off" else 48},
                "sampling": {"analysis_families": [{
                    "family_id": "shared_pca_states",
                    "analysis_family": "Shared common PCA/FES/K-means/MSM",
                    "effective_raw_stride_display": cell,
                }]},
            }))
        matrix = build_plan_matrix(reports)
        markdown = render_plan_matrix_markdown(matrix)
        self.assertIn("| Analysis family | 8 h reduced | 48 h reduced |", markdown)
        self.assertIn("| Shared common PCA/FES/K-means/MSM | Off | 50 |", markdown)

    def test_custom_adapter_writes_scheduler_neutral_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = (
                "#!/usr/bin/env bash\n#SBATCH --time=00:30:00\n"
                "#SBATCH --cpus-per-task=1\n#SBATCH --mem=2G\nset -euo pipefail\n"
            )
            (root / "run_preflight.slurm").write_text(worker, encoding="utf-8")
            (root / "run_finalize_reporting.slurm").write_text(worker, encoding="utf-8")
            result = prepare_execution_artifacts(root, {
                "execution": {
                    "submission_adapter": "custom", "maximum_parallel_cpus": 2,
                    "maximum_hours_per_cpu": 1, "maximum_memory_gib": 8,
                },
                "reporting": {
                    "resource_table_enabled": False,
                    "finding_picker_enabled": False,
                },
            })
            contract = json.loads((root / "launcher-contract.json").read_text())
            self.assertEqual(result["adapter"], "custom")
            self.assertEqual(result["next_command"].rsplit("/", 1)[-1], "run-custom.sh")
            self.assertEqual(contract["resource_envelope"]["maximum_parallel_cpus"], 2)
            self.assertEqual([row["phase_id"] for row in contract["phases"]], [
                "preflight", "final_reporting",
            ])


if __name__ == "__main__":
    unittest.main()
