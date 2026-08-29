import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.execution_resources import (
    ExecutionResourceError,
    analysis_report_sidecar,
    summarize_execution_resources,
)
from salsbury_md_analysis.finding_picker import FindingPickerError, prioritize_findings


class FinalReportingTests(unittest.TestCase):
    def test_resource_table_excludes_uninstrumented_integrated_reporting_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis_path = root / "results" / "rmsd" / "report.json"
            analysis_path.parent.mkdir(parents=True)
            analysis_path.write_text(json.dumps({
                "module_id": "rmsd",
                "technical_status": "complete",
                "frame_count": 20,
                "observation_accounting": {
                    "source_physical_frame_count": 20,
                    "symmetry_expanded_observation_count": 20,
                },
                "execution_resources": {
                    "computer_hostname": "node1",
                    "requested_cpu_count": 1,
                    "requested_memory": "1024",
                    "wall_seconds": 1.0,
                    "total_cpu_seconds": 1.0,
                    "maximum_resident_memory_mib": 1.0,
                },
            }), encoding="utf-8")
            integrated_path = root / "results" / "integrated-comparison" / "report.json"
            integrated_path.parent.mkdir(parents=True)
            integrated_path.write_text(json.dumps({
                "module_id": "integrated_comparison",
                "technical_status": "complete",
                "scientific_status": "not evaluated",
            }), encoding="utf-8")

            resource_report = summarize_execution_resources(root)

            self.assertEqual(resource_report["row_count"], 1)
            payload = json.loads(
                Path(resource_report["json_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["rows"][0]["module_id"], "rmsd")

    def test_picker_accounts_for_every_complete_report_without_promoting_qc(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write(name, payload):
                path = root / "results" / name / "report.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({
                    "technical_status": "complete", **payload,
                }), encoding="utf-8")

            write("alternative", {
                "module_id": "alternative_clustering",
                "algorithm_results": [{
                    "algorithm": "partition_around_medoids",
                    "silhouette": 0.51,
                    "cluster_sizes": [7, 5],
                }],
            })
            write("structural-qc", {
                "module_id": "structural_integrity_qc",
                "qc_status": "findings_require_review",
                "qc_finding_count": 2,
            })
            write("cache", {"module_id": "coordinate_cache"})
            write("dihedrals", {
                "module_id": "dihedral_distributions",
                "circular_summaries": [],
            })

            report = prioritize_findings(root, maximum_findings=1)
            accounting = {
                row["module_id"]: row for row in report["module_accounting"]
            }
            self.assertEqual(report["reviewed_report_count"], 4)
            self.assertEqual(report["reviewed_module_count"], 4)
            self.assertEqual(report["silent_omission_count"], 0)
            self.assertEqual(
                accounting["alternative_clustering"]["disposition"],
                "ranked_candidates",
            )
            self.assertEqual(
                accounting["structural_integrity_qc"]["disposition"],
                "quality_control",
            )
            self.assertEqual(
                accounting["coordinate_cache"]["disposition"],
                "technical_support",
            )
            self.assertEqual(
                accounting["dihedral_distributions"]["disposition"],
                "reviewed_no_automatic_highlight",
            )
            self.assertEqual(report["quality_control_record_count"], 1)
            self.assertEqual(report["headline_count"], 1)
            self.assertEqual(report["secondary_count"], 0)
            self.assertEqual(report["searchable_candidate_count"], 1)
            self.assertEqual(
                report["headline_findings"], report["findings"]
            )
            self.assertFalse(any(
                row["module_id"] == "structural_integrity_qc"
                for row in report["findings"]
            ))

    def test_picker_scales_to_twenty_system_all_pair_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "results" / "rmsf" / "report.json"
            report_path.parent.mkdir(parents=True)
            systems = []
            for index in range(20):
                systems.append({
                    "system_id": f"variant-{index:02d}",
                    "atom_statistics": [{
                        "common_atom_index": 0,
                        "chain_id": "A",
                        "residue_name": "ALA",
                        "residue_number": 10,
                        "insertion_code": "",
                        "atom_name": "CA",
                        "frame_pooled_rmsf_angstrom": 1.0 + index / 10.0,
                    }],
                })
            report_path.write_text(json.dumps({
                "module_id": "pooled_rmsf",
                "technical_status": "complete",
                "systems": systems,
            }), encoding="utf-8")
            standard_report = prioritize_findings(root)
            self.assertEqual(standard_report["headline_count"], 10)
            self.assertEqual(standard_report["secondary_count"], 40)
            self.assertEqual(standard_report["reported_count"], 50)
            self.assertEqual(
                standard_report["additional_candidate_count"], 160
            )
            self.assertEqual(
                standard_report["searchable_candidate_count"], 210
            )
            self.assertEqual(
                standard_report["presentation_contract"]["status"],
                "satisfied",
            )
            self.assertEqual(
                standard_report["presentation_contract"][
                    "headline_selection"
                ],
                "bh_significance_at_boundary",
            )
            self.assertEqual(len(standard_report["all_candidates"]), 210)
            persisted = json.loads(
                (root / "prioritized_findings.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(persisted["all_candidates"]), 210)
            self.assertEqual(
                len((root / "prioritized_findings.csv").read_text(
                    encoding="utf-8"
                ).splitlines()),
                211,
            )

            (root / "analysis-config.json").write_text(json.dumps({
                "reporting": {
                    "minimum_headline_findings": 10,
                    "headline_findings": 10,
                    "maximum_findings": 50,
                },
            }), encoding="utf-8")
            ten_headline_report = prioritize_findings(root)
            self.assertEqual(ten_headline_report["headline_count"], 10)
            self.assertEqual(ten_headline_report["secondary_count"], 40)
            self.assertEqual(
                ten_headline_report["presentation_contract"]["status"],
                "satisfied",
            )

            report = prioritize_findings(root, maximum_findings=500)
            self.assertEqual(report["scientific_status"], "not evaluated")
            self.assertEqual(report["headline_count"], 10)
            self.assertEqual(report["secondary_count"], 200)
            self.assertEqual(report["searchable_candidate_count"], 210)
            self.assertEqual(len(report["all_candidates"]), 210)
            self.assertTrue(all(
                row["presentation_tier"] == "headline"
                for row in report["all_candidates"][:10]
            ))
            self.assertTrue(all(
                row["presentation_tier"] == "secondary"
                for row in report["all_candidates"][10:]
            ))
            pairwise = [
                row for row in report["findings"]
                if row["comparison_family"] == "pooled_rmsf:pairwise_atom_difference"
            ]
            self.assertEqual(len(pairwise), 190)
            self.assertEqual(report["candidate_count"], 210)

            (root / "analysis-config.json").write_text(json.dumps({
                "comparisons": {
                    "mode": "reference_vs_all",
                    "reference_system_id": "variant-00",
                    "alpha": 0.05,
                },
            }), encoding="utf-8")
            reference_report = prioritize_findings(root, maximum_findings=500)
            reference_pairs = [
                row for row in reference_report["findings"]
                if row["comparison_family"] == "pooled_rmsf:pairwise_atom_difference"
            ]
            self.assertEqual(len(reference_pairs), 19)
            self.assertTrue(all("variant-00" in row["system_ids"] for row in reference_pairs))

    def test_picker_uses_supported_significance_to_choose_ten_to_twelve_headlines(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "results" / "observables" / "report.json"
            report_path.parent.mkdir(parents=True)

            def write(significant_count):
                report_path.write_text(json.dumps({
                    "module_id": "optional_observables",
                    "technical_status": "complete",
                    "finding_candidates": [
                        {
                            "category": "other_physical",
                            "statement": f"Candidate {index:02d}",
                            "effect_value": float(60 - index),
                            "p_value": (
                                0.001 if index < significant_count else None
                            ),
                            "evidence_level": "inferential",
                            "comparison_family": "fixture:headline_boundary",
                        }
                        for index in range(60)
                    ],
                }), encoding="utf-8")

            for significant_count, expected_headlines in (
                (0, 10), (11, 11), (12, 12)
            ):
                write(significant_count)
                report = prioritize_findings(root)
                self.assertEqual(report["headline_count"], expected_headlines)
                self.assertEqual(
                    report["secondary_count"], 50 - expected_headlines
                )
                self.assertEqual(report["searchable_candidate_count"], 60)
                self.assertEqual(report["additional_candidate_count"], 10)
                self.assertEqual(
                    report["presentation_contract"]["selected_headline_count"],
                    expected_headlines,
                )

    def test_picker_names_residue_and_interaction_level_findings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write(command, payload):
                path = root / "results" / command / "report.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({
                    "technical_status": "complete", **payload,
                }), encoding="utf-8")

            atom_1 = {
                "common_atom_index": 0, "chain_id": "A", "residue_name": "ALA",
                "residue_number": 10, "insertion_code": "", "atom_name": "CA",
            }
            atom_2 = {
                "common_atom_index": 1, "chain_id": "A", "residue_name": "GLY",
                "residue_number": 11, "insertion_code": "", "atom_name": "CA",
            }
            write("rmsf", {
                "module_id": "pooled_rmsf", "systems": [
                    {"system_id": "control", "atom_statistics": [
                        {**atom_1, "frame_pooled_rmsf_angstrom": 1.0},
                        {**atom_2, "frame_pooled_rmsf_angstrom": 2.0},
                    ]},
                    {"system_id": "variant", "atom_statistics": [
                        {**atom_1, "frame_pooled_rmsf_angstrom": 3.0},
                        {**atom_2, "frame_pooled_rmsf_angstrom": 1.5},
                    ]},
                ],
            })
            write("dccm", {
                "module_id": "dccm", "analysis_atoms": [atom_1, atom_2],
                "systems": [
                    {"system_id": "control", "frame_pooled_dccm": {
                        "matrix": [[1.0, 0.8], [0.8, 1.0]],
                    }},
                    {"system_id": "variant", "frame_pooled_dccm": {
                        "matrix": [[1.0, -0.2], [-0.2, 1.0]],
                    }},
                ],
            })
            write("hydrogen-bond-discovery", {
                "module_id": "hydrogen_bond_discovery",
                "candidate_dictionary": [{
                    "bond_id": "D0-H2-A1", "donor_atom_index": 0,
                    "hydrogen_atom_index": 2, "acceptor_atom_index": 1,
                }],
                "atom_dictionary": [
                    {"atom_index": 0, "identity": atom_1},
                    {"atom_index": 1, "identity": atom_2},
                ],
                "frame_bond_matrix": [
                    {"system_id": "control", "present_bond_ids": ["D0-H2-A1"]},
                    {"system_id": "control", "present_bond_ids": ["D0-H2-A1"]},
                    {"system_id": "variant", "present_bond_ids": []},
                    {"system_id": "variant", "present_bond_ids": ["D0-H2-A1"]},
                ],
            })
            write("sasa", {
                "module_id": "solvent_accessible_surface_area",
                "replicas": [
                    {"system_id": system_id, "replica_id": "replica-1",
                     "per_residue_summaries": [{
                         "chain_id": "A", "residue_number": 10,
                         "insertion_code": "", "residue_name": "ALA",
                         "summary_angstrom2": {"mean": value},
                     }]}
                    for system_id, value in (("control", 10.0), ("variant", 14.0))
                ],
            })
            for system_id, system_values in (
                ("control", (("replica-1", 4.0), ("replica-2", 5.0))),
                ("variant", (("replica-1", 8.0), ("replica-2", 9.0))),
            ):
                write(f"optional-observables-{system_id}", {
                    "module_id": "optional_observables",
                    "feature_reports": [
                        {"system_id": system_id, "replica_id": replica_id,
                         "feature_id": "active-site-distance", "kind": "distance",
                         "question": "active-site separation",
                         "distance_summary_angstrom": {"count": 100, "mean": value}}
                        for replica_id, value in system_values
                    ],
                })
            for command, module_id, metric in (
                ("nucleic-acid-geometry", "nucleic_acid_geometry", "minor_groove_width"),
                ("ion-geometry", "ion_coordination_geometry", "coordination_number"),
            ):
                write(command, {
                    "module_id": module_id,
                    "replica_reports": [
                        {"system_id": system_id, "replica_id": replica_id,
                         "metric_summaries": {metric: {"mean": value}},
                         "late_minus_early_metric_means": {metric: value / 10.0}}
                        for system_id, replica_id, value in (
                            ("control", "replica-1", 4.0),
                            ("control", "replica-2", 5.0),
                            ("variant", "replica-1", 7.0),
                            ("variant", "replica-2", 8.0),
                        )
                    ],
                })
            write("rdf", {
                "module_id": "radial_distribution_functions",
                "feature_reports": [
                    {"system_id": system_id, "replica_id": replica_id,
                     "feature_id": "mg-dna", "bins": [{
                         "bin_index": 3, "center_radius_angstrom": 2.1, "g_r": value,
                     }]}
                    for system_id, replica_id, value in (
                        ("control", "replica-1", 2.0),
                        ("control", "replica-2", 2.2),
                        ("variant", "replica-1", 4.0),
                        ("variant", "replica-2", 4.2),
                    )
                ],
            })
            write("dihedrals", {
                "module_id": "dihedral_distributions",
                "circular_summaries": [
                    {"system_id": system_id, "replica_id": "replica-1",
                     "chain_id": "A", "residue_number": 10,
                     "insertion_code": "", "angle_type": "chi",
                     "mean_angle_degrees": value}
                    for system_id, value in (("control", 20.0), ("variant", 80.0))
                ],
            })
            write("water-networks", {
                "module_id": "water_mediated_hydrogen_bond_networks",
                "endpoint_dictionary": [
                    {"system_id": system_id, "replica_id": "replica-1",
                     "atom_index": atom_index, "identity": identity}
                    for system_id in ("control", "variant")
                    for atom_index, identity in ((0, atom_1), (1, atom_2))
                ],
                "observed_bridge_dictionary": [{
                    "bridge_id": "W0-1", "first_endpoint_atom_index": 0,
                    "second_endpoint_atom_index": 1,
                }],
                "bridge_occupancies": [
                    {"system_id": system_id, "replica_id": "replica-1",
                     "bridge_id": "W0-1", "cutoff_id": "primary",
                     "occupancy_fraction": value}
                    for system_id, value in (("control", 0.2), ("variant", 0.7))
                ],
            })
            report = prioritize_findings(root, maximum_findings=50)
            self.assertEqual(report["scientific_status"], "not evaluated")
            statements = "\n".join(row["statement"] for row in report["findings"])
            self.assertIn("Largest descriptive atom-level RMSF difference", statements)
            self.assertIn("A:ALA10:CA", statements)
            self.assertIn("Largest descriptive DCCM difference", statements)
            self.assertIn("Largest descriptive direct-hydrogen-bond occupancy", statements)
            self.assertIn("Largest descriptive residue SASA difference", statements)
            self.assertIn("Largest descriptive declared-observable difference", statements)
            observable_finding = next(
                row for row in report["findings"]
                if row["comparison_family"] == "optional_observables:pairwise_feature_difference"
            )
            self.assertEqual(len(observable_finding["report_paths"]), 2)
            self.assertIn("Largest descriptive nucleic_acid_geometry metric difference", statements)
            self.assertIn("Largest descriptive ion_coordination_geometry metric difference", statements)
            self.assertIn("Largest descriptive RDF difference", statements)
            self.assertIn("Largest descriptive circular-mean dihedral difference", statements)
            self.assertIn("Largest descriptive shared one-water-bridge occupancy", statements)
            self.assertEqual(
                [row["finding_id"] for row in report["findings"]],
                [f"finding-{index:06d}" for index in range(1, report["reported_count"] + 1)],
            )
            self.assertTrue(all(
                row["statistically_significant"] is None for row in report["findings"]
            ))

    def test_resource_table_and_multisystem_finding_picker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "results" / "pca-fes-basins" / "report.json"
            report_path.parent.mkdir(parents=True)
            report_path.write_text(json.dumps({
                "module_id": "pca_fes_basins",
                "technical_status": "complete",
                "observation_accounting": {
                    "source_physical_frame_count": 30000,
                    "symmetry_expanded_observation_count": 60000,
                },
                "execution_resources": {
                    "computer_hostname": "node1", "slurm_job_id": "1",
                    "requested_cpu_count": 4, "requested_memory": "65536",
                    "requested_time_limit": "04:00:00", "wall_seconds": 12.0,
                    "total_cpu_seconds": 30.0, "maximum_resident_memory_mib": 500.0,
                },
                "primary_smoothing_sigma_bins": 1.0,
                "landscape": {"basins": [{
                    "basin_id": 1, "assigned_fraction": 0.7,
                }]},
                "state_population_comparison": {
                    "pairwise_system_differences": [{
                        "left_system_id": "control", "right_system_id": "variant",
                        "state_fraction_differences": [{
                            "state_id": 1,
                            "left_minus_right_fraction_of_all_evaluated": 0.25,
                        }],
                    }],
                },
            }), encoding="utf-8")
            alternative_path = (
                root / "results" / "alternative-clustering" / "report.json"
            )
            alternative_path.parent.mkdir(parents=True)
            alternative_path.write_text(json.dumps({
                "module_id": "alternative_clustering",
                "technical_status": "complete",
                "observation_count": 60000,
                "fit_observation_count": 3000,
                "full_assignment_observation_count": 60000,
                "feature_contract": {
                    "observation_count": 60000,
                    "observation_accounting": {
                        "source_physical_frame_count": 30000,
                        "selected_physical_frame_count": 30000,
                        "symmetry_expanded_observation_count": 60000,
                        "basis_selected_physical_frame_count": 750,
                        "basis_member_observation_count": 1500,
                        "member_count": 2,
                    },
                    "symmetry_expansion": {"member_count": 2},
                },
                "algorithm_results": [{
                    "algorithm": "partition_around_medoids",
                    "silhouette_evaluation": {
                        "evaluated_observation_count": 1000,
                    },
                }],
                "execution_resources": {
                    "computer_hostname": "node2", "slurm_job_id": "2",
                    "requested_cpu_count": 4, "requested_memory": "65536",
                    "requested_time_limit": "04:00:00", "wall_seconds": 30.0,
                    "total_cpu_seconds": 80.0, "maximum_resident_memory_mib": 700.0,
                },
            }), encoding="utf-8")
            for path in (report_path, alternative_path):
                raw = path.read_bytes()
                payload = json.loads(raw)
                sidecar = analysis_report_sidecar(
                    payload, path,
                    report_sha256=hashlib.sha256(raw).hexdigest(),
                    report_size_bytes=len(raw),
                )
                Path(str(path) + ".summary.json").write_text(
                    json.dumps(sidecar), encoding="utf-8"
                )
            (root / "workflow-stages.json").write_text(json.dumps({
                "stages": [{"commands": ["pca-fes-basins"]}],
            }), encoding="utf-8")
            resource_report = summarize_execution_resources(root)
            self.assertEqual(resource_report["row_count"], 2)
            self.assertEqual(resource_report["scientific_status"], "not evaluated")
            csv_text = Path(resource_report["csv_path"]).read_text(encoding="utf-8")
            self.assertIn("60000", csv_text)
            markdown_text = Path(resource_report["markdown_path"]).read_text(
                encoding="utf-8"
            )
            for field in (
                "module_id", "technical_status", "requested_memory",
                "maximum_resident_memory_mib", "slurm_job_id",
                "model_fit_equivalent_physical_frames",
            ):
                self.assertIn(field, markdown_text)
            resource_payload = json.loads(
                Path(resource_report["json_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(resource_payload["scientific_status"], "not evaluated")
            alternative = next(
                row for row in resource_payload["rows"]
                if row["module_id"] == "alternative_clustering"
            )
            self.assertEqual(alternative["basis_selected_physical_frames"], 750)
            self.assertEqual(alternative["basis_member_observations"], 1500)
            self.assertEqual(alternative["model_fit_observations"], 3000)
            self.assertEqual(alternative["model_fit_equivalent_physical_frames"], 1500)
            self.assertEqual(alternative["full_assignment_observations"], 60000)
            self.assertEqual(alternative["silhouette_evaluation_observations"], 1000)
            findings = prioritize_findings(root, maximum_findings=10)
            self.assertGreaterEqual(findings["reported_count"], 2)
            self.assertEqual(findings["findings"][0]["category"], "free_energy_surface")
            self.assertIsNone(findings["findings"][0]["statistically_significant"])

    def test_sidecars_are_rejected_after_report_or_size_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "results" / "pca-fes-basins" / "report.json"
            report_path.parent.mkdir(parents=True)
            payload = {
                "module_id": "pca_fes_basins",
                "technical_status": "complete",
                "observation_accounting": {
                    "source_physical_frame_count": 10,
                    "symmetry_expanded_observation_count": 10,
                },
                "execution_resources": {
                    "computer_hostname": "node1",
                    "requested_cpu_count": 1,
                    "requested_memory": "1024",
                    "wall_seconds": 1.0,
                    "total_cpu_seconds": 1.0,
                    "maximum_resident_memory_mib": 1.0,
                },
                "landscape": {"basins": []},
            }
            raw = json.dumps(payload).encode("utf-8")
            report_path.write_bytes(raw)
            sidecar = analysis_report_sidecar(
                payload, report_path,
                report_sha256=hashlib.sha256(raw).hexdigest(),
                report_size_bytes=len(raw),
            )
            sidecar_path = Path(str(report_path) + ".summary.json")
            sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
            prioritize_findings(root)
            summarize_execution_resources(root)

            report_path.write_bytes(raw + b"\n")
            with self.assertRaises(FindingPickerError):
                prioritize_findings(root)
            with self.assertRaises(ExecutionResourceError):
                summarize_execution_resources(root)

            report_path.write_bytes(raw)
            sidecar["report_size_bytes"] = len(raw) + 1
            sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
            with self.assertRaises(FindingPickerError):
                prioritize_findings(root)
            with self.assertRaises(ExecutionResourceError):
                summarize_execution_resources(root)

    def test_resource_table_uses_instrumented_planner_frame_coverage_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "results" / "secondary-structure" / "report.json"
            report_path.parent.mkdir(parents=True)
            report_path.write_text(json.dumps({
                "module_id": "secondary_structure",
                "technical_status": "complete",
                "planner_benchmark": {
                    "frame_coverage": {
                        "estimator_selected_frame_count": 300,
                        "symmetry_expanded_observation_count": 300,
                    },
                },
                "execution_resources": {
                    "computer_hostname": "node1",
                    "requested_cpu_count": 1,
                    "requested_memory": "32768",
                    "wall_seconds": 1426.0,
                    "total_cpu_seconds": 1400.0,
                    "maximum_resident_memory_mib": 181.0,
                },
            }), encoding="utf-8")
            resource_report = summarize_execution_resources(root)
            payload = json.loads(
                Path(resource_report["json_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["rows"][0]["selected_source_physical_frames"], 300)
            self.assertEqual(payload["rows"][0]["symmetry_expanded_observations"], 300)
            hbond_path = root / "results" / "hydrogen-bond-discovery" / "report.json"
            hbond_path.parent.mkdir(parents=True)
            hbond_path.write_text(json.dumps({
                "module_id": "hydrogen_bond_discovery",
                "technical_status": "complete",
                "evaluated_frame_count": 4200,
                "candidate_diagnostics": [
                    {"segment_id": f"candidate-{index}", "evaluated_frame_count": 4200}
                    for index in range(3)
                ],
                "execution_resources": {
                    "computer_hostname": "node1", "requested_cpu_count": 1,
                    "requested_memory": "32768", "wall_seconds": 10.0,
                    "total_cpu_seconds": 9.0, "maximum_resident_memory_mib": 100.0,
                },
            }), encoding="utf-8")
            expanded = summarize_execution_resources(root)
            expanded_payload = json.loads(
                Path(expanded["json_path"]).read_text(encoding="utf-8")
            )
            hbond = next(
                row for row in expanded_payload["rows"]
                if row["module_id"] == "hydrogen_bond_discovery"
            )
            self.assertEqual(hbond["selected_source_physical_frames"], 4200)
            self.assertIsNone(hbond["full_assignment_observations"])
            dihedral_path = root / "results" / "dihedrals" / "report.json"
            dihedral_path.parent.mkdir(parents=True)
            dihedral_path.write_text(json.dumps({
                "module_id": "dihedral_distributions",
                "technical_status": "complete",
                "observation_count": 8736000,
                "planner_benchmark": {
                    "frame_coverage": {
                        "estimator_selected_frame_count": 4200,
                        "symmetry_expanded_observation_count": 4200,
                    },
                },
                "execution_resources": {
                    "computer_hostname": "node1", "requested_cpu_count": 1,
                    "requested_memory": "32768", "wall_seconds": 10.0,
                    "total_cpu_seconds": 9.0, "maximum_resident_memory_mib": 100.0,
                },
            }), encoding="utf-8")
            final = summarize_execution_resources(root)
            final_payload = json.loads(
                Path(final["json_path"]).read_text(encoding="utf-8")
            )
            dihedral = next(
                row for row in final_payload["rows"]
                if row["module_id"] == "dihedral_distributions"
            )
            self.assertEqual(dihedral["selected_source_physical_frames"], 4200)
            self.assertEqual(dihedral["symmetry_expanded_observations"], 4200)
            self.assertIsNone(dihedral["full_assignment_observations"])

    def test_atom_frame_observations_do_not_replace_physical_frame_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = root / "project.json"
            project_path.write_text(json.dumps({
                "definitions": {
                    "solvent_accessible_surface_area": {
                        "maximum_surface_atoms": 2380,
                    },
                },
            }), encoding="utf-8")
            report_path = root / "results" / "sasa" / "report.json"
            report_path.parent.mkdir(parents=True)
            payload = {
                "module_id": "solvent_accessible_surface_area",
                "technical_status": "complete",
                "project_manifest_path": str(project_path),
                "frame_selection": {
                    "source_frame_count": 100,
                    "selected_frame_count": 100,
                    "coverage_fraction": 1.0,
                },
                "observation_count": 238000,
                "execution_resources": {
                    "computer_hostname": "node1",
                    "requested_cpu_count": 1,
                    "requested_memory": "32768",
                    "wall_seconds": 10.0,
                    "total_cpu_seconds": 9.0,
                    "maximum_resident_memory_mib": 100.0,
                },
            }
            raw = json.dumps(payload).encode("utf-8")
            report_path.write_bytes(raw)
            sidecar = analysis_report_sidecar(
                payload, report_path,
                report_sha256=hashlib.sha256(raw).hexdigest(),
                report_size_bytes=len(raw),
            )
            evidence = sidecar["resource_evidence"]
            self.assertEqual(evidence["selected_source_physical_frames"], 100)
            self.assertEqual(evidence["symmetry_expanded_observations"], 100)

            segment_payload = {
                **payload,
                "frame_selection": None,
                "module_id": "dihedral_distributions",
                "observation_count": 148000,
                "segment_reports": [{
                    "system_id": "system-1",
                    "replica_id": "replica-1",
                    "segment_id": "production",
                    "source_frame_count": 100,
                    "evaluated_frame_count": 100,
                    "torsion_definition_count": 1480,
                }],
            }
            segment_raw = json.dumps(segment_payload).encode("utf-8")
            segment_sidecar = analysis_report_sidecar(
                segment_payload, report_path,
                report_sha256=hashlib.sha256(segment_raw).hexdigest(),
                report_size_bytes=len(segment_raw),
            )
            segment_evidence = segment_sidecar["resource_evidence"]
            self.assertEqual(
                segment_evidence["selected_source_physical_frames"], 100
            )
            self.assertEqual(
                segment_evidence["symmetry_expanded_observations"], 100
            )

            scalar_payload = {
                **payload,
                "frame_selection": None,
                "module_id": "scalar_feature_distributions",
                "observation_count": 600,
                "planner_benchmark": {
                    "frame_coverage": {
                        "estimator_selected_frame_count": 600,
                        "symmetry_expanded_observation_count": 600,
                    },
                },
                "distribution_reports": [
                    {
                        "feature_id": f"feature-{feature}",
                        "assignments": [
                            {
                                "system_id": "system-1",
                                "replica_id": "replica-1",
                                "segment_id": "production",
                                "source_frame_index": frame,
                            }
                            for frame in range(100)
                        ],
                    }
                    for feature in range(6)
                ],
            }
            scalar_raw = json.dumps(scalar_payload).encode("utf-8")
            scalar_sidecar = analysis_report_sidecar(
                scalar_payload, report_path,
                report_sha256=hashlib.sha256(scalar_raw).hexdigest(),
                report_size_bytes=len(scalar_raw),
            )
            scalar_evidence = scalar_sidecar["resource_evidence"]
            self.assertEqual(
                scalar_evidence["selected_source_physical_frames"], 100
            )
            self.assertEqual(
                scalar_evidence["symmetry_expanded_observations"], 100
            )

    def test_cross_report_hydrogen_bonds_and_ions_use_chemical_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "analysis-config.json").write_text(json.dumps({
                "comparisons": {"mode": "all_pairs", "alpha": 0.05},
            }), encoding="utf-8")
            donor = {
                "chain_id": "A", "residue_name": "LYS", "residue_number": 10,
                "insertion_code": "", "atom_name": "NZ",
            }
            acceptor = {
                "chain_id": "B", "residue_name": "ASP", "residue_number": 20,
                "insertion_code": "", "atom_name": "OD1",
            }
            for system_id, offset, occupancy in (
                ("control", 0, 0.8), ("variant", 100, 0.2),
            ):
                hbond_path = (
                    root / "results" / f"hbond-{system_id}" / "report.json"
                )
                hbond_path.parent.mkdir(parents=True)
                hbond_path.write_text(json.dumps({
                    "module_id": "hydrogen_bond_discovery",
                    "technical_status": "complete",
                    "atom_dictionary": [
                        {"atom_index": offset, "identity": donor},
                        {"atom_index": offset + 1, "identity": acceptor},
                    ],
                    "candidate_dictionary": [{
                        "bond_id": f"bond-{system_id}",
                        "donor_atom_index": offset,
                        "hydrogen_atom_index": offset + 2,
                        "acceptor_atom_index": offset + 1,
                    }],
                    "occupancies": [{
                        "system_id": system_id, "replica_id": "replica-1",
                        "bond_id": f"bond-{system_id}",
                        "evaluated_frame_count": 100,
                        "present_frame_count": round(100 * occupancy),
                        "occupancy_fraction": occupancy,
                    }],
                }), encoding="utf-8")
                ion_path = (
                    root / "results" / f"ions-{system_id}" / "report.json"
                )
                ion_path.parent.mkdir()
                ion_path.write_text(json.dumps({
                    "module_id": "ion_atmosphere", "technical_status": "complete",
                    "per_ion_inner_shell_persistence": [{
                        "system_id": system_id, "replica_id": "replica-1",
                        "species": "K", "ion_atom_index": offset + 50,
                        "evaluated_frame_count": 100,
                        "inner_shell_occupancy": occupancy,
                    }],
                }), encoding="utf-8")

            report = prioritize_findings(root, maximum_findings=50)
            families = {row["comparison_family"] for row in report["findings"]}
            self.assertIn(
                "hydrogen_bond_discovery:chemical_identity_pairwise_difference",
                families,
            )
            self.assertIn(
                "ion_atmosphere:pairwise_species_maximum_difference", families
            )
            hydrogen = next(
                row for row in report["findings"]
                if row["comparison_family"].startswith(
                    "hydrogen_bond_discovery:chemical_identity"
                )
            )
            self.assertAlmostEqual(hydrogen["effect_value"], 0.6)
            self.assertEqual(len(hydrogen["report_paths"]), 2)


if __name__ == "__main__":
    unittest.main()
