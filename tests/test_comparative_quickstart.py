import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.comparative_quickstart import (
    _automatic_context_slurm_files,
    prepare_comparative_analysis,
    prepare_comparative_analysis_memory_fit,
    prepare_comparative_analysis_resource_fit,
)
from salsbury_md_analysis.quickstart import QuickstartError
from tests.test_quickstart import _write_dcd, _write_inputs, _write_oligomer_inputs


class ComparativeQuickstartTests(unittest.TestCase):
    def test_protein_only_comparison_marks_helical_mechanics_inapplicable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdb, psf, trajectories = _write_inputs(root)
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
            config = root / "config.json"
            config.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "modules": {"helical_mechanics": {"enabled": True}},
            }), encoding="utf-8")
            output = root / "analysis"
            prepare_comparative_analysis(
                request_path=request,
                output_directory=output,
                project_id="protein-only-comparison",
                config_path=config,
            )
            availability = json.loads(
                (output / "helical-mechanics-availability-control.json")
                .read_text()
            )
        self.assertEqual(availability["availability_status"], "not_available")
        self.assertEqual(
            availability["availability_reason"], "no_duplex_dna_or_rna"
        )

    def test_enabled_available_duplex_mechanics_gets_per_system_planner_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atoms = []
            names = (
                ("C1'", "C"), ("N9", "N"), ("C8", "C"), ("N7", "N"),
                ("C5", "C"), ("C6", "C"), ("N1", "N"), ("C2", "C"),
                ("N3", "N"), ("C4", "C"),
            )
            for chain_index, chain in enumerate(("A", "B")):
                for atom_index, (name, element) in enumerate(names):
                    atoms.append((
                        name, chain, atom_index * 0.3,
                        chain_index * 5.0, 0.1 * (atom_index % 2), element,
                    ))
            pdb = root / "duplex.pdb"
            pdb.write_text("".join(
                f"ATOM  {serial:5d} {name:^4s}  DA {chain}   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
                for serial, (name, chain, x, y, z, element) in enumerate(atoms, 1)
            ) + "END\n", encoding="utf-8")
            bonds = [(index, index + 1) for index in range(1, 10)] + [
                (index, index + 1) for index in range(11, 20)
            ]
            psf = root / "duplex.psf"
            psf.write_text(
                "PSF\n\n      20 !NATOM\n"
                + "".join(
                    f"{index:8d} SEG {index:4d} DA C1 C 0.0 12.0\n"
                    for index in range(1, 21)
                )
                + f"{len(bonds):8d} !NBOND: bonds\n"
                + "".join(f"{left:8d}{right:8d}" for left, right in bonds)
                + "\n", encoding="utf-8",
            )
            trajectory = root / "duplex.dcd"
            _write_dcd(trajectory, 20, 40)
            request = root / "comparison.json"
            request.write_text(json.dumps({
                "request_schema": "salsbury-comparative-analysis-input-v1",
                "systems": [{
                    "system_id": system_id,
                    "pdb": str(pdb), "psf": str(psf),
                    "trajectories": [str(trajectory)],
                    "frame_interval_ps": 10.0,
                } for system_id in ("control", "variant")],
            }), encoding="utf-8")
            dssr = root / "x3dna-dssr"
            dssr.write_text(
                f"#!{sys.executable}\n"
                "import json, pathlib, sys\n"
                "if '--version' in sys.argv:\n"
                " print('DSSR 2.9-test'); raise SystemExit(0)\n"
                "output = next(value.split('=', 1)[1] for value in sys.argv if value.startswith('--output='))\n"
                "step = {'shift':0.1,'slide':0.2,'rise':3.4,'tilt':1.0,'roll':2.0,'twist':34.0}\n"
                "pathlib.Path(output).write_text(json.dumps({'pairs':[{},{}], 'helices':[{}], 'stems':[{'steps':[step]}]}))\n",
                encoding="utf-8",
            )
            os.chmod(dssr, 0o755)
            config = root / "config.json"
            config.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "modules": {
                    "helical_mechanics": {"enabled": True},
                    "energetic_network_embeddings": {"enabled": True},
                },
            }), encoding="utf-8")
            output = root / "analysis"
            prepare_comparative_analysis(
                request_path=request, output_directory=output,
                project_id="duplex-comparison", config_path=config,
                dssr_executable=str(dssr),
            )
            campaign = json.loads(
                (output / "campaign-resource-plan.json").read_text()
            )
            projects = {
                system_id: json.loads(
                    (output / f"project-chemical_{system_id}.json").read_text()
                )
                for system_id in ("control", "variant")
            }
            worker = (
                output / "run_automatic_context_stage_1_array.slurm"
            ).read_text()
            energetic_availability = json.loads(
                (output / "energetic-network-embeddings-availability.json")
                .read_text()
            )
        tasks = {row["task_id"]: row for row in campaign["tasks"]}
        for system_id in ("control", "variant"):
            structure_id = (
                f"context:chemical_{system_id}:nucleic_acid_structure"
            )
            mechanics_id = f"context:chemical_{system_id}:helical_mechanics"
            self.assertIn(structure_id, tasks)
            self.assertIn(mechanics_id, tasks)
            self.assertEqual(
                tasks[structure_id]["balance_group"],
                tasks[mechanics_id]["balance_group"],
            )
            self.assertEqual(
                tasks[structure_id]["selected_physical_frames_per_replica"],
                tasks[mechanics_id]["selected_physical_frames_per_replica"],
            )
            self.assertIn(
                "helical_mechanics", projects[system_id]["requested_modules"]
            )
            self.assertEqual(
                len(projects[system_id]["definitions"]
                    ["nucleic_acid_structure"]["numeric_queries"]),
                6,
            )
        self.assertIn("'helical-mechanics'", worker)
        self.assertIn(
            "SALSBURY_MD_ANALYSIS_NUCLEIC_ACID_STRUCTURE_REPORT", worker
        )
        self.assertIn(
            "Validated cache unavailable for nucleic_acid_structure; "
            "continuing without that prerequisite report",
            worker,
        )
        self.assertEqual(
            energetic_availability["availability_status"], "not_available"
        )
        self.assertEqual(
            energetic_availability["availability_reason"],
            "not applicable: protein residues are absent from control, variant",
        )
    def test_comparison_resource_fit_preserves_protected_core(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdb, psf, trajectories = _write_oligomer_inputs(root)
            request_path = root / "comparison.json"
            request_path.write_text(json.dumps({
                "request_schema": "salsbury-comparative-analysis-input-v1",
                "systems": [
                    {
                        "system_id": system_id,
                        "pdb": str(pdb),
                        "psf": str(psf),
                        "trajectories": [str(path) for path in trajectories],
                        "frame_interval_ps": 10.0,
                    }
                    for system_id in ("control", "variant")
                ],
            }), encoding="utf-8")
            config_path = root / "constrained.json"
            config_path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "execution": {
                    "maximum_memory_gib": 4.0,
                    "well_calibrated_memory_uncertainty_factor": 2.0,
                    "poorly_calibrated_memory_uncertainty_factor": 2.0,
                },
            }), encoding="utf-8")
            output = root / "comparison-resource-fit"
            report = prepare_comparative_analysis_resource_fit(
                request_path=request_path,
                output_directory=output,
                project_id="comparison-resource-fit",
                config_path=config_path,
            )
            self.assertEqual(report["technical_status"], "complete")
            resource_fit = json.loads(
                (output / "resource-fit-report.json").read_text()
            )
            self.assertTrue(resource_fit["automatic_changes_applied"])
            self.assertTrue(resource_fit["protected_set_preserved"])
            reduced = json.loads(
                (output / "analysis-config.resource-fit.json").read_text()
            )
            self.assertTrue(
                reduced["modules"]["structural_integrity_qc"]["enabled"]
            )
            self.assertTrue(
                reduced["modules"]["integrated_comparison"]["enabled"]
            )
    def test_comparison_memory_fallback_writes_explicit_reduced_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdb, psf, trajectories = _write_oligomer_inputs(root)
            request_path = root / "comparison.json"
            request_path.write_text(json.dumps({
                "request_schema": "salsbury-comparative-analysis-input-v1",
                "systems": [
                    {
                        "system_id": system_id,
                        "pdb": str(pdb),
                        "psf": str(psf),
                        "trajectories": [str(path) for path in trajectories],
                        "frame_interval_ps": 10.0,
                    }
                    for system_id in ("control", "variant")
                ],
            }), encoding="utf-8")
            config_path = root / "low-memory.json"
            config_path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "execution": {
                    "maximum_memory_gib": 4.0,
                    "well_calibrated_memory_uncertainty_factor": 2.0,
                    "poorly_calibrated_memory_uncertainty_factor": 2.0,
                },
            }), encoding="utf-8")
            output = root / "comparison-memory-fit"
            report = prepare_comparative_analysis_memory_fit(
                request_path=request_path,
                output_directory=output,
                project_id="comparison-memory-fit",
                config_path=config_path,
            )
            self.assertEqual(report["technical_status"], "complete")
            memory = json.loads(
                (output / "memory-feasibility-report.json").read_text()
            )
            self.assertTrue(memory["automatic_changes_applied"])
            self.assertTrue(memory["final_memory"]["fits_configured_memory"])
            self.assertIn(
                "modules.solvent_accessible_surface_area.enabled",
                memory["directly_disabled_configuration_switches"],
            )
            reduced = json.loads(
                (output / "analysis-config.memory-fit.json").read_text()
            )
            self.assertFalse(
                reduced["modules"]["solvent_accessible_surface_area"]["enabled"]
            )

    def test_automatic_context_jobs_respect_hard_campaign_wall_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "project-chemical_control.json").write_text(json.dumps({
                "analysis_output_root": "results/control/chemical-context",
            }), encoding="utf-8")
            generated, counts = _automatic_context_slurm_files(
                root,
                "hard-wall-limit",
                [{
                    "project_filename": "project-chemical_control.json",
                    "commands": [
                        "trajectory-features", "scalar-distributions",
                        "scalar-threshold-states",
                    ],
                }],
                target_wall_hours=24.0,
                python_executable="/usr/bin/python3",
                package_root="/tmp/package/src",
            )
            self.assertEqual(counts, {0: 1, 1: 2})
            workers = [
                (root / filename).read_text(encoding="utf-8")
                for filename in generated
            ]
            self.assertTrue(all("#SBATCH --time=24:00:00" in text for text in workers))
            self.assertTrue(all("#SBATCH --time=24:30:00" not in text for text in workers))
            stage_one = (root / "run_automatic_context_stage_1_array.slurm").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "SALSBURY_MD_ANALYSIS_TRAJECTORY_FEATURES_REPORT", stage_one
            )
            self.assertIn(
                "Validated cache unavailable for trajectory_features; recomputing",
                stage_one,
            )
            self.assertIn(
                "unset SALSBURY_MD_ANALYSIS_TRAJECTORY_FEATURES_REPORT",
                stage_one,
            )
            self.assertNotIn(
                r"\nunset SALSBURY_MD_ANALYSIS_TRAJECTORY_FEATURES_REPORT",
                stage_one,
            )
            for filename in generated:
                syntax = subprocess.run(
                    ["bash", "-n", str(root / filename)],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_per_system_conformational_branches_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdb, psf, trajectories = _write_oligomer_inputs(root)
            request = {
                "request_schema": "salsbury-comparative-analysis-input-v1",
                "systems": [
                    {
                        "system_id": system_id,
                        "pdb": str(pdb),
                        "psf": str(psf),
                        "trajectories": [str(path) for path in trajectories],
                        "frame_interval_ps": 10.0,
                    }
                    for system_id in ("control", "variant")
                ],
            }
            request_path = root / "comparison.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "comparisons": {"run_per_system_analysis": False},
            }), encoding="utf-8")
            output = root / "analysis"
            prepare_comparative_analysis(
                request_path=request_path,
                output_directory=output,
                project_id="shared-only",
                config_path=config_path,
            )
            campaign = json.loads(
                (output / "campaign-resource-plan.json").read_text()
            )
            self.assertEqual(
                campaign["planning_algorithm"],
                "globally_coupled_integer_stride_iteration_v2",
            )
            self.assertTrue(campaign["planning_converged"])
            self.assertGreaterEqual(campaign["planning_iterations"], 2)
            self.assertEqual(campaign["per_system_view_count"], 0)
            self.assertGreater(campaign["shared_basis_view_count"], 0)
            self.assertFalse((output / "system-control.json").exists())

    def test_preparation_accepts_portable_bond_json_connectivity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdb, _, trajectories = _write_oligomer_inputs(root)
            bonds = []
            for base in (0, 8):
                bonds.extend([
                    [base, base + 1], [base + 1, base + 2],
                    [base + 2, base + 3], [base + 1, base + 4],
                    [base + 5, base + 6], [base + 6, base + 7],
                ])
            connectivity = root / "dimer.bonds.json"
            connectivity.write_text(json.dumps({
                "format": "salsbury-bonds-v1",
                "atom_count": 16,
                "index_base": 0,
                "bonds": bonds,
            }), encoding="utf-8")
            system_xml = root / "system.xml"
            system_xml.write_text(
                "<System><Forces><Force type=\"NonbondedForce\"><Particles>"
                + "".join(
                    '<Particle q="0.0" sig="0.35" eps="0.4184"/>'
                    for _ in range(16)
                )
                + "</Particles><Exceptions/></Force></Forces></System>",
                encoding="utf-8",
            )
            request = {
                "request_schema": "salsbury-comparative-analysis-input-v1",
                "systems": [
                    {
                        "system_id": system_id,
                        "pdb": str(pdb),
                        "connectivity": str(connectivity),
                        "force_field_parameters": {
                            "format": "openmm_system_xml_v1",
                            "files": [str(system_xml)],
                        },
                        "trajectories": [str(path) for path in trajectories],
                        "frame_interval_ps": 10.0,
                    }
                    for system_id in ("control", "variant")
                ],
            }
            request_path = root / "comparison.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            report = prepare_comparative_analysis(
                request_path=request_path,
                output_directory=root / "analysis",
                project_id="portable-connectivity",
            )
            self.assertEqual(report["technical_status"], "complete")
            project = json.loads((root / "analysis" / "project.json").read_text())
            self.assertEqual(
                project["reference_connectivity"], str(connectivity.resolve())
            )
            system = json.loads(
                (root / "analysis" / "system.json").read_text()
            )
            self.assertEqual(
                system["systems"][0]["replicas"][0]["force_field_parameters"],
                {
                    "format": "openmm_system_xml_v1",
                    "files": [str(system_xml.resolve())],
                },
            )

    def test_preparation_accepts_wt_plus_twenty_variant_panel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdb, psf, trajectories = _write_inputs(root)
            request = {
                "request_schema": "salsbury-comparative-analysis-input-v1",
                "systems": [
                    {
                        "system_id": f"variant-{index:02d}",
                        "pdb": str(pdb),
                        "psf": str(psf),
                        "trajectories": [str(path) for path in trajectories],
                        "frame_interval_ps": 10.0,
                    }
                    for index in range(21)
                ],
            }
            request_path = root / "panel.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            report = prepare_comparative_analysis(
                request_path=request_path,
                output_directory=root / "panel-analysis",
                project_id="wt-plus-twenty-variant-panel",
            )
            self.assertEqual(report["system_count"], 21)
            self.assertEqual(report["replica_count"], 63)
            self.assertEqual(report["total_source_frame_count"], 1260)
            config = json.loads(
                (root / "panel-analysis" / "analysis-config.json").read_text()
            )
            self.assertEqual(config["comparisons"]["mode"], "all_pairs")
            campaign = json.loads(
                (root / "panel-analysis" / "campaign-resource-plan.json")
                .read_text()
            )
            capacity = campaign["workflow_parallel_capacity"]
            self.assertEqual(
                capacity["coordinate_cache_replica_parallel_cpu_ceiling"], 63
            )
            self.assertEqual(capacity["useful_parallel_cpu_ceiling"], 380)

    def test_comparison_rejects_disabling_protected_common_pca(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdb, psf, trajectories = _write_inputs(root)
            request_path = root / "comparison.json"
            request_path.write_text(json.dumps({
                "request_schema": "salsbury-comparative-analysis-input-v1",
                "systems": [
                    {
                        "system_id": system_id,
                        "pdb": str(pdb),
                        "psf": str(psf),
                        "trajectories": [str(path) for path in trajectories],
                        "frame_interval_ps": 10.0,
                    }
                    for system_id in ("control", "variant")
                ],
            }), encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "modules": {"common_pca": {"enabled": False}},
            }), encoding="utf-8")
            output = root / "analysis"
            with self.assertRaisesRegex(
                QuickstartError,
                "protected module common_pca cannot be disabled",
            ):
                prepare_comparative_analysis(
                    request_path=request_path,
                    output_directory=output,
                    project_id="no-common-pca",
                    config_path=config_path,
                )

    def test_schema_v2_preserves_segmented_replica_boundaries_and_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_root = root / "control"
            second_root = root / "variant"
            first_root.mkdir()
            second_root.mkdir()
            first_pdb, first_psf, first_dcds = _write_oligomer_inputs(first_root)
            second_pdb, second_psf, second_dcds = _write_oligomer_inputs(second_root)
            systems = []
            for system_id, pdb, psf, dcds in (
                ("control", first_pdb, first_psf, first_dcds),
                ("variant", second_pdb, second_psf, second_dcds),
            ):
                replicas = []
                for replica_index, trajectory in enumerate(dcds, start=1):
                    continuation = trajectory.with_name(
                        f"continuation-{replica_index}.dcd"
                    )
                    _write_dcd(continuation, 16, 40)
                    replicas.append({
                        "replica_id": f"replica-{replica_index}",
                        "segments": [
                            {
                                "segment_id": "segment-3to4ns",
                                "trajectory": str(trajectory),
                                "first_frame_time_ps": 3010.0,
                                "frame_interval_ps": 10.0,
                            },
                            {
                                "segment_id": "segment-4to5ns",
                                "trajectory": str(continuation),
                                "first_frame_time_ps": 4010.0,
                                "frame_interval_ps": 10.0,
                                "continuous_with_previous": False,
                            },
                        ],
                    })
                systems.append({
                    "system_id": system_id,
                    "pdb": str(pdb),
                    "psf": str(psf),
                    "replicas": replicas,
                })
            request_path = root / "segmented-comparison.json"
            request_path.write_text(json.dumps({
                "request_schema": "salsbury-comparative-analysis-input-v2",
                "systems": systems,
            }), encoding="utf-8")
            output = root / "analysis"
            report = prepare_comparative_analysis(
                request_path=request_path,
                output_directory=output,
                project_id="segmented-control-vs-variant",
            )
            self.assertEqual(report["technical_status"], "complete")
            self.assertEqual(report["replica_count"], 6)
            self.assertEqual(report["total_source_frame_count"], 840)
            system = json.loads((output / "system.json").read_text())
            first_replica = system["systems"][0]["replicas"][0]
            self.assertEqual(len(first_replica["segments"]), 2)
            self.assertNotIn(
                "continuous_with_previous", first_replica["segments"][0]
            )
            self.assertFalse(
                first_replica["segments"][1]["continuous_with_previous"]
            )
            campaign = json.loads(
                (output / "campaign-resource-plan.json").read_text()
            )
            rmsd = next(
                row for row in campaign["tasks"]
                if row["task_id"] == "direct:replica_rmsd_rg"
            )
            self.assertEqual(
                rmsd["source_frames_per_replica"], [140, 140, 140, 140, 140, 140]
            )

    def test_prepares_shared_views_for_two_unequal_length_oligomer_systems(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_root = root / "control"
            second_root = root / "variant"
            first_root.mkdir()
            second_root.mkdir()
            first_pdb, first_psf, first_dcds = _write_oligomer_inputs(first_root)
            second_pdb, second_psf, second_dcds = _write_oligomer_inputs(second_root)
            _write_dcd(second_dcds[0], 16, 120)
            request = {
                "request_schema": "salsbury-comparative-analysis-input-v1",
                "systems": [
                    {
                        "system_id": "control",
                        "pdb": str(first_pdb),
                        "psf": str(first_psf),
                        "trajectories": [str(path) for path in first_dcds],
                        "frame_interval_ps": 10.0,
                    },
                    {
                        "system_id": "variant",
                        "pdb": str(second_pdb),
                        "psf": str(second_psf),
                        "trajectories": [str(path) for path in second_dcds],
                        "frame_interval_ps": 10.0,
                    },
                ],
            }
            request_path = root / "comparison.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            output = root / "analysis"
            report = prepare_comparative_analysis(
                request_path=request_path,
                output_directory=output,
                project_id="control-vs-variant",
            )
            self.assertEqual(report["technical_status"], "complete")
            self.assertEqual(report["system_count"], 2)
            self.assertEqual(report["replica_count"], 6)
            self.assertEqual(report["total_source_frame_count"], 620)
            project = json.loads((output / "project.json").read_text())
            self.assertEqual(project["common_atom_policy"], "position")
            self.assertEqual(project["reference_system"], "control")
            self.assertNotIn("common_pca", project["requested_modules"])
            campaign = json.loads(
                (output / "campaign-resource-plan.json").read_text()
            )
            for candidate in output.glob("project*.json"):
                self.assertNotIn(
                    '"mode": "uniform_per_replica_budget_v1"',
                    candidate.read_text(encoding="utf-8"),
                    msg=f"generic workflow emitted legacy near-uniform sampling in {candidate.name}",
                )
            task_ids = {row["task_id"] for row in campaign["tasks"]}
            self.assertNotIn("direct:common_pca", task_ids)
            self.assertIn("base:rmsf_permutation_inference", task_ids)
            self.assertIn("view:global_common_heavy:common_pca", task_ids)
            self.assertIn("view:global_common_heavy:pca_fes_basins", task_ids)
            self.assertIn(
                "view:system_control__global_common_heavy:common_pca", task_ids
            )
            self.assertIn(
                "view:system_variant__global_common_heavy:pca_fes_basins",
                task_ids,
            )
            task_rows = {row["task_id"]: row for row in campaign["tasks"]}
            pam = task_rows[
                "view:global_common_heavy:alternative_clustering:pam"
            ]
            mixture = task_rows[
                "view:global_common_heavy:alternative_clustering:gaussian_mixture"
            ]
            self.assertEqual(pam["execution_bundle_id"], mixture["execution_bundle_id"])
            self.assertNotEqual(pam["balance_group"], mixture["balance_group"])
            self.assertIsNotNone(pam["power_law_cost_model"])
            control_pca = task_rows[
                "view:system_control__global_common_heavy:common_pca"
            ]
            variant_pca = task_rows[
                "view:system_variant__global_common_heavy:common_pca"
            ]
            self.assertEqual(
                control_pca["balance_group"], variant_pca["balance_group"]
            )
            self.assertEqual(
                control_pca["integer_stride"], variant_pca["integer_stride"]
            )
            stride = control_pca["integer_stride"]
            self.assertEqual(
                control_pca["selected_physical_frames_per_replica"],
                [((count - 1) // stride) + 1 for count in (100, 100, 100)],
            )
            self.assertEqual(
                variant_pca["selected_physical_frames_per_replica"],
                [((count - 1) // stride) + 1 for count in (120, 100, 100)],
            )
            self.assertTrue((output / "system-control.json").is_file())
            self.assertTrue(
                (output / "conformational-views-control.json").is_file()
            )
            control_project = json.loads(
                (output / "project-system_control__global_common_heavy.json")
                .read_text()
            )
            self.assertRegex(
                control_project["system_manifest"],
                r"^coordinate-cache/system-cache-[0-9a-f]{10}\.json$",
            )
            self.assertEqual(control_project["reference_system"], "control")
            self.assertEqual(
                control_project["analysis_output_root"],
                "results/per-system/control/conformational-views/global_common_heavy",
            )
            plan = json.loads((output / "conformational-views.json").read_text())
            self.assertEqual(
                plan["planning_schema"],
                "salsbury-comparative-conformational-view-plan-v1",
            )
            view_ids = {row["view_id"] for row in plan["views"]}
            self.assertEqual(
                view_ids,
                {
                    "global_common_heavy", "chemical_interface",
                    "macromolecular_trace", "oligomer_member_common_heavy",
                    "oligomer_member_interface_common_heavy",
                },
            )
            oligomer = next(
                row for row in plan["views"]
                if row["view_id"] == "oligomer_member_common_heavy"
            )
            self.assertEqual(oligomer["physical_projection_frame_count"], 620)
            subprocess.run(
                ["bash", "-n", str(output / "submit-conformational-views.sh")],
                check=True,
            )
            self.assertTrue((output / "run_view_preflight_0.slurm").is_file())
            main_submit = (output / "submit.sh").read_text()
            self.assertIn(
                'submit-conformational-views.sh" "$STAGE_1_JOB"', main_submit
            )
            workflow = json.loads((output / "workflow-stages.json").read_text())
            self.assertEqual(workflow["maximum_parallel_cpus"], 16)
            self.assertTrue(workflow["coordinate_cache"]["enabled"])
            self.assertEqual(workflow["coordinate_cache"]["worker_processes"], 6)
            self.assertTrue((output / "run_coordinate_cache.slurm").is_file())
            self.assertNotIn(
                '--dependency="afterok:$CACHE_JOB"', main_submit
            )
            view_submit = (output / "submit-conformational-views.sh").read_text()
            self.assertIn("--array=0-0%1", view_submit)
            self.assertIn(
                "afterany:$SYSTEM_PREFLIGHT_0_JOB:$SYSTEM_PREFLIGHT_1_JOB:"
                "$SYSTEM_PREFLIGHT_2_JOB",
                view_submit,
            )
            self.assertEqual(
                oligomer["symmetry_expanded_projection_observation_count"], 1240
            )
            member_interface = next(
                row for row in plan["views"]
                if row["view_id"] == "oligomer_member_interface_common_heavy"
            )
            self.assertEqual(member_interface["physical_projection_frame_count"], 620)
            self.assertEqual(
                member_interface["symmetry_expanded_projection_observation_count"],
                1240,
            )
            config = json.loads((output / "analysis-config.json").read_text())
            self.assertEqual(config["default_module_policy"], "all_applicable")
            self.assertEqual(config["comparisons"]["mode"], "all_pairs")
            self.assertTrue(
                config["modules"]["integrated_comparison"]["protected"]
            )
            coverage = json.loads((output / "module-coverage.json").read_text())
            self.assertEqual(coverage["comparison_system_ids"], ["control", "variant"])
            self.assertEqual(
                coverage["module_status"]["rmsf_permutation_inference"]["status"],
                "automatic",
            )
            self.assertEqual(
                coverage["module_status"]["integrated_comparison"]["status"],
                "automatic",
            )
            self.assertIn(
                "rmsf-permutation-from-report",
                (output / "run_reporting_rmsf_permutation_inference.slurm").read_text(),
            )
            self.assertIn(
                "integrate-comparison-results",
                (output / "run_reporting_integrated_comparison.slurm").read_text(),
            )
            finalizer = (output / "run_finalize_reporting.slurm").read_text()
            self.assertIn("FINAL_REPORTING_STATUS=0", finalizer)
            self.assertNotIn("rmsf-permutation-from-report", finalizer)
            self.assertNotIn("integrate-comparison-results", finalizer)
            self.assertIn('exit "$FINAL_REPORTING_STATUS"', finalizer)
            planning = json.loads((output / "planning-report.json").read_text())
            rmsf_family = next(
                row for row in planning["sampling"]["analysis_families"]
                if row["family_id"] == "rmsf_inference"
            )
            self.assertEqual(rmsf_family["status"], "on")
            subprocess.run(["bash", "-n", str(output / "submit.sh")], check=True)
            subprocess.run(
                ["bash", "-n", str(output / "submit-conformational-views.sh")],
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
