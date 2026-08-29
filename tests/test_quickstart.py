import json
import subprocess
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.quickstart import (
    _GENERIC_CHEMISTRY_COMMANDS,
    _GENERIC_STAGE_COMMANDS,
    QuickstartError,
    _composition,
    _discover_dssp_executable,
    _hydrogen_bond_feature_observation_gate,
    _secondary_structure_applicable,
    prepare_standard_analysis,
    prepare_standard_analysis_memory_fit,
)


def _record(payload: bytes) -> bytes:
    marker = struct.pack("<i", len(payload))
    return marker + payload + marker


def _write_dcd(path: Path, atom_count: int, frames: int) -> None:
    header = bytearray(84)
    header[:4] = b"CORD"
    struct.pack_into("<3i", header, 4, frames, 0, 1)
    title = struct.pack("<i", 1) + b"quickstart fixture".ljust(80)
    path.write_bytes(
        _record(bytes(header))
        + _record(title)
        + _record(struct.pack("<i", atom_count))
    )


def _write_inputs(root: Path):
    pdb = root / "system.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1       2.900   0.000   0.000  1.00  0.00           C\n"
        "ATOM      4  O   ALA A   1       3.500   1.000   0.000  1.00  0.00           O\n"
        "END\n",
        encoding="utf-8",
    )
    psf = root / "system.psf"
    psf.write_text(
        "PSF\n\n       4 !NATOM\n"
        "       1 SEG 1 ALA N  N  0.0 14.0\n"
        "       2 SEG 1 ALA CA CT 0.0 12.0\n"
        "       3 SEG 1 ALA C  C  0.0 12.0\n"
        "       4 SEG 1 ALA O  O  0.0 16.0\n"
        "       3 !NBOND: bonds\n       1       2       2       3       3       4\n",
        encoding="utf-8",
    )
    trajectories = []
    for index in range(3):
        trajectory = root / f"replica-{index + 1}.dcd"
        _write_dcd(trajectory, 4, 20)
        trajectories.append(trajectory)
    return pdb, psf, trajectories


def _write_ion_inputs(root: Path):
    pdb = root / "ion-system.pdb"
    pdb.write_text(
        "CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1\n"
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1       2.900   0.000   0.000  1.00  0.00           C\n"
        "ATOM      4  O   ALA A   1       3.500   1.000   0.000  1.00  0.00           O\n"
        "HETATM    5  K   POT I   1       5.000   1.000   0.000  1.00  0.00           K\n"
        "HETATM    6  CL  CLA I   2      28.000   0.000   0.000  1.00  0.00          CL\n"
        "HETATM    7  OH2 TIP W   1       8.000   8.000   8.000  1.00  0.00           O\n"
        "HETATM    8  H1  TIP W   1       8.900   8.000   8.000  1.00  0.00           H\n"
        "HETATM    9  H2  TIP W   1       7.700   8.850   8.000  1.00  0.00           H\n"
        "END\n",
        encoding="utf-8",
    )
    psf = root / "ion-system.psf"
    psf.write_text(
        "PSF\n\n       9 !NATOM\n"
        + "".join(
            f"{index:8d} SEG {index:4d} RES A C 0.0 12.0\n"
            for index in range(1, 10)
        )
        + "       5 !NBOND: bonds\n"
        "       1       2       2       3       3       4"
        "       7       8       7       9\n",
        encoding="utf-8",
    )
    trajectories = []
    for index in range(3):
        trajectory = root / f"ion-replica-{index + 1}.dcd"
        _write_dcd(trajectory, 9, 100)
        trajectories.append(trajectory)
    return pdb, psf, trajectories


def _write_oligomer_inputs(root: Path):
    pdb = root / "dimer.pdb"
    rows = []
    atoms = []
    for protein_chain, dna_chain, offset in (("A", "C", 0.0), ("B", "D", 30.0)):
        atoms.extend([
            ("N", "ALA", protein_chain, 1, offset, 0.0, 0.0, "N"),
            ("CA", "ALA", protein_chain, 1, offset + 1.0, 0.0, 0.0, "C"),
            ("C", "ALA", protein_chain, 1, offset + 2.0, 0.0, 0.0, "C"),
            ("O", "ALA", protein_chain, 1, offset + 2.0, 1.0, 0.0, "O"),
            ("CB", "ALA", protein_chain, 1, offset + 1.0, 1.0, 0.0, "C"),
            ("P", "DG", dna_chain, 10, offset + 2.0, 3.0, 0.0, "P"),
            ("C1'", "DG", dna_chain, 10, offset + 2.0, 4.0, 0.0, "C"),
            ("O4'", "DG", dna_chain, 10, offset + 2.0, 5.0, 0.0, "O"),
        ])
    for serial, (name, residue, chain, number, x, y, z, element) in enumerate(atoms, start=1):
        rows.append(
            f"ATOM  {serial:5d} {name:^4s} {residue:>3s} {chain}{number:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
        )
    pdb.write_text("".join(rows) + "END\n", encoding="utf-8")
    bonds = []
    for base in (1, 9):
        bonds.extend([
            (base, base + 1), (base + 1, base + 2), (base + 2, base + 3),
            (base + 1, base + 4), (base + 5, base + 6), (base + 6, base + 7),
        ])
    bond_values = "".join(f"{left:8d}{right:8d}" for left, right in bonds)
    psf = root / "dimer.psf"
    psf.write_text(
        "PSF\n\n      16 !NATOM\n"
        + "".join(
            f"{index:8d} SEG {index:4d} RES A C 0.0 12.0\n"
            for index in range(1, 17)
        )
        + f"{len(bonds):8d} !NBOND: bonds\n{bond_values}\n",
        encoding="utf-8",
    )
    trajectories = []
    for index in range(3):
        trajectory = root / f"dimer-replica-{index + 1}.dcd"
        _write_dcd(trajectory, 16, 100)
        trajectories.append(trajectory)
    return pdb, psf, trajectories


class QuickstartTests(unittest.TestCase):
    def test_deac_config_activates_profiled_slurm_launcher(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "inputs"
            source.mkdir()
            output = Path(temporary) / "deac-analysis"
            pdb, psf, trajectories = _write_inputs(source)
            report = prepare_standard_analysis(
                pdb_path=pdb,
                psf_path=psf,
                trajectories=trajectories,
                output_directory=output,
                project_id="deac-profile-test",
                frame_interval_ps=10.0,
                config_path=repository / "profiles/analysis/deac-default.json",
            )
            worker = (output / "run_stage_0_array.slurm").read_text(
                encoding="utf-8"
            )
            submit = (output / "submit.sh").read_text(encoding="utf-8")
            retained = json.loads((output / "slurm-profile.json").read_text())
            scheduler = json.loads(
                (output / "scheduler-resource-requests.json").read_text()
            )
        self.assertEqual(report["execution_adapter"], "slurm")
        self.assertEqual(report["slurm_profile_id"], "wfu-deac-salsbury-group-v1")
        self.assertTrue(report["next_command"].endswith("./submit.sh"))
        self.assertEqual(retained["account"], "salsburygrp")
        self.assertIn("#SBATCH --account=salsburygrp", worker)
        self.assertIn("#SBATCH --partition=small", worker)
        self.assertIn("/opt/scyld/slurm/bin/sbatch", submit)
        stage_request = scheduler["scripts"]["run_stage_0_array.slurm"]
        self.assertTrue(stage_request["planner_task_ids"])
        self.assertIn(f"#SBATCH --time={stage_request['slurm_time']}", worker)
        self.assertIn(f"#SBATCH --mem={stage_request['slurm_memory']}", worker)
        self.assertIn(
            "/software/salsbury-md-analysis/environments/v76/",
            retained["environment"]["python_executable"],
        )

    def test_hydrogen_bond_gate_matches_candidate_count_times_selected_frames(self):
        sampling_plan = {
            "dimensions": {
                "hydrogen_bond_candidate_planning": {
                    "status": "complete",
                    "common_candidate_count": 125_001,
                },
            },
            "method_plans": [{
                "module_id": "hydrogen_bond_discovery",
                "selected_frame_count": 18_528,
            }],
        }
        self.assertEqual(
            _hydrogen_bond_feature_observation_gate(sampling_plan),
            125_001 * 18_528,
        )

    def test_composition_distinguishes_dna_from_protein_and_detects_mixed_complex(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dna = root / "dna.pdb"
            dna.write_text(
                "ATOM      1  N   DG  D   1       0.000   0.000   0.000  1.00  0.00           N\n"
                "ATOM      2  CA  DG  D   1       1.000   0.000   0.000  1.00  0.00           C\n"
                "ATOM      3  C   DG  D   1       2.000   0.000   0.000  1.00  0.00           C\n"
                "ATOM      4  P   DG  D   1       0.000   1.000   0.000  1.00  0.00           P\n"
                "ATOM      5  C1' DG  D   1       1.000   1.000   0.000  1.00  0.00           C\n"
                "END\n",
                encoding="utf-8",
            )
            dna_composition = _composition(dna)
            self.assertFalse(dna_composition["has_protein"])
            self.assertTrue(dna_composition["has_nucleic_acid"])

            mixed = root / "mixed.pdb"
            mixed.write_text(
                dna.read_text(encoding="utf-8").replace("END\n", "")
                + "ATOM      6  N   ALA A   2       0.000   3.000   0.000  1.00  0.00           N\n"
                + "ATOM      7  CA  ALA A   2       1.000   3.000   0.000  1.00  0.00           C\n"
                + "ATOM      8  C   ALA A   2       2.000   3.000   0.000  1.00  0.00           C\n"
                + "END\n",
                encoding="utf-8",
            )
            mixed_composition = _composition(mixed)
            self.assertTrue(mixed_composition["has_protein"])
            self.assertTrue(mixed_composition["has_nucleic_acid"])

    def test_dssp_requires_detected_protein_even_when_executable_exists(self):
        self.assertFalse(
            _secondary_structure_applicable(
                {"has_protein": False, "has_nucleic_acid": True},
                "/validated/bin/mkdssp",
            )
        )
        self.assertTrue(
            _secondary_structure_applicable(
                {"has_protein": True, "has_nucleic_acid": True},
                "/validated/bin/mkdssp",
            )
        )
        self.assertFalse(
            _secondary_structure_applicable(
                {"has_protein": True, "has_nucleic_acid": False}, None
            )
        )

    def test_oligomer_quickstart_generates_member_safe_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdb, psf, trajectories = _write_oligomer_inputs(root)
            output = root / "analysis"
            report = prepare_standard_analysis(
                pdb_path=pdb, psf_path=psf, trajectories=trajectories,
                output_directory=output, project_id="dimer", frame_interval_ps=10.0,
            )
            self.assertEqual(report["technical_status"], "complete")
            plan = json.loads((output / "conformational-views.json").read_text())
            view = next(
                row for row in plan["views"]
                if row["view_id"] == "oligomer_member_common_heavy"
            )
            self.assertEqual(view["symmetry_expanded_projection_observation_count"], 600)
            interface_view = next(
                row for row in plan["views"]
                if row["view_id"] == "oligomer_member_interface_common_heavy"
            )
            self.assertEqual(
                interface_view["symmetry_expanded_projection_observation_count"], 600
            )
            project = json.loads(
                (output / "project-oligomer_member_common_heavy.json").read_text()
            )
            self.assertNotIn(
                "pald", project["definitions"]["alternative_clustering"]["algorithms"]
            )
            self.assertNotIn("pald_community_analysis", project["definitions"])
            self.assertNotIn("pald_community_analysis", project["requested_modules"])
            self.assertEqual(
                project["definitions"]["clustering_kmeans"]["feature_source"],
                "tica",
            )
            self.assertEqual(
                project["definitions"]["markov_state_models"]["assignment_sources"],
                ["best_clustering", "pca_fes_basins"],
            )
            self.assertEqual(
                project["definitions"]["markov_state_models"]["maximum_states"],
                250,
            )
            self.assertEqual(
                project["definitions"]["common_pca"]["symmetry_expansion"]["member_count"], 2
            )
            self.assertEqual(
                project["system_manifest"],
                "coordinate-cache/system-cache.json",
            )
            self.assertEqual(
                project["definitions"]["state_coordinate_exports"]
                ["coordinate_selection"],
                "molecular_payload",
            )
            self.assertEqual(
                project["definitions"]["state_coordinate_exports"]
                ["maximum_states"],
                250,
            )
            self.assertEqual(
                project["definitions"]["state_coordinate_exports"]
                ["maximum_total_frames"],
                500,
            )
            self.assertTrue(
                (output / "project-oligomer_member_interface_common_heavy.json").is_file()
            )
            self.assertTrue((output / "analysis-config.json").is_file())
            analysis_config = json.loads(
                (output / "analysis-config.json").read_text()
            )
            self.assertEqual(len(analysis_config["clustering"]["methods"]), 11)
            self.assertFalse(
                analysis_config["clustering"]["methods"]["hdbscan"]["enabled"]
            )
            self.assertTrue(all(
                row["enabled"]
                for method, row in analysis_config["clustering"]["methods"].items()
                if method != "hdbscan"
            ))
            self.assertFalse(
                analysis_config["community_analysis"]["pald"]["enabled"]
            )
            campaign = json.loads(
                (output / "campaign-resource-plan.json").read_text()
            )
            self.assertEqual(campaign["technical_status"], "complete")
            self.assertEqual(
                campaign["planning_scope"],
                "complete generated base including inferred chemistry plus "
                "conformational-view campaign",
            )
            task_rows = {row["task_id"]: row for row in campaign["tasks"]}
            self.assertIn("preprocessing:coordinate_cache", task_rows)
            self.assertFalse(
                task_rows["preprocessing:coordinate_cache"]["subsampling_triggered"]
            )
            self.assertIn("base:correlation_networks", task_rows)
            self.assertIn("base:convergence_uncertainty", task_rows)
            member_pca = task_rows[
                "view:oligomer_member_common_heavy:common_pca"
            ]
            member_fes = task_rows[
                "view:oligomer_member_common_heavy:pca_fes_basins"
            ]
            self.assertEqual(
                member_pca["selected_physical_frames_per_replica"],
                member_fes["selected_physical_frames_per_replica"],
            )
            self.assertEqual(
                member_pca["selected_member_observation_count"],
                2 * member_pca["selected_physical_frame_count"],
            )
            self.assertIn("campaign-resource-plan.json", report["generated_files"])
            self.assertIn("run_coordinate_cache.slurm", report["generated_files"])
            self.assertIn(
                "run-instrumented",
                (output / "run_view_oligomer_member_common_heavy_stage_0.slurm").read_text(),
            )
            submit_text = (output / "submit-conformational-views.sh").read_text()
            self.assertIn(
                "printf 'FINAL_JOB_IDS=%s\\n' \"${VIEW_0_STAGE_2_JOB}:"
                "${VIEW_1_STAGE_2_JOB}:${VIEW_2_STAGE_2_JOB}:"
                "${VIEW_3_STAGE_2_JOB}\"",
                submit_text,
            )
            self.assertNotIn("printf 'FINAL_JOB_IDS=${", submit_text)
            subprocess.run(["bash", "-n", str(output / "submit.sh")], check=True)
            subprocess.run([
                "bash", "-n", str(output / "submit-conformational-views.sh")
            ], check=True)
            subprocess.run([
                "bash", "-n", str(output / "run_finalize_reporting.slurm")
            ], check=True)
            finalizer_text = (output / "run_finalize_reporting.slurm").read_text()
            self.assertIn('rm "$RESOURCE_TMP"\n', finalizer_text)
            self.assertIn('rm "$FINDING_TMP"\n', finalizer_text)

    def test_finds_dssp_beside_active_python_when_not_on_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            interpreter = root / "python"
            interpreter.write_text("", encoding="utf-8")
            dssp = root / "mkdssp"
            dssp.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            dssp.chmod(0o755)
            with patch("salsbury_md_analysis.quickstart.shutil.which", return_value=None):
                with patch("salsbury_md_analysis.quickstart.sys.executable", str(interpreter)):
                    self.assertEqual(_discover_dssp_executable(None), str(dssp.resolve()))

    def test_invalid_declared_dssp_fails_during_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = str(Path(temporary) / "missing-mkdssp")
            with patch("salsbury_md_analysis.quickstart.shutil.which", return_value=None):
                with self.assertRaisesRegex(QuickstartError, "declared DSSP executable"):
                    _discover_dssp_executable(missing)

    def test_three_character_charmm_tip_is_counted_as_water_not_solute(self):
        with tempfile.TemporaryDirectory() as temporary:
            pdb = Path(temporary) / "tip-water.pdb"
            pdb.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
                "ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00  0.00           C\n"
                "ATOM      3  C   ALA A   1       2.900   0.000   0.000  1.00  0.00           C\n"
                "ATOM      4  O   ALA A   1       3.500   1.000   0.000  1.00  0.00           O\n"
                "HETATM    5  OH2 TIP W   2       8.000   8.000   8.000  1.00  0.00           O\n"
                "HETATM    6  H1  TIP W   2       8.900   8.000   8.000  1.00  0.00           H\n"
                "HETATM    7  H2  TIP W   2       7.700   8.850   8.000  1.00  0.00           H\n"
                "HETATM    8  OH2 TIP W   2      18.000  18.000  18.000  1.00  0.00           O\n"
                "HETATM    9  H1  TIP W   2      18.900  18.000  18.000  1.00  0.00           H\n"
                "HETATM   10  H2  TIP W   2      17.700  18.850  18.000  1.00  0.00           H\n"
                "END\n",
                encoding="utf-8",
            )
            composition = _composition(pdb)
            self.assertEqual(composition["water_residue_count"], 2)
            self.assertEqual(composition["solute_heavy_atom_count"], 4)

    def test_prepares_ready_to_submit_project_without_copying_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "inputs"
            source.mkdir()
            output = Path(temporary) / "analysis-v1"
            pdb, psf, trajectories = _write_inputs(source)
            report = prepare_standard_analysis(
                pdb_path=pdb,
                psf_path=psf,
                trajectories=trajectories,
                output_directory=output,
                project_id="tutorial-system",
                frame_interval_ps=10.0,
            )
            self.assertEqual(report["technical_status"], "complete")
            self.assertEqual(report["reference_connectivity_check"]["bond_count"], 3)
            project = json.loads((output / "project.json").read_text())
            sampling = json.loads((output / "sampling-plan.json").read_text())
            self.assertEqual(project["reference_connectivity"], str(psf.resolve()))
            self.assertEqual(
                project["definitions"]["hydrogen_bond_discovery"]["output_mode"],
                "sparse_packed_v2",
            )
            self.assertEqual(
                sampling["default_resource_envelope"]["target_wall_hours_per_method"],
                24.0,
            )
            self.assertEqual(
                sampling["campaign_resource_plan"]["raw_capacity_cpu_hours"],
                384.0,
            )
            safety = sampling["campaign_resource_plan"][
                "resource_safety_margins"
            ]
            self.assertEqual(safety["modeled_task_time_factor"], 1.5)
            self.assertEqual(safety["analysis_memory_model_factor"], 1.25)
            self.assertEqual(safety["scheduler_memory_safety_factor"], 1.5)
            self.assertEqual(safety["scheduler_memory_overhead_gib"], 1.0)
            self.assertTrue((output / "submit.sh").stat().st_mode & 0o100)
            self.assertTrue((output / "run-local.sh").stat().st_mode & 0o100)
            self.assertEqual(report["execution_adapter"], "local")
            self.assertTrue(report["next_command"].endswith("./run-local.sh"))
            local_plan = json.loads(
                (output / "local-execution-plan.json").read_text()
            )
            self.assertEqual(
                local_plan["local_execution_plan_schema"],
                "salsbury-local-execution-plan-v4",
            )
            self.assertEqual(local_plan["dependency_model"], "task_dag_v1")
            self.assertEqual(local_plan["maximum_parallel_cpus"], 8)
            campaign = json.loads(
                (output / "campaign-resource-plan.json").read_text()
            )
            self.assertEqual(campaign["maximum_parallel_cpus_input"], 16)
            self.assertEqual(campaign["effective_parallel_cpu_cap"], 8)
            self.assertEqual(
                campaign["resource_warnings"][0]["code"],
                "REQUESTED_CPUS_EXCEED_USEFUL_PARALLELISM",
            )
            self.assertIn(
                "Slurm submission will be changed to 8 CPUs",
                campaign["resource_warnings"][0]["message"],
            )
            self.assertEqual(local_plan["maximum_parallel_memory_gib"], 128.0)
            worker_paths = sorted(output.glob("run_stage_*_array.slurm"))
            workers = [path.read_text(encoding="utf-8") for path in worker_paths]
            worker = "\n".join(workers)
            preflight = (output / "run_preflight.slurm").read_text(encoding="utf-8")
            submit = (output / "submit.sh").read_text(encoding="utf-8")
            stages = json.loads((output / "workflow-stages.json").read_text())
            views = json.loads((output / "conformational-views.json").read_text())
            stage_lengths = [len(stage["commands"]) for stage in stages["stages"]]
            self.assertIn(stage_lengths[0], (8, 9))
            self.assertEqual(stage_lengths[1:], [2])
            for stage_worker in workers:
                self.assertIn("#SBATCH --time=24:00:00", stage_worker)
                self.assertIn("SALSBURY_MD_ANALYSIS_PYTHONPATH", stage_worker)
                self.assertIn("export PYTHONPATH=", stage_worker)
            self.assertIn("SALSBURY_MD_ANALYSIS_PYTHONPATH", preflight)
            self.assertIn("SALSBURY_MD_ANALYSIS_PYTHONPATH", submit)
            self.assertIn("preflight-system", preflight)
            self.assertIn(
                '--dependency="afterok:$CACHE_JOB"', submit
            )
            self.assertIn("--dependency=\"afterok:$PREFLIGHT_JOB\"", submit)
            self.assertIn("--dependency=\"afterok:$STAGE_0_JOB\"", submit)
            self.assertIn('FINAL_DEPENDENCIES="${STAGE_1_JOB}"', submit)
            self.assertIn(
                '--dependency="afterok:$FINAL_DEPENDENCIES"', submit
            )
            self.assertIn('PREFLIGHT_JOB="${PREFLIGHT_JOB%%;*}"', submit)
            self.assertIn("submit-conformational-views.sh", submit)
            self.assertIn("Reusing complete result", worker)
            self.assertIn("refusing to overwrite it", worker)
            self.assertIn('ln "$TMP" "$FINAL"', worker)
            self.assertIn("Revalidated and reused complete preflight", preflight)
            self.assertIn('ln "$TMP" "$FINAL"', preflight)
            self.assertIn("cmp -s", preflight)
            for stage in range(len(worker_paths)):
                subprocess.run(
                    ["bash", "-n", str(output / f"run_stage_{stage}_array.slurm")],
                    check=True,
                )
            subprocess.run(
                ["bash", "-n", str(output / "run_preflight.slurm")],
                check=True,
            )
            subprocess.run(
                ["bash", "-n", str(output / "run_coordinate_cache.slurm")],
                check=True,
            )
            subprocess.run(["bash", "-n", str(output / "submit.sh")], check=True)
            self.assertNotIn("run_module_array.slurm", report["generated_files"])
            self.assertIn("workflow-stages.json", report["generated_files"])
            self.assertIn('"correlation-networks"', worker)
            self.assertIn("SALSBURY_MD_ANALYSIS_COMMON_PCA_REPORT", workers[1])
            self.assertIn("SALSBURY_MD_ANALYSIS_DCCM_REPORT", workers[1])
            self.assertIn("SALSBURY_MD_ANALYSIS_RMSD_RG_REPORT", workers[1])
            self.assertIn("SALSBURY_MD_ANALYSIS_PREFLIGHT_REPORT", workers[1])
            view_workers = [
                path.read_text(encoding="utf-8")
                for path in output.glob("run_view_*_stage_*.slurm")
            ]
            self.assertTrue(view_workers)
            combined_view_workers = "\n".join(view_workers)
            self.assertIn('"information-correlation"', combined_view_workers)
            self.assertIn('"information-dynamics"', combined_view_workers)
            self.assertIn('"grouped-ml"', combined_view_workers)
            self.assertIn(
                "SALSBURY_MD_ANALYSIS_KMEANS_REPORT", combined_view_workers
            )
            self.assertIn(
                "SALSBURY_MD_ANALYSIS_PREFLIGHT_REPORT", combined_view_workers
            )
            for view_worker in view_workers:
                if "SALSBURY_MD_ANALYSIS_COMMON_PCA_REPORT" in view_worker:
                    self.assertIn(
                        "SALSBURY_MD_ANALYSIS_PREFLIGHT_REPORT", view_worker
                    )
            self.assertIn("generalized_correlation_and_information", project["definitions"])
            self.assertEqual(views["system_classification"], "protein_only")
            self.assertEqual(
                [view["view_id"] for view in views["views"]],
                ["global_common_heavy", "macromolecular_trace"],
            )
            self.assertTrue((output / "project-global_common_heavy.json").is_file())
            self.assertFalse((output / "project-macromolecular_trace.json").exists())
            retained_config = json.loads(
                (output / "analysis-config.json").read_text(encoding="utf-8")
            )
            self.assertFalse(
                retained_config["views"]["macromolecular_trace"]["enabled"]
            )
            self.assertFalse(
                retained_config["views"]["macromolecular_trace"]
                ["state_trajectory_exports_enabled"]
            )
            global_project = json.loads(
                (output / "project-global_common_heavy.json").read_text()
            )
            global_pca = global_project["definitions"]["common_pca"]
            self.assertEqual(
                global_project["selections"]["analysis"],
                global_project["selections"][global_pca["analysis_selection"]],
            )
            self.assertEqual(
                global_project["selections"]["alignment"],
                global_project["selections"][global_pca["alignment_selection"]],
            )
            self.assertEqual(
                global_project["definitions"]["state_coordinate_exports"]
                ["coordinate_selection"],
                "molecular_payload",
            )
            view_submit = output / "submit-conformational-views.sh"
            subprocess.run(["bash", "-n", str(view_submit)], check=True)
            for stage in range(3):
                view_worker = output / f"run_view_global_common_heavy_stage_{stage}.slurm"
                self.assertTrue(view_worker.is_file())
                subprocess.run(["bash", "-n", str(view_worker)], check=True)
            self.assertIn(
                '"state-coordinate-exports"',
                (output / "run_view_global_common_heavy_stage_2.slurm").read_text(),
            )
            self.assertEqual(len(list(output.glob("*.dcd"))), 0)

    def test_single_system_preparation_accepts_portable_bond_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "inputs"
            source.mkdir()
            output = Path(temporary) / "analysis-v1"
            pdb, _psf, trajectories = _write_inputs(source)
            connectivity = source / "system.bonds.json"
            connectivity.write_text(json.dumps({
                "format": "salsbury-bonds-v1",
                "atom_count": 4,
                "index_base": 0,
                "bonds": [[0, 1], [1, 2], [2, 3]],
                "provenance": {"source": "unit-test"},
            }), encoding="utf-8")
            report = prepare_standard_analysis(
                pdb_path=pdb,
                psf_path=connectivity,
                trajectories=trajectories,
                output_directory=output,
                project_id="portable-connectivity",
                frame_interval_ps=10.0,
            )
            self.assertEqual(report["technical_status"], "complete")
            self.assertEqual(
                report["reference_connectivity_check"]["connectivity_format"],
                "salsbury-bonds-v1",
            )
            project = json.loads((output / "project.json").read_text())
            self.assertEqual(
                project["reference_connectivity"], str(connectivity.resolve())
            )

    @patch(
        "salsbury_md_analysis.quickstart._discover_dssp_executable",
        return_value=None,
    )
    def test_memory_fallback_writes_requested_and_reduced_configs(
        self, _discover_dssp,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "inputs"
            source.mkdir()
            pdb, psf, trajectories = _write_inputs(source)
            config_path = root / "low-memory.json"
            config_path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "execution": {"maximum_memory_gib": 4.0},
            }), encoding="utf-8")
            output = root / "analysis-memory-fit"
            report = prepare_standard_analysis_memory_fit(
                pdb_path=pdb,
                psf_path=psf,
                trajectories=trajectories,
                output_directory=output,
                project_id="low-memory-system",
                frame_interval_ps=10.0,
                config_path=config_path,
            )
            self.assertEqual(report["technical_status"], "complete")
            memory = json.loads(
                (output / "memory-feasibility-report.json").read_text()
            )
            self.assertTrue(memory["automatic_changes_applied"])
            self.assertTrue(memory["requested_memory"]["memory_shortfall_gib"] > 0)
            self.assertTrue(memory["final_memory"]["fits_configured_memory"])
            requested = json.loads(
                (output / "analysis-config.requested.json").read_text()
            )
            reduced = json.loads(
                (output / "analysis-config.memory-fit.json").read_text()
            )
            disabled = memory["directly_disabled_configuration_switches"]
            self.assertEqual(
                disabled,
                ["modules.solvent_accessible_surface_area.enabled"],
            )
            self.assertTrue(
                requested["modules"]["solvent_accessible_surface_area"]["enabled"]
            )
            self.assertFalse(
                reduced["modules"]["solvent_accessible_surface_area"]["enabled"]
            )

    @patch("salsbury_md_analysis.quickstart.export_pdb_connectivity")
    def test_single_system_preparation_can_generate_openmm_connectivity(
        self, export_connectivity,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "inputs"
            source.mkdir()
            output = Path(temporary) / "analysis-v1"
            pdb, _psf, trajectories = _write_inputs(source)
            export_connectivity.return_value = {
                "format": "salsbury-bonds-v1",
                "atom_count": 4,
                "index_base": 0,
                "bonds": [[0, 1], [1, 2], [2, 3]],
                "provenance": {
                    "generator": "mock-openmm",
                    "scientific_status": "requires topology-owner review",
                },
            }
            report = prepare_standard_analysis(
                pdb_path=pdb,
                psf_path=None,
                trajectories=trajectories,
                output_directory=output,
                project_id="openmm-connectivity",
                frame_interval_ps=10.0,
                generate_connectivity_openmm=True,
            )
            self.assertEqual(report["technical_status"], "complete")
            self.assertEqual(report["connectivity_source"], "openmm_pdb_topology")
            generated = (
                output / "generated-connectivity" /
                "openmm-connectivity.bonds.json"
            )
            self.assertTrue(generated.is_file())
            project = json.loads((output / "project.json").read_text())
            self.assertEqual(
                project["reference_connectivity"], str(generated.resolve())
            )
            system = json.loads((output / "system.json").read_text())
            self.assertEqual(
                system["systems"][0]["replicas"][0]["connectivity"],
                str(generated.resolve()),
            )
            self.assertNotIn(str(generated), project["protected_locations"])
            export_connectivity.assert_called_once_with(
                pdb.resolve(), additional_bond_definitions=[]
            )

    def test_missing_connectivity_requires_explicit_openmm_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "inputs"
            source.mkdir()
            pdb, _psf, trajectories = _write_inputs(source)
            with self.assertRaisesRegex(
                QuickstartError, "generate-connectivity-openmm"
            ):
                prepare_standard_analysis(
                    pdb_path=pdb,
                    psf_path=None,
                    trajectories=trajectories,
                    output_directory=Path(temporary) / "analysis-v1",
                    project_id="missing-connectivity",
                    frame_interval_ps=10.0,
                )

    def test_inferred_chemistry_commands_have_dependency_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "inputs"
            source.mkdir()
            output = Path(temporary) / "analysis-v1"
            pdb, psf, trajectories = _write_ion_inputs(source)
            report = prepare_standard_analysis(
                pdb_path=pdb, psf_path=psf, trajectories=trajectories,
                output_directory=output, project_id="ion-staging",
                frame_interval_ps=10.0,
            )
            self.assertEqual(report["technical_status"], "complete")
            stages = json.loads((output / "workflow-stages.json").read_text())
            stage_zero = next(row for row in stages["stages"] if row["stage"] == 0)
            stage_one = next(row for row in stages["stages"] if row["stage"] == 1)
            self.assertTrue(
                set(stage_zero["commands"]).issubset(_GENERIC_STAGE_COMMANDS[0])
            )
            self.assertIn("trajectory-features", stage_zero["commands"])
            self.assertNotIn("observables", stage_zero["commands"])
            self.assertIn("scalar-distributions", stage_one["commands"])
            self.assertIn("scalar-threshold-states", stage_one["commands"])
            stage_one_worker = (output / "run_stage_1_array.slurm").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "SALSBURY_MD_ANALYSIS_TRAJECTORY_FEATURES_REPORT",
                stage_one_worker,
            )
            campaign = json.loads(
                (output / "campaign-resource-plan.json").read_text()
            )
            task_ids = {row["task_id"] for row in campaign["tasks"]}
            self.assertIn("base:trajectory_features", task_ids)
            self.assertIn("base:scalar_feature_distributions", task_ids)
            self.assertIn("base:scalar_threshold_states", task_ids)
            self.assertNotIn("base:optional_observables", task_ids)
            staged = set().union(*_GENERIC_STAGE_COMMANDS.values())
            self.assertTrue(
                set(_GENERIC_CHEMISTRY_COMMANDS.values()).issubset(staged)
            )

    def test_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdb, psf, trajectories = _write_inputs(root)
            output = root / "used"
            output.mkdir()
            (output / "existing.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaises(QuickstartError):
                prepare_standard_analysis(
                    pdb_path=pdb,
                    psf_path=psf,
                    trajectories=trajectories,
                    output_directory=output,
                    project_id="tutorial-system",
                    frame_interval_ps=10.0,
                )

    def test_rejects_geometrically_incompatible_pdb_psf_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "inputs"
            source.mkdir()
            output = Path(temporary) / "analysis-v1"
            pdb, psf, trajectories = _write_inputs(source)
            text = pdb.read_text(encoding="utf-8")
            pdb.write_text(text.replace("   3.500   1.000   0.000", "  30.000   1.000   0.000"), encoding="utf-8")
            with self.assertRaisesRegex(
                QuickstartError, "PDB/connectivity reference geometry is incompatible"
            ):
                prepare_standard_analysis(
                    pdb_path=pdb,
                    psf_path=psf,
                    trajectories=trajectories,
                    output_directory=output,
                    project_id="broken-reference",
                    frame_interval_ps=10.0,
                )
            self.assertFalse(output.exists())

    def test_applies_planned_integer_strides_to_streaming_modules(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "inputs"
            source.mkdir()
            output = Path(temporary) / "large-analysis-v1"
            pdb, psf, trajectories = _write_inputs(source)
            for trajectory in trajectories:
                _write_dcd(trajectory, atom_count=4, frames=300_000)
            prepare_standard_analysis(
                pdb_path=pdb,
                psf_path=psf,
                trajectories=trajectories,
                output_directory=output,
                project_id="large-tutorial-system",
                frame_interval_ps=10.0,
            )
            project = json.loads((output / "project.json").read_text())
            plans = {
                row["module_id"]: row
                for row in json.loads((output / "sampling-plan.json").read_text())
                ["method_plans"]
            }
            structural = plans["structural_integrity_qc"]
            structural_stride = structural["frame_selection"]["stride"]
            self.assertGreater(structural_stride, 1)
            self.assertEqual(
                project["definitions"]["structural_qc"]["frame_selection"],
                structural["frame_selection"],
            )
            self.assertEqual(
                structural["selected_frame_count"],
                3 * ((300_000 + structural_stride - 1) // structural_stride),
            )
            for module_id, definition_id in (
                ("replica_rmsd_rg", "replica_rmsd_rg"),
                ("pooled_rmsf", "pooled_rmsf"),
                ("dihedral_distributions", "dihedral_distributions"),
            ):
                stride = plans[module_id]["frame_stride"]
                self.assertGreater(stride, 1)
                self.assertEqual(project["definitions"][definition_id]["frame_stride"], stride)
                self.assertEqual(
                    plans[module_id]["selected_frame_count"],
                    3 * ((300_000 + stride - 1) // stride),
                )
            individual = plans["individual_pca"]
            individual_selection = individual["frame_selection"]
            individual_stride = individual_selection["stride"]
            individual_definition = project["definitions"]["individual_pca"]
            self.assertGreater(individual_stride, 1)
            self.assertEqual(
                individual_definition["frame_selection"], individual_selection
            )
            self.assertEqual(
                individual_definition["projection_frame_selection"],
                individual_selection,
            )
            self.assertEqual(
                individual["selected_frame_count"],
                3 * ((300_000 + individual_stride - 1) // individual_stride),
            )


if __name__ == "__main__":
    unittest.main()
