import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.analysis_config import DEFAULT_DISABLED_MODULES
from salsbury_md_analysis.preflight import (
    probe_connectivity,
    probe_topology,
    probe_trajectory,
)
from salsbury_md_analysis.quickstart import prepare_standard_analysis


ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "tutorials" / "nemo_zinc_finger"
DATA = TUTORIAL / "data"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NemoZincFingerTutorialTests(unittest.TestCase):
    def test_energetic_network_module_is_explicitly_unavailable_for_psf_fixture(self):
        acceptance = json.loads(
            (ROOT / "validation" /
             "nemo_energetic_network_embeddings_acceptance.json")
            .read_text(encoding="utf-8")
        )
        module = acceptance["module_acceptance"]
        planner = acceptance["planner_acceptance"]
        self.assertTrue(module["configuration_enabled"])
        self.assertEqual(module["availability_status"], "not_available")
        self.assertFalse(module["analysis_performed"])
        self.assertFalse(planner["planner_task_created"])
        self.assertEqual(planner["energetic_network_task_count"], 0)
        self.assertFalse(
            acceptance["scope_boundary"]["solvent_inclusive_extension_implemented"]
        )
        self.assertFalse(
            acceptance["scope_boundary"]["signed_energy_extension_implemented"]
        )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nemo-energetic-availability"
            report = prepare_standard_analysis(
                pdb_path=DATA / "nemo_zinc_finger.pdb",
                psf_path=DATA / "nemo_zinc_finger.psf",
                trajectories=[DATA / "nemo_zinc_finger_1000_frames.dcd"],
                output_directory=output,
                project_id="nemo-energetic-availability-test",
                frame_interval_ps=0.2,
                config_path=TUTORIAL / "energetic-network-analysis-config.json",
            )
            self.assertEqual(report["technical_status"], "complete")
            availability = json.loads(
                (output / "energetic-network-embeddings-availability.json")
                .read_text(encoding="utf-8")
            )
            campaign = json.loads(
                (output / "campaign-resource-plan.json").read_text(encoding="utf-8")
            )
            project = json.loads(
                (output / "project.json").read_text(encoding="utf-8")
            )
        self.assertEqual(availability["availability_status"], "not_available")
        self.assertFalse(availability["planner_task_created"])
        self.assertNotIn("energetic_network_embeddings", project["requested_modules"])
        self.assertIn("energetic_network_embeddings", project["definitions"])
        self.assertFalse(any(
            row.get("module_id") == "energetic_network_embeddings"
            for row in campaign["tasks"]
        ))

    def test_extended_experimental_fixture_enables_new_paper_methods(self):
        config = json.loads(
            (TUTORIAL / "interaction-fingerprint-analysis-config.json")
            .read_text(encoding="utf-8")
        )
        self.assertTrue(config["enable_all_experimental_modules"])
        self.assertTrue(
            config["modules"]["interaction_persistence"]["enabled"]
        )
        self.assertTrue(
            config["modules"]["spatial_interaction_ensembles"]["enabled"]
        )
        self.assertTrue(
            config["modules"]["random_feature_koopman"]["enabled"]
        )
        acceptance = json.loads(
            (ROOT / "validation" /
             "nemo_persistence_random_feature_koopman_acceptance.json")
            .read_text(encoding="utf-8")
        )
        persistence = acceptance["interaction_persistence_acceptance"]
        nonlinear = acceptance["random_feature_koopman_acceptance"]
        self.assertEqual(persistence["evaluated_frame_count"], 1000)
        self.assertEqual(persistence["primary_zero_gap_pass_count"], 58)
        self.assertEqual(nonlinear["seed_evaluation_count"], 24)
        self.assertEqual(nonlinear["selection_status"], "no_stable_candidate")
        self.assertEqual(acceptance["scientific_status"], "not evaluated")

        spatial = json.loads(
            (ROOT / "validation" /
             "nemo_spatial_interaction_ensembles_acceptance.json")
            .read_text(encoding="utf-8")
        )
        module = spatial["module_acceptance"]
        planner = spatial["planner_acceptance"]
        self.assertEqual(planner["upstream_module_id"], "interaction_fingerprints")
        self.assertEqual(planner["selected_physical_frame_count"], 1000)
        self.assertEqual(module["point_observation_count"], 16064)
        self.assertEqual(module["selected_spatial_mode_candidate_count"], 9)
        self.assertEqual(
            module["zinc_site_mode_inference_status"],
            "withheld_by_exact_mode_resource_gate",
        )
        self.assertEqual(module["pairwise_system_spatial_difference_count"], 0)
        self.assertEqual(module["scientific_status"], "not evaluated")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nemo-all-experimental"
            prepared = prepare_standard_analysis(
                pdb_path=DATA / "nemo_zinc_finger.pdb",
                psf_path=DATA / "nemo_zinc_finger.psf",
                trajectories=[DATA / "nemo_zinc_finger_1000_frames.dcd"],
                output_directory=output,
                project_id="nemo-all-experimental-planner-test",
                frame_interval_ps=0.2,
                config_path=(
                    TUTORIAL / "interaction-fingerprint-analysis-config.json"
                ),
            )
            campaign = json.loads(
                (output / "campaign-resource-plan.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(prepared["technical_status"], "complete")
        coverage = campaign["experimental_module_coverage"]
        rows = {
            row["module_id"]: row for row in coverage["modules"]
        }
        self.assertEqual(set(rows), DEFAULT_DISABLED_MODULES)
        self.assertEqual(coverage["enabled_module_count"], 13)
        self.assertEqual(coverage["planned_module_count"], 11)
        self.assertEqual(coverage["not_available_module_count"], 2)
        self.assertEqual(
            rows["energetic_network_embeddings"]["planner_status"],
            "not_available",
        )
        self.assertEqual(
            rows["helical_mechanics"]["planner_status"], "not_available"
        )
        for module_id in DEFAULT_DISABLED_MODULES.difference({
            "energetic_network_embeddings", "helical_mechanics",
        }):
            self.assertEqual(rows[module_id]["planner_status"], "planned")
            self.assertTrue(rows[module_id]["planner_task_ids"])
        combined_acceptance = json.loads(
            (ROOT / "validation" /
             "nemo_all_experimental_methods_acceptance.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(
            combined_acceptance["planner_acceptance"][
                "enabled_experimental_module_count"
            ],
            len(DEFAULT_DISABLED_MODULES),
        )
        self.assertEqual(
            combined_acceptance["finding_picker_acceptance"][
                "missing_experimental_modules"
            ],
            [],
        )
        self.assertEqual(
            combined_acceptance["interactive_dashboard_acceptance"][
                "missing_experimental_modules"
            ],
            [],
        )

    def test_fixture_matches_documented_lineage_and_dimensions(self):
        provenance = json.loads(
            (DATA / "FIXTURE_PROVENANCE.json").read_text(encoding="utf-8")
        )
        records = {row["path"]: row for row in provenance["derived_records"]}
        for name, record in records.items():
            self.assertEqual(sha256(DATA / name), record["sha256"])

        pdb = probe_topology(DATA / "nemo_zinc_finger.pdb")
        psf = probe_connectivity(DATA / "nemo_zinc_finger.psf")
        dcd = probe_trajectory(DATA / "nemo_zinc_finger_1000_frames.dcd")
        self.assertEqual(pdb["atom_count"], 423)
        self.assertEqual(psf["atom_count"], 423)
        self.assertEqual(dcd["atom_count"], 423)
        self.assertEqual(dcd["declared_frame_count"], 1000)

    def test_generic_preparation_recognizes_protein_and_zinc(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "tutorial-run"
            report = prepare_standard_analysis(
                pdb_path=DATA / "nemo_zinc_finger.pdb",
                psf_path=DATA / "nemo_zinc_finger.psf",
                trajectories=[DATA / "nemo_zinc_finger_1000_frames.dcd"],
                output_directory=output,
                project_id="nemo-zinc-finger-tutorial-test",
                frame_interval_ps=0.2,
                config_path=TUTORIAL / "analysis-config.json",
            )
            self.assertEqual(report["technical_status"], "complete")
            self.assertEqual(report["execution_adapter"], "local")

            views = json.loads(
                (output / "conformational-views.json").read_text(encoding="utf-8")
            )
            self.assertEqual(views["system_classification"], "protein_only")
            config = json.loads(
                (output / "analysis-config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config["execution"]["maximum_parallel_cpus"], 2)
            self.assertEqual(config["execution"]["maximum_hours_per_cpu"], 1.0)
            self.assertEqual(config["execution"]["maximum_memory_gib"], 32.0)
            self.assertTrue(config["inference"]["ion_site_classification_enabled"])
            self.assertFalse(
                config["views"]["global_common_heavy"]
                ["state_trajectory_exports_enabled"]
            )
            project = json.loads(
                (output / "project.json").read_text(encoding="utf-8")
            )
            kmeans = project["definitions"]["clustering_kmeans"]
            self.assertEqual(
                kmeans["initialization_methods"],
                ["nani_strat_all", "nani_strat_reduced"],
            )
            self.assertEqual(kmeans["nani_percentage"], 10)
            self.assertEqual(
                kmeans["silhouette_random_seeds"], [0, 7, 19, 41]
            )
            self.assertNotIn("random_seeds", kmeans)
            self.assertNotIn(
                "profile_clustering",
                project["definitions"]["correlation_networks"],
            )

            chemistry = json.loads(
                (output / "automatic-chemical-context.json").read_text(
                    encoding="utf-8"
                )
            )
            candidates = chemistry["inference"]["ion_candidates"]
            self.assertIn(422, [row["ion_atom_index"] for row in candidates])
            self.assertEqual(chemistry["inference"]["ion_atmosphere_species"], ["ZN"])
            coverage = json.loads(
                (output / "module-coverage.json").read_text(encoding="utf-8")
            )
            statuses = coverage["module_status"]
            self.assertEqual(statuses["ion_coordination_geometry"]["status"], "automatic")
            self.assertEqual(statuses["ion_atmosphere"]["status"], "automatic")


if __name__ == "__main__":
    unittest.main()
