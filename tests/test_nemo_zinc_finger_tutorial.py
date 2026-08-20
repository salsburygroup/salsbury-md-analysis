import hashlib
import json
import tempfile
import unittest
from pathlib import Path

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
