import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.molecular_evidence import finding_signature, panel_matches_finding
from salsbury_md_analysis.presentation_artifacts import _export_inventory_artifacts, PresentationArtifactError


class MolecularEvidenceTests(unittest.TestCase):
    def test_exact_claim_coordinates_and_saved_view_required(self):
        finding = {"module_id": "alternative_clustering", "statement": "State 1 differs",
                   "presentation_target": {"algorithm": "pam"}}
        panel = {"artifact_type": "figure", "purpose": "structural_figure", "molecular_evidence": {
            "finding_signature_sha256": finding_signature(finding),
            "coordinate_artifacts": [{"artifact_id": "pdb", "sha256": "a"*64}],
            "saved_view_artifact_id": "view", "saved_view_sha256": "b"*64}}
        artifacts = {"pdb": {"artifact_type": "structure", "artifact_sha256": "a"*64},
                     "view": {"artifact_type": "table", "purpose": "molecular_saved_view", "artifact_sha256": "b"*64}}
        self.assertTrue(panel_matches_finding(panel, finding, artifacts, set(artifacts)))
        self.assertTrue(panel_matches_finding(panel, {**finding, "finding_id": "new-rank"}, artifacts, set(artifacts)))
        for update in ({"statement": "State 2 differs"}, {"presentation_target": {"algorithm": "kmeans"}}):
            self.assertFalse(panel_matches_finding(panel, {**finding, **update}, artifacts, set(artifacts)))
        self.assertFalse(panel_matches_finding(panel, finding, artifacts, {"pdb"}))
        for key in ("coordinate_artifacts", "saved_view_sha256", "finding_signature_sha256"):
            bad = copy.deepcopy(panel)
            bad["molecular_evidence"].pop(key)
            self.assertFalse(panel_matches_finding(bad, finding, artifacts, set(artifacts)))
        bad = copy.deepcopy(artifacts)
        bad["pdb"]["artifact_sha256"] = "c"*64
        self.assertFalse(panel_matches_finding(panel, finding, bad, set(bad)))

    def test_state_pdb_paths_preserve_export_identity_and_verify_declared_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = []
            hashes = set()
            for scope in ("shared", "individual"):
                source = root / scope
                source.mkdir()
                pdb = source / "representative.pdb"
                pdb.write_text("REMARK " + scope + "\nEND\n")
                sha = hashlib.sha256(pdb.read_bytes()).hexdigest()
                hashes.add(sha)
                report = {"module_id": "state_coordinate_exports", "source_module_id": "pca_fes_basins",
                          "export_directory": str(source), "settings": {"export_id": scope},
                          "outputs": [{"system_id": "control", "state_id": 1, "representatives": [{
                              "path": pdb.name, "sha256": sha, "replica_id": "rep1", "source_frame_index": 42}]}]}
                path = source / "report.json"
                path.write_text(json.dumps(report))
                _export_inventory_artifacts(root / "artifacts", path, report, "state_coordinate_exports", artifacts)
            structures = [a for a in artifacts if a["artifact_type"] == "structure"]
            self.assertEqual(len({a["relative_path"] for a in artifacts}), len(artifacts))
            self.assertEqual(len({a["artifact_id"] for a in artifacts}), len(artifacts))
            self.assertEqual(len({a["relative_path"] for a in structures}), 2)
            self.assertEqual(len({a["artifact_id"] for a in structures}), 2)
            self.assertEqual({hashlib.sha256((root / "artifacts" / a["relative_path"]).read_bytes()).hexdigest() for a in structures}, hashes)
            self.assertTrue(all(a["context"]["source_frame_index"] == 42 for a in structures))
            report["outputs"][0]["representatives"][0]["sha256"] = "0"*64
            with self.assertRaisesRegex(PresentationArtifactError, "hash mismatch"):
                _export_inventory_artifacts(root / "bad", path, report, "state_coordinate_exports", [])
