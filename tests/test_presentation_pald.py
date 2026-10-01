"""PaLD presentations must not turn sampled communities into full populations."""

import copy
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.presentation_artifacts import (
    PresentationArtifactError, generate_presentation_artifacts,
)


def report_fixture():
    observations = [{"system_id": sid, "replica_id": rid, "segment_id": "full",
                     "source_frame_index": i*10, "pald_sample_index": i,
                     "community_id": cid, "local_depth": depth,
                     "boundary_cohesion_fraction": boundary}
                    for i, (sid, rid, cid, depth, boundary) in enumerate([
                        ("A", "r1", 1, .6, .1), ("A", "r2", 1, .5, .2),
                        ("A", "r2", 2, .4, .3), ("B", "r1", 2, .3, .4)])]
    return {"module_id": "pald_community_analysis", "technical_status": "complete",
            "scientific_status": "not evaluated", "source_observation_count": 80,
            "sampled_observation_count": 4, "sampled_observations": observations,
            "communities": [{"community_id": cid, "sampled_population": 2,
                             "sampled_population_fraction": .5,
                             "mean_local_depth": depth, "core_observation": observations[index]}
                            for cid, index, depth in [(1, 0, .55), (2, 2, .35)]],
            "strongest_intercommunity_ties": [{"left_sample_index": 0,
                                              "right_sample_index": 2, "mutual_cohesion": .01}]}


class PaldPresentationTests(unittest.TestCase):
    def write_report(self, root, report, relative="conformational-views/global_common_heavy/pald-community"):
        path = root / "results" / relative / "report.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(report))
        return path

    def test_denominators_provenance_and_complete_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = report_fixture()
            original = copy.deepcopy(report)
            path = self.write_report(root, report)
            expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = generate_presentation_artifacts(root)
            self.assertEqual(report, original)
            self.assertEqual(manifest["unadapted_report_count"], 0)
            self.assertEqual(manifest["technical_status"], "complete")
            self.assertEqual(manifest["reviewed_reports"][0]["presentation_adapter"], "complete")
            figures = [a for a in manifest["artifacts"] if a["artifact_type"] == "figure"]
            self.assertEqual(len(figures), 3)
            for artifact in manifest["artifacts"]:
                self.assertEqual(artifact["source_report_sha256"], [expected_hash])
                self.assertEqual(artifact["context"]["denominator_scope"], "sampled_observations")
                self.assertIn("not all-frame populations", artifact["context"]["limitations"])
                output = root / "presentation-artifacts" / artifact["relative_path"]
                self.assertTrue(output.is_file())
                if artifact["artifact_type"] == "figure":
                    text = output.read_text()
                    self.assertIn("4 of 80", text)
                    self.assertIn("Sampled observation fraction", text)
                    self.assertNotIn(">Frame fraction<", text)
            systems = next(a for a in manifest["artifacts"] if a["purpose"] == "system_sampled_communities")
            with (root / "presentation-artifacts" / systems["relative_path"]).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            a_first = next(r for r in rows if r["system_id"] == "A" and r["community_id"] == "1")
            b_first = next(r for r in rows if r["system_id"] == "B" and r["community_id"] == "1")
            self.assertAlmostEqual(float(a_first["fraction_of_system_sample"]), 2/3)
            self.assertEqual(float(b_first["fraction_of_system_sample"]), 0)
            for artifact in manifest["artifacts"]:
                if artifact["artifact_type"] == "table" and artifact["purpose"] in {"local_depth", "boundary_cohesion_fraction"}:
                    with (root / "presentation-artifacts" / artifact["relative_path"]).open() as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertEqual(sum(int(row["count"]) for row in rows), 4)
                    self.assertAlmostEqual(sum(float(row["fraction"]) for row in rows), 1)
                    self.assertFalse(any("angstrom" in key for key in rows[0]))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected_hash)

    def test_missing_or_inconsistent_source_values_fail_closed(self):
        for mutate in (
            lambda r: r.update(sampled_observation_count=5),
            lambda r: r["communities"][0].update(sampled_population_fraction=.25),
            lambda r: r["communities"].pop(),
            lambda r: r["sampled_observations"][1].update(pald_sample_index=0),
            lambda r: r["sampled_observations"][0].update(local_depth=float("nan")),
            lambda r: r["sampled_observations"][0].update(boundary_cohesion_fraction=1.1),
        ):
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temporary:
                report = report_fixture()
                mutate(report)
                root = Path(temporary)
                self.write_report(root, report)
                with self.assertRaises(PresentationArtifactError):
                    generate_presentation_artifacts(root)

    def test_scopes_do_not_collide_and_source_scientific_status_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for sid in ("A", "B"):
                for view in ("global_common_heavy", "macromolecular_trace"):
                    paths.append(self.write_report(root, report_fixture(),
                        f"per-system/{sid}/conformational-views/{view}/pald-community"))
            manifest = generate_presentation_artifacts(root)
            self.assertEqual(len(manifest["reviewed_reports"]), 4)
            ids = [a["artifact_id"] for a in manifest["artifacts"]]
            filenames = [a["relative_path"] for a in manifest["artifacts"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(filenames), len(set(filenames)))
            self.assertTrue(all(json.loads(p.read_text())["scientific_status"] == "not evaluated" for p in paths))


if __name__ == "__main__":
    unittest.main()
