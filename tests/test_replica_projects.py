import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.manifests import load_json
from salsbury_md_analysis.replica_projects import materialized_replica_project_shards


class ReplicaProjectShardTests(unittest.TestCase):
    def test_materializes_absolute_identity_preserving_projects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            topology = root / "topology.pdb"
            trajectory = root / "trajectory.pdb"
            topology.write_text(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
                encoding="utf-8",
            )
            trajectory.write_text(topology.read_text(encoding="utf-8"), encoding="utf-8")
            system = {
                "systems": [{
                    "system_id": "sys", "replicas": [
                        {
                            "replica_id": replica_id,
                            "topology": "topology.pdb",
                            "segments": [{
                                "segment_id": "seg", "trajectory": "trajectory.pdb",
                                "timing": {"first_frame_time": 0.0, "frame_interval": 1.0, "unit": "ns"},
                            }],
                        }
                        for replica_id in ("r1", "r2")
                    ],
                }]
            }
            (root / "system.json").write_text(json.dumps(system), encoding="utf-8")
            project = {
                "project_id": "test", "analysis_profile": "test",
                "system_manifest": "system.json", "analysis_output_root": str(root / "out"),
                "sampling_mode": "UNBIASED_MD", "protected_locations": [],
                "coordinate_unit": "angstrom", "time_unit": "ns",
                "periodic_coordinate_policy": "reject",
                "selections": {"alignment": {"preset": "all"}, "analysis": {"preset": "all"}},
            }
            project_path = root / "project.json"
            project_path.write_text(json.dumps(project), encoding="utf-8")
            with materialized_replica_project_shards(project_path) as (shards, original):
                self.assertEqual([shard.identity for shard in shards], [("sys", "r1"), ("sys", "r2")])
                for shard in shards:
                    shard_project = load_json(Path(shard.payload["project_path"]))
                    shard_system = load_json(Path(shard_project["system_manifest"]))
                    replica = shard_system["systems"][0]["replicas"][0]
                    self.assertTrue(Path(replica["topology"]).is_absolute())
                    self.assertTrue(Path(replica["segments"][0]["trajectory"]).is_absolute())
                self.assertEqual(original["project_id"], "test")


if __name__ == "__main__":
    unittest.main()
