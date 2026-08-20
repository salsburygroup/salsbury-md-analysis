import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.nucleic_acid_structure import (
    build_dssr_json_command,
    nucleic_acid_structure_project,
    parse_dssr_collection_counts,
    extract_numeric_json_path,
)


def _atom(serial, name, x, element):
    return (
        f"ATOM  {serial:5d} {name:^4s}  DA A   1    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {element:>2s}\n"
    )


def _project(root: Path) -> Path:
    executable = root / "fake-dssr"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "if '--version' in sys.argv:\n"
        "    print('x3dna-dssr 2.4.0-test')\n"
        "    raise SystemExit(0)\n"
        "output = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--output='))\n"
        "pathlib.Path(output).write_text(json.dumps({'pairs': [{'bp1_params': {'shear': 0.1}}, {'bp1_params': {'shear': -0.2}}], 'helices': [{}]}))\n",
        encoding="utf-8",
    )
    os.chmod(executable, 0o755)
    topology = _atom(1, "P", 0.0, "P") + _atom(2, "C1'", 2.0, "C") + "END\n"
    (root / "topology.pdb").write_text(topology, encoding="ascii")
    (root / "trajectory.xyz").write_text(
        "2\nframe 0\nP 0 0 0\nC 2 0 0\n"
        "2\nframe 1\nP 0 0 0\nC 2.1 0 0\n",
        encoding="ascii",
    )
    (root / "system.json").write_text(json.dumps({
        "systems": [{"system_id": "dna", "replicas": [{
            "replica_id": "r1", "topology": "topology.pdb",
            "segments": [{
                "segment_id": "s1", "trajectory": "trajectory.xyz",
                "timing": {"first_frame_time": 0, "frame_interval": 1, "unit": "ps"},
            }],
        }]}],
    }), encoding="utf-8")
    project = {
        "project_id": "dssr-test", "analysis_profile": "standard_md_v1",
        "system_manifest": "system.json", "analysis_output_root": "outputs",
        "sampling_mode": "UNBIASED_MD", "coordinate_unit": "angstrom",
        "time_unit": "ps", "periodic_coordinate_policy": "reject",
        "selections": {"alignment": {"preset": "all"}, "analysis": {"preset": "all"}},
        "definitions": {"nucleic_acid_structure": {
            "method": "x3dna-dssr-json", "executable": str(executable),
            "frame_stride": 1, "maximum_frames": 4, "timeout_seconds": 10,
            "json_collection_fields": ["pairs", "helices", "junctions"],
            "numeric_queries": [{
                "query_id": "base-pair-shear",
                "path": ["pairs", "*", "bp1_params", "shear"],
                "missing_policy": "fail",
            }],
        }},
        "requested_modules": ["nucleic_acid_structure"],
        "protected_locations": ["/protected/example"],
    }
    path = root / "project.json"
    path.write_text(json.dumps(project), encoding="utf-8")
    return path


class NucleicAcidStructureTests(unittest.TestCase):
    def test_command_and_json_count_contract_are_explicit(self):
        command = build_dssr_json_command("dssr", Path("in.pdb"), Path("out.json"))
        self.assertEqual(command[:3], ["dssr", "--json", "--more"])
        self.assertEqual(
            parse_dssr_collection_counts(
                {"pairs": [{}, {}], "helices": 1},
                ["pairs", "helices", "junctions"],
            ),
            {"pairs": 2, "helices": 1, "junctions": 0},
        )
        self.assertEqual(
            extract_numeric_json_path(
                {"pairs": [{"shear": 0.1}, {"shear": -0.2}]},
                ["pairs", "*", "shear"],
            ),
            [0.1, -0.2],
        )

    def test_project_runs_without_shell_and_retains_version_and_replica_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = nucleic_acid_structure_project(_project(Path(temporary)))
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["evaluated_frame_count"], 2)
        self.assertFalse(report["implementation"]["shell"])
        self.assertIn("2.4.0-test", report["implementation"]["version_output"])
        self.assertEqual(
            report["frame_reports"][0]["collection_counts"],
            {"pairs": 2, "helices": 1, "junctions": 0},
        )
        self.assertEqual(
            report["replica_summaries"][0]["collection_count_summaries"]["pairs"]["mean"],
            2.0,
        )
        self.assertEqual(
            report["frame_reports"][0]["numeric_queries"][0]["values"],
            [0.1, -0.2],
        )


if __name__ == "__main__":
    unittest.main()
