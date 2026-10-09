"""A zero-observation segment still has a coordinate integrity contract."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from salsbury_md_analysis.coordinate_cache import _DCDWriter
from salsbury_md_analysis.coordinates import CoordinateFrame, CoordinateReadError
from salsbury_md_analysis.frame_sampling import plan_frame_selection
from salsbury_md_analysis.pca import PCAAnalysisError, _scan_replica, common_pca_project
from salsbury_md_analysis.pca_math import CartesianCovariance
from salsbury_md_analysis.periodic import PeriodicFrameProcessor, PeriodicReconstructionError
from test_pca import _frame, _pdb_atom, _write_project


def scan_segment(path, *, atom_count=1, processor=None, state=None, selected=None, timing=None):
    segment = {"segment_id": "short", "trajectory": str(path),
               "continuous_with_previous": False,
               "timing": timing or {"first_frame_time": 18800, "frame_interval": 100, "unit": "ps"}}
    plan = SimpleNamespace(
        system_id="synthetic", replica_id="r1", replica={"segments": [segment]},
        target_atoms=[None] * atom_count,
        alignment=SimpleNamespace(target_indices=(0,), reference_indices=(0,)),
        analysis=SimpleNamespace(target_indices=(0,), reference_indices=(0,)),
    )
    processor = processor or PeriodicFrameProcessor("preprocessed_make_whole", atom_count)
    with patch("salsbury_md_analysis.pca.PeriodicFrameProcessor.from_replica", return_value=processor):
        return _scan_replica(
            plan, {}, path.parent / "system.json", [(0., 0., 0.)],
            "angstrom", "ps", "unwrap_continuous", 1,
            {("synthetic", "r1", "short"): set() if selected is None else selected},
            state=state,
        )[0]


class PCAEmptySelectionTests(unittest.TestCase):
    def test_incident_two_model_pdb_is_valid_but_contributes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "two-model.pdb"
            path.write_text("MODEL        1\n" + _pdb_atom(1, "P", 0, 0, 0, "P")
                            + "ENDMDL\nMODEL        2\n" + _pdb_atom(1, "P", 1, 0, 0, "P")
                            + "ENDMDL\nEND\n")
            before = path.read_bytes()
            state = CartesianCovariance(3)
            row = scan_segment(path, state=state)
            self.assertEqual(row["observed_frame_count"], 2)
            self.assertEqual(row["evaluated_frame_count"], 0)
            self.assertEqual(state.count, 0)
            self.assertIsNone(row["evaluated_time_range"])
            self.assertEqual(row["timing"]["first_frame_time"], 18800)
            self.assertEqual(path.read_bytes(), before)

    def test_empty_corrupt_nonfinite_and_wrong_topology_still_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            for text in ("", "1\ntruncated\n", "1\nbad\nC nan 0 0\n",
                         "2\nwrong topology\nC 0 0 0\nC 1 0 0\n"):
                with self.subTest(text=text):
                    path = Path(tmp) / "bad.xyz"
                    path.write_text(text)
                    with self.assertRaises((PCAAnalysisError, CoordinateReadError,
                                            PeriodicReconstructionError)):
                        scan_segment(path)

    def test_periodic_cache_payload_and_box_are_validated_not_reconstructed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.dcd"
            writer = _DCDWriter(path, atom_count=1, frame_count=2, starting_step=0,
                                save_interval=1, unit_cell_present=True)
            cell = ((10., 0., 0.), (0., 10., 0.), (0., 0., 10.))
            for index in range(2):
                writer.write(CoordinateFrame(index, [(float(index), 0., 0.)],
                                              "angstrom", True, cell), [0])
            writer.close()
            before = path.read_bytes()
            processor = PeriodicFrameProcessor("preprocessed_make_whole", 1,
                                                preprocessed_cache_identity={"verified": True})
            with patch("salsbury_md_analysis.periodic._make_whole_components",
                       side_effect=AssertionError("cache must not be reconstructed")):
                row = scan_segment(path, processor=processor)
            self.assertEqual(row["observed_frame_count"], 2)
            self.assertEqual(row["periodic_cell_frame_count"], 2)
            self.assertEqual(row["evaluated_frame_count"], 0)
            self.assertEqual(path.read_bytes(), before)
            # A valid header alone must not excuse truncated coordinate data.
            path.write_bytes(before[:-5])
            with self.assertRaises(CoordinateReadError):
                scan_segment(path, processor=processor)

    def test_segmented_pool_matches_continuous_reader_oracle_without_changing_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_path = _write_project(root, True)
            project = json.loads(project_path.read_text())
            definition = project["definitions"]["common_pca"]
            definition["frame_selection"] = {"mode": "integer_stride_per_replica_v1", "stride": 3}
            definition["projection_frame_selection"] = {"mode": "fixed_stride_v1"}
            definition["projection_frame_stride"] = 1
            project_path.write_text(json.dumps(project))
            manifest_path = root / "system.json"
            manifest = json.loads(manifest_path.read_text())
            for system in manifest["systems"]:
                segments = []
                for n, count in enumerate((2, 1, 5)):
                    name = f"{system['system_id']}-{n}.xyz"
                    (root / name).write_text("".join(_frame(i + n, f"{n}-{i}") for i in range(count)))
                    segments.append({"segment_id": f"s{n}", "trajectory": name,
                                     "continuous_with_previous": False,
                                     "timing": {"first_frame_time": n * 100, "frame_interval": 2, "unit": "ps"}})
                system["replicas"][0]["segments"] = segments
            manifest_path.write_text(json.dumps(manifest))
            originals = {p: p.read_bytes() for p in root.iterdir() if p.is_file()}
            frozen, summary = plan_frame_selection(manifest, manifest_path, "angstrom",
                                                   definition["frame_selection"], frame_stride=1)
            frozen_copy = copy.deepcopy(frozen)
            self.assertEqual(frozen[("short", "r1", "s1")], set())
            self.assertEqual(frozen[("short", "r1", "s2")], {0})
            actual = common_pca_project(project_path)
            # Old full-reader path already handles empty selections correctly.
            # No periodic cells in this oracle; use its continuous reader while
            # retaining exactly the same fit/projection selectors and topology.
            def full_reader(project, replica, system_path, atom_count):
                return PeriodicFrameProcessor("unwrap_continuous", atom_count)
            with patch("salsbury_md_analysis.pca.PeriodicFrameProcessor.from_replica", side_effect=full_reader):
                oracle = common_pca_project(project_path)
            self.assertEqual(actual["basis"], oracle["basis"])
            self.assertEqual(actual["basis"]["evaluated_frame_count"], summary["selected_frame_count"])
            for a_system, b_system in zip(actual["systems"], oracle["systems"]):
                a_rows = a_system["replicas"][0]["segments"]
                b_rows = b_system["replicas"][0]["segments"]
                self.assertEqual([r["projections"] for r in a_rows], [r["projections"] for r in b_rows])
                self.assertEqual([r["frame_axis"] for r in a_rows], [r["frame_axis"] for r in b_rows])
                self.assertEqual(sum(r["evaluated_frame_count"] for r in a_rows), 8)
            self.assertEqual(frozen, frozen_copy)
            self.assertTrue(all(p.read_bytes() == data for p, data in originals.items()))

    def test_whole_replica_without_basis_observations_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_project(Path(tmp), True)
            project = json.loads(path.read_text())
            project["definitions"]["common_pca"]["frame_selection"] = {
                "mode": "integer_stride_per_replica_v1", "stride": 100,
            }
            path.write_text(json.dumps(project))
            with self.assertRaisesRegex(PCAAnalysisError, "no evaluated basis frames"):
                common_pca_project(path)

    def test_continuous_unwrap_still_processes_unselected_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.dcd"
            writer = _DCDWriter(path, atom_count=1, frame_count=2, starting_step=0,
                                save_interval=1, unit_cell_present=True)
            cell = ((10., 0., 0.), (0., 10., 0.), (0., 0., 10.))
            for index, x in enumerate((9., 1.)):
                writer.write(CoordinateFrame(index, [(x, 0., 0.)], "angstrom", True, cell), [0])
            writer.close()
            processor = PeriodicFrameProcessor(
                "unwrap_continuous", 1, bonds=((0, 0),), settings={
                    "maximum_bond_length_angstrom": 3.,
                    "cycle_closure_tolerance_angstrom": .25,
                    "maximum_anchor_displacement_angstrom": 4.,
                })
            # Feed ordinary raw frames to exercise continuous reconstruction.
            from dataclasses import replace
            from salsbury_md_analysis.coordinates import iter_coordinate_frames
            frames = [replace(f, coordinate_representation="raw")
                      for f in iter_coordinate_frames(path, "angstrom")]
            with patch("salsbury_md_analysis.pca.iter_coordinate_frames", return_value=iter(frames)):
                row = scan_segment(path, processor=processor)
            self.assertEqual(row["observed_frame_count"], 2)
            self.assertEqual(row["evaluated_frame_count"], 0)
            self.assertAlmostEqual(processor.checkpoint_state()["previous_anchors"][0][0], 11.)
            processor.begin_segment(False)
            self.assertIsNone(processor.checkpoint_state()["previous_anchors"])


if __name__ == "__main__":
    unittest.main()
