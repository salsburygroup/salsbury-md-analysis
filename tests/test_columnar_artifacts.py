import tempfile
import unittest
from pathlib import Path

import numpy as np

from salsbury_md_analysis.columnar_artifacts import (
    AtomicColumnarBundle,
    ColumnarArtifactError,
    iter_columnar_records,
    load_columnar_table,
)


class ColumnarArtifactTests(unittest.TestCase):
    def test_atomic_hash_bound_table_is_memory_mapped_and_streamable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            bundle = AtomicColumnarBundle(root)
            reference = bundle.write_table(
                "segment-00000/feature-0000",
                {
                    "source_frame_index": np.asarray([0, 10], dtype=np.int64),
                    "axis_value": np.asarray([0.0, 1.0], dtype=np.float64),
                    "values": np.asarray([[2.0], [3.0]], dtype=np.float64),
                },
                constants={"axis_kind": "time"},
                provenance={"module_id": "trajectory_features"},
            )
            self.assertFalse(root.exists())
            bundle.publish()
            table = load_columnar_table(reference)
            self.assertIsInstance(table["arrays"]["values"], np.memmap)
            rows = list(iter_columnar_records(reference))
            self.assertEqual(rows[0]["source_frame_index"], 0)
            self.assertEqual(rows[1]["values"], [3.0])
            self.assertEqual(rows[1]["axis_kind"], "time")

    def test_array_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            bundle = AtomicColumnarBundle(root)
            reference = bundle.write_table(
                "table",
                {"value": np.asarray([1.0, 2.0])},
                constants={},
                provenance={"module_id": "test"},
            )
            bundle.publish()
            array_path = next((root / "table").glob("*.npy"))
            with array_path.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ColumnarArtifactError, "hash mismatch"):
                load_columnar_table(reference)


if __name__ == "__main__":
    unittest.main()
