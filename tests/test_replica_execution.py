import unittest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.replica_execution import (
    ReplicaExecutionError,
    ReplicaShard,
    execute_replica_map_reduce,
    execute_replica_workers,
)
from salsbury_md_analysis.replica_module_execution import (
    _distributed_worker_entry,
)


def _square(shard):
    return int(shard.payload) ** 2


def _sum_in_stable_order(partials):
    return sum(row.value for row in partials)


class ReplicaExecutionTests(unittest.TestCase):
    def test_distributed_worker_entry_preserves_rank_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "partials"
            output.mkdir()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "output_directory": str(output),
                "shards": [
                    {
                        "ordinal": index,
                        "system_id": "system",
                        "replica_id": f"replica-{index}",
                        "segment_ids": ["segment"],
                        "payload": {"value": index},
                    }
                    for index in range(2)
                ],
            }), encoding="utf-8")
            with patch.dict(os.environ, {
                "SLURM_PROCID": "0", "SLURM_NTASKS": "2"
            }), patch(
                "salsbury_md_analysis.replica_module_execution._module_worker",
                side_effect=lambda shard: {"ordinal": shard.ordinal},
            ):
                _distributed_worker_entry(manifest)
            first = json.loads(
                (output / "partial-00000.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first["replica_id"], "replica-0")
            self.assertEqual(first["value"], {"ordinal": 0})
            self.assertFalse((output / "partial-00001.json").exists())
    @staticmethod
    def _shards():
        return [
            ReplicaShard(0, "system-a", "replica-1", ("segment-1",), 2),
            ReplicaShard(1, "system-a", "replica-2", ("segment-1", "segment-2"), 3),
        ]

    def test_parallel_map_reduce_preserves_identity_and_order(self):
        result, evidence = execute_replica_map_reduce(
            self._shards(), _square, _sum_in_stable_order,
            maximum_workers=2, scheduler_cpu_limit=2,
        )
        self.assertEqual(result, 13)
        self.assertEqual(evidence.workers_used, 2)
        self.assertEqual(
            evidence.stable_reduction_order,
            (("system-a", "replica-1"), ("system-a", "replica-2")),
        )
        self.assertTrue(evidence.segment_boundaries_preserved)

    def test_scheduler_limit_caps_internal_workers(self):
        partials, evidence = execute_replica_workers(
            self._shards(), _square,
            maximum_workers=8, scheduler_cpu_limit=1,
        )
        self.assertEqual([row.value for row in partials], [4, 9])
        self.assertEqual(evidence.workers_used, 1)

    def test_duplicate_identity_fails_closed(self):
        shards = self._shards()
        shards[1] = ReplicaShard(
            1, "system-a", "replica-1", ("segment-2",), 3
        )
        with self.assertRaisesRegex(ReplicaExecutionError, "duplicate replica"):
            execute_replica_workers(
                shards, _square, maximum_workers=1, scheduler_cpu_limit=1
            )

    def test_segment_identity_is_required(self):
        with self.assertRaisesRegex(ReplicaExecutionError, "segment identity"):
            execute_replica_workers(
                [ReplicaShard(0, "system-a", "replica-1", (), 2)],
                _square, maximum_workers=1, scheduler_cpu_limit=1,
            )


if __name__ == "__main__":
    unittest.main()
