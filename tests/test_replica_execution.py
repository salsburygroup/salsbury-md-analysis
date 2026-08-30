import unittest

from salsbury_md_analysis.replica_execution import (
    ReplicaExecutionError,
    ReplicaShard,
    execute_replica_map_reduce,
    execute_replica_workers,
)


def _square(shard):
    return int(shard.payload) ** 2


def _sum_in_stable_order(partials):
    return sum(row.value for row in partials)


class ReplicaExecutionTests(unittest.TestCase):
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
