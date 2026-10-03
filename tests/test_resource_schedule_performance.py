"""Compression must reproduce the dense token policy, not approximate it."""
import copy
import hashlib
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from salsbury_md_analysis import resource_schedule as rs
from salsbury_md_analysis.resource_planning import plan_campaign_resource_budget


class DenseTokenPool:
    """Independent, intentionally slow specification of the original policy."""

    def __init__(self, count):
        self.tokens = [(0.0, None)] * count

    def __len__(self):
        return len(self.tokens)

    def select(self, count):
        indices = sorted(range(len(self.tokens)),
                         key=lambda i: (self.tokens[i][0], i))[:count]
        return rs._TokenSelection(
            tuple((i, i + 1) for i in indices), len(indices),
            max([self.tokens[i][0] for i in indices] + [0.0]),
            frozenset(self.tokens[i][1] for i in indices if self.tokens[i][1] is not None))

    def assign(self, selection, finish, owner):
        for lo, hi in selection.ranges:
            for index in range(lo, hi):
                self.tokens[index] = (finish, owner)


def scheduler_fixture(seed):
    rng = random.Random(seed)
    node_count = 1 + seed % 3
    rows = []
    for i in range(18):
        nodes = rng.randint(1, node_count)
        per_node = rng.randint(1, 4)
        cpus = (nodes - 1) * per_node + rng.randint(1, per_node)
        mem = rng.choice([0.25, 0.251, 1, 3, 4, 7])
        deps = [str(j) for j in range(i) if rng.random() < 0.10]
        waits = [str(j) for j in range(i) if rng.random() < 0.06]
        hours = rng.choice([0, 0.1, 0.25, 1, 2])
        rows.append(dict(item_id=str(i), submission_index=i, cpu_slots=cpus,
            memory_gib=nodes * mem, requested_memory_gib=mem, node_count=nodes,
            workers_per_node=per_node, planned_wall_hours=hours, wall_hours=hours + 1,
            depends_on_task_ids=deps, wait_for_task_ids=waits,
            schedule_order_key=f'key-{18-i}'))
    return rows, dict(maximum_cpus=4 * node_count, maximum_memory=8 * node_count,
        node_policy=dict(cpus_per_node=4, memory_gib_per_node=8,
                         maximum_nodes_per_campaign=node_count, memory_reserve_gib=1))


class ResourceSchedulePerformanceTests(unittest.TestCase):
    def test_token_ranges_match_dense_selection_and_ownership(self):
        for size in [1, 7, 64, 901]:
            rng = random.Random(size)
            pool, dense = rs._TokenPool(size), DenseTokenPool(size)
            for step in range(100):
                count = rng.randint(0, size + 2)
                actual, expected = pool.select(count), dense.select(count)
                indices = [i for lo, hi in actual.ranges for i in range(lo, hi)]
                self.assertEqual(indices, [i for lo, hi in expected.ranges for i in range(lo, hi)])
                self.assertEqual(actual[1:], expected[1:])
                self.assertIs(pool.select(count), actual)
                finish = actual.ready + rng.choice([0, 0.1, 1, 2])
                owner = f'task-{step}'
                pool.assign(actual, finish, owner)
                dense.assign(expected, finish, owner)
                expanded = [(available, previous) for lo, hi, available, previous in pool.runs
                            for _ in range(lo, hi)]
                self.assertEqual(expanded, dense.tokens)

    def test_complete_schedules_equal_dense_oracle(self):
        # Frozen full outputs also guard the greedy ready-group optimization,
        # independently of the dense-vs-compressed token implementation below.
        golden = json.loads((Path(__file__).parent / 'fixtures/resource_schedule_pr137_hashes.json').read_text())
        for seed in range(80):
            rows, kwargs = scheduler_fixture(seed)
            original = copy.deepcopy(rows)
            actual = rs.schedule_resource_tasks(rows, **kwargs)
            with patch.object(rs, '_TokenPool', DenseTokenPool):
                expected = rs.schedule_resource_tasks(rows, **kwargs)
            self.assertEqual(actual, expected, seed)
            digest = hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            self.assertEqual(digest, golden['hashes'][str(seed)], seed)
            self.assertEqual(rows, original)

    def test_aggregate_only_and_resource_failures_equal_dense(self):
        rows, _ = scheduler_fixture(3)
        for cpus, memory in [(8, 16), (1, 1), (4, 0.25)]:
            kwargs = dict(maximum_cpus=cpus, maximum_memory=memory, node_policy={})
            outcomes = []
            for factory in (rs._TokenPool, DenseTokenPool):
                with patch.object(rs, '_TokenPool', factory):
                    try:
                        outcomes.append(rs.schedule_resource_tasks(rows, **kwargs))
                    except rs.ResourceScheduleError as exc:
                        outcomes.append(str(exc))
            self.assertEqual(*outcomes)

    def test_capacity_does_not_expand_into_individual_tokens(self):
        pool = rs._TokenPool(10**9)
        for i in range(20):
            selection = pool.select(12345 + i)
            pool.assign(selection, i + 1, str(i))
        self.assertLessEqual(len(pool.runs), 41)
        self.assertLessEqual(len(pool.select(10**9).ranges), 41)

    def test_distinct_aggregate_and_node_demands_are_not_aliased(self):
        rows, kwargs = scheduler_fixture(0)
        for row in rows:
            row['memory_gib'] = row['requested_memory_gib'] / 2
        actual = rs.schedule_resource_tasks(rows, **kwargs)
        with patch.object(rs, '_TokenPool', DenseTokenPool):
            expected = rs.schedule_resource_tasks(rows, **kwargs)
        self.assertEqual(actual, expected)

    def test_sampling_and_feasibility_match_dense_policy(self):
        tasks = [dict(task_id=str(i), execution_bundle_id=str(i), module_id='test',
            source_frames_per_replica=[1000, 1000], minimum_frames_per_replica=50,
            maximum_frames_per_replica=1000, effective_cpu_cap=1,
            estimated_peak_memory_gib=1 + i % 3, dependency_stage=i,
            cpu_seconds_per_physical_frame=0.5 + i,
            planning_dependencies=dict(depends_on_bundle_ids=([str(i-2)] if i>1 else []),
                                       wait_for_bundle_ids=[])) for i in range(8)]
        original = copy.deepcopy(tasks)
        for hours in [0.1, 0.5, 1, 4, 12]:
            kwargs = dict(maximum_parallel_cpus=4, maximum_wall_hours=hours,
                maximum_memory_gib=16, planning_utilization=1, pilot_budget_fraction=0,
                memory_safety_factor=1.5, memory_overhead_gib=1,
                maximum_cpus_per_node=4, maximum_memory_gib_per_node=16, maximum_nodes=1)
            actual = plan_campaign_resource_budget(tasks, **kwargs)
            with patch.object(rs, '_TokenPool', DenseTokenPool):
                expected = plan_campaign_resource_budget(tasks, **kwargs)
            self.assertEqual(actual, expected)
        self.assertEqual(tasks, original)


if __name__ == '__main__':
    unittest.main()
