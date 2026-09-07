import unittest
from salsbury_md_analysis.resource_planning import plan_campaign_resource_budget


class CacheScanParallelPlanningTests(unittest.TestCase):
    def plan(self, counts, source, scan_fraction=1.0, serial=0.1, slots=2):
        total_worker_hours = sum(scan_fraction*s + (1-scan_fraction)*c for s,c in zip(source, counts))/3600
        task = dict(task_id='cache', module_id='coordinate_cache', dependency_stage=0,
            effective_cpu_cap=slots, parallel_execution_model='replica_worker_exact_global_reducer_v1',
            parallel_worker_count=2, estimated_peak_memory_gib_per_parallel_worker=1.,
            reducer_memory_gib=1., estimated_peak_memory_gib=2.,
            source_frames_per_replica=counts, minimum_frames_per_replica=max(counts),
            maximum_frames_per_replica=max(counts), replica_sampling_mode='independent_all_available',
            fixed_cpu_hours=serial+sum(source)*scan_fraction/3600,
            cpu_seconds_per_physical_frame=1-scan_fraction,
            coordinate_cache_original_fixed_cpu_hours=serial,
            coordinate_cache_full_scan_fraction=scan_fraction,
            coordinate_cache_raw_source_frames_per_replica=source)
        p=plan_campaign_resource_budget([task],maximum_parallel_cpus=slots,
            maximum_wall_hours=10., maximum_memory_gib=10.)
        return p['tasks'][0], total_worker_hours+serial

    def test_full_scan_parallelizes_even_when_all_cpu_cost_is_stride_fixed(self):
        row,cpu=self.plan([100,100],[10000,10000])
        self.assertAlmostEqual(row['estimated_cpu_hours'],cpu)
        self.assertAlmostEqual(row['estimated_wall_hours_at_effective_cpu_cap'],0.1+10000/3600)

    def test_unequal_source_lengths_use_raw_scan_weights(self):
        row,cpu=self.plan([100,100],[10000,1000])
        self.assertAlmostEqual(row['estimated_cpu_hours'],cpu)
        self.assertAlmostEqual(row['estimated_wall_hours_at_effective_cpu_cap'],0.1+10000/3600)

    def test_single_worker_and_mixed_scan_write_work(self):
        row,cpu=self.plan([100,200],[10000,1000],scan_fraction=.5,slots=1)
        self.assertAlmostEqual(row['estimated_wall_hours_at_effective_cpu_cap'],cpu)


if __name__=='__main__': unittest.main()
