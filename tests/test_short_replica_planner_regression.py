import unittest
from salsbury_md_analysis.resource_planning import plan_campaign_resource_budget, ResourcePlanningError
from salsbury_md_analysis.scientific_sampling import profile_contract, scientific_sampling_profile

class ShortReplicaPlanningTests(unittest.TestCase):
    def task(self):
        return dict(task_id='short',module_id='replica_rmsd_rg',dependency_stage=0,
            effective_cpu_cap=1,source_frames_per_replica=[47,10000],
            minimum_frames_per_replica=100,maximum_frames_per_replica=10000,
            cpu_seconds_per_physical_frame=.1,fixed_cpu_hours=0.,estimated_peak_memory_gib=1.,
            priority_weight=1.,replica_sampling_mode='balanced_pooled',
            scientific_sampling_requirements=profile_contract(scientific_sampling_profile('replica_rmsd_rg')))
    def test_short_nonempty_replica_rejects_trial_zero_then_uses_all(self):
        plan=plan_campaign_resource_budget([self.task()],maximum_parallel_cpus=2,
            maximum_wall_hours=24,maximum_memory_gib=10,planning_utilization=1.,pilot_budget_fraction=0.)
        self.assertEqual(plan['feasibility_status'],'feasible')
        self.assertEqual(plan['tasks'][0]['integer_stride'],1)
        self.assertEqual(plan['tasks'][0]['selected_physical_frames_per_replica'],[47,10000])
    def test_explicit_zero_coverage_stride_is_rejected(self):
        task=self.task();task['required_integer_stride']=100
        with self.assertRaises(ResourcePlanningError):
            plan_campaign_resource_budget([task],maximum_parallel_cpus=2,
                maximum_wall_hours=24,maximum_memory_gib=10,planning_utilization=1.,pilot_budget_fraction=0.)

if __name__=='__main__': unittest.main()
