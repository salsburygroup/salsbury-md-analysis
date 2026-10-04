"""Planning-only fixtures; no trajectories are analyzed or cluster jobs run."""
from contextlib import redirect_stderr
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from salsbury_md_analysis.analysis_config import AnalysisConfigError, default_analysis_config, load_analysis_config
from salsbury_md_analysis.planning_diagnostics import planning_diagnostics
from salsbury_md_analysis.planning_reuse import reuse_planning_work
from salsbury_md_analysis.planning_search import (
    minimum_then_refine, search_scope, record_schedule_work, RefinementBudgetExceeded,
)
from salsbury_md_analysis.resource_planning import PlanningSearchError, ResourcePlanningError, plan_campaign_resource_budget
from salsbury_md_analysis.scientific_sampling import scientific_sampling_profile, profile_contract
from salsbury_md_analysis.resource_schedule import schedule_resource_tasks
from salsbury_md_analysis.quickstart import prepare_standard_analysis_resource_fit
from tests.test_core_first_pruning import task, LIMITS
from tests.test_quickstart import _write_inputs
from tests.test_resource_schedule_performance import scheduler_fixture


class BoundedPlanningTests(unittest.TestCase):
    def test_rejected_full_minimum_never_enters_refinement(self):
        calls = []
        def planner(rows, **kwargs):
            calls.append(kwargs['_minimum_only'])
            return {'feasibility_status': 'infeasible'}
        result = minimum_then_refine(planner, [])
        self.assertEqual(result['feasibility_status'], 'infeasible')
        self.assertEqual(calls, [True])

    def test_refinement_budget_returns_unchanged_feasible_minimum(self):
        minimum = {'feasibility_status': 'feasible', 'tasks': [{'selected': [250, 250]}]}
        def planner(rows, **kwargs):
            if kwargs['_minimum_only']:
                record_schedule_work()  # feasibility is not charged to refinement
                return deepcopy(minimum)
            for _ in range(5):
                record_schedule_work()
            self.fail('work cap should have interrupted this search')
        with search_scope(2) as state:
            result = minimum_then_refine(planner, [])
            self.assertEqual(state['refinement_calls'], 2)
            self.assertEqual(state['minimum_calls'], 1)
        self.assertEqual(result['tasks'], minimum['tasks'])
        self.assertEqual(result['planning_refinement']['status'], 'validated_minimum_fallback')
        self.assertEqual(minimum, {'feasibility_status': 'feasible', 'tasks': [{'selected': [250, 250]}]})

    def test_zero_budget_and_nested_calls_share_limit(self):
        calls = []
        def planner(rows, **kwargs):
            calls.append(kwargs['_minimum_only'])
            return {'feasibility_status': 'feasible'}
        with search_scope(0):
            with search_scope(100):
                result = minimum_then_refine(planner, [])
        self.assertEqual(calls, [True])
        self.assertEqual(result['planning_refinement']['maximum_schedule_calls'], 0)

    def test_invalid_core_probe_requires_full_check_and_limit_means_unknown(self):
        def planner(rows, **kwargs):
            return {'feasibility_status': 'infeasible' if kwargs['_minimum_only'] else 'feasible'}
        with search_scope(10):
            result = minimum_then_refine(planner, [], verify_rejected_minimum=True)
        self.assertEqual(result['feasibility_status'], 'feasible')
        with search_scope(0), self.assertRaises(PlanningSearchError) as caught:
            minimum_then_refine(planner, [], verify_rejected_minimum=True)
        self.assertTrue(caught.exception.diagnostics['search_budget_exhausted'])

    def test_unexpected_errors_are_not_converted_to_fallback(self):
        def planner(rows, **kwargs):
            if not kwargs['_minimum_only']:
                raise TypeError('actual defect')
            return {'feasibility_status': 'feasible'}
        with self.assertRaisesRegex(TypeError, 'actual defect'):
            minimum_then_refine(planner, [])

    def test_real_native_planner_obeys_zero_limit_and_scientific_floor(self):
        row = task('qc', 'structural_integrity_qc', 0)
        row.update(minimum_frames_per_replica=250, cpu_seconds_per_physical_frame=.01,
                   planning_dependencies={'depends_on_bundle_ids': [], 'wait_for_bundle_ids': []})
        before = deepcopy(row)
        with search_scope(0):
            result = minimum_then_refine(plan_campaign_resource_budget, [row], **LIMITS)
        self.assertEqual(row, before)
        self.assertEqual(result['tasks'][0]['selected_physical_frames_per_replica'], [250])
        self.assertEqual(result['tasks'][0]['integer_stride'], 4)
        self.assertEqual(result['feasibility_status'], 'feasible')

    def test_integer_rounding_preserves_stronger_task_floor(self):
        row = task('qc', 'structural_integrity_qc', 0)
        row.update(minimum_frames_per_replica=301,
                   scientific_sampling_requirements=profile_contract(scientific_sampling_profile('structural_integrity_qc')),
                   scientific_minimum_frames_per_replica=250)
        result = plan_campaign_resource_budget([row], _minimum_only=True, **LIMITS)
        self.assertEqual(result['tasks'][0]['selected_physical_frames_per_replica'], [333])
        self.assertEqual(result['tasks'][0]['integer_stride'], 3)
        row['required_integer_stride'] = 4
        with self.assertRaisesRegex(ResourcePlanningError, 'violates a scientific sampling floor'):
            plan_campaign_resource_budget([row], _minimum_only=True, **LIMITS)

    def test_nonzero_bound_stops_real_native_refinement(self):
        row = task('qc', 'structural_integrity_qc', 0)
        row.update(minimum_frames_per_replica=250, cpu_seconds_per_physical_frame=.01,
                   planning_dependencies={'depends_on_bundle_ids': [], 'wait_for_bundle_ids': []})
        with search_scope(1) as state:
            result = minimum_then_refine(plan_campaign_resource_budget, [row], **LIMITS)
            self.assertEqual(state['refinement_calls'], 1)
        self.assertEqual(result['planning_refinement']['status'], 'validated_minimum_fallback')
        self.assertEqual(result['tasks'][0]['selected_physical_frames_per_replica'], [250])

    def test_stronger_shared_projection_floor_disappears_with_optional_consumer(self):
        parent = task('pca', 'common_pca', 0)
        parent.update(minimum_frames_per_replica=250, balance_group='view',
                      scientific_sampling_requirements=profile_contract(scientific_sampling_profile('common_pca')))
        child = deepcopy(parent)
        child.update(task_id='grouped', module_id='grouped_ml', minimum_frames_per_replica=301)
        full = plan_campaign_resource_budget([parent, child], _minimum_only=True, **LIMITS)
        self.assertTrue(all(t['selected_physical_frames_per_replica'] == [333] for t in full['tasks']))
        core = plan_campaign_resource_budget([parent], _minimum_only=True, **LIMITS)
        self.assertEqual(core['tasks'][0]['selected_physical_frames_per_replica'], [250])
        self.assertEqual(parent['minimum_frames_per_replica'], 250)

    def test_schedule_reuse_preserves_complete_layout_and_ownership(self):
        rows, kwargs = scheduler_fixture(1)
        oracle = schedule_resource_tasks(rows, **kwargs)
        @reuse_planning_work
        def execute():
            from salsbury_md_analysis.planning_reuse import _active
            a = schedule_resource_tasks(rows, **kwargs)
            a[0]['planned_wall_hours'] = 99999
            b = schedule_resource_tasks(rows, **kwargs)
            self.assertEqual(b, oracle)
            self.assertEqual(_active.get()['schedule_hits'], 1)
            self.assertEqual(_active.get()['schedule_misses'], 1)
            altered = deepcopy(rows)
            altered[0]['planned_wall_hours'] += 1
            schedule_resource_tasks(altered, **kwargs)
            self.assertEqual(_active.get()['schedule_misses'], 2)
        execute()
        execute()  # distinct invocations never share cached schedules

    def test_native_preparation_replays_selected_minimum_without_refinement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdb, psf, trajectories = _write_inputs(root)
            sources = [pdb, psf, *trajectories]
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            before = [digest(p) for p in sources]
            config = root / 'config.json'
            config.write_text(json.dumps({'config_schema': 'salsbury-analysis-config-v1',
                'planning': {'maximum_refinement_schedule_calls': 0}}))
            with redirect_stderr(io.StringIO()), planning_diagnostics(root/'out', command='test'):
                report = prepare_standard_analysis_resource_fit(pdb_path=pdb, psf_path=psf,
                    trajectories=trajectories, output_directory=root/'out', project_id='bounded',
                    frame_interval_ps=10.0, config_path=config)
            fit = report['resource_fit']
            selected = json.loads((root/'out/resource-fit-evidence'/f"{fit['selected_validated_trial']}.json").read_text())['plan']
            final = json.loads((root/'out/campaign-resource-plan.json').read_text())
            fields = lambda plan: {t['task_id']: (t['integer_stride'],t['selected_physical_frames_per_replica']) for t in plan['tasks']}
            self.assertEqual(fields(final), fields(selected))
            self.assertTrue(final['internal_validated_sampling_replay'])
            self.assertEqual(final['planning_refinement']['status'], 'validated_minimum_fallback')
            self.assertEqual(final['native_schedule_validation']['status'], 'complete')
            self.assertFalse(final['scientific_sampling_feasibility']['below_standard_tasks'])
            self.assertEqual(before, [digest(p) for p in sources])
            human = (root/'out/planning-report.md').read_text()
            self.assertIn('validated_minimum_fallback', human)
            self.assertIn('maximum information or optimality has not been established', human)
            report_json = json.loads((root/'out/planning-report.json').read_text())
            self.assertEqual(report_json['planning_refinement'], final['planning_refinement'])

    def test_schedule_cache_invalidates_dependencies_and_resource_policies(self):
        rows, kwargs = scheduler_fixture(0)
        @reuse_planning_work
        def execute():
            from salsbury_md_analysis.planning_reuse import _active
            schedule_resource_tasks(rows, **kwargs)
            for change in ('memory', 'cpu', 'reserve', 'dependencies'):
                altered_rows, altered_kwargs = deepcopy(rows), deepcopy(kwargs)
                if change == 'memory':
                    altered_kwargs['maximum_memory'] += 1
                elif change == 'cpu':
                    altered_kwargs['maximum_cpus'] += 1
                elif change == 'reserve':
                    altered_kwargs['node_policy']['memory_reserve_gib'] = .5
                else:
                    altered_rows[-1]['depends_on_task_ids'] = ['0']
                actual = schedule_resource_tasks(altered_rows, **altered_kwargs)
                expected = schedule_resource_tasks.__wrapped__(altered_rows, **altered_kwargs)
                self.assertEqual(actual, expected)
            self.assertEqual(_active.get()['schedule_misses'], 5)
            self.assertEqual(_active.get()['schedule_hits'], 0)
        execute()

    def test_exact_schedule_hits_do_not_exhaust_refinement_budget(self):
        rows, kwargs = scheduler_fixture(0)
        @reuse_planning_work
        def execute():
            with search_scope(1) as state:
                state['phase'] = 'refinement'
                original = schedule_resource_tasks(rows, **kwargs)
                for _ in range(5):
                    self.assertEqual(schedule_resource_tasks(rows, **kwargs), original)
                self.assertEqual(state['refinement_calls'], 1)
                altered = deepcopy(rows)
                altered[0]['planned_wall_hours'] += .1
                with self.assertRaises(RefinementBudgetExceeded):
                    schedule_resource_tasks(altered, **kwargs)
        execute()

    def test_config_rejects_noninteger_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            for limit in (-1, 1.5, True, None, '100'):
                path.write_text(json.dumps({'config_schema':'salsbury-analysis-config-v1',
                    'planning': {'maximum_refinement_schedule_calls': limit}}))
                with self.assertRaisesRegex(AnalysisConfigError, 'maximum_refinement_schedule_calls'):
                    load_analysis_config(path, ['replica_rmsd_rg'], [])
