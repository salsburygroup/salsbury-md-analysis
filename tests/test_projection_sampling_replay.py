"""Projection-stream replay regressions; synthetic planning, never science."""
from contextlib import redirect_stderr
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from salsbury_md_analysis.campaign_planning import _view_tasks
from salsbury_md_analysis.comparative_quickstart import prepare_comparative_analysis_resource_fit
from salsbury_md_analysis.comparative_quickstart import prepare_comparative_analysis
from salsbury_md_analysis.resource_fit_preparation import prepare_core_first_resource_fit
from salsbury_md_analysis.quickstart import QuickstartError
from salsbury_md_analysis.fixed_sampling import FixedSamplingError, freeze_task_sampling
from salsbury_md_analysis.frame_sampling import integer_stride_selected_count
from salsbury_md_analysis.imwkmeans_resources import REFERENCE_SETTINGS
from tests.test_quickstart import _write_ion_inputs, _write_dcd


def view_project(projection_stride, members=1):
    pca = {'maximum_features': 12, 'frame_stride': 1,
           'frame_selection': {'mode': 'integer_stride_per_replica_v1', 'stride': 500},
           'projection_frame_stride': 1,
           'projection_frame_selection': {'mode': 'integer_stride_per_replica_v1', 'stride': projection_stride}}
    if members > 1:
        pca['symmetry_expansion'] = {'member_count': members}
    return {'requested_modules': ['common_pca', 'clustering_imwkmeans'],
            'definitions': {'common_pca': pca, 'clustering_imwkmeans': {**deepcopy(REFERENCE_SETTINGS),
                'k_values': [2, 3], 'minkowski_p_values': [1.5, 2.0],
                'initialization_ranks': [0], 'maximum_iterations': 100, 'fit_stride': 8}}}


class ProjectionSamplingReplayTests(unittest.TestCase):
    def test_fit_tasks_use_projection_stream_for_shared_system_and_member_views(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, members, counts in (
                ('global_common_heavy', 1, [100_000]*12),
                ('system_A__global_common_heavy', 1, [100_000]*6),
                ('oligomer_member_common_heavy', 2, [100_000]*12),
            ):
                for cache_stride, projection_stride in ((1, 50), (2, 25), (1, 1)):
                    with self.subTest(view=name, cache=cache_stride, projection=projection_stride):
                        source = [integer_stride_selected_count(n, cache_stride) for n in counts]
                        expected = [integer_stride_selected_count(n, projection_stride) for n in source]
                        path = Path(directory)/f'project-{name}.json'
                        path.write_text(json.dumps(view_project(projection_stride, members)))
                        rows = _view_tasks(path, source, 1000, time_safety_factor=1.5,
                            frame_intervals_ns_per_replica=[.01*cache_stride]*len(counts),
                            source_time_spans_ns_per_replica=[999.99]*len(counts))
                        fit = next(row for row in rows if row['module_id']=='clustering_imwkmeans')
                        self.assertEqual(fit['source_frames_per_replica'], expected)
                        self.assertEqual(fit['maximum_frames_per_replica'], max(expected))
                        self.assertEqual(fit['member_observation_multiplier'], members)
                        self.assertEqual(fit['scientific_minimum_frames_per_replica'], 250)
                        self.assertEqual(fit['frame_intervals_ns_per_replica'], [.01*cache_stride*projection_stride]*len(counts))
                        # Iteration starts from raw resource counts; fixed
                        # replay starts from cached counts. Both must produce
                        # the identical derived fit stream and PCA basis cost.
                        iterated = _view_tasks(path, counts, 1000, time_safety_factor=1.5,
                            selector_source_integer_stride=cache_stride,
                            frame_intervals_ns_per_replica=[.01]*len(counts),
                            source_time_spans_ns_per_replica=[999.99]*len(counts))
                        fit_iteration = next(row for row in iterated if row['module_id']=='clustering_imwkmeans')
                        for key in ('source_frames_per_replica', 'maximum_frames_per_replica', 'frame_intervals_ns_per_replica'):
                            self.assertEqual(fit_iteration[key], fit[key])
                        self.assertEqual(iterated[0]['basis_selected_physical_frames_per_replica'],
                                         rows[0]['basis_selected_physical_frames_per_replica'])
                        self.assertEqual(iterated[0]['fixed_cpu_hours'], rows[0]['fixed_cpu_hours'])
                        frozen = deepcopy(fit)
                        frozen.update(integer_stride=8,
                            selected_physical_frames_per_replica=[integer_stride_selected_count(n,8) for n in expected])
                        replayed = freeze_task_sampling([fit], {'tasks': {fit['task_id']: frozen}})[0]
                        self.assertEqual(replayed['required_integer_stride'], 8)
                        self.assertEqual(replayed['selected_physical_frames_per_replica'], frozen['selected_physical_frames_per_replica'])
                        altered = deepcopy(fit)
                        altered['source_frames_per_replica'][0] -= 1
                        with self.assertRaisesRegex(FixedSamplingError, 'source stream differs'):
                            freeze_task_sampling([altered], {'tasks': {fit['task_id']: frozen}})

    @patch('salsbury_md_analysis.comparative_quickstart._discover_dssp_executable', return_value=None)
    def test_native_bounded_fallback_replays_strided_projections(self, _dssp):
        for cache_stride in (1, 2):
            with self.subTest(cache_stride=cache_stride):
                self._native_replay(cache_stride)

    @patch('salsbury_md_analysis.comparative_quickstart._discover_dssp_executable', return_value=None)
    def test_failed_materialization_keeps_exact_trial_documents(self, _dssp):
        self._native_replay(2, fail_materialization=True)

    def _native_replay(self, cache_stride, fail_materialization=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdb, psf, trajectories = _write_ion_inputs(root)
            for trajectory in trajectories:
                _write_dcd(trajectory, 9, 10_000)
            sources = [pdb, psf, *trajectories]
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            before = [digest(p) for p in sources]
            request = root/'request.json'
            request.write_text(json.dumps({'request_schema':'salsbury-comparative-analysis-input-v1',
                'systems':[{'system_id':name, 'pdb':str(pdb), 'psf':str(psf),
                            'trajectories':[str(p) for p in trajectories], 'frame_interval_ps':10}
                           for name in ('A','B')]}))
            config = root/'config.json'
            config.write_text(json.dumps({'config_schema':'salsbury-analysis-config-v1',
                'planning':{'maximum_refinement_schedule_calls':0},
                'execution':{'maximum_parallel_cpus':44, 'maximum_memory_gib':185,
                    'overall_stride_candidates':[cache_stride], 'coordinate_cache_materialization':'planned_strided'}}))
            documents = {}
            def prepare(**kwargs):
                if '_fixed_sampling_replay' in kwargs:
                    trial = kwargs['_fixed_sampling_replay'].parent
                    for path in trial.iterdir():
                        if path.name in ('fixed-sampling-schedule.json', 'sampling-plan.json') or path.name.startswith(('project', 'system')) and path.suffix=='.json':
                            documents[path.name] = path.read_bytes()
                    raise QuickstartError('injected final materialization failure')
                return prepare_comparative_analysis(**kwargs)
            if fail_materialization:
                with self.assertRaisesRegex(QuickstartError, 'injected final'), redirect_stderr(io.StringIO()):
                    prepare_core_first_resource_fit(prepare=prepare,
                        common=dict(request_path=request, project_id='failure-evidence'),
                        destination=root/'out', target_wall_hours=168, config_path=config)
                self.assertIn('fixed-sampling-schedule.json', documents)
                self.assertIn('sampling-plan.json', documents)
                self.assertIn('project-global_common_heavy.json', documents)
                self.assertIn('system.json', documents)
                for evidence_root in (root/'out.resource-fit-evidence', root/'out/resource-fit-evidence'):
                    receipt = json.loads((evidence_root/'requested.json').read_text())
                    self.assertEqual(receipt['status'], 'validated')
                    hashes = receipt['preparation_documents']['sha256']
                    for name, data in documents.items():
                        saved = evidence_root/'requested'/name
                        self.assertEqual(saved.read_bytes(), data)
                        self.assertEqual(hashes[f'requested/{name}'], hashlib.sha256(data).hexdigest())
                self.assertFalse((root/'out/submit.sh').exists())
                self.assertEqual(before, [digest(p) for p in sources])
                return
            with redirect_stderr(io.StringIO()):
                report = prepare_comparative_analysis_resource_fit(request_path=request,
                    output_directory=root/'out', project_id='strided-replay', config_path=config,
                    target_wall_hours=168)
            fit_report = report['resource_fit']
            self.assertEqual(fit_report['selected_validated_trial'], 'requested')
            trial = json.loads((root/'out/resource-fit-evidence/requested.json').read_text())['plan']
            final = json.loads((root/'out/campaign-resource-plan.json').read_text())
            fields = lambda p: {r['task_id']:(r['source_frames_per_replica'], r['integer_stride'],
                r['selected_physical_frames_per_replica']) for r in p['tasks']}
            self.assertEqual(fields(final), fields(trial))
            self.assertEqual(final['global_stride_coupling']['selected_coordinate_cache_integer_stride'], cache_stride)
            self.assertEqual(final['native_schedule_validation']['status'], 'complete')
            self.assertEqual(final['planning_refinement']['status'], 'validated_minimum_fallback')
            self.assertFalse(final['scientific_sampling_feasibility']['below_standard_tasks'])
            fits = [r for r in final['tasks'] if r['module_id']=='clustering_imwkmeans']
            self.assertTrue(fits)
            parents = {r['task_id']:r for r in final['tasks'] if r['module_id']=='common_pca'}
            self.assertTrue(any(parents[f"view:{r['workflow_id']}:common_pca"]['integer_stride']>1 for r in fits))
            self.assertTrue(any(r['integer_stride']>1 for r in fits))
            for fit in fits:
                self.assertEqual(fit['source_frames_per_replica'],
                    parents[f"view:{fit['workflow_id']}:common_pca"]['selected_physical_frames_per_replica'])
            for parent in parents.values():
                project = json.loads((root/'out'/f"project-{parent['workflow_id']}.json").read_text())
                stride = project['definitions']['common_pca']['frame_selection'].get('stride', 1)
                self.assertEqual(parent['basis_selected_physical_frames_per_replica'],
                    [integer_stride_selected_count(n, stride) for n in parent['source_frames_per_replica']])
            receipt = json.loads((root/'out/resource-fit-evidence/requested.json').read_text())
            for relative, sha in receipt['preparation_documents']['sha256'].items():
                self.assertEqual(digest(root/'out/resource-fit-evidence'/relative), sha)
                self.assertIn(f'resource-fit-evidence/{relative}', report['generated_files'])
            self.assertEqual(before, [digest(p) for p in sources])
