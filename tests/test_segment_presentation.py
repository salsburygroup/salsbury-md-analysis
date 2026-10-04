import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from salsbury_md_analysis.presentation_artifacts import generate_presentation_artifacts, PresentationArtifactError


class SegmentPresentationTests(unittest.TestCase):
    def test_hydrogen_bond_primary_chart_uses_complete_pooled_denominators(self):
        from test_segment_reporting import report
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.write(root, 'hydrogen-bond-discovery', dict(report(), technical_status='complete'))
            result = generate_presentation_artifacts(root)
            tables = [a for a in result['artifacts'] if a['artifact_type'] == 'table'
                      and a.get('primary_human_output')]
            self.assertEqual(len(tables), 1)
            with (root / 'presentation-artifacts' / tables[0]['relative_path']).open() as handle:
                rows = {r['system_id']: r for r in csv.DictReader(handle)}
            self.assertEqual({r['evaluated_frame_count'] for r in rows.values()}, {'100'})
            self.assertAlmostEqual(float(rows['A']['occupancy_fraction']), .06)
            self.assertAlmostEqual(float(rows['B']['occupancy_fraction']), .18)

    def write(self, root, rel, doc):
        path = root / 'results' / rel / 'report.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(doc))
        return path

    def tensor(self):
        return {'technical_status':'complete','module_id':'information_dynamics',
                'settings':{'component_indices':[1,3], 'feature_source':'common_pca'},
                'analyses':{'coskewness':{'feature_count':2,'observation_count':500,
                    'coskewness':[[[1.,-.2],[-.2,.7]],[[-.2,.7],[.7,-1.5]]]}}}

    def test_every_tensor_cell_preserved_and_scopes_do_not_collide(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); before={}
            for system in ['A','B']:
                for view in ['heavy','trace']:
                    p=self.write(root, f'per-system/{system}/conformational-views/{view}/information-dynamics', self.tensor())
                    before[p]=hashlib.sha256(p.read_bytes()).hexdigest()
            report=generate_presentation_artifacts(root)
            self.assertEqual(report['unadapted_report_count'],0)
            self.assertEqual(report['artifact_count'],16)
            paths=[a['relative_path'] for a in report['artifacts']]
            self.assertEqual(len(paths),len(set(paths)))
            for path in paths:
                p=root/'presentation-artifacts'/path
                if p.suffix=='.csv':
                    with p.open() as f: rows=list(csv.DictReader(f))
                    self.assertEqual(len(rows),4)
                    fixed=int(rows[0]['fixed_feature'].split()[1]);k=[1,3].index(fixed)
                    self.assertEqual([float(x['coskewness']) for x in rows],
                                     [v for r in self.tensor()['analyses']['coskewness']['coskewness'][k] for v in r])
                if p.suffix=='.svg':
                    ET.parse(p)
                    self.assertIn('Shared scale',p.read_text())
                    self.assertNotIn('Atom or residue',p.read_text())
            self.assertTrue(all(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p,digest in before.items()))

    def test_not_estimable_metrics_preserve_abstention(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for system in ['A','B']:
                self.write(root,f'per-system/{system}/chemical-context/nucleic-acid-geometry',{
                    'technical_status':'complete','module_id':'nucleic_acid_geometry',
                    'distribution_reports':[{'metric_id':'ring:one:plane_rms_angstrom',
                      'status':'not_estimable','reason':'scalar distribution contains an empty trajectory segment'}]})
            result=generate_presentation_artifacts(root)
            self.assertEqual(result['unadapted_report_count'],0)
            self.assertTrue(all(r['presentation_adapter']=='unavailable_with_explanation' for r in result['reviewed_reports']))
            self.assertEqual(len(set(a['relative_path'] for a in result['artifacts'])),4)
            for a in result['artifacts']:
                if a['artifact_type']=='table':
                    text=(root/'presentation-artifacts'/a['relative_path']).read_text()
                    self.assertIn('not_estimable',text)
                    self.assertIn('empty trajectory segment',text)
                    self.assertNotIn('fraction',text)

    def test_invalid_tensor_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);doc=self.tensor();doc['analyses']['coskewness']['coskewness'][0][0]=[float('nan'),0]
            self.write(root,'information-dynamics',doc)
            with self.assertRaises(PresentationArtifactError):generate_presentation_artifacts(root)

    def test_matrix_figures_distinguish_scope_system_and_view(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for scope in ['conformational-views', 'per-system/A/conformational-views']:
                for view in ['heavy','trace']:
                    self.write(root,f'{scope}/{view}/information-correlation',{
                        'technical_status':'complete','module_id':'generalized_correlation_and_information',
                        'systems':[{'system_id':'A','generalized_correlation':[[1.,.2],[.2,1.]]}]})
            result=generate_presentation_artifacts(root)
            paths=[a['relative_path'] for a in result['artifacts']]
            self.assertEqual(len(paths),8)
            self.assertEqual(len(paths),len(set(paths)))

    def test_ion_distribution_paths_preserve_system(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for system,value in [('A',.2),('B',.8)]:
                self.write(root,f'per-system/{system}/chemical-context/ion-geometry',{
                    'technical_status':'complete','module_id':'ion_coordination_geometry',
                    'distribution_reports':[{'metric_id':'same_metric','status':'complete',
                        'histogram':[{'center':1.,'fraction':value},{'center':2.,'fraction':1-value}]}]})
            result=generate_presentation_artifacts(root)
            paths=[a['relative_path'] for a in result['artifacts']]
            self.assertEqual(len(paths),4)
            self.assertEqual(len(paths),len(set(paths)))


if __name__=='__main__':unittest.main()
