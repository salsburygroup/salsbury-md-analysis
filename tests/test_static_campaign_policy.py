import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.scalar_distributions import analyze_scalar_distribution
from salsbury_md_analysis.water_mediated_hydrogen_bonds import _residence_runs
from salsbury_md_analysis.pca_fes import _population_tables
from salsbury_md_analysis.trajectory_contracts import normalize_segment_axis
from salsbury_md_analysis.coordinate_cache import build_coordinate_cache, validate_reusable_coordinate_cache, CoordinateCacheError
from salsbury_md_analysis.coordinates import iter_coordinate_frames
from test_coordinate_cache import write_dcd


class StaticCampaignPolicyTests(unittest.TestCase):
    def test_static_histogram_preserves_counts_without_residence(self):
        rows = [{'source_frame_index':i,'value':float(i % 3)} for i in range(12)]
        args = dict(binning_rule='explicit',padding_fraction=0.,bin_count=3)
        with patch.dict(os.environ, {'SALSBURY_STATIC_ENSEMBLE':'0'}):
            normal = analyze_scalar_distribution([({'replica_id':'r'},rows)],**args)
        with patch.dict(os.environ, {'SALSBURY_STATIC_ENSEMBLE':'1'}):
            static = analyze_scalar_distribution([({'replica_id':'r'},rows)],**args)
            reverse = analyze_scalar_distribution([({'replica_id':'r'},list(reversed(rows)))],**args)
        self.assertEqual(normal['histogram'], static['histogram'])
        self.assertEqual(static['histogram'], reverse['histogram'])
        self.assertIsNone(static['residence_runs'])
        self.assertEqual(static['residence_by_bin'], [])
        self.assertEqual(len(static['assignments']), 12)

    def test_static_water_and_fes_do_not_evaluate_time_order(self):
        rows=[dict(system_id='s',replica_id='r',segment_id='g',source_frame_index=i,basin_id=i%2+1) for i in range(4)]
        with patch.dict(os.environ, {'SALSBURY_STATIC_ENSEMBLE':'1'}):
            self.assertEqual(_residence_runs([{}]), ([], []))
            populations, blocks = _population_tables(rows,[1,2],2,True)
            self.assertEqual(len(populations),1)
            self.assertEqual(blocks,[])
            self.assertEqual(normalize_segment_axis({'timing':{'first_frame_time':10,'frame_interval':100,'unit':'ps'}},'ns')['kind'],'sample_index')

    def test_static_cache_is_independent_and_not_reused_as_continuous(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            pdb=root/'system.pdb'
            pdb.write_text('ATOM      1  C1  LIG A   1       9.500   0.000   0.000  1.00  0.00           C\nATOM      2  C2  LIG A   1       0.500   0.000   0.000  1.00  0.00           C\nHETATM    3  O   TIP W   2       5.000   0.000   0.000  1.00  0.00           O\nHETATM    4  H1  TIP W   2       5.500   0.000   0.000  1.00  0.00           H\nHETATM    5 MG   MG  M   3       8.000   0.000   0.000  1.00  0.00          MG\nEND\n')
            bonds=root/'bonds.json'
            bonds.write_text(json.dumps(dict(format='salsbury-bonds-v1',atom_count=5,index_base=0,bonds=[[0,1],[2,3]])))
            trajectory=root/'test.dcd';write_dcd(trajectory)
            manifest=root/'system.json'
            manifest.write_text(json.dumps({'systems':[{'system_id':'s','replicas':[{'replica_id':'r','topology':str(pdb),'connectivity':str(bonds),'segments':[{'segment_id':'g','trajectory':str(trajectory),'timing':{'first_frame_time':0.,'frame_interval':1.,'unit':'ps'}}]}]}]}))
            out=root/'cache'
            with patch.dict(os.environ, {'SALSBURY_STATIC_ENSEMBLE':'1'}):
                report=build_coordinate_cache(manifest,out)
                self.assertEqual(report['rows'][0]['periodic_reconstruction']['policy'],'make_whole')
                self.assertEqual(report['coordinate_representation'],'independent_make_whole_unaligned_strided')
                validate_reusable_coordinate_cache(out,manifest)
            with patch.dict(os.environ, {'SALSBURY_STATIC_ENSEMBLE':'0'}):
                with self.assertRaises(CoordinateCacheError):
                    validate_reusable_coordinate_cache(out,manifest)


if __name__ == '__main__':
    unittest.main()
