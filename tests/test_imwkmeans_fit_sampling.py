import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from salsbury_md_analysis.clustering import (assign_imwkmeans, clustering_imwkmeans_project,
    _minkowski_distance_power, _imwkmeans_settings, ClusteringAnalysisError)


class ImwkmeansFitSamplingTests(unittest.TestCase):
    def project(self):
        return {"definitions":{"clustering_imwkmeans":{
            "feature_source":"common_pca","component_indices":[1,2],"standardize_features":True,
            "k_values":[2],"minkowski_p_values":[1.5,2.0],"initialization_ranks":[0,1],
            "maximum_iterations":100,"objective_tolerance":1e-10,"minimum_cluster_size":2,
            "weight_dispersion_floor":1e-12,"maximum_silhouette_observations":100,
            "fit_stride":2,"assignment_chunk_size":3}}}

    def test_weighted_assignment_matches_scalar_objective_and_chunk_boundaries(self):
        vectors = [[i/10, (i%3)/5] for i in range(41)]
        centers, weights = [[0,0],[2,1]], [[.2,.8],[.6,.4]]
        for p in (1.5,2,3):
            expected = [min(range(2), key=lambda c: (_minkowski_distance_power(v,centers[c],weights[c],p), c)) for v in vectors]
            for chunk in (1,7,100):
                self.assertEqual(assign_imwkmeans(vectors,centers,weights,p,chunk), expected)

    def test_one_pooled_fit_all_observations_and_lineage_retained(self):
        points = [(-3+i*.02,-2+i*.01) for i in range(6)] + [(3+i*.02,2+i*.01) for i in range(6)]
        fake_pca = {"project_manifest_sha256":"a"*64,"system_manifest_path":"/tmp/system.json",
                    "system_manifest_sha256":"b"*64,"input_content_signature_sha256":"c"*64,"issues":[],
                    "systems":[{"system_id":"A","replicas":[{
                        "replica_id":f"r{replica}","segments":[{"segment_id":f"s{segment}","projections":[
                            {"source_frame_index":i,"sample_index":segment*6+i,"scores_angstrom":list(point)}
                            for i,point in enumerate(points[segment*6:(segment+1)*6])]} for segment in range(2)]}
                        for replica in range(2)]}]}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"project.json"
            path.write_text(json.dumps(self.project()))
            with patch("salsbury_md_analysis.feature_matrix.common_pca_project",return_value=fake_pca):
                result = clustering_imwkmeans_project(path)
        self.assertEqual(result["fit_observation_count"],12)
        self.assertEqual(result["full_assignment_observation_count"],24)
        self.assertEqual(result["selected_model"]["cluster_sizes"],[12,12])
        self.assertEqual(result["selected_model"]["fit_cluster_sizes"],[6,6])
        self.assertEqual({a["replica_id"] for a in result["assignments"]},{"r0","r1"})
        self.assertEqual({a["segment_id"] for a in result["assignments"]},{"s0","s1"})
        self.assertEqual(len(result["assignments"]),24)
        self.assertEqual(result["selected_model"]["objective_scope"],"pooled_fit_sample")

    def test_invalid_stride_rejected_before_analysis(self):
        for stride in (0,-1,1.5,True):
            project = self.project()
            project["definitions"]["clustering_imwkmeans"]["fit_stride"] = stride
            with self.assertRaises(ClusteringAnalysisError):
                _imwkmeans_settings(project)
