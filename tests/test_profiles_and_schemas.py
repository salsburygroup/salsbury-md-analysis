import json
import unittest
from pathlib import Path

from salsbury_md_analysis.analysis_config import default_analysis_config
from salsbury_md_analysis.registry import MODULES


ROOT = Path(__file__).resolve().parents[1]


class ProfileAndSchemaTests(unittest.TestCase):
    def test_analysis_profiles_reference_registered_modules(self):
        registered = {module.module_id for module in MODULES}
        paths = sorted((ROOT / "profiles").glob("standard_*.json"))
        self.assertEqual([path.name for path in paths], ["standard_md_v1.json"])
        for path in paths:
            profile = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(profile["modules"]), len(set(profile["modules"])), path.name)
            self.assertEqual(set(profile["modules"]), registered, path.name)

    def test_all_schemas_are_json_objects(self):
        schemas = sorted((ROOT / "schemas").glob("*.json"))
        self.assertTrue(schemas)
        for path in schemas:
            schema = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(schema.get("type"), "object", path.name)
            self.assertIn("$schema", schema, path.name)

    def test_analysis_config_schema_exposes_every_clustering_method(self):
        schema = json.loads(
            (ROOT / "schemas" / "analysis-config.schema.json").read_text(
                encoding="utf-8"
            )
        )
        methods = schema["properties"]["clustering"]["properties"]["methods"][
            "properties"
        ]
        self.assertEqual(set(methods), {
            "kmeans", "hdbscan", "intelligent_minkowski_weighted_kmeans",
            "pam", "minkowski_weighted_pam", "ward", "gaussian_mixture",
            "variational_gaussian_mixture", "affinity_propagation",
            "mean_shift", "quality_threshold",
        })
        pald = schema["properties"]["community_analysis"]["properties"]["pald"]
        self.assertFalse(pald["properties"]["enabled"]["default"])
        self.assertFalse(
            pald["properties"]["community_msm_enabled"]["default"]
        )
        self.assertEqual(
            schema["properties"]["clustering"]["properties"]["feature_space"][
                "default"
            ],
            "tica",
        )

    def test_analysis_config_schema_accepts_generated_metadata_and_defaults(self):
        schema = json.loads(
            (ROOT / "schemas" / "analysis-config.schema.json").read_text(
                encoding="utf-8"
            )
        )
        generated = default_analysis_config(
            [module.module_id for module in MODULES], ["global_common_heavy"]
        )
        module_properties = schema["properties"]["modules"][
            "additionalProperties"
        ]["properties"]
        for field in generated["modules"][MODULES[0].module_id]:
            self.assertIn(field, module_properties)
        self.assertTrue(module_properties["protected"]["readOnly"])
        execution_schema = schema["properties"]["execution"]["properties"]
        self.assertEqual(
            execution_schema["submission_adapter"]["default"],
            generated["execution"]["submission_adapter"],
        )
        planning_schema = schema["properties"]["planning"]["properties"]
        for field, value in generated["planning"].items():
            self.assertEqual(planning_schema[field]["default"], value)

    def test_project_schema_accepts_tica_clustering_and_dual_msm_sources(self):
        schema = json.loads(
            (ROOT / "schemas" / "project.schema.json").read_text(
                encoding="utf-8"
            )
        )
        definitions = schema["properties"]["definitions"]["properties"]
        for module_id in (
            "clustering_kmeans", "clustering_hdbscan",
            "clustering_imwkmeans", "alternative_clustering",
        ):
            self.assertIn("tica", definitions[module_id]["properties"][
                "feature_source"
            ]["enum"])
        sources = definitions["markov_state_models"]["properties"][
            "assignment_sources"
        ]
        self.assertEqual(
            set(sources["items"]["enum"]),
            {"best_clustering", "pca_fes_basins"},
        )

    def test_reporting_standard_is_nonredundant_and_scott_first(self):
        profile = json.loads(
            (ROOT / "reporting" / "reporting_standard_v1.json").read_text(
                encoding="utf-8"
            )
        )
        rows = profile["presentation_priority"]
        self.assertEqual([row["rank"] for row in rows], list(range(1, 7)))
        self.assertEqual(len({row["result"] for row in rows}), 6)
        self.assertEqual(
            profile["time_series_policy"]["default_histogram_rule"], "scott"
        )
        self.assertEqual(
            profile["time_series_policy"]["exceptions"]["rmsd"],
            "replica_resolved_time_series_primary",
        )
        cluster_policy = profile["cluster_partition_interpretation_policy"]
        self.assertEqual(
            cluster_policy["replica_or_preparation_association"],
            "report_and_retain",
        )
        self.assertEqual(
            set(cluster_policy["retained_outputs"]),
            {
                "assignments", "population_tables", "representative_structures",
                "state_trajectories",
            },
        )


if __name__ == "__main__":
    unittest.main()
