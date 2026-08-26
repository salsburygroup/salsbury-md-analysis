import unittest

from salsbury_md_analysis.helical_mechanics import build_helical_mechanics
from salsbury_md_analysis.nucleic_acid_structure import (
    discover_helical_step_parameter_path,
)


def _settings(status="available"):
    return {
        "source_module": "nucleic_acid_structure",
        "duplex_collection_field": "stems",
        "descriptor_query_ids": {
            name: f"helical-step-{name}"
            for name in ("shift", "slide", "rise", "tilt", "roll", "twist")
        },
        "angular_input_unit": "degrees",
        "minimum_frames_per_step": 8,
        "minimum_frames_per_state": 4,
        "maximum_states": 2,
        "minimum_silhouette_for_state_split": 0.2,
        "covariance_eigenvalue_floor_fraction": 1.0e-6,
        "maximum_steps": 10,
        "preparation_availability": {"status": status, "reason": None},
    }


class HelicalMechanicsTests(unittest.TestCase):
    def test_discovers_installed_dssr_step_object_path(self):
        contract = discover_helical_step_parameter_path({
            "stems": [{"steps": [{
                "shift": 0.1, "slide": 0.2, "rise": 3.4,
                "tilt": 1.0, "roll": 2.0, "twist": 34.0,
            }]}],
        })
        self.assertEqual(contract["object_path"], ["stems", "*", "steps", "*"])
        self.assertEqual(contract["fields"]["twist"], "twist")

    def test_duplex_frames_produce_state_specific_mechanics_and_coupling(self):
        frames = []
        components = ("shift", "slide", "rise", "tilt", "roll", "twist")
        for index in range(24):
            group = 0.0 if index < 12 else 5.0
            values = {
                "shift": [group + 0.03 * index, group + 0.02 * index],
                "slide": [0.1 * index, 0.08 * index],
                "rise": [3.2 + 0.01 * index, 3.4 + 0.01 * index],
                "tilt": [group + 0.2 * index, group + 0.1 * index],
                "roll": [2.0 + 0.15 * index, 3.0 + 0.12 * index],
                "twist": [32.0 + 0.1 * index, 34.0 + 0.1 * index],
            }
            frames.append({
                "system_id": "duplex", "replica_id": "r1",
                "segment_id": "production", "source_frame_index": index,
                "collection_counts": {"stems": 1},
                "numeric_queries": [{
                    "query_id": f"helical-step-{name}", "values": values[name],
                } for name in components],
            })
        source = {
            "module_id": "nucleic_acid_structure", "technical_status": "complete",
            "implementation": {
                "executable_path": "/opt/x3dna-dssr",
                "version_output": "DSSR test",
            },
            "frame_reports": frames,
        }
        report = build_helical_mechanics(
            source, _settings(), temperature_kelvin=300.0
        )
        self.assertEqual(report["availability_status"], "available")
        self.assertTrue(report["analysis_performed"])
        self.assertEqual(len(report["step_state_models"]), 2)
        self.assertEqual(len(report["neighbor_step_couplings"]), 1)
        self.assertTrue(report["step_state_models"][0]["states"])
        self.assertIn(
            "stiffness_matrix_kcal_per_mol_mixed_coordinates",
            report["step_state_models"][0]["states"][0],
        )

    def test_unavailable_preparation_gate_is_a_complete_noop_contract(self):
        report = build_helical_mechanics(
            None, _settings(status="not_available"), temperature_kelvin=300.0
        )
        self.assertEqual(report["availability_status"], "not_available")
        self.assertFalse(report["analysis_performed"])


if __name__ == "__main__":
    unittest.main()
