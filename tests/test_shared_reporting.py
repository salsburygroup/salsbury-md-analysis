import unittest

from salsbury_md_analysis.atom_mapping import AtomRecord
from salsbury_md_analysis.provenance import stable_json_sha256
from salsbury_md_analysis.reporting import atom_identity_record, issue_record


class SharedReportingTests(unittest.TestCase):
    def test_stable_json_signature_is_order_independent_and_ascii_canonical(self):
        first = {"b": 1, "a": "µ"}
        second = {"a": "µ", "b": 1}
        expected = "7b0cd9f1641262553fbbbd783da3c9f002390aae870873366d8786810e630821"
        self.assertEqual(stable_json_sha256(first), expected)
        self.assertEqual(stable_json_sha256(second), expected)

    def test_issue_record_preserves_common_fields_and_optional_details(self):
        self.assertEqual(
            issue_record(
                "warning", "TEST_WARNING", "system/replica", "test message",
                frame_index=7,
            ),
            {
                "severity": "warning",
                "code": "TEST_WARNING",
                "location": "system/replica",
                "message": "test message",
                "frame_index": 7,
            },
        )

    def test_atom_identity_record_supports_local_and_common_indices(self):
        atom = AtomRecord(
            atom_index=4,
            serial=9,
            atom_name="CA",
            altloc="",
            residue_name="ALA",
            chain_id="A",
            residue_number=12,
            insertion_code="",
            element="C",
        )
        expected = {
            "reference_atom_index": 4,
            "serial": 9,
            "atom_name": "CA",
            "element": "C",
            "residue_name": "ALA",
            "chain_id": "A",
            "residue_number": 12,
            "insertion_code": "",
            "altloc": "",
        }
        self.assertEqual(atom_identity_record(atom), expected)
        self.assertEqual(
            atom_identity_record(atom, common_index=2),
            {"common_atom_index": 2, **expected},
        )


if __name__ == "__main__":
    unittest.main()
