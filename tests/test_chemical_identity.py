import unittest

from salsbury_md_analysis.chemical_identity import (
    ANION_ELEMENTS,
    CATION_ELEMENTS,
    ION_RESIDUES,
    NUCLEIC_RESIDUES,
    PROTEIN_RESIDUES,
    SOLVENT_AND_ION_RESIDUES,
    WATER_RESIDUES,
)


class ChemicalIdentityVocabularyTests(unittest.TestCase):
    def test_supported_generic_system_vocabulary_covers_documented_classes(self):
        self.assertTrue({"HIS", "HID", "HIE", "HIP"}.issubset(PROTEIN_RESIDUES))
        self.assertTrue({"DA", "DT", "RA", "RU", "8OG"}.issubset(NUCLEIC_RESIDUES))
        self.assertTrue({"NA", "K", "MG", "CA", "ZN", "FE", "CL"}.issubset(ION_RESIDUES))
        self.assertTrue({"NA", "K", "MG", "CA", "ZN", "FE"}.issubset(CATION_ELEMENTS))
        self.assertIn("CL", ANION_ELEMENTS)

    def test_solvent_and_ion_union_is_exact(self):
        self.assertEqual(SOLVENT_AND_ION_RESIDUES, WATER_RESIDUES | ION_RESIDUES)
        self.assertTrue(WATER_RESIDUES.isdisjoint(PROTEIN_RESIDUES))


if __name__ == "__main__":
    unittest.main()
