import unittest

from salsbury_md_analysis.integrated import IntegratedAnalysisError, json_pointer


class IntegratedTests(unittest.TestCase):
    def test_json_pointer_resolves_dict_list_and_escapes(self):
        document = {"rows": [{"a/b": {"~key": 7}}]}
        self.assertEqual(json_pointer(document, "/rows/0/a~1b/~0key"), 7)
        with self.assertRaises(IntegratedAnalysisError):
            json_pointer(document, "/rows/2")


if __name__ == "__main__":
    unittest.main()
