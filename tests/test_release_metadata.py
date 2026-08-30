import configparser
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseMetadataTests(unittest.TestCase):
    def test_version_is_consistent_across_package_citation_and_readme(self):
        parser = configparser.ConfigParser()
        parser.read(ROOT / "setup.cfg", encoding="utf-8")
        package_version = parser["metadata"]["version"]

        init_text = (ROOT / "src" / "salsbury_md_analysis" / "__init__.py").read_text(
            encoding="utf-8"
        )
        init_match = re.search(r'^__version__\s*=\s*"([^"]+)"$', init_text, re.MULTILINE)
        self.assertIsNotNone(init_match)
        self.assertEqual(init_match.group(1), package_version)

        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        citation_match = re.search(r'^version:\s*"([^"]+)"$', citation, re.MULTILINE)
        self.assertIsNotNone(citation_match)
        self.assertEqual(citation_match.group(1), package_version)

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(f"Version {package_version}", readme)

if __name__ == "__main__":
    unittest.main()
