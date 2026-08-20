import runpy
import unittest
from unittest.mock import patch


class ModuleEntrypointTests(unittest.TestCase):
    def test_python_m_propagates_cli_exit_status(self):
        with patch("salsbury_md_analysis.cli.main", return_value=7):
            with self.assertRaises(SystemExit) as caught:
                runpy.run_module("salsbury_md_analysis", run_name="__main__")
        self.assertEqual(caught.exception.code, 7)


if __name__ == "__main__":
    unittest.main()
