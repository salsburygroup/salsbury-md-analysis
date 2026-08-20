import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.upstream_cache import (
    load_cached_project_report,
    project_module_contract_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UpstreamCacheTests(unittest.TestCase):
    def _fixture(self, root: Path):
        system = root / "system.json"
        system.write_text('{"systems": []}\n', encoding="utf-8")
        project = root / "project.json"
        project.write_text(
            json.dumps({
                "project_id": "original",
                "system_manifest": "system.json",
                "analysis_output_root": "original-results",
                "requested_modules": ["common_pca"],
                "definitions": {"common_pca": {"component_count": 3}},
            }) + "\n",
            encoding="utf-8",
        )
        report = root / "common-pca.json"
        report.write_text(
            json.dumps(
                {
                    "module_id": "common_pca",
                    "technical_status": "complete",
                    "project_manifest_path": str(project),
                    "project_manifest_sha256": _sha256(project),
                    "module_contract_sha256": project_module_contract_sha256(
                        "common_pca", project
                    ),
                    "system_manifest_path": str(system),
                    "system_manifest_sha256": _sha256(system),
                    "input_content_signature_sha256": "a" * 64,
                    "systems": [],
                }
            ),
            encoding="utf-8",
        )
        preflight = root / "preflight.json"
        preflight.write_text(
            json.dumps(
                {
                    "technical_status": "complete",
                    "content_hashes_included": True,
                    "manifest_path": str(system),
                    "manifest_sha256": _sha256(system),
                    "input_content_signature_sha256": "a" * 64,
                }
            ),
            encoding="utf-8",
        )
        return project, system, report, preflight

    def test_matching_cache_is_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            project, _, report, preflight = self._fixture(Path(temporary))
            with patch.dict(
                os.environ,
                {
                    "SALSBURY_MD_ANALYSIS_COMMON_PCA_REPORT": str(report),
                    "SALSBURY_MD_ANALYSIS_PREFLIGHT_REPORT": str(preflight),
                },
                clear=False,
            ):
                observed = load_cached_project_report(
                    "common_pca",
                    project,
                    hash_content=True,
                    error_type=ValueError,
                )
            self.assertEqual(observed["module_id"], "common_pca")
            payload = json.loads(preflight.read_text(encoding="utf-8"))
            payload["input_content_signature_sha256"] = "b" * 64
            preflight.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "SALSBURY_MD_ANALYSIS_COMMON_PCA_REPORT": str(report),
                    "SALSBURY_MD_ANALYSIS_PREFLIGHT_REPORT": str(preflight),
                },
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "does not match preflight"):
                    load_cached_project_report(
                        "common_pca", project, hash_content=True,
                        error_type=ValueError,
                    )

    def test_missing_preflight_rehashes_current_inputs_instead_of_trusting_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            project, _, report, _ = self._fixture(Path(temporary))
            with patch.dict(
                os.environ,
                {"SALSBURY_MD_ANALYSIS_COMMON_PCA_REPORT": str(report)},
                clear=True,
            ), patch(
                "salsbury_md_analysis.upstream_cache.compile_project_context_file",
                return_value={"input_content_signature_sha256": "a" * 64},
            ) as compile_context:
                observed = load_cached_project_report(
                    "common_pca", project, hash_content=True,
                    error_type=ValueError,
                )
            self.assertEqual(observed["module_id"], "common_pca")
            compile_context.assert_called_once()
            args, kwargs = compile_context.call_args
            self.assertEqual(args[0], project.resolve())
            self.assertEqual(kwargs, {"hash_content": True})

    def test_configured_cache_fails_closed_after_manifest_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            project, system, report, preflight = self._fixture(Path(temporary))
            system.write_text('{"systems": [], "changed": true}\n', encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "SALSBURY_MD_ANALYSIS_COMMON_PCA_REPORT": str(report),
                    "SALSBURY_MD_ANALYSIS_PREFLIGHT_REPORT": str(preflight),
                },
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "system-manifest hash"):
                    load_cached_project_report(
                        "common_pca",
                        project,
                        hash_content=True,
                        error_type=ValueError,
                    )

    def test_equivalent_module_contract_can_reuse_cache_across_project_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, _, report, preflight = self._fixture(root)
            variant = root / "variant.json"
            payload = json.loads(project.read_text(encoding="utf-8"))
            payload.update({
                "project_id": "alternative-calibration-750",
                "analysis_output_root": "calibration-results",
                "requested_modules": ["alternative_clustering"],
            })
            payload["definitions"]["alternative_clustering"] = {
                "fit_budget": 750
            }
            variant.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "SALSBURY_MD_ANALYSIS_COMMON_PCA_REPORT": str(report),
                    "SALSBURY_MD_ANALYSIS_PREFLIGHT_REPORT": str(preflight),
                },
                clear=False,
            ):
                observed = load_cached_project_report(
                    "common_pca", variant, hash_content=True,
                    error_type=ValueError,
                )
            self.assertEqual(observed["module_id"], "common_pca")
            payload["definitions"]["common_pca"]["component_count"] = 4
            variant.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "SALSBURY_MD_ANALYSIS_COMMON_PCA_REPORT": str(report),
                    "SALSBURY_MD_ANALYSIS_PREFLIGHT_REPORT": str(preflight),
                },
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "module-contract"):
                    load_cached_project_report(
                        "common_pca", variant, hash_content=True,
                        error_type=ValueError,
                    )

    def test_legacy_report_without_contract_reuses_hash_matched_original_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, _, report, preflight = self._fixture(root)
            report_payload = json.loads(report.read_text(encoding="utf-8"))
            report_payload.pop("module_contract_sha256")
            report.write_text(json.dumps(report_payload), encoding="utf-8")

            variant = root / "variant.json"
            payload = json.loads(project.read_text(encoding="utf-8"))
            payload.update({
                "project_id": "state-export-recovery",
                "analysis_output_root": "recovery-results",
                "requested_modules": ["state_coordinate_exports"],
            })
            payload["definitions"]["state_coordinate_exports"] = {
                "maximum_states": 250
            }
            variant.write_text(json.dumps(payload), encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "SALSBURY_MD_ANALYSIS_COMMON_PCA_REPORT": str(report),
                    "SALSBURY_MD_ANALYSIS_PREFLIGHT_REPORT": str(preflight),
                },
                clear=False,
            ):
                observed = load_cached_project_report(
                    "common_pca", variant, hash_content=True,
                    error_type=ValueError,
                )
            self.assertEqual(observed["module_id"], "common_pca")

    def test_legacy_report_fails_if_original_project_no_longer_matches_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, _, report, preflight = self._fixture(root)
            report_payload = json.loads(report.read_text(encoding="utf-8"))
            report_payload.pop("module_contract_sha256")
            report.write_text(json.dumps(report_payload), encoding="utf-8")

            variant = root / "variant.json"
            payload = json.loads(project.read_text(encoding="utf-8"))
            payload.update({
                "project_id": "state-export-recovery",
                "analysis_output_root": "recovery-results",
            })
            variant.write_text(json.dumps(payload), encoding="utf-8")
            project.write_text(project.read_text(encoding="utf-8") + "\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "SALSBURY_MD_ANALYSIS_COMMON_PCA_REPORT": str(report),
                    "SALSBURY_MD_ANALYSIS_PREFLIGHT_REPORT": str(preflight),
                },
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "module-contract"):
                    load_cached_project_report(
                        "common_pca", variant, hash_content=True,
                        error_type=ValueError,
                    )

    def test_unconfigured_cache_returns_none(self):
        with tempfile.TemporaryDirectory() as temporary:
            project, _, _, _ = self._fixture(Path(temporary))
            with patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(
                    load_cached_project_report(
                        "common_pca",
                        project,
                        hash_content=True,
                        error_type=ValueError,
                    )
                )


if __name__ == "__main__":
    unittest.main()
