"""Fail-closed reuse of immutable upstream reports in staged workflows."""

from __future__ import annotations

import json
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, Optional, Type

from .context import compile_project_context_file
from .manifests import load_json, resolve_manifest_path, sha256_file


_ENVIRONMENT_VARIABLES = {
    "common_pca": "SALSBURY_MD_ANALYSIS_COMMON_PCA_REPORT",
    "dccm": "SALSBURY_MD_ANALYSIS_DCCM_REPORT",
    "replica_rmsd_rg": "SALSBURY_MD_ANALYSIS_RMSD_RG_REPORT",
    "clustering_kmeans": "SALSBURY_MD_ANALYSIS_KMEANS_REPORT",
    "clustering_hdbscan": "SALSBURY_MD_ANALYSIS_HDBSCAN_REPORT",
    "clustering_imwkmeans": "SALSBURY_MD_ANALYSIS_IMWKMEANS_REPORT",
    "alternative_clustering": "SALSBURY_MD_ANALYSIS_ALTERNATIVE_CLUSTERING_REPORT",
    "pald_community_analysis": "SALSBURY_MD_ANALYSIS_PALD_COMMUNITY_REPORT",
    "pca_fes_basins": "SALSBURY_MD_ANALYSIS_FES_REPORT",
    "trajectory_features": "SALSBURY_MD_ANALYSIS_TRAJECTORY_FEATURES_REPORT",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PREFLIGHT_ENVIRONMENT_VARIABLE = "SALSBURY_MD_ANALYSIS_PREFLIGHT_REPORT"


def project_module_contract_sha256(
    module_id: str, project_path: Path
) -> str:
    """Hash the scientific contract for one module, excluding downstream fields."""

    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    definitions = project.get("definitions")
    if not isinstance(definitions, dict) or not isinstance(definitions.get(module_id), dict):
        raise ValueError(f"project contains no definitions.{module_id} object")
    excluded = {
        "project_id", "analysis_output_root", "requested_modules",
        "protected_locations", "definitions",
    }
    contract = {
        key: value for key, value in project.items() if key not in excluded
    }
    contract["definitions"] = {module_id: definitions[module_id]}
    for key in (
        "system_manifest", "reference_structure", "reference_connectivity"
    ):
        value = contract.get(key)
        if isinstance(value, str) and value.strip():
            contract[key] = str(resolve_manifest_path(value, source))
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cached_project_report(
    module_id: str,
    project_path: Path,
    *,
    hash_content: bool,
    error_type: Type[ValueError],
) -> Optional[Dict[str, object]]:
    """Return a matching complete cached report, or ``None`` when not configured.

    A configured cache never falls back silently. The report must match the
    current project and system manifest bytes; content-hashing callers also
    require a complete input-content signature from the upstream run.
    """

    try:
        variable = _ENVIRONMENT_VARIABLES[module_id]
    except KeyError as exc:
        raise error_type(f"unsupported upstream cache module: {module_id}") from exc
    declared = os.environ.get(variable)
    if not declared:
        return None
    report_path = Path(declared).expanduser().resolve(strict=False)
    if not report_path.is_file():
        raise error_type(f"{variable} does not name a readable report: {report_path}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise error_type(f"cached {module_id} report is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise error_type(f"cached {module_id} report must contain a JSON object")
    if payload.get("technical_status") != "complete":
        raise error_type(f"cached {module_id} report is not technically complete")
    if payload.get("module_id") != module_id:
        raise error_type(
            f"cached report module is {payload.get('module_id')!r}; expected {module_id!r}"
        )

    project_source = Path(project_path).expanduser().resolve(strict=False)
    reported_project = Path(
        str(payload.get("project_manifest_path", ""))
    ).expanduser().resolve(strict=False)
    exact_project = (
        payload.get("project_manifest_sha256") == sha256_file(project_source)
        and reported_project == project_source
    )
    if not exact_project:
        reported_contract = payload.get("module_contract_sha256")
        try:
            current_contract = project_module_contract_sha256(
                module_id, project_source
            )
        except (OSError, ValueError) as exc:
            raise error_type(str(exc)) from exc
        if reported_contract != current_contract:
            # Reports produced before module-contract hashes were recorded can
            # still be reused safely when their original project manifest is
            # present and byte-for-byte matches the hash embedded in the
            # report.  Recompute the old and current module contracts with the
            # current canonical algorithm; downstream-only recovery changes
            # may then reuse the immutable upstream result without weakening
            # the system/input gates below.
            reported_project_sha256 = payload.get("project_manifest_sha256")
            legacy_contract_matches = False
            try:
                reported_project_is_authentic = (
                    reported_project.is_file()
                    and isinstance(reported_project_sha256, str)
                    and _SHA256.fullmatch(reported_project_sha256) is not None
                    and sha256_file(reported_project) == reported_project_sha256
                )
                if reported_project_is_authentic:
                    legacy_contract_matches = (
                        project_module_contract_sha256(module_id, reported_project)
                        == current_contract
                    )
            except (OSError, ValueError):
                legacy_contract_matches = False
            if not legacy_contract_matches:
                raise error_type(
                    f"cached {module_id} report project and module-contract hashes do not match"
                )

    project = load_json(project_source)
    system_text = project.get("system_manifest")
    if not isinstance(system_text, str) or not system_text.strip():
        raise error_type("current project has no valid system_manifest")
    system_source = resolve_manifest_path(system_text, project_source)
    if payload.get("system_manifest_sha256") != sha256_file(system_source):
        raise error_type(f"cached {module_id} report system-manifest hash does not match")
    reported_system = Path(
        str(payload.get("system_manifest_path", ""))
    ).expanduser().resolve(strict=False)
    if reported_system != system_source:
        raise error_type(f"cached {module_id} report system path does not match")

    if hash_content:
        signature = payload.get("input_content_signature_sha256")
        if not isinstance(signature, str) or _SHA256.fullmatch(signature) is None:
            raise error_type(
                f"cached {module_id} report lacks a complete input-content signature"
            )
        preflight_text = os.environ.get(_PREFLIGHT_ENVIRONMENT_VARIABLE)
        if not preflight_text:
            try:
                current_context = compile_project_context_file(
                    project_source, hash_content=True
                )
            except (OSError, ValueError) as exc:
                raise error_type(
                    "current input content could not be revalidated for cached "
                    f"{module_id}: {exc}"
                ) from exc
            if current_context.get("input_content_signature_sha256") != signature:
                raise error_type(
                    f"cached {module_id} input-content signature does not match "
                    "the current input files"
                )
            return payload
        preflight_path = Path(preflight_text).expanduser().resolve(strict=False)
        if not preflight_path.is_file():
            raise error_type(
                f"{_PREFLIGHT_ENVIRONMENT_VARIABLE} does not name a readable report: "
                f"{preflight_path}"
            )
        try:
            preflight = json.loads(
                preflight_path.read_text(encoding="utf-8", errors="strict")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise error_type(f"input preflight report is unreadable: {exc}") from exc
        if not isinstance(preflight, dict):
            raise error_type("input preflight report must contain a JSON object")
        if preflight.get("technical_status") != "complete":
            raise error_type("input preflight report is not technically complete")
        if preflight.get("content_hashes_included") is not True:
            raise error_type("input preflight report does not include content hashes")
        if preflight.get("manifest_sha256") != sha256_file(system_source):
            raise error_type("input preflight system-manifest hash does not match")
        preflight_system = Path(
            str(preflight.get("manifest_path", ""))
        ).expanduser().resolve(strict=False)
        if preflight_system != system_source:
            raise error_type("input preflight system path does not match")
        if preflight.get("input_content_signature_sha256") != signature:
            raise error_type(
                f"cached {module_id} input-content signature does not match preflight"
            )
    return payload
