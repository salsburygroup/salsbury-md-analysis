"""Identity checks for coordinate-derived figures attached to reader findings."""
from __future__ import annotations

import hashlib
import json


def finding_signature(row):
    """Bind a panel to a scientific claim, not a rank that can change on replay."""
    identity = {key: row.get(key) for key in (
        "module_id", "statement", "system_ids", "view_ids", "presentation_target",
    )}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def panel_matches_finding(panel, row, artifacts, verified):
    """Require verified coordinates, a saved view and the exact claim identity.

    This verifies provenance, not whether the chosen view explains the science.
    External molecular renderers can supply this same manifest contract.
    """
    evidence = panel.get("molecular_evidence", {})
    if not isinstance(evidence, dict):
        return False
    if (panel.get("artifact_type") != "figure"
            or panel.get("purpose") not in {"structural_figure", "representative_structure_figure"}
            or evidence.get("finding_signature_sha256") != finding_signature(row)):
        return False
    coordinates = evidence.get("coordinate_artifacts", [])
    if not isinstance(coordinates, list) or not coordinates:
        return False
    for source in coordinates:
        if not isinstance(source, dict):
            return False
        aid = source.get("artifact_id")
        artifact = artifacts.get(aid, {})
        if (aid not in verified or artifact.get("artifact_type") != "structure"
                or source.get("sha256") != artifact.get("artifact_sha256")):
            return False
    view_id = evidence.get("saved_view_artifact_id")
    view = artifacts.get(view_id, {})
    return bool(view_id in verified and view.get("artifact_type") == "table"
                and view.get("purpose") == "molecular_saved_view"
                and evidence.get("saved_view_sha256") == view.get("artifact_sha256"))
