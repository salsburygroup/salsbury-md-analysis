"""Verify a generated molecular cache before skipping repeated unwrapping."""

from pathlib import Path
from typing import Mapping

from .manifests import load_json, resolve_manifest_path, sha256_file


_RECOGNIZED_CACHES = {"continuous_unwrap_strided_molecular_payload_v2",
                      "independent_make_whole_molecular_payload_v1"}


def validate_cached_manifest(data: Mapping, manifest: Path) -> list[dict]:
    """Validate materialized recognized caches with the runtime contract.

    Raw inputs need no cache sidecar. Cache payload digests are mandatory even
    when ordinary preflight inventory hashing was not requested. Arbitrary
    derived manifest names are valid only when the report binds their identity.
    This function never exempts incomplete caches on the basis of a future job.
    """
    from .preflight import probe_topology

    systems = data.get("systems", [])
    if not any(s.get("metadata", {}).get("coordinate_cache") in _RECOGNIZED_CACHES
               for s in systems):
        return []
    identities = []
    for system in systems:
        static = (system.get("metadata", {}).get("coordinate_cache")
                  == "independent_make_whole_molecular_payload_v1")
        for replica in system.get("replicas", []):
            topology = resolve_manifest_path(replica["topology"], manifest)
            atom_count = int(probe_topology(topology)["atom_count"])
            identity = discover_cached_replica(manifest, replica, atom_count, static=static)
            if identity is None:
                raise ValueError("recognized cache is not declared in the saved system manifest")
            identities.append({"system_id": system["system_id"],
                               "replica_id": replica["replica_id"], **identity})
    return identities


def cache_manifest_digest(report: Mapping, report_path: Path, manifest: Path) -> str:
    """Resolve a pooled or declared per-system manifest, never an arbitrary subset."""
    manifest = manifest.resolve()
    candidates = [{"path": report.get("cached_system_manifest"),
                   "sha256": report.get("cached_system_manifest_sha256")}]
    per_system = report.get("cached_per_system_manifests", {})
    if not isinstance(per_system, Mapping):
        raise ValueError("preprocessed cache per-system identities must be an object")
    candidates.extend(per_system.values())
    for row in candidates:
        if (isinstance(row, Mapping) and isinstance(row.get("path"), str)
                and resolve_manifest_path(row["path"], report_path) == manifest):
            digest = sha256_file(manifest)
            if digest != row.get("sha256"):
                raise ValueError("preprocessed cache system manifest hash does not match")
            return digest
    raise ValueError("preprocessed cache report names a different system manifest")


def discover_cached_replica(manifest: Path, replica: Mapping, atom_count: int,
                            *, static: bool) -> dict | None:
    """Recognize only cache manifests with verified report and payload identities.

    Ordinary trajectories, even a DCD with a copied cache title, cannot enter
    this path. No source trajectory or project is modified. Hash only the
    current replica's cache files, not all other replicas on every scan.
    """
    manifest = Path(manifest).resolve()
    document = load_json(manifest)
    systems = document.get("systems", [])
    declarations = {s.get("metadata", {}).get("coordinate_cache") for s in systems}
    if not declarations.intersection(_RECOGNIZED_CACHES):
        return None
    if not declarations.issubset(_RECOGNIZED_CACHES):
        raise ValueError("mixed raw/cache manifest cannot skip reconstruction")
    report_path = manifest.parent / "coordinate-cache-report.json"
    report = load_json(report_path)
    representation = ("independent_make_whole_unaligned_strided" if static
                      else "continuous_unwrap_unaligned_strided")
    if (report.get("technical_status") != "complete"
            or report.get("coordinate_representation") != representation
            or report.get("selection") != "molecular_payload"
            or report.get("source_frame_scan") != "all source frames decoded in order"):
        raise ValueError("coordinate cache lacks a complete matching reconstruction contract")
    manifest_digest = cache_manifest_digest(report, report_path, manifest)
    matches = [(s["system_id"], r["replica_id"]) for s in systems
               for r in s.get("replicas", []) if r == replica]
    if len(matches) != 1:
        raise ValueError("cached replica does not match exactly one manifest entry")
    rows = [r for r in report.get("rows", [])
            if (r.get("system_id"), r.get("replica_id")) == matches[0]]
    if len(rows) != 1 or rows[0].get("cached_atom_count") != atom_count:
        raise ValueError("coordinate cache replica identity or atom count differs")
    row = rows[0]
    for field in ("topology", "connectivity"):
        path = resolve_manifest_path(replica[field], manifest)
        if sha256_file(path) != row.get(field + "_sha256"):
            raise ValueError("cached " + field + " hash does not match")
    segments = [s for s in row.get("segments", []) if s.get("cache") is not None]
    if len(segments) != len(replica.get("segments", [])):
        raise ValueError("coordinate cache segment coverage differs")
    for declared, saved in zip(replica["segments"], segments):
        path = resolve_manifest_path(declared["trajectory"], manifest)
        identity = saved["cache"]
        if (declared["segment_id"] != saved["segment_id"]
                or path != resolve_manifest_path(identity["path"], report_path)
                or sha256_file(path) != identity["sha256"]):
            raise ValueError("cached trajectory path, segment identity or hash differs")
    return {"cache_report": str(report_path),
            "cache_report_sha256": sha256_file(report_path),
            "cached_system_manifest_sha256": manifest_digest,
            "coordinate_representation": representation,
            "validation": "manifest_and_replica_payload_hashes_v1"}
