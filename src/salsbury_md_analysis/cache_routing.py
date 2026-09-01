"""Validated routing of base analyses to a molecular-payload coordinate cache."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Mapping, Sequence

from .coordinate_cache import (
    coordinate_cache_prefix,
    coordinate_cache_system_manifest_filename,
    validate_reusable_coordinate_cache,
)
from .manifests import load_json, validate_project


class CacheRoutingError(ValueError):
    """Raised when a base module cannot be routed without changing its atoms."""


_CACHE_COMPATIBLE_BASE_MODULES = {
    "replica_rmsd_rg", "pooled_rmsf", "dccm", "individual_pca",
    "dihedral_distributions", "hydrogen_bond_discovery",
    "secondary_structure", "solvent_accessible_surface_area",
    "nucleic_acid_structure", "nucleic_acid_geometry",
    "ion_coordination_geometry", "ion_atmosphere", "trajectory_features",
    "optional_observables", "allosteric_pathways",
    "ensemble_pocket_dynamics",
}

_ORIGINAL_SOLVATED_MODULES = {
    "water_mediated_hydrogen_bond_networks",
    "radial_distribution_functions",
    "multivalent_molecular_bridges",
    "hydration_density_channels",
    "spatial_interaction_ensembles",
}


def cache_compatibility(
    module_id: str, project: Mapping[str, object]
) -> Dict[str, object]:
    """Return one explicit base-project cache routing decision."""

    if module_id in _ORIGINAL_SOLVATED_MODULES:
        return {
            "module_id": module_id,
            "cache_compatible": False,
            "reason": "module may require bulk-solvent coordinates",
        }
    if module_id == "structural_integrity_qc":
        return {
            "module_id": module_id,
            "cache_compatible": True,
            "route": "validated_internal_replica_cache",
            "reason": (
                "structural QC retains original-source provenance while its "
                "replica workers consume the validated cache"
            ),
        }
    if module_id == "energetic_network_embeddings":
        return {
            "module_id": module_id,
            "cache_compatible": False,
            "reason": (
                "force-field parameters require the original atom-order-matched "
                "PSF, PRMTOP/PARM7, or serialized OpenMM System connectivity"
            ),
        }
    if module_id not in _CACHE_COMPATIBLE_BASE_MODULES:
        return {
            "module_id": module_id,
            "cache_compatible": False,
            "reason": "module is infrastructure, derived reporting, or not a base estimator",
        }
    definitions = project.get("definitions")
    definition = (
        definitions.get(module_id) if isinstance(definitions, Mapping) else None
    )
    if _contains_explicit_atom_indices(definition):
        return {
            "module_id": module_id,
            "cache_compatible": False,
            "reason": (
                "module contains source atom-index definitions; a heterogeneous "
                "multi-system cache cannot apply one remapping to every system"
            ),
        }
    if module_id == "hydrogen_bond_discovery":
        if not isinstance(definition, Mapping) or definition.get("water_policy") != "exclude":
            return {
                "module_id": module_id,
                "cache_compatible": False,
                "reason": "hydrogen-bond discovery includes or ambiguously handles water",
            }
    return {
        "module_id": module_id,
        "cache_compatible": True,
        "route": "validated_cache_backed_base_project",
        "reason": "all required atoms are in the molecular-payload cache",
    }


def _contains_explicit_atom_indices(value: object) -> bool:
    """Return whether a definition embeds source-topology atom indices."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            label = str(key)
            if (
                label == "atom_index"
                or label.endswith("_atom_index")
                or label == "atom_indices"
                or label.endswith("_atom_indices")
            ):
                return True
            if _contains_explicit_atom_indices(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_explicit_atom_indices(item) for item in value)
    return False


def _write_validated_project(path: Path, payload: Mapping[str, object]) -> str:
    """Atomically write and validate one cache-backed project."""

    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    validate_project(payload, source_path=temporary, check_paths=True)
    temporary.replace(path)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def cache_routing_plan(project: Mapping[str, object]) -> Dict[str, object]:
    requested = project.get("requested_modules")
    if not isinstance(requested, list):
        raise CacheRoutingError("project requested_modules must be an array")
    decisions = [cache_compatibility(str(value), project) for value in requested]
    return {
        "cache_routing_schema": "salsbury-base-cache-routing-v1",
        "technical_status": "complete",
        "decisions": decisions,
        "cache_project_modules": sorted(
            str(row["module_id"])
            for row in decisions
            if row.get("route") == "validated_cache_backed_base_project"
        ),
        "internal_cache_modules": sorted(
            str(row["module_id"])
            for row in decisions
            if row.get("route") == "validated_internal_replica_cache"
        ),
        "original_project_modules": sorted(
            str(row["module_id"])
            for row in decisions if not bool(row["cache_compatible"])
        ),
    }


def _remap_atom_indices(value: object, mapping: Mapping[int, int]) -> object:
    """Remap atom-index fields in a module definition, failing on removed atoms."""

    def remap_one(raw: object, field: str) -> int:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise CacheRoutingError(f"{field} must contain integer atom indices")
        if raw not in mapping:
            raise CacheRoutingError(
                f"{field} references source atom {raw}, which is absent from the cache"
            )
        return mapping[raw]

    def visit(raw: object, field: str = "definition") -> object:
        if isinstance(raw, dict):
            output = {}
            for key, nested in raw.items():
                label = str(key)
                if label == "atom_index" or label.endswith("_atom_index"):
                    output[key] = remap_one(nested, label)
                elif label == "atom_indices" or label.endswith("_atom_indices"):
                    if not isinstance(nested, list):
                        raise CacheRoutingError(f"{label} must be an atom-index array")
                    output[key] = [remap_one(item, label) for item in nested]
                else:
                    output[key] = visit(nested, label)
            return output
        if isinstance(raw, list):
            return [visit(item, field) for item in raw]
        return deepcopy(raw)

    return visit(value)


def materialize_cache_backed_base_project(
    source_project_path: Path,
    cache_directory: Path,
    output_path: Path,
) -> Dict[str, object]:
    """Validate a cache and atom-remap one deterministic base project."""

    source_path = Path(source_project_path).expanduser().resolve(strict=True)
    project = load_json(source_path)
    if not isinstance(project, dict):
        raise CacheRoutingError("source project is not an object")
    raw_manifest = project.get("system_manifest")
    if not isinstance(raw_manifest, str):
        raise CacheRoutingError("source project lacks system_manifest")
    source_manifest = Path(raw_manifest)
    if not source_manifest.is_absolute():
        source_manifest = source_path.parent / source_manifest
    validation = validate_reusable_coordinate_cache(
        Path(cache_directory), source_manifest
    )
    report = load_json(Path(str(validation["cache_report"])))
    report_rows = report.get("rows") if isinstance(report, dict) else None
    if not isinstance(report_rows, list) or not report_rows:
        raise CacheRoutingError("coordinate cache report has no replica mappings")
    mappings = []
    for row in report_rows:
        if not isinstance(row, dict):
            raise CacheRoutingError("coordinate cache report row is invalid")
        indices = row.get("source_atom_indices_in_cache_order")
        if not isinstance(indices, list) or not indices:
            raise CacheRoutingError(
                "coordinate cache lacks source-to-cache atom-index provenance"
            )
        mappings.append(tuple(int(value) for value in indices))
    routing = cache_routing_plan(project)
    cache_modules = set(routing["cache_project_modules"])
    definitions = project.get("definitions")
    if not isinstance(definitions, dict):
        raise CacheRoutingError("source project lacks definitions")
    mapping_is_shared = all(value == mappings[0] for value in mappings[1:])
    source_to_cache = {
        source_index: cache_index
        for cache_index, source_index in enumerate(mappings[0])
    }
    remapped_definitions = deepcopy(definitions)
    if mapping_is_shared:
        for module_id in cache_modules:
            if module_id in remapped_definitions:
                remapped_definitions[module_id] = _remap_atom_indices(
                    remapped_definitions[module_id], source_to_cache
                )
    elif any(
        _contains_explicit_atom_indices(remapped_definitions.get(module_id))
        for module_id in cache_modules
    ):
        # cache_routing_plan normally excludes these modules before the cache
        # is built.  Keep the runtime materializer fail closed if a hand-edited
        # routing file attempts to put one back.
        raise CacheRoutingError(
            "heterogeneous cache mappings cannot share explicit atom-index "
            "definitions; route those modules to their original projects"
        )
    cached_manifest = Path(str(validation["cached_system_manifest"]))
    cached = load_json(cached_manifest)
    systems = cached.get("systems") if isinstance(cached, dict) else None
    if not isinstance(systems, list) or not systems or not isinstance(systems[0], dict):
        raise CacheRoutingError("cached system manifest has no systems")
    replicas = systems[0].get("replicas")
    if not isinstance(replicas, list) or not replicas or not isinstance(replicas[0], dict):
        raise CacheRoutingError("cached system manifest has no replicas")
    first_system = str(systems[0]["system_id"])
    first_replica = str(replicas[0]["replica_id"])
    prefix = coordinate_cache_prefix(first_system, first_replica)
    output = Path(output_path).expanduser().resolve(strict=False)
    payload = deepcopy(project)
    payload.update({
        "system_manifest": str(cached_manifest),
        "reference_structure": str(cached_manifest.parent / f"{prefix}.pdb"),
        "reference_connectivity": str(
            cached_manifest.parent / f"{prefix}.bonds.json"
        ),
        "periodic_coordinate_policy": "preprocessed_make_whole",
        "definitions": remapped_definitions,
        "requested_modules": sorted(cache_modules),
        "preprocessed_coordinate_source": {
            "cache_report": validation["cache_report"],
            "cache_report_sha256": validation["cache_report_sha256"],
        },
    })
    pooled_sha256 = _write_validated_project(output, payload)

    per_system_cache_projects = []
    per_system_manifests = report.get("cached_per_system_manifests")
    if not isinstance(per_system_manifests, Mapping):
        raise CacheRoutingError(
            "coordinate cache report lacks per-system cached manifests"
        )
    rows_by_system: Dict[str, list[Mapping[str, object]]] = {}
    for row in report_rows:
        system_id = str(row.get("system_id", ""))
        if not system_id:
            raise CacheRoutingError("coordinate cache report row lacks system_id")
        rows_by_system.setdefault(system_id, []).append(row)
    for cached_system in systems:
        if not isinstance(cached_system, Mapping):
            raise CacheRoutingError("cached system manifest contains an invalid system")
        system_id = str(cached_system.get("system_id", ""))
        manifest_row = per_system_manifests.get(system_id)
        if not isinstance(manifest_row, Mapping):
            raise CacheRoutingError(
                f"coordinate cache lacks a per-system manifest for {system_id}"
            )
        manifest_name = str(manifest_row.get(
            "path", coordinate_cache_system_manifest_filename(system_id)
        ))
        per_system_manifest = cached_manifest.parent / manifest_name
        replicas = cached_system.get("replicas")
        if (
            not isinstance(replicas, list)
            or not replicas
            or not isinstance(replicas[0], Mapping)
        ):
            raise CacheRoutingError(
                f"cached system {system_id} has no replicas"
            )
        first_replica_id = str(replicas[0]["replica_id"])
        system_prefix = coordinate_cache_prefix(system_id, first_replica_id)
        system_payload = deepcopy(payload)
        system_payload.update({
            "project_id": f"{project.get('project_id', 'project')}--cache--{system_id}",
            "system_manifest": str(per_system_manifest),
            "reference_structure": str(
                cached_manifest.parent / f"{system_prefix}.pdb"
            ),
            "reference_connectivity": str(
                cached_manifest.parent / f"{system_prefix}.bonds.json"
            ),
        })
        safe_system = hashlib.sha256(system_id.encode("utf-8")).hexdigest()[:10]
        system_output = output.with_name(
            f"{output.stem}--system-{safe_system}{output.suffix}"
        )
        system_sha256 = _write_validated_project(system_output, system_payload)
        system_mappings = []
        for row in rows_by_system.get(system_id, []):
            indices = row.get("source_atom_indices_in_cache_order")
            if isinstance(indices, list):
                system_mappings.append(tuple(int(value) for value in indices))
        per_system_cache_projects.append({
            "system_id": system_id,
            "cache_project": str(system_output),
            "cache_project_sha256": system_sha256,
            "cached_system_manifest": str(per_system_manifest),
            "replica_count": len(replicas),
            "source_index_mapping_count": len(system_mappings),
            "source_index_mappings_identical_within_system": (
                bool(system_mappings)
                and all(value == system_mappings[0] for value in system_mappings[1:])
            ),
        })
    routing.update({
        "cache_validation": validation,
        "cache_project": str(output),
        "cache_project_sha256": pooled_sha256,
        "cache_mapping_policy": (
            "one shared source-to-cache mapping"
            if mapping_is_shared else
            "chemistry-based definitions on a heterogeneous per-replica cache; "
            "explicit source atom-index modules remain on original projects"
        ),
        "source_index_mappings_identical_across_replicas": mapping_is_shared,
        "source_to_cache_atom_mapping_count": (
            len(source_to_cache) if mapping_is_shared else None
        ),
        "per_system_cache_projects": per_system_cache_projects,
    })
    return routing
