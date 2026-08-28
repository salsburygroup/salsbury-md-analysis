"""Frame-aligned, chemically typed interaction fingerprints.

The module is deliberately a report aggregator.  It never rereads coordinates
and never treats a missing upstream frame as an interaction-negative frame.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple

from .context import compile_project_context_file
from .hydrogen_bond_sparse import SparseHydrogenBondError, packed_present_indices
from .manifests import ManifestValidationError, load_json
from .upstream_cache import load_cached_project_report
from .validation import positive_integer


class InteractionFingerprintError(ValueError):
    """Raised when interaction reports cannot be joined without ambiguity."""


FrameKey = Tuple[str, str, str, int]


_SUPPORTED_SOURCES = (
    "hydrogen_bond_discovery",
    "water_mediated_hydrogen_bond_networks",
    "ion_coordination_geometry",
    "ion_atmosphere",
    "multivalent_molecular_bridges",
    "hydration_density_channels",
)


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = (
        definitions.get("interaction_fingerprints")
        if isinstance(definitions, dict) else None
    )
    required = {
        "source_modules", "frame_join_policy", "minimum_feature_occupancy",
        "maximum_features", "maximum_pair_comparisons",
        "minimum_pair_observations", "minimum_cooccurrence_count",
    }
    if (
        not isinstance(raw, dict)
        or set(raw) != required
        or raw.get("frame_join_policy") != "pairwise_complete_observations_v1"
    ):
        raise InteractionFingerprintError(
            "definitions.interaction_fingerprints fields do not match the contract"
        )
    sources = raw["source_modules"]
    if (
        not isinstance(sources, list) or not sources
        or any(not isinstance(value, str) for value in sources)
        or len(set(sources)) != len(sources)
        or not set(sources) <= set(_SUPPORTED_SOURCES)
    ):
        raise InteractionFingerprintError(
            "source_modules must be unique supported interaction modules"
        )
    occupancy = raw["minimum_feature_occupancy"]
    if (
        isinstance(occupancy, bool) or not isinstance(occupancy, (int, float))
        or not math.isfinite(float(occupancy))
        or not 0.0 <= float(occupancy) <= 1.0
    ):
        raise InteractionFingerprintError(
            "minimum_feature_occupancy must be finite and within [0, 1]"
        )
    result = dict(raw)
    result["source_modules"] = tuple(sources)
    result["minimum_feature_occupancy"] = float(occupancy)
    for name in (
        "maximum_features", "maximum_pair_comparisons",
        "minimum_pair_observations", "minimum_cooccurrence_count",
    ):
        result[name] = positive_integer(
            raw[name], name, error_type=InteractionFingerprintError
        )
    return result


def _key(row: Mapping[str, object]) -> FrameKey:
    try:
        source_index = row.get("source_frame_index", row.get("frame_index"))
        values = (
            str(row["system_id"]), str(row["replica_id"]),
            str(row["segment_id"]), int(source_index),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InteractionFingerprintError(
            "upstream frame lacks an exact system/replica/segment/source-frame identity"
        ) from exc
    if values[3] < 0:
        raise InteractionFingerprintError("source_frame_index must be nonnegative")
    return values


def _feature_id(source: str, interaction_type: str, identity: str) -> str:
    return f"{source}|{interaction_type}|{identity}"


def _add_feature(
    dictionary: MutableMapping[str, Dict[str, object]],
    *, source: str, interaction_type: str, identity: str,
    definition: Mapping[str, object],
) -> str:
    feature_id = _feature_id(source, interaction_type, identity)
    row = {
        "feature_id": feature_id,
        "source_module": source,
        "interaction_type": interaction_type,
        "source_identity": identity,
        "definition": dict(definition),
    }
    prior = dictionary.setdefault(feature_id, row)
    if prior != row:
        raise InteractionFingerprintError(
            f"feature identity {feature_id!r} has inconsistent definitions"
        )
    return feature_id


def _hydrogen_bond_features(
    report: Mapping[str, object], dictionary: MutableMapping[str, Dict[str, object]]
) -> Dict[FrameKey, set[str]]:
    candidates = report.get("candidate_dictionary")
    frames = report.get("frame_bond_matrix")
    if not isinstance(candidates, list) or not isinstance(frames, list):
        raise InteractionFingerprintError("hydrogen-bond report is incomplete")
    ids = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise InteractionFingerprintError("hydrogen-bond candidate is not an object")
        identity = str(candidate.get("bond_id", f"candidate-{index}"))
        ids.append(_add_feature(
            dictionary, source="hydrogen_bond_discovery",
            interaction_type="direct_hydrogen_bond", identity=identity,
            definition=candidate,
        ))
    result: Dict[FrameKey, set[str]] = {}
    for frame in frames:
        if not isinstance(frame, dict):
            raise InteractionFingerprintError("hydrogen-bond frame is not an object")
        if frame.get("representation") == "sparse_packed_v2":
            try:
                present = list(packed_present_indices(frame, "primary"))
            except SparseHydrogenBondError as exc:
                raise InteractionFingerprintError(str(exc)) from exc
        elif isinstance(frame.get("primary_present_candidate_indices"), list):
            present = list(frame["primary_present_candidate_indices"])
        elif isinstance(frame.get("cutoff_present_candidate_indices"), dict):
            present = list(frame["cutoff_present_candidate_indices"].get("primary", []))
        elif isinstance(frame.get("binary_values"), list):
            present = [
                index for index, value in enumerate(frame["binary_values"])
                if value == 1
            ]
        else:
            raise InteractionFingerprintError(
                "hydrogen-bond frame has no supported primary-presence representation"
            )
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            or index < 0 or index >= len(ids) for index in present
        ):
            raise InteractionFingerprintError("hydrogen-bond candidate index is invalid")
        result[_key(frame)] = {ids[index] for index in present}
    return result


def _water_features(
    report: Mapping[str, object], dictionary: MutableMapping[str, Dict[str, object]]
) -> Dict[FrameKey, set[str]]:
    frames = report.get("frame_networks")
    if not isinstance(frames, list):
        raise InteractionFingerprintError("water-network report is incomplete")
    result = {}
    for frame in frames:
        if not isinstance(frame, dict):
            raise InteractionFingerprintError("water-network frame is not an object")
        bridges = frame.get("primary_bridge_water_ids")
        if not isinstance(bridges, dict):
            raise InteractionFingerprintError("water-network frame lacks primary bridges")
        present = set()
        for bridge_id, water_ids in sorted(bridges.items()):
            present.add(_add_feature(
                dictionary, source="water_mediated_hydrogen_bond_networks",
                interaction_type="one_water_hydrogen_bond_bridge",
                identity=str(bridge_id),
                definition={"bridge_id": str(bridge_id), "water_identity_policy": "any"},
            ))
        result[_key(frame)] = present
    return result


def _ion_geometry_features(
    report: Mapping[str, object], dictionary: MutableMapping[str, Dict[str, object]]
) -> Dict[FrameKey, set[str]]:
    frames = report.get("frame_reports")
    if not isinstance(frames, list):
        raise InteractionFingerprintError("ion-geometry report is incomplete")
    result = {}
    for frame in frames:
        if not isinstance(frame, dict) or not isinstance(frame.get("ion_sites"), list):
            raise InteractionFingerprintError("ion-geometry frame is malformed")
        present = set()
        for site in frame["ion_sites"]:
            if not isinstance(site, dict) or not isinstance(site.get("bound_ligands"), list):
                raise InteractionFingerprintError("ion site is malformed")
            site_id = str(site.get("site_id", "unnamed"))
            for ligand in site["bound_ligands"]:
                if not isinstance(ligand, dict) or "atom_index" not in ligand:
                    raise InteractionFingerprintError("bound ligand is malformed")
                identity = f"{site_id}:ligand-atom-{int(ligand['atom_index'])}"
                present.add(_add_feature(
                    dictionary, source="ion_coordination_geometry",
                    interaction_type="ion_ligand_coordination", identity=identity,
                    definition={
                        "site_id": site_id,
                        "ion_atom_index": site.get("ion_atom_index"),
                        "ligand_atom_index": int(ligand["atom_index"]),
                        "ion_identity": site.get("ion_identity"),
                        "ligand_identity": ligand.get("atom_identity"),
                    },
                ))
        result[_key(frame)] = present
    return result


def _ion_atmosphere_features(
    report: Mapping[str, object], dictionary: MutableMapping[str, Dict[str, object]]
) -> Dict[FrameKey, set[str]]:
    frames = report.get("frame_records")
    if not isinstance(frames, list):
        raise InteractionFingerprintError("ion-atmosphere report is incomplete")
    result = {}
    for frame in frames:
        if not isinstance(frame, dict) or not isinstance(frame.get("species"), dict):
            raise InteractionFingerprintError("ion-atmosphere frame is malformed")
        present = set()
        for species, species_row in sorted(frame["species"].items()):
            if not isinstance(species_row, dict) or not isinstance(species_row.get("targets"), dict):
                raise InteractionFingerprintError("ion-atmosphere species row is malformed")
            charge = str(species_row.get("charge_class", "unknown"))
            for target_id, target in sorted(species_row["targets"].items()):
                counts = target.get("ion_count_within_shell") if isinstance(target, dict) else None
                if not isinstance(counts, dict):
                    raise InteractionFingerprintError("ion shell row is malformed")
                for cutoff, count in sorted(counts.items(), key=lambda item: float(item[0])):
                    if int(count) <= 0:
                        continue
                    identity = f"{species}:{target_id}:within-{cutoff}-angstrom"
                    present.add(_add_feature(
                        dictionary, source="ion_atmosphere",
                        interaction_type="ion_shell_presence", identity=identity,
                        definition={
                            "species": str(species), "charge_class": charge,
                            "target_id": str(target_id),
                            "cutoff_angstrom": float(cutoff),
                            "presence_rule": "one_or_more_ions",
                        },
                    ))
        result[_key(frame)] = present
    return result


def _multivalent_features(
    report: Mapping[str, object], dictionary: MutableMapping[str, Dict[str, object]]
) -> Dict[FrameKey, set[str]]:
    summaries = report.get("frame_summaries")
    hyperedges = report.get("bridge_hyperedges")
    if not isinstance(summaries, list) or not isinstance(hyperedges, list):
        raise InteractionFingerprintError("multivalent-bridge report is incomplete")
    result = {_key(row): set() for row in summaries if isinstance(row, dict)}
    compact_dictionary = report.get("bridge_feature_dictionary")
    if isinstance(compact_dictionary, list) and all(
        isinstance(row, dict)
        and isinstance(row.get("active_bridge_feature_indices"), list)
        for row in summaries
    ):
        by_index = {}
        for row in compact_dictionary:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("feature_index"), int)
                or not isinstance(row.get("contacted_residue_ids"), list)
            ):
                raise InteractionFingerprintError(
                    "multivalent compact feature dictionary is malformed"
                )
            residues = sorted(str(value) for value in row["contacted_residue_ids"])
            mediator_type = str(row.get("mediator_type", "unknown"))
            by_index[int(row["feature_index"])] = _add_feature(
                dictionary,
                source="multivalent_molecular_bridges",
                interaction_type="multivalent_mediator_bridge",
                identity=mediator_type + ":" + "+".join(residues),
                definition={
                    "mediator_type": mediator_type,
                    "contacted_residue_ids": residues,
                    "mediator_kind": row.get("mediator_kind"),
                },
            )
        for summary in summaries:
            assert isinstance(summary, dict)
            target = result[_key(summary)]
            for index in summary["active_bridge_feature_indices"]:
                if isinstance(index, bool) or not isinstance(index, int) or index not in by_index:
                    raise InteractionFingerprintError(
                        "multivalent frame names an unknown compact feature index"
                    )
                target.add(by_index[index])
        return result
    for edge in hyperedges:
        if not isinstance(edge, dict) or not isinstance(edge.get("contacted_residues"), list):
            raise InteractionFingerprintError("multivalent bridge hyperedge is malformed")
        mediator = edge.get("mediator")
        if not isinstance(mediator, dict):
            raise InteractionFingerprintError("multivalent bridge mediator is malformed")
        residues = sorted(
            str(row.get("residue", {}).get("residue_id"))
            for row in edge["contacted_residues"]
            if isinstance(row, dict) and isinstance(row.get("residue"), dict)
        )
        if len(residues) < 2:
            raise InteractionFingerprintError("multivalent bridge has fewer than two residues")
        mediator_type = str(mediator.get("mediator_type", "unknown"))
        identity = mediator_type + ":" + "+".join(residues)
        feature = _add_feature(
            dictionary, source="multivalent_molecular_bridges",
            interaction_type="multivalent_mediator_bridge", identity=identity,
            definition={
                "mediator_type": mediator_type,
                "contacted_residue_ids": residues,
                "mediator_kind": mediator.get("mediator_kind"),
            },
        )
        result.setdefault(_key(edge), set()).add(feature)
    return result


def _hydration_density_features(
    report: Mapping[str, object], dictionary: MutableMapping[str, Dict[str, object]]
) -> Dict[FrameKey, set[str]]:
    frames = report.get("frame_feature_records")
    components = report.get("density_components")
    if not isinstance(frames, list) or not isinstance(components, list):
        raise InteractionFingerprintError("hydration-density report is incomplete")
    component_by_id = {
        str(row["feature_id"]): row
        for row in components
        if isinstance(row, dict) and isinstance(row.get("feature_id"), str)
    }
    result: Dict[FrameKey, set[str]] = {}
    for frame in frames:
        if not isinstance(frame, dict) or not isinstance(frame.get("active_feature_ids"), list):
            raise InteractionFingerprintError("hydration-density frame is malformed")
        present = set()
        for source_id in frame["active_feature_ids"]:
            component = component_by_id.get(str(source_id))
            if component is None:
                raise InteractionFingerprintError(
                    "hydration-density frame names an unknown density component"
                )
            species = str(component.get("species", "unknown"))
            present.add(_add_feature(
                dictionary, source="hydration_density_channels",
                interaction_type=(
                    "aligned_water_density_component"
                    if species == "water" else "aligned_ion_density_component"
                ),
                identity=str(source_id),
                definition={
                    "species": species,
                    "centroid_angstrom": component.get("centroid_angstrom"),
                    "volume_angstrom3": component.get("volume_angstrom3"),
                    "geometric_channel_candidate": component.get("geometric_channel_candidate"),
                    "presence_rule": "one_or_more_species_atoms_in_component_voxels",
                },
            ))
        result[_key(frame)] = present
    return result


_EXTRACTORS = {
    "hydrogen_bond_discovery": _hydrogen_bond_features,
    "water_mediated_hydrogen_bond_networks": _water_features,
    "ion_coordination_geometry": _ion_geometry_features,
    "ion_atmosphere": _ion_atmosphere_features,
    "multivalent_molecular_bridges": _multivalent_features,
    "hydration_density_channels": _hydration_density_features,
}


def build_interaction_fingerprints(
    reports: Mapping[str, Mapping[str, object]],
    settings: Mapping[str, object],
) -> Dict[str, object]:
    """Build sparse fingerprints with explicit per-source missingness."""

    dictionary: Dict[str, Dict[str, object]] = {}
    source_frames: Dict[str, Dict[FrameKey, set[str]]] = {}
    for source in settings["source_modules"]:  # type: ignore[index]
        report = reports.get(str(source))
        if report is None:
            continue
        if report.get("module_id") != source or report.get("technical_status") != "complete":
            raise InteractionFingerprintError(f"{source} report is not technically complete")
        source_frames[str(source)] = _EXTRACTORS[str(source)](report, dictionary)
    if not source_frames:
        return {
            "availability_status": "not_available",
            "availability_reason": "no_configured_interaction_source_reports",
            "available_source_modules": [],
            "feature_dictionary": [], "frame_fingerprints": [],
            "feature_occupancies": [], "cooccurrence_edges": [],
        }
    if len(dictionary) > int(settings["maximum_features"]):
        raise InteractionFingerprintError(
            f"feature count {len(dictionary)} exceeds maximum_features"
        )
    feature_frames: Dict[str, set[FrameKey]] = defaultdict(set)
    for frames in source_frames.values():
        for frame_key, present in frames.items():
            for feature in present:
                feature_frames[feature].add(frame_key)
    retained = set()
    occupancies = []
    for feature_id, feature in sorted(dictionary.items()):
        source = str(feature["source_module"])
        denominator = len(source_frames[source])
        present = len(feature_frames.get(feature_id, set()))
        occupancy = present / denominator if denominator else 0.0
        if occupancy >= float(settings["minimum_feature_occupancy"]):
            retained.add(feature_id)
            occupancies.append({
                "feature_id": feature_id,
                "source_module": source,
                "evaluated_frame_count": denominator,
                "present_frame_count": present,
                "occupancy_fraction": occupancy,
            })
    if len(retained) > int(settings["maximum_features"]):
        raise InteractionFingerprintError("retained features exceed maximum_features")

    all_keys = sorted(set().union(*(set(rows) for rows in source_frames.values())))
    frame_rows = []
    for frame_key in all_keys:
        available = sorted(source for source, rows in source_frames.items() if frame_key in rows)
        present = sorted(set().union(*(
            source_frames[source][frame_key] for source in available
        )).intersection(retained))
        frame_rows.append({
            "system_id": frame_key[0], "replica_id": frame_key[1],
            "segment_id": frame_key[2], "source_frame_index": frame_key[3],
            "available_source_modules": available,
            "present_feature_ids": present,
        })

    feature_ids = sorted(retained)
    observed_joint_counts: Counter[Tuple[str, str]] = Counter()
    for frame in frame_rows:
        present = frame["present_feature_ids"]
        assert isinstance(present, list)
        observed_joint_counts.update(combinations(present, 2))
        if len(observed_joint_counts) > int(settings["maximum_pair_comparisons"]):
            raise InteractionFingerprintError(
                "distinct observed feature pairs exceed maximum_pair_comparisons"
            )
    cooccurrence = []
    source_key_sets = {source: set(rows) for source, rows in source_frames.items()}
    for (left, right), observed_joint_count in sorted(observed_joint_counts.items()):
        if observed_joint_count < int(settings["minimum_cooccurrence_count"]):
            continue
        left_source = str(dictionary[left]["source_module"])
        right_source = str(dictionary[right]["source_module"])
        common = source_key_sets[left_source].intersection(source_key_sets[right_source])
        if len(common) < int(settings["minimum_pair_observations"]):
            continue
        left_set = feature_frames.get(left, set()).intersection(common)
        right_set = feature_frames.get(right, set()).intersection(common)
        joint = left_set.intersection(right_set)
        if len(joint) != observed_joint_count:
            raise InteractionFingerprintError(
                "co-occurrence accumulator disagrees with exact frame identity sets"
            )
        union = left_set.union(right_set)
        denominator = len(common)
        p_left = len(left_set) / denominator
        p_right = len(right_set) / denominator
        p_joint = len(joint) / denominator
        variance = p_left * (1.0 - p_left) * p_right * (1.0 - p_right)
        phi = (p_joint - p_left * p_right) / math.sqrt(variance) if variance > 0 else None
        cooccurrence.append({
            "feature_i": left, "feature_j": right,
            "interaction_type_i": dictionary[left]["interaction_type"],
            "interaction_type_j": dictionary[right]["interaction_type"],
            "pairwise_complete_frame_count": denominator,
            "cooccurrence_frame_count": len(joint),
            "jaccard": len(joint) / len(union) if union else 0.0,
            "p_j_given_i": len(joint) / len(left_set) if left_set else None,
            "p_i_given_j": len(joint) / len(right_set) if right_set else None,
            "phi_correlation": phi,
        })

    coverage = []
    for source, rows in sorted(source_frames.items()):
        coverage.append({
            "source_module": source,
            "evaluated_frame_count": len(rows),
            "feature_count": sum(
                dictionary[feature]["source_module"] == source for feature in retained
            ),
        })
    return {
        "availability_status": "available",
        "availability_reason": None,
        "available_source_modules": sorted(source_frames),
        "source_coverage": coverage,
        "feature_dictionary": [dictionary[key] for key in feature_ids],
        "frame_fingerprints": frame_rows,
        "feature_occupancies": occupancies,
        "cooccurrence_edges": cooccurrence,
        "frame_join_contract": (
            "union of exact frame identities with per-source availability; occupancies "
            "use source-specific denominators and feature pairs use only frames observed "
            "by both source modules"
        ),
    }


def interaction_fingerprints_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    reports = {}
    for module_id in settings["source_modules"]:  # type: ignore[index]
        report = load_cached_project_report(
            str(module_id), source, hash_content=hash_content,
            error_type=InteractionFingerprintError,
        )
        if report is not None:
            reports[str(module_id)] = report
    context = compile_project_context_file(source, hash_content=hash_content)
    result = build_interaction_fingerprints(reports, settings)
    issues = [issue for issue in context.get("issues", []) if isinstance(issue, dict)]
    if result["availability_status"] == "not_available":
        issues.append({
            "severity": "warning", "code": "INTERACTION_FINGERPRINTS_NOT_AVAILABLE",
            "message": "No configured complete interaction source report was supplied.",
        })
    return {
        "module_id": "interaction_fingerprints",
        "technical_status": "complete", "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": context["system_manifest_path"],
        "system_manifest_sha256": context["system_manifest_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "settings": settings, **result,
        "evaluated_frame_count": len(result["frame_fingerprints"]),
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Interactions remain operational geometric or chemical definitions inherited from each source report.",
            "Missing upstream frames are explicit missing observations and are never encoded as absent interactions.",
            "Co-occurrence and phi correlation are descriptive associations, not energetic coupling, causality, or mechanism.",
            "Feature occupancies and frame counts are not independent-replica uncertainty.",
        ],
    }


def interaction_fingerprints_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return interaction_fingerprints_project(project_path, hash_content=hash_content)
    except (
        InteractionFingerprintError, ManifestValidationError, OSError,
        KeyError, TypeError, ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "interaction_fingerprints",
            "technical_status": "failed", "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [{
                "severity": "error", "code": "INTERACTION_FINGERPRINTS_INVALID",
                "message": message,
            } for message in messages],
        }
