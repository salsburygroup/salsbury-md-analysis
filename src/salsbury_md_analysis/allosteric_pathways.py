"""Experimental contact-occupancy allosteric communication pathways."""

from __future__ import annotations

import heapq
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .atom_mapping import AtomMappingError, AtomRecord, read_topology_atoms
from .context import compile_project_context_file
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .frame_sampling import (
    frame_selected, normalize_frame_selection, plan_frame_selection,
    reader_frame_indices,
)
from .geometry import GeometryError, apply_transform, best_fit_transform
from .manifests import (
    ManifestValidationError, load_json, resolve_manifest_path, sha256_file,
)
from .moments import DisplacementCovariance, MomentError
from .periodic import PeriodicFrameProcessor, PeriodicReconstructionError
from .reporting import atom_identity_record
from .selections import AtomCorrespondence, build_common_correspondences
from .trajectory_contracts import require_periodic_policy
from .validation import positive_integer


class AllostericPathwayError(ValueError):
    """Raised when a physical-network pathway contract is invalid."""


Adjacency = List[List[Tuple[int, float, float]]]


def _numeric_symmetric_matrix(
    matrix: Sequence[Sequence[float]], name: str, maximum_nodes: int,
    *, lower: float | None = None, upper: float | None = None,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[0] != values.shape[1]
        or values.shape[0] > maximum_nodes
    ):
        raise AllostericPathwayError(
            f"{name} must be a square matrix with 2..maximum_nodes rows"
        )
    if not np.isfinite(values).all():
        raise AllostericPathwayError(f"{name} contains non-finite values")
    scale = max(1.0, float(np.max(np.abs(values))))
    if not np.allclose(values, values.T, rtol=1.0e-10, atol=1.0e-12 * scale):
        raise AllostericPathwayError(f"{name} must be symmetric")
    if lower is not None and np.any(values < lower):
        raise AllostericPathwayError(f"{name} contains values below {lower}")
    if upper is not None and np.any(values > upper):
        raise AllostericPathwayError(f"{name} contains values above {upper}")
    return values


def _network(
    contact_occupancy_matrix: Sequence[Sequence[float]],
    minimum_contact_occupancy: float,
    distance_epsilon: float,
    maximum_nodes: int,
) -> Tuple[np.ndarray, Adjacency, List[Dict[str, object]]]:
    occupancy = _numeric_symmetric_matrix(
        contact_occupancy_matrix, "contact_occupancy_matrix", maximum_nodes,
        lower=0.0, upper=1.0,
    )
    if (
        isinstance(minimum_contact_occupancy, bool)
        or not isinstance(minimum_contact_occupancy, (int, float))
        or not math.isfinite(float(minimum_contact_occupancy))
        or not 0.0 <= float(minimum_contact_occupancy) <= 1.0
    ):
        raise AllostericPathwayError(
            "minimum_contact_occupancy must be between zero and one"
        )
    if (
        isinstance(distance_epsilon, bool)
        or not isinstance(distance_epsilon, (int, float))
        or not math.isfinite(float(distance_epsilon))
        or not 0.0 < float(distance_epsilon) < 1.0
    ):
        raise AllostericPathwayError("distance_epsilon must be between zero and one")
    size = len(occupancy)
    adjacency: Adjacency = [[] for _ in range(size)]
    edges = []
    for left in range(size - 1):
        for right in range(left + 1, size):
            weight = float(occupancy[left, right])
            if weight <= 0.0 or weight < float(minimum_contact_occupancy):
                continue
            distance = -math.log(max(weight, float(distance_epsilon))) + float(
                distance_epsilon
            )
            adjacency[left].append((right, distance, weight))
            adjacency[right].append((left, distance, weight))
            edges.append({
                "node_i": left,
                "node_j": right,
                "contact_occupancy": weight,
                "path_distance": distance,
            })
    for neighbors in adjacency:
        neighbors.sort()
    return occupancy, adjacency, edges


def _dijkstra(
    adjacency: Adjacency, source: int, tolerance: float
) -> Tuple[List[float], List[int], List[List[int]], List[int]]:
    size = len(adjacency)
    distances = [math.inf] * size
    path_counts = [0] * size
    predecessors: List[List[int]] = [[] for _ in range(size)]
    canonical_parent = [-1] * size
    distances[source] = 0.0
    path_counts[source] = 1
    queue = [(0.0, source)]
    order = []
    settled = [False] * size
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances[node] + tolerance or settled[node]:
            continue
        settled[node] = True
        order.append(node)
        for neighbor, edge_distance, _ in adjacency[node]:
            candidate = distance + edge_distance
            if candidate < distances[neighbor] - tolerance:
                distances[neighbor] = candidate
                path_counts[neighbor] = path_counts[node]
                predecessors[neighbor] = [node]
                canonical_parent[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
            elif math.isclose(
                candidate, distances[neighbor], rel_tol=tolerance, abs_tol=tolerance
            ):
                path_counts[neighbor] += path_counts[node]
                predecessors[neighbor].append(node)
                predecessors[neighbor].sort()
                if canonical_parent[neighbor] < 0 or node < canonical_parent[neighbor]:
                    canonical_parent[neighbor] = node
    return distances, path_counts, predecessors, canonical_parent


def _canonical_path(parent: Sequence[int], source: int, target: int) -> List[int]:
    if source == target:
        return [source]
    path = [target]
    while path[-1] != source:
        predecessor = parent[path[-1]]
        if predecessor < 0:
            return []
        path.append(predecessor)
    return list(reversed(path))


def weighted_betweenness_centrality(
    adjacency: Adjacency, equality_tolerance: float = 1.0e-12
) -> List[float]:
    """Return normalized undirected weighted Brandes centrality."""

    size = len(adjacency)
    centrality = [0.0] * size
    for source in range(size):
        distances, sigma, predecessors, _ = _dijkstra(
            adjacency, source, equality_tolerance
        )
        order = sorted(
            (node for node in range(size) if math.isfinite(distances[node])),
            key=lambda node: (distances[node], node),
        )
        dependency = [0.0] * size
        for node in reversed(order):
            if sigma[node]:
                coefficient = (1.0 + dependency[node]) / sigma[node]
                for predecessor in predecessors[node]:
                    dependency[predecessor] += sigma[predecessor] * coefficient
            if node != source:
                centrality[node] += dependency[node]
    centrality = [value / 2.0 for value in centrality]
    denominator = (size - 1) * (size - 2) / 2.0
    if denominator > 0.0:
        centrality = [value / denominator for value in centrality]
    return centrality


def _minmax(values: Sequence[float]) -> List[float]:
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum, rel_tol=1.0e-15, abs_tol=1.0e-15):
        return [0.0] * len(values)
    return [(value - minimum) / (maximum - minimum) for value in values]


def allosteric_pathway_network(
    contact_occupancy_matrix: Sequence[Sequence[float]],
    source_node_indices: Sequence[int],
    sink_node_indices: Sequence[int],
    *,
    minimum_contact_occupancy: float = 0.5,
    distance_epsilon: float = 1.0e-12,
    shortest_path_equality_tolerance: float = 1.0e-12,
    maximum_nodes: int = 2000,
    dependency_matrix: Sequence[Sequence[float]] | None = None,
) -> Dict[str, object]:
    """Build contact-frequency pathways and optional local dependency scores."""

    maximum_nodes = positive_integer(maximum_nodes, "maximum_nodes")
    occupancy, adjacency, edges = _network(
        contact_occupancy_matrix, minimum_contact_occupancy,
        distance_epsilon, maximum_nodes,
    )
    size = len(occupancy)
    sources = tuple(source_node_indices)
    sinks = tuple(sink_node_indices)
    for values, name in ((sources, "source_node_indices"), (sinks, "sink_node_indices")):
        if (
            not values
            or len(set(values)) != len(values)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
            or any(value < 0 or value >= size for value in values)
        ):
            raise AllostericPathwayError(
                f"{name} must contain unique zero-based node indices"
            )
    if set(sources).intersection(sinks):
        raise AllostericPathwayError("source and sink node sets must not overlap")
    if (
        isinstance(shortest_path_equality_tolerance, bool)
        or not isinstance(shortest_path_equality_tolerance, (int, float))
        or not math.isfinite(float(shortest_path_equality_tolerance))
        or float(shortest_path_equality_tolerance) <= 0.0
    ):
        raise AllostericPathwayError(
            "shortest_path_equality_tolerance must be finite and positive"
        )
    tolerance = float(shortest_path_equality_tolerance)

    pair_reports = []
    node_usage = np.zeros(size, dtype=float)
    internal_node_usage = np.zeros(size, dtype=float)
    edge_usage = {
        (int(edge["node_i"]), int(edge["node_j"])): 0.0 for edge in edges
    }
    reachable_pairs = 0
    for source in sources:
        distances, counts_from_source, _, parent = _dijkstra(
            adjacency, source, tolerance
        )
        for sink in sinks:
            if not math.isfinite(distances[sink]) or counts_from_source[sink] == 0:
                pair_reports.append({
                    "source_node_index": source,
                    "sink_node_index": sink,
                    "reachable": False,
                    "path_distance": None,
                    "equal_shortest_path_count": 0,
                    "canonical_shortest_path": [],
                })
                continue
            distances_to_sink, counts_to_sink, _, _ = _dijkstra(
                adjacency, sink, tolerance
            )
            total_paths = counts_from_source[sink]
            reachable_pairs += 1
            for node in range(size):
                if math.isclose(
                    distances[node] + distances_to_sink[node], distances[sink],
                    rel_tol=tolerance, abs_tol=tolerance,
                ):
                    fraction = (
                        counts_from_source[node] * counts_to_sink[node] / total_paths
                    )
                    node_usage[node] += fraction
                    if node not in {source, sink}:
                        internal_node_usage[node] += fraction
            for left, right in edge_usage:
                edge_distance = next(
                    distance for neighbor, distance, _ in adjacency[left]
                    if neighbor == right
                )
                paths_through = 0
                if math.isclose(
                    distances[left] + edge_distance + distances_to_sink[right],
                    distances[sink], rel_tol=tolerance, abs_tol=tolerance,
                ):
                    paths_through += counts_from_source[left] * counts_to_sink[right]
                if math.isclose(
                    distances[right] + edge_distance + distances_to_sink[left],
                    distances[sink], rel_tol=tolerance, abs_tol=tolerance,
                ):
                    paths_through += counts_from_source[right] * counts_to_sink[left]
                edge_usage[(left, right)] += paths_through / total_paths
            pair_reports.append({
                "source_node_index": source,
                "sink_node_index": sink,
                "reachable": True,
                "path_distance": distances[sink],
                "equal_shortest_path_count": total_paths,
                "canonical_shortest_path": _canonical_path(parent, source, sink),
            })
    requested_pairs = len(sources) * len(sinks)
    if reachable_pairs:
        node_usage /= reachable_pairs
        internal_node_usage /= reachable_pairs
        edge_usage = {
            edge: value / reachable_pairs for edge, value in edge_usage.items()
        }
    betweenness = weighted_betweenness_centrality(adjacency, tolerance)
    report: Dict[str, object] = {
        "node_count": size,
        "edge_count": len(edges),
        "source_node_indices": list(sources),
        "sink_node_indices": list(sinks),
        "minimum_contact_occupancy": float(minimum_contact_occupancy),
        "distance_transform": "negative natural logarithm of contact occupancy plus distance_epsilon",
        "distance_epsilon": float(distance_epsilon),
        "shortest_path_equality_tolerance": tolerance,
        "edges": edges,
        "source_sink_paths": pair_reports,
        "requested_source_sink_pair_count": requested_pairs,
        "reachable_source_sink_pair_count": reachable_pairs,
        "all_source_sink_pairs_connected": reachable_pairs == requested_pairs,
        "node_shortest_path_participation": node_usage.tolist(),
        "internal_node_shortest_path_participation": internal_node_usage.tolist(),
        "edge_shortest_path_participation": [
            {"node_i": left, "node_j": right, "participation": value}
            for (left, right), value in edge_usage.items()
        ],
        "weighted_betweenness_centrality": betweenness,
        "path_participation_contract": "fraction across all equal shortest paths, averaged over reachable declared source-sink pairs",
    }
    if dependency_matrix is not None:
        dependency = _numeric_symmetric_matrix(
            dependency_matrix, "dependency_matrix", maximum_nodes
        )
        if dependency.shape != occupancy.shape:
            raise AllostericPathwayError(
                "dependency_matrix shape must match contact_occupancy_matrix"
            )
        factors = []
        for node, neighbors in enumerate(adjacency):
            denominator = sum(weight for _, _, weight in neighbors)
            factors.append(
                sum(
                    weight * abs(float(dependency[node, neighbor]))
                    for neighbor, _, weight in neighbors
                ) / denominator
                if denominator else 0.0
            )
        normalized_betweenness = _minmax(betweenness)
        normalized_factors = _minmax(factors)
        report.update({
            "neighbor_correlation_factor": factors,
            "neighbor_correlation_factor_contract": "contact-occupancy-weighted mean absolute dependency with retained physical neighbors",
            "normalized_weighted_betweenness": normalized_betweenness,
            "normalized_neighbor_correlation_factor": normalized_factors,
            "combined_allosteric_score": [
                (centrality + factor) / 2.0
                for centrality, factor in zip(
                    normalized_betweenness, normalized_factors
                )
            ],
            "combined_score_contract": "arithmetic mean of independently min-max normalized weighted betweenness and neighbor-correlation factor",
        })
    return report


_TRAJECTORY_FIELDS = {
    "node_selection", "alignment_selection", "minimum_reference_coverage",
    "frame_stride", "frame_selection", "contact_cutoff_angstrom",
    "minimum_sequence_separation", "minimum_evaluated_frames_per_system",
    "minimum_variance_angstrom2",
}
_COMMON_FIELDS = {
    "network_source", "network_path", "source_node_indices", "sink_node_indices",
    "minimum_contact_occupancy", "distance_epsilon",
    "shortest_path_equality_tolerance", "maximum_nodes",
    "neighbor_correlation_factor_enabled",
}


def _finite_number(
    value: object, name: str, *, minimum: float | None = None,
    maximum: float | None = None, strictly_positive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise AllostericPathwayError(f"{name} must be finite")
    number = float(value)
    if strictly_positive and number <= 0.0:
        raise AllostericPathwayError(f"{name} must be positive")
    if minimum is not None and number < minimum:
        raise AllostericPathwayError(f"{name} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise AllostericPathwayError(f"{name} must be at most {maximum}")
    return number


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("allosteric_pathways") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict):
        raise AllostericPathwayError("definitions.allosteric_pathways must be an object")
    required = _COMMON_FIELDS | _TRAJECTORY_FIELDS
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required))
    if missing or unknown:
        raise AllostericPathwayError(
            "allosteric-pathway settings mismatch; missing="
            + ",".join(missing) + "; unknown=" + ",".join(unknown)
        )
    if raw["network_source"] not in {"trajectory", "external_json"}:
        raise AllostericPathwayError(
            "network_source must be trajectory or external_json"
        )
    if not isinstance(raw["network_path"], str):
        raise AllostericPathwayError("network_path must be a string")
    if raw["network_source"] == "external_json" and not raw["network_path"].strip():
        raise AllostericPathwayError(
            "network_path must be nonempty when network_source is external_json"
        )
    if raw["network_source"] == "trajectory" and raw["network_path"].strip():
        raise AllostericPathwayError(
            "network_path must be empty when network_source is trajectory"
        )
    for field in ("source_node_indices", "sink_node_indices"):
        if not isinstance(raw[field], list):
            raise AllostericPathwayError(f"{field} must be an array")
    positive_integer(raw["maximum_nodes"], "maximum_nodes")
    if not isinstance(raw["neighbor_correlation_factor_enabled"], bool):
        raise AllostericPathwayError(
            "neighbor_correlation_factor_enabled must be boolean"
        )
    for field in ("node_selection", "alignment_selection"):
        if not isinstance(raw[field], str) or not raw[field].strip():
            raise AllostericPathwayError(f"{field} must be a nonempty selection name")
    _finite_number(
        raw["minimum_reference_coverage"], "minimum_reference_coverage",
        minimum=0.0, maximum=1.0,
    )
    positive_integer(raw["frame_stride"], "frame_stride")
    positive_integer(
        raw["minimum_evaluated_frames_per_system"],
        "minimum_evaluated_frames_per_system",
    )
    _finite_number(
        raw["contact_cutoff_angstrom"], "contact_cutoff_angstrom",
        strictly_positive=True,
    )
    positive_integer(raw["minimum_sequence_separation"], "minimum_sequence_separation")
    _finite_number(
        raw["minimum_variance_angstrom2"], "minimum_variance_angstrom2",
        strictly_positive=True,
    )
    _finite_number(
        raw["minimum_contact_occupancy"], "minimum_contact_occupancy",
        minimum=0.0, maximum=1.0,
    )
    _finite_number(
        raw["distance_epsilon"], "distance_epsilon", strictly_positive=True,
    )
    _finite_number(
        raw["shortest_path_equality_tolerance"],
        "shortest_path_equality_tolerance", strictly_positive=True,
    )
    result = dict(raw)
    result["frame_selection"] = normalize_frame_selection(
        raw["frame_selection"], int(raw["frame_stride"]),
        error_type=AllostericPathwayError,
    )
    return result


def _network_payload(path: Path, ncf_enabled: bool) -> Dict[str, object]:
    payload = load_json(path)
    allowed = {
        "network_schema", "nodes", "contact_occupancy_matrix", "dependency_matrix"
    }
    required = allowed.difference({"dependency_matrix"})
    if not required.issubset(payload) or set(payload).difference(allowed):
        raise AllostericPathwayError("network file fields are invalid")
    if payload["network_schema"] != "salsbury-residue-contact-network-v1":
        raise AllostericPathwayError("network_schema is unsupported")
    nodes = payload["nodes"]
    if not isinstance(nodes, list) or not nodes:
        raise AllostericPathwayError("nodes must be a nonempty array")
    node_ids = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or not isinstance(node.get("node_id"), str) or not node["node_id"].strip():
            raise AllostericPathwayError(f"nodes[{index}].node_id must be nonempty")
        node_ids.append(node["node_id"])
    if len(set(node_ids)) != len(node_ids):
        raise AllostericPathwayError("node_id values must be unique")
    matrix = payload["contact_occupancy_matrix"]
    if not isinstance(matrix, list) or len(matrix) != len(nodes):
        raise AllostericPathwayError(
            "contact_occupancy_matrix row count must match nodes"
        )
    if ncf_enabled and "dependency_matrix" not in payload:
        raise AllostericPathwayError(
            "neighbor correlation factor requires dependency_matrix"
        )
    return payload


def _coordinates_at(
    coordinates: Sequence[Tuple[float, float, float]], indices: Sequence[int]
) -> Tuple[Tuple[float, float, float], ...]:
    try:
        return tuple(coordinates[index] for index in indices)
    except IndexError as exc:
        raise AllostericPathwayError(
            "atom correspondence index exceeds coordinate atom count"
        ) from exc


def _mapping_sets(
    reference_atoms: Sequence[AtomRecord],
    target_atom_sets: Sequence[Sequence[AtomRecord]],
    selections: Mapping[str, object],
    settings: Mapping[str, object],
    policy: str,
) -> Tuple[Dict[str, AtomCorrespondence], ...]:
    results: List[Dict[str, AtomCorrespondence]] = [{} for _ in target_atom_sets]
    for role, field in (("alignment", "alignment_selection"), ("node", "node_selection")):
        selection_id = str(settings[field])
        definition = selections.get(selection_id)
        if not isinstance(definition, dict):
            raise AllostericPathwayError(
                f"{field} names undefined selection {selection_id!r}"
            )
        mappings = build_common_correspondences(
            reference_atoms,
            target_atom_sets,
            definition,
            selection_id,
            policy,
            float(settings["minimum_reference_coverage"]),
        )
        for result, mapping in zip(results, mappings):
            result[role] = mapping
    return tuple(results)


def _residue_nodes(atoms: Sequence[AtomRecord]) -> List[Dict[str, object]]:
    residue_keys = [
        (atom.chain_id, atom.residue_number, atom.insertion_code)
        for atom in atoms
    ]
    if len(set(residue_keys)) != len(residue_keys):
        raise AllostericPathwayError(
            "node_selection must resolve to exactly one representative atom per residue"
        )
    nodes = []
    for index, atom in enumerate(atoms):
        identity = atom_identity_record(atom, index)
        identity.update({
            "node_index": index,
            "node_id": (
                f"{atom.chain_id or '_'}:{atom.residue_number}"
                f"{atom.insertion_code}:{atom.residue_name}"
            ),
        })
        nodes.append(identity)
    return nodes


def _sequence_exclusion_mask(
    atoms: Sequence[AtomRecord], minimum_sequence_separation: int
) -> np.ndarray:
    """Return pairs excluded because they are close along one declared chain."""

    positions: Dict[Tuple[str, int, str], int] = {}
    next_by_chain: Dict[str, int] = {}
    for atom in atoms:
        key = (atom.chain_id, atom.residue_number, atom.insertion_code)
        positions[key] = next_by_chain.get(atom.chain_id, 0)
        next_by_chain[atom.chain_id] = positions[key] + 1
    size = len(atoms)
    excluded = np.eye(size, dtype=bool)
    for left in range(size - 1):
        left_atom = atoms[left]
        left_position = positions[
            (left_atom.chain_id, left_atom.residue_number, left_atom.insertion_code)
        ]
        for right in range(left + 1, size):
            right_atom = atoms[right]
            if left_atom.chain_id != right_atom.chain_id:
                continue
            right_position = positions[
                (right_atom.chain_id, right_atom.residue_number, right_atom.insertion_code)
            ]
            if abs(left_position - right_position) < minimum_sequence_separation:
                excluded[left, right] = excluded[right, left] = True
    return excluded


def _dependency_matrix(
    state: DisplacementCovariance, minimum_variance: float
) -> List[List[float]]:
    matrix = state.correlation_matrix(minimum_variance)
    if any(value is None for row in matrix for value in row):
        raise AllostericPathwayError(
            "trajectory dependency matrix contains undefined zero-variance nodes; "
            "adjust the node selection or minimum_variance_angstrom2"
        )
    return [[float(value) for value in row] for row in matrix]


def _trajectory_network_payloads(
    project: Mapping[str, object], source: Path, settings: Mapping[str, object],
    *, hash_content: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    """Build per-system residue-contact occupancies from selected trajectory frames."""

    context = compile_project_context_file(source, hash_content=hash_content)
    contract = context["contract"]
    assert isinstance(contract, dict)
    selections = contract["selections"]
    units = contract["units"]
    assert isinstance(selections, dict) and isinstance(units, dict)
    coordinate_unit = str(units["coordinates"])
    periodic_policy = require_periodic_policy(
        contract.get("periodic_coordinate_policy")
    )
    reference_value = project.get("reference_structure")
    policy_value = project.get("common_atom_policy")
    if not isinstance(reference_value, str) or not reference_value.strip():
        raise AllostericPathwayError(
            "reference_structure is required for trajectory-derived pathways"
        )
    if not isinstance(policy_value, str) or not policy_value.strip():
        raise AllostericPathwayError(
            "common_atom_policy is required for trajectory-derived pathways"
        )
    reference_path = resolve_manifest_path(reference_value, source)
    _, reference_atoms = read_topology_atoms(reference_path)
    try:
        raw_reference_frame = next(
            iter_coordinate_frames(reference_path, coordinate_unit)
        )
    except StopIteration as exc:
        raise AllostericPathwayError(
            "reference_structure contains no coordinate frame"
        ) from exc
    reference_processor = PeriodicFrameProcessor.from_reference(
        project, source, len(reference_atoms)
    )
    reference_frame = reference_processor.process(
        raw_reference_frame, str(reference_path)
    )
    if reference_frame.atom_count != len(reference_atoms):
        raise AllostericPathwayError(
            f"reference coordinate count {reference_frame.atom_count} does not "
            f"match reference topology count {len(reference_atoms)}"
        )

    system_path = Path(str(context["system_manifest_path"]))
    system_manifest = load_json(system_path)
    systems = system_manifest["systems"]
    assert isinstance(systems, list)
    frame_plan, frame_report = plan_frame_selection(
        system_manifest,
        system_path,
        coordinate_unit,
        settings["frame_selection"],  # type: ignore[arg-type]
        frame_stride=int(settings["frame_stride"]),
        error_type=AllostericPathwayError,
    )

    topology_records: List[Dict[str, object]] = []
    for system in systems:
        assert isinstance(system, dict)
        for replica in system["replicas"]:
            assert isinstance(replica, dict)
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology_path)
            topology_records.append({
                "key": (str(system["system_id"]), str(replica["replica_id"])),
                "path": topology_path,
                "atoms": atoms,
            })
    mappings = _mapping_sets(
        reference_atoms,
        [record["atoms"] for record in topology_records],  # type: ignore[list-item]
        selections,
        settings,
        str(policy_value),
    )
    topology_by_key: Dict[Tuple[str, str], Dict[str, object]] = {}
    for record, mapping in zip(topology_records, mappings):
        record["mappings"] = mapping
        topology_by_key[record["key"]] = record  # type: ignore[index]
    node_atoms = mappings[0]["node"].reference_atoms
    nodes = _residue_nodes(node_atoms)
    node_count = len(nodes)
    if node_count < 2 or node_count > int(settings["maximum_nodes"]):
        raise AllostericPathwayError(
            f"trajectory node selection contains {node_count} nodes; expected 2.."
            f"{settings['maximum_nodes']}"
        )
    excluded = _sequence_exclusion_mask(
        node_atoms, int(settings["minimum_sequence_separation"])
    )
    cutoff2 = float(settings["contact_cutoff_angstrom"]) ** 2
    minimum_frames = int(settings["minimum_evaluated_frames_per_system"])
    system_payloads: List[Dict[str, object]] = []
    for system in systems:
        assert isinstance(system, dict)
        system_id = str(system["system_id"])
        contact_counts = np.zeros((node_count, node_count), dtype=np.int64)
        covariance = DisplacementCovariance(node_count)
        segment_reports: List[Dict[str, object]] = []
        for replica in system["replicas"]:
            assert isinstance(replica, dict)
            replica_id = str(replica["replica_id"])
            topology = topology_by_key[(system_id, replica_id)]
            target_atoms = topology["atoms"]
            mapping = topology["mappings"]
            assert isinstance(target_atoms, list) and isinstance(mapping, dict)
            alignment = mapping["alignment"]
            node_mapping = mapping["node"]
            assert isinstance(alignment, AtomCorrespondence)
            assert isinstance(node_mapping, AtomCorrespondence)
            processor = PeriodicFrameProcessor.from_replica(
                project, replica, system_path, len(target_atoms)
            )
            reconstruction_indices = tuple(sorted(
                set(alignment.target_indices) | set(node_mapping.target_indices)
            ))
            for segment in replica["segments"]:
                assert isinstance(segment, dict)
                segment_id = str(segment["segment_id"])
                trajectory_path = resolve_manifest_path(
                    str(segment["trajectory"]), system_path
                )
                selected_indices = frame_plan[(system_id, replica_id, segment_id)]
                evaluated = 0
                processor.begin_segment(
                    bool(segment.get("continuous_with_previous", False))
                )
                for raw_frame in iter_coordinate_frames(
                    trajectory_path,
                    coordinate_unit,
                    reader_frame_indices(selected_indices, processor.policy),
                ):
                    selected = frame_selected(
                        raw_frame.frame_index,
                        selected_indices,
                        int(settings["frame_stride"]),
                    )
                    if not selected and processor.policy != "unwrap_continuous":
                        continue
                    frame = processor.process(
                        raw_frame,
                        f"{system_id}/{replica_id}/{segment_id}/frame-{raw_frame.frame_index}",
                        reconstruction_indices,
                    )
                    if not selected:
                        continue
                    node_coordinates = np.asarray(
                        _coordinates_at(
                            frame.coordinates_angstrom, node_mapping.target_indices
                        ),
                        dtype=float,
                    )
                    delta = node_coordinates[:, None, :] - node_coordinates[None, :, :]
                    contacts = np.einsum("ijk,ijk->ij", delta, delta) <= cutoff2
                    contacts[excluded] = False
                    contact_counts += contacts.astype(np.int64)
                    transform = best_fit_transform(
                        _coordinates_at(
                            frame.coordinates_angstrom, alignment.target_indices
                        ),
                        _coordinates_at(
                            reference_frame.coordinates_angstrom,
                            alignment.reference_indices,
                        ),
                    )
                    covariance.update(apply_transform(
                        _coordinates_at(
                            frame.coordinates_angstrom, node_mapping.target_indices
                        ),
                        transform,
                    ))
                    evaluated += 1
                segment_reports.append({
                    "replica_id": replica_id,
                    "segment_id": segment_id,
                    "trajectory_path": str(trajectory_path),
                    "trajectory_sha256": (
                        sha256_file(trajectory_path) if hash_content else None
                    ),
                    "evaluated_frame_count": evaluated,
                })
        if covariance.count < minimum_frames:
            raise AllostericPathwayError(
                f"system {system_id} produced {covariance.count} selected frames; "
                f"minimum_evaluated_frames_per_system is {minimum_frames}"
            )
        occupancy = contact_counts.astype(float) / covariance.count
        dependency = (
            _dependency_matrix(
                covariance, float(settings["minimum_variance_angstrom2"])
            )
            if settings["neighbor_correlation_factor_enabled"] else None
        )
        system_payloads.append({
            "system_id": system_id,
            "network_schema": "salsbury-residue-contact-network-v1",
            "nodes": nodes,
            "contact_occupancy_matrix": occupancy.tolist(),
            "dependency_matrix": dependency,
            "selected_physical_frame_count": covariance.count,
            "segments": segment_reports,
        })
    derivation = {
        "network_source": "trajectory",
        "contact_definition": "representative-atom distance occupancy",
        "contact_cutoff_angstrom": float(settings["contact_cutoff_angstrom"]),
        "node_selection": str(settings["node_selection"]),
        "alignment_selection_for_dependency": str(settings["alignment_selection"]),
        "minimum_sequence_separation": int(settings["minimum_sequence_separation"]),
        "sequence_exclusion_contract": (
            "within-chain node pairs separated by fewer than the configured number "
            "of residue positions are excluded"
        ),
        "dependency_definition": (
            "trajectory DCCM of fitted representative-atom displacement vectors"
            if settings["neighbor_correlation_factor_enabled"] else None
        ),
        "frame_selection": frame_report,
        "periodic_coordinate_policy": periodic_policy,
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "input_content_signature_sha256": context[
            "input_content_signature_sha256"
        ],
        "content_hashes_included": bool(context["content_hashes_included"]),
    }
    return derivation, nodes, system_payloads


def allosteric_pathways_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    network_source = str(settings["network_source"])
    if network_source == "external_json":
        network_path = resolve_manifest_path(str(settings["network_path"]), source)
        payloads = [{
            "system_id": None,
            **_network_payload(
                network_path, bool(settings["neighbor_correlation_factor_enabled"])
            ),
            "selected_physical_frame_count": 0,
        }]
        nodes = payloads[0]["nodes"]
        derivation: Dict[str, object] = {
            "network_source": "external_json",
            "network_path": str(network_path),
            "network_sha256": sha256_file(network_path),
        }
    else:
        derivation, nodes, payloads = _trajectory_network_payloads(
            project, source, settings, hash_content=hash_content
        )
    system_reports = []
    issues: List[Dict[str, object]] = []
    for payload in payloads:
        network = allosteric_pathway_network(
            payload["contact_occupancy_matrix"],  # type: ignore[arg-type]
            settings["source_node_indices"],  # type: ignore[arg-type]
            settings["sink_node_indices"],  # type: ignore[arg-type]
            minimum_contact_occupancy=float(settings["minimum_contact_occupancy"]),
            distance_epsilon=float(settings["distance_epsilon"]),
            shortest_path_equality_tolerance=float(
                settings["shortest_path_equality_tolerance"]
            ),
            maximum_nodes=int(settings["maximum_nodes"]),
            dependency_matrix=(
                payload.get("dependency_matrix")  # type: ignore[arg-type]
                if settings["neighbor_correlation_factor_enabled"] else None
            ),
        )
        connected = bool(network["all_source_sink_pairs_connected"])
        if not connected:
            issues.append({
                "severity": "warning",
                "code": "SOURCE_SINK_PATHS_DISCONNECTED",
                "location": str(payload.get("system_id") or "external_json"),
                "message": (
                    f"{network['reachable_source_sink_pair_count']} of "
                    f"{network['requested_source_sink_pair_count']} declared source-sink "
                    "pairs are connected; complete pathway interpretation is blocked"
                ),
            })
        system_reports.append({
            "system_id": payload.get("system_id"),
            "selected_physical_frame_count": int(
                payload["selected_physical_frame_count"]
            ),
            "segments": payload.get("segments", []),
            "network": network,
            "pathway_validity_status": "passed" if connected else "failed",
        })
    all_connected = all(
        report["pathway_validity_status"] == "passed" for report in system_reports
    )
    selected_frames = sum(
        int(report["selected_physical_frame_count"]) for report in system_reports
    )
    report = {
        "module_id": "allosteric_pathways",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "pathway_validity_status": "passed" if all_connected else "failed",
        "project_manifest_path": str(source),
        "project_manifest_sha256": sha256_file(source),
        "content_hashes_included": hash_content,
        "settings": settings,
        "network_derivation": derivation,
        "nodes": nodes,
        "observation_accounting": {
            "selected_physical_frame_count": selected_frames,
            "symmetry_expanded_observation_count": selected_frames,
            "accounting_basis": (
                "selected trajectory frames used directly for per-system contact "
                "occupancies and fitted displacement dependencies"
                if network_source == "trajectory" else
                "external aggregate network; no trajectory frames are read by this module"
            ),
        },
        "systems": system_reports,
        "error_count": 0,
        "warning_count": len(issues),
        "issues": issues,
        "limitations": [
            "In trajectory mode, physical edges are representative-atom distance occupancies calculated from the selected frames; correlation alone is never accepted as a contact edge.",
            "Representative-atom cutoff, within-chain sequence exclusion, and occupancy threshold sensitivity must be evaluated before interpretation.",
            "Negative-log occupancy distances rank high-persistence routes and do not measure transmission time, energy, or causal information flow.",
            "Path participation averages across all equal shortest paths; the canonical path is only a deterministic representative.",
            "Betweenness and path usage are sensitive to contact definition, occupancy cutoff, source/sink selection, and trajectory sampling.",
            "The optional neighbor-correlation factor is the explicitly reported local occupancy-weighted dependency definition in this codebase; equivalence to an external SenseNet implementation is not claimed.",
            "The combined score averages separately min-max normalized centrality and local dependency; it is a prioritization device, not an inferential statistic.",
            "Technical completion and network connectivity do not establish allostery, mechanism, convergence, or scientific validity.",
        ],
    }
    if network_source == "external_json":
        report.update({
            "network_path": derivation["network_path"],
            "network_sha256": derivation["network_sha256"],
            "network": system_reports[0]["network"],
        })
    else:
        report.update({
            "system_manifest_path": derivation["system_manifest_path"],
            "system_manifest_sha256": derivation["system_manifest_sha256"],
            "input_content_signature_sha256": derivation[
                "input_content_signature_sha256"
            ],
        })
    return report


def allosteric_pathways_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return allosteric_pathways_project(project_path, hash_content=hash_content)
    except (
        AllostericPathwayError, AtomMappingError, CoordinateReadError,
        GeometryError, ManifestValidationError, MomentError, OSError, KeyError,
        PeriodicReconstructionError, TypeError, ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "allosteric_pathways",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "pathway_validity_status": "not evaluated",
            "project_manifest_path": str(
                Path(project_path).expanduser().resolve(strict=False)
            ),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {
                    "severity": "error",
                    "code": "ALLOSTERIC_PATHWAYS_INVALID",
                    "message": message,
                }
                for message in messages
            ],
        }
