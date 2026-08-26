"""Experimental DFI/DCI analysis from common-PCA trajectory covariances."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .manifests import ManifestValidationError, load_json
from .pca import common_pca_project
from .validation import positive_integer


class PerturbationResponseError(ValueError):
    """Raised when a perturbation-response contract is incomplete or invalid."""


def _percentile_ranks(values: Sequence[float]) -> List[float]:
    """Return the inclusive empirical percentile used for %DFI and %DCI."""

    return [
        sum(candidate <= value for candidate in values) / len(values)
        for value in values
    ]


def perturbation_response_indices(
    cartesian_covariance: Sequence[Sequence[float]],
    functional_site_node_indices: Sequence[int],
    *,
    random_force_directions: int = 250,
    random_seed: int = 20260824,
    maximum_nodes: int = 2000,
    include_self_perturbations: bool = True,
) -> Dict[str, object]:
    """Calculate DFI and functional-site DCI from a Cartesian covariance.

    The covariance is interpreted through linear response, ``delta_R = C F``.
    For each source node, deterministic seeded random unit-force directions are
    applied and the mean magnitude of the target-node response is recorded.
    Rows in the returned response matrix are targets and columns are sources.
    """

    covariance = np.asarray(cartesian_covariance, dtype=float)
    maximum_nodes = positive_integer(maximum_nodes, "maximum_nodes")
    random_force_directions = positive_integer(
        random_force_directions, "random_force_directions"
    )
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise PerturbationResponseError("cartesian_covariance must be square")
    if covariance.shape[0] < 3 or covariance.shape[0] % 3:
        raise PerturbationResponseError(
            "cartesian_covariance dimensions must equal three times the node count"
        )
    node_count = covariance.shape[0] // 3
    if node_count > maximum_nodes:
        raise PerturbationResponseError(
            f"covariance contains {node_count} nodes; maximum_nodes is {maximum_nodes}"
        )
    if not np.isfinite(covariance).all():
        raise PerturbationResponseError("cartesian_covariance contains non-finite values")
    scale = max(1.0, float(np.max(np.abs(covariance))))
    if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=1.0e-12 * scale):
        raise PerturbationResponseError("cartesian_covariance must be symmetric")
    sites = tuple(functional_site_node_indices)
    if (
        not sites
        or len(set(sites)) != len(sites)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in sites)
        or any(value < 0 or value >= node_count for value in sites)
    ):
        raise PerturbationResponseError(
            "functional_site_node_indices must be unique zero-based node indices"
        )
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise PerturbationResponseError("random_seed must be a nonnegative integer")
    if not isinstance(include_self_perturbations, bool):
        raise PerturbationResponseError("include_self_perturbations must be boolean")

    blocks = covariance.reshape(node_count, 3, node_count, 3).transpose(0, 2, 1, 3)
    generator = np.random.default_rng(random_seed)
    response = np.zeros((node_count, node_count), dtype=float)
    for source in range(node_count):
        forces = generator.normal(size=(random_force_directions, 3))
        norms = np.linalg.norm(forces, axis=1)
        while np.any(norms <= 1.0e-15):
            forces[norms <= 1.0e-15] = generator.normal(
                size=(int(np.sum(norms <= 1.0e-15)), 3)
            )
            norms = np.linalg.norm(forces, axis=1)
        forces /= norms[:, None]
        displacements = np.einsum(
            "tab,db->dta", blocks[:, source, :, :], forces, optimize=True
        )
        response[:, source] = np.mean(np.linalg.norm(displacements, axis=2), axis=0)
    if not include_self_perturbations:
        np.fill_diagonal(response, 0.0)

    row_sums = response.sum(axis=1)
    total = float(row_sums.sum())
    if total <= 1.0e-15:
        raise PerturbationResponseError(
            "covariance produced no perturbation response above numerical zero"
        )
    dfi = (row_sums / total).tolist()
    all_counts = np.full(node_count, node_count, dtype=float)
    if not include_self_perturbations:
        all_counts -= 1.0
    all_means = row_sums / all_counts
    functional = response[:, sites].sum(axis=1)
    functional_counts = np.full(node_count, len(sites), dtype=float)
    if not include_self_perturbations:
        functional_counts -= np.asarray(
            [int(target in sites) for target in range(node_count)], dtype=float
        )
    if np.any(functional_counts <= 0.0):
        raise PerturbationResponseError(
            "excluding self perturbations leaves a node with no functional-site source"
        )
    functional_means = functional / functional_counts
    if np.any(all_means <= 1.0e-15):
        raise PerturbationResponseError(
            "at least one node has no global response for DCI normalization"
        )
    dci = (functional_means / all_means).tolist()
    return {
        "node_count": node_count,
        "functional_site_node_indices": list(sites),
        "response_matrix": response.tolist(),
        "matrix_orientation": "row target node; column perturbed source node",
        "dfi": dfi,
        "dfi_percentile": _percentile_ranks(dfi),
        "dci": dci,
        "dci_percentile": _percentile_ranks(dci),
        "random_force_directions": random_force_directions,
        "random_seed": random_seed,
        "include_self_perturbations": include_self_perturbations,
        "dfi_definition": "target response summed over source perturbations and normalized by the response sum over all target-source pairs",
        "dci_definition": "mean target response to functional-site perturbations divided by that target's mean response to all source perturbations",
    }


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = (
        definitions.get("perturbation_response_dynamics")
        if isinstance(definitions, dict) else None
    )
    if not isinstance(raw, dict):
        raise PerturbationResponseError(
            "definitions.perturbation_response_dynamics must be an object"
        )
    required = {
        "feature_source", "functional_site_node_indices",
        "random_force_directions", "random_seed", "maximum_nodes",
        "minimum_observations_per_system",
        "minimum_cumulative_explained_variance",
        "include_self_perturbations",
    }
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required))
    if missing or unknown:
        raise PerturbationResponseError(
            "perturbation-response settings mismatch; missing="
            + ",".join(missing) + "; unknown=" + ",".join(unknown)
        )
    if raw["feature_source"] != "common_pca":
        raise PerturbationResponseError("feature_source must be common_pca")
    sites = raw["functional_site_node_indices"]
    if not isinstance(sites, list):
        raise PerturbationResponseError(
            "functional_site_node_indices must be an array"
        )
    for field in (
        "random_force_directions", "maximum_nodes",
        "minimum_observations_per_system",
    ):
        positive_integer(raw[field], field)
    seed = raw["random_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise PerturbationResponseError("random_seed must be a nonnegative integer")
    threshold = raw["minimum_cumulative_explained_variance"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise PerturbationResponseError(
            "minimum_cumulative_explained_variance must be between zero and one"
        )
    if not isinstance(raw["include_self_perturbations"], bool):
        raise PerturbationResponseError(
            "include_self_perturbations must be boolean"
        )
    return dict(raw)


def _basis_loadings(
    pca: Mapping[str, object]
) -> Tuple[np.ndarray, float, List[Dict[str, object]]]:
    basis_payload = pca.get("basis")
    payload = basis_payload.get("pca") if isinstance(basis_payload, dict) else None
    components = payload.get("components") if isinstance(payload, dict) else None
    if not isinstance(components, list) or not components:
        raise PerturbationResponseError("common PCA report contains no components")
    rows = []
    for component in components:
        loadings = component.get("loadings") if isinstance(component, dict) else None
        if not isinstance(loadings, list) or not loadings:
            raise PerturbationResponseError("common PCA component lacks atom loadings")
        rows.append([
            coordinate
            for atom in loadings
            for coordinate in (
                float(atom["loading_x"]),
                float(atom["loading_y"]),
                float(atom["loading_z"]),
            )
        ])
    basis = np.asarray(rows, dtype=float)
    if not np.isfinite(basis).all() or len({len(row) for row in rows}) != 1:
        raise PerturbationResponseError("common PCA loadings are inconsistent")
    explained = components[-1].get("cumulative_explained_variance_fraction")
    if isinstance(explained, bool) or not isinstance(explained, (int, float)):
        raise PerturbationResponseError(
            "common PCA components lack cumulative explained variance"
        )
    mean_structure = payload.get("mean_structure") if isinstance(payload, dict) else None
    if not isinstance(mean_structure, list) or len(mean_structure) * 3 != basis.shape[1]:
        raise PerturbationResponseError(
            "common PCA mean-structure node count is inconsistent with loadings"
        )
    if not all(isinstance(row, dict) for row in mean_structure):
        raise PerturbationResponseError("common PCA mean-structure rows are invalid")
    return basis, float(explained), [dict(row) for row in mean_structure]


def _system_scores(system: Mapping[str, object], component_count: int) -> np.ndarray:
    rows: List[List[float]] = []
    replicas = system.get("replicas")
    if not isinstance(replicas, list):
        raise PerturbationResponseError("common PCA system contains no replicas")
    for replica in replicas:
        segments = replica.get("segments") if isinstance(replica, dict) else None
        if not isinstance(segments, list):
            raise PerturbationResponseError("common PCA replica contains no segments")
        for segment in segments:
            projections = segment.get("projections") if isinstance(segment, dict) else None
            if not isinstance(projections, list):
                raise PerturbationResponseError(
                    "common PCA segment contains no projections"
                )
            for projection in projections:
                scores = projection.get("scores_angstrom") if isinstance(projection, dict) else None
                if not isinstance(scores, list) or len(scores) != component_count:
                    raise PerturbationResponseError(
                        "common PCA projection component count is inconsistent"
                    )
                rows.append([float(value) for value in scores])
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise PerturbationResponseError("common PCA projections are invalid")
    return values


def perturbation_response_dynamics_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    pca = common_pca_project(source, hash_content=hash_content)
    if pca.get("technical_status") != "complete":
        raise PerturbationResponseError("common PCA report is not technically complete")
    basis, explained, analysis_nodes = _basis_loadings(pca)
    minimum_explained = float(settings["minimum_cumulative_explained_variance"])
    if explained < minimum_explained:
        raise PerturbationResponseError(
            f"common PCA basis captures {explained:.6g} cumulative variance; "
            f"minimum is {minimum_explained:.6g}"
        )
    systems = pca.get("systems")
    if not isinstance(systems, list) or not systems:
        raise PerturbationResponseError("common PCA report contains no systems")
    reports = []
    for system in systems:
        if not isinstance(system, dict):
            raise PerturbationResponseError("common PCA system entry is invalid")
        scores = _system_scores(system, basis.shape[0])
        minimum = int(settings["minimum_observations_per_system"])
        if len(scores) < minimum:
            raise PerturbationResponseError(
                f"system {system.get('system_id')} has {len(scores)} projections; "
                f"minimum_observations_per_system is {minimum}"
            )
        centered = scores - scores.mean(axis=0)
        score_covariance = centered.T @ centered / len(scores)
        cartesian_covariance = basis.T @ score_covariance @ basis
        indices = perturbation_response_indices(
            cartesian_covariance,
            settings["functional_site_node_indices"],  # type: ignore[arg-type]
            random_force_directions=int(settings["random_force_directions"]),
            random_seed=int(settings["random_seed"]),
            maximum_nodes=int(settings["maximum_nodes"]),
            include_self_perturbations=bool(settings["include_self_perturbations"]),
        )
        reports.append({
            "system_id": system["system_id"],
            "observation_count": len(scores),
            "score_covariance_denominator": "population_N",
            **indices,
        })
    reference_id = str(pca["reference_system_id"])
    reference = next(
        (report for report in reports if report["system_id"] == reference_id), None
    )
    if reference is None:
        raise PerturbationResponseError(
            "common PCA reference system is absent from projection systems"
        )
    for report in reports:
        report["difference_from_reference"] = {
            "reference_system_id": reference_id,
            "dfi": [
                float(value) - float(reference_value)
                for value, reference_value in zip(report["dfi"], reference["dfi"])
            ],
            "dci": [
                float(value) - float(reference_value)
                for value, reference_value in zip(report["dci"], reference["dci"])
            ],
        }
    pca_accounting = pca.get("observation_accounting")
    projection_selection = pca.get("projection_frame_selection")
    physical_frames = (
        pca_accounting.get("selected_physical_frame_count")
        if isinstance(pca_accounting, dict)
        else projection_selection.get("selected_frame_count")
        if isinstance(projection_selection, dict)
        else None
    )
    if isinstance(physical_frames, bool) or not isinstance(physical_frames, int):
        raise PerturbationResponseError(
            "common PCA report lacks exact projection-frame accounting"
        )
    observation_count = sum(int(report["observation_count"]) for report in reports)
    return {
        "module_id": "perturbation_response_dynamics",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": pca["project_manifest_sha256"],
        "system_manifest_path": pca["system_manifest_path"],
        "system_manifest_sha256": pca["system_manifest_sha256"],
        "input_content_signature_sha256": pca["input_content_signature_sha256"],
        "settings": settings,
        "reference_system_id": reference_id,
        "analysis_nodes": analysis_nodes,
        "common_pca_component_count": int(basis.shape[0]),
        "common_pca_cumulative_explained_variance": explained,
        "covariance_contract": (
            "per-system population covariance of common-PCA scores, mapped back "
            "to the retained shared Cartesian subspace"
        ),
        "observation_accounting": {
            "selected_physical_frame_count": physical_frames,
            "symmetry_expanded_observation_count": observation_count,
            "accounting_basis": "common-PCA projection identities consumed by per-system covariance estimates",
        },
        "systems": reports,
        "error_count": 0,
        "warning_count": 0,
        "issues": [],
        "limitations": [
            "DFI/DCI are calculated at the selected analysis-node level; residue interpretation requires one declared representative atom per residue, normally C-alpha for proteins.",
            "The response covariance is restricted to retained common-PCA components; cumulative explained variance and the configured acceptance gate are reported.",
            "Seeded random unit forces reproduce the perturbation-response convention but require force-count and seed sensitivity checks.",
            "System covariances are frame pooled; frames are not independent uncertainty units and replica-level uncertainty is not inferred.",
            "DFI and DCI depend on alignment, atom mapping, functional-site selection, PCA dimensionality, and sampling adequacy.",
            "Dynamic coupling is an association under linear response and does not establish directionality, causality, an allosteric pathway, or mechanism.",
            "Technical completion does not establish convergence, scientific validity, or biological importance.",
        ],
    }


def perturbation_response_dynamics_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return perturbation_response_dynamics_project(
            project_path, hash_content=hash_content
        )
    except (
        PerturbationResponseError, ManifestValidationError, OSError,
        KeyError, TypeError, ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "perturbation_response_dynamics",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(
                Path(project_path).expanduser().resolve(strict=False)
            ),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {
                    "severity": "error",
                    "code": "PERTURBATION_RESPONSE_INVALID",
                    "message": message,
                }
                for message in messages
            ],
        }
