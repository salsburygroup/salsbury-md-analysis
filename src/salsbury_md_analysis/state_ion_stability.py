"""State-conditioned ion-site occupancy and positional stability.

Input coordinates must already be aligned on the state-defining polymer frame.
Equivalent ions are assigned to spatial sites within each frame, so exchange of
topology ion identities does not masquerade as site motion.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


class StateIonStabilityError(ValueError):
    """Raised when aligned state/ion observations are incomplete."""


def _settings(raw: object) -> Dict[str, object]:
    values = raw if isinstance(raw, dict) else {}
    allowed = {
        "site_discovery_radius_angstrom",
        "site_assignment_cutoff_angstrom",
        "minimum_state_frames",
        "minimum_site_occupancy_fraction",
        "maximum_site_rmsf_angstrom",
        "maximum_sites_per_species",
    }
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise StateIonStabilityError(
            "unknown state-ion stability settings: " + ", ".join(unknown)
        )
    result: Dict[str, object] = {
        "site_discovery_radius_angstrom": values.get(
            "site_discovery_radius_angstrom", 1.5
        ),
        "site_assignment_cutoff_angstrom": values.get(
            "site_assignment_cutoff_angstrom", 2.25
        ),
        "minimum_state_frames": values.get("minimum_state_frames", 20),
        "minimum_site_occupancy_fraction": values.get(
            "minimum_site_occupancy_fraction", 0.5
        ),
        "maximum_site_rmsf_angstrom": values.get(
            "maximum_site_rmsf_angstrom", 1.0
        ),
        "maximum_sites_per_species": values.get("maximum_sites_per_species", 32),
    }
    for field in (
        "site_discovery_radius_angstrom",
        "site_assignment_cutoff_angstrom",
        "maximum_site_rmsf_angstrom",
    ):
        value = result[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise StateIonStabilityError(f"{field} must be finite and positive")
        result[field] = float(value)
    occupancy = result["minimum_site_occupancy_fraction"]
    if (
        isinstance(occupancy, bool)
        or not isinstance(occupancy, (int, float))
        or not math.isfinite(float(occupancy))
        or not 0.0 < float(occupancy) <= 1.0
    ):
        raise StateIonStabilityError(
            "minimum_site_occupancy_fraction must be within (0, 1]"
        )
    result["minimum_site_occupancy_fraction"] = float(occupancy)
    for field in ("minimum_state_frames", "maximum_sites_per_species"):
        value = result[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise StateIonStabilityError(f"{field} must be a positive integer")
    return result


def _ion_rows(frame: Mapping[str, object]) -> List[Dict[str, object]]:
    rows = frame.get("ions")
    if not isinstance(rows, list):
        raise StateIonStabilityError("every state frame must contain an ions array")
    normalized = []
    identities = set()
    for row in rows:
        if not isinstance(row, dict):
            raise StateIonStabilityError("ion observations must be objects")
        identity = str(row.get("ion_id", "")).strip()
        species = str(row.get("element", "")).strip().upper()
        coordinates = row.get("coordinates_angstrom")
        if not identity or not species:
            raise StateIonStabilityError("ion_id and element must be nonempty")
        if identity in identities:
            raise StateIonStabilityError("ion_id values must be unique within a frame")
        identities.add(identity)
        if (
            not isinstance(coordinates, list)
            or len(coordinates) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in coordinates
            )
        ):
            raise StateIonStabilityError(
                "coordinates_angstrom must contain three finite numbers"
            )
        normalized.append({
            "ion_id": identity,
            "element": species,
            "coordinates_angstrom": [float(value) for value in coordinates],
        })
    return normalized


def _discover_centers(
    observations: Sequence[Tuple[int, str, np.ndarray]],
    radius: float,
    maximum_sites: int,
) -> List[np.ndarray]:
    if not observations:
        return []
    points = np.asarray([row[2] for row in observations], dtype=np.float64)
    remaining = list(range(len(points)))
    centers = []
    while remaining and len(centers) < maximum_sites:
        best_index = min(
            remaining,
            key=lambda candidate: (
                -sum(
                    float(np.linalg.norm(points[candidate] - points[other])) <= radius
                    for other in remaining
                ),
                candidate,
            ),
        )
        neighbors = [
            other for other in remaining
            if float(np.linalg.norm(points[best_index] - points[other])) <= radius
        ]
        centers.append(np.mean(points[neighbors], axis=0))
        removed = set(neighbors)
        remaining = [index for index in remaining if index not in removed]
    return centers


def _assign_frames(
    frame_rows: Sequence[Mapping[str, object]],
    species: str,
    centers: Sequence[np.ndarray],
    cutoff: float,
) -> List[List[Tuple[str, np.ndarray, int]]]:
    assignments: List[List[Tuple[str, np.ndarray, int]]] = [
        [] for _ in centers
    ]
    for frame_index, frame in enumerate(frame_rows):
        ions = [row for row in frame["ions"] if row["element"] == species]
        if not ions or not centers:
            continue
        coordinates = np.asarray(
            [row["coordinates_angstrom"] for row in ions], dtype=np.float64
        )
        costs = np.linalg.norm(
            np.asarray(centers)[:, None, :] - coordinates[None, :, :], axis=2
        )
        site_indices, ion_indices = linear_sum_assignment(costs)
        for site_index, ion_index in zip(site_indices, ion_indices):
            if float(costs[site_index, ion_index]) <= cutoff:
                assignments[int(site_index)].append((
                    str(ions[int(ion_index)]["ion_id"]),
                    coordinates[int(ion_index)],
                    frame_index,
                ))
    return assignments


def analyze_state_ion_stability(request: Mapping[str, object]) -> Dict[str, object]:
    """Find occupied, positionally stable ion sites within each structural state."""

    if request.get("coordinates_aligned_to_polymer") is not True:
        raise StateIonStabilityError(
            "coordinates_aligned_to_polymer must be true; ion RMSF is undefined "
            "until every state frame uses one declared polymer alignment"
        )
    raw_frames = request.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise StateIonStabilityError("frames must be a nonempty array")
    settings = _settings(request.get("settings"))
    grouped: Dict[Tuple[str, int], List[Dict[str, object]]] = defaultdict(list)
    seen = set()
    for raw in raw_frames:
        if not isinstance(raw, dict):
            raise StateIonStabilityError("state frames must be objects")
        system_id = str(raw.get("system_id", "")).strip()
        state_id = raw.get("state_id")
        frame_id = str(raw.get("frame_id", "")).strip()
        if (
            not system_id
            or isinstance(state_id, bool)
            or not isinstance(state_id, int)
            or state_id <= 0
            or not frame_id
        ):
            raise StateIonStabilityError(
                "every frame requires system_id, positive state_id, and frame_id"
            )
        identity = (system_id, state_id, frame_id)
        if identity in seen:
            raise StateIonStabilityError("state frame identities must be unique")
        seen.add(identity)
        grouped[(system_id, state_id)].append({
            "frame_id": frame_id,
            "ions": _ion_rows(raw),
        })

    reports = []
    for (system_id, state_id), frames in sorted(grouped.items()):
        species_values = sorted({
            row["element"] for frame in frames for row in frame["ions"]
        })
        if len(frames) < int(settings["minimum_state_frames"]):
            reports.append({
                "system_id": system_id,
                "state_id": state_id,
                "evaluated_frame_count": len(frames),
                "technical_status": "not_estimable",
                "reason": "minimum_state_frames_not_met",
                "stable_sites": [],
            })
            continue
        species_reports = []
        for species in species_values:
            observations = [
                (frame_index, str(ion["ion_id"]), np.asarray(ion["coordinates_angstrom"]))
                for frame_index, frame in enumerate(frames)
                for ion in frame["ions"] if ion["element"] == species
            ]
            centers = _discover_centers(
                observations,
                float(settings["site_discovery_radius_angstrom"]),
                int(settings["maximum_sites_per_species"]),
            )
            assignments = _assign_frames(
                frames, species, centers,
                float(settings["site_assignment_cutoff_angstrom"]),
            )
            # Refine once after unique per-frame site assignment.
            refined = [
                np.mean([row[1] for row in rows], axis=0) if rows else centers[index]
                for index, rows in enumerate(assignments)
            ]
            assignments = _assign_frames(
                frames, species, refined,
                float(settings["site_assignment_cutoff_angstrom"]),
            )
            sites = []
            for site_index, rows in enumerate(assignments, start=1):
                occupied_frames = sorted({row[2] for row in rows})
                occupancy = len(occupied_frames) / len(frames)
                if rows:
                    positions = np.asarray([row[1] for row in rows], dtype=np.float64)
                    center = np.mean(positions, axis=0)
                    rmsf = float(np.sqrt(np.mean(np.sum((positions - center) ** 2, axis=1))))
                else:
                    center = refined[site_index - 1]
                    rmsf = None
                stable = (
                    occupancy >= float(settings["minimum_site_occupancy_fraction"])
                    and rmsf is not None
                    and rmsf <= float(settings["maximum_site_rmsf_angstrom"])
                )
                sites.append({
                    "site_id": f"{species}-site-{site_index:03d}",
                    "element": species,
                    "center_angstrom": [float(value) for value in center],
                    "occupied_frame_count": len(occupied_frames),
                    "occupancy_fraction": occupancy,
                    "positional_rmsf_angstrom": rmsf,
                    "stable_for_default_state_view": stable,
                    "frame_assignments": [{
                        "frame_id": str(frames[frame_index]["frame_id"]),
                        "ion_id": ion_id,
                        "coordinates_angstrom": [float(value) for value in position],
                    } for ion_id, position, frame_index in rows],
                })
            species_reports.append({
                "element": species,
                "candidate_site_count": len(sites),
                "stable_site_count": sum(
                    row["stable_for_default_state_view"] for row in sites
                ),
                "sites": sites,
            })
        reports.append({
            "system_id": system_id,
            "state_id": state_id,
            "evaluated_frame_count": len(frames),
            "technical_status": "complete",
            "species": species_reports,
            "stable_sites": [
                {"element": species["element"], **site}
                for species in species_reports for site in species["sites"]
                if site["stable_for_default_state_view"]
            ],
        })
    return {
        "module_id": "state_conditioned_ion_stability",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "settings": settings,
        "coordinates_aligned_to_polymer": True,
        "state_reports": reports,
        "limitations": [
            "Stable sites require both state-conditioned occupancy and positional RMSF gates.",
            "Equivalent ions are matched to spatial sites within each frame; topology ion identity is not treated as site identity.",
            "The calculation describes positionally stable ion sites and does not establish biological binding, affinity, oxidation state, or mechanism.",
        ],
    }


def analyze_state_ion_stability_safe(request: Mapping[str, object]) -> Dict[str, object]:
    try:
        return analyze_state_ion_stability(request)
    except (StateIonStabilityError, ValueError, FloatingPointError) as exc:
        return {
            "module_id": "state_conditioned_ion_stability",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "issues": [{
                "severity": "error",
                "code": "STATE_ION_STABILITY_INVALID",
                "message": str(exc),
            }],
        }
