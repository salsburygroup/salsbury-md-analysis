"""Deterministic PCA occupancy landscapes, thermodynamic FES, and basins."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from .clustering import (
    ClusteringAnalysisError,
    adjusted_rand_index,
    silhouette_score_report,
)
from .manifests import ManifestValidationError, load_json
from .pca import PCAAnalysisError, common_pca_project
from .state_populations import summarize_state_populations
from .upstream_cache import load_cached_project_report


KB_KCAL_PER_MOL_K = 0.00198720425864083


class PCAFESAnalysisError(ValueError):
    """Raised when a PCA landscape contract is incomplete or misleading."""


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    if not isinstance(definitions, dict):
        raise PCAFESAnalysisError("project definitions.pca_fes_basins is required")
    raw = definitions.get("pca_fes_basins")
    if not isinstance(raw, dict):
        raise PCAFESAnalysisError("definitions.pca_fes_basins must be an object")
    required = {
        "x_component",
        "y_component",
        "padding_fraction",
        "minimum_bin_count",
        "population_block_size_frames",
        "include_partial_final_block",
        "maximum_grid_cells",
        "density_estimator",
    }
    optional = {
        "bins_x", "bins_y", "binning_rule",
        "minimum_bins_per_axis", "maximum_bins_per_axis",
        "smoothing_sigmas_bins", "primary_smoothing_sigma_bins",
        "maximum_silhouette_observations", "silhouette_random_seed",
    }
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required | optional))
    if missing:
        raise PCAFESAnalysisError(
            "definitions.pca_fes_basins is missing required fields: "
            + ", ".join(missing)
        )
    if unknown:
        raise PCAFESAnalysisError(
            "definitions.pca_fes_basins contains unknown fields: "
            + ", ".join(unknown)
        )
    result: Dict[str, object] = {}
    for field in (
        "x_component",
        "y_component",
        "minimum_bin_count",
        "population_block_size_frames",
        "maximum_grid_cells",
    ):
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PCAFESAnalysisError(f"{field} must be a positive integer")
        result[field] = value
    if result["x_component"] == result["y_component"]:
        raise PCAFESAnalysisError("x_component and y_component must be distinct")
    binning_rule = raw.get("binning_rule", "explicit")
    if binning_rule not in {"explicit", "scott", "freedman_diaconis", "rice"}:
        raise PCAFESAnalysisError(
            "binning_rule must be explicit, scott, freedman_diaconis, or rice"
        )
    result["binning_rule"] = binning_rule
    if binning_rule == "explicit":
        if "bins_x" not in raw or "bins_y" not in raw:
            raise PCAFESAnalysisError("explicit binning requires bins_x and bins_y")
        if "minimum_bins_per_axis" in raw or "maximum_bins_per_axis" in raw:
            raise PCAFESAnalysisError(
                "explicit binning cannot declare automatic bin-count gates"
            )
        for field in ("bins_x", "bins_y"):
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise PCAFESAnalysisError(f"{field} must be an integer of at least 2")
            result[field] = value
        if int(result["bins_x"]) * int(result["bins_y"]) > int(
            result["maximum_grid_cells"]
        ):
            raise PCAFESAnalysisError("requested grid exceeds maximum_grid_cells")
    else:
        if "bins_x" in raw or "bins_y" in raw:
            raise PCAFESAnalysisError(
                "automatic binning cannot also declare bins_x or bins_y"
            )
        for field in ("minimum_bins_per_axis", "maximum_bins_per_axis"):
            value = raw.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise PCAFESAnalysisError(
                    f"automatic binning requires {field} as an integer of at least 2"
                )
            result[field] = value
        if int(result["minimum_bins_per_axis"]) > int(result["maximum_bins_per_axis"]):
            raise PCAFESAnalysisError(
                "minimum_bins_per_axis cannot exceed maximum_bins_per_axis"
            )
    padding = raw["padding_fraction"]
    if (
        isinstance(padding, bool)
        or not isinstance(padding, (int, float))
        or not math.isfinite(float(padding))
        or float(padding) < 0.0
    ):
        raise PCAFESAnalysisError("padding_fraction must be finite and nonnegative")
    result["padding_fraction"] = float(padding)
    if not isinstance(raw["include_partial_final_block"], bool):
        raise PCAFESAnalysisError("include_partial_final_block must be boolean")
    result["include_partial_final_block"] = raw["include_partial_final_block"]
    if raw["density_estimator"] != "histogram":
        raise PCAFESAnalysisError(
            "density_estimator currently supports only the explicit value histogram"
        )
    result["density_estimator"] = "histogram"
    smoothing = raw.get("smoothing_sigmas_bins", [0.0])
    if (
        not isinstance(smoothing, list)
        or not smoothing
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in smoothing
        )
    ):
        raise PCAFESAnalysisError(
            "smoothing_sigmas_bins must be a nonempty array of finite nonnegative numbers"
        )
    normalized_smoothing = [float(value) for value in smoothing]
    if len(set(normalized_smoothing)) != len(normalized_smoothing):
        raise PCAFESAnalysisError("smoothing_sigmas_bins must contain unique values")
    primary = raw.get("primary_smoothing_sigma_bins", normalized_smoothing[0])
    if (
        isinstance(primary, bool)
        or not isinstance(primary, (int, float))
        or not math.isfinite(float(primary))
        or float(primary) not in normalized_smoothing
    ):
        raise PCAFESAnalysisError(
            "primary_smoothing_sigma_bins must be one declared smoothing value"
        )
    result["smoothing_sigmas_bins"] = normalized_smoothing
    result["primary_smoothing_sigma_bins"] = float(primary)
    maximum_silhouette = raw.get("maximum_silhouette_observations", 1_000)
    if (
        isinstance(maximum_silhouette, bool)
        or not isinstance(maximum_silhouette, int)
        or maximum_silhouette <= 0
    ):
        raise PCAFESAnalysisError(
            "maximum_silhouette_observations must be a positive integer"
        )
    silhouette_seed = raw.get("silhouette_random_seed", 0)
    if isinstance(silhouette_seed, bool) or not isinstance(silhouette_seed, int):
        raise PCAFESAnalysisError("silhouette_random_seed must be an integer")
    result["maximum_silhouette_observations"] = maximum_silhouette
    result["silhouette_random_seed"] = silhouette_seed
    return result


def _cell_index(value: float, lower: float, width: float, count: int) -> int:
    return max(0, min(count - 1, int((value - lower) / width)))


def _neighbors(cell: Tuple[int, int], bins_x: int, bins_y: int) -> Iterable[Tuple[int, int]]:
    x, y = cell
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            candidate = (x + dx, y + dy)
            if 0 <= candidate[0] < bins_x and 0 <= candidate[1] < bins_y:
                yield candidate


def select_bin_counts(
    points: Sequence[Tuple[float, float]],
    rule: str,
    padding_fraction: float,
    minimum_bins_per_axis: int,
    maximum_bins_per_axis: int,
) -> Dict[str, object]:
    """Select independent axis counts using a declared histogram rule."""

    if rule not in {"scott", "freedman_diaconis", "rice"}:
        raise PCAFESAnalysisError("automatic binning rule is unsupported")
    if len(points) < 2 or not all(
        math.isfinite(x) and math.isfinite(y) for x, y in points
    ):
        raise PCAFESAnalysisError("automatic binning requires finite PCA points")
    if minimum_bins_per_axis < 2 or maximum_bins_per_axis < minimum_bins_per_axis:
        raise PCAFESAnalysisError("automatic bin-count gates are invalid")

    axis_values = ([point[0] for point in points], [point[1] for point in points])
    raw_counts = []
    widths = []
    count = len(points)
    for values in axis_values:
        ordered = sorted(values)
        span = ordered[-1] - ordered[0]
        if span <= 0.0:
            raise PCAFESAnalysisError(
                "automatic binning requires nonconstant values on both axes"
            )
        if rule == "rice":
            raw_count = int(math.ceil(2.0 * count ** (1.0 / 3.0)))
            width = span * (1.0 + 2.0 * padding_fraction) / raw_count
        else:
            if rule == "scott":
                mean = sum(values) / count
                scale = math.sqrt(sum((value - mean) ** 2 for value in values) / count)
                width = 3.5 * scale * count ** (-1.0 / 3.0)
            else:
                def percentile(fraction: float) -> float:
                    position = (count - 1) * fraction
                    lower = int(math.floor(position))
                    upper = int(math.ceil(position))
                    if lower == upper:
                        return ordered[lower]
                    weight = position - lower
                    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

                iqr = percentile(0.75) - percentile(0.25)
                width = 2.0 * iqr * count ** (-1.0 / 3.0)
            if not math.isfinite(width) or width <= 0.0:
                raise PCAFESAnalysisError(
                    f"{rule} bin width is undefined for an axis; use explicit bins"
                )
            raw_count = int(
                math.ceil(span * (1.0 + 2.0 * padding_fraction) / width)
            )
        raw_counts.append(max(1, raw_count))
        widths.append(width)
    selected = [
        min(maximum_bins_per_axis, max(minimum_bins_per_axis, value))
        for value in raw_counts
    ]
    return {
        "rule": rule,
        "observation_count": count,
        "raw_bins_x": raw_counts[0],
        "raw_bins_y": raw_counts[1],
        "bins_x": selected[0],
        "bins_y": selected[1],
        "rule_width_x": widths[0],
        "rule_width_y": widths[1],
        "minimum_bins_per_axis": minimum_bins_per_axis,
        "maximum_bins_per_axis": maximum_bins_per_axis,
        "bin_count_clamped": selected != raw_counts,
        "axis_contract": "each PCA coordinate is binned independently",
    }


def build_landscape(
    points: Sequence[Tuple[float, float]],
    *,
    bins_x: int,
    bins_y: int,
    padding_fraction: float,
    minimum_bin_count: int,
    temperature_kelvin: Optional[float],
    smoothing_sigma_bins: float = 0.0,
    fixed_bounds: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    """Return a histogram landscape and deterministic density-catchment basins.

    Raw histogram counts are always retained.  A nonzero Gaussian sigma changes
    only the density used to locate minima and route catchments; it never changes
    raw frame populations.  The kernel is evaluated in bin units with zero
    density outside the declared grid and renormalized after boundary loss.
    """

    if len(points) < 2:
        raise PCAFESAnalysisError("at least two PCA points are required")
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in points):
        raise PCAFESAnalysisError("PCA projections contain a non-finite value")
    if (
        isinstance(smoothing_sigma_bins, bool)
        or not isinstance(smoothing_sigma_bins, (int, float))
        or not math.isfinite(float(smoothing_sigma_bins))
        or float(smoothing_sigma_bins) < 0.0
    ):
        raise PCAFESAnalysisError(
            "smoothing_sigma_bins must be finite and nonnegative"
        )
    smoothing_sigma_bins = float(smoothing_sigma_bins)
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    if fixed_bounds is None:
        x_span = max(x_values) - min(x_values)
        y_span = max(y_values) - min(y_values)
        if x_span <= 0.0 or y_span <= 0.0:
            raise PCAFESAnalysisError(
                "selected PCA components must each span more than one value"
            )
        x_lower = min(x_values) - padding_fraction * x_span
        x_upper = max(x_values) + padding_fraction * x_span
        y_lower = min(y_values) - padding_fraction * y_span
        y_upper = max(y_values) + padding_fraction * y_span
        bounds_source = "derived_from_these_points"
    else:
        required_bounds = {
            "x_min_angstrom", "x_max_angstrom",
            "y_min_angstrom", "y_max_angstrom",
        }
        if set(fixed_bounds) != required_bounds:
            raise PCAFESAnalysisError(
                "fixed_bounds must contain exactly x_min_angstrom, "
                "x_max_angstrom, y_min_angstrom, and y_max_angstrom"
            )
        values = {key: float(fixed_bounds[key]) for key in required_bounds}
        if not all(math.isfinite(value) for value in values.values()):
            raise PCAFESAnalysisError("fixed_bounds values must be finite")
        x_lower = values["x_min_angstrom"]
        x_upper = values["x_max_angstrom"]
        y_lower = values["y_min_angstrom"]
        y_upper = values["y_max_angstrom"]
        if x_upper <= x_lower or y_upper <= y_lower:
            raise PCAFESAnalysisError("fixed_bounds must have positive axis spans")
        if (
            min(x_values) < x_lower or max(x_values) > x_upper
            or min(y_values) < y_lower or max(y_values) > y_upper
        ):
            raise PCAFESAnalysisError("a PCA point lies outside fixed_bounds")
        bounds_source = "fixed_shared_grid"
    x_width = (x_upper - x_lower) / bins_x
    y_width = (y_upper - y_lower) / bins_y
    counts = [[0 for _ in range(bins_y)] for _ in range(bins_x)]
    point_cells = []
    for x, y in points:
        cell = (
            _cell_index(x, x_lower, x_width, bins_x),
            _cell_index(y, y_lower, y_width, bins_y),
        )
        counts[cell[0]][cell[1]] += 1
        point_cells.append(cell)

    raw_density = np.asarray(counts, dtype=float)
    if smoothing_sigma_bins == 0.0:
        surface_density = raw_density.copy()
    else:
        surface_density = gaussian_filter(
            raw_density,
            sigma=smoothing_sigma_bins,
            mode="constant",
            cval=0.0,
            truncate=4.0,
        )
        retained_mass = float(surface_density.sum())
        if retained_mass <= 0.0 or not np.isfinite(surface_density).all():
            raise PCAFESAnalysisError("Gaussian smoothing produced an invalid density")
        surface_density *= len(points) / retained_mass

    eligible = {
        (x, y)
        for x in range(bins_x)
        for y in range(bins_y)
        if counts[x][y] >= minimum_bin_count
    }
    if not eligible:
        raise PCAFESAnalysisError(
            "no grid cell meets minimum_bin_count; lower the prespecified threshold"
        )

    successors: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for cell in eligible:
        candidates = [cell] + [neighbor for neighbor in _neighbors(cell, bins_x, bins_y) if neighbor in eligible]
        successors[cell] = min(
            candidates,
            key=lambda candidate: (
                -float(surface_density[candidate[0], candidate[1]]), candidate
            ),
        )

    roots: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for cell in eligible:
        trail = []
        current = cell
        while current not in roots and successors[current] != current:
            trail.append(current)
            current = successors[current]
        root = roots.get(current, current)
        roots[current] = root
        for visited in trail:
            roots[visited] = root

    root_populations: Dict[Tuple[int, int], int] = {}
    for cell, root in roots.items():
        root_populations[root] = root_populations.get(root, 0) + counts[cell[0]][cell[1]]
    ordered_roots = sorted(
        root_populations,
        key=lambda root: (-root_populations[root], root),
    )
    basin_by_root = {root: index + 1 for index, root in enumerate(ordered_roots)}
    cell_basins = {cell: basin_by_root[root] for cell, root in roots.items()}
    point_assignments = [cell_basins.get(cell) for cell in point_cells]
    maximum_surface_density = max(
        float(surface_density[x, y]) for x, y in eligible
    )
    cell_area = x_width * y_width
    grid = []
    for x in range(bins_x):
        for y in range(bins_y):
            count = counts[x][y]
            probability = count / len(points)
            surface_count = float(surface_density[x, y])
            surface_probability = surface_count / len(points)
            row: Dict[str, object] = {
                "x_bin": x,
                "y_bin": y,
                "x_center_angstrom": x_lower + (x + 0.5) * x_width,
                "y_center_angstrom": y_lower + (y + 0.5) * y_width,
                "count": count,
                "probability": probability,
                "probability_density_per_angstrom2": probability / cell_area,
                "surface_count": surface_count,
                "surface_probability": surface_probability,
                "surface_probability_density_per_angstrom2": (
                    surface_probability / cell_area
                ),
                "basin_id": cell_basins.get((x, y)),
            }
            if surface_count > 0.0:
                relative_log_occupancy = -math.log(
                    surface_count / maximum_surface_density
                )
                if temperature_kelvin is None:
                    row["relative_occupancy_score"] = relative_log_occupancy
                else:
                    row["relative_free_energy_kcal_per_mol"] = (
                        KB_KCAL_PER_MOL_K
                        * temperature_kelvin
                        * relative_log_occupancy
                    )
            else:
                if temperature_kelvin is None:
                    row["relative_occupancy_score"] = None
                else:
                    row["relative_free_energy_kcal_per_mol"] = None
            grid.append(row)

    basins = []
    for root in ordered_roots:
        basin_id = basin_by_root[root]
        population = sum(
            1 for assignment in point_assignments if assignment == basin_id
        )
        basins.append({
            "basin_id": basin_id,
            "root_x_bin": root[0],
            "root_y_bin": root[1],
            "root_x_center_angstrom": x_lower + (root[0] + 0.5) * x_width,
            "root_y_center_angstrom": y_lower + (root[1] + 0.5) * y_width,
            "root_count": counts[root[0]][root[1]],
            "root_surface_count": float(surface_density[root[0], root[1]]),
            "assigned_count": population,
            "assigned_fraction": population / len(points),
        })
    return {
        "bounds": {
            "x_min_angstrom": x_lower,
            "x_max_angstrom": x_upper,
            "y_min_angstrom": y_lower,
            "y_max_angstrom": y_upper,
        },
        "bounds_source": bounds_source,
        "bin_widths_angstrom": {"x": x_width, "y": y_width},
        "smoothing": {
            "kernel": "scipy.ndimage.gaussian_filter",
            "sigma_bins": smoothing_sigma_bins,
            "boundary_mode": "constant_zero_then_mass_renormalized",
            "truncate_sigma": 4.0,
            "raw_counts_preserved": True,
            "basin_eligibility": "raw_count_at_least_minimum_bin_count",
        },
        "grid": grid,
        "basins": basins,
        "point_assignments": point_assignments,
        "assigned_count": sum(value is not None for value in point_assignments),
        "unassigned_count": sum(value is None for value in point_assignments),
    }


def _projection_records(pca_report: Mapping[str, object]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    systems = pca_report.get("systems")
    assert isinstance(systems, list)
    for system in systems:
        assert isinstance(system, dict)
        for replica in system["replicas"]:
            assert isinstance(replica, dict)
            for segment in replica["segments"]:
                assert isinstance(segment, dict)
                projections = segment["projections"]
                assert isinstance(projections, list)
                for row in projections:
                    assert isinstance(row, dict)
                    records.append({
                        "system_id": str(system["system_id"]),
                        "replica_id": str(replica["replica_id"]),
                        "segment_id": str(segment["segment_id"]),
                        **row,
                    })
    return records


def _population_row(assignments: Sequence[Optional[int]], basin_ids: Sequence[int]) -> Dict[str, object]:
    count = len(assignments)
    return {
        "evaluated_count": count,
        "assigned_count": sum(value is not None for value in assignments),
        "unassigned_count": sum(value is None for value in assignments),
        "basin_populations": [
            {
                "basin_id": basin_id,
                "count": sum(value == basin_id for value in assignments),
                "fraction_of_all_evaluated": (
                    sum(value == basin_id for value in assignments) / count if count else None
                ),
            }
            for basin_id in basin_ids
        ],
    }


def _assignment_rows(
    records: Sequence[Mapping[str, object]],
    points: Sequence[Tuple[float, float]],
    assignments: Sequence[Optional[int]],
    smoothing_sigma_bins: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for record, point, assignment in zip(records, points, assignments):
        rows.append({
            "system_id": record["system_id"],
            "replica_id": record["replica_id"],
            "segment_id": record["segment_id"],
            "source_frame_index": record["source_frame_index"],
            **(
                {"member_id": record["member_id"]}
                if "member_id" in record else {}
            ),
            **(
                {"sample_index": record["sample_index"]}
                if "sample_index" in record
                else {"time": record["time"], "time_unit": record["time_unit"]}
            ),
            "pc_x_angstrom": point[0],
            "pc_y_angstrom": point[1],
            "smoothing_sigma_bins": smoothing_sigma_bins,
            "basin_id": assignment,
        })
    return rows


def _population_tables(
    assignment_rows: Sequence[Mapping[str, object]],
    basin_ids: Sequence[int],
    block_size: int,
    include_partial_final_block: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    replica_populations: List[Dict[str, object]] = []
    replica_keys = sorted({
        (str(row["system_id"]), str(row["replica_id"]))
        for row in assignment_rows
    })
    for system_id, replica_id in replica_keys:
        values = [
            row["basin_id"] for row in assignment_rows
            if row["system_id"] == system_id and row["replica_id"] == replica_id
        ]
        replica_populations.append({
            "system_id": system_id,
            "replica_id": replica_id,
            **_population_row(values, basin_ids),  # type: ignore[arg-type]
        })

    block_populations: List[Dict[str, object]] = []
    segment_keys = sorted({
        (
            str(row["system_id"]), str(row["replica_id"]),
            str(row["segment_id"]), str(row.get("member_id", "")),
        )
        for row in assignment_rows
    })
    for system_id, replica_id, segment_id, member_id in segment_keys:
        segment_rows = [
            row for row in assignment_rows
            if row["system_id"] == system_id
            and row["replica_id"] == replica_id
            and row["segment_id"] == segment_id
            and str(row.get("member_id", "")) == member_id
        ]
        for start in range(0, len(segment_rows), block_size):
            block = segment_rows[start : start + block_size]
            if len(block) < block_size and not include_partial_final_block:
                continue
            block_populations.append({
                "system_id": system_id,
                "replica_id": replica_id,
                "segment_id": segment_id,
                **({"member_id": member_id} if member_id else {}),
                "block_index": start // block_size,
                "source_frame_index_start": block[0]["source_frame_index"],
                "source_frame_index_end": block[-1]["source_frame_index"],
                **_population_row(
                    [row["basin_id"] for row in block],  # type: ignore[list-item]
                    basin_ids,
                ),
            })
    return replica_populations, block_populations


def _smoothing_comparison(
    primary: Sequence[Optional[int]],
    alternate: Sequence[Optional[int]],
    alternate_sigma_bins: float,
) -> Dict[str, object]:
    paired = [
        (left, right)
        for left, right in zip(primary, alternate)
        if left is not None and right is not None
    ]
    cross_tabulation = []
    for left in sorted({value[0] for value in paired}):
        for right in sorted({value[1] for value in paired}):
            count = sum(pair == (left, right) for pair in paired)
            if count:
                cross_tabulation.append({
                    "primary_basin_id": left,
                    "alternate_basin_id": right,
                    "count": count,
                })
    return {
        "alternate_smoothing_sigma_bins": alternate_sigma_bins,
        "jointly_assigned_count": len(paired),
        "jointly_assigned_fraction": len(paired) / len(primary) if primary else None,
        "adjusted_rand_index": (
            adjusted_rand_index(
                [int(value[0]) for value in paired],
                [int(value[1]) for value in paired],
            )
            if paired else None
        ),
        "basin_cross_tabulation": cross_tabulation,
    }


def basin_silhouette_report(
    points: Sequence[Tuple[float, float]],
    assignments: Sequence[Optional[int]],
    maximum_exact_observations: int,
    random_seed: int = 0,
) -> Dict[str, object]:
    """Score the geometric separation of assigned FES basin points.

    Basin labels remain defined exclusively by the occupancy/free-energy
    watershed.  Silhouette is a secondary separability diagnostic and cannot
    establish that a basin is a thermodynamic or kinetic state.
    """

    assigned = [
        (point, int(assignment))
        for point, assignment in zip(points, assignments)
        if assignment is not None
    ]
    labels = {label for _, label in assigned}
    base = {
        "diagnostic_role": "secondary_geometric_separability",
        "defines_fes_basins": False,
        "assigned_observation_count": len(assigned),
        "unassigned_observation_count": len(points) - len(assigned),
        "cluster_count": len(labels),
    }
    if len(assigned) < 2 or len(labels) < 2:
        return {
            **base,
            "status": "not_calculable",
            "score": None,
            "reason": "silhouette requires at least two assigned observations and two basins",
        }
    try:
        report = silhouette_score_report(
            [point for point, _ in assigned],
            [label for _, label in assigned],
            maximum_exact_observations,
            random_seed,
        )
    except ClusteringAnalysisError as exc:
        return {**base, "status": "not_calculable", "score": None, "reason": str(exc)}
    return {
        **base,
        "status": "complete",
        **report,
        "interpretation": (
            "Silhouette measures separation of watershed-assigned points in the "
            "selected PCA plane; it is not a free-energy, population, metastability, "
            "or kinetic criterion."
        ),
    }


def pca_fes_basins_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Run common PCA and construct a mode-aware two-dimensional landscape."""

    source = Path(project_path).expanduser().resolve(strict=False)
    cached = load_cached_project_report(
        "pca_fes_basins",
        source,
        hash_content=hash_content,
        error_type=PCAFESAnalysisError,
    )
    if cached is not None:
        return cached
    project = load_json(source)
    settings = _settings(project)
    sampling_mode = str(project["sampling_mode"])
    weights = project.get("statistical_weights")
    if weights is not None:
        raise PCAFESAnalysisError(
            "statistical_weights are not implemented for pca_fes_basins; refusing to ignore them"
        )
    if sampling_mode in {"BIASED_MD", "ENHANCED_SAMPLING"}:
        raise PCAFESAnalysisError(
            f"{sampling_mode} requires a validated reweighting model before an FES can be calculated"
        )
    if sampling_mode not in {"UNBIASED_MD", "AI_ENSEMBLE"}:
        raise PCAFESAnalysisError(
            "pca_fes_basins currently supports UNBIASED_MD and non-thermodynamic AI_ENSEMBLE occupancy only"
        )
    temperature: Optional[float]
    issues: List[Dict[str, object]] = []
    if sampling_mode == "UNBIASED_MD":
        raw_temperature = project.get("temperature_kelvin")
        if (
            isinstance(raw_temperature, bool)
            or not isinstance(raw_temperature, (int, float))
            or not math.isfinite(float(raw_temperature))
            or float(raw_temperature) <= 0.0
        ):
            raise PCAFESAnalysisError(
                "positive temperature_kelvin is required for an UNBIASED_MD free-energy surface"
            )
        temperature = float(raw_temperature)
        landscape_kind = "thermodynamic_relative_free_energy"
    else:
        temperature = None
        landscape_kind = "nonthermodynamic_relative_occupancy"
        issues.append({
            "severity": "warning",
            "code": "AI_ENSEMBLE_NOT_THERMODYNAMIC",
            "location": str(source),
            "message": (
                "AI ensemble counts are reported as relative occupancy only; "
                "they are not converted to kcal/mol or called thermodynamic populations"
            ),
        })

    pca_report = common_pca_project(source, hash_content=hash_content)
    pca_issues = pca_report.get("issues", [])
    if isinstance(pca_issues, list):
        issues.extend(
            issue for issue in pca_issues if isinstance(issue, dict)
        )
    records = _projection_records(pca_report)
    x_index = int(settings["x_component"]) - 1
    y_index = int(settings["y_component"]) - 1
    points = []
    for record in records:
        scores = record["scores_angstrom"]
        assert isinstance(scores, list)
        if max(x_index, y_index) >= len(scores):
            raise PCAFESAnalysisError(
                "selected landscape component exceeds the components returned by common_pca"
            )
        points.append((float(scores[x_index]), float(scores[y_index])))
    if settings["binning_rule"] == "explicit":
        bins_x = int(settings["bins_x"])
        bins_y = int(settings["bins_y"])
        binning = {
            "rule": "explicit", "bins_x": bins_x, "bins_y": bins_y,
            "axis_contract": "each PCA coordinate uses its declared bin count",
        }
    else:
        binning = select_bin_counts(
            points,
            str(settings["binning_rule"]),
            float(settings["padding_fraction"]),
            int(settings["minimum_bins_per_axis"]),
            int(settings["maximum_bins_per_axis"]),
        )
        bins_x = int(binning["bins_x"])
        bins_y = int(binning["bins_y"])
        if bins_x * bins_y > int(settings["maximum_grid_cells"]):
            raise PCAFESAnalysisError(
                "automatic binning exceeds maximum_grid_cells; tighten per-axis gates"
            )
    smoothing_landscapes: List[Dict[str, object]] = []
    for sigma in settings["smoothing_sigmas_bins"]:  # type: ignore[union-attr]
        smoothing_sigma_bins = float(sigma)
        landscape = build_landscape(
            points,
            bins_x=bins_x,
            bins_y=bins_y,
            padding_fraction=float(settings["padding_fraction"]),
            minimum_bin_count=int(settings["minimum_bin_count"]),
            temperature_kelvin=temperature,
            smoothing_sigma_bins=smoothing_sigma_bins,
        )
        landscape["binning"] = binning
        shared_bounds = landscape["bounds"]
        assignments = landscape.pop("point_assignments")
        assert isinstance(assignments, list)
        basin_silhouette = basin_silhouette_report(
            points,
            assignments,  # type: ignore[arg-type]
            int(settings["maximum_silhouette_observations"]),
            int(settings["silhouette_random_seed"]),
        )
        basin_ids = [
            int(basin["basin_id"])
            for basin in landscape["basins"]  # type: ignore[union-attr]
        ]
        assignment_rows = _assignment_rows(
            records, points, assignments, smoothing_sigma_bins
        )
        replica_populations, block_populations = _population_tables(
            assignment_rows,
            basin_ids,
            int(settings["population_block_size_frames"]),
            bool(settings["include_partial_final_block"]),
        )
        per_system_landscapes = []
        for system_id in sorted({str(record["system_id"]) for record in records}):
            system_indices = [
                index for index, record in enumerate(records)
                if str(record["system_id"]) == system_id
            ]
            system_records = [records[index] for index in system_indices]
            system_points = [points[index] for index in system_indices]
            if len(system_points) < 2:
                issues.append({
                    "severity": "warning",
                    "code": "PER_SYSTEM_LANDSCAPE_INSUFFICIENT_POINTS",
                    "location": system_id,
                    "message": (
                        "Per-system landscape was not constructed because fewer "
                        "than two projected frames were available"
                    ),
                })
                per_system_landscapes.append({
                    "system_id": system_id,
                    "technical_status": "not_constructed",
                    "reason": "fewer_than_two_projected_frames",
                    "normalization_scope": "within_system",
                    "common_grid_with_pooled_landscape": True,
                })
                continue
            try:
                system_landscape = build_landscape(
                    system_points,
                    bins_x=bins_x,
                    bins_y=bins_y,
                    padding_fraction=float(settings["padding_fraction"]),
                    minimum_bin_count=int(settings["minimum_bin_count"]),
                    temperature_kelvin=temperature,
                    smoothing_sigma_bins=smoothing_sigma_bins,
                    fixed_bounds=shared_bounds,  # type: ignore[arg-type]
                )
            except PCAFESAnalysisError as exc:
                issues.append({
                    "severity": "warning",
                    "code": "PER_SYSTEM_LANDSCAPE_GATE_NOT_MET",
                    "location": system_id,
                    "message": str(exc),
                })
                per_system_landscapes.append({
                    "system_id": system_id,
                    "technical_status": "not_constructed",
                    "reason": str(exc),
                    "normalization_scope": "within_system",
                    "common_grid_with_pooled_landscape": True,
                })
                continue
            system_landscape["binning"] = binning
            system_assignments = system_landscape.pop("point_assignments")
            assert isinstance(system_assignments, list)
            system_basin_silhouette = basin_silhouette_report(
                system_points,
                system_assignments,  # type: ignore[arg-type]
                int(settings["maximum_silhouette_observations"]),
                int(settings["silhouette_random_seed"]),
            )
            system_basin_ids = [
                int(basin["basin_id"])
                for basin in system_landscape["basins"]  # type: ignore[union-attr]
            ]
            system_assignment_rows = _assignment_rows(
                system_records,
                system_points,
                system_assignments,
                smoothing_sigma_bins,
            )
            system_replica_populations, system_block_populations = _population_tables(
                system_assignment_rows,
                system_basin_ids,
                int(settings["population_block_size_frames"]),
                bool(settings["include_partial_final_block"]),
            )
            per_system_landscapes.append({
                "system_id": system_id,
                "technical_status": "complete",
                "normalization_scope": "within_system",
                "common_grid_with_pooled_landscape": True,
                "landscape": system_landscape,
                "basin_silhouette": system_basin_silhouette,
                "frame_assignments": system_assignment_rows,
                "replica_populations": system_replica_populations,
                "block_populations": system_block_populations,
            })
        smoothing_landscapes.append({
            "smoothing_sigma_bins": smoothing_sigma_bins,
            "landscape": landscape,
            "basin_silhouette": basin_silhouette,
            "frame_assignments": assignment_rows,
            "replica_populations": replica_populations,
            "block_populations": block_populations,
            "state_population_comparison": summarize_state_populations(
                assignment_rows, "basin_id"
            ),
            "per_system_landscapes": per_system_landscapes,
        })

    primary_sigma = float(settings["primary_smoothing_sigma_bins"])
    primary = next(
        row for row in smoothing_landscapes
        if row["smoothing_sigma_bins"] == primary_sigma
    )
    primary_assignments = [
        row["basin_id"] for row in primary["frame_assignments"]  # type: ignore[union-attr]
    ]
    smoothing_sensitivity = [
        _smoothing_comparison(
            primary_assignments,  # type: ignore[arg-type]
            [
                row["basin_id"]
                for row in alternate["frame_assignments"]  # type: ignore[union-attr]
            ],
            float(alternate["smoothing_sigma_bins"]),
        )
        for alternate in smoothing_landscapes
        if alternate is not primary
    ]

    error_count = sum(issue["severity"] == "error" for issue in issues)
    warning_count = sum(issue["severity"] == "warning" for issue in issues)
    return {
        "module_id": "pca_fes_basins",
        "technical_status": "failed" if error_count else "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": pca_report["project_manifest_sha256"],
        "system_manifest_path": pca_report["system_manifest_path"],
        "system_manifest_sha256": pca_report["system_manifest_sha256"],
        "input_content_signature_sha256": pca_report["input_content_signature_sha256"],
        "contract_signature_sha256": pca_report["contract_signature_sha256"],
        "content_hashes_included": hash_content,
        "sampling_mode": sampling_mode,
        "landscape_kind": landscape_kind,
        "temperature_kelvin": temperature,
        "settings": settings,
        "pca_basis": {
            "module_id": "common_pca",
            "basis_weighting": pca_report["basis"]["basis_weighting"],  # type: ignore[index]
            "x_component": settings["x_component"],
            "y_component": settings["y_component"],
        },
        "primary_smoothing_sigma_bins": primary_sigma,
        "landscape": primary["landscape"],
        "basin_silhouette": primary["basin_silhouette"],
        "frame_assignments": primary["frame_assignments"],
        "replica_populations": primary["replica_populations"],
        "block_populations": primary["block_populations"],
        "state_population_comparison": summarize_state_populations(
            primary["frame_assignments"], "basin_id"  # type: ignore[arg-type]
        ),
        "per_system_landscapes": primary["per_system_landscapes"],
        "smoothing_landscapes": smoothing_landscapes,
        "smoothing_sensitivity": smoothing_sensitivity,
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": issues,
        "limitations": [
            "Histogram edges, resolution, minimum occupancy, PCA components, basis weighting, and Gaussian smoothing are analysis choices requiring sensitivity runs.",
            "Gaussian smoothing changes density minima and catchments but never changes or substitutes for raw frame counts.",
            "Each per-system surface is normalized independently on the pooled PCA grid; relative free-energy or occupancy offsets must not be compared between systems.",
            "Per-system basin identifiers describe system-conditional catchments and do not imply correspondence to pooled or other-system basin identifiers.",
            "The deterministic steepest-occupancy watershed defines descriptive basins; it does not prove metastability or kinetics.",
            "FES-basin silhouette is a secondary PCA-plane separability diagnostic and never defines or selects a free-energy basin.",
            "Frame and block populations are descriptive and are not automatically independent uncertainty units.",
            "AI_ENSEMBLE counts are not thermodynamic populations and are never converted to free energies by this module.",
            "Biased and enhanced-sampling inputs fail closed until a validated weighting implementation exists.",
            "Technical completion does not establish convergence, adequate sampling, mechanisms, or scientific validity.",
        ],
    }


def pca_fes_basins_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Return a machine-readable failure rather than an uncaught exception."""

    try:
        return pca_fes_basins_project(project_path, hash_content=hash_content)
    except (ManifestValidationError, PCAAnalysisError, PCAFESAnalysisError, OSError) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "pca_fes_basins",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "PCA_FES_INVALID", "message": message}
                for message in messages
            ],
        }
