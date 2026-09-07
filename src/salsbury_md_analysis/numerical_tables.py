"""Lossless tabular views of retained FES cells and DCCM entries.

These writers do not fit, smooth, resample, or normalize the source results.
Matrix rows are streamed so the CSV export needs no second dense matrix.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


FES_CELL_FIELDS = (
    "x_bin", "y_bin", "x_center_angstrom", "y_center_angstrom", "count",
    "probability", "probability_density_per_angstrom2", "surface_count",
    "surface_probability", "surface_probability_density_per_angstrom2",
    "relative_free_energy_kcal_per_mol", "relative_occupancy_score", "basin_id",
)


def _value_status(value: object) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("numerical table value must be a number or null")
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "positive_infinity" if value > 0 else "negative_infinity"
    return "finite"


def _write(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            # Null/nonfinite numbers are blank, never zero. Status columns retain
            # the distinction for the plotted energy/occupancy/correlation value.
            writer.writerow({key: None if isinstance(value, float) and not math.isfinite(value)
                             else value for key, value in row.items()})


def write_fes_grid(path: Path, landscape: Mapping[str, object], metadata: Mapping[str, object]) -> int:
    """Write every supplied grid cell in source order, with labeled coordinates."""
    grid = landscape.get("grid")
    if not isinstance(grid, list) or not grid:
        raise ValueError("FES numerical export requires a nonempty grid")
    coordinates = set()
    extra_fields = set()
    for cell in grid:
        if not isinstance(cell, dict):
            raise ValueError("FES grid cells must be objects")
        indices = (cell.get("x_bin"), cell.get("y_bin"))
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices):
            raise ValueError("FES grid indices must be nonnegative integers")
        if indices in coordinates:
            raise ValueError("duplicate FES grid cell")
        coordinates.add(indices)
        if any(isinstance(v, (dict, list)) for v in cell.values()):
            raise ValueError("FES grid cell contains a non-tabular field")
        extra_fields.update(set(cell) - set(FES_CELL_FIELDS))
    nx = max(i for i, _ in coordinates) + 1
    ny = max(j for _, j in coordinates) + 1
    if len(coordinates) != nx * ny:
        raise ValueError("FES grid is incomplete; missing cells cannot be invented")
    metadata = dict(metadata)
    widths = landscape.get("bin_widths_angstrom", {})
    bounds = landscape.get("bounds", {})
    for axis in ("x", "y"):
        metadata[f"{axis}_bin_width_angstrom"] = widths.get(axis) if isinstance(widths, dict) else None
        for edge in ("min", "max"):
            key = f"{axis}_{edge}_angstrom"
            metadata[key] = bounds.get(key) if isinstance(bounds, dict) else None
    fields = list(metadata) + list(FES_CELL_FIELDS) + sorted(extra_fields) + [
        "free_energy_status", "occupancy_score_status",
    ]
    if len(fields) != len(set(fields)):
        raise ValueError("FES metadata and cell columns overlap")

    def rows():
        for cell in grid:
            yield {
                **metadata, **cell,
                "free_energy_status": _value_status(cell["relative_free_energy_kcal_per_mol"])
                if "relative_free_energy_kcal_per_mol" in cell else "not_reported",
                "occupancy_score_status": _value_status(cell["relative_occupancy_score"])
                if "relative_occupancy_score" in cell else "not_reported",
            }

    _write(path, fields, rows())
    return len(grid)


def validate_dccm_matrix(matrix: object, atoms: Sequence[Mapping[str, object]]) -> int:
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("DCCM numerical export requires a nonempty square matrix")
    n = len(matrix)
    if any(not isinstance(row, list) or len(row) != n for row in matrix):
        raise ValueError("DCCM numerical export requires a square matrix")
    if atoms and len(atoms) != n:
        raise ValueError("DCCM atom identities do not match the matrix dimensions")
    for row in matrix:
        for value in row:
            _value_status(value)
    return n


def _atom_label(atoms: Sequence[Mapping[str, object]], index: int) -> str:
    if not atoms or not isinstance(atoms[index], dict):
        return str(index)
    atom = atoms[index]
    return (f"{atom.get('chain_id') or '_'}:{atom.get('residue_name', 'UNK')}"
            f"{atom.get('residue_number', '?')}{atom.get('insertion_code') or ''}:"
            f"{atom.get('atom_name', '?')}")


def write_dccm_matrix(path: Path, matrix: object, atoms: Sequence[Mapping[str, object]],
                      system_id: str, *, right_matrix: object = None,
                      right_system_id: str | None = None) -> int:
    """Write all n*n entries, including the diagonal and both triangles."""
    n = validate_dccm_matrix(matrix, atoms)
    difference = right_matrix is not None
    if difference and validate_dccm_matrix(right_matrix, atoms) != n:
        raise ValueError("DCCM difference matrices have different dimensions")
    labels = [_atom_label(atoms, i) for i in range(n)]
    system_field = "left_system_id" if difference else "system_id"
    fields = [system_field, "atom_i", "atom_j", "atom_i_label", "atom_j_label"]
    if difference:
        fields += ["right_system_id", "left_correlation", "right_correlation",
                   "left_value_status", "right_value_status", "left_minus_right"]
    else:
        fields += ["correlation"]
    fields += ["value_status"]

    def rows():
        for i, row in enumerate(matrix):
            for j, left in enumerate(row):
                result = {system_field: system_id, "atom_i": i, "atom_j": j,
                          "atom_i_label": labels[i], "atom_j_label": labels[j]}
                status = _value_status(left)
                if difference:
                    right = right_matrix[i][j]
                    right_status = _value_status(right)
                    value = left - right if status == right_status == "finite" else None
                    result.update(right_system_id=right_system_id,
                                  left_correlation=left, right_correlation=right,
                                  left_value_status=status, right_value_status=right_status,
                                  left_minus_right=value,
                                  value_status=_value_status(value))
                else:
                    result.update(correlation=left, value_status=status)
                yield result

    _write(path, fields, rows())
    return n * n
