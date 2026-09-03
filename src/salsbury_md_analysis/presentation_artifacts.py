"""Versioned, portable contracts for human-facing analysis artifacts.

The scientific JSON reports remain the source records.  This module defines
the figures, tables, and structures that a person sees first and gives the
finding picker and interactive viewer stable identifiers for them.
"""

from __future__ import annotations

import hashlib
import csv
import html
import itertools
import json
import math
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .manifests import load_json, sha256_file


PRESENTATION_MANIFEST_SCHEMA = "salsbury-presentation-artifacts-v1"
FINDING_TARGET_SCHEMA = "salsbury-finding-target-v1"
ARTIFACT_TYPES = frozenset({"figure", "table", "structure"})


ANALYSIS_CLASSES = {
    "structural_integrity_qc": "quality_control",
    "convergence_uncertainty": "quality_control",
    "coordinate_cache": "technical_support",
    "integrated_comparison": "comparisons",
    "pca_fes_basins": "free_energy_surfaces",
    "common_pca": "conformational_ensembles",
    "replica_rmsd_rg": "rmsd_and_radius_of_gyration",
    "pooled_rmsf": "rmsf",
    "rmsf_permutation_inference": "comparisons",
    "rmsf_visualization_export": "rmsf",
    "dccm": "coupled_interactions",
    "correlation_networks": "coupled_interactions",
    "generalized_correlation_and_information": "coupled_interactions",
    "information_dynamics": "coupled_interactions",
    "hydrogen_bonds": "hydrogen_bonds",
    "hydrogen_bond_discovery": "hydrogen_bonds",
    "hydrogen_bond_comparison": "hydrogen_bonds",
    "water_mediated_hydrogen_bond_networks": "hydrogen_bonds",
    "solvent_accessible_surface_area": "solvent_exposure",
    "secondary_structure": "secondary_structure",
    "dihedral_distributions": "internal_coordinates",
    "nucleic_acid_geometry": "nucleic_acid_structure",
    "nucleic_acid_structure": "nucleic_acid_structure",
    "ion_atmosphere": "ions_and_solvation",
    "ion_coordination_geometry": "ions_and_solvation",
    "radial_distribution_functions": "ions_and_solvation",
    "state_conditioned_ion_stability": "ions_and_solvation",
    "scalar_feature_distributions": "feature_distributions",
    "scalar_threshold_states": "molecular_states",
    "trajectory_features": "feature_distributions",
    "alternative_clustering": "clustering",
    "clustering_hdbscan": "clustering",
    "clustering_imwkmeans": "clustering",
    "clustering_kmeans": "clustering",
    "pald_community_analysis": "clustering",
    "markov_state_models": "kinetic_models",
    "time_lagged_independent_component_analysis": "kinetic_models",
    "representative_frames": "molecular_states",
    "state_coordinate_exports": "molecular_states",
    "grouped_ml": "machine_learning",
    "grouped_regularized_classification": "machine_learning",
    "optional_observables": "other_observables",
}


class PresentationArtifactError(ValueError):
    """Raised when a human-facing artifact contract is incomplete."""


def _slug(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return text or "all"


def stable_artifact_id(
    artifact_type: str,
    module_id: str,
    purpose: str,
    *,
    context: Optional[Mapping[str, object]] = None,
) -> str:
    """Return a readable ID whose suffix protects against slug collisions."""

    if artifact_type not in ARTIFACT_TYPES:
        raise PresentationArtifactError(
            "artifact_type must be figure, table, or structure"
        )
    if not str(module_id).strip() or not str(purpose).strip():
        raise PresentationArtifactError("module_id and purpose must be nonempty")
    normalized = {
        str(key): context[key] for key in sorted(context or {})
    }
    digest = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return "-".join((
        artifact_type,
        _slug(module_id),
        _slug(purpose),
        digest,
    ))


def finding_target(
    *,
    module_id: str,
    purpose: str,
    context: Optional[Mapping[str, object]] = None,
    preferred_artifact_types: Sequence[str] = ("figure", "table"),
) -> Dict[str, object]:
    """Describe the exact evidence panel a finding should open."""

    invalid = sorted(set(preferred_artifact_types).difference(ARTIFACT_TYPES))
    if invalid:
        raise PresentationArtifactError(
            "unsupported preferred artifact types: " + ", ".join(invalid)
        )
    if not preferred_artifact_types:
        raise PresentationArtifactError(
            "preferred_artifact_types must contain at least one type"
        )
    return {
        "target_schema": FINDING_TARGET_SCHEMA,
        "module_id": str(module_id),
        "purpose": str(purpose),
        "context": dict(context or {}),
        "preferred_artifact_types": list(preferred_artifact_types),
    }


def artifact_record(
    *,
    artifact_type: str,
    module_id: str,
    purpose: str,
    title: str,
    relative_path: str,
    source_report_paths: Sequence[str],
    source_report_sha256: Sequence[str],
    context: Optional[Mapping[str, object]] = None,
    primary_human_output: bool = True,
    media_type: Optional[str] = None,
    analysis_class: Optional[str] = None,
) -> Dict[str, object]:
    """Construct and validate one manifest record."""

    if artifact_type not in ARTIFACT_TYPES:
        raise PresentationArtifactError(
            "artifact_type must be figure, table, or structure"
        )
    if len(source_report_paths) != len(source_report_sha256) or not source_report_paths:
        raise PresentationArtifactError(
            "source report paths and hashes must be nonempty and have equal length"
        )
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise PresentationArtifactError("relative_path must stay within the artifact root")
    if not str(title).strip():
        raise PresentationArtifactError("artifact title must be nonempty")
    normalized_context = dict(context or {})
    return {
        "artifact_id": stable_artifact_id(
            artifact_type, module_id, purpose, context=normalized_context
        ),
        "artifact_type": artifact_type,
        "module_id": str(module_id),
        "analysis_class": str(
            analysis_class or ANALYSIS_CLASSES.get(str(module_id), "other_analyses")
        ),
        "purpose": str(purpose),
        "title": str(title),
        "relative_path": path.as_posix(),
        "media_type": media_type,
        "context": normalized_context,
        "primary_human_output": bool(primary_human_output),
        "source_report_paths": list(map(str, source_report_paths)),
        "source_report_sha256": list(map(str, source_report_sha256)),
    }


def validate_manifest(manifest: Mapping[str, object]) -> None:
    """Fail closed on duplicate IDs, unsafe paths, or incomplete provenance."""

    if manifest.get("presentation_manifest_schema") != PRESENTATION_MANIFEST_SCHEMA:
        raise PresentationArtifactError("unsupported presentation manifest schema")
    rows = manifest.get("artifacts")
    if not isinstance(rows, list):
        raise PresentationArtifactError("presentation manifest artifacts must be an array")
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise PresentationArtifactError(f"artifact {index} must be an object")
        artifact_id = row.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise PresentationArtifactError(f"artifact {index} has no artifact_id")
        if artifact_id in seen:
            raise PresentationArtifactError(f"duplicate artifact_id: {artifact_id}")
        seen.add(artifact_id)
        if row.get("artifact_type") not in ARTIFACT_TYPES:
            raise PresentationArtifactError(f"artifact {artifact_id} has invalid type")
        path = Path(str(row.get("relative_path", "")))
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise PresentationArtifactError(f"artifact {artifact_id} has unsafe path")
        sources = row.get("source_report_paths")
        hashes = row.get("source_report_sha256")
        if (
            not isinstance(sources, list)
            or not isinstance(hashes, list)
            or not sources
            or len(sources) != len(hashes)
        ):
            raise PresentationArtifactError(
                f"artifact {artifact_id} has incomplete source provenance"
            )
        if not isinstance(row.get("artifact_sha256"), str) or len(str(row["artifact_sha256"])) != 64:
            raise PresentationArtifactError(
                f"artifact {artifact_id} lacks its content hash"
            )
        if not isinstance(row.get("artifact_size_bytes"), int) or int(row["artifact_size_bytes"]) < 1:
            raise PresentationArtifactError(
                f"artifact {artifact_id} lacks its byte count"
            )


def write_manifest(
    output_path: Path,
    artifacts: Iterable[Mapping[str, object]],
    *,
    analysis_root: Path,
) -> Dict[str, object]:
    """Write a deterministic manifest after validating the complete record set."""

    artifact_root = output_path.parent
    expanded = []
    for raw in artifacts:
        row = dict(raw)
        artifact_path = artifact_root / str(row.get("relative_path", ""))
        if not artifact_path.is_file():
            raise PresentationArtifactError(
                f"declared artifact is absent: {artifact_path}"
            )
        row["artifact_sha256"] = sha256_file(artifact_path)
        row["artifact_size_bytes"] = artifact_path.stat().st_size
        expanded.append(row)
    rows = sorted(
        expanded,
        key=lambda row: (str(row.get("module_id")), str(row.get("artifact_id"))),
    )
    manifest = {
        "presentation_manifest_schema": PRESENTATION_MANIFEST_SCHEMA,
        "technical_status": "complete",
        "analysis_root": str(Path(analysis_root).resolve()),
        "artifact_count": len(rows),
        "artifacts": rows,
    }
    validate_manifest(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


_WFU_COLORS = (
    "#9E7E38", "#000000", "#A6192E", "#CEB888", "#53565A",
    "#6F5A2A", "#7C1D2D", "#D7C89B", "#7A7D80", "#B99A4A",
)


def human_label(value: object) -> str:
    """Turn internal identifiers into compact labels without losing meaning."""

    text = str(value)
    aliases = {
        "global_common_heavy": "Whole complex",
        "protein_common_heavy": "Protein",
        "nucleic_common_heavy": "Nucleic acid",
        "interface_common_heavy": "Protein-nucleic acid interface",
        "shared_interface_heavy": "Oligomer interface",
        "oligomer_members_pooled": "Pooled oligomer members",
    }
    if text in aliases:
        return aliases[text]
    return re.sub(r"\s+", " ", text.replace("_", " ").replace("-", " ")).strip()


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _svg_document(width: int, height: int, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title">'
        f'<title id="title">{html.escape(title)}</title>'
        '<rect width="100%" height="100%" fill="#FFFFFF"/>'
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#000000}'
        '.axis{stroke:#000;stroke-width:1}.grid{stroke:#D7D7D7;stroke-width:.7}'
        '.small{font-size:11px}.label{font-size:13px}.title{font-size:18px;font-weight:700}'
        '.subtitle{font-size:12px;fill:#53565A}</style>'
        f'{body}</svg>\n'
    )


def _write_svg(path: Path, width: int, height: int, body: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_svg_document(width, height, body, title), encoding="utf-8")


def _state_population_rows(comparison: object) -> List[Dict[str, object]]:
    if not isinstance(comparison, dict):
        return []
    systems = comparison.get("system_populations")
    if not isinstance(systems, list):
        return []
    rows = []
    for system in systems:
        if not isinstance(system, dict):
            continue
        for state in system.get("state_populations", []):
            if not isinstance(state, dict):
                continue
            value = state.get("fraction_of_all_evaluated")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            rows.append({
                "system_id": str(system.get("system_id")),
                "state_id": state.get("state_id"),
                "count": state.get("count"),
                "fraction_of_all_evaluated": float(value),
                "percentage_of_all_evaluated": 100.0 * float(value),
                "evaluated_count": system.get("evaluated_count"),
                "assigned_count": system.get("assigned_count"),
                "assigned_coverage_fraction": system.get("assigned_coverage_fraction"),
            })
    return rows


def _state_population_svg(rows: Sequence[Mapping[str, object]], title: str) -> Tuple[int, int, str]:
    systems = list(dict.fromkeys(str(row["system_id"]) for row in rows))
    states = sorted({int(row["state_id"]) for row in rows})
    width = 980
    row_height = 34
    top = 92
    left = 190
    plot_width = 720
    height = top + row_height * len(systems) + 94
    values = {
        (str(row["system_id"]), int(row["state_id"])): float(
            row["fraction_of_all_evaluated"]
        )
        for row in rows
    }
    body = [
        f'<text class="title" x="24" y="30">{html.escape(title)}</text>',
        '<text class="subtitle" x="24" y="52">Fractions use every evaluated observation; unassigned observations remain visible as unused bar length.</text>',
    ]
    for tick in range(0, 101, 20):
        x = left + plot_width * tick / 100.0
        body.append(f'<line class="grid" x1="{x:.1f}" y1="{top-12}" x2="{x:.1f}" y2="{top+row_height*len(systems)}"/>')
        body.append(f'<text class="small" x="{x:.1f}" y="{top+row_height*len(systems)+20}" text-anchor="middle">{tick}%</text>')
    for row_index, system_id in enumerate(systems):
        y = top + row_index * row_height
        body.append(
            f'<text class="label" x="{left-10}" y="{y+19}" text-anchor="end">'
            f'{html.escape(human_label(system_id))}</text>'
        )
        x = left
        for color_index, state_id in enumerate(states):
            fraction = values.get((system_id, state_id), 0.0)
            segment = plot_width * max(0.0, fraction)
            color = _WFU_COLORS[color_index % len(_WFU_COLORS)]
            body.append(
                f'<rect x="{x:.2f}" y="{y}" width="{segment:.2f}" height="24" '
                f'fill="{color}" data-system="{html.escape(system_id)}" data-state="{state_id}"/>'
            )
            if segment >= 42:
                foreground = "#FFFFFF" if color not in {"#CEB888", "#D7C89B"} else "#000000"
                body.append(
                    f'<text class="small" x="{x+segment/2:.2f}" y="{y+16}" '
                    f'text-anchor="middle" style="fill:{foreground}">{100*fraction:.1f}%</text>'
                )
            x += segment
        body.append(f'<rect x="{left}" y="{y}" width="{plot_width}" height="24" fill="none" stroke="#000"/>')
    legend_y = top + row_height * len(systems) + 52
    cursor = left
    for color_index, state_id in enumerate(states):
        color = _WFU_COLORS[color_index % len(_WFU_COLORS)]
        body.append(f'<rect x="{cursor}" y="{legend_y-12}" width="14" height="14" fill="{color}"/>')
        body.append(f'<text class="small" x="{cursor+20}" y="{legend_y}">State {state_id}</text>')
        cursor += 78
    return width, height, "".join(body)


def _finite(value: object) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _color_ramp(value: float, lower: float, upper: float, *, diverging: bool = False) -> str:
    if upper <= lower:
        position = 0.0
    else:
        position = max(0.0, min(1.0, (value - lower) / (upper - lower)))
    if diverging:
        if position < 0.5:
            local = position * 2.0
            start, end = (49, 90, 138), (255, 255, 255)
        else:
            local = (position - 0.5) * 2.0
            start, end = (255, 255, 255), (166, 25, 46)
    else:
        start, end = (206, 184, 136), (166, 25, 46)
        local = position
    rgb = tuple(round(start[index] + local * (end[index] - start[index])) for index in range(3))
    return "#%02X%02X%02X" % rgb


def _fes_svg(landscape: Mapping[str, object], title: str, x_component: int, y_component: int) -> Tuple[int, int, str]:
    grid = [row for row in landscape.get("grid", []) if isinstance(row, dict)]
    if not grid:
        raise PresentationArtifactError("FES landscape has no grid")
    bins_x = max(int(row["x_bin"]) for row in grid) + 1
    bins_y = max(int(row["y_bin"]) for row in grid) + 1
    key = (
        "relative_free_energy_kcal_per_mol"
        if any("relative_free_energy_kcal_per_mol" in row for row in grid)
        else "relative_occupancy_score"
    )
    values = [_finite(row.get(key)) for row in grid]
    finite = [value for value in values if value is not None]
    if not finite:
        raise PresentationArtifactError("FES landscape contains no finite display values")
    lower, upper = min(finite), max(finite)
    width, height = 850, 720
    left, top, side = 92, 72, 560
    cell_x, cell_y = side / bins_x, side / bins_y
    body = [
        f'<text class="title" x="24" y="30">{html.escape(title)}</text>',
        f'<text class="subtitle" x="24" y="50">Primary surface; {bins_x} by {bins_y} common grid.</text>',
    ]
    by_cell = {(int(row["x_bin"]), int(row["y_bin"])): row for row in grid}
    for x_index in range(bins_x):
        for y_index in range(bins_y):
            row = by_cell.get((x_index, y_index), {})
            value = _finite(row.get(key))
            fill = "#EFEFEF" if value is None else _color_ramp(value, lower, upper)
            x = left + x_index * cell_x
            y = top + (bins_y - y_index - 1) * cell_y
            body.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_x+.2:.2f}" height="{cell_y+.2:.2f}" fill="{fill}"/>')
    body.append(f'<rect x="{left}" y="{top}" width="{side}" height="{side}" fill="none" stroke="#000"/>')
    basins = [row for row in landscape.get("basins", []) if isinstance(row, dict)]
    for basin in basins:
        x = left + (float(basin.get("root_x_bin", 0)) + 0.5) * cell_x
        y = top + (bins_y - float(basin.get("root_y_bin", 0)) - 0.5) * cell_y
        fraction = _finite(basin.get("assigned_fraction"))
        label = f"{basin.get('basin_id')}"
        if fraction is not None:
            label += f" ({fraction:.1%})"
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" fill="#FFFFFF" stroke="#000"/>')
        body.append(f'<text class="small" x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle">{html.escape(label)}</text>')
    body.append(f'<text class="label" x="{left+side/2}" y="{top+side+42}" text-anchor="middle">PC{x_component} (Å)</text>')
    body.append(f'<text class="label" transform="translate(28 {top+side/2}) rotate(-90)" text-anchor="middle">PC{y_component} (Å)</text>')
    legend_x = left + side + 52
    legend_y = top + 38
    legend_height = 360
    steps = 60
    for index in range(steps):
        value = upper - (upper - lower) * index / max(1, steps - 1)
        y = legend_y + legend_height * index / steps
        body.append(f'<rect x="{legend_x}" y="{y:.2f}" width="26" height="{legend_height/steps+1:.2f}" fill="{_color_ramp(value, lower, upper)}"/>')
    legend_label = "Relative free energy (kcal/mol)" if key.startswith("relative_free") else "Relative occupancy score"
    body.append(f'<text class="label" x="{legend_x+13}" y="{legend_y-14}" text-anchor="middle">{html.escape(legend_label)}</text>')
    body.append(f'<text class="small" x="{legend_x+36}" y="{legend_y+5}">{upper:.2f}</text>')
    body.append(f'<text class="small" x="{legend_x+36}" y="{legend_y+legend_height}">{lower:.2f}</text>')
    return width, height, "".join(body)


def _matrix_reduce(matrix: Sequence[Sequence[object]], maximum: int = 96) -> Tuple[List[List[Optional[float]]], int]:
    size = len(matrix)
    stride = max(1, math.ceil(size / maximum))
    reduced = []
    for row_start in range(0, size, stride):
        row_values = []
        for column_start in range(0, size, stride):
            values = []
            for row in matrix[row_start:row_start + stride]:
                if not isinstance(row, list):
                    continue
                for value in row[column_start:column_start + stride]:
                    number = _finite(value)
                    if number is not None:
                        values.append(number)
            row_values.append(sum(values) / len(values) if values else None)
        reduced.append(row_values)
    return reduced, stride


def _matrix_svg(matrix: Sequence[Sequence[object]], title: str, legend_label: str, *, difference: bool) -> Tuple[int, int, str]:
    reduced, stride = _matrix_reduce(matrix)
    size = len(reduced)
    values = [value for row in reduced for value in row if value is not None]
    if not values:
        raise PresentationArtifactError("matrix has no finite values")
    bound = max(abs(value) for value in values) if difference else 1.0
    lower, upper = -bound, bound
    width, height = 820, 720
    left, top, side = 88, 68, 560
    cell = side / size
    body = [
        f'<text class="title" x="24" y="30">{html.escape(title)}</text>',
        f'<text class="subtitle" x="24" y="50">Display block size: {stride} source matrix row(s) per cell.</text>',
    ]
    for row_index, row in enumerate(reduced):
        for column_index, value in enumerate(row):
            fill = "#EFEFEF" if value is None else _color_ramp(value, lower, upper, diverging=True)
            x = left + column_index * cell
            y = top + row_index * cell
            body.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell+.2:.2f}" height="{cell+.2:.2f}" fill="{fill}"/>')
    body.append(f'<rect x="{left}" y="{top}" width="{side}" height="{side}" fill="none" stroke="#000"/>')
    body.append(f'<text class="label" x="{left+side/2}" y="{top+side+40}" text-anchor="middle">Atom or residue index</text>')
    body.append(f'<text class="label" transform="translate(26 {top+side/2}) rotate(-90)" text-anchor="middle">Atom or residue index</text>')
    legend_x = left + side + 55
    legend_y = top + 52
    legend_height = 330
    for index in range(60):
        value = upper - (upper - lower) * index / 59
        y = legend_y + legend_height * index / 60
        body.append(f'<rect x="{legend_x}" y="{y:.2f}" width="26" height="{legend_height/60+1:.2f}" fill="{_color_ramp(value, lower, upper, diverging=True)}"/>')
    body.append(f'<text class="label" x="{legend_x+13}" y="{legend_y-16}" text-anchor="middle">{html.escape(legend_label)}</text>')
    body.append(f'<text class="small" x="{legend_x+36}" y="{legend_y+5}">{upper:+.2f}</text>')
    body.append(f'<text class="small" x="{legend_x+36}" y="{legend_y+legend_height/2+4}">0</text>')
    body.append(f'<text class="small" x="{legend_x+36}" y="{legend_y+legend_height}">{lower:+.2f}</text>')
    return width, height, "".join(body)


def _downsample_xy(
    x_values: Sequence[float], y_values: Sequence[float], maximum: int = 2000
) -> Tuple[List[float], List[float]]:
    count = min(len(x_values), len(y_values))
    if count <= maximum:
        return list(x_values[:count]), list(y_values[:count])
    stride = max(1, math.floor(count / maximum))
    indices = list(range(0, count, stride))[:maximum]
    return [float(x_values[index]) for index in indices], [float(y_values[index]) for index in indices]


def _line_svg(
    series: Sequence[Tuple[str, Sequence[float], Sequence[float]]],
    title: str,
    x_label: str,
    y_label: str,
) -> Tuple[int, int, str]:
    normalized = []
    for label, x_values, y_values in series:
        pairs = [
            (float(x), float(y))
            for x, y in zip(x_values, y_values)
            if _finite(x) is not None and _finite(y) is not None
        ]
        if pairs:
            x, y = zip(*pairs)
            reduced_x, reduced_y = _downsample_xy(x, y)
            normalized.append((str(label), reduced_x, reduced_y))
    if not normalized:
        raise PresentationArtifactError("line figure contains no finite observations")
    all_x = [value for _, x, _ in normalized for value in x]
    all_y = [value for _, _, y in normalized for value in y]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0
    width, height = 980, 660
    left, top, plot_width, plot_height = 92, 72, 690, 500
    body = [
        f'<text class="title" x="24" y="30">{html.escape(title)}</text>',
        '<text class="subtitle" x="24" y="50">Lines are drawn from the values retained in the analysis report.</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        x = left + fraction * plot_width
        y = top + plot_height - fraction * plot_height
        body.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_height}"/>')
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}"/>')
        body.append(f'<text class="small" x="{x:.1f}" y="{top+plot_height+20}" text-anchor="middle">{x_min+(x_max-x_min)*fraction:.3g}</text>')
        body.append(f'<text class="small" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{y_min+(y_max-y_min)*fraction:.3g}</text>')
    for index, (label, x_values, y_values) in enumerate(normalized):
        color = _WFU_COLORS[index % len(_WFU_COLORS)]
        points = " ".join(
            f'{left+(x-x_min)/(x_max-x_min)*plot_width:.2f},{top+plot_height-(y-y_min)/(y_max-y_min)*plot_height:.2f}'
            for x, y in zip(x_values, y_values)
        )
        body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.86"/>')
        legend_y = top + index * 18
        if legend_y < top + plot_height:
            body.append(f'<line x1="{left+plot_width+22}" y1="{legend_y}" x2="{left+plot_width+40}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
            body.append(f'<text class="small" x="{left+plot_width+46}" y="{legend_y+4}">{html.escape(label)}</text>')
    body.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#000"/>')
    body.append(f'<text class="label" x="{left+plot_width/2}" y="{height-34}" text-anchor="middle">{html.escape(x_label)}</text>')
    body.append(f'<text class="label" transform="translate(26 {top+plot_height/2}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>')
    return width, height, "".join(body)


def _bar_svg(
    rows: Sequence[Mapping[str, object]],
    title: str,
    label_key: str,
    value_key: str,
    y_label: str,
    *,
    maximum_rows: int = 60,
) -> Tuple[int, int, str]:
    values = []
    for row in rows[:maximum_rows]:
        value = _finite(row.get(value_key))
        if value is not None:
            values.append((str(row.get(label_key, "")), value))
    if not values:
        raise PresentationArtifactError("bar figure contains no finite observations")
    width = max(840, min(1600, 140 + 32 * len(values)))
    height = 650
    left, top, plot_width, plot_height = 78, 72, width - 140, 470
    lower = min(0.0, min(value for _, value in values))
    upper = max(0.0, max(value for _, value in values))
    if lower == upper:
        upper = lower + 1.0
    baseline = top + plot_height - (0.0 - lower) / (upper - lower) * plot_height
    body = [
        f'<text class="title" x="24" y="30">{html.escape(title)}</text>',
        f'<line class="axis" x1="{left}" y1="{baseline:.1f}" x2="{left+plot_width}" y2="{baseline:.1f}"/>',
    ]
    slot = plot_width / len(values)
    for index, (label, value) in enumerate(values):
        value_y = top + plot_height - (value - lower) / (upper - lower) * plot_height
        y = min(value_y, baseline)
        bar_height = max(1.0, abs(value_y - baseline))
        x = left + index * slot + slot * 0.12
        body.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{slot*0.76:.2f}" height="{bar_height:.2f}" fill="{_WFU_COLORS[index % len(_WFU_COLORS)]}"/>')
        body.append(f'<text class="small" transform="translate({x+slot*.38:.2f} {top+plot_height+12}) rotate(60)" text-anchor="start">{html.escape(label)}</text>')
    for tick in range(6):
        fraction = tick / 5
        y = top + plot_height - fraction * plot_height
        value = lower + fraction * (upper - lower)
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}"/>')
        body.append(f'<text class="small" x="{left-8}" y="{y+4:.1f}" text-anchor="end">{value:.3g}</text>')
    body.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#000"/>')
    body.append(f'<text class="label" transform="translate(22 {top+plot_height/2}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>')
    return width, height, "".join(body)


def _scott_histogram(values: Sequence[float]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    finite = [float(value) for value in values if _finite(value) is not None]
    if not finite:
        return [], {"rule": "scott", "observation_count": 0}
    lower, upper = min(finite), max(finite)
    count = len(finite)
    if count > 1:
        mean = sum(finite) / count
        variance = sum((value - mean) ** 2 for value in finite) / (count - 1)
        sample_sd = math.sqrt(max(0.0, variance))
    else:
        sample_sd = 0.0
    raw_width = 3.5 * sample_sd * count ** (-1.0 / 3.0) if sample_sd > 0.0 else 0.0
    if upper == lower:
        bin_count = 1
        width = 1.0
        lower_edge = lower - 0.5
    else:
        raw_bins = math.ceil((upper - lower) / raw_width) if raw_width > 0.0 else 1
        bin_count = max(5, min(200, raw_bins))
        width = (upper - lower) / bin_count
        lower_edge = lower
    counts = [0] * bin_count
    for value in finite:
        index = min(bin_count - 1, max(0, int((value - lower_edge) / width)))
        counts[index] += 1
    rows = []
    for index, bin_observations in enumerate(counts):
        edge = lower_edge + index * width
        rows.append({
            "bin_id": index + 1,
            "lower_edge_angstrom": edge,
            "upper_edge_angstrom": edge + width,
            "center_angstrom": edge + width / 2.0,
            "count": bin_observations,
            "fraction": bin_observations / count,
        })
    return rows, {
        "rule": "scott", "observation_count": count, "sample_sd_angstrom": sample_sd,
        "raw_rule_width_angstrom": raw_width, "used_bin_width_angstrom": width,
        "bin_count": bin_count,
    }


def _histogram_svg(
    rows: Sequence[Mapping[str, object]], title: str, x_label: str
) -> Tuple[int, int, str]:
    usable = [
        row for row in rows
        if _finite(row.get("lower_edge_angstrom")) is not None
        and _finite(row.get("upper_edge_angstrom")) is not None
        and _finite(row.get("fraction")) is not None
    ]
    if not usable:
        raise PresentationArtifactError("histogram contains no finite bins")
    lower = float(usable[0]["lower_edge_angstrom"])
    upper = float(usable[-1]["upper_edge_angstrom"])
    maximum = max(float(row["fraction"]) for row in usable)
    maximum = maximum if maximum > 0 else 1.0
    width, height = 920, 650
    left, top, plot_width, plot_height = 90, 72, 750, 480
    body = [f'<text class="title" x="24" y="30">{html.escape(title)}</text>']
    for row in usable:
        bin_lower = float(row["lower_edge_angstrom"])
        bin_upper = float(row["upper_edge_angstrom"])
        fraction = float(row["fraction"])
        x = left + (bin_lower - lower) / max(upper - lower, 1e-12) * plot_width
        bar_width = (bin_upper - bin_lower) / max(upper - lower, 1e-12) * plot_width
        bar_height = fraction / maximum * plot_height
        body.append(
            f'<rect x="{x:.2f}" y="{top+plot_height-bar_height:.2f}" '
            f'width="{max(1.0, bar_width):.2f}" height="{bar_height:.2f}" '
            'fill="#9E7E38" stroke="#000000" stroke-width="0.35"/>'
        )
    for tick in range(6):
        fraction = tick / 5
        x = left + fraction * plot_width
        y = top + plot_height - fraction * plot_height
        body.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_height}"/>')
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}"/>')
        body.append(f'<text class="small" x="{x:.1f}" y="{top+plot_height+20}" text-anchor="middle">{lower+(upper-lower)*fraction:.3g}</text>')
        body.append(f'<text class="small" x="{left-8}" y="{y+4:.1f}" text-anchor="end">{maximum*fraction:.3g}</text>')
    body.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#000"/>')
    body.append(f'<text class="label" x="{left+plot_width/2}" y="{height-34}" text-anchor="middle">{html.escape(x_label)}</text>')
    body.append(f'<text class="label" transform="translate(25 {top+plot_height/2}) rotate(-90)" text-anchor="middle">Frame fraction</text>')
    return width, height, "".join(body)


def _report_context(path: Path, report: Mapping[str, object]) -> Dict[str, object]:
    parts = path.parts
    view_id = None
    system_id = None
    if "per-system" in parts:
        index = parts.index("per-system")
        if index + 1 < len(parts):
            system_id = parts[index + 1]
    if "conformational-views" in parts:
        index = parts.index("conformational-views")
        if index + 1 < len(parts):
            view_id = parts[index + 1]
    context = {}
    if system_id:
        context["system_id"] = system_id
        context["analysis_scope"] = "per_system"
    elif view_id:
        context["analysis_scope"] = "pooled_system_comparison"
    if view_id:
        context["view_id"] = view_id
    return context


def _source(path: Path) -> Tuple[List[str], List[str]]:
    return [str(path)], [sha256_file(path)]


def _add_state_population_artifacts(
    output_root: Path,
    path: Path,
    report: Mapping[str, object],
    module_id: str,
    title_prefix: str,
    artifacts: List[Dict[str, object]],
) -> None:
    rows = _state_population_rows(report.get("state_population_comparison"))
    if not rows:
        return
    context = _report_context(path, report)
    directory = output_root / _slug(module_id) / _slug(context.get("view_id", "all"))
    table_path = directory / "state-populations.csv"
    figure_path = directory / "state-populations.svg"
    _write_csv(table_path, (
        "system_id", "state_id", "count", "fraction_of_all_evaluated",
        "percentage_of_all_evaluated", "evaluated_count", "assigned_count",
        "assigned_coverage_fraction",
    ), rows)
    width, height, body = _state_population_svg(rows, f"{title_prefix} populations by system")
    _write_svg(figure_path, width, height, body, f"{title_prefix} populations by system")
    sources, hashes = _source(path)
    common = {
        "module_id": module_id,
        "purpose": "state_populations",
        "source_report_paths": sources,
        "source_report_sha256": hashes,
        "context": {
            **context,
            "system_ids": sorted({str(row["system_id"]) for row in rows}),
            "state_ids": sorted({int(row["state_id"]) for row in rows}),
        },
    }
    artifacts.append(artifact_record(
        artifact_type="table", title=f"{title_prefix} populations by system",
        relative_path=str(table_path.relative_to(output_root)), media_type="text/csv",
        **common,
    ))
    artifacts.append(artifact_record(
        artifact_type="figure", title=f"{title_prefix} populations by system",
        relative_path=str(figure_path.relative_to(output_root)), media_type="image/svg+xml",
        **common,
    ))


def _pca_fes_artifacts(output_root: Path, path: Path, report: Mapping[str, object], artifacts: List[Dict[str, object]]) -> None:
    view = _report_context(path, report)
    view_label = human_label(view.get("view_id", "Whole complex"))
    sigma = report.get("primary_smoothing_sigma_bins")
    landscape = report.get("landscape")
    sources, hashes = _source(path)
    directory = output_root / "pca-fes-basins" / _slug(view.get("view_id", "all"))
    if isinstance(landscape, dict):
        figure_path = directory / "primary-fes.svg"
        basis = report.get("pca_basis")
        x_component = int(basis.get("x_component", 1)) if isinstance(basis, dict) else 1
        y_component = int(basis.get("y_component", 2)) if isinstance(basis, dict) else 2
        title = f"{view_label}: primary PCA free-energy surface"
        width, height, body = _fes_svg(landscape, title, x_component, y_component)
        _write_svg(figure_path, width, height, body, title)
        artifacts.append(artifact_record(
            artifact_type="figure", module_id="pca_fes_basins", purpose="primary_fes",
            title=title, relative_path=str(figure_path.relative_to(output_root)),
            source_report_paths=sources, source_report_sha256=hashes,
            context={**view, "smoothing_sigma_bins": sigma},
            media_type="image/svg+xml",
        ))
    _add_state_population_artifacts(
        output_root, path, report, "pca_fes_basins", f"{view_label} FES basin", artifacts
    )
    sensitivity = report.get("smoothing_sensitivity")
    if isinstance(sensitivity, list):
        rows = [row for row in sensitivity if isinstance(row, dict)]
        rows.insert(0, {
            "smoothing_sigma_bins": sigma,
            "role": "configured_primary",
            "adjusted_rand_to_primary": 1.0,
        })
        fields = sorted({str(key) for row in rows for key in row})
        table_path = directory / "smoothing-sensitivity.csv"
        _write_csv(table_path, fields, rows)
        artifacts.append(artifact_record(
            artifact_type="table", module_id="pca_fes_basins",
            purpose="smoothing_sensitivity", title=f"{view_label}: FES smoothing sensitivity",
            relative_path=str(table_path.relative_to(output_root)),
            source_report_paths=sources, source_report_sha256=hashes,
            context={**view, "primary_smoothing_sigma_bins": sigma}, media_type="text/csv",
            primary_human_output=False,
        ))


def _clustering_artifacts(output_root: Path, path: Path, report: Mapping[str, object], module_id: str, artifacts: List[Dict[str, object]]) -> None:
    view = _report_context(path, report)
    method_label = human_label(module_id.replace("clustering_", ""))
    model = report.get("selected_model")
    if isinstance(model, dict):
        rows = []
        diagnostics = report.get("grid_diagnostics")
        if isinstance(diagnostics, list):
            for row in diagnostics:
                if isinstance(row, dict):
                    rows.append(dict(row))
        if not rows:
            rows = [dict(model)]
        fields = sorted({str(key) for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
        directory = output_root / _slug(module_id) / _slug(view.get("view_id", "all"))
        table_path = directory / "model-selection.csv"
        _write_csv(table_path, fields, rows)
        sources, hashes = _source(path)
        feature_contract = report.get("feature_contract")
        feature_source = None
        if isinstance(feature_contract, dict):
            feature_source = feature_contract.get("feature_source", feature_contract.get("source_module_id"))
        artifacts.append(artifact_record(
            artifact_type="table", module_id=module_id, purpose="model_selection",
            title=f"{method_label}: model selection",
            relative_path=str(table_path.relative_to(output_root)),
            source_report_paths=sources, source_report_sha256=hashes,
            context={**view, "feature_source": feature_source}, media_type="text/csv",
        ))
        plotted = []
        for index, row in enumerate(rows, start=1):
            score = _finite(row.get("silhouette"))
            if score is None:
                continue
            label = row.get("k", row.get("cluster_count", index))
            plotted.append({"partition": f"k={label}", "silhouette": score})
        if plotted:
            figure_path = directory / "model-selection.svg"
            title = f"{method_label}: model selection by silhouette score"
            width, height, body = _bar_svg(
                plotted, title, "partition", "silhouette", "Silhouette score"
            )
            _write_svg(figure_path, width, height, body, title)
            artifacts.append(artifact_record(
                artifact_type="figure", module_id=module_id,
                purpose="model_selection", title=title,
                relative_path=str(figure_path.relative_to(output_root)),
                source_report_paths=sources, source_report_sha256=hashes,
                context={**view, "feature_source": feature_source},
                media_type="image/svg+xml",
            ))
    _add_state_population_artifacts(
        output_root, path, report, module_id, f"{method_label} cluster", artifacts
    )


def _alternative_clustering_artifacts(
    output_root: Path, path: Path, report: Mapping[str, object], artifacts: List[Dict[str, object]]
) -> None:
    rows = []
    for item in report.get("algorithm_results", []):
        if not isinstance(item, dict):
            continue
        silhouette = _finite(item.get("silhouette"))
        if silhouette is None:
            continue
        cluster_sizes = item.get("full_cluster_sizes", item.get("cluster_sizes"))
        rows.append({
            "algorithm": item.get("algorithm"),
            "silhouette": silhouette,
            "cluster_count": len(cluster_sizes) if isinstance(cluster_sizes, list) else item.get("cluster_count"),
            "fit_observation_count": item.get("fit_observation_count"),
            "assignment_observation_count": item.get("assignment_observation_count"),
            "assignment_coverage_fraction": item.get("assignment_coverage_fraction"),
        })
    rows.sort(key=lambda row: float(row["silhouette"]), reverse=True)
    if not rows:
        return
    title = "Alternative clustering methods ranked by silhouette score"
    _register_pair(
        output_root, path, artifacts, module_id="alternative_clustering",
        purpose="model_selection", title=title,
        directory=output_root / "alternative-clustering" / _slug(_report_context(path, report).get("view_id", "all")),
        rows=rows,
        fieldnames=("algorithm", "silhouette", "cluster_count", "fit_observation_count", "assignment_observation_count", "assignment_coverage_fraction"),
        svg=_bar_svg(rows, title, "algorithm", "silhouette", "Silhouette score"),
        context=_report_context(path, report),
    )


def _atom_label(identity: object, index: int) -> str:
    if not isinstance(identity, dict):
        return str(index)
    chain = str(identity.get("chain_id", "_")) or "_"
    residue = str(identity.get("residue_name", "UNK"))
    number = identity.get("residue_number", "?")
    atom = str(identity.get("atom_name", "?"))
    return f"{chain}:{residue}{number}:{atom}"


def _dccm_artifacts(output_root: Path, path: Path, report: Mapping[str, object], artifacts: List[Dict[str, object]]) -> None:
    systems = [row for row in report.get("systems", []) if isinstance(row, dict)]
    atoms = report.get("analysis_atoms")
    atom_rows = atoms if isinstance(atoms, list) else []
    sources, hashes = _source(path)
    matrices: Dict[str, List[List[object]]] = {}
    directory = output_root / "dccm"
    for system in systems:
        system_id = str(system.get("system_id"))
        pooled = system.get("frame_pooled_dccm")
        matrix = pooled.get("matrix") if isinstance(pooled, dict) else None
        if not isinstance(matrix, list):
            continue
        matrices[system_id] = matrix
        figure_path = directory / f"{_slug(system_id)}.svg"
        title = f"{human_label(system_id)} dynamic cross-correlation"
        width, height, body = _matrix_svg(matrix, title, "Correlation", difference=False)
        _write_svg(figure_path, width, height, body, title)
        artifacts.append(artifact_record(
            artifact_type="figure", module_id="dccm", purpose="system_matrix",
            title=title, relative_path=str(figure_path.relative_to(output_root)),
            source_report_paths=sources, source_report_sha256=hashes,
            context={"system_id": system_id}, media_type="image/svg+xml",
        ))
    for left, right in itertools.combinations(sorted(matrices), 2):
        left_matrix, right_matrix = matrices[left], matrices[right]
        size = min(len(left_matrix), len(right_matrix))
        difference: List[List[Optional[float]]] = []
        values = []
        for row_index in range(size):
            row = []
            left_row = left_matrix[row_index]
            right_row = right_matrix[row_index]
            width = min(len(left_row), len(right_row), size)
            for column_index in range(width):
                left_value = _finite(left_row[column_index])
                right_value = _finite(right_row[column_index])
                value = left_value - right_value if left_value is not None and right_value is not None else None
                row.append(value)
                if value is not None and row_index != column_index:
                    values.append((abs(value), value, row_index, column_index, left_value, right_value))
            difference.append(row)
        values.sort(reverse=True)
        top = values[:50]
        pair_context = {"left_system_id": left, "right_system_id": right}
        if top:
            pair_context.update({"atom_i": top[0][2], "atom_j": top[0][3]})
        figure_path = directory / "comparisons" / f"{_slug(left)}-minus-{_slug(right)}.svg"
        table_path = directory / "comparisons" / f"{_slug(left)}-minus-{_slug(right)}.csv"
        title = f"Dynamic cross-correlation difference: {human_label(left)} minus {human_label(right)}"
        width, height, body = _matrix_svg(difference, title, "Correlation difference", difference=True)
        _write_svg(figure_path, width, height, body, title)
        table_rows = [{
            "rank": rank,
            "left_system_id": left,
            "right_system_id": right,
            "atom_i": item[2],
            "atom_j": item[3],
            "atom_i_label": _atom_label(atom_rows[item[2]] if item[2] < len(atom_rows) else None, item[2]),
            "atom_j_label": _atom_label(atom_rows[item[3]] if item[3] < len(atom_rows) else None, item[3]),
            "left_correlation": item[4],
            "right_correlation": item[5],
            "left_minus_right": item[1],
        } for rank, item in enumerate(top, start=1)]
        _write_csv(table_path, (
            "rank", "left_system_id", "right_system_id", "atom_i", "atom_j",
            "atom_i_label", "atom_j_label", "left_correlation",
            "right_correlation", "left_minus_right",
        ), table_rows)
        for artifact_type, purpose, artifact_path, media_type in (
            ("figure", "pairwise_difference", figure_path, "image/svg+xml"),
            ("table", "pairwise_difference", table_path, "text/csv"),
        ):
            artifacts.append(artifact_record(
                artifact_type=artifact_type, module_id="dccm", purpose=purpose,
                title=title, relative_path=str(artifact_path.relative_to(output_root)),
                source_report_paths=sources, source_report_sha256=hashes,
                context=pair_context, media_type=media_type,
            ))


def _register_pair(
    output_root: Path,
    path: Path,
    artifacts: List[Dict[str, object]],
    *,
    module_id: str,
    purpose: str,
    title: str,
    directory: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
    svg: Tuple[int, int, str],
    context: Optional[Mapping[str, object]] = None,
    primary_human_output: bool = True,
) -> None:
    table_path = directory / f"{_slug(purpose)}.csv"
    figure_path = directory / f"{_slug(purpose)}.svg"
    _write_csv(table_path, fieldnames, rows)
    _write_svg(figure_path, *svg, title)
    sources, hashes = _source(path)
    for artifact_type, artifact_path, media_type in (
        ("figure", figure_path, "image/svg+xml"),
        ("table", table_path, "text/csv"),
    ):
        artifacts.append(artifact_record(
            artifact_type=artifact_type,
            module_id=module_id,
            purpose=purpose,
            title=title,
            relative_path=str(artifact_path.relative_to(output_root)),
            source_report_paths=sources,
            source_report_sha256=hashes,
            context=context,
            media_type=media_type,
            primary_human_output=primary_human_output,
        ))


def _rmsd_rg_artifacts(
    output_root: Path, path: Path, report: Mapping[str, object], artifacts: List[Dict[str, object]]
) -> None:
    time_unit = str(report.get("time_unit", "trajectory time"))
    metrics = (
        ("rmsd_angstrom", "RMSD", "RMSD (Å)"),
        ("radius_of_gyration_angstrom", "radius of gyration", "Radius of gyration (Å)"),
    )
    for system in report.get("systems", []):
        if not isinstance(system, dict):
            continue
        system_id = str(system.get("system_id"))
        for metric, label, y_label in metrics:
            series = []
            rows = []
            for replica in system.get("replicas", []):
                if not isinstance(replica, dict):
                    continue
                replica_id = str(replica.get("replica_id"))
                for segment in replica.get("segments", []):
                    if not isinstance(segment, dict):
                        continue
                    segment_id = str(segment.get("segment_id"))
                    values = []
                    times = []
                    for row in segment.get("timeseries", []):
                        if not isinstance(row, dict):
                            continue
                        value, time = _finite(row.get(metric)), _finite(row.get("time"))
                        if value is None or time is None:
                            continue
                        times.append(time)
                        values.append(value)
                        rows.append({
                            "system_id": system_id, "replica_id": replica_id,
                            "segment_id": segment_id, "time": time,
                            "time_unit": time_unit, metric: value,
                        })
                    if values:
                        series.append((f"{replica_id} · {segment_id}", times, values))
            if not rows:
                continue
            title = f"{human_label(system_id)} {label}"
            purpose = f"{metric}_timeseries"
            directory = output_root / "replica-rmsd-rg" / _slug(system_id)
            _register_pair(
                output_root, path, artifacts, module_id="replica_rmsd_rg",
                purpose=purpose, title=title, directory=directory, rows=rows,
                fieldnames=("system_id", "replica_id", "segment_id", "time", "time_unit", metric),
                svg=_line_svg(series, title, f"Time ({time_unit})", y_label),
                context={"system_id": system_id, "metric": metric},
                # RMSD is interpreted from its replica-resolved time series.
                # Radius of gyration is presented first as a Scott-rule
                # distribution; its trajectory trace remains available as a
                # secondary diagnostic.
                primary_human_output=(metric == "rmsd_angstrom"),
            )
            if metric == "radius_of_gyration_angstrom":
                histogram, binning = _scott_histogram(
                    [float(row[metric]) for row in rows]
                )
                if histogram:
                    histogram_title = f"{human_label(system_id)} radius-of-gyration distribution"
                    histogram_context = {
                        "system_id": system_id, "metric": metric,
                        "binning_rule": "scott", "bin_count": binning["bin_count"],
                        "scott_width_angstrom": binning["raw_rule_width_angstrom"],
                    }
                    _register_pair(
                        output_root, path, artifacts, module_id="replica_rmsd_rg",
                        purpose="radius_of_gyration_histogram", title=histogram_title,
                        directory=directory,
                        rows=histogram,
                        fieldnames=(
                            "bin_id", "lower_edge_angstrom", "upper_edge_angstrom",
                            "center_angstrom", "count", "fraction",
                        ),
                        svg=_histogram_svg(
                            histogram, histogram_title, "Radius of gyration (Å)"
                        ),
                        context=histogram_context,
                    )


def _rmsf_artifacts(
    output_root: Path, path: Path, report: Mapping[str, object], artifacts: List[Dict[str, object]]
) -> None:
    profiles: Dict[str, Dict[int, Dict[str, object]]] = {}
    for system in report.get("systems", []):
        if not isinstance(system, dict):
            continue
        system_id = str(system.get("system_id"))
        rows = []
        x_values, y_values = [], []
        for index, atom in enumerate(system.get("atom_statistics", [])):
            if not isinstance(atom, dict):
                continue
            value = _finite(atom.get("frame_pooled_rmsf_angstrom"))
            if value is None:
                continue
            atom_index = int(atom.get("common_atom_index", index))
            x_values.append(float(atom_index))
            y_values.append(value)
            rows.append({
                "system_id": system_id,
                "atom_index": atom_index,
                "atom_label": _atom_label(atom, atom_index),
                "rmsf_angstrom": value,
            })
        if not rows:
            continue
        profiles[system_id] = {int(row["atom_index"]): row for row in rows}
        title = f"{human_label(system_id)} atomic fluctuations"
        _register_pair(
            output_root, path, artifacts, module_id="pooled_rmsf",
            purpose="rmsf_profile", title=title,
            directory=output_root / "pooled-rmsf" / _slug(system_id),
            rows=rows, fieldnames=("system_id", "atom_index", "atom_label", "rmsf_angstrom"),
            svg=_line_svg([("Pooled RMSF", x_values, y_values)], title, "Common atom index", "RMSF (Å)"),
            context={"system_id": system_id},
        )
    for left, right in itertools.combinations(sorted(profiles), 2):
        common = sorted(set(profiles[left]).intersection(profiles[right]))
        rows = [{
            "atom_index": index,
            "atom_label": profiles[left][index]["atom_label"],
            "left_system_id": left, "right_system_id": right,
            "left_rmsf_angstrom": profiles[left][index]["rmsf_angstrom"],
            "right_rmsf_angstrom": profiles[right][index]["rmsf_angstrom"],
            "left_minus_right_rmsf_angstrom": float(profiles[left][index]["rmsf_angstrom"]) - float(profiles[right][index]["rmsf_angstrom"]),
        } for index in common]
        if not rows:
            continue
        title = f"RMSF difference: {human_label(left)} minus {human_label(right)}"
        context = {
            "system_ids": [left, right], "left_system_id": left,
            "right_system_id": right,
        }
        _register_pair(
            output_root, path, artifacts, module_id="pooled_rmsf",
            purpose="pairwise_comparison", title=title,
            directory=output_root / "comparisons" / "pooled-rmsf" / f"{_slug(left)}-and-{_slug(right)}",
            rows=rows,
            fieldnames=("atom_index", "atom_label", "left_system_id", "right_system_id", "left_rmsf_angstrom", "right_rmsf_angstrom", "left_minus_right_rmsf_angstrom"),
            svg=_line_svg(
                [("RMSF difference", [float(row["atom_index"]) for row in rows], [float(row["left_minus_right_rmsf_angstrom"]) for row in rows])],
                title, "Common atom index", "RMSF difference (Å)",
            ),
            context=context,
        )


def _convergence_artifacts(
    output_root: Path, path: Path, report: Mapping[str, object], artifacts: List[Dict[str, object]]
) -> None:
    rows = []
    for item in report.get("series_diagnostics", []):
        if not isinstance(item, dict):
            continue
        ess = item.get("effective_sample_size")
        value = _finite(ess.get("estimate") if isinstance(ess, dict) else ess)
        if value is None and isinstance(ess, dict):
            value = _finite(ess.get("value"))
        if value is None:
            continue
        label = " · ".join((str(item.get("system_id")), str(item.get("replica_id")), str(item.get("metric"))))
        rows.append({
            "series": label, "system_id": item.get("system_id"),
            "replica_id": item.get("replica_id"), "metric": item.get("metric"),
            "effective_sample_size": value,
            "split_mean_difference_in_sd": item.get("split_mean_difference_in_sd"),
        })
    if not rows:
        return
    title = "Sampling diagnostics by system, replica, and metric"
    _register_pair(
        output_root, path, artifacts, module_id="convergence_uncertainty",
        purpose="effective_sample_size", title=title,
        directory=output_root / "quality-control" / "sampling-diagnostics",
        rows=rows, fieldnames=("series", "system_id", "replica_id", "metric", "effective_sample_size", "split_mean_difference_in_sd"),
        svg=_bar_svg(rows, title, "series", "effective_sample_size", "Effective sample size"),
        context={"diagnostic": "effective_sample_size"},
    )


def _qc_artifacts(
    output_root: Path, path: Path, report: Mapping[str, object], artifacts: List[Dict[str, object]]
) -> None:
    rows = []
    counts = []
    for system in report.get("systems", []):
        if not isinstance(system, dict):
            continue
        system_id = str(system.get("system_id"))
        count = 0
        for replica in system.get("replicas", []):
            if not isinstance(replica, dict):
                continue
            replica_id = str(replica.get("replica_id"))
            findings = replica.get("findings", replica.get("qc_findings", []))
            if isinstance(findings, list):
                for finding in findings:
                    if isinstance(finding, dict):
                        rows.append({"system_id": system_id, "replica_id": replica_id, **finding})
                        count += 1
        counts.append({"system_id": system_id, "finding_count": count})
    if not counts:
        return
    if not rows:
        rows = [{"system_id": row["system_id"], "replica_id": "", "finding": "No findings observed"} for row in counts]
    fields = sorted({str(key) for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    title = "Structural-integrity QC findings by system"
    _register_pair(
        output_root, path, artifacts, module_id="structural_integrity_qc",
        purpose="qc_findings", title=title,
        directory=output_root / "quality-control" / "structural-integrity",
        rows=rows, fieldnames=fields,
        svg=_bar_svg(counts, title, "system_id", "finding_count", "QC finding count"),
        context={"qc_status": report.get("qc_status")},
    )


def _rdf_artifacts(
    output_root: Path, path: Path, report: Mapping[str, object], artifacts: List[Dict[str, object]]
) -> None:
    base_context = _report_context(path, report)
    system_id = str(base_context.get("system_id", "all systems"))
    for index, feature in enumerate(report.get("feature_reports", [])):
        if not isinstance(feature, dict):
            continue
        bins = [row for row in feature.get("bins", []) if isinstance(row, dict)]
        x_values = [_finite(row.get("center_radius_angstrom")) for row in bins]
        y_values = [_finite(row.get("g_r")) for row in bins]
        pairs = [(x, y) for x, y in zip(x_values, y_values) if x is not None and y is not None]
        if not pairs:
            continue
        feature_id = str(feature.get("feature_id", f"feature-{index+1}"))
        replica_id = str(feature.get("replica_id", f"replica-{index+1}"))
        title = f"{human_label(system_id)} {human_label(replica_id)} radial distribution: {human_label(feature_id)}"
        _register_pair(
            output_root, path, artifacts, module_id="radial_distribution_functions",
            purpose=f"rdf_{feature_id}_{replica_id}", title=title,
            directory=output_root / "radial-distribution-functions" / _slug(system_id) / _slug(replica_id),
            rows=bins, fieldnames=sorted({str(key) for row in bins for key in row}),
            svg=_line_svg([(human_label(feature_id), [x for x, _ in pairs], [y for _, y in pairs])], title, "Radius (Å)", "g(r)"),
            context={**base_context, "feature_id": feature_id, "replica_id": replica_id},
        )


def _scalar_distribution_artifacts(
    output_root: Path, path: Path, report: Mapping[str, object], artifacts: List[Dict[str, object]]
) -> None:
    base_context = _report_context(path, report)
    system_id = str(base_context.get("system_id", "all systems"))
    for index, distribution in enumerate(report.get("distribution_reports", [])):
        if not isinstance(distribution, dict):
            continue
        rows = [row for row in distribution.get("histogram", []) if isinstance(row, dict)]
        pairs = [(_finite(row.get("center")), _finite(row.get("fraction"))) for row in rows]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if not pairs:
            continue
        feature_id = str(distribution.get("feature_id", f"feature-{index+1}"))
        rule = distribution.get("binning", {}).get("rule") if isinstance(distribution.get("binning"), dict) else None
        title = f"{human_label(system_id)} distribution: {human_label(feature_id)}"
        if rule:
            title += f" ({human_label(rule)} bins)"
        _register_pair(
            output_root, path, artifacts, module_id="scalar_feature_distributions",
            purpose=f"distribution_{feature_id}", title=title,
            directory=output_root / "scalar-feature-distributions" / _slug(system_id),
            rows=rows, fieldnames=sorted({str(key) for row in rows for key in row}),
            svg=_line_svg([(human_label(feature_id), [x for x, _ in pairs], [y for _, y in pairs])], title, human_label(feature_id), "Frame fraction"),
            context={**base_context, "feature_id": feature_id, "binning_rule": rule},
        )


def _distribution_report_artifacts(
    output_root: Path, path: Path, report: Mapping[str, object], module_id: str,
    artifacts: List[Dict[str, object]],
) -> None:
    context = _report_context(path, report)
    for index, distribution in enumerate(report.get("distribution_reports", [])):
        if not isinstance(distribution, dict) or distribution.get("status") == "not_estimable":
            continue
        histogram = [row for row in distribution.get("histogram", []) if isinstance(row, dict)]
        pairs = [(_finite(row.get("center")), _finite(row.get("fraction"))) for row in histogram]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if not pairs:
            continue
        metric = str(distribution.get("metric_id", distribution.get("feature_id", f"metric-{index+1}")))
        title = f"{human_label(module_id)} distribution: {human_label(metric)}"
        _register_pair(
            output_root, path, artifacts, module_id=module_id,
            purpose=f"distribution_{metric}", title=title,
            directory=output_root / _slug(module_id) / "distributions",
            rows=histogram,
            fieldnames=sorted({str(key) for row in histogram for key in row}),
            svg=_line_svg(
                [(human_label(metric), [x for x, _ in pairs], [y for _, y in pairs])],
                title, human_label(metric), "Frame fraction",
            ),
            context={**context, "metric_id": metric, "binning_rule": distribution.get("binning", {}).get("rule") if isinstance(distribution.get("binning"), dict) else None},
        )


def _sasa_artifacts(
    output_root: Path, path: Path, report: Mapping[str, object], artifacts: List[Dict[str, object]]
) -> None:
    for replica in report.get("replicas", []):
        if not isinstance(replica, dict):
            continue
        system_id, replica_id = str(replica.get("system_id")), str(replica.get("replica_id"))
        context = {"system_id": system_id, "replica_id": replica_id}
        directory = output_root / "solvent-accessible-surface-area" / _slug(system_id) / _slug(replica_id)
        distribution = replica.get("total_sasa_distribution")
        if isinstance(distribution, dict):
            histogram = [row for row in distribution.get("histogram", []) if isinstance(row, dict)]
            pairs = [(_finite(row.get("center")), _finite(row.get("fraction"))) for row in histogram]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if pairs:
                title = f"{human_label(system_id)} {human_label(replica_id)} total SASA distribution"
                _register_pair(
                    output_root, path, artifacts, module_id="solvent_accessible_surface_area",
                    purpose="total_sasa_distribution", title=title, directory=directory,
                    rows=histogram, fieldnames=sorted({str(key) for row in histogram for key in row}),
                    svg=_line_svg([("SASA", [x for x, _ in pairs], [y for _, y in pairs])], title, "Total SASA (Å²)", "Frame fraction"),
                    context={**context, "binning_rule": distribution.get("binning", {}).get("rule") if isinstance(distribution.get("binning"), dict) else None},
                )
        residue_rows = []
        for residue in replica.get("per_residue_summaries", []):
            if not isinstance(residue, dict):
                continue
            summary = residue.get("summary_angstrom2")
            mean = _finite(summary.get("mean") if isinstance(summary, dict) else None)
            if mean is None:
                continue
            label = f"{residue.get('chain_id', '_')}:{residue.get('residue_name', '')}{residue.get('residue_number', '')}{residue.get('insertion_code', '')}"
            residue_rows.append({"residue": label, "mean_sasa_angstrom2": mean, **{str(k): v for k, v in residue.items() if k != "summary_angstrom2"}})
        if residue_rows:
            title = f"{human_label(system_id)} {human_label(replica_id)} residue SASA"
            _register_pair(
                output_root, path, artifacts, module_id="solvent_accessible_surface_area",
                purpose="residue_sasa", title=title, directory=directory,
                rows=residue_rows, fieldnames=sorted({str(key) for row in residue_rows for key in row}),
                svg=_bar_svg(residue_rows, title, "residue", "mean_sasa_angstrom2", "Mean SASA (Å²)", maximum_rows=60),
                context=context,
            )


def _secondary_structure_artifacts(
    output_root: Path, path: Path, report: Mapping[str, object], artifacts: List[Dict[str, object]]
) -> None:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = {}
    for row in report.get("residue_populations", report.get("populations", [])):
        if isinstance(row, dict):
            groups.setdefault((str(row.get("system_id")), str(row.get("replica_id")), str(row.get("chain_id"))), []).append(row)
    for (system_id, replica_id, chain_id), group in groups.items():
        group = sorted(group, key=lambda row: int(row.get("dssp_sequential_residue_number", row.get("original_residue_number", 0))))
        codes = sorted({str(code) for row in group for code in (row.get("code_fractions", {}) if isinstance(row.get("code_fractions"), dict) else {})})
        x_values = [float(row.get("original_residue_number", index + 1)) for index, row in enumerate(group)]
        series = []
        table_rows = []
        for code in codes:
            fractions = [float(row.get("code_fractions", {}).get(code, 0.0)) for row in group]
            series.append((code, x_values, fractions))
        for row in group:
            base = {key: value for key, value in row.items() if key not in {"code_counts", "code_fractions"}}
            for code in codes:
                base[f"fraction_{code}"] = row.get("code_fractions", {}).get(code, 0.0)
            table_rows.append(base)
        if not series:
            continue
        title = f"{human_label(system_id)} {human_label(replica_id)} chain {chain_id} secondary structure"
        _register_pair(
            output_root, path, artifacts, module_id="secondary_structure",
            purpose="residue_state_fractions", title=title,
            directory=output_root / "secondary-structure" / _slug(system_id) / _slug(replica_id) / _slug(chain_id),
            rows=table_rows, fieldnames=sorted({str(key) for row in table_rows for key in row}),
            svg=_line_svg(series, title, "Residue number", "Frame fraction"),
            context={"system_id": system_id, "replica_id": replica_id, "chain_id": chain_id},
        )


def _occupancy_artifacts(
    output_root: Path, path: Path, report: Mapping[str, object], module_id: str,
    row_key: str, id_keys: Sequence[str], artifacts: List[Dict[str, object]],
) -> None:
    source_rows = [row for row in report.get(row_key, []) if isinstance(row, dict)]
    if not source_rows:
        return
    rows = []
    for row in source_rows:
        value = _finite(row.get("occupancy_fraction", row.get("bound_fraction", row.get("inner_shell_occupancy"))))
        if value is None:
            continue
        label = " · ".join(str(row.get(key, "")) for key in id_keys if row.get(key) is not None)
        rows.append({"feature": label, "occupancy_fraction": value, **row})
    rows.sort(key=lambda row: float(row["occupancy_fraction"]), reverse=True)
    if not rows:
        return
    title = f"{human_label(module_id)} occupancies"
    _register_pair(
        output_root, path, artifacts, module_id=module_id,
        purpose=f"{row_key}_occupancy", title=title,
        directory=output_root / _slug(module_id) / "occupancies",
        rows=rows, fieldnames=sorted({str(key) for row in rows for key in row if not isinstance(row[key], (dict, list))}),
        svg=_bar_svg(rows, title, "feature", "occupancy_fraction", "Frame fraction", maximum_rows=50),
        context=_report_context(path, report),
    )


def _dihedral_artifacts(
    output_root: Path, path: Path, report: Mapping[str, object], artifacts: List[Dict[str, object]]
) -> None:
    summaries = [row for row in report.get("circular_summaries", []) if isinstance(row, dict)]
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = {}
    for row in summaries:
        grouped.setdefault((str(row.get("system_id")), str(row.get("replica_id")), str(row.get("angle_type"))), []).append(row)
    for (system_id, replica_id, angle_type), group in grouped.items():
        rows = []
        for item in sorted(group, key=lambda row: int(row.get("residue_number", 0))):
            mean = _finite(item.get("mean_angle_degrees"))
            if mean is None:
                continue
            rows.append({"residue": f"{item.get('chain_id', '_')}:{item.get('residue_number')}{item.get('insertion_code', '')}", "circular_mean_degrees": mean, "circular_variance": item.get("circular_variance")})
        if not rows:
            continue
        title = f"{human_label(system_id)} {human_label(replica_id)} {human_label(angle_type)} dihedrals"
        _register_pair(
            output_root, path, artifacts, module_id="dihedral_distributions",
            purpose=f"{angle_type}_residue_profile", title=title,
            directory=output_root / "dihedral-distributions" / _slug(system_id) / _slug(replica_id),
            rows=rows, fieldnames=("residue", "circular_mean_degrees", "circular_variance"),
            svg=_bar_svg(rows, title, "residue", "circular_mean_degrees", "Circular mean (degrees)"),
            context={"system_id": system_id, "replica_id": replica_id, "angle_type": angle_type},
        )


def _nested_state_population_artifacts(
    output_root: Path, path: Path, report: Mapping[str, object], module_id: str,
    artifacts: List[Dict[str, object]],
) -> None:
    rows = report.get("state_reports")
    if not isinstance(rows, list):
        return
    for index, state_report in enumerate(rows):
        if not isinstance(state_report, dict):
            continue
        synthetic = {
            "state_population_comparison": state_report.get("state_population_comparison")
        }
        feature_id = str(
            state_report.get("feature_id", state_report.get("state_analysis_id", f"state-{index+1}"))
        )
        _add_state_population_artifacts(
            output_root, path, synthetic, module_id,
            f"{human_label(feature_id)} state", artifacts,
        )


def _matrix_report_artifacts(
    output_root: Path, path: Path, report: Mapping[str, object], module_id: str,
    artifacts: List[Dict[str, object]],
) -> None:
    context = _report_context(path, report)
    candidates: List[Tuple[str, List[List[object]], Dict[str, object]]] = []
    if module_id == "generalized_correlation_and_information":
        for system in report.get("systems", []):
            if not isinstance(system, dict):
                continue
            system_id = str(system.get("system_id"))
            for key in ("generalized_correlation", "normalized_mutual_information", "mutual_information_nats"):
                matrix = system.get(key)
                if isinstance(matrix, list) and matrix and isinstance(matrix[0], list):
                    candidates.append((key, matrix, {**context, "system_id": system_id}))
    elif module_id == "information_dynamics":
        analyses = report.get("analyses")
        if isinstance(analyses, dict):
            for analysis_id, analysis in analyses.items():
                if not isinstance(analysis, dict):
                    continue
                preferred = {
                    "lagged_cross_correlation": "lagged_cross_correlation",
                    "transfer_entropy": "transfer_entropy_nats",
                }.get(str(analysis_id))
                keys = ([preferred] if preferred else []) + [
                    key for key in analysis if key != preferred
                ]
                for key in keys:
                    value = analysis.get(key)
                    if (
                        isinstance(value, list) and value
                        and isinstance(value[0], list) and value[0]
                        and _finite(value[0][0]) is not None
                    ):
                        candidates.append((f"{analysis_id}_{key}", value, context))
                        break
    elif module_id in {"grouped_ml", "grouped_regularized_classification"}:
        metrics = report.get("pooled_held_out_metrics")
        if isinstance(metrics, dict):
            matrix = metrics.get("confusion_matrix")
            if isinstance(matrix, list) and matrix and isinstance(matrix[0], list):
                candidates.append(("confusion_matrix", matrix, context))
    sources, hashes = _source(path)
    for name, matrix, matrix_context in candidates:
        title = f"{human_label(module_id)}: {human_label(name)}"
        figure_path = output_root / _slug(module_id) / _slug(matrix_context.get("system_id", matrix_context.get("view_id", "all"))) / f"{_slug(name)}.svg"
        values = [_finite(value) for row in matrix if isinstance(row, list) for value in row]
        difference = any(value is not None and value < 0 for value in values)
        width, height, body = _matrix_svg(matrix, title, human_label(name), difference=difference)
        _write_svg(figure_path, width, height, body, title)
        artifacts.append(artifact_record(
            artifact_type="figure", module_id=module_id, purpose=name, title=title,
            relative_path=str(figure_path.relative_to(output_root)),
            source_report_paths=sources, source_report_sha256=hashes,
            context=matrix_context, media_type="image/svg+xml",
        ))
        table_rows = []
        for row_index, row in enumerate(matrix):
            if not isinstance(row, list):
                continue
            for column_index, value in enumerate(row):
                number = _finite(value)
                if number is not None:
                    table_rows.append({"row_index": row_index, "column_index": column_index, "value": number})
        table_path = figure_path.with_suffix(".csv")
        _write_csv(table_path, ("row_index", "column_index", "value"), table_rows)
        artifacts.append(artifact_record(
            artifact_type="table", module_id=module_id, purpose=name, title=title,
            relative_path=str(table_path.relative_to(output_root)),
            source_report_paths=sources, source_report_sha256=hashes,
            context=matrix_context, media_type="text/csv",
        ))


def _network_artifacts(
    output_root: Path, path: Path, report: Mapping[str, object], artifacts: List[Dict[str, object]]
) -> None:
    for system in report.get("systems", []):
        if not isinstance(system, dict):
            continue
        system_id = str(system.get("system_id"))
        for index, matrix in enumerate(system.get("matrices", [])):
            if not isinstance(matrix, dict):
                continue
            network = matrix.get("network")
            if not isinstance(network, dict):
                continue
            edges = [row for row in network.get("edges", []) if isinstance(row, dict)]
            edges = sorted(edges, key=lambda row: abs(float(row.get("weight", 0.0))), reverse=True)
            rows = [{
                "edge": f"{row.get('node_i')}–{row.get('node_j')}",
                "node_i": row.get("node_i"), "node_j": row.get("node_j"),
                "weight": row.get("weight"), "absolute_weight": row.get("absolute_weight"),
                "sign": row.get("sign"),
            } for row in edges]
            if not rows:
                continue
            kind = str(matrix.get("matrix_kind", f"network-{index+1}"))
            title = f"{human_label(system_id)} strongest {human_label(kind)} network edges"
            _register_pair(
                output_root, path, artifacts, module_id="correlation_networks",
                purpose=f"network_edges_{kind}", title=title,
                directory=output_root / "correlation-networks" / _slug(system_id),
                rows=rows,
                fieldnames=("edge", "node_i", "node_j", "weight", "absolute_weight", "sign"),
                svg=_bar_svg(rows, title, "edge", "weight", "Edge weight", maximum_rows=30),
                context={"system_id": system_id, "matrix_kind": kind},
            )


def _msm_artifacts(
    output_root: Path, path: Path, report: Mapping[str, object], artifacts: List[Dict[str, object]]
) -> None:
    rows = []
    for item in report.get("clustering_state_model_comparison", []):
        if not isinstance(item, dict):
            continue
        rows.append({
            "candidate_id": item.get("candidate_id"),
            "state_count": item.get("state_count"),
            "geometric_score": item.get("geometric_score"),
            "assignment_coverage_fraction": item.get("assignment_coverage_fraction"),
            "kinetic_validation_status": item.get("kinetic_validation_status"),
        })
    fes = report.get("fes_state_model")
    if isinstance(fes, dict):
        rows.append({
            "candidate_id": fes.get("candidate_id", "FES states"),
            "state_count": fes.get("state_count"),
            "geometric_score": fes.get("geometric_score"),
            "assignment_coverage_fraction": fes.get("assignment_coverage_fraction"),
            "kinetic_validation_status": fes.get("kinetic_validation_status"),
        })
    if not rows:
        return
    title = "Markov-state candidate comparison"
    _register_pair(
        output_root, path, artifacts, module_id="markov_state_models",
        purpose="state_model_comparison", title=title,
        directory=output_root / "markov-state-models" / _slug(_report_context(path, report).get("view_id", "all")),
        rows=rows,
        fieldnames=("candidate_id", "state_count", "geometric_score", "assignment_coverage_fraction", "kinetic_validation_status"),
        svg=_bar_svg(rows, title, "candidate_id", "geometric_score", "Geometric score"),
        context=_report_context(path, report),
    )


def _export_inventory_artifacts(
    output_root: Path, path: Path, report: Mapping[str, object], module_id: str,
    artifacts: List[Dict[str, object]],
) -> None:
    if module_id == "representative_frames":
        source_rows = [row for row in report.get("representatives", []) if isinstance(row, dict)]
        rows = [{
            "state": f"{row.get('system_id', 'all')} · {row.get('state_id')}",
            "system_id": row.get("system_id"), "state_id": row.get("state_id"),
            "representative_rank": row.get("representative_rank"),
            "replica_id": row.get("replica_id"), "segment_id": row.get("segment_id"),
            "source_frame_index": row.get("source_frame_index"),
            "distance_to_state_center_squared": row.get("distance_to_state_center_squared"),
            "count": 1,
        } for row in source_rows]
        purpose = "representative_inventory"
        title = "Observed representative frames by molecular state"
    else:
        source_rows = [row for row in report.get("outputs", []) if isinstance(row, dict)]
        rows = [{
            "state": f"{row.get('system_id', 'all')} · {row.get('state_id')}",
            "system_id": row.get("system_id"), "state_id": row.get("state_id"),
            "trajectory_frame_count": row.get("trajectory_frame_count", 0),
            "representative_count": len(row.get("representatives", [])) if isinstance(row.get("representatives"), list) else 0,
            "trajectory_path": row.get("trajectory_path"),
        } for row in source_rows]
        purpose = "state_export_inventory"
        title = "State-coordinate output inventory"
    if not rows:
        return
    value_key = "count" if module_id == "representative_frames" else "trajectory_frame_count"
    _register_pair(
        output_root, path, artifacts, module_id=module_id, purpose=purpose,
        title=title,
        directory=output_root / _slug(module_id) / _slug(_report_context(path, report).get("view_id", "all")),
        rows=rows, fieldnames=sorted({str(key) for row in rows for key in row}),
        svg=_bar_svg(rows, title, "state", value_key, "Output frame count" if value_key != "count" else "Representative count"),
        context=_report_context(path, report),
    )
    if module_id != "state_coordinate_exports":
        return
    ion_report = report.get("state_conditioned_ion_stability")
    if isinstance(ion_report, dict):
        site_rows = []
        for state in ion_report.get("state_reports", []):
            if not isinstance(state, dict):
                continue
            for species in state.get("species_reports", []):
                if not isinstance(species, dict):
                    continue
                for site in species.get("sites", []):
                    if not isinstance(site, dict):
                        continue
                    site_rows.append({
                        "site": f"{state.get('system_id')} · state {state.get('state_id')} · {site.get('site_id')}",
                        "system_id": state.get("system_id"), "state_id": state.get("state_id"),
                        "element": site.get("element"), "site_id": site.get("site_id"),
                        "occupancy_fraction": site.get("occupancy_fraction"),
                        "positional_rmsf_angstrom": site.get("positional_rmsf_angstrom"),
                        "stable_for_default_state_view": site.get("stable_for_default_state_view"),
                    })
        if site_rows:
            title = "State-conditioned ion-site occupancies"
            _register_pair(
                output_root, path, artifacts,
                module_id="state_conditioned_ion_stability",
                purpose="state_ion_sites", title=title,
                directory=output_root / "state-conditioned-ion-stability",
                rows=site_rows,
                fieldnames=("site", "system_id", "state_id", "element", "site_id", "occupancy_fraction", "positional_rmsf_angstrom", "stable_for_default_state_view"),
                svg=_bar_svg(site_rows, title, "site", "occupancy_fraction", "State-conditioned occupancy fraction", maximum_rows=60),
                context=_report_context(path, report),
            )
    export_root_raw = report.get("export_directory")
    export_root = Path(str(export_root_raw)) if export_root_raw else None
    sources, hashes = _source(path)
    for output in source_rows:
        representatives = output.get("representatives")
        if not isinstance(representatives, list):
            continue
        for representative in representatives:
            if not isinstance(representative, dict) or not representative.get("path"):
                continue
            candidate = Path(str(representative["path"]))
            if not candidate.is_absolute() and export_root is not None:
                candidate = export_root / candidate
            if not candidate.is_file():
                continue
            system_id = str(output.get("system_id", representative.get("system_id", "all")))
            state_id = output.get("state_id", representative.get("state_id"))
            rank = representative.get("representative_rank", 1)
            relative = Path("state-structures") / _slug(system_id) / f"state-{state_id}" / f"representative-{rank}.pdb"
            copied = output_root / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, copied)
            artifacts.append(artifact_record(
                artifact_type="structure", module_id="state_coordinate_exports",
                purpose="representative_structure",
                title=f"{human_label(system_id)} state {state_id} representative {rank}",
                relative_path=str(relative), source_report_paths=sources,
                source_report_sha256=hashes,
                context={
                    **_report_context(path, report), "system_id": system_id,
                    "state_id": state_id, "representative_rank": rank,
                    "state_conditioned_stable_ion_count": representative.get("state_conditioned_stable_ion_count", 0),
                },
                media_type="chemical/x-pdb",
            ))


def _integrated_comparison_artifacts(
    output_root: Path, path: Path, report: Mapping[str, object], artifacts: List[Dict[str, object]]
) -> None:
    findings = [row for row in report.get("comparison_findings", []) if isinstance(row, dict)]
    groups: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, object]]] = {}
    for row in findings:
        effect = _finite(row.get("effect_value"))
        systems = tuple(sorted(set(map(str, row.get("system_ids", []))))) if isinstance(row.get("system_ids"), list) else ()
        module_id = str(row.get("module_id", "integrated_comparison"))
        if effect is None or len(systems) < 2:
            continue
        groups.setdefault((module_id, systems), []).append({
            "finding": str(row.get("statement", "")),
            "comparison_family": row.get("comparison_family"),
            "effect_value": effect,
            "absolute_effect_value": abs(effect),
            "p_value": row.get("p_value"),
            "adjusted_p_value": row.get("adjusted_p_value"),
        })
    for (module_id, systems), rows in sorted(groups.items()):
        rows.sort(key=lambda row: float(row["absolute_effect_value"]), reverse=True)
        left, right = systems[0], systems[1]
        context = {
            "system_ids": list(systems),
            **({"left_system_id": left, "right_system_id": right} if len(systems) == 2 else {}),
        }
        # Prefer the module's own presentation view when one exists.  The
        # integrated report is an index of comparisons, not a second owner of
        # the same figure/table path.  Registering both would overwrite the
        # module artifact and create duplicate stable IDs.
        if any(
            artifact.get("module_id") == module_id
            and artifact.get("purpose") == "pairwise_comparison"
            and artifact.get("context") == context
            for artifact in artifacts
        ):
            continue
        for index, row in enumerate(rows, start=1):
            row["finding_number"] = index
            row["finding_label"] = f"Finding {index}"
            row["left_system_id"] = left
            row["right_system_id"] = right
        system_label = ", ".join(human_label(value) for value in systems)
        title = f"{human_label(module_id)} comparison: {system_label}"
        _register_pair(
            output_root, path, artifacts, module_id=module_id,
            purpose="pairwise_comparison", title=title,
            directory=output_root / "comparisons" / _slug(module_id) / _slug("-and-".join(systems)),
            rows=rows,
            fieldnames=("finding_number", "finding_label", "left_system_id", "right_system_id", "comparison_family", "effect_value", "absolute_effect_value", "p_value", "adjusted_p_value", "finding"),
            svg=_bar_svg(rows, title, "finding_label", "effect_value", "Reported effect (see table for units)", maximum_rows=30),
            context=context,
        )
    overview = []
    for (module_id, systems), rows in sorted(groups.items()):
        overview.append({
            "comparison": f"{human_label(module_id)} · {human_label(systems[0])} vs {human_label(systems[1])}",
            "module_id": module_id, "left_system_id": systems[0], "right_system_id": systems[1],
            "finding_count": len(rows),
        })
    if overview:
        title = "Integrated comparison coverage"
        _register_pair(
            output_root, path, artifacts, module_id="integrated_comparison",
            purpose="comparison_coverage", title=title,
            directory=output_root / "comparisons" / "overview",
            rows=overview,
            fieldnames=("comparison", "module_id", "left_system_id", "right_system_id", "finding_count"),
            svg=_bar_svg(overview, title, "comparison", "finding_count", "Candidate comparison count", maximum_rows=60),
            context={"system_count": len(report.get("comparison_system_ids", []))},
        )


def _flatten_numeric_scalars(
    value: object, prefix: str = "", *, depth: int = 0, limit: int = 80
) -> List[Dict[str, object]]:
    if depth > 5:
        return []
    rows: List[Dict[str, object]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"execution_resources", "planner_benchmark", "settings", "thresholds"}:
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            number = _finite(child)
            if number is not None:
                rows.append({"measure": child_prefix, "value": number})
            elif isinstance(child, dict):
                rows.extend(_flatten_numeric_scalars(child, child_prefix, depth=depth + 1, limit=limit))
            if len(rows) >= limit:
                break
    return rows[:limit]


def _find_tabular_rows(value: object, prefix: str = "", *, depth: int = 0) -> Optional[Tuple[str, List[Dict[str, object]]]]:
    if depth > 5:
        return None
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"issues", "limitations", "settings", "execution_resources", "planner_benchmark"}:
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, list) and child and all(isinstance(row, dict) for row in child[:20]):
                rows = []
                for row in child:
                    flat = {str(k): v for k, v in row.items() if not isinstance(v, (dict, list))}
                    if any(_finite(v) is not None for v in flat.values()):
                        rows.append(flat)
                if len(rows) >= 2:
                    return child_prefix, rows
            found = _find_tabular_rows(child, child_prefix, depth=depth + 1)
            if found:
                return found
    return None


def _generic_artifacts(
    output_root: Path, path: Path, report: Mapping[str, object], module_id: str,
    artifacts: List[Dict[str, object]],
) -> None:
    """Create a truthful minimum presentation for modules without a richer adapter."""

    found = _find_tabular_rows(report)
    rows: List[Dict[str, object]]
    purpose: str
    title: str
    label_key: str
    value_key: str
    if found:
        source_name, rows = found
        fields = sorted({str(key) for row in rows for key in row})
        numeric_fields = [
            field for field in fields
            if sum(_finite(row.get(field)) is not None for row in rows) >= 2
        ]
        if not numeric_fields:
            rows = []
        else:
            value_key = numeric_fields[0]
            label_candidates = [field for field in fields if field.endswith("_id") or field in {"name", "metric", "feature", "method"}]
            label_key = label_candidates[0] if label_candidates else "presentation_row"
            for index, row in enumerate(rows, start=1):
                row.setdefault("presentation_row", index)
            purpose = f"reported_{_slug(source_name)}"
            title = f"{human_label(module_id)}: {human_label(source_name.split('.')[-1])}"
    else:
        rows = []
    if not rows:
        rows = _flatten_numeric_scalars(report)
        label_key, value_key = "measure", "value"
        purpose = "reported_measures"
        title = f"{human_label(module_id)} reported measures"
    if not rows:
        raise PresentationArtifactError(
            f"complete report {path} has no supported human-facing numerical output"
        )
    fields = sorted({str(key) for row in rows for key in row})
    _register_pair(
        output_root, path, artifacts, module_id=module_id, purpose=purpose,
        title=title,
        directory=output_root / _slug(module_id) / _slug(_report_context(path, report).get("system_id", _report_context(path, report).get("view_id", "all"))),
        rows=rows, fieldnames=fields,
        svg=_bar_svg(rows, title, label_key, value_key, human_label(value_key), maximum_rows=40),
        context=_report_context(path, report),
    )


def generate_presentation_artifacts(
    analysis_root: Path,
    *,
    output_root: Optional[Path] = None,
) -> Dict[str, object]:
    """Generate primary human figures and tables from complete scientific reports."""

    root = Path(analysis_root).expanduser().resolve(strict=True)
    destination = (
        Path(output_root).expanduser().resolve(strict=False)
        if output_root is not None else root / "presentation-artifacts"
    )
    if destination.exists():
        raise PresentationArtifactError(
            f"presentation artifact root already exists: {destination}"
        )
    destination.mkdir(parents=True)
    artifacts: List[Dict[str, object]] = []
    reviewed = []
    report_records = []
    for path in sorted((root / "results").glob("**/report.json")):
        report = load_json(path)
        module_id = str(report.get("module_id", path.parent.name))
        report_records.append((path, report, module_id))
    # Module-owned artifacts must exist before the integrated comparison
    # chooses whether it needs a fallback presentation view.
    report_records.sort(
        key=lambda item: (item[2] == "integrated_comparison", str(item[0]))
    )
    for path, report, module_id in report_records:
        if report.get("technical_status") != "complete":
            continue
        before = len(artifacts)
        if module_id == "pca_fes_basins":
            _pca_fes_artifacts(destination, path, report, artifacts)
        elif module_id.startswith("clustering_"):
            _clustering_artifacts(destination, path, report, module_id, artifacts)
        elif module_id == "alternative_clustering":
            _alternative_clustering_artifacts(
                destination, path, report, artifacts
            )
        elif module_id == "dccm":
            _dccm_artifacts(destination, path, report, artifacts)
        elif module_id == "replica_rmsd_rg":
            _rmsd_rg_artifacts(destination, path, report, artifacts)
        elif module_id == "pooled_rmsf":
            _rmsf_artifacts(destination, path, report, artifacts)
        elif module_id == "convergence_uncertainty":
            _convergence_artifacts(destination, path, report, artifacts)
        elif module_id == "structural_integrity_qc":
            _qc_artifacts(destination, path, report, artifacts)
        elif module_id == "radial_distribution_functions":
            _rdf_artifacts(destination, path, report, artifacts)
        elif module_id == "scalar_feature_distributions":
            _scalar_distribution_artifacts(destination, path, report, artifacts)
        elif module_id == "solvent_accessible_surface_area":
            _sasa_artifacts(destination, path, report, artifacts)
        elif module_id == "secondary_structure":
            _secondary_structure_artifacts(destination, path, report, artifacts)
        elif module_id == "dihedral_distributions":
            _dihedral_artifacts(destination, path, report, artifacts)
        elif module_id in {"ion_coordination_geometry", "nucleic_acid_geometry"}:
            _distribution_report_artifacts(
                destination, path, report, module_id, artifacts
            )
        elif module_id == "hydrogen_bonds":
            _occupancy_artifacts(
                destination, path, report, module_id, "occupancies",
                ("system_id", "replica_id", "feature_id"), artifacts,
            )
        elif module_id == "hydrogen_bond_discovery":
            _occupancy_artifacts(
                destination, path, report, module_id, "occupancies",
                ("system_id", "replica_id", "bond_id"), artifacts,
            )
        elif module_id == "water_mediated_hydrogen_bond_networks":
            _occupancy_artifacts(
                destination, path, report, module_id, "bridge_occupancies",
                ("system_id", "replica_id", "bridge_id"), artifacts,
            )
        elif module_id == "ion_atmosphere":
            _occupancy_artifacts(
                destination, path, report, module_id,
                "per_ion_inner_shell_persistence",
                ("system_id", "replica_id", "species", "ion_atom_index"), artifacts,
            )
        elif module_id == "scalar_threshold_states":
            _nested_state_population_artifacts(
                destination, path, report, module_id, artifacts
            )
        elif module_id in {
            "generalized_correlation_and_information", "information_dynamics",
            "grouped_ml", "grouped_regularized_classification",
        }:
            _matrix_report_artifacts(
                destination, path, report, module_id, artifacts
            )
        elif module_id == "correlation_networks":
            _network_artifacts(destination, path, report, artifacts)
        elif module_id == "markov_state_models":
            _msm_artifacts(destination, path, report, artifacts)
        elif module_id == "integrated_comparison":
            _integrated_comparison_artifacts(
                destination, path, report, artifacts
            )
        elif module_id in {"representative_frames", "state_coordinate_exports"}:
            _export_inventory_artifacts(
                destination, path, report, module_id, artifacts
            )
        if len(artifacts) == before:
            _generic_artifacts(destination, path, report, module_id, artifacts)
        reviewed.append({
            "module_id": module_id,
            "report_path": str(path),
            "report_sha256": sha256_file(path),
            "generated_artifact_count": len(artifacts) - before,
            "presentation_adapter": (
                "complete" if len(artifacts) > before else "failed"
            ),
        })
    manifest = write_manifest(
        destination / "presentation-manifest.json", artifacts, analysis_root=root
    )
    manifest["reviewed_reports"] = reviewed
    manifest["adapted_report_count"] = sum(
        row["presentation_adapter"] == "complete" for row in reviewed
    )
    manifest["unadapted_report_count"] = sum(
        row["presentation_adapter"] != "complete" for row in reviewed
    )
    manifest["coverage_policy"] = (
        "Every technically complete analysis report has at least one labeled figure; "
        "a table is also emitted whenever the report exposes tabular numerical values."
    )
    if manifest["unadapted_report_count"]:
        raise PresentationArtifactError(
            f"{manifest['unadapted_report_count']} complete reports lack presentation artifacts"
        )
    (destination / "presentation-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
