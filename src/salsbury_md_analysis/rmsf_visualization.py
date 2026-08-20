"""Export RMSF values as PDB B factors plus a reproducible VMD cartoon view."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .atom_mapping import AtomRecord, read_pdb_atoms
from .manifests import load_json


class RMSFVisualizationError(ValueError):
    """Raised when an RMSF report cannot be mapped onto its PDB reference."""


def _residue_key(record: Mapping[str, object]) -> Tuple[object, ...]:
    return (
        str(record.get("chain_id", "")),
        int(record["residue_number"]),
        str(record.get("insertion_code", "")),
        str(record.get("residue_name", "")),
    )


def _atom_key(atom: AtomRecord) -> Tuple[object, ...]:
    return (
        atom.chain_id, atom.residue_number, atom.insertion_code,
        atom.residue_name, atom.atom_name, atom.altloc,
    )


def _row_atom_key(record: Mapping[str, object]) -> Tuple[object, ...]:
    return (
        str(record.get("chain_id", "")), int(record["residue_number"]),
        str(record.get("insertion_code", "")), str(record.get("residue_name", "")),
        str(record["atom_name"]), str(record.get("altloc", "")),
    )


def build_rmsf_bfactor_pdb(
    reference_path: Path,
    atom_statistics: Sequence[Mapping[str, object]],
    *,
    aggregation: str = "residue_mean",
    unmapped_bfactor: float = 0.0,
) -> Dict[str, object]:
    """Return PDB text with RMSF encoded in the temperature-factor field."""

    if aggregation not in {"residue_mean", "atom"}:
        raise RMSFVisualizationError("aggregation must be residue_mean or atom")
    source = Path(reference_path).expanduser().resolve(strict=False)
    if source.suffix.lower() not in {".pdb", ".ent"}:
        raise RMSFVisualizationError("RMSF visualization currently requires a PDB reference")
    atoms = read_pdb_atoms(source)
    atom_values: Dict[Tuple[object, ...], float] = {}
    residue_values: Dict[Tuple[object, ...], list[float]] = {}
    for row in atom_statistics:
        value = row.get("frame_pooled_rmsf_angstrom", row.get("rmsf_angstrom"))
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RMSFVisualizationError("atom statistics lack a numeric RMSF value")
        numeric = float(value)
        if numeric < 0.0 or numeric > 999.99:
            raise RMSFVisualizationError("RMSF B factors must be between 0 and 999.99")
        atom_values[_row_atom_key(row)] = numeric
        residue_values.setdefault(_residue_key(row), []).append(numeric)
    if not atom_values:
        raise RMSFVisualizationError("atom_statistics must be nonempty")
    residue_means = {
        key: sum(values) / len(values) for key, values in residue_values.items()
    }

    values_by_atom = []
    mapped_values = []
    mapped_count = 0
    for atom in atoms:
        if aggregation == "atom":
            value = atom_values.get(_atom_key(atom))
        else:
            value = residue_means.get((
                atom.chain_id, atom.residue_number, atom.insertion_code,
                atom.residue_name,
            ))
        if value is None:
            value = float(unmapped_bfactor)
        else:
            mapped_count += 1
            mapped_values.append(value)
        if value < 0.0 or value > 999.99:
            raise RMSFVisualizationError("unmapped_bfactor must be between 0 and 999.99")
        values_by_atom.append(value)

    lines = source.read_text(encoding="utf-8", errors="strict").splitlines(keepends=True)
    output = [
        "REMARK 950 SALSBURY_MD_ANALYSIS RMSF ANGSTROM STORED IN B FACTOR\n",
        f"REMARK 950 AGGREGATION {aggregation.upper()} UNMAPPED {unmapped_bfactor:.2f}\n",
    ]
    atom_index = 0
    saw_model = False
    in_first_model = False
    first_model_complete = False
    for line in lines:
        record = line[:6].strip().upper()
        if record == "MODEL":
            if not saw_model:
                saw_model = True
                in_first_model = True
            elif first_model_complete:
                output.append(line)
                continue
        elif record == "ENDMDL" and in_first_model:
            in_first_model = False
            first_model_complete = True
        if record in {"ATOM", "HETATM"} and (not saw_model or in_first_model):
            if atom_index >= len(values_by_atom):
                raise RMSFVisualizationError("PDB atom records changed during export")
            newline = "\n" if line.endswith("\n") else ""
            body = line[:-1] if newline else line
            body = body.ljust(66)
            line = body[:60] + f"{values_by_atom[atom_index]:6.2f}" + body[66:] + newline
            atom_index += 1
        output.append(line)
    if atom_index != len(atoms):
        raise RMSFVisualizationError("not all first-model PDB atoms were exported")
    return {
        "pdb_text": "".join(output),
        "aggregation": aggregation,
        "reference_atom_count": len(atoms),
        "mapped_output_atom_count": mapped_count,
        "unmapped_output_atom_count": len(atoms) - mapped_count,
        "minimum_mapped_rmsf_angstrom": min(mapped_values) if mapped_values else None,
        "maximum_mapped_rmsf_angstrom": max(mapped_values) if mapped_values else None,
    }


def export_rmsf_visualization(
    report_path: Path,
    system_id: str,
    output_prefix: Path,
    *,
    reference_path: Optional[Path] = None,
    aggregation: str = "residue_mean",
    overwrite: bool = False,
) -> Dict[str, object]:
    """Write a B-factor PDB and VMD NewCartoon/Beta rendering script."""

    report_source = Path(report_path).expanduser().resolve(strict=False)
    report = load_json(report_source)
    systems = report.get("systems")
    if not isinstance(systems, list):
        raise RMSFVisualizationError("RMSF report has no systems array")
    matches = [row for row in systems if isinstance(row, dict) and row.get("system_id") == system_id]
    if len(matches) != 1:
        raise RMSFVisualizationError("system_id is absent or duplicated in the RMSF report")
    system = matches[0]
    statistics = system.get("atom_statistics")
    if not isinstance(statistics, list):
        raise RMSFVisualizationError("selected RMSF system has no atom_statistics")
    if reference_path is None:
        reference = report.get("reference")
        value = reference.get("path") if isinstance(reference, dict) else None
        if not isinstance(value, str) or not value:
            raise RMSFVisualizationError("RMSF report has no reference path")
        reference_path = Path(value)
    result = build_rmsf_bfactor_pdb(
        reference_path, statistics, aggregation=aggregation
    )
    prefix = Path(output_prefix).expanduser().resolve(strict=False)
    pdb_path = prefix.with_suffix(".rmsf_bfactor.pdb")
    vmd_path = prefix.with_suffix(".rmsf_cartoon.vmd.tcl")
    for target in (pdb_path, vmd_path):
        if target.exists() and not overwrite:
            raise RMSFVisualizationError(
                f"refusing to overwrite existing output: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
    minimum = result["minimum_mapped_rmsf_angstrom"]
    maximum = result["maximum_mapped_rmsf_angstrom"]
    if minimum is None or maximum is None:
        raise RMSFVisualizationError("no RMSF values mapped to the reference")
    script = (
        f"mol new {{{pdb_path.name}}} type pdb waitfor all\n"
        "mol delrep 0 top\n"
        "mol representation NewCartoon\n"
        "mol color Beta\n"
        "mol selection {protein or nucleic}\n"
        "mol material Opaque\n"
        "mol addrep top\n"
        "color scale method BWR\n"
        f"mol scaleminmax top 0 {float(minimum):.6f} {float(maximum):.6f}\n"
    )
    pdb_path.write_text(str(result.pop("pdb_text")), encoding="utf-8")
    vmd_path.write_text(script, encoding="utf-8")
    return {
        "module_id": "rmsf_visualization_export",
        "technical_status": "complete",
        "scientific_status": "visualization export only",
        "system_id": system_id,
        "source_report_path": str(report_source),
        "reference_path": str(Path(reference_path).expanduser().resolve(strict=False)),
        "pdb_path": str(pdb_path),
        "vmd_script_path": str(vmd_path),
        **result,
        "interpretation": (
            "PDB B factors contain RMSF in angstrom; the VMD script renders a "
            "NewCartoon colored by the Beta field. RMSF is not a crystallographic B factor."
        ),
    }
