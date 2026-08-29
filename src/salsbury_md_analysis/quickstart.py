"""Safe one-command setup for routine PDB/connectivity/DCD analyses.

This layer deliberately automates file inventory, conservative generic
settings, frame budgets, and cluster submission scaffolding.  It does not
invent residue-specific hypotheses or silently guess physical time.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from .atom_mapping import AtomMappingError, read_pdb_atoms
from .analysis_config import (
    COMMAND_MODULES,
    DEFAULT_DISABLED_MODULES,
    AnalysisConfigError,
    apply_module_configuration,
    enabled_modules,
    load_analysis_config,
    make_memory_fit_config,
)
from .automatic_sampling import automatic_sampling_plan, plan_cartesian_pca_basis
from .automatic_chemistry import (
    AutomaticChemistryError, infer_standard_chemistry_definitions,
)
from .campaign_planning import (
    CampaignPlanningError,
    plan_and_apply_complete_campaign,
)
from .chemical_identity import ION_RESIDUES, WATER_RESIDUES
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .execution_adapters import (
    ExecutionAdapterError, _active_python_executable, prepare_execution_artifacts,
)
from .energetic_network_embeddings import probe_energetic_parameter_source
from .conformational_views import plan_conformational_views
from .coordinate_cache import (
    coordinate_cache_prefix,
    coordinate_cache_system_manifest_filename,
    validate_reusable_coordinate_cache,
)
from .geometry import distance3
from .hydrogen_bond_chemistry import NUCLEIC_RESIDUES, PROTEIN_RESIDUES
from .frame_sampling import (
    integer_stride_for_budget,
    integer_stride_selected_count,
)
from .manifests import load_json, validate_project, validate_system
from .openmm_connectivity import (
    OpenMMConnectivityError,
    export_pdb_connectivity,
)
from .preflight import (
    FileProbeError, probe_connectivity, probe_topology, probe_trajectory,
)
from .planning_report import PlanningReportError, write_planning_report
from .periodic import (
    PeriodicReconstructionError,
    load_connectivity,
    make_whole_coordinates,
)
from .resource_planning import plan_alternative_clustering_fit_strides
from .registry import list_modules
from .selections import select_atoms
from .nucleic_acid_structure import probe_dssr_reference_duplex


class QuickstartError(ValueError):
    """Raised when a safe runnable project cannot be prepared."""


class QuickstartPlanningError(QuickstartError):
    """Raised with a complete plan when an execution envelope is infeasible."""

    def __init__(
        self, message: str, *, plan: Mapping[str, object],
        analysis_config: Mapping[str, object], output_directory: Path,
    ) -> None:
        super().__init__(message)
        self.plan = deepcopy(dict(plan))
        self.analysis_config = deepcopy(dict(analysis_config))
        self.output_directory = output_directory


class QuickstartMemoryError(QuickstartPlanningError):
    """Raised with a complete plan when enabled minima exceed the memory cap."""


def _experimental_planner_coverage(
    analysis_config: Mapping[str, object],
    campaign_resource_plan: Mapping[str, object],
    exclusions: Mapping[str, str],
) -> Dict[str, object]:
    """Audit every default-off method against its resolved planner state."""

    modules = analysis_config.get("modules")
    tasks = campaign_resource_plan.get("tasks")
    if not isinstance(modules, Mapping) or not isinstance(tasks, list):
        raise QuickstartError("experimental planner coverage inputs are incomplete")
    task_ids_by_module: Dict[str, list[str]] = {}
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        module_id = str(task.get("module_id", ""))
        task_id = task.get("task_id")
        if module_id and isinstance(task_id, str):
            task_ids_by_module.setdefault(module_id, []).append(task_id)
    rows = []
    missing = []
    for module_id in sorted(DEFAULT_DISABLED_MODULES):
        module_config = modules.get(module_id)
        enabled = bool(
            isinstance(module_config, Mapping)
            and module_config.get("enabled") is True
        )
        task_ids = sorted(task_ids_by_module.get(module_id, []))
        if not enabled:
            status = "disabled_by_config"
            reason = "default-off experimental method was not enabled"
        elif task_ids:
            status = "planned"
            reason = "enabled and available; planner task created"
        elif module_id in exclusions:
            status = "not_available"
            reason = str(exclusions[module_id])
        else:
            status = "missing_planner_task"
            reason = "enabled method has no planner task or availability exclusion"
            missing.append(module_id)
        rows.append({
            "module_id": module_id,
            "configuration_enabled": enabled,
            "planner_status": status,
            "planner_task_ids": task_ids,
            "reason": reason,
        })
    if missing:
        raise QuickstartError(
            "enabled experimental modules lack planner accounting: "
            + ", ".join(missing)
        )
    return {
        "coverage_schema": "salsbury-experimental-planner-coverage-v1",
        "module_count": len(rows),
        "enabled_module_count": sum(
            row["configuration_enabled"] is True for row in rows
        ),
        "planned_module_count": sum(
            row["planner_status"] == "planned" for row in rows
        ),
        "not_available_module_count": sum(
            row["planner_status"] == "not_available" for row in rows
        ),
        "modules": rows,
    }


def _record_conformational_experimental_exclusions(
    root: Path,
    view_project_files: Sequence[str],
    exclusions: Dict[str, str],
    analysis_config: Optional[Mapping[str, object]] = None,
) -> None:
    """Record method contracts that make an enabled view method inapplicable."""

    requested_modules = set()
    for filename in view_project_files:
        project = load_json(root / filename)
        requested = project.get("requested_modules")
        if isinstance(requested, list):
            requested_modules.update(str(value) for value in requested)
    modules = (
        analysis_config.get("modules")
        if isinstance(analysis_config, Mapping) else None
    )
    perturbation = (
        modules.get("perturbation_response_dynamics")
        if isinstance(modules, Mapping) else None
    )
    options = (
        perturbation.get("options")
        if isinstance(perturbation, Mapping) else None
    )
    functional_sites = (
        options.get("functional_site_node_indices")
        if isinstance(options, Mapping) else None
    )
    if (
        "perturbation_response_dynamics" not in requested_modules
        and analysis_config is not None
        and (not isinstance(functional_sites, list) or not functional_sites)
    ):
        exclusions.setdefault(
            "perturbation_response_dynamics",
            (
                "not available: functional_site_node_indices were not declared; "
                "the generic workflow does not guess a biological functional site"
            ),
        )
    elif "perturbation_response_dynamics" not in requested_modules:
        exclusions.setdefault(
            "perturbation_response_dynamics",
            (
                "not applicable: the current DFI/DCI contract requires an enabled "
                "macromolecular-trace view with one node per residue; no such view "
                "was generated for this system"
            ),
        )


_EXPERIMENTAL_INPUT_COMMANDS = {
    "trajectory_reweighting": "trajectory-reweighting",
    "allosteric_pathways": "allosteric-pathways",
    "multivalent_molecular_bridges": "multivalent-bridges",
    "hydration_density_channels": "hydration-density-channels",
}


def _apply_experimental_input_gates(
    definitions: Dict[str, object],
    commands: Sequence[str],
    requested: Sequence[str],
    exclusions: Dict[str, str],
    composition: Mapping[str, object],
) -> tuple[list[str], list[str]]:
    """Remove enabled experimental methods lacking required scientific input."""

    unavailable: Dict[str, str] = {}
    reweighting = definitions.get("trajectory_reweighting")
    if (
        "trajectory_reweighting" in requested
        and isinstance(reweighting, Mapping)
        and not str(reweighting.get("weights_path", "")).strip()
    ):
        unavailable["trajectory_reweighting"] = (
            "not available: weights_path was not declared; the generic workflow "
            "does not synthesize frame weights"
        )
    pathways = definitions.get("allosteric_pathways")
    if "allosteric_pathways" in requested and isinstance(pathways, Mapping):
        sources = pathways.get("source_node_indices")
        sinks = pathways.get("sink_node_indices")
        if (
            not isinstance(sources, list)
            or not sources
            or not isinstance(sinks, list)
            or not sinks
        ):
            unavailable["allosteric_pathways"] = (
                "not available: source_node_indices and sink_node_indices were "
                "not declared; the generic workflow does not guess biological "
                "pathway endpoints"
            )
    residue_names = {
        str(value).upper() for value in composition.get("residue_names", [])
    }
    water_present = int(composition.get("water_residue_count", 0)) > 0
    ions_present = bool(composition.get("ion_atom_indices"))
    bridges = definitions.get("multivalent_molecular_bridges")
    if "multivalent_molecular_bridges" in requested and isinstance(bridges, Mapping):
        explicit = {
            str(value).upper()
            for value in bridges.get("mediator_residue_names", [])
        }
        mediator_present = (
            (bool(bridges.get("include_supported_ions")) and ions_present)
            or (bool(bridges.get("include_recognized_waters")) and water_present)
            or bool(explicit.intersection(residue_names))
        )
        if not mediator_present:
            unavailable["multivalent_molecular_bridges"] = (
                "not applicable: no enabled ion, water, or declared mediator "
                "residue is present in the reference topology"
            )
    hydration = definitions.get("hydration_density_channels")
    if "hydration_density_channels" in requested and isinstance(hydration, Mapping):
        additional = {
            str(value).upper()
            for value in hydration.get("additional_residue_names", [])
        }
        particle_present = (
            (bool(hydration.get("include_recognized_waters")) and water_present)
            or (bool(hydration.get("include_supported_ions")) and ions_present)
            or bool(additional.intersection(residue_names))
        )
        if not particle_present:
            unavailable["hydration_density_channels"] = (
                "not applicable: no enabled water, ion, or additional particle "
                "residue is present in the reference topology"
            )
    if not unavailable:
        return list(commands), list(requested)
    for module_id, reason in unavailable.items():
        definitions.pop(module_id, None)
        exclusions[module_id] = reason
    removed_commands = {
        _EXPERIMENTAL_INPUT_COMMANDS[module_id] for module_id in unavailable
    }
    return (
        [command for command in commands if command not in removed_commands],
        [module_id for module_id in requested if module_id not in unavailable],
    )


def _force_field_parameter_spec(
    *, charmm_parameter_files: Sequence[Path] = (),
    openmm_system_xml: Optional[Path] = None,
    gromacs_tpr: Optional[Path] = None,
) -> Optional[Dict[str, object]]:
    """Normalize one explicit force-field source for energetic analyses."""

    modes = sum((
        bool(charmm_parameter_files), openmm_system_xml is not None,
        gromacs_tpr is not None,
    ))
    if modes > 1:
        raise QuickstartError(
            "choose only one energetic parameter source: CHARMM parameter files, "
            "OpenMM System XML, or GROMACS TPR"
        )
    if charmm_parameter_files:
        files = [
            Path(path).expanduser().resolve(strict=True)
            for path in charmm_parameter_files
        ]
        unsupported = [
            str(path) for path in files
            if path.suffix.lower() not in {".prm", ".par", ".str", ".inp"}
        ]
        if unsupported:
            raise QuickstartError(
                "CHARMM energetic parameter files must use .prm, .par, .str, or .inp: "
                + ", ".join(unsupported)
            )
        return {
            "format": "charmm_parameter_files_v1",
            "files": [str(path) for path in files],
        }
    if openmm_system_xml is not None:
        path = Path(openmm_system_xml).expanduser().resolve(strict=True)
        if path.suffix.lower() != ".xml":
            raise QuickstartError("serialized OpenMM System input must use .xml")
        return {"format": "openmm_system_xml_v1", "files": [str(path)]}
    if gromacs_tpr is not None:
        path = Path(gromacs_tpr).expanduser().resolve(strict=True)
        if path.suffix.lower() != ".tpr":
            raise QuickstartError("GROMACS compiled parameter input must use .tpr")
        return {"format": "gromacs_tpr_v1", "files": [str(path)]}
    return None


_GENERIC_DIRECT_ESTIMATORS = (
    "structural_integrity_qc", "replica_rmsd_rg", "pooled_rmsf", "dccm",
    "individual_pca", "dihedral_distributions",
    "hydrogen_bond_discovery", "solvent_accessible_surface_area",
    "multivalent_molecular_bridges", "hydration_density_channels",
    "ensemble_pocket_dynamics",
)

_GENERIC_CHEMISTRY_COMMANDS = {
    "trajectory_features": "trajectory-features",
    "optional_observables": "observables",
    "radial_distribution_functions": "rdf",
    "scalar_feature_distributions": "scalar-distributions",
    "scalar_threshold_states": "scalar-threshold-states",
    "nucleic_acid_structure": "nucleic-acid-structure",
    "nucleic_acid_geometry": "nucleic-acid-geometry",
    "ion_coordination_geometry": "ion-geometry",
    "ion_atmosphere": "ion-atmosphere",
}


def _secondary_structure_applicable(
    composition: Mapping[str, object], dssp_executable: Optional[str]
) -> bool:
    """Return whether protein-specific DSSP analysis can be executed."""

    return bool(dssp_executable) and bool(composition.get("has_protein"))


def _applicable_sampling_modules(
    composition: Mapping[str, object],
    analysis_config: Mapping[str, object],
    *,
    dssp_executable: Optional[str],
    energetic_parameter_available: bool = False,
) -> list[str]:
    """Return applicable direct and frame-inheriting base estimators."""

    enabled = enabled_modules(analysis_config)
    conditional_direct = {
        "multivalent_molecular_bridges", "hydration_density_channels"
    }
    result = [
        module_id for module_id in _GENERIC_DIRECT_ESTIMATORS
        if module_id in enabled and module_id not in conditional_direct
    ]
    modules = analysis_config.get("modules")
    assert isinstance(modules, Mapping)
    residue_names = {
        str(value).upper() for value in composition.get("residue_names", [])
    }
    water_present = int(composition.get("water_residue_count", 0)) > 0
    ions_present = bool(composition.get("ion_atom_indices"))
    bridge_config = modules.get("multivalent_molecular_bridges")
    bridge_options = (
        bridge_config.get("options")
        if isinstance(bridge_config, Mapping) else None
    )
    if "multivalent_molecular_bridges" in enabled:
        options = bridge_options if isinstance(bridge_options, Mapping) else {}
        explicit = {
            str(value).upper()
            for value in options.get("mediator_residue_names", [])
        }
        if (
            (bool(options.get("include_supported_ions", ions_present)) and ions_present)
            or (
                bool(options.get("include_recognized_waters", water_present))
                and water_present
            )
            or bool(explicit.intersection(residue_names))
        ):
            result.append("multivalent_molecular_bridges")
    hydration_config = modules.get("hydration_density_channels")
    hydration_options = (
        hydration_config.get("options")
        if isinstance(hydration_config, Mapping) else None
    )
    if "hydration_density_channels" in enabled:
        options = hydration_options if isinstance(hydration_options, Mapping) else {}
        additional = {
            str(value).upper()
            for value in options.get("additional_residue_names", [])
        }
        if (
            (bool(options.get("include_supported_ions", ions_present)) and ions_present)
            or (
                bool(options.get("include_recognized_waters", water_present))
                and water_present
            )
            or bool(additional.intersection(residue_names))
        ):
            result.append("hydration_density_channels")
    allosteric_config = modules.get("allosteric_pathways")
    allosteric_options = (
        allosteric_config.get("options")
        if isinstance(allosteric_config, Mapping) else None
    )
    if "allosteric_pathways" in enabled:
        options = allosteric_options if isinstance(allosteric_options, Mapping) else {}
        sources = options.get("source_node_indices")
        sinks = options.get("sink_node_indices")
        if (
            isinstance(sources, list)
            and sources
            and isinstance(sinks, list)
            and sinks
        ):
            result.append("allosteric_pathways")
    if energetic_parameter_available and "energetic_network_embeddings" in enabled:
        result.append("energetic_network_embeddings")
    if (
        int(composition["water_residue_count"]) > 0
        and "water_mediated_hydrogen_bond_networks" in enabled
    ):
        result.append("water_mediated_hydrogen_bond_networks")
    if (
        _secondary_structure_applicable(composition, dssp_executable)
        and "secondary_structure" in enabled
    ):
        result.append("secondary_structure")
    if composition.get("ion_atom_indices"):
        for module_id in ("ion_coordination_geometry", "ion_atmosphere"):
            if module_id in enabled:
                result.append(module_id)
        if (
            int(composition["water_residue_count"]) > 0
            and "radial_distribution_functions" in enabled
        ):
            result.append("radial_distribution_functions")
    if bool(composition.get("has_nucleic_acid")):
        if "nucleic_acid_geometry" in enabled:
            result.append("nucleic_acid_geometry")
    return result or ["provenance_manifest"]


def _safe_id(value: str, label: str) -> str:
    text = value.strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", text):
        raise QuickstartError(
            f"{label} must start with a letter and contain only letters, numbers, '.', '_', or '-'"
        )
    return text


def _discover_dssp_executable(explicit: Optional[str]) -> Optional[str]:
    """Find DSSP on PATH or beside the active Python interpreter."""

    if explicit:
        found = shutil.which(explicit)
        candidate = Path(explicit).expanduser()
        if found:
            return str(Path(found).resolve(strict=True))
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve(strict=True))
        raise QuickstartError(
            f"declared DSSP executable is not an executable file or command: {explicit}"
        )
    for name in ("mkdssp", "dssp"):
        found = shutil.which(name)
        if found:
            return str(Path(found).resolve(strict=True))
    # Preserve a virtual environment's interpreter path. Resolving this
    # symlink can escape the environment and make generated launchers lose the
    # dependencies that were installed there.
    interpreter_directory = Path(_active_python_executable()).parent
    for name in ("mkdssp", "dssp"):
        candidate = interpreter_directory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve(strict=True))
    return None


def _discover_dssr_executable(explicit: Optional[str]) -> Optional[str]:
    """Find x3dna-dssr without treating its separate license as bundled."""

    if explicit:
        found = shutil.which(explicit)
        candidate = Path(explicit).expanduser()
        if found:
            return str(Path(found).resolve(strict=True))
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve(strict=True))
        raise QuickstartError(
            f"declared DSSR executable is not an executable file or command: {explicit}"
        )
    found = shutil.which("x3dna-dssr")
    if found:
        return str(Path(found).resolve(strict=True))
    candidate = Path(_active_python_executable()).parent / "x3dna-dssr"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate.resolve(strict=True))
    return None


def _json_write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_new_directory(path: Path) -> Path:
    target = path.expanduser().resolve(strict=False)
    if target.exists():
        if not target.is_dir():
            raise QuickstartError(f"output path exists and is not a directory: {target}")
        if any(target.iterdir()):
            raise QuickstartError(
                f"output directory is not empty: {target}; choose a new versioned directory"
            )
    else:
        target.mkdir(parents=True)
    return target


def _composition(pdb_path: Path) -> Dict[str, object]:
    try:
        atoms = read_pdb_atoms(pdb_path)
    except AtomMappingError as exc:
        raise QuickstartError(str(exc)) from exc
    residue_atoms: Dict[tuple[object, ...], list[object]] = {}
    for atom in atoms:
        key = (atom.chain_id, atom.residue_number, atom.insertion_code, atom.residue_name)
        residue_atoms.setdefault(key, []).append(atom)
    # Large legacy PDB files can reuse residue numbers after the fixed-width
    # field wraps. Count physical water oxygens instead of unique residue keys;
    # connectivity-backed discovery applies the stricter molecule check later.
    water_count = sum(
        atom.residue_name.upper() in WATER_RESIDUES
        and atom.element.upper() == "O"
        for atom in atoms
    )
    ion_indices = [
        atom.atom_index for atom in atoms
        if atom.residue_name.upper() in ION_RESIDUES
    ]
    protein = any(
        str(key[3]).upper() in PROTEIN_RESIDUES
        and {"N", "CA", "C"}.issubset({atom.atom_name.upper() for atom in members})
        for key, members in residue_atoms.items()
    )
    nucleic_acid = any(
        str(key[3]).upper() in NUCLEIC_RESIDUES
        and "C1'" in {atom.atom_name.upper().replace("*", "'") for atom in members}
        for key, members in residue_atoms.items()
    )
    selections = {
        "alignment": {"preset": "macromolecular_backbone"},
        "protein_alignment": {"preset": "backbone"},
        "analysis": {"preset": "complex_trace"},
        "macromolecular_trace": {"preset": "complex_trace"},
        "global_common_heavy": {"preset": "solute_heavy"},
        "mapping": {"preset": "solute_heavy"},
        "solute_heavy": {"preset": "solute_heavy"},
        "molecular_payload": {"preset": "molecular_payload"},
        "all_heavy": {"preset": "heavy"},
    }
    try:
        trace_count = len(select_atoms(atoms, selections["analysis"], "analysis"))
    except AtomMappingError:
        selections["alignment"] = {"preset": "solute_heavy"}
        selections["analysis"] = {"preset": "solute_heavy"}
        trace_count = len(select_atoms(atoms, selections["analysis"], "analysis"))
    solute_heavy_count = len(
        select_atoms(atoms, selections["solute_heavy"], "solute_heavy")
    )
    try:
        reference_frame = next(iter_coordinate_frames(pdb_path, "angstrom"))
        conformational_views = plan_conformational_views(
            atoms, reference_frame.coordinates_angstrom
        )
    except (CoordinateReadError, StopIteration, ValueError, AtomMappingError) as exc:
        raise QuickstartError(
            "could not construct outcome-independent conformational views: " + str(exc)
        ) from exc
    for view in conformational_views["views"]:
        assert isinstance(view, dict)
        if "selection" not in view:
            # Symmetry-expanded oligomer views use member-local canonical atom
            # maps, not one whole-topology selection.
            continue
        selection_id = str(view["selection_id"])
        selection = view["selection"]
        assert isinstance(selection, dict)
        selections[selection_id] = selection
    return {
        "atom_count": len(atoms),
        "residue_count": len(residue_atoms),
        "water_residue_count": water_count,
        "ion_atom_indices": ion_indices,
        "has_protein": protein,
        "has_nucleic_acid": nucleic_acid,
        "residue_names": sorted({str(key[3]).upper() for key in residue_atoms}),
        "trace_atom_count": trace_count,
        "solute_heavy_atom_count": solute_heavy_count,
        "selections": selections,
        "conformational_view_plan": conformational_views,
    }


def _validate_reference_connectivity(
    pdb_path: Path, connectivity_path: Path, atom_count: int,
    *, maximum_bond_length_angstrom: float = 4.0,
    cycle_closure_tolerance_angstrom: float = 0.05,
) -> Dict[str, object]:
    """Fail before setup when the PDB and explicit bonds are incompatible."""

    try:
        bonds, identity = load_connectivity(connectivity_path, atom_count)
        frame = next(iter_coordinate_frames(pdb_path, "angstrom"))
        if frame.cell_vectors_angstrom is not None:
            make_whole_coordinates(
                frame.coordinates_angstrom,
                frame.cell_vectors_angstrom,
                bonds,
                maximum_bond_length_angstrom,
                cycle_closure_tolerance_angstrom,
            )
        else:
            for first, second in bonds:
                length = distance3(
                    frame.coordinates_angstrom[first],
                    frame.coordinates_angstrom[second],
                )
                if length > maximum_bond_length_angstrom:
                    raise PeriodicReconstructionError(
                        f"bond {first}-{second} length {length:.6g} angstrom exceeds "
                        f"gate {maximum_bond_length_angstrom:.6g}"
                    )
    except (
        CoordinateReadError, PeriodicReconstructionError, OSError, StopIteration,
    ) as exc:
        raise QuickstartError(
            "PDB/connectivity reference geometry is incompatible: " + str(exc)
            + "; supply a PDB in the connectivity atom order whose bonded geometry is valid "
              "(for example, an accepted trajectory frame)"
        ) from exc
    return {
        "connectivity_format": identity["format"],
        "bond_count": identity["bond_count"],
        "periodic_cell_present": frame.cell_vectors_angstrom is not None,
        "maximum_bond_length_angstrom": maximum_bond_length_angstrom,
        "cycle_closure_tolerance_angstrom": cycle_closure_tolerance_angstrom,
    }


def _sampling_rows(plan: Mapping[str, object]) -> Dict[str, Mapping[str, object]]:
    rows = plan.get("method_plans")
    assert isinstance(rows, list)
    return {
        str(row["module_id"]): row for row in rows if isinstance(row, dict)
    }


def _frame_selection(rows: Mapping[str, Mapping[str, object]], module_id: str) -> Dict[str, object]:
    row = rows.get(module_id)
    if row is None:
        return {"mode": "fixed_stride_v1"}
    selection = row.get("frame_selection")
    if not isinstance(selection, dict):
        return {"mode": "fixed_stride_v1"}
    return dict(selection)


def _frame_stride(rows: Mapping[str, Mapping[str, object]], module_id: str) -> int:
    row = rows.get(module_id)
    value = row.get("frame_stride", 1) if row is not None else 1
    selection = row.get("frame_selection") if row is not None else None
    if (
        isinstance(selection, dict)
        and selection.get("mode") == "integer_stride_per_replica_v1"
    ):
        value = selection.get("stride")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QuickstartError(f"sampling plan returned an invalid frame stride for {module_id}")
    return value


def _hydrogen_bond_feature_observation_gate(
    sampling_plan: Mapping[str, object],
) -> int:
    """Return the exact default candidate-by-selected-frame execution gate."""

    rows = _sampling_rows(sampling_plan)
    row = rows.get("hydrogen_bond_discovery")
    dimensions = sampling_plan.get("dimensions")
    candidate_plan = (
        dimensions.get("hydrogen_bond_candidate_planning")
        if isinstance(dimensions, dict) else None
    )
    if (
        row is not None
        and isinstance(candidate_plan, dict)
        and candidate_plan.get("status") == "complete"
    ):
        selected_frames = row.get("selected_frame_count")
        candidate_count = candidate_plan.get("common_candidate_count")
        if (
            isinstance(selected_frames, int)
            and not isinstance(selected_frames, bool)
            and selected_frames > 0
            and isinstance(candidate_count, int)
            and not isinstance(candidate_count, bool)
            and candidate_count > 0
        ):
            return selected_frames * candidate_count
    # Backward-compatible fail-safe for explicit manifests or older planning
    # records without a topology-derived candidate universe.
    return 2_000_000_000


def _generic_definitions(
    composition: Mapping[str, object],
    sampling_plan: Mapping[str, object],
    *,
    frame_counts_per_replica: Sequence[int],
    dssp_executable: Optional[str],
    dssr_probe: Mapping[str, object],
) -> tuple[Dict[str, object], list[str], Dict[str, str]]:
    if not frame_counts_per_replica or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in frame_counts_per_replica
    ):
        raise QuickstartError(
            "frame_counts_per_replica must contain positive integers"
        )
    if sum(frame_counts_per_replica) < 2:
        raise QuickstartError(
            "the pooled workflow requires at least two physical frames"
        )
    rows = _sampling_rows(sampling_plan)
    total_frames = sum(frame_counts_per_replica)
    minimum_replica_frames = min(frame_counts_per_replica)
    trace_atoms = int(composition["trace_atom_count"])
    solute_heavy = int(composition["solute_heavy_atom_count"])
    component_count = max(2, min(10, 3 * trace_atoms, minimum_replica_frames - 1))
    k_max = max(2, min(12, total_frames // 20))
    k_values = list(range(2, k_max + 1))
    # Retain the paper's 10% default while guaranteeing that even a very small
    # prepared project supplies at least k_max deterministic NANI candidates.
    nani_percentage = max(10, math.ceil(100 * k_max / total_frames))
    definitions: Dict[str, object] = {
        "structural_qc": {
            "near_coincident_distance_angstrom": 0.5,
            "maximum_near_coincident_pairs_per_frame": 100,
            "maximum_absolute_coordinate_angstrom": 1_000_000.0,
            "maximum_frame_atom_displacement_angstrom": 100.0,
            "frame_displacement_selection": "solute_heavy",
            "chemical_integrity": {
                "maximum_peptide_bond_angstrom": 1.8,
                "maximum_trans_omega_deviation_degrees": 45.0,
                "minimum_ca_chirality_volume_angstrom3": 0.1,
                "steric_clash_scale": 0.55,
                "maximum_steric_clashes_per_frame": 100_000,
                "allow_cis_proline": True,
                "declared_covalent_links": [],
            },
            "frame_stride": 1,
            "frame_selection": _frame_selection(
                rows, "structural_integrity_qc"
            ),
            "checkpointing": {
                "enabled": True,
                "within_segment_interval_seconds": 7200.0,
            },
        },
        "replica_rmsd_rg": {
            "alignment_selection": "alignment",
            "rmsd_selection": "analysis",
            "rg_selection": "solute_heavy",
            "minimum_reference_coverage": 0.95,
            "frame_stride": _frame_stride(rows, "replica_rmsd_rg"),
        },
        "pooled_rmsf": {
            "alignment_selection": "alignment",
            "analysis_selection": "analysis",
            "minimum_reference_coverage": 0.95,
            "frame_stride": _frame_stride(rows, "pooled_rmsf"),
            "time_block_size_frames": max(10, minimum_replica_frames // 10),
            "include_partial_final_block": True,
            "minimum_replicas_for_uncertainty": 2,
        },
        "dccm": {
            "alignment_selection": "alignment",
            "analysis_selection": "analysis",
            "minimum_reference_coverage": 0.95,
            "frame_stride": 1,
            "frame_selection": _frame_selection(rows, "dccm"),
            "maximum_atoms": max(1, trace_atoms),
            "minimum_evaluated_frames_per_replica": 2,
            "minimum_variance_angstrom2": 1.0e-12,
        },
        "individual_pca": {
            "alignment_selection": "alignment",
            "analysis_selection": "analysis",
            "minimum_reference_coverage": 0.95,
            "frame_stride": 1,
            "frame_selection": _frame_selection(rows, "individual_pca"),
            "projection_frame_stride": 1,
            "projection_frame_selection": _frame_selection(
                rows, "individual_pca"
            ),
            "maximum_features": max(3, 3 * trace_atoms),
            "component_count": component_count,
            "minimum_evaluated_frames_per_replica": 2,
        },
        "common_pca": {
            "alignment_selection": "alignment",
            "analysis_selection": "analysis",
            "minimum_reference_coverage": 0.95,
            "frame_stride": 1,
            "frame_selection": _frame_selection(rows, "common_pca"),
            "projection_frame_stride": 1,
            "projection_frame_selection": {"mode": "fixed_stride_v1"},
            "maximum_features": max(3, 3 * trace_atoms),
            "component_count": component_count,
            "minimum_evaluated_frames_per_replica": 2,
            "basis_weighting": "frame",
        },
        "generalized_correlation_and_information": {
            "feature_source": "common_pca",
            "component_indices": list(range(1, min(5, component_count) + 1)),
            "bin_count": 20,
            "minimum_observations_per_replica": 50,
            "maximum_features": min(5, component_count),
        },
        "information_dynamics": {
            "feature_source": "common_pca",
            "component_indices": list(range(1, min(5, component_count) + 1)),
            "analyses": ["transfer_entropy", "lagged_cross_correlation", "coskewness"],
            "lag_frames": max(1, min(10, minimum_replica_frames // 20)),
            "bin_count": 10,
            "minimum_pairs": 20,
            "maximum_features": min(5, component_count),
            "maximum_tensor_elements": 1_000,
        },
        "perturbation_response_dynamics": {
            "feature_source": "common_pca",
            "functional_site_node_indices": [],
            "random_force_directions": 250,
            "random_seed": 20260824,
            "maximum_nodes": max(1, trace_atoms),
            "minimum_observations_per_system": 2,
            "minimum_cumulative_explained_variance": 0.0,
            "include_self_perturbations": True,
        },
        "trajectory_reweighting": {
            "observable_source": "common_pca",
            "weights_path": "",
            "normalization_scope": "per_system",
            "minimum_kish_effective_sample_size": 20.0,
            "minimum_kish_ratio": 0.05,
            "maximum_single_frame_weight": 0.25,
        },
        "correlation_networks": {
            "matrix_kinds": ["frame_pooled_dccm", "difference_from_reference_dccm"],
            "absolute_threshold": 0.3,
            "include_negative": True,
            "maximum_nodes": max(1, trace_atoms),
            "profile_clustering": {
                "input_mode": "profiles",
                "minimum_cluster_size": 5,
                "minimum_samples": 3,
                "cluster_selection_method": "eom",
                "allow_single_cluster": False,
            },
        },
        "allosteric_pathways": {
            "network_source": "trajectory",
            "network_path": "",
            "node_selection": "analysis",
            "alignment_selection": "alignment",
            "minimum_reference_coverage": 0.95,
            "frame_stride": 1,
            "frame_selection": _frame_selection(rows, "dccm"),
            "contact_cutoff_angstrom": 8.0,
            "minimum_sequence_separation": 2,
            "minimum_evaluated_frames_per_system": 20,
            "minimum_variance_angstrom2": 1.0e-12,
            "source_node_indices": [],
            "sink_node_indices": [],
            "minimum_contact_occupancy": 0.5,
            "distance_epsilon": 1.0e-12,
            "shortest_path_equality_tolerance": 1.0e-12,
            "maximum_nodes": max(2, trace_atoms),
            "neighbor_correlation_factor_enabled": True,
        },
        "energetic_network_embeddings": {
            "parameter_source": "force_field_parameter_source_auto_v1",
            "atom_scope": "strict_common_complete_protein_residues_v1",
            "periodic_pair_treatment": "nonperiodic_made_whole_cpptraj_pairwise_v1",
            "electrostatic_reporting_threshold_kcal_per_mol": 0.0001,
            "vdw_reporting_threshold_kcal_per_mol": 0.0001,
            "network_edge_threshold": 0.003,
            "heat_diffusion_time": 6.0,
            "embedding_component_count": 3,
            "frame_stride": 1,
            "frame_selection": _frame_selection(
                rows, "energetic_network_embeddings"
            ),
            "minimum_evaluated_frames_per_system": 20,
            "maximum_common_protein_residues": 2_000,
            "maximum_selected_atom_pairs": 10_000_000,
            "maximum_atom_pair_frame_evaluations": 500_000_000,
            "pair_chunk_size": 250_000,
            "maximum_heat_kernel_elements": 50_000_000,
            "maximum_vdw_to_electrostatic_ratio": 0.10,
        },
        "multivalent_molecular_bridges": {
            "frame_stride": 1,
            "frame_selection": _frame_selection(
                rows, "multivalent_molecular_bridges"
            ),
            "maximum_frames": int(
                rows.get("multivalent_molecular_bridges", {}).get(
                    "selected_frame_count", total_frames
                )
            ),
            "include_supported_ions": bool(composition["ion_atom_indices"]),
            "include_recognized_waters": (
                int(composition["water_residue_count"]) > 0
            ),
            "mediator_residue_names": [],
            "solute_residue_classes": ["protein", "nucleic_acid"],
            "solute_residue_names": [],
            "mediator_atom_elements": [],
            "solute_atom_elements": ["N", "O", "S", "P"],
            "contact_cutoff_angstrom": 4.0,
            "water_contact_cutoff_angstrom": 3.5,
            "minimum_distinct_residues": 2,
            "maximum_neighbor_pairs_per_frame": 2_000_000,
            # Detailed hyperedges are retained as a deterministic min-hash
            # sample; aggregate occupancies and compact frame features still
            # use every observed bridge.
            "maximum_bridge_records": 100_000,
            "minimum_evaluated_frames_per_system": 10,
        },
        "hydration_density_channels": {
            "alignment_selection": "alignment",
            "reference_extent_selection": "solute_heavy",
            "minimum_reference_coverage": 0.95,
            "frame_stride": 1,
            "frame_selection": _frame_selection(rows, "hydration_density_channels"),
            "maximum_frames": int(
                rows.get("hydration_density_channels", {}).get(
                    "selected_frame_count", total_frames
                )
            ),
            "include_recognized_waters": int(composition["water_residue_count"]) > 0,
            "include_supported_ions": bool(composition["ion_atom_indices"]),
            "additional_residue_names": [],
            "grid_spacing_angstrom": 1.5,
            "grid_padding_angstrom": 8.0,
            "minimum_voxel_frame_occupancy": 0.10,
            "minimum_component_voxels": 1,
            "minimum_channel_depth_angstrom": 4.5,
            "maximum_grid_voxels": 500_000,
            "maximum_particle_observations": 500_000_000,
            "maximum_sparse_frame_voxels": 250_000_000,
            "minimum_evaluated_frames_per_system": 10,
        },
        "ensemble_pocket_dynamics": {
            "backend": "native_frequency_grid_v2",
            "alignment_selection": "alignment",
            "solute_selection": "solute_heavy",
            "minimum_reference_coverage": 0.95,
            "frame_stride": 1,
            "frame_selection": _frame_selection(rows, "ensemble_pocket_dynamics"),
            "maximum_frames": int(
                rows.get("ensemble_pocket_dynamics", {}).get(
                    "selected_frame_count", total_frames
                )
            ),
            "grid_spacing_angstrom": 2.0,
            "grid_padding_angstrom": 4.0,
            "minimum_clearance_angstrom": 1.4,
            "maximum_surface_distance_angstrom": 4.5,
            "minimum_seed_clearance_angstrom": 2.5,
            "minimum_seed_separation_angstrom": 5.5,
            "pocket_growth_radius_angstrom": 4.0,
            "neighborhood_radius_angstrom": 6.0,
            "minimum_nearby_atoms": 4,
            "minimum_nearby_residues": 3,
            "minimum_occupied_directions": 5,
            "maximum_directional_imbalance": 0.55,
            "minimum_pocket_voxels": 2,
            "maximum_pockets_per_frame": 24,
            "residue_jaccard_threshold": 0.50,
            "maximum_centroid_distance_angstrom": 6.0,
            "maximum_grid_voxels": 250_000,
            "maximum_pocket_instances": 2_000_000,
            "maximum_tracking_comparisons": 100_000_000,
            "minimum_evaluated_frames_per_system": 10,
            "minimum_region_frequency_fraction": 0.05,
            "minimum_region_voxels": 4,
            "maximum_frequency_regions": 128,
            "representative_frames_per_region": 2,
            "maximum_sparse_frame_voxels": 250_000_000,
        },
        "interaction_fingerprints": {
            "source_modules": [
                "hydrogen_bond_discovery",
                "water_mediated_hydrogen_bond_networks",
                "ion_coordination_geometry", "ion_atmosphere",
                "multivalent_molecular_bridges",
                "hydration_density_channels",
            ],
            "frame_join_policy": "pairwise_complete_observations_v1",
            "minimum_feature_occupancy": 0.0,
            "maximum_features": 10_000,
            "maximum_pair_comparisons": 2_000_000,
            "minimum_pair_observations": 10,
            "minimum_cooccurrence_count": 2,
        },
        "spatial_interaction_ensembles": {
            "source_module": "interaction_fingerprints",
            "alignment_selection": "alignment",
            "minimum_reference_coverage": 0.95,
            "point_construction_policy": "endpoint_partner_coordinates_v1",
            "minimum_point_observations": 20,
            "minimum_distinct_frames": 20,
            "time_block_count": 4,
            "mode_k_values": [2, 3],
            "minimum_mode_observations": 10,
            "minimum_mode_fraction": 0.10,
            "minimum_mode_silhouette": 0.35,
            "minimum_mode_centroid_separation_angstrom": 1.0,
            "minimum_mode_time_blocks": 2,
            "minimum_mode_replicas": 1,
            "maximum_superfeatures": 10_000,
            "maximum_point_observations": 5_000_000,
            "maximum_exact_mode_points": 1_000,
            "maximum_mode_iterations": 100,
            "mode_center_tolerance_angstrom": 1.0e-6,
        },
        "interaction_persistence": {
            "source_module": "interaction_fingerprints",
            "gap_tolerance_observations": [0, 1],
            "minimum_observations_per_series": 10,
            "minimum_complete_events": 2,
            "maximum_features": 10_000,
            "maximum_event_records": 5_000_000,
            "maximum_interval_relative_deviation": 0.01,
        },
        "helical_mechanics": {
            "source_module": "nucleic_acid_structure",
            "duplex_collection_field": "stems",
            "descriptor_query_ids": {
                name: f"helical-step-{name}"
                for name in ("shift", "slide", "rise", "tilt", "roll", "twist")
            },
            "angular_input_unit": "degrees",
            "minimum_frames_per_step": 20,
            "minimum_frames_per_state": 12,
            "maximum_states": 3,
            "minimum_silhouette_for_state_split": 0.25,
            "covariance_eigenvalue_floor_fraction": 1.0e-6,
            "maximum_steps": 1_000,
            "preparation_availability": deepcopy(dict(dssr_probe)),
        },
        "time_lagged_independent_component_analysis": {
            "feature_source": "common_pca",
            "component_indices": list(range(1, min(5, component_count) + 1)),
            "lag_frames": max(1, min(10, minimum_replica_frames // 20)),
            "component_count": min(3, component_count),
            "covariance_regularization": 1.0e-8,
            "covariance_eigenvalue_cutoff": 1.0e-10,
            "minimum_pairs_per_segment": 10,
            "maximum_features": min(5, component_count),
        },
        "random_feature_koopman": {
            "source_module": "time_lagged_independent_component_analysis",
            "component_indices": list(range(1, min(3, component_count) + 1)),
            "lag_frames": max(1, min(10, minimum_replica_frames // 20)),
            "component_count": min(3, component_count),
            "random_feature_counts": [32, 64],
            "bandwidth_scales": [0.5, 1.0, 2.0],
            "random_seeds": [0, 7, 19, 41],
            "cross_validation_folds": 5,
            "covariance_regularization": 1.0e-8,
            "covariance_eigenvalue_cutoff": 1.0e-10,
            "minimum_pairs_per_segment": 10,
            "maximum_bandwidth_observations": 500,
            "maximum_feature_matrix_elements": 50_000_000,
            "maximum_seed_vamp_e_relative_range": 0.25,
            "minimum_seed_subspace_similarity": 0.70,
        },
        "pca_fes_basins": {
            "x_component": 1,
            "y_component": 2,
            "binning_rule": "scott",
            "minimum_bins_per_axis": 10,
            "maximum_bins_per_axis": 100,
            "smoothing_sigmas_bins": [0.0, 0.5, 1.0, 2.0],
            "primary_smoothing_sigma_bins": 1.0,
            "maximum_silhouette_observations": 1_000,
            "silhouette_random_seed": 0,
            "padding_fraction": 0.05,
            "minimum_bin_count": 1,
            "population_block_size_frames": max(10, minimum_replica_frames // 10),
            "include_partial_final_block": True,
            "maximum_grid_cells": 10_000,
            "density_estimator": "histogram",
        },
        "clustering_kmeans": {
            "feature_source": "tica",
            "component_indices": [1, 2, 3] if component_count >= 3 else [1, 2],
            "standardize_features": True,
            "k_values": k_values,
            "initialization_methods": ["nani_strat_all", "nani_strat_reduced"],
            "nani_percentage": nani_percentage,
            "silhouette_random_seeds": [0, 7, 19, 41],
            "maximum_iterations": 500,
            "center_tolerance": 1.0e-6,
            "minimum_cluster_size": 5,
            "maximum_silhouette_observations": 1_000,
        },
        "clustering_hdbscan": {
            "feature_source": "tica",
            "component_indices": [1, 2, 3] if component_count >= 3 else [1, 2],
            "standardize_features": True,
            "minimum_cluster_sizes": [10, 25, 50],
            "minimum_samples_values": [5, 10],
            "cluster_selection_method": "eom",
            "allow_single_cluster": False,
            "minimum_retained_fraction": 0.5,
            "maximum_silhouette_observations": 1_000,
        },
        "clustering_imwkmeans": {
            "feature_source": "tica",
            "component_indices": [1, 2, 3] if component_count >= 3 else [1, 2],
            "standardize_features": True,
            "k_values": k_values,
            "minkowski_p_values": [1.5, 2.0, 3.0],
            "initialization_ranks": [0, 1, 2, 3],
            "maximum_iterations": 500,
            "objective_tolerance": 1.0e-6,
            "minimum_cluster_size": 5,
            "weight_dispersion_floor": 1.0e-8,
            "maximum_silhouette_observations": 1_000,
        },
        "alternative_clustering": {
            "feature_source": "tica",
            "component_indices": [1, 2, 3] if component_count >= 3 else [1, 2],
            "standardize_features": True,
            "algorithms": [
                "pam", "mwpam", "ward", "gaussian_mixture",
                "variational_gaussian_mixture", "affinity_propagation",
                "mean_shift", "quality_threshold",
            ],
            "k": min(6, k_max),
            "k_values": k_values,
            "random_seed": 0,
            "random_seeds": [0, 7],
            "maximum_iterations": 500,
            "minkowski_p": 2.0,
            "minkowski_p_values": [1.5, 2.0, 3.0],
            "quality_threshold_cutoff": 1.0,
            "quality_threshold_cutoffs": [0.5, 1.0, 2.0],
            "affinity_damping": 0.75,
            "affinity_damping_values": [0.65, 0.75, 0.85],
            "mean_shift_bandwidth": None,
            "mean_shift_bandwidth_values": [None],
            "maximum_observations": min(total_frames, 10_000),
            "maximum_exact_silhouette_observations": 1_000,
        },
        "pald_community_analysis": {
            "feature_source": "tica",
            "component_indices": [1, 2, 3] if component_count >= 3 else [1, 2],
            "standardize_features": True,
            "maximum_observations": min(total_frames, 500),
            "community_msm_enabled": False,
            "maximum_reported_intercommunity_ties": 100,
        },
        "representative_frames": {
            "source": "clustering_kmeans",
            "representatives_per_state": 1,
            "maximum_states": 50,
            "maximum_candidates": max(1, total_frames),
        },
        "markov_state_models": {
            "assignment_sources": ["best_clustering", "pca_fes_basins"],
            "lag_frames": [1, 2, 5, 10],
            "estimators": ["reversible_symmetrized", "nonreversible_mle"],
            "minimum_transition_count": 1,
            "maximum_states": 250,
            "ck_multiples": [2, 3],
            "maximum_ck_rmse": 0.25,
            "vamp_cross_validation_folds": 5,
            "vamp_regularization": 1.0e-8,
            "maximum_implied_timescale_relative_range": 0.5,
            "bootstrap_repeats": 100,
            "bootstrap_block_length_frames": max(
                20, min(500, minimum_replica_frames // 20)
            ),
            "bootstrap_confidence_level": 0.95,
            "random_seed": 0,
        },
        "reactive_path_ensembles": {
            "assignment_source": "clustering_kmeans",
            "endpoint_mode": "automatic_recurrent_pair",
            "source_state_ids": [],
            "sink_state_ids": [],
            "feature_indices": [1, 2, 3] if component_count >= 3 else [1, 2],
            "feature_scaling": "zscore",
            "minimum_pair_events_for_automatic_selection": 2,
            "sakoe_chiba_fraction": 0.2,
            "maximum_paths_per_direction": 100,
            "maximum_path_frames": 500,
            "maximum_pairwise_dtw_cells": 20_000_000,
            "maximum_path_clusters": 4,
            "minimum_path_cluster_size": 2,
            "minimum_complete_paths_for_comparison": 10,
            "minimum_complete_paths_per_direction": 3,
            "minimum_replicas_with_complete_paths": 2,
            "minimum_complete_paths_for_kinetics": 50,
            "minimum_complete_paths_per_direction_for_kinetics": 10,
            "minimum_replicas_with_complete_paths_for_kinetics": 3,
            "require_validated_msm_for_kinetics": True,
        },
        "grouped_ml": {
            "feature_source": "clustering_kmeans_features",
            "target_source": "clustering_kmeans_assignments",
            "group_strategy": "segment_time_blocks",
            "group_block_size_frames": max(10, minimum_replica_frames // 10),
            "estimator": "decision_tree",
            "maximum_depth": 4,
            "minimum_leaf_size": 10,
            "maximum_thresholds_per_feature": 64,
            "permutation_repeats": 20,
            "random_seed": 0,
            "minimum_groups": 4,
            "maximum_observations": max(1, total_frames),
        },
        "dihedral_distributions": {
            "angle_types": (
                (["phi", "psi", "omega", "chi1", "chi2", "chi3", "chi4", "chi5"]
                 if bool(composition.get("has_protein")) else [])
                + ([
                    "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
                    "chi", "nu0", "nu1", "nu2", "nu3", "nu4",
                ] if bool(composition.get("has_nucleic_acid")) else [])
            ),
            "frame_stride": _frame_stride(rows, "dihedral_distributions"),
            "histogram_bins": 72,
            "maximum_reference_peptide_bond_angstrom": 1.8,
            "maximum_reference_phosphodiester_bond_angstrom": 2.2,
            "maximum_observations": max(
                1_000_000,
                total_frames * int(composition["residue_count"]) * 16,
            ),
        },
        "hydrogen_bond_discovery": {
            "chemistry_policy": "automatic_topology_templates_v1",
            "interaction_scope": "all_solute",
            "cutoff_policy": {"preset": "mdanalysis_compatible_v1"},
            "exclude_same_residue": True,
            "water_policy": "exclude",
            "frame_stride": 1,
            "frame_selection": _frame_selection(rows, "hydrogen_bond_discovery"),
            "maximum_reference_donor_hydrogen_bond_angstrom": 1.35,
            "maximum_candidate_bonds": 2_000_000,
            "maximum_feature_observations": (
                _hydrogen_bond_feature_observation_gate(sampling_plan)
            ),
            "output_mode": "sparse_packed_v2",
            "candidate_chunk_size": 4096,
            "candidate_harmonization": "intersection_by_atom_index_v1",
        },
        "solvent_accessible_surface_area": {
            "surface_selection": "solute_heavy",
            "occluder_selection": "solute_heavy",
            "probe_radius_angstrom": 1.4,
            "sphere_point_count": 960,
            "frame_stride": 1,
            "frame_selection": _frame_selection(rows, "solvent_accessible_surface_area"),
            "maximum_surface_atoms": max(1, solute_heavy),
            "maximum_observations": max(1, total_frames * solute_heavy),
            "output_detail": "bounded_summary_v1",
        },
        "convergence_uncertainty": {
            "source_module": "replica_rmsd_rg",
            "metrics": ["rmsd_angstrom", "radius_of_gyration_angstrom"],
            "block_size_frames": max(10, minimum_replica_frames // 10),
            "include_partial_final_block": True,
            "minimum_blocks": 4,
            "minimum_effective_sample_size": 20.0,
            "maximum_split_mean_difference_in_sd": 1.0,
            "replica_diagnostics": False,
        },
    }
    commands = [
        "structural-qc", "rmsd-rg", "rmsf", "dccm", "individual-pca",
        "common-pca", "information-correlation", "information-dynamics",
        "perturbation-response",
        "trajectory-reweighting",
        "correlation-networks", "allosteric-pathways",
        "energetic-network-embeddings",
        "multivalent-bridges", "hydration-density-channels",
        "ensemble-pocket-dynamics", "interaction-fingerprints",
        "spatial-interaction-ensembles", "interaction-persistence",
        "helical-mechanics", "tica",
        "random-feature-koopman",
        "pca-fes-basins", "cluster-kmeans",
        "cluster-imwkmeans", "alternative-clustering", "pald-community",
        "representative-frames",
        "markov-models", "grouped-ml", "dihedrals", "hydrogen-bond-discovery",
        "sasa", "convergence",
    ]
    exclusions: Dict[str, str] = {}
    if int(composition["water_residue_count"]) > 0:
        water_selection = _frame_selection(rows, "water_mediated_hydrogen_bond_networks")
        water_row = rows.get("water_mediated_hydrogen_bond_networks")
        water_selected = (
            int(water_row["selected_frame_count"])
            if water_row is not None else total_frames
        )
        definitions["water_mediated_hydrogen_bond_networks"] = {
            "chemistry_policy": "automatic_topology_templates_v1",
            "interaction_scope": "all_solute",
            "water_identity_policy": "standard_residue_names_connectivity_v1",
            "maximum_bridge_length": 1,
            "exclude_same_residue_endpoints": True,
            "frame_stride": 1,
            "frame_selection": water_selection,
            "cutoff_policy": {"preset": "mdanalysis_compatible_v1"},
            "maximum_reference_donor_hydrogen_bond_angstrom": 1.35,
            "neighbor_search": "cell_list_v1",
            "maximum_solute_endpoints": max(2, solute_heavy),
            "maximum_waters": max(1, int(composition["water_residue_count"])),
            "maximum_evaluated_frames": max(1, water_selected),
            "maximum_neighbor_pairs_per_frame": 20_000_000,
            "maximum_bridge_paths_per_frame": 5_000_000,
            "maximum_sparse_records": 50_000_000,
        }
        commands.append("water-mediated-hydrogen-bonds")
    else:
        exclusions["water_mediated_hydrogen_bond_networks"] = "no standard water residues detected"
    if _secondary_structure_applicable(composition, dssp_executable):
        definitions["secondary_structure"] = {
            "method": "mkdssp",
            "executable": dssp_executable,
            "frame_stride": 1,
            "frame_selection": _frame_selection(rows, "secondary_structure"),
            "maximum_frames": int(
                rows.get("secondary_structure", {}).get(
                    "selected_frame_count", total_frames
                )
            ),
        }
        commands.append("secondary-structure")
    elif not bool(composition.get("has_protein")):
        exclusions["secondary_structure"] = "no protein residues detected"
    else:
        exclusions["secondary_structure"] = "mkdssp was not found; install it or pass --dssp-executable"
    commands.append("cluster-hdbscan")
    return definitions, commands, exclusions


_VIEW_STAGE_COMMANDS = {
    0: ("common-pca",),
    1: (
        "information-correlation", "information-dynamics",
        "perturbation-response", "trajectory-reweighting", "tica",
        "pca-fes-basins", "cluster-kmeans", "cluster-hdbscan",
        "cluster-imwkmeans", "alternative-clustering", "pald-community",
    ),
    2: (
        "random-feature-koopman",
        "representative-frames", "state-coordinate-exports",
        "markov-models", "grouped-ml",
    ),
    3: ("reactive-path-ensembles",),
}

_VIEW_COMMANDS = frozenset(
    command
    for stage_commands in _VIEW_STAGE_COMMANDS.values()
    for command in stage_commands
)
_VIEW_MODULE_IDS = frozenset(COMMAND_MODULES[command] for command in _VIEW_COMMANDS)

_GENERIC_STAGE_COMMANDS = {
    0: {
        "structural-qc", "rmsd-rg", "rmsf", "dccm", "individual-pca",
        "common-pca", "dihedrals", "hydrogen-bond-discovery", "sasa",
        "water-mediated-hydrogen-bonds", "secondary-structure",
        "trajectory-features", "observables", "rdf", "ion-atmosphere",
        "ion-geometry", "nucleic-acid-structure", "nucleic-acid-geometry",
        "multivalent-bridges", "hydration-density-channels",
        "ensemble-pocket-dynamics",
        "energetic-network-embeddings",
    },
    1: {
        "information-correlation", "information-dynamics",
        "perturbation-response",
        "trajectory-reweighting",
        "correlation-networks", "allosteric-pathways", "tica",
        "pca-fes-basins", "cluster-kmeans",
        "cluster-imwkmeans", "alternative-clustering", "convergence",
        "cluster-hdbscan", "pald-community", "scalar-distributions",
        "scalar-threshold-states", "interaction-fingerprints",
        "helical-mechanics",
    },
    2: {
        "spatial-interaction-ensembles", "interaction-persistence",
        "representative-frames",
        "markov-models", "grouped-ml",
    },
}


def _exclude_conformational_views_from_base_workflow(
    commands: Sequence[str], requested: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Keep conformational analyses in explicit view projects, never the base project."""

    return (
        [command for command in commands if command not in _VIEW_COMMANDS],
        [module_id for module_id in requested if module_id not in _VIEW_MODULE_IDS],
    )


def _view_requested_modules(root: Path, generated: Sequence[str]) -> set[str]:
    requested: set[str] = set()
    for filename in generated:
        if not filename.startswith("project-") or not filename.endswith(".json"):
            continue
        project = load_json(root / filename)
        raw = project.get("requested_modules", [])
        if isinstance(raw, list):
            requested.update(str(module_id) for module_id in raw)
    return requested


def _coordinate_cache_enabled(
    analysis_config: Mapping[str, object], view_ids: Sequence[str]
) -> bool:
    execution = analysis_config.get("execution")
    if not isinstance(execution, dict):
        raise QuickstartError("analysis execution configuration is unavailable")
    mode = str(execution.get("coordinate_cache", "auto"))
    if mode == "off":
        return False
    if mode not in {"auto", "required"}:
        raise QuickstartError(f"unsupported coordinate cache mode: {mode}")
    if mode == "required" and not view_ids:
        raise QuickstartError(
            "coordinate cache is required but no conformational view is executable"
        )
    return bool(view_ids)


def _configure_coordinate_cache_views(
    root: Path,
    view_ids: Sequence[str],
    *,
    cache_stride: int,
    cache_directory: Optional[Path] = None,
) -> list[str]:
    """Point conformational views at the future unwrapped working cache."""

    external = cache_directory is not None
    cache_root = (
        Path(cache_directory).expanduser().resolve(strict=True)
        if external else root / "coordinate-cache"
    )
    records = []
    for view_id in view_ids:
        project_path = root / f"project-{view_id}.json"
        project = load_json(project_path)
        if not isinstance(project, dict):
            raise QuickstartError(f"view project is not an object: {project_path}")
        original_manifest_name = str(project["system_manifest"])
        original_manifest_path = root / original_manifest_name
        original_manifest = load_json(original_manifest_path)
        systems = original_manifest.get("systems")
        if not isinstance(systems, list) or not systems:
            raise QuickstartError(
                f"view {view_id} source system manifest contains no systems"
            )
        system_by_id = {
            str(row["system_id"]): row
            for row in systems if isinstance(row, dict)
        }
        reference_system = str(project["reference_system"])
        reference_row = system_by_id.get(reference_system)
        if not isinstance(reference_row, dict):
            raise QuickstartError(
                f"view {view_id} reference system is absent from its source manifest"
            )
        replicas = reference_row.get("replicas")
        if not isinstance(replicas, list) or not replicas or not isinstance(replicas[0], dict):
            raise QuickstartError(
                f"view {view_id} reference system contains no replica"
            )
        reference_replica = str(replicas[0]["replica_id"])
        prefix = coordinate_cache_prefix(reference_system, reference_replica)
        if original_manifest_name == "system.json":
            cache_manifest_name = "system-cache.json"
        elif len(systems) == 1:
            cache_manifest_name = coordinate_cache_system_manifest_filename(
                str(systems[0]["system_id"])
            )
        else:
            raise QuickstartError(
                f"view {view_id} has an unsupported cache manifest scope"
            )
        project.update({
            "system_manifest": (
                str(cache_root / cache_manifest_name)
                if external else f"coordinate-cache/{cache_manifest_name}"
            ),
            "reference_structure": (
                str(cache_root / f"{prefix}.pdb")
                if external else f"coordinate-cache/{prefix}.pdb"
            ),
            "reference_connectivity": (
                str(cache_root / f"{prefix}.bonds.json")
                if external else f"coordinate-cache/{prefix}.bonds.json"
            ),
        })
        _json_write(project_path, project)
        validate_project(project, source_path=project_path, check_paths=False)
        records.append({
            "view_id": view_id,
            "project_manifest": project_path.name,
            "source_system_manifest": original_manifest_name,
            "cached_system_manifest": project["system_manifest"],
            "reference_system": reference_system,
            "reference_replica": reference_replica,
            "cached_reference_structure": project["reference_structure"],
            "cached_reference_connectivity": project["reference_connectivity"],
        })
    contract = {
        "cache_contract_schema": "salsbury-coordinate-cache-workflow-v1",
        "technical_status": "reused" if external else "planned",
        "cache_output_directory": str(cache_root),
        "source_system_manifest": "system.json",
        "coordinate_representation": (
            "continuous_unwrap_strided_molecular_payload_v2"
        ),
        "source_frame_scan": "all_frames_continuous_unwrap",
        "cache_stride": cache_stride,
        "external_cache_reused": external,
        "base_workflow_uses_original_solvated_trajectories": True,
        "conformational_views_use_cache": True,
        "alignment_is_performed_downstream_per_view": True,
        "bulk_water_excluded": True,
        "frame_identity_preserved": True,
        "views": records,
    }
    _json_write(root / "coordinate-cache-contract.json", contract)
    return ["coordinate-cache-contract.json"]


def _conformational_view_projects(
    root: Path,
    base_project: Mapping[str, object],
    composition: Mapping[str, object],
    *,
    frame_counts_per_replica: Sequence[int],
    analysis_config: Mapping[str, object],
    workflow_prefix: str = "",
    plan_filename: str = "conformational-views.json",
    output_root_prefix: str = "results/conformational-views",
    workflow_scope: str = "shared_or_single_system",
    workflow_system_id: Optional[str] = None,
) -> tuple[list[str], list[str]]:
    """Write independently configurable topology-derived conformational projects."""

    raw_plan = composition["conformational_view_plan"]
    assert isinstance(raw_plan, dict)
    plan = deepcopy(raw_plan)
    views = plan["views"]
    assert isinstance(views, list)
    base_definitions = base_project["definitions"]
    assert isinstance(base_definitions, dict)
    if "common_pca" not in base_definitions:
        for view in views:
            assert isinstance(view, dict)
            view["execution"] = (
                "not generated because common_pca is disabled by analysis config"
            )
        plan["workflow_scope"] = workflow_scope
        plan["workflow_prefix"] = workflow_prefix
        plan["workflow_system_id"] = workflow_system_id
        _json_write(root / plan_filename, plan)
        return [], [plan_filename]
    view_definition_ids = {
        "common_pca", "generalized_correlation_and_information",
        "information_dynamics", "perturbation_response_dynamics",
        "trajectory_reweighting", "time_lagged_independent_component_analysis",
        "random_feature_koopman",
        "pca_fes_basins", "clustering_kmeans", "clustering_hdbscan",
        "clustering_imwkmeans", "alternative_clustering",
        "pald_community_analysis", "representative_frames",
        "state_coordinate_exports", "markov_state_models", "grouped_ml",
        "reactive_path_ensembles",
    }
    requested = [
        "common_pca", "generalized_correlation_and_information",
        "information_dynamics", "perturbation_response_dynamics",
        "trajectory_reweighting",
        "time_lagged_independent_component_analysis",
        "random_feature_koopman",
        "pca_fes_basins", "clustering_kmeans", "clustering_hdbscan",
        "clustering_imwkmeans", "alternative_clustering",
        "pald_community_analysis", "representative_frames",
        "state_coordinate_exports", "markov_state_models", "grouped_ml",
        "reactive_path_ensembles",
    ]
    generated: list[str] = []
    executable_views: list[str] = []
    if not frame_counts_per_replica:
        raise QuickstartError("conformational views require at least one replica")
    replica_count = len(frame_counts_per_replica)
    minimum_replica_frames = min(frame_counts_per_replica)
    total_source_frames = sum(frame_counts_per_replica)
    exports_config = analysis_config["exports"]
    assert isinstance(exports_config, dict)
    for view in views:
        assert isinstance(view, dict)
        view_id = str(view["view_id"])
        execution_view_id = (
            f"{workflow_prefix}__{view_id}" if workflow_prefix else view_id
        )
        view_config = analysis_config["views"]
        assert isinstance(view_config, dict)
        configured_view = view_config.get(view_id, {})
        assert isinstance(configured_view, dict)
        if configured_view.get("enabled") is False:
            view["execution"] = "disabled by analysis config"
            continue
        atom_count = int(view["atom_count_in_reference"])
        feature_count = 3 * atom_count
        symmetry = view.get("symmetry_expansion")
        member_count = (
            int(symmetry["member_count"])
            if isinstance(symmetry, dict) else 1
        )
        try:
            resource_plan = plan_cartesian_pca_basis(
                feature_count,
                [value * member_count for value in frame_counts_per_replica],
                component_count=min(10, feature_count, max(2, minimum_replica_frames - 1)),
                minimum_basis_frames_per_replica=min(
                    minimum_replica_frames * member_count, 20 * member_count
                ),
            )
        except ValueError as exc:
            raise QuickstartError(f"cannot plan {view_id} PCA: {exc}") from exc
        member_basis_budget = min(
            int(value) for value in resource_plan["basis_frames_per_replica"]
        )
        basis_budget = min(
            minimum_replica_frames,
            max(1, member_basis_budget // member_count),
        )
        if member_count > 1:
            basis_stride = integer_stride_for_budget(
                list(frame_counts_per_replica),
                basis_budget,
                error_type=QuickstartError,
            )
            basis_physical_counts = [
                integer_stride_selected_count(value, basis_stride)
                for value in frame_counts_per_replica
            ]
            basis_physical_total = sum(basis_physical_counts)
            resource_plan.update({
                "source_physical_frames_per_replica": list(frame_counts_per_replica),
                "source_physical_frame_count": total_source_frames,
                "member_observation_multiplier": member_count,
                "basis_physical_frames_per_replica": basis_physical_counts,
                "basis_physical_frame_count": basis_physical_total,
                "basis_member_observation_count": basis_physical_total * member_count,
                "projection_physical_frame_count": total_source_frames,
                "projection_member_observation_count": (
                    total_source_frames * member_count
                ),
                "independent_sampling_unit": "original simulation replica and physical time block",
            })
            resource_plan["basis_frame_selection"] = (
                {"mode": "integer_stride_per_replica_v1", "stride": basis_stride}
                if basis_stride > 1
                else {"mode": "fixed_stride_v1"}
            )
        view_project = deepcopy(base_project)
        view_project["project_id"] = (
            f"{base_project['project_id']}-{execution_view_id}"
        )
        output_root = f"{output_root_prefix}/{view_id}"
        view_project["analysis_output_root"] = output_root
        view_project["definitions"] = {
            key: deepcopy(value)
            for key, value in base_definitions.items()
            if key in view_definition_ids
        }
        definitions = view_project["definitions"]
        assert isinstance(definitions, dict)
        if view_id != "macromolecular_trace":
            # The current DFI/DCI contract is residue/node based.  The trace
            # view supplies one representative macromolecular node per residue;
            # heavy-atom and interface views would change the scientific unit.
            definitions.pop("perturbation_response_dynamics", None)
        total_frames = total_source_frames
        export_stride = max(1, math.ceil(total_frames / 200))
        definitions["state_coordinate_exports"] = {
            "source": "pca_fes_basins",
            "export_id": f"{execution_view_id}-fes-sigma1",
            "trajectory_format": "pdb",
            "representatives_per_state": 1,
            "frame_stride_within_state": export_stride,
            "maximum_states": 250,
            "maximum_frames_per_state": 250,
            "maximum_total_frames": 500,
            "existing_output_policy": "fail",
            "coordinate_selection": (
                "molecular_payload"
                if exports_config.get("payload") == "complete_solute"
                else str(view.get("selection_id", "analysis"))
            ),
            "fes_smoothing_sigma_bins": 1.0,
            "write_trajectories": bool(
                configured_view.get("state_trajectory_exports_enabled", True)
            ),
        }
        pca = definitions["common_pca"]
        assert isinstance(pca, dict)
        view_selections = view_project["selections"]
        assert isinstance(view_selections, dict)
        if member_count > 1:
            assert isinstance(symmetry, dict)
            pca["symmetry_expansion"] = deepcopy(symmetry)
            if exports_config.get("payload") != "complete_solute":
                definitions["state_coordinate_exports"].pop(  # type: ignore[union-attr]
                    "coordinate_selection", None
                )
        else:
            pca["analysis_selection"] = str(view["selection_id"])
            pca["alignment_selection"] = str(view["alignment_selection_id"])
            view_selections["analysis"] = deepcopy(
                view_selections[pca["analysis_selection"]]
            )
            view_selections["alignment"] = deepcopy(
                view_selections[pca["alignment_selection"]]
            )
        pca["maximum_features"] = feature_count
        pca["component_count"] = int(resource_plan["component_count"])
        pca["frame_stride"] = 1
        pca["frame_selection"] = resource_plan["basis_frame_selection"]
        pca["projection_frame_stride"] = 1
        pca["projection_frame_selection"] = resource_plan["projection_frame_selection"]
        if resource_plan["solver_method"] == "randomized_truncated_svd_v1":
            pca["solver"] = {
                "method": "randomized_truncated_svd_v1",
                "oversampling": int(resource_plan["randomized_solver_oversampling"]),
                "power_iterations": 4,
                "power_iteration_schedule": [4, 8, 12],
                "random_seed": 20260812,
                "maximum_sample_matrix_elements": resource_plan[
                    "maximum_sample_matrix_elements"
                ],
                "maximum_relative_residual": 1.0e-3,
            }
        else:
            pca["solver"] = {"method": "dense_covariance_v1"}
        effective_observations = total_source_frames * member_count
        representative = definitions.get("representative_frames")
        if isinstance(representative, dict):
            representative["maximum_candidates"] = effective_observations
        grouped = definitions.get("grouped_ml")
        if isinstance(grouped, dict):
            grouped["maximum_observations"] = effective_observations
        alternative = definitions.get("alternative_clustering")
        if isinstance(alternative, dict):
            execution = analysis_config["execution"]
            assert isinstance(execution, dict)
            fit_plan = plan_alternative_clustering_fit_strides(
                frame_counts_per_replica,
                member_observation_multiplier=member_count,
                algorithms=[str(value) for value in alternative["algorithms"]],
                target_wall_hours=float(execution["maximum_hours_per_cpu"]),
            )
            alternative["fit_sampling"] = fit_plan
            running = [
                plan for plan in fit_plan["algorithm_plans"].values()
                if isinstance(plan, dict) and plan.get("execution") == "run"
            ]
            alternative["maximum_observations"] = max(
                (
                    int(plan["selected_fit_observation_count"])
                    for plan in running
                ),
                default=1,
            )
        module_options = configured_view.get("module_options", {})
        assert isinstance(module_options, dict)
        for module_id, options in module_options.items():
            definition_id = (
                "structural_qc" if module_id == "structural_integrity_qc" else module_id
            )
            if definition_id not in definitions:
                raise QuickstartError(
                    f"view {view_id} cannot configure unavailable module {module_id}"
                )
            definition = definitions[definition_id]
            assert isinstance(definition, dict) and isinstance(options, dict)
            definition.update(deepcopy(options))
        enabled = enabled_modules(analysis_config)
        view_requested_modules = [
            module_id for module_id in requested
            if module_id in enabled
            and (
                ("structural_qc" if module_id == "structural_integrity_qc" else module_id)
                in definitions
            )
        ]
        view_project["requested_modules"] = view_requested_modules
        if not view_project["requested_modules"]:
            view["execution"] = (
                "not generated because configuration disabled every dependent "
                "conformational-view module"
            )
            continue
        filename = f"project-{execution_view_id}.json"
        path = root / filename
        _json_write(path, view_project)
        validate_project(view_project, source_path=path, check_paths=True)
        generated.append(filename)
        executable_views.append(execution_view_id)
        view.update({
            "execution_view_id": execution_view_id,
            "workflow_scope": workflow_scope,
            "workflow_system_id": workflow_system_id,
            "project_manifest": filename,
            "analysis_output_root": output_root,
            "basis_fit_maximum_frames_per_replica": basis_budget,
            "basis_fit_maximum_member_observations_per_replica": (
                basis_budget * member_count
            ),
            "projection_policy": "all source frames",
            "physical_projection_frame_count": total_source_frames,
            "symmetry_expanded_projection_observation_count": effective_observations,
            "member_observations_are_independent_replicas": False,
            "cartesian_feature_count": feature_count,
            "resource_plan": resource_plan,
            "solver": pca["solver"],
            "representative_structure_exports_enabled": (
                "state_coordinate_exports" in view_requested_modules
            ),
            "state_trajectory_exports_enabled": bool(
                definitions["state_coordinate_exports"]["write_trajectories"]
            ),
            "execution": "additional automatic conformational-view workflow",
        })
    plan["workflow_scope"] = workflow_scope
    plan["workflow_prefix"] = workflow_prefix
    plan["workflow_system_id"] = workflow_system_id
    _json_write(root / plan_filename, plan)
    generated.append(plan_filename)
    return executable_views, generated


def _conformational_view_slurm_files(
    root: Path,
    project_id: str,
    view_ids: Sequence[str],
    *,
    target_wall_hours: float,
    python_executable: str,
    package_root: str,
    maximum_parallel_cpus: int,
) -> list[str]:
    if not view_ids:
        return []
    view_wall_minutes = int(math.ceil(target_wall_hours * 60.0))
    wall_limit = f"{view_wall_minutes // 60:02d}:{view_wall_minutes % 60:02d}:00"
    # Every current view command is one process. Reserve one CPU for it and
    # bound aggregate array width across views in the submitter below.
    view_cpus_per_task = 1
    generated: list[str] = []
    submission_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"ROOT={json.dumps(str(root))}",
        'UPSTREAM_JOB="${1:?preflight job id is required}"',
    ]
    view_projects = {
        view_id: load_json(root / f"project-{view_id}.json")
        for view_id in view_ids
    }
    manifest_preflights: Dict[str, tuple[str, str]] = {
        "system.json": ("UPSTREAM_JOB", str(root / "preflight.report.json"))
    }
    nonshared_manifests = sorted({
        str(project.get("system_manifest"))
        for project in view_projects.values()
        if str(project.get("system_manifest")) != "system.json"
    })
    previous_preflight_batch: list[str] = []
    current_preflight_batch: list[str] = []
    for manifest_index, manifest_filename in enumerate(nonshared_manifests):
        if (
            manifest_index > 0
            and manifest_index % maximum_parallel_cpus == 0
        ):
            previous_preflight_batch = current_preflight_batch
            current_preflight_batch = []
        manifest_path = root / manifest_filename
        preflight_path = root / f"preflight-{Path(manifest_filename).stem}.report.json"
        filename = f"run_view_preflight_{manifest_index}.slurm"
        worker = f"""#!/usr/bin/env bash
#SBATCH --job-name=sma-{project_id[:12]}-vp{manifest_index}
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output={root}/logs/%j-view-preflight-{manifest_index}.out
#SBATCH --error={root}/logs/%j-view-preflight-{manifest_index}.err
set -euo pipefail
ROOT={json.dumps(str(root))}
MANIFEST={json.dumps(str(manifest_path))}
FINAL={json.dumps(str(preflight_path))}
PYTHON_DEFAULT={json.dumps(python_executable)}
PYTHON="${{SALSBURY_MD_ANALYSIS_PYTHON:-$PYTHON_DEFAULT}}"
PACKAGE_ROOT_DEFAULT={json.dumps(package_root)}
PACKAGE_ROOT="${{SALSBURY_MD_ANALYSIS_PYTHONPATH:-$PACKAGE_ROOT_DEFAULT}}"
export PYTHONPATH="$PACKAGE_ROOT${{PYTHONPATH:+:$PYTHONPATH}}"
TMP="$FINAL.tmp.$SLURM_JOB_ID"
"$PYTHON" -m salsbury_md_analysis preflight-system "$MANIFEST" --hash-content > "$TMP"
"$PYTHON" - "$TMP" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding='utf-8'))
if report.get('technical_status') != 'complete':
    raise SystemExit('per-system preflight did not complete')
PY
if [[ -e "$FINAL" ]]; then
  if ! cmp -s "$TMP" "$FINAL"; then
    printf 'Refreshed per-system preflight differs from retained report.\n' >&2
    exit 1
  fi
  rm "$TMP"
else
  ln "$TMP" "$FINAL"
  rm "$TMP"
fi
"""
        (root / filename).write_text(worker, encoding="utf-8")
        generated.append(filename)
        variable = f"SYSTEM_PREFLIGHT_{manifest_index}_JOB"
        preflight_dependencies = ["UPSTREAM_JOB", *previous_preflight_batch]
        preflight_dependency_expression = ":".join(
            f"${name}" for name in preflight_dependencies
        )
        submission_lines.extend([
            f'{variable}=$(sbatch --parsable '
            f'--dependency="afterok:{preflight_dependency_expression}" '
            f'"$ROOT/{filename}")',
            f'{variable}="${{{variable}%%;*}}"',
            f"printf 'Submitted per-system preflight %s.\\n' \"${variable}\"",
        ])
        current_preflight_batch.append(variable)
        manifest_preflights[manifest_filename] = (variable, str(preflight_path))
    all_system_preflight_variables = [
        variable for manifest, (variable, _) in manifest_preflights.items()
        if manifest != "system.json"
    ]
    stage_specs: Dict[int, list[Dict[str, object]]] = {
        stage: [] for stage in _VIEW_STAGE_COMMANDS
    }
    for view_index, view_id in enumerate(view_ids):
        project_path = root / f"project-{view_id}.json"
        view_project = view_projects[view_id]
        requested_modules = set(view_project.get("requested_modules", []))
        output_root = root / str(view_project["analysis_output_root"])
        manifest_filename = str(view_project["system_manifest"])
        view_preflight_variable, preflight_report_path = manifest_preflights[
            manifest_filename
        ]
        for stage, default_commands in _VIEW_STAGE_COMMANDS.items():
            commands = tuple(
                command for command in default_commands
                if COMMAND_MODULES[command] in requested_modules
            )
            if not commands:
                continue
            filename = f"run_view_{view_id}_stage_{stage}.slurm"
            command_lines = "\n".join(f"  {json.dumps(command)}" for command in commands)
            cache_exports = ""
            if stage >= 1:
                cache_exports += (
                    f"export SALSBURY_MD_ANALYSIS_PREFLIGHT_REPORT="
                    f"{json.dumps(preflight_report_path)}\n"
                )
                cache_exports += _validated_cache_export_shell(
                    "SALSBURY_MD_ANALYSIS_COMMON_PCA_REPORT",
                    output_root / "common-pca/report.json", "common_pca",
                ) + "\n"
            if stage >= 2:
                cache_exports += "\n".join((
                    _validated_cache_export_shell(
                        "SALSBURY_MD_ANALYSIS_TICA_REPORT",
                        output_root / "tica/report.json",
                        "time_lagged_independent_component_analysis",
                    ),
                    _validated_cache_export_shell(
                        "SALSBURY_MD_ANALYSIS_KMEANS_REPORT",
                        output_root / "cluster-kmeans/report.json",
                        "clustering_kmeans",
                    ),
                    _validated_cache_export_shell(
                        "SALSBURY_MD_ANALYSIS_HDBSCAN_REPORT",
                        output_root / "cluster-hdbscan/report.json",
                        "clustering_hdbscan",
                    ),
                    _validated_cache_export_shell(
                        "SALSBURY_MD_ANALYSIS_IMWKMEANS_REPORT",
                        output_root / "cluster-imwkmeans/report.json",
                        "clustering_imwkmeans",
                    ),
                    _validated_cache_export_shell(
                        "SALSBURY_MD_ANALYSIS_ALTERNATIVE_CLUSTERING_REPORT",
                        output_root / "alternative-clustering/report.json",
                        "alternative_clustering",
                    ),
                    _validated_cache_export_shell(
                        "SALSBURY_MD_ANALYSIS_FES_REPORT",
                        output_root / "pca-fes-basins/report.json",
                        "pca_fes_basins",
                    ),
                )) + "\n"
            if stage >= 3 and "markov_state_models" in requested_modules:
                cache_exports += _validated_cache_export_shell(
                    "SALSBURY_MD_ANALYSIS_MSM_REPORT",
                    output_root / "markov-models/report.json",
                    "markov_state_models",
                ) + "\n"
            worker = f"""#!/usr/bin/env bash
#SBATCH --job-name=sma-{project_id[:12]}-{view_index}-v{stage}
#SBATCH --time={wall_limit}
#SBATCH --cpus-per-task={view_cpus_per_task}
#SBATCH --mem={32 if stage in (0, 1) else 8}G
#SBATCH --output={root}/logs/%A_%a-view-{view_id}-stage-{stage}.out
#SBATCH --error={root}/logs/%A_%a-view-{view_id}-stage-{stage}.err
set -euo pipefail
ROOT={json.dumps(str(root))}
PROJECT={json.dumps(str(project_path))}
OUTPUT_ROOT={json.dumps(str(output_root))}
COMMANDS=(
{command_lines}
)
COMMAND="${{COMMANDS[$SLURM_ARRAY_TASK_ID]}}"
PYTHON_DEFAULT={json.dumps(python_executable)}
PYTHON="${{SALSBURY_MD_ANALYSIS_PYTHON:-$PYTHON_DEFAULT}}"
PACKAGE_ROOT_DEFAULT={json.dumps(package_root)}
PACKAGE_ROOT="${{SALSBURY_MD_ANALYSIS_PYTHONPATH:-$PACKAGE_ROOT_DEFAULT}}"
export PYTHONPATH="$PACKAGE_ROOT${{PYTHONPATH:+:$PYTHONPATH}}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
{cache_exports}mkdir -p "$OUTPUT_ROOT/$COMMAND" "$ROOT/logs"
TMP="$OUTPUT_ROOT/$COMMAND/report.json.tmp.$SLURM_JOB_ID"
FINAL="$OUTPUT_ROOT/$COMMAND/report.json"
SUMMARY="$FINAL.summary.json"
if [[ -e "$FINAL" ]]; then
  "$PYTHON" - "$FINAL" "$SUMMARY" <<'PY'
import hashlib, json, sys
report_path, summary_path = sys.argv[1:]
summary = json.load(open(summary_path, encoding='utf-8'))
digest = hashlib.sha256()
with open(report_path, 'rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(chunk)
if summary.get('technical_status') != 'complete' or summary.get('report_sha256') != digest.hexdigest():
    raise SystemExit('existing view report summary is incomplete or hash-mismatched')
PY
  exit 0
fi
SUMMARY_TMP="$TMP.summary.json"
"$PYTHON" -m salsbury_md_analysis run-instrumented "$COMMAND" "$PROJECT" \
  --hash-content --summary-sidecar "$SUMMARY_TMP" --installed-report-path "$FINAL" > "$TMP"
"$PYTHON" - "$TMP" "$SUMMARY_TMP" <<'PY'
import hashlib, json, sys
report_path, summary_path = sys.argv[1:]
summary = json.load(open(summary_path, encoding='utf-8'))
digest = hashlib.sha256()
with open(report_path, 'rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(chunk)
if summary.get('technical_status') != 'complete' or summary.get('report_sha256') != digest.hexdigest():
    raise SystemExit('view report summary is incomplete or hash-mismatched')
PY
ln "$SUMMARY_TMP" "$SUMMARY"
ln "$TMP" "$FINAL"
rm "$TMP" "$SUMMARY_TMP"
"""
            (root / filename).write_text(worker, encoding="utf-8")
            generated.append(filename)
            variable = f"VIEW_{view_index}_STAGE_{stage}_JOB"
            stage_specs[stage].append({
                "variable": variable,
                "filename": filename,
                "view_id": view_id,
                "command_count": len(commands),
                "preflight_variable": view_preflight_variable,
            })

    previous_stage_variables = (
        all_system_preflight_variables or ["UPSTREAM_JOB"]
    )
    final_view_variables: list[str] = []
    for stage in sorted(stage_specs):
        specs = stage_specs[stage]
        if not specs:
            continue
        batches: list[list[Dict[str, object]]] = []
        batch: list[Dict[str, object]] = []
        batch_width = 0
        for spec in specs:
            width = int(spec["command_count"])
            if batch and batch_width + width > maximum_parallel_cpus:
                batches.append(batch)
                batch = []
                batch_width = 0
            batch.append(spec)
            batch_width += width
        if batch:
            batches.append(batch)
        previous_batch_variables: list[str] = []
        stage_variables: list[str] = []
        for stage_batch in batches:
            batch_variables: list[str] = []
            for spec in stage_batch:
                variable = str(spec["variable"])
                dependencies = list(previous_stage_variables)
                dependencies.extend(previous_batch_variables)
                if stage == min(stage_specs):
                    dependencies.append(str(spec["preflight_variable"]))
                dependency_expression = ":".join(
                    f"${name}" for name in dict.fromkeys(dependencies)
                )
                width = int(spec["command_count"])
                submission_lines.extend([
                    f'{variable}=$(sbatch --parsable '
                    f'--dependency="afterok:{dependency_expression}" '
                    f'--array=0-{width - 1}%{min(width, maximum_parallel_cpus)} '
                    f'"$ROOT/{spec["filename"]}")',
                    f'{variable}="${{{variable}%%;*}}"',
                    f"printf 'Submitted {spec['view_id']} conformational stage "
                    f"{stage} %s.\\n' \"${variable}\"",
                ])
                batch_variables.append(variable)
                stage_variables.append(variable)
            previous_batch_variables = batch_variables
        previous_stage_variables = stage_variables
        final_view_variables = stage_variables
    final_job_expression = ":".join(
        f"${{{variable}}}" for variable in final_view_variables
    )
    submission_lines.append(
        f"printf 'FINAL_JOB_IDS=%s\\n' \"{final_job_expression}\""
    )
    submit_name = "submit-conformational-views.sh"
    (root / submit_name).write_text("\n".join(submission_lines) + "\n", encoding="utf-8")
    os.chmod(root / submit_name, 0o755)
    generated.append(submit_name)
    return generated


def _effective_parallel_cpu_cap(
    campaign_resource_plan: Mapping[str, object],
) -> int:
    """Return the resolved launcher cap while retaining the user's request."""

    requested = int(campaign_resource_plan["maximum_parallel_cpus_input"])
    value = campaign_resource_plan.get("effective_parallel_cpu_cap", requested)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QuickstartError(
            "campaign resource plan has an invalid effective parallel CPU cap"
        )
    if value > requested:
        raise QuickstartError(
            "campaign resource plan cannot raise the requested parallel CPU cap"
        )
    return value


def _execution_config_for_parallel_cpu_cap(
    analysis_config: Mapping[str, object], effective_parallel_cpu_cap: int,
) -> Dict[str, object]:
    """Copy a user config and apply only its resolved execution CPU cap."""

    resolved = deepcopy(dict(analysis_config))
    execution = resolved.get("execution")
    if not isinstance(execution, dict):
        raise QuickstartError("analysis config lacks an execution object")
    execution["maximum_parallel_cpus"] = effective_parallel_cpu_cap
    return resolved


def _validated_cache_export_shell(
    variable: str, report_path: Optional[Path], module_id: str, *,
    report_shell_expression: Optional[str] = None,
    fallback_action: str = "recomputing from project inputs",
) -> str:
    """Export a matching cache or leave the consumer free to recompute it."""

    if report_shell_expression is None:
        if report_path is None:
            raise QuickstartError("cache report path is required")
        prefix = ""
        report = json.dumps(str(report_path))
        summary = json.dumps(str(report_path) + ".summary.json")
    else:
        prefix = f"CACHE_REPORT={report_shell_expression}\\n"
        report = '"$CACHE_REPORT"'
        summary = '"$CACHE_REPORT.summary.json"'
    return f"""{prefix}unset {variable}
if [[ -f {report} && -f {summary} ]]; then
  export {variable}={report}
  if ! "$PYTHON" - "$PROJECT" {report} {summary} <<'PY'
import hashlib, json, sys
from pathlib import Path
from salsbury_md_analysis.upstream_cache import load_cached_project_report
project_path, report_path, summary_path = sys.argv[1:]
with open(report_path, encoding='utf-8') as handle:
    report = json.load(handle)
with open(summary_path, encoding='utf-8') as handle:
    summary = json.load(handle)
digest = hashlib.sha256()
with open(report_path, 'rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(chunk)
if report.get('technical_status') != 'complete':
    raise SystemExit(1)
if report.get('module_id') != {json.dumps(module_id)}:
    raise SystemExit(1)
if summary.get('technical_status') != 'complete':
    raise SystemExit(1)
if summary.get('report_sha256') != digest.hexdigest():
    raise SystemExit(1)
load_cached_project_report(
    {json.dumps(module_id)}, Path(project_path), hash_content=True,
    error_type=ValueError,
)
PY
  then
    unset {variable}
  fi
fi
if [[ -z "${{{variable}:-}}" ]]; then
  printf 'Validated cache unavailable for {module_id}; {fallback_action}.\\n' >&2
fi"""


def _slurm_files(
    root: Path, project_id: str, commands: Sequence[str], *, target_wall_hours: float,
    python_executable: str, package_root: str,
    conformational_view_ids: Sequence[str] = (),
    resource_table_enabled: bool = True,
    finding_picker_enabled: bool = True,
    maximum_parallel_cpus: int = 1,
    coordinate_cache_enabled: bool = False,
    coordinate_cache_workers: int = 1,
    coordinate_cache_stride: int = 1,
    automatic_context_stage_counts: Optional[Mapping[int, int]] = None,
    rmsf_permutation_enabled: bool = False,
    integrated_comparison_enabled: bool = False,
) -> list[str]:
    stage_members = _GENERIC_STAGE_COMMANDS
    stages = {
        stage: [command for command in commands if command in members]
        for stage, members in stage_members.items()
    }
    stages = {stage: values for stage, values in stages.items() if values}
    known_commands = set().union(*stage_members.values())
    unknown = sorted(set(commands).difference(known_commands))
    if unknown:
        raise QuickstartError(
            "generated workflow has commands without a dependency stage: "
            + ", ".join(unknown)
        )
    if not stages:
        raise QuickstartError("analysis config disables every executable analysis module")
    wall_minutes = int(math.ceil(float(target_wall_hours) * 60.0))
    wall_limit = f"{wall_minutes // 60:02d}:{wall_minutes % 60:02d}:00"
    coordinate_cache_worker = ""
    coordinate_cache_filename = None
    cache_submit_lines: list[str] = []
    preflight_submission = 'PREFLIGHT_JOB=$(sbatch --parsable "$ROOT/run_preflight.slurm")'
    if coordinate_cache_enabled:
        if (
            isinstance(coordinate_cache_workers, bool)
            or not isinstance(coordinate_cache_workers, int)
            or coordinate_cache_workers <= 0
            or coordinate_cache_workers > maximum_parallel_cpus
        ):
            raise QuickstartError(
                "coordinate cache worker count must be within the CPU envelope"
            )
        if (
            isinstance(coordinate_cache_stride, bool)
            or not isinstance(coordinate_cache_stride, int)
            or coordinate_cache_stride <= 0
        ):
            raise QuickstartError("coordinate cache stride must be a positive integer")
        campaign_plan = load_json(root / "campaign-resource-plan.json")
        cache_tasks = [
            row for row in campaign_plan.get("tasks", [])
            if isinstance(row, dict)
            and row.get("task_id") == "preprocessing:coordinate_cache"
        ]
        if len(cache_tasks) != 1:
            raise QuickstartError(
                "enabled coordinate cache lacks one campaign resource task"
            )
        cache_hours = float(
            cache_tasks[0]["estimated_wall_hours_at_effective_cpu_cap"]
        )
        cache_minutes = min(
            wall_minutes,
            max(30, int(math.ceil(cache_hours * 60.0)) + 30),
        )
        cache_wall_limit = (
            f"{cache_minutes // 60:02d}:{cache_minutes % 60:02d}:00"
        )
        cache_memory_gib = max(4, math.ceil(0.5 * coordinate_cache_workers))
        coordinate_cache_filename = "run_coordinate_cache.slurm"
        coordinate_cache_worker = f"""#!/usr/bin/env bash
#SBATCH --job-name=sma-{project_id[:16]}-cache
#SBATCH --time={cache_wall_limit}
#SBATCH --cpus-per-task={coordinate_cache_workers}
#SBATCH --mem={cache_memory_gib}G
#SBATCH --output={root}/logs/%j-coordinate-cache.out
#SBATCH --error={root}/logs/%j-coordinate-cache.err
set -euo pipefail
ROOT={json.dumps(str(root))}
PYTHON_DEFAULT={json.dumps(python_executable)}
PYTHON="${{SALSBURY_MD_ANALYSIS_PYTHON:-$PYTHON_DEFAULT}}"
PACKAGE_ROOT_DEFAULT={json.dumps(package_root)}
PACKAGE_ROOT="${{SALSBURY_MD_ANALYSIS_PYTHONPATH:-$PACKAGE_ROOT_DEFAULT}}"
export PYTHONPATH="$PACKAGE_ROOT${{PYTHONPATH:+:$PYTHONPATH}}"
mkdir -p "$ROOT/results/coordinate-cache" "$ROOT/logs"
FINAL="$ROOT/results/coordinate-cache/report.json"
SUMMARY="$FINAL.summary.json"
if [[ -e "$FINAL" ]]; then
  "$PYTHON" - "$FINAL" "$SUMMARY" "$ROOT/coordinate-cache/system-cache.json" <<'PY'
import hashlib, json, sys
report_path, summary_path, manifest_path = sys.argv[1:]
report = json.load(open(report_path, encoding='utf-8'))
summary = json.load(open(summary_path, encoding='utf-8'))
digest = hashlib.sha256(open(report_path, 'rb').read()).hexdigest()
if report.get('technical_status') != 'complete':
    raise SystemExit('existing coordinate-cache report is incomplete')
if report.get('cache_stride') != {coordinate_cache_stride}:
    raise SystemExit('existing coordinate-cache stride does not match the plan')
if summary.get('technical_status') != 'complete' or summary.get('report_sha256') != digest:
    raise SystemExit('existing coordinate-cache sidecar is incomplete or hash-mismatched')
cached = json.load(open(manifest_path, encoding='utf-8'))
if not cached.get('systems'):
    raise SystemExit('existing coordinate-cache manifest has no systems')
PY
  printf 'Reusing complete coordinate cache %s.\n' "$ROOT/coordinate-cache"
  exit 0
fi
if [[ -e "$ROOT/coordinate-cache" ]]; then
  printf 'Coordinate cache exists without an accepted report; preserving it and failing closed.\n' >&2
  exit 1
fi
TMP="$FINAL.tmp.$SLURM_JOB_ID"
SUMMARY_TMP="$TMP.summary.json"
"$PYTHON" -m salsbury_md_analysis run-coordinate-cache-instrumented \
  "$ROOT/system.json" --output "$ROOT/coordinate-cache" \
  --workers {coordinate_cache_workers} --cache-stride {coordinate_cache_stride} \
  --summary-sidecar "$SUMMARY_TMP" \
  --installed-report-path "$FINAL" > "$TMP"
"$PYTHON" - "$TMP" "$SUMMARY_TMP" "$ROOT/coordinate-cache/system-cache.json" <<'PY'
import hashlib, json, sys
report_path, summary_path, manifest_path = sys.argv[1:]
report = json.load(open(report_path, encoding='utf-8'))
summary = json.load(open(summary_path, encoding='utf-8'))
digest = hashlib.sha256(open(report_path, 'rb').read()).hexdigest()
if report.get('technical_status') != 'complete':
    raise SystemExit('coordinate-cache report did not complete')
if report.get('cache_stride') != {coordinate_cache_stride}:
    raise SystemExit('coordinate-cache report stride does not match the plan')
if summary.get('technical_status') != 'complete' or summary.get('report_sha256') != digest:
    raise SystemExit('coordinate-cache sidecar is incomplete or hash-mismatched')
cached = json.load(open(manifest_path, encoding='utf-8'))
if not cached.get('systems'):
    raise SystemExit('coordinate-cache manifest has no systems')
PY
ln "$SUMMARY_TMP" "$SUMMARY"
ln "$TMP" "$FINAL"
rm "$TMP" "$SUMMARY_TMP"
"""
        cache_submit_lines = [
            'CACHE_JOB=$(sbatch --parsable "$ROOT/run_coordinate_cache.slurm")',
            'CACHE_JOB="${CACHE_JOB%%;*}"',
        ]
        preflight_submission = (
            'PREFLIGHT_JOB=$(sbatch --parsable '
            '--dependency="afterok:$CACHE_JOB" "$ROOT/run_preflight.slurm")'
        )
    interaction_source_exports = "\n".join(
        _validated_cache_export_shell(
            variable, root / f"results/{command}/report.json", module_id,
            fallback_action="continuing without that optional source report",
        )
        for command, variable, module_id in (
            ("hydrogen-bond-discovery", "SALSBURY_MD_ANALYSIS_HYDROGEN_BOND_DISCOVERY_REPORT", "hydrogen_bond_discovery"),
            ("water-mediated-hydrogen-bonds", "SALSBURY_MD_ANALYSIS_WATER_HYDROGEN_BOND_REPORT", "water_mediated_hydrogen_bond_networks"),
            ("ion-geometry", "SALSBURY_MD_ANALYSIS_ION_GEOMETRY_REPORT", "ion_coordination_geometry"),
            ("ion-atmosphere", "SALSBURY_MD_ANALYSIS_ION_ATMOSPHERE_REPORT", "ion_atmosphere"),
            ("multivalent-bridges", "SALSBURY_MD_ANALYSIS_MULTIVALENT_BRIDGES_REPORT", "multivalent_molecular_bridges"),
            ("hydration-density-channels", "SALSBURY_MD_ANALYSIS_HYDRATION_DENSITY_REPORT", "hydration_density_channels"),
            ("nucleic-acid-structure", "SALSBURY_MD_ANALYSIS_NUCLEIC_ACID_STRUCTURE_REPORT", "nucleic_acid_structure"),
        )
        if command in commands
    )
    cache_exports = {
        0: "",
        1: "\n".join((
            f"export SALSBURY_MD_ANALYSIS_PREFLIGHT_REPORT={json.dumps(str(root / 'preflight.report.json'))}",
            _validated_cache_export_shell(
                "SALSBURY_MD_ANALYSIS_COMMON_PCA_REPORT",
                root / "results/common-pca/report.json", "common_pca",
            ),
            _validated_cache_export_shell(
                "SALSBURY_MD_ANALYSIS_DCCM_REPORT",
                root / "results/dccm/report.json", "dccm",
            ),
            _validated_cache_export_shell(
                "SALSBURY_MD_ANALYSIS_RMSD_RG_REPORT",
                root / "results/rmsd-rg/report.json", "replica_rmsd_rg",
            ),
            _validated_cache_export_shell(
                "SALSBURY_MD_ANALYSIS_TRAJECTORY_FEATURES_REPORT",
                root / "results/trajectory-features/report.json",
                "trajectory_features",
            ),
            interaction_source_exports,
        )),
        2: "\n".join((
            f"export SALSBURY_MD_ANALYSIS_PREFLIGHT_REPORT={json.dumps(str(root / 'preflight.report.json'))}",
            _validated_cache_export_shell(
                "SALSBURY_MD_ANALYSIS_COMMON_PCA_REPORT",
                root / "results/common-pca/report.json", "common_pca",
            ),
            _validated_cache_export_shell(
                "SALSBURY_MD_ANALYSIS_TICA_REPORT",
                root / "results/tica/report.json",
                "time_lagged_independent_component_analysis",
            ),
            _validated_cache_export_shell(
                "SALSBURY_MD_ANALYSIS_KMEANS_REPORT",
                root / "results/cluster-kmeans/report.json", "clustering_kmeans",
            ),
            _validated_cache_export_shell(
                "SALSBURY_MD_ANALYSIS_HDBSCAN_REPORT",
                root / "results/cluster-hdbscan/report.json", "clustering_hdbscan",
            ),
            _validated_cache_export_shell(
                "SALSBURY_MD_ANALYSIS_IMWKMEANS_REPORT",
                root / "results/cluster-imwkmeans/report.json", "clustering_imwkmeans",
            ),
            _validated_cache_export_shell(
                "SALSBURY_MD_ANALYSIS_ALTERNATIVE_CLUSTERING_REPORT",
                root / "results/alternative-clustering/report.json", "alternative_clustering",
            ),
            _validated_cache_export_shell(
                "SALSBURY_MD_ANALYSIS_FES_REPORT",
                root / "results/pca-fes-basins/report.json", "pca_fes_basins",
            ),
            _validated_cache_export_shell(
                "SALSBURY_MD_ANALYSIS_INTERACTION_FINGERPRINTS_REPORT",
                root / "results/interaction-fingerprints/report.json",
                "interaction_fingerprints",
            ),
        )),
    }

    def worker_text(stage: int, stage_commands: Sequence[str]) -> str:
        command_lines = "\n".join(
            f"  {json.dumps(command)}" for command in stage_commands
        )
        return f"""#!/usr/bin/env bash
#SBATCH --job-name=sma-{project_id[:20]}
#SBATCH --time={wall_limit}
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output={root}/logs/%A_%a.out
#SBATCH --error={root}/logs/%A_%a.err
set -euo pipefail
ROOT={json.dumps(str(root))}
PROJECT="$ROOT/project.json"
COMMANDS=(
{command_lines}
)
COMMAND="${{COMMANDS[$SLURM_ARRAY_TASK_ID]}}"
PYTHON_DEFAULT={json.dumps(python_executable)}
PYTHON="${{SALSBURY_MD_ANALYSIS_PYTHON:-$PYTHON_DEFAULT}}"
PACKAGE_ROOT_DEFAULT={json.dumps(package_root)}
PACKAGE_ROOT="${{SALSBURY_MD_ANALYSIS_PYTHONPATH:-$PACKAGE_ROOT_DEFAULT}}"
export PYTHONPATH="$PACKAGE_ROOT${{PYTHONPATH:+:$PYTHONPATH}}"
{cache_exports[stage]}
mkdir -p "$ROOT/results/$COMMAND" "$ROOT/logs"
TMP="$ROOT/results/$COMMAND/report.json.tmp.$SLURM_JOB_ID"
FINAL="$ROOT/results/$COMMAND/report.json"
SUMMARY="$FINAL.summary.json"
if [[ -e "$FINAL" ]]; then
  "$PYTHON" - "$FINAL" "$SUMMARY" <<'PY'
import hashlib, json, sys
report_path, summary_path = sys.argv[1:]
summary = json.load(open(summary_path, encoding='utf-8'))
digest = hashlib.sha256()
with open(report_path, 'rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(chunk)
if summary.get('technical_status') != 'complete' or summary.get('report_sha256') != digest.hexdigest():
    raise SystemExit('existing module report summary is incomplete or hash-mismatched; refusing to overwrite it')
PY
  printf 'Reusing complete result %s.\n' "$FINAL"
  exit 0
fi
SUMMARY_TMP="$TMP.summary.json"
"$PYTHON" -m salsbury_md_analysis run-instrumented "$COMMAND" "$PROJECT" \
  --hash-content --summary-sidecar "$SUMMARY_TMP" --installed-report-path "$FINAL" > "$TMP"
"$PYTHON" - "$TMP" "$SUMMARY_TMP" <<'PY'
import hashlib, json, sys
report_path, summary_path = sys.argv[1:]
summary = json.load(open(summary_path, encoding='utf-8'))
digest = hashlib.sha256()
with open(report_path, 'rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(chunk)
if summary.get('technical_status') != 'complete' or summary.get('report_sha256') != digest.hexdigest():
    raise SystemExit('module report summary is incomplete or hash-mismatched')
PY
ln "$SUMMARY_TMP" "$SUMMARY"
ln "$TMP" "$FINAL"
rm "$TMP" "$SUMMARY_TMP"
"""

    preflight = f"""#!/usr/bin/env bash
#SBATCH --job-name=sma-{project_id[:20]}-preflight
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output={root}/logs/%j-preflight.out
#SBATCH --error={root}/logs/%j-preflight.err
set -euo pipefail
ROOT={json.dumps(str(root))}
PYTHON_DEFAULT={json.dumps(python_executable)}
PYTHON="${{SALSBURY_MD_ANALYSIS_PYTHON:-$PYTHON_DEFAULT}}"
PACKAGE_ROOT_DEFAULT={json.dumps(package_root)}
PACKAGE_ROOT="${{SALSBURY_MD_ANALYSIS_PYTHONPATH:-$PACKAGE_ROOT_DEFAULT}}"
export PYTHONPATH="$PACKAGE_ROOT${{PYTHONPATH:+:$PYTHONPATH}}"
TMP="$ROOT/preflight.report.json.tmp.$SLURM_JOB_ID"
FINAL="$ROOT/preflight.report.json"
if [[ -e "$FINAL" ]]; then
  "$PYTHON" - "$FINAL" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding='utf-8'))
if report.get('technical_status') != 'complete':
    raise SystemExit('existing preflight report is not technically complete; refusing to overwrite it')
PY
  "$PYTHON" -m salsbury_md_analysis preflight-system "$ROOT/system.json" --hash-content > "$TMP"
  "$PYTHON" - "$TMP" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding='utf-8'))
if report.get('technical_status') != 'complete':
    raise SystemExit('refreshed preflight report is not technically complete')
PY
  if ! cmp -s "$TMP" "$FINAL"; then
    printf 'Refreshed preflight differs from retained report; refusing cached-result reuse.\n' >&2
    exit 1
  fi
  rm "$TMP"
  printf 'Revalidated and reused complete preflight %s.\n' "$FINAL"
  exit 0
fi
"$PYTHON" -m salsbury_md_analysis preflight-system "$ROOT/system.json" --hash-content > "$TMP"
"$PYTHON" - "$TMP" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding='utf-8'))
if report.get('technical_status') != 'complete':
    raise SystemExit('preflight report is not technically complete')
PY
ln "$TMP" "$FINAL"
rm "$TMP"
"""
    stage_submit_lines = []
    previous_job = "PREFLIGHT_JOB"
    for stage in sorted(stages):
        variable = f"STAGE_{stage}_JOB"
        stage_submit_lines.extend([
            f'{variable}=$(sbatch --parsable --dependency="afterok:${previous_job}" '
            f'--array=0-{len(stages[stage]) - 1}%{maximum_parallel_cpus} '
            f'"$ROOT/run_stage_{stage}_array.slurm")',
            f'{variable}="${{{variable}%%;*}}"',
            f"printf 'Submitted analysis stage {stage} %s with {len(stages[stage])} methods.\\n' \"${variable}\"",
        ])
        previous_job = variable
    stage_submit_text = "\n".join(stage_submit_lines)
    context_submit_lines: list[str] = []
    context_previous_job = "PREFLIGHT_JOB"
    context_final_job: Optional[str] = None
    maximum_base_stage_width = max(len(values) for values in stages.values())
    context_parallel_cap = max(
        1, maximum_parallel_cpus - maximum_base_stage_width
    )
    for context_stage, count in sorted(
        (automatic_context_stage_counts or {}).items()
    ):
        if count <= 0:
            continue
        variable = f"CONTEXT_STAGE_{context_stage}_JOB"
        context_submit_lines.extend([
            f'{variable}=$(sbatch --parsable '
            f'--dependency="afterok:${context_previous_job}" '
            f'--array=0-{count - 1}%{min(count, context_parallel_cap)} '
            f'"$ROOT/run_automatic_context_stage_{context_stage}_array.slurm")',
            f'{variable}="${{{variable}%%;*}}"',
            f"printf 'Submitted automatic chemical-context stage "
            f"{context_stage} %s with {count} methods.\\n' \"${variable}\"",
        ])
        context_previous_job = variable
        context_final_job = variable
    context_submit_text = "\n".join(context_submit_lines)
    context_final_dependency = (
        f':${{{context_final_job}}}' if context_final_job else ""
    )
    view_submit_line = (
        "\n".join([
            f'VIEW_SUBMISSION_OUTPUT=$("$ROOT/submit-conformational-views.sh" '
            f'"${previous_job}{context_final_dependency}")',
            "printf '%s\\n' \"$VIEW_SUBMISSION_OUTPUT\"",
            "VIEW_FINAL_LINE=\"${VIEW_SUBMISSION_OUTPUT##*$'\\n'}\"",
            'VIEW_FINAL_JOBS="${VIEW_FINAL_LINE#FINAL_JOB_IDS=}"',
            'FINAL_DEPENDENCIES="${FINAL_DEPENDENCIES}:$VIEW_FINAL_JOBS"',
        ]) if conformational_view_ids else ""
    )
    reporting_commands: list[tuple[str, str]] = []
    if rmsf_permutation_enabled:
        reporting_commands.append(("rmsf_permutation_inference", """RMSF_INFERENCE_DIR="$ROOT/results/rmsf-permutation-inference"
mkdir -p "$RMSF_INFERENCE_DIR"
RMSF_INFERENCE_TMP="$RMSF_INFERENCE_DIR/report.json.tmp.$SLURM_JOB_ID"
RMSF_INFERENCE_FINAL="$RMSF_INFERENCE_DIR/report.json"
if [[ -e "$RMSF_INFERENCE_FINAL" ]]; then
  printf 'RMSF permutation report already exists; refusing overwrite: %s\n' "$RMSF_INFERENCE_FINAL" >&2
  exit 1
fi
"$PYTHON" -m salsbury_md_analysis rmsf-permutation-from-report \
  "$ROOT/results/rmsf/report.json" "$ROOT/analysis-config.json" > "$RMSF_INFERENCE_TMP"
"$PYTHON" - "$RMSF_INFERENCE_TMP" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding='utf-8'))
if report.get('technical_status') != 'complete':
    raise SystemExit('RMSF permutation inference did not complete')
PY
ln "$RMSF_INFERENCE_TMP" "$RMSF_INFERENCE_FINAL"
rm "$RMSF_INFERENCE_TMP"
"""))
    if integrated_comparison_enabled:
        reporting_commands.append(("integrated_comparison", """INTEGRATED_DIR="$ROOT/results/integrated-comparison"
mkdir -p "$INTEGRATED_DIR"
INTEGRATED_TMP="$INTEGRATED_DIR/report.json.tmp.$SLURM_JOB_ID"
INTEGRATED_FINAL="$INTEGRATED_DIR/report.json"
if [[ -e "$INTEGRATED_FINAL" ]]; then
  printf 'Integrated comparison report already exists; refusing overwrite: %s\n' "$INTEGRATED_FINAL" >&2
  exit 1
fi
"$PYTHON" -m salsbury_md_analysis integrate-comparison-results \
  "$ROOT" > "$INTEGRATED_TMP"
"$PYTHON" - "$INTEGRATED_TMP" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding='utf-8'))
contract = report.get('integration_contract', {})
if report.get('technical_status') != 'complete':
    raise SystemExit('integrated comparison did not complete')
if report.get('unreviewed_complete_report_count') != 0:
    raise SystemExit('integrated comparison silently omitted a completed report')
if contract.get('all_completed_reports_reviewed') is not True:
    raise SystemExit('integrated comparison lacks all-report review evidence')
PY
ln "$INTEGRATED_TMP" "$INTEGRATED_FINAL"
rm "$INTEGRATED_TMP"
"""))
    if resource_table_enabled:
        reporting_commands.append(("resource_summary", """RESOURCE_TMP="$ROOT/final-resource-summary.json.tmp.$SLURM_JOB_ID"
RESOURCE_FINAL="$ROOT/final-resource-summary.json"
if [[ -e "$RESOURCE_FINAL" ]]; then
  printf 'Final resource summary already exists; refusing overwrite: %s\\n' "$RESOURCE_FINAL" >&2
  exit 1
fi
"$PYTHON" -m salsbury_md_analysis summarize-execution-resources "$ROOT" > "$RESOURCE_TMP"
"$PYTHON" - "$RESOURCE_TMP" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding='utf-8'))
if report.get('technical_status') != 'complete' or report.get('scientific_status') != 'not evaluated':
    raise SystemExit('final resource/frame table lacks its required technical/scientific status boundary')
PY
ln "$RESOURCE_TMP" "$RESOURCE_FINAL"
rm "$RESOURCE_TMP"
"""))
    if finding_picker_enabled:
        reporting_commands.append(("finding_picker", """FINDING_TMP="$ROOT/final-findings-summary.json.tmp.$SLURM_JOB_ID"
FINDING_FINAL="$ROOT/final-findings-summary.json"
if [[ -e "$FINDING_FINAL" ]]; then
  printf 'Final finding summary already exists; refusing overwrite: %s\\n' "$FINDING_FINAL" >&2
  exit 1
fi
"$PYTHON" -m salsbury_md_analysis prioritize-findings "$ROOT" > "$FINDING_TMP"
"$PYTHON" - "$FINDING_TMP" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding='utf-8'))
if report.get('technical_status') != 'complete' or report.get('scientific_status') != 'not evaluated':
    raise SystemExit('finding prioritization lacks its required technical/scientific status boundary')
PY
ln "$FINDING_TMP" "$FINDING_FINAL"
rm "$FINDING_TMP"
"""))
    early_reporting_ids = {
        "rmsf_permutation_inference", "integrated_comparison",
    }
    early_reporting_workers: Dict[str, str] = {}
    retained_reporting_commands: list[tuple[str, str]] = []
    for reporting_id, reporting_command in reporting_commands:
        if reporting_id not in early_reporting_ids:
            retained_reporting_commands.append((reporting_id, reporting_command))
            continue
        filename = f"run_reporting_{reporting_id}.slurm"
        early_reporting_workers[filename] = f"""#!/usr/bin/env bash
#SBATCH --job-name=sma-{project_id[:16]}-{reporting_id[:8]}
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output={root}/logs/%j-{reporting_id}.out
#SBATCH --error={root}/logs/%j-{reporting_id}.err
set -euo pipefail
ROOT={json.dumps(str(root))}
PYTHON_DEFAULT={json.dumps(python_executable)}
PYTHON="${{SALSBURY_MD_ANALYSIS_PYTHON:-$PYTHON_DEFAULT}}"
PACKAGE_ROOT_DEFAULT={json.dumps(package_root)}
PACKAGE_ROOT="${{SALSBURY_MD_ANALYSIS_PYTHONPATH:-$PACKAGE_ROOT_DEFAULT}}"
export PYTHONPATH="$PACKAGE_ROOT${{PYTHONPATH:+:$PYTHONPATH}}"
{reporting_command}
"""
    reporting_commands = retained_reporting_commands
    if reporting_commands:
        wrapped_reporting_commands = ["FINAL_REPORTING_STATUS=0"]
        for reporting_id, reporting_command in reporting_commands:
            wrapped_reporting_commands.append(f"""set +e
(
set -euo pipefail
{reporting_command}
)
REPORTING_COMPONENT_STATUS=$?
set -e
if (( REPORTING_COMPONENT_STATUS != 0 )); then
  printf 'Final reporting component failed: {reporting_id} (exit %s)\\n' "$REPORTING_COMPONENT_STATUS" >&2
  FINAL_REPORTING_STATUS=1
fi""")
        wrapped_reporting_commands.append('exit "$FINAL_REPORTING_STATUS"')
        reporting_command_text = "\n".join(wrapped_reporting_commands)
    else:
        reporting_command_text = (
            "printf '{\"technical_status\":\"complete\",\"scientific_status\":\"not evaluated\",\"reporting_disabled\":true}\\n' "
            '"> \"$ROOT/final-reporting-disabled.json\"'
        )
    finalizer = f"""#!/usr/bin/env bash
#SBATCH --job-name=sma-{project_id[:20]}-final
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output={root}/logs/%j-final-reporting.out
#SBATCH --error={root}/logs/%j-final-reporting.err
set -euo pipefail
ROOT={json.dumps(str(root))}
PYTHON_DEFAULT={json.dumps(python_executable)}
PYTHON="${{SALSBURY_MD_ANALYSIS_PYTHON:-$PYTHON_DEFAULT}}"
PACKAGE_ROOT_DEFAULT={json.dumps(package_root)}
PACKAGE_ROOT="${{SALSBURY_MD_ANALYSIS_PYTHONPATH:-$PACKAGE_ROOT_DEFAULT}}"
export PYTHONPATH="$PACKAGE_ROOT${{PYTHONPATH:+:$PYTHONPATH}}"
{reporting_command_text}
"""
    cache_submit_text = "\n".join(cache_submit_lines)
    cache_status_text = (
        "printf 'Submitted coordinate-cache job %s.\\n' \"$CACHE_JOB\""
        if coordinate_cache_enabled else ""
    )
    submit = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT={json.dumps(str(root))}
mkdir -p "$ROOT/logs" "$ROOT/results"
PYTHON_DEFAULT={json.dumps(python_executable)}
PYTHON="${{SALSBURY_MD_ANALYSIS_PYTHON:-$PYTHON_DEFAULT}}"
PACKAGE_ROOT_DEFAULT={json.dumps(package_root)}
PACKAGE_ROOT="${{SALSBURY_MD_ANALYSIS_PYTHONPATH:-$PACKAGE_ROOT_DEFAULT}}"
export PYTHONPATH="$PACKAGE_ROOT${{PYTHONPATH:+:$PYTHONPATH}}"
"$PYTHON" -m salsbury_md_analysis validate-manifest system "$ROOT/system.json" --check-paths
"$PYTHON" -m salsbury_md_analysis validate-manifest project "$ROOT/project.json" --check-paths
{cache_submit_text}
{preflight_submission}
PREFLIGHT_JOB="${{PREFLIGHT_JOB%%;*}}"
{stage_submit_text}
{context_submit_text}
FINAL_DEPENDENCIES="${{{previous_job}}}{context_final_dependency}"
{view_submit_line}
FINAL_JOB=$(sbatch --parsable --dependency="afterok:$FINAL_DEPENDENCIES" "$ROOT/run_finalize_reporting.slurm")
FINAL_JOB="${{FINAL_JOB%%;*}}"
{cache_status_text}
printf 'Submitted preflight/hash job %s.\\n' "$PREFLIGHT_JOB"
printf 'Submitted final resource/frame reporting job %s.\\n' "$FINAL_JOB"
printf 'Results will appear under %s/results.\\n' "$ROOT"
"""
    generated = []
    for stage in sorted(stages):
        filename = f"run_stage_{stage}_array.slurm"
        (root / filename).write_text(
            worker_text(stage, stages[stage]), encoding="utf-8"
        )
        generated.append(filename)
    (root / "run_preflight.slurm").write_text(preflight, encoding="utf-8")
    if coordinate_cache_filename is not None:
        (root / coordinate_cache_filename).write_text(
            coordinate_cache_worker, encoding="utf-8"
        )
    for filename, worker in sorted(early_reporting_workers.items()):
        (root / filename).write_text(worker, encoding="utf-8")
        generated.append(filename)
    (root / "run_finalize_reporting.slurm").write_text(finalizer, encoding="utf-8")
    (root / "submit.sh").write_text(submit, encoding="utf-8")
    _json_write(
        root / "workflow-stages.json",
        {
            "workflow_schema": "salsbury-staged-workflow-v2",
            "authoritative_dependency_graph": "local-execution-plan.json",
            "dependency_policy": (
                "stage numbers group generated commands; they do not make every "
                "task depend on the preceding stage. The execution adapter derives "
                "success-only edges from each task's irreplaceable inputs."
            ),
            "maximum_parallel_cpus": maximum_parallel_cpus,
            "coordinate_cache": {
                "enabled": coordinate_cache_enabled,
                "worker_processes": (
                    coordinate_cache_workers if coordinate_cache_enabled else 0
                ),
                "dependency": (
                    "only before preflights whose manifests are built inside the "
                    "coordinate cache; base-input preflight and analyses are independent"
                ),
                "source_scope": "all original physical frames",
            },
            "automatic_context_stages": {
                str(stage): {
                    "task_count": count,
                    "array_parallelism_cap": min(count, context_parallel_cap),
                    "dependency": "per-task graph in local-execution-plan.json",
                }
                for stage, count in sorted(
                    (automatic_context_stage_counts or {}).items()
                )
            },
            "slurm_array_parallelism_contract": (
                "Resource waves reserve no more than maximum_parallel_cpus and "
                "aggregate memory at one time; resource barriers do not create "
                "scientific success dependencies."
            ),
            "stages": [
                {
                    "stage": stage,
                    "commands": stages[stage],
                    "dependency": "per-task graph in local-execution-plan.json",
                    "upstream_report_reuse": {
                        0: [],
                        1: [
                            "common_pca", "dccm", "replica_rmsd_rg",
                            "trajectory_features",
                        ],
                        2: ["common_pca", "clustering_kmeans"],
                    }[stage],
                }
                for stage in sorted(stages)
            ],
        },
    )
    os.chmod(root / "submit.sh", 0o755)
    return [
        "run_preflight.slurm", "run_finalize_reporting.slurm", *generated,
        *([coordinate_cache_filename] if coordinate_cache_filename else []),
        "workflow-stages.json", "submit.sh",
    ]


def prepare_standard_analysis(
    *,
    pdb_path: Path,
    psf_path: Optional[Path],
    trajectories: Sequence[Path],
    output_directory: Path,
    project_id: str,
    frame_interval_ps: float,
    first_frame_time_ps: float = 0.0,
    temperature_kelvin: float = 300.0,
    target_wall_hours: Optional[float] = None,
    dssp_executable: Optional[str] = None,
    dssr_executable: Optional[str] = None,
    config_path: Optional[Path] = None,
    generate_connectivity_openmm: bool = False,
    openmm_bond_definitions: Sequence[Path] = (),
    energetic_charmm_parameter_files: Sequence[Path] = (),
    energetic_openmm_system_xml: Optional[Path] = None,
    energetic_gromacs_tpr: Optional[Path] = None,
) -> Dict[str, object]:
    """Prepare manifests, budgets, and a local or Slurm workflow without touching inputs."""

    project_id = _safe_id(project_id, "project_id")
    if not trajectories:
        raise QuickstartError("at least one trajectory is required")
    for value, label in (
        (frame_interval_ps, "frame_interval_ps"),
        (temperature_kelvin, "temperature_kelvin"),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise QuickstartError(f"{label} must be finite and positive")
    pdb = pdb_path.expanduser().resolve(strict=True)
    if psf_path is not None and generate_connectivity_openmm:
        raise QuickstartError(
            "choose either supplied connectivity or --generate-connectivity-openmm, not both"
        )
    if psf_path is None and not generate_connectivity_openmm:
        raise QuickstartError(
            "explicit PSF, PRMTOP/PARM7, or bond JSON connectivity is required unless "
            "--generate-connectivity-openmm is selected"
        )
    psf = (
        psf_path.expanduser().resolve(strict=True)
        if psf_path is not None else None
    )
    bond_definition_paths = [
        path.expanduser().resolve(strict=True) for path in openmm_bond_definitions
    ]
    force_field_parameters = _force_field_parameter_spec(
        charmm_parameter_files=energetic_charmm_parameter_files,
        openmm_system_xml=energetic_openmm_system_xml,
        gromacs_tpr=energetic_gromacs_tpr,
    )
    if bond_definition_paths and not generate_connectivity_openmm:
        raise QuickstartError(
            "--openmm-bond-definitions requires --generate-connectivity-openmm"
        )
    trajectory_paths = [path.expanduser().resolve(strict=True) for path in trajectories]
    if pdb.suffix.lower() not in {".pdb", ".ent"}:
        raise QuickstartError("--pdb must name a PDB file")
    if any(path.suffix.lower() != ".dcd" for path in trajectory_paths):
        raise QuickstartError("every --trajectory must name a DCD file")
    try:
        pdb_probe = probe_topology(pdb)
        trajectory_probes = [probe_trajectory(path) for path in trajectory_paths]
    except (FileProbeError, OSError) as exc:
        raise QuickstartError(str(exc)) from exc
    atom_count = int(pdb_probe["atom_count"])
    generated_connectivity = None
    if psf is None:
        try:
            generated_connectivity = export_pdb_connectivity(
                pdb, additional_bond_definitions=bond_definition_paths
            )
        except (OpenMMConnectivityError, OSError) as exc:
            raise QuickstartError(str(exc)) from exc
        if int(generated_connectivity["atom_count"]) != atom_count:
            raise QuickstartError("PDB and OpenMM-generated connectivity atom counts differ")
        with tempfile.TemporaryDirectory(prefix="salsbury-openmm-connectivity-") as temporary:
            temporary_connectivity = Path(temporary) / "generated.bonds.json"
            _json_write(temporary_connectivity, generated_connectivity)
            reference_connectivity_check = _validate_reference_connectivity(
                pdb, temporary_connectivity, atom_count
            )
    else:
        try:
            psf_probe = probe_connectivity(psf)
        except (FileProbeError, OSError) as exc:
            raise QuickstartError(str(exc)) from exc
        if int(psf_probe["atom_count"]) != atom_count:
            raise QuickstartError("PDB and connectivity atom counts differ")
        reference_connectivity_check = _validate_reference_connectivity(
            pdb, psf, atom_count
        )
    frame_counts = []
    for path, probe in zip(trajectory_paths, trajectory_probes):
        if int(probe["atom_count"]) != atom_count:
            raise QuickstartError(
                f"trajectory atom count differs from PDB/connectivity: {path}"
            )
        frame_counts.append(int(probe["declared_frame_count"]))
    if len(set(frame_counts)) != 1:
        raise QuickstartError(
            "the first seamless workflow requires equal-length replica trajectories; "
            "use an explicit manifest for unequal or segmented replicas"
        )
    root = _require_new_directory(output_directory)
    generated_connectivity_file = None
    if generated_connectivity is not None:
        generated_directory = root / "generated-connectivity"
        generated_directory.mkdir()
        generated_connectivity_file = (
            generated_directory / f"{project_id}.bonds.json"
        )
        _json_write(generated_connectivity_file, generated_connectivity)
        psf = generated_connectivity_file.resolve(strict=True)
    assert psf is not None
    composition = _composition(pdb)
    registry_ids = [module.module_id for module in list_modules()]
    raw_view_plan = composition["conformational_view_plan"]
    assert isinstance(raw_view_plan, dict)
    raw_views = raw_view_plan["views"]
    assert isinstance(raw_views, list)
    view_ids_for_config = [
        str(view["view_id"]) for view in raw_views if isinstance(view, dict)
    ]
    try:
        analysis_config = load_analysis_config(
            config_path, registry_ids, view_ids_for_config
        )
    except (AnalysisConfigError, OSError) as exc:
        raise QuickstartError(str(exc)) from exc
    dssr = _discover_dssr_executable(dssr_executable)
    if not bool(composition.get("has_nucleic_acid")):
        dssr_probe: Dict[str, object] = {
            "status": "not_available", "reason": "no_duplex_dna_or_rna",
            "executable": dssr,
        }
    elif dssr is None:
        dssr_probe = {
            "status": "not_available", "reason": "dssr_not_installed",
            "executable": None,
        }
    else:
        dssr_probe = probe_dssr_reference_duplex(dssr, pdb)
    if target_wall_hours is not None:
        if (
            isinstance(target_wall_hours, bool)
            or not isinstance(target_wall_hours, (int, float))
            or not math.isfinite(float(target_wall_hours))
            or float(target_wall_hours) <= 0.0
        ):
            raise QuickstartError("target_wall_hours must be finite and positive")
        execution = analysis_config["execution"]
        assert isinstance(execution, dict)
        execution["maximum_hours_per_cpu"] = float(target_wall_hours)
        execution["maximum_total_cpu_hours"] = (
            int(execution["maximum_parallel_cpus"]) * float(target_wall_hours)
        )
    system = {
        "systems": [{
            "system_id": project_id,
            "metadata": {
                "prepared_by": "salsbury-md-analysis prepare-analysis",
                "timing_source": "user-declared --frame-interval-ps",
            },
            "replicas": [
                {
                    "replica_id": f"replica-{index + 1}",
                    "topology": str(pdb),
                    "connectivity": str(psf),
                    **(
                        {"force_field_parameters": deepcopy(force_field_parameters)}
                        if force_field_parameters is not None else {}
                    ),
                    "segments": [{
                        "segment_id": "production",
                        "trajectory": str(path),
                        "timing": {
                            "first_frame_time": float(first_frame_time_ps),
                            "frame_interval": float(frame_interval_ps),
                            "unit": "ps",
                        },
                    }],
                }
                for index, path in enumerate(trajectory_paths)
            ],
        }]
    }
    system_path = root / "system.json"
    _json_write(system_path, system)
    validate_system(system, source_path=system_path, check_paths=True)
    energetic_parameter_probe = probe_energetic_parameter_source(
        pdb, psf, force_field_parameters,
    )
    energetic_parameter_available = (
        energetic_parameter_probe.get("availability_status") == "available"
    )
    execution_config = analysis_config["execution"]
    assert isinstance(execution_config, dict)
    coordinate_cache_input = execution_config.get("coordinate_cache_input")
    if coordinate_cache_input is not None:
        try:
            cache_reuse = validate_reusable_coordinate_cache(
                Path(str(coordinate_cache_input)), system_path
            )
        except (OSError, ValueError) as exc:
            raise QuickstartError(str(exc)) from exc
        _json_write(root / "coordinate-cache-reuse.json", cache_reuse)
    dssp = _discover_dssp_executable(dssp_executable)
    sampling_plan = automatic_sampling_plan(
        system_path,
        simulation_kind="unbiased_md",
        module_ids=_applicable_sampling_modules(
            composition, analysis_config, dssp_executable=dssp,
            energetic_parameter_available=energetic_parameter_available,
        ),
        b_vs_2b=bool(analysis_config["sampling"]["b_vs_2b_sensitivity"]),  # type: ignore[index]
        replica_diagnostics=bool(
            analysis_config["sampling"]["optional_replica_diagnostics"]  # type: ignore[index]
        ),
        target_wall_seconds=float(
            analysis_config["execution"]["maximum_hours_per_cpu"]  # type: ignore[index]
        ) * 3600.0,
        time_safety_factor=float(
            analysis_config["execution"]["time_safety_factor"]  # type: ignore[index]
        ),
        campaign_execution=analysis_config["execution"],  # type: ignore[arg-type]
    )
    definitions, commands, exclusions = _generic_definitions(
        composition,
        sampling_plan,
        frame_counts_per_replica=frame_counts,
        dssp_executable=dssp,
        dssr_probe=dssr_probe,
    )
    inference_config = analysis_config["inference"]
    assert isinstance(inference_config, dict)
    inferred_modules: list[str] = []
    if bool(inference_config["automatic_chemical_context"]):
        sampling_rows = _sampling_rows(sampling_plan)
        maximum_by_module = {
            module_id: int(row.get("selected_frame_count", sum(frame_counts)))
            for module_id, row in sampling_rows.items()
            if isinstance(row.get("selected_frame_count"), int)
            and not isinstance(row.get("selected_frame_count"), bool)
        }
        try:
            inferred = infer_standard_chemistry_definitions(
                pdb,
                maximum_frames_by_module=maximum_by_module,
                total_source_frames=sum(frame_counts),
                dssr_executable=dssr,
                ion_site_classification_enabled=bool(
                    inference_config["ion_site_classification_enabled"]
                ),
            )
        except (AutomaticChemistryError, OSError) as exc:
            raise QuickstartError(
                "automatic chemical-context inference failed: " + str(exc)
            ) from exc
        inferred_definitions = inferred["definitions"]
        assert isinstance(inferred_definitions, dict)
        dssr_definition = inferred_definitions.get("nucleic_acid_structure")
        parameter_contract = dssr_probe.get("helical_step_parameters")
        if (
            isinstance(dssr_definition, dict)
            and isinstance(parameter_contract, dict)
            and isinstance(parameter_contract.get("object_path"), list)
            and isinstance(parameter_contract.get("fields"), dict)
        ):
            object_path = list(parameter_contract["object_path"])
            fields = parameter_contract["fields"]
            dssr_definition["numeric_queries"] = [{
                "query_id": f"helical-step-{name}",
                "path": [*object_path, str(fields[name])],
                "missing_policy": "fail",
            } for name in ("shift", "slide", "rise", "tilt", "roll", "twist")]
        definitions.update(deepcopy(inferred_definitions))
        inferred_modules = [str(value) for value in inferred["applicable_modules"]]
        commands.extend(
            _GENERIC_CHEMISTRY_COMMANDS[module_id]
            for module_id in inferred_modules
            if module_id in _GENERIC_CHEMISTRY_COMMANDS
        )
        exclusions.update({
            str(key): str(value)
            for key, value in inferred["not_applicable_modules"].items()
        })
        _json_write(root / "automatic-chemical-context.json", inferred)
    requested = [
        "provenance_manifest", "preflight_inventory", "common_atom_mapping",
        "structural_integrity_qc", "replica_rmsd_rg", "pooled_rmsf", "dccm",
        "generalized_correlation_and_information", "information_dynamics",
        "perturbation_response_dynamics", "trajectory_reweighting",
        "correlation_networks", "individual_pca", "common_pca",
        "allosteric_pathways", "multivalent_molecular_bridges",
        "energetic_network_embeddings",
        "hydration_density_channels", "ensemble_pocket_dynamics",
        "interaction_fingerprints", "spatial_interaction_ensembles",
        "helical_mechanics",
        "interaction_persistence", "random_feature_koopman",
        "time_lagged_independent_component_analysis",
        "pca_fes_basins", "clustering_kmeans", "clustering_hdbscan",
        "clustering_imwkmeans", "alternative_clustering",
        "pald_community_analysis", "representative_frames",
        "markov_state_models", "grouped_ml", "dihedral_distributions",
        "reactive_path_ensembles",
        "hydrogen_bond_discovery",
        "solvent_accessible_surface_area", "convergence_uncertainty",
    ]
    if "water_mediated_hydrogen_bond_networks" in definitions:
        requested.append("water_mediated_hydrogen_bond_networks")
    if "secondary_structure" in definitions:
        requested.append("secondary_structure")
    requested.extend(
        module_id for module_id in inferred_modules if module_id not in requested
    )
    try:
        definitions, commands, requested, config_disabled = apply_module_configuration(
            definitions, commands, requested, analysis_config
        )
    except AnalysisConfigError as exc:
        raise QuickstartError(str(exc)) from exc
    commands, requested = _apply_experimental_input_gates(
        definitions, commands, requested, exclusions, composition
    )
    helical_definition = definitions.get("helical_mechanics")
    if isinstance(helical_definition, dict):
        helical_definition["preparation_availability"] = deepcopy(dssr_probe)
    helical_availability_file: Optional[str] = None
    helical_source_configured = "nucleic_acid_structure" in definitions
    if (
        "helical_mechanics" in requested
        and (
            dssr_probe.get("status") != "available"
            or not helical_source_configured
        )
    ):
        commands = [command for command in commands if command != "helical-mechanics"]
        requested = [module_id for module_id in requested if module_id != "helical_mechanics"]
        reason = str(
            dssr_probe.get("reason", "dssr_or_duplex_unavailable")
            if dssr_probe.get("status") != "available"
            else "nucleic_acid_structure_source_not_configured"
        )
        exclusions["helical_mechanics"] = reason
        helical_availability_file = "helical-mechanics-availability.json"
        _json_write(root / helical_availability_file, {
            "module_id": "helical_mechanics",
            "technical_status": "complete", "scientific_status": "not evaluated",
            "availability_status": "not_available",
            "availability_reason": reason,
            "planner_task_created": False,
            "preparation_probe": dssr_probe,
        })
    energetic_availability_file: Optional[str] = None
    if (
        "energetic_network_embeddings" in requested
        and (
            not bool(composition.get("has_protein"))
            or not energetic_parameter_available
        )
    ):
        commands = [
            command for command in commands
            if command != "energetic-network-embeddings"
        ]
        requested = [
            module_id for module_id in requested
            if module_id != "energetic_network_embeddings"
        ]
        reason = (
            "not applicable: system contains no protein residues"
            if not bool(composition.get("has_protein"))
            else (
                "not available: "
                + str(energetic_parameter_probe.get(
                    "availability_reason", "no compatible interaction parameters"
                ))
            )
        )
        exclusions["energetic_network_embeddings"] = reason
        energetic_availability_file = "energetic-network-embeddings-availability.json"
        _json_write(root / energetic_availability_file, {
            "module_id": "energetic_network_embeddings",
            "technical_status": "complete", "scientific_status": "not evaluated",
            "availability_status": "not_available",
            "availability_reason": reason,
            "planner_task_created": False,
            "supplied_connectivity": str(psf) if psf is not None else None,
            "preparation_probe": energetic_parameter_probe,
        })
    commands, requested = _exclude_conformational_views_from_base_workflow(
        commands, requested
    )
    project = {
        "project_id": project_id,
        "analysis_profile": "standard_md_v1",
        "system_manifest": "system.json",
        "analysis_output_root": "results",
        "reference_system": project_id,
        "temperature_kelvin": float(temperature_kelvin),
        "sampling_mode": "UNBIASED_MD",
        "coordinate_unit": "angstrom",
        "time_unit": "ps",
        "periodic_coordinate_policy": "unwrap_continuous",
        "periodic_reconstruction": {
            "maximum_bond_length_angstrom": 4.0,
            "cycle_closure_tolerance_angstrom": 0.05,
            "maximum_anchor_displacement_angstrom": 100.0,
        },
        "reference_structure": str(pdb),
        "reference_connectivity": str(psf),
        "common_atom_policy": "strict",
        "selections": composition["selections"],
        "definitions": definitions,
        "requested_modules": requested,
        "protected_locations": [
            str(path) for path in [
                pdb,
                *([] if generated_connectivity_file is not None else [psf]),
                *bond_definition_paths,
                *(
                    [Path(str(value)) for value in force_field_parameters["files"]]
                    if force_field_parameters is not None else []
                ),
                *trajectory_paths,
            ]
        ],
    }
    project_path = root / "project.json"
    _json_write(project_path, project)
    validate_project(project, source_path=project_path, check_paths=True)
    _json_write(root / "sampling-plan.json", sampling_plan)
    _json_write(root / "analysis-config.json", analysis_config)
    view_ids, view_project_files = _conformational_view_projects(
        root,
        project,
        composition,
        frame_counts_per_replica=frame_counts,
        analysis_config=analysis_config,
    )
    try:
        campaign_resource_plan = plan_and_apply_complete_campaign(
            root=root,
            sampling_plan=sampling_plan,
            analysis_config=analysis_config,
            view_project_files=view_project_files,
            base_project_path=project_path,
            time_safety_factor=float(
                analysis_config["execution"]["time_safety_factor"]  # type: ignore[index]
            ),
        )
    except CampaignPlanningError as exc:
        if exc.plan is not None:
            _json_write(root / "campaign-resource-plan.json", exc.plan)
            memory = exc.plan.get("memory_feasibility")
            if isinstance(memory, Mapping) and not bool(
                memory.get("fits_configured_memory", True)
            ):
                memory_report = {
                    "report_schema": "salsbury-memory-feasibility-report-v1",
                    "technical_status": "complete",
                    "planning_status": "insufficient_memory",
                    "automatic_changes_applied": False,
                    **deepcopy(dict(memory)),
                    "requested_config_path": str(root / "analysis-config.json"),
                    "proposed_action": (
                        "Increase execution.maximum_memory_gib or rerun preparation "
                        "with --auto-disable-to-fit-memory to create and replan an "
                        "explicit reduced configuration in a new output directory."
                    ),
                }
                _json_write(root / "memory-feasibility-report.json", memory_report)
                raise QuickstartMemoryError(
                    str(exc), plan=exc.plan, analysis_config=analysis_config,
                    output_directory=root,
                ) from exc
            raise QuickstartPlanningError(
                str(exc), plan=exc.plan, analysis_config=analysis_config,
                output_directory=root,
            ) from exc
        raise QuickstartError(str(exc)) from exc
    _record_conformational_experimental_exclusions(
        root, view_project_files, exclusions, analysis_config
    )
    # Preserve the complete plan even if the coverage audit itself detects an
    # integration defect. This makes the failure diagnosable without rerunning
    # input discovery.
    _json_write(root / "campaign-resource-plan.json", campaign_resource_plan)
    experimental_planner_coverage = _experimental_planner_coverage(
        analysis_config, campaign_resource_plan, exclusions
    )
    campaign_resource_plan["experimental_module_coverage"] = (
        experimental_planner_coverage
    )
    _json_write(root / "sampling-plan.json", sampling_plan)
    _json_write(root / "campaign-resource-plan.json", campaign_resource_plan)
    effective_parallel_cpu_cap = _effective_parallel_cpu_cap(
        campaign_resource_plan
    )
    coordinate_cache_enabled = _coordinate_cache_enabled(
        analysis_config, view_ids
    )
    coordinate_cache_build_required = (
        coordinate_cache_enabled and coordinate_cache_input is None
    )
    cache_coupling = campaign_resource_plan.get("global_stride_coupling")
    coordinate_cache_stride = (
        int(cache_coupling["selected_coordinate_cache_integer_stride"])
        if coordinate_cache_enabled
        and isinstance(cache_coupling, dict)
        and isinstance(
            cache_coupling.get("selected_coordinate_cache_integer_stride"), int
        )
        else 1
    )
    coordinate_cache_files = (
        _configure_coordinate_cache_views(
            root, view_ids, cache_stride=coordinate_cache_stride,
            cache_directory=(
                Path(str(coordinate_cache_input))
                if coordinate_cache_input is not None else None
            ),
        )
        if coordinate_cache_enabled else []
    )
    coordinate_cache_workers = min(
        effective_parallel_cpu_cap,
        len(trajectory_paths),
    )
    deferred = {
        **exclusions,
        **config_disabled,
        "trajectory_features": "requires a declared scientific feature rather than an arbitrary atom pair",
        "scalar_feature_distributions": "runs after a question-linked scalar trajectory feature is declared",
        "scalar_threshold_states": "requires a physically justified threshold and sensitivity range",
        "hydrogen_bonds": (
            "optional manual fixed-feature override; automatic chemistry-backed "
            "donor-hydrogen-acceptor discovery is the production default"
        ),
        "hydrogen_bond_comparison": "requires at least two chemically mapped conditions",
        "hydrogen_bond_patterns": "runs after a direct-hydrogen-bond report has defined frame patterns",
        "grouped_regularized_classification": "requires at least two conditions and discovered hydrogen-bond features",
        "state_coordinate_exports": "runs after the user accepts a fitted FES or clustering state definition",
        "representative_structures": (
            "optional coordinate-space mean/medoid utility; observed state-centered "
            "representative frames and coordinate exports are automatic"
        ),
        "optional_observables": "requires a residue- or question-specific definition; deliberately deferred",
        "radial_distribution_functions": "requires explicit chemically meaningful atom groups",
        "nucleic_acid_geometry": "automatic ring and stacking definition generation is not yet enabled in the generic initializer",
        "ion_coordination_geometry": "automatic bound-ion ligand-shell definition generation is not yet enabled in the generic initializer",
        "ion_atmosphere": "requires supported ions and polar non-solvent solute atoms; automatic inference is configurable",
        "nucleic_acid_structure": "requires a separately licensed x3dna-dssr executable",
        "rmsf_permutation_inference": "requires a declared exchangeable-unit comparison",
        "integrated_comparison": "runs after accepted upstream reports are selected",
    }
    automatic_ids = set(requested) | _view_requested_modules(root, view_project_files)
    deferred = {
        module_id: reason for module_id, reason in deferred.items()
        if module_id not in automatic_ids
    }
    unaccounted = sorted(set(registry_ids).difference(automatic_ids, deferred))
    if unaccounted:
        raise QuickstartError(
            "quickstart module accounting is incomplete: " + ", ".join(unaccounted)
        )
    coverage = {
        "coverage_schema": "salsbury-self-service-module-coverage-v1",
        "registry_module_count": len(registry_ids),
        "automatically_requested_modules": requested,
        "automatically_executed_commands": commands,
        "deferred_or_context_specific": deferred,
        "module_status": {
            module_id: (
                {"status": "automatic", "reason": "included in the generated workflow"}
                if module_id in automatic_ids
                else {"status": "deferred", "reason": deferred[module_id]}
            )
            for module_id in registry_ids
        },
        "experimental_planner_coverage": experimental_planner_coverage,
        "interpretation": (
            "Deferred modules are visible, not silently omitted. The matched TREX validation "
            "suite is the comprehensive all-module software exercise; this initializer avoids "
            "guessing question-specific residue definitions."
        ),
    }
    _json_write(root / "module-coverage.json", coverage)
    (root / "logs").mkdir()
    (root / "results").mkdir()
    view_slurm_files = _conformational_view_slurm_files(
        root,
        project_id,
        view_ids,
        target_wall_hours=float(
            analysis_config["execution"]["maximum_hours_per_cpu"]  # type: ignore[index]
        ),
        python_executable=_active_python_executable(),
        package_root=str(Path(__file__).resolve(strict=True).parents[1]),
        maximum_parallel_cpus=effective_parallel_cpu_cap,
    )
    slurm_files = _slurm_files(
        root, project_id, commands,
        target_wall_hours=float(
            analysis_config["execution"]["maximum_hours_per_cpu"]  # type: ignore[index]
        ),
        python_executable=_active_python_executable(),
        package_root=str(Path(__file__).resolve(strict=True).parents[1]),
        conformational_view_ids=view_ids,
        resource_table_enabled=bool(analysis_config["reporting"]["resource_table_enabled"]),  # type: ignore[index]
        finding_picker_enabled=bool(analysis_config["reporting"]["finding_picker_enabled"]),  # type: ignore[index]
        maximum_parallel_cpus=effective_parallel_cpu_cap,
        coordinate_cache_enabled=coordinate_cache_build_required,
        coordinate_cache_workers=coordinate_cache_workers,
        coordinate_cache_stride=coordinate_cache_stride,
    )
    try:
        execution_artifacts = prepare_execution_artifacts(
            root,
            _execution_config_for_parallel_cpu_cap(
                analysis_config, effective_parallel_cpu_cap
            ),
        )
        planning_report_files = write_planning_report(root)
    except (ExecutionAdapterError, PlanningReportError, OSError) as exc:
        raise QuickstartError(str(exc)) from exc
    active_launcher = (
        "submit.sh" if execution_artifacts["adapter"] == "slurm"
        else "run-custom.sh" if execution_artifacts["adapter"] == "custom"
        else "run-local.sh"
    )
    adapter_description = (
        f"Slurm profile `{execution_artifacts['slurm_profile_id']}`"
        if execution_artifacts["adapter"] == "slurm"
        else (
            "external launcher contract"
            if execution_artifacts["adapter"] == "custom"
            else "local dependency-aware executor"
        )
    )
    readme = f"""# {project_id}: generated Salsbury MD analysis

Inputs were inspected read-only. The PDB, connectivity, and {len(trajectory_paths)} DCD files are
referenced by absolute path and are not copied or modified.

Review `planning-report.md` first. It lists every analysis family, the effective raw
stride over the original trajectory, and every explicitly disabled, deferred, or
inapplicable capability. Exact decisions remain in `planning-report.json`,
`campaign-resource-plan.json`, and `sampling-plan.json`. The configured whole-campaign envelope is
{analysis_config['execution']['maximum_parallel_cpus']} CPUs for at most
{analysis_config['execution']['maximum_hours_per_cpu']:g} wall hours with a 1.5 timing
safety factor. Then submit:

```bash
cd {root}
./{active_launcher}
```

The active execution adapter is the {adapter_description}. Change
`execution.submission_adapter` in the preparation config to `local`, `slurm`, or
`custom`; Slurm mode also requires `execution.slurm_profile`. All adapters execute the same
worker scripts and therefore retain the same dependencies, atomic reports, hashes,
frame selections, and scientific definitions. `execution-adapter.json` records the
choice and `local-execution-plan.json` records the workstation dependency plan.

The generated workflow covers generic structure, motion, FES, clustering, interactions,
surface, and convergence analyses. Every task declares the reports it consumes, so a
failure skips only its true descendants while unrelated work continues. Expensive PCA,
DCCM, RMSD/Rg, and K-means reports are computed once and reused only after their
project, system, and input-content hashes match. `module-coverage.json` names every
deferred capability and why it was not guessed. Residue-specific questions are
intentionally outside this first zero-configuration workflow. `conformational-views.json`
    records the topology-derived global-common-heavy, chemical-interface when applicable,
    and optional macromolecular-trace views. The active launcher runs every enabled,
    automatically applicable view; the trace view is disabled by default. Conformational
    views read a reusable, made-whole molecular-payload
    cache built from every physical frame; base solvent-dependent analyses continue to
    read the original solvated trajectories. The launchers retain the
Python executable and package source used here; after an intentional installation move,
override them with `SALSBURY_MD_ANALYSIS_PYTHON` and
`SALSBURY_MD_ANALYSIS_PYTHONPATH`.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    return {
        "technical_status": "complete",
        "project_id": project_id,
        "output_directory": str(root),
        "atom_count": atom_count,
        "replica_count": len(trajectory_paths),
        "frames_per_replica": frame_counts[0],
        "reference_connectivity_check": reference_connectivity_check,
        "connectivity_source": (
            "openmm_pdb_topology" if generated_connectivity_file is not None
            else "user_supplied"
        ),
        "configured_campaign_wall_hours": float(
            analysis_config["execution"]["maximum_hours_per_cpu"]  # type: ignore[index]
        ),
        "configured_campaign_parallel_cpus": int(
            analysis_config["execution"]["maximum_parallel_cpus"]  # type: ignore[index]
        ),
        "generated_files": [
            "system.json", "project.json", "sampling-plan.json",
            "campaign-resource-plan.json", "module-coverage.json",
            *(["coordinate-cache-reuse.json"] if coordinate_cache_input is not None else []),
            *(
                [str(generated_connectivity_file.relative_to(root))]
                if generated_connectivity_file is not None else []
            ),
            *view_project_files, *coordinate_cache_files, *view_slurm_files,
            "analysis-config.json", *slurm_files,
            *execution_artifacts["generated_files"], *planning_report_files,
            "README.md",
        ],
        "execution_adapter": execution_artifacts["adapter"],
        "slurm_profile_id": execution_artifacts["slurm_profile_id"],
        "next_command": execution_artifacts["next_command"],
    }


def prepare_standard_analysis_memory_fit(
    *,
    pdb_path: Path,
    psf_path: Optional[Path],
    trajectories: Sequence[Path],
    output_directory: Path,
    project_id: str,
    frame_interval_ps: float,
    first_frame_time_ps: float = 0.0,
    temperature_kelvin: float = 300.0,
    target_wall_hours: Optional[float] = None,
    dssp_executable: Optional[str] = None,
    dssr_executable: Optional[str] = None,
    config_path: Optional[Path] = None,
    generate_connectivity_openmm: bool = False,
    openmm_bond_definitions: Sequence[Path] = (),
    energetic_charmm_parameter_files: Sequence[Path] = (),
    energetic_openmm_system_xml: Optional[Path] = None,
    energetic_gromacs_tpr: Optional[Path] = None,
) -> Dict[str, object]:
    """Prepare, explicitly reduce memory-incompatible modules, and replan.

    This is intentionally separate from the default fail-closed entry point.
    It performs a disposable planning pass, preserves the fully resolved user
    request, and only then writes a final runnable directory from an explicit
    reduced config.
    """

    destination = output_directory.expanduser().resolve(strict=False)
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise QuickstartError(
            f"output directory is not empty: {destination}; choose a new "
            "versioned directory"
        )
    common = {
        "pdb_path": pdb_path,
        "psf_path": psf_path,
        "trajectories": trajectories,
        "project_id": project_id,
        "frame_interval_ps": frame_interval_ps,
        "first_frame_time_ps": first_frame_time_ps,
        "temperature_kelvin": temperature_kelvin,
        "dssp_executable": dssp_executable,
        "dssr_executable": dssr_executable,
        "generate_connectivity_openmm": generate_connectivity_openmm,
        "openmm_bond_definitions": openmm_bond_definitions,
        "energetic_charmm_parameter_files": energetic_charmm_parameter_files,
        "energetic_openmm_system_xml": energetic_openmm_system_xml,
        "energetic_gromacs_tpr": energetic_gromacs_tpr,
    }
    with tempfile.TemporaryDirectory(
        prefix="salsbury-memory-fit-planning-"
    ) as temporary:
        temporary_root = Path(temporary)
        probe_output = temporary_root / "requested-plan"
        try:
            probe = prepare_standard_analysis(
                **common,
                output_directory=probe_output,
                target_wall_hours=target_wall_hours,
                config_path=config_path,
            )
            requested_config = load_json(probe_output / "analysis-config.json")
            active_config = deepcopy(requested_config)
            initial_plan = load_json(
                probe_output / "campaign-resource-plan.json"
            )
            requested_plan = deepcopy(initial_plan)
            direct_disabled: list[str] = []
            transitive_disabled: list[str] = []
        except QuickstartMemoryError as exc:
            requested_config = deepcopy(exc.analysis_config)
            active_config = deepcopy(exc.analysis_config)
            initial_plan = deepcopy(exc.plan)
            requested_plan = deepcopy(exc.plan)
            direct_disabled = []
            transitive_disabled = []
            for iteration in range(1, len(requested_config["modules"]) + 3):
                memory = initial_plan.get("memory_feasibility")
                if not isinstance(memory, Mapping):
                    raise QuickstartError(
                        "memory fallback received no memory feasibility report"
                    )
                modules_to_disable = memory.get(
                    "configuration_switches_to_disable_to_fit_configured_memory",
                    memory.get("modules_to_disable_to_fit_configured_memory", []),
                )
                if not isinstance(modules_to_disable, list) or not modules_to_disable:
                    raise QuickstartError(
                        "memory fallback could not identify a configurable module"
                    )
                active_config, direct, transitive = make_memory_fit_config(
                    active_config,
                    [str(value) for value in modules_to_disable],
                )
                direct_disabled.extend(
                    value for value in direct if value not in direct_disabled
                )
                transitive_disabled.extend(
                    value for value in transitive
                    if value not in transitive_disabled
                    and value not in direct_disabled
                )
                reduced_path = temporary_root / f"memory-fit-{iteration}.json"
                _json_write(reduced_path, active_config)
                retry_output = temporary_root / f"replanned-{iteration}"
                try:
                    retry = prepare_standard_analysis(
                        **common,
                        output_directory=retry_output,
                        target_wall_hours=None,
                        config_path=reduced_path,
                    )
                    probe = retry
                    probe_output = retry_output
                    break
                except QuickstartMemoryError as retry_exc:
                    initial_plan = deepcopy(retry_exc.plan)
            else:
                raise QuickstartError(
                    "memory fallback did not converge after disabling every "
                    "reported oversized module"
                )

        final_config_path = temporary_root / "resolved-memory-fit-config.json"
        _json_write(final_config_path, active_config)
        report = prepare_standard_analysis(
            **common,
            output_directory=destination,
            target_wall_hours=None,
            config_path=final_config_path,
        )
        final_plan = load_json(destination / "campaign-resource-plan.json")
        final_memory = final_plan.get("memory_feasibility", {})
        initial_memory = requested_plan.get("memory_feasibility", {})
        fallback_report = {
            "report_schema": "salsbury-memory-feasibility-report-v1",
            "technical_status": "complete",
            "planning_status": "replanned_with_explicit_reduced_config",
            "automatic_changes_applied": bool(
                direct_disabled or transitive_disabled
            ),
            "requested_memory": deepcopy(initial_memory),
            "final_memory": deepcopy(final_memory),
            "directly_disabled_modules_or_features": sorted(direct_disabled),
            "directly_disabled_configuration_switches": sorted(
                direct_disabled
            ),
            "transitively_disabled_modules": sorted(transitive_disabled),
            "requested_config": "analysis-config.requested.json",
            "resolved_config": "analysis-config.memory-fit.json",
            "original_request_preserved": True,
        }
        _json_write(
            destination / "analysis-config.requested.json", requested_config
        )
        _json_write(
            destination / "analysis-config.memory-fit.json", active_config
        )
        _json_write(
            destination / "memory-feasibility-report.json", fallback_report
        )
        generated = report.get("generated_files")
        if isinstance(generated, list):
            generated.extend([
                "analysis-config.requested.json",
                "analysis-config.memory-fit.json",
                "memory-feasibility-report.json",
            ])
        report["memory_fit"] = fallback_report
        return report
