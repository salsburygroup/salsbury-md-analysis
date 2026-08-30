"""Self-service preparation of shared-basis analyses for multiple systems."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import sys
import tempfile
from typing import Dict, Mapping, Optional, Sequence

from .analysis_config import (
    AnalysisConfigError,
    apply_module_configuration,
    load_analysis_config,
    make_memory_fit_config,
    make_resource_fit_config,
)
from .atom_mapping import AtomMappingError, read_pdb_atoms
from .automatic_sampling import automatic_sampling_plan
from .automatic_chemistry import (
    AutomaticChemistryError,
    infer_standard_chemistry_definitions,
)
from .campaign_planning import (
    CampaignPlanningError,
    plan_and_apply_complete_campaign,
)
from .conformational_views import plan_comparative_conformational_views
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .execution_adapters import (
    ExecutionAdapterError, _active_python_executable, prepare_execution_artifacts,
)
from .energetic_network_embeddings import probe_energetic_parameter_source
from .manifests import (
    load_json, resolve_manifest_path, validate_project, validate_system,
)
from .nucleic_acid_structure import probe_dssr_reference_duplex
from .preflight import (
    FileProbeError,
    probe_connectivity,
    probe_topology,
    probe_trajectory,
)
from .planning_report import PlanningReportError, write_planning_report
from .quickstart import (
    QuickstartError,
    QuickstartMemoryError,
    QuickstartPlanningError,
    _composition,
    _apply_experimental_input_gates,
    _applicable_sampling_modules,
    _configure_coordinate_cache_views,
    _conformational_view_projects,
    _conformational_view_slurm_files,
    _coordinate_cache_enabled,
    _discover_dssr_executable,
    _discover_dssp_executable,
    _exclude_conformational_views_from_base_workflow,
    _experimental_planner_coverage,
    _effective_parallel_cpu_cap,
    _execution_config_for_parallel_cpu_cap,
    _force_field_parameter_spec,
    _generic_definitions,
    _json_write,
    _record_conformational_experimental_exclusions,
    _require_new_directory,
    _safe_id,
    _slurm_files,
    _validated_cache_export_shell,
    _validate_reference_connectivity,
)
from .coordinate_cache import validate_reusable_coordinate_cache
from .registry import list_modules


_ALLOWED_SYSTEM_FIELDS = {
    "system_id", "pdb", "psf", "connectivity", "trajectories",
    "frame_interval_ps", "first_frame_time_ps", "replicas",
    "force_field_parameters",
}
_ALLOWED_REPLICA_FIELDS = {"replica_id", "segments"}
_ALLOWED_SEGMENT_FIELDS = {
    "segment_id", "trajectory", "frame_interval_ps", "first_frame_time_ps",
    "continuous_with_previous",
}


def _positive_finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise QuickstartError(f"{label} must be finite and positive")
    return float(value)


def _finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise QuickstartError(f"{label} must be finite")
    return float(value)


def _deferred_modules(exclusions: Dict[str, str], disabled: Dict[str, str]) -> Dict[str, str]:
    return {
        **exclusions,
        **disabled,
        "trajectory_features": "requires a declared scientific feature rather than an arbitrary atom pair",
        "scalar_feature_distributions": "runs after a question-linked scalar feature is declared",
        "scalar_threshold_states": "requires a physically justified threshold and sensitivity range",
        "hydrogen_bonds": (
            "optional manual fixed-feature override; automatic chemistry-backed "
            "donor-hydrogen-acceptor discovery is the production default"
        ),
        "hydrogen_bond_comparison": "runs after chemically mapped condition reports are complete",
        "hydrogen_bond_patterns": "runs after direct-hydrogen-bond reports define frame patterns",
        "grouped_regularized_classification": "runs after cross-condition hydrogen-bond features are accepted",
        "state_coordinate_exports": "automatic for non-trace FES views; extra exports require an accepted state definition",
        "representative_structures": (
            "optional coordinate-space mean/medoid utility; observed state-centered "
            "representative frames and coordinate exports are automatic"
        ),
        "optional_observables": "requires a residue- or question-specific definition",
        "radial_distribution_functions": "requires explicit chemically meaningful atom groups",
        "nucleic_acid_geometry": "automatic ring and stacking definitions are not yet enabled in the generic initializer",
        "ion_coordination_geometry": "automatic bound-ion ligand-shell definitions are not yet enabled in the generic initializer",
        "ion_atmosphere": "requires supported ions and polar non-solvent solute atoms; automatic inference is configurable",
        "nucleic_acid_structure": "requires a separately licensed x3dna-dssr executable",
        "helical_mechanics": "requires an enabled module and a per-system DSSR duplex-availability probe",
        "rmsf_permutation_inference": "requires a declared exchangeable-unit comparison",
        "integrated_comparison": "runs after accepted upstream reports are selected",
    }


_AUTOMATIC_CONTEXT_COMMANDS = {
    "trajectory_features": "trajectory-features",
    "optional_observables": "observables",
    "radial_distribution_functions": "rdf",
    "scalar_feature_distributions": "scalar-distributions",
    "scalar_threshold_states": "scalar-threshold-states",
    "nucleic_acid_structure": "nucleic-acid-structure",
    "nucleic_acid_geometry": "nucleic-acid-geometry",
    "ion_coordination_geometry": "ion-geometry",
    "ion_atmosphere": "ion-atmosphere",
    "helical_mechanics": "helical-mechanics",
}

_AUTOMATIC_CONTEXT_STAGES = {
    "trajectory-features": 0,
    "observables": 0,
    "rdf": 0,
    "nucleic-acid-structure": 0,
    "nucleic-acid-geometry": 0,
    "ion-geometry": 0,
    "ion-atmosphere": 0,
    "helical-mechanics": 1,
    "scalar-distributions": 1,
    "scalar-threshold-states": 1,
}


def _automatic_context_project(
    *,
    root: Path,
    system_id: str,
    system_manifest_filename: str,
    base_project: Mapping[str, object],
    reference_structure: Path,
    reference_connectivity: Path,
    source_frame_counts: Sequence[int],
    composition: Mapping[str, object],
    analysis_config: Mapping[str, object],
    dssr_executable: Optional[str],
) -> tuple[Optional[str], Dict[str, object], Dict[str, str], list[str]]:
    """Create one topology-local, outcome-independent chemistry project."""

    inference_config = analysis_config["inference"]
    assert isinstance(inference_config, dict)
    if not bool(inference_config["automatic_chemical_context"]):
        return None, {}, {
            module_id: "automatic chemical-context inference disabled by analysis config"
            for module_id in _AUTOMATIC_CONTEXT_COMMANDS
        }, []
    total_frames = sum(int(value) for value in source_frame_counts)
    try:
        inferred = infer_standard_chemistry_definitions(
            reference_structure,
            maximum_frames_by_module={
                module_id: total_frames for module_id in _AUTOMATIC_CONTEXT_COMMANDS
            },
            total_source_frames=total_frames,
            dssr_executable=dssr_executable,
            ion_site_classification_enabled=bool(
                inference_config["ion_site_classification_enabled"]
            ),
        )
    except (AutomaticChemistryError, OSError) as exc:
        raise QuickstartError(
            f"automatic chemical-context inference failed for {system_id}: {exc}"
        ) from exc
    raw_definitions = inferred["definitions"]
    applicable = inferred["applicable_modules"]
    not_applicable = inferred["not_applicable_modules"]
    assert isinstance(raw_definitions, dict)
    assert isinstance(applicable, list)
    assert isinstance(not_applicable, dict)
    config_modules = analysis_config["modules"]
    assert isinstance(config_modules, dict)
    helical_config = config_modules.get("helical_mechanics")
    helical_requested = (
        isinstance(helical_config, dict)
        and helical_config.get("enabled") is True
    )
    helical_probe: Dict[str, object] = {
        "status": "not_available", "reason": "helical_mechanics_disabled",
        "executable": dssr_executable,
    }
    if helical_requested:
        if not bool(composition.get("has_nucleic_acid")):
            helical_probe = {
                "status": "not_available", "reason": "no_duplex_dna_or_rna",
                "executable": dssr_executable,
            }
        elif dssr_executable is None:
            helical_probe = {
                "status": "not_available", "reason": "dssr_not_installed",
                "executable": None,
            }
        else:
            helical_probe = probe_dssr_reference_duplex(
                dssr_executable, reference_structure
            )
        source_definition = raw_definitions.get("nucleic_acid_structure")
        base_definitions = base_project.get("definitions")
        helical_definition = (
            base_definitions.get("helical_mechanics")
            if isinstance(base_definitions, dict) else None
        )
        parameters = helical_probe.get("helical_step_parameters")
        if (
            helical_probe.get("status") == "available"
            and isinstance(source_definition, dict)
            and isinstance(helical_definition, dict)
            and isinstance(parameters, dict)
            and isinstance(parameters.get("object_path"), list)
            and isinstance(parameters.get("fields"), dict)
        ):
            object_path = list(parameters["object_path"])
            fields = parameters["fields"]
            source_definition["numeric_queries"] = [{
                "query_id": f"helical-step-{name}",
                "path": [*object_path, str(fields[name])],
                "missing_policy": "fail",
            } for name in ("shift", "slide", "rise", "tilt", "roll", "twist")]
            configured_helical = deepcopy(helical_definition)
            configured_helical["preparation_availability"] = deepcopy(helical_probe)
            raw_definitions["helical_mechanics"] = configured_helical
            if "helical_mechanics" not in applicable:
                applicable.append("helical_mechanics")
        else:
            reason = str(
                helical_probe.get("reason")
                or "nucleic_acid_structure_source_not_configured"
            )
            not_applicable["helical_mechanics"] = reason
        inferred["dssr_duplex_probe"] = deepcopy(helical_probe)
    raw_commands = [
        _AUTOMATIC_CONTEXT_COMMANDS[module_id]
        for module_id in applicable
        if module_id in _AUTOMATIC_CONTEXT_COMMANDS
    ]
    try:
        definitions, commands, requested, disabled = apply_module_configuration(
            raw_definitions, raw_commands, applicable, analysis_config
        )
    except AnalysisConfigError as exc:
        raise QuickstartError(str(exc)) from exc
    inference_filename = f"automatic-chemical-context-{system_id}.json"
    generated = [inference_filename]
    if helical_requested and "helical_mechanics" not in requested:
        reason = str(
            helical_probe.get("reason")
            or disabled.get(
                "helical_mechanics", "nucleic_acid_structure_source_not_configured"
            )
        )
        not_applicable["helical_mechanics"] = reason
        availability_filename = f"helical-mechanics-availability-{system_id}.json"
        _json_write(root / availability_filename, {
            "module_id": "helical_mechanics",
            "system_id": system_id,
            "technical_status": "complete",
            "scientific_status": "not evaluated",
            "availability_status": "not_available",
            "availability_reason": reason,
            "planner_task_created": False,
            "preparation_probe": helical_probe,
        })
        generated.append(availability_filename)
    _json_write(root / inference_filename, inferred)
    if not commands:
        return None, inferred, {**not_applicable, **disabled}, generated
    project = deepcopy(dict(base_project))
    project.update({
        "project_id": f"{base_project['project_id']}-{system_id}-chemical-context",
        "system_manifest": system_manifest_filename,
        "analysis_output_root": f"results/per-system/{system_id}/chemical-context",
        "reference_system": system_id,
        "reference_structure": str(reference_structure),
        "reference_connectivity": str(reference_connectivity),
        "common_atom_policy": "strict",
        "definitions": definitions,
        "requested_modules": requested,
        "protected_locations": [
            str(reference_structure), str(reference_connectivity),
        ],
    })
    filename = f"project-chemical_{system_id}.json"
    path = root / filename
    _json_write(path, project)
    validate_project(project, source_path=path, check_paths=True)
    return filename, inferred, {**not_applicable, **disabled}, [*generated, filename]


def _automatic_context_slurm_files(
    root: Path,
    project_id: str,
    context_projects: Sequence[Mapping[str, object]],
    *,
    target_wall_hours: float,
    python_executable: str,
    package_root: str,
) -> tuple[list[str], Dict[int, int]]:
    """Write scheduler tasks for the generated per-system chemistry projects."""

    by_stage: Dict[int, list[tuple[str, str, str]]] = {}
    for spec in context_projects:
        project_filename = str(spec["project_filename"])
        project = load_json(root / project_filename)
        output_root = root / str(project["analysis_output_root"])
        for command in spec["commands"]:
            command_text = str(command)
            stage = _AUTOMATIC_CONTEXT_STAGES[command_text]
            by_stage.setdefault(stage, []).append(
                (project_filename, command_text, str(output_root / command_text))
            )
    generated: list[str] = []
    stage_counts: Dict[int, int] = {}
    # The campaign limit is a hard per-job ceiling.  Preparation and reporting
    # overhead must not silently extend automatic-context jobs past it.
    wall_minutes = int(math.ceil(target_wall_hours * 60.0))
    wall_limit = f"{wall_minutes // 60:02d}:{wall_minutes % 60:02d}:00"
    for stage, rows in sorted(by_stage.items()):
        if not rows:
            continue
        projects = "\n".join(f"  {value!r}" for value, _, _ in rows)
        commands = "\n".join(f"  {value!r}" for _, value, _ in rows)
        outputs = "\n".join(f"  {value!r}" for _, _, value in rows)
        filename = f"run_automatic_context_stage_{stage}_array.slurm"
        worker = f"""#!/usr/bin/env bash
#SBATCH --job-name=sma-{project_id[:14]}-chem-{stage}
#SBATCH --time={wall_limit}
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output={root}/logs/%A_%a-context-stage-{stage}.out
#SBATCH --error={root}/logs/%A_%a-context-stage-{stage}.err
set -euo pipefail
ROOT={str(root)!r}
PROJECTS=(
{projects}
)
COMMANDS=(
{commands}
)
OUTPUTS=(
{outputs}
)
PROJECT="$ROOT/${{PROJECTS[$SLURM_ARRAY_TASK_ID]}}"
COMMAND="${{COMMANDS[$SLURM_ARRAY_TASK_ID]}}"
OUTPUT="${{OUTPUTS[$SLURM_ARRAY_TASK_ID]}}"
PYTHON_DEFAULT={python_executable!r}
PYTHON="${{SALSBURY_MD_ANALYSIS_PYTHON:-$PYTHON_DEFAULT}}"
PACKAGE_ROOT_DEFAULT={package_root!r}
PACKAGE_ROOT="${{SALSBURY_MD_ANALYSIS_PYTHONPATH:-$PACKAGE_ROOT_DEFAULT}}"
export PYTHONPATH="$PACKAGE_ROOT${{PYTHONPATH:+:$PYTHONPATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
if [[ "$COMMAND" == "scalar-distributions" || "$COMMAND" == "scalar-threshold-states" ]]; then
  {_validated_cache_export_shell(
      "SALSBURY_MD_ANALYSIS_TRAJECTORY_FEATURES_REPORT", None,
      "trajectory_features",
      report_shell_expression='"${OUTPUT%/*}/trajectory-features/report.json"',
  )}
fi
if [[ "$COMMAND" == "helical-mechanics" ]]; then
  {_validated_cache_export_shell(
      "SALSBURY_MD_ANALYSIS_NUCLEIC_ACID_STRUCTURE_REPORT", None,
      "nucleic_acid_structure",
      report_shell_expression='"${OUTPUT%/*}/nucleic-acid-structure/report.json"',
      fallback_action="continuing without that prerequisite report",
  )}
fi
mkdir -p "$OUTPUT" "$ROOT/logs"
FINAL="$OUTPUT/report.json"
SUMMARY="$FINAL.summary.json"
if [[ -e "$FINAL" ]]; then
  "$PYTHON" - "$FINAL" "$SUMMARY" <<'PY'
import hashlib, json, sys
report_path, summary_path = sys.argv[1:]
summary = json.load(open(summary_path, encoding='utf-8'))
digest = hashlib.sha256(open(report_path, 'rb').read()).hexdigest()
if summary.get('technical_status') != 'complete' or summary.get('report_sha256') != digest:
    raise SystemExit('existing automatic-context report is incomplete or hash-mismatched')
PY
  exit 0
fi
TMP="$FINAL.tmp.$SLURM_JOB_ID"
SUMMARY_TMP="$TMP.summary.json"
"$PYTHON" -m salsbury_md_analysis run-instrumented "$COMMAND" "$PROJECT" \
  --hash-content --summary-sidecar "$SUMMARY_TMP" \
  --installed-report-path "$FINAL" > "$TMP"
"$PYTHON" - "$TMP" "$SUMMARY_TMP" <<'PY'
import hashlib, json, sys
report_path, summary_path = sys.argv[1:]
summary = json.load(open(summary_path, encoding='utf-8'))
digest = hashlib.sha256(open(report_path, 'rb').read()).hexdigest()
if summary.get('technical_status') != 'complete' or summary.get('report_sha256') != digest:
    raise SystemExit('automatic-context report is incomplete or hash-mismatched')
PY
ln "$SUMMARY_TMP" "$SUMMARY"
ln "$TMP" "$FINAL"
rm "$TMP" "$SUMMARY_TMP"
"""
        (root / filename).write_text(worker, encoding="utf-8")
        generated.append(filename)
        stage_counts[stage] = len(rows)
    return generated, stage_counts


def prepare_comparative_analysis(
    *,
    request_path: Path,
    output_directory: Path,
    project_id: str,
    temperature_kelvin: float = 300.0,
    target_wall_hours: Optional[float] = None,
    dssp_executable: Optional[str] = None,
    dssr_executable: Optional[str] = None,
    config_path: Optional[Path] = None,
    protected_core_only: bool = False,
    uniform_cache_stride: bool = False,
) -> Dict[str, object]:
    """Prepare one shared-basis, common-grid workflow for two or more systems."""

    project_id = _safe_id(project_id, "project_id")
    temperature = _positive_finite(temperature_kelvin, "temperature_kelvin")
    target_hours = (
        _positive_finite(target_wall_hours, "target_wall_hours")
        if target_wall_hours is not None else None
    )
    source = Path(request_path).expanduser().resolve(strict=True)
    request = load_json(source)
    if not isinstance(request, dict):
        raise QuickstartError("comparative input request must be a JSON object")
    unknown_top = sorted(set(request).difference({"request_schema", "systems"}))
    if unknown_top:
        raise QuickstartError(
            "comparative input request has unknown fields: " + ", ".join(unknown_top)
        )
    request_schema = request.get("request_schema")
    if request_schema not in {
        "salsbury-comparative-analysis-input-v1",
        "salsbury-comparative-analysis-input-v2",
    }:
        raise QuickstartError(
            "request_schema must be salsbury-comparative-analysis-input-v1 or "
            "salsbury-comparative-analysis-input-v2"
        )
    raw_systems = request.get("systems")
    if not isinstance(raw_systems, list) or len(raw_systems) < 2:
        raise QuickstartError("comparative analysis requires at least two systems")

    system_rows = []
    system_ids = []
    frame_counts = []
    protected_paths = []
    compositions = []
    references = []
    system_inputs = []
    connectivity_checks: Dict[str, object] = {}
    for raw in raw_systems:
        if not isinstance(raw, dict):
            raise QuickstartError("every comparative system must be an object")
        unknown = sorted(set(raw).difference(_ALLOWED_SYSTEM_FIELDS))
        if unknown:
            raise QuickstartError(
                "comparative system has unknown fields: " + ", ".join(unknown)
            )
        system_id = _safe_id(str(raw.get("system_id", "")), "system_id")
        system_ids.append(system_id)
        pdb = Path(str(raw.get("pdb", ""))).expanduser().resolve(strict=True)
        raw_psf = raw.get("psf")
        raw_connectivity = raw.get("connectivity")
        if (raw_psf is None) == (raw_connectivity is None):
            raise QuickstartError(
                f"{system_id} requires exactly one of psf or connectivity"
            )
        connectivity = Path(
            str(raw_psf if raw_psf is not None else raw_connectivity)
        ).expanduser().resolve(strict=True)
        raw_force_field_parameters = raw.get("force_field_parameters")
        force_field_parameters = None
        if raw_force_field_parameters is not None:
            if (
                not isinstance(raw_force_field_parameters, dict)
                or set(raw_force_field_parameters).difference({"format", "files"})
                or not isinstance(raw_force_field_parameters.get("format"), str)
                or not isinstance(raw_force_field_parameters.get("files"), list)
                or not raw_force_field_parameters["files"]
            ):
                raise QuickstartError(
                    f"{system_id}.force_field_parameters requires format and files"
                )
            parameter_paths = [
                resolve_manifest_path(str(value), source)
                for value in raw_force_field_parameters["files"]
            ]
            format_name = str(raw_force_field_parameters["format"])
            if format_name == "charmm_parameter_files_v1":
                force_field_parameters = _force_field_parameter_spec(
                    charmm_parameter_files=parameter_paths,
                )
            elif format_name == "openmm_system_xml_v1":
                if len(parameter_paths) != 1:
                    raise QuickstartError(
                        f"{system_id} OpenMM System XML requires exactly one file"
                    )
                force_field_parameters = _force_field_parameter_spec(
                    openmm_system_xml=parameter_paths[0],
                )
            elif format_name == "gromacs_tpr_v1":
                if len(parameter_paths) != 1:
                    raise QuickstartError(
                        f"{system_id} GROMACS TPR requires exactly one file"
                    )
                force_field_parameters = _force_field_parameter_spec(
                    gromacs_tpr=parameter_paths[0],
                )
            else:
                raise QuickstartError(
                    f"{system_id}.force_field_parameters has unsupported format "
                    f"{format_name!r}"
                )
        if pdb.suffix.lower() not in {".pdb", ".ent"}:
            raise QuickstartError(f"{system_id}.pdb must name a PDB file")
        if raw_psf is not None and connectivity.suffix.lower() != ".psf":
            raise QuickstartError(f"{system_id}.psf must name a PSF file")
        try:
            pdb_probe = probe_topology(pdb)
            connectivity_probe = probe_connectivity(connectivity)
        except (FileProbeError, OSError) as exc:
            raise QuickstartError(str(exc)) from exc
        atom_count = int(pdb_probe["atom_count"])
        if int(connectivity_probe["atom_count"]) != atom_count:
            raise QuickstartError(
                f"{system_id} PDB and connectivity atom counts differ"
            )
        connectivity_checks[system_id] = _validate_reference_connectivity(
            pdb, connectivity, atom_count
        )
        replicas = []
        trajectories = []
        system_frame_counts = []
        raw_replicas = raw.get("replicas")
        trajectory_values = raw.get("trajectories")
        if request_schema == "salsbury-comparative-analysis-input-v1":
            if raw_replicas is not None:
                raise QuickstartError(
                    f"{system_id}.replicas requires comparative input schema v2"
                )
            interval = _positive_finite(
                raw.get("frame_interval_ps"), f"{system_id}.frame_interval_ps"
            )
            first = _finite(
                raw.get("first_frame_time_ps", 0.0),
                f"{system_id}.first_frame_time_ps",
            )
            if not isinstance(trajectory_values, list) or not trajectory_values:
                raise QuickstartError(
                    f"{system_id}.trajectories must be a nonempty list"
                )
            raw_replicas = [
                {
                    "replica_id": f"replica-{index}",
                    "segments": [{
                        "segment_id": "production",
                        "trajectory": value,
                        "first_frame_time_ps": first,
                        "frame_interval_ps": interval,
                        "continuous_with_previous": False,
                    }],
                }
                for index, value in enumerate(trajectory_values, start=1)
            ]
        else:
            forbidden = [
                field for field in (
                    "trajectories", "frame_interval_ps", "first_frame_time_ps"
                )
                if field in raw
            ]
            if forbidden:
                raise QuickstartError(
                    f"{system_id} schema-v2 systems use replicas/segments, not: "
                    + ", ".join(forbidden)
                )
            if not isinstance(raw_replicas, list) or not raw_replicas:
                raise QuickstartError(
                    f"{system_id}.replicas must be a nonempty list under schema v2"
                )
        assert isinstance(raw_replicas, list)
        replica_ids = []
        for replica_index, raw_replica in enumerate(raw_replicas):
            if not isinstance(raw_replica, dict):
                raise QuickstartError(
                    f"{system_id}.replicas[{replica_index}] must be an object"
                )
            unknown_replica = sorted(
                set(raw_replica).difference(_ALLOWED_REPLICA_FIELDS)
            )
            if unknown_replica:
                raise QuickstartError(
                    f"{system_id}.replicas[{replica_index}] has unknown fields: "
                    + ", ".join(unknown_replica)
                )
            replica_id = _safe_id(
                str(raw_replica.get("replica_id", "")), "replica_id"
            )
            if replica_id in replica_ids:
                raise QuickstartError(
                    f"{system_id} contains duplicate replica_id {replica_id!r}"
                )
            replica_ids.append(replica_id)
            raw_segments = raw_replica.get("segments")
            if not isinstance(raw_segments, list) or not raw_segments:
                raise QuickstartError(
                    f"{system_id}.{replica_id}.segments must be a nonempty list"
                )
            segments = []
            segment_ids = []
            replica_frame_count = 0
            for segment_index, raw_segment in enumerate(raw_segments):
                if not isinstance(raw_segment, dict):
                    raise QuickstartError(
                        f"{system_id}.{replica_id}.segments[{segment_index}] "
                        "must be an object"
                    )
                unknown_segment = sorted(
                    set(raw_segment).difference(_ALLOWED_SEGMENT_FIELDS)
                )
                if unknown_segment:
                    raise QuickstartError(
                        f"{system_id}.{replica_id}.segments[{segment_index}] has "
                        "unknown fields: " + ", ".join(unknown_segment)
                    )
                segment_id = _safe_id(
                    str(raw_segment.get("segment_id", "")), "segment_id"
                )
                if segment_id in segment_ids:
                    raise QuickstartError(
                        f"{system_id}.{replica_id} contains duplicate segment_id "
                        f"{segment_id!r}"
                    )
                segment_ids.append(segment_id)
                trajectory = Path(
                    str(raw_segment.get("trajectory", ""))
                ).expanduser().resolve(strict=True)
                if trajectory.suffix.lower() != ".dcd":
                    raise QuickstartError(
                        f"{system_id}.{replica_id}.{segment_id}.trajectory must "
                        "name a DCD file"
                    )
                interval = _positive_finite(
                    raw_segment.get("frame_interval_ps"),
                    f"{system_id}.{replica_id}.{segment_id}.frame_interval_ps",
                )
                first = _finite(
                    raw_segment.get("first_frame_time_ps", 0.0),
                    f"{system_id}.{replica_id}.{segment_id}.first_frame_time_ps",
                )
                continuous = raw_segment.get("continuous_with_previous", False)
                if not isinstance(continuous, bool):
                    raise QuickstartError(
                        f"{system_id}.{replica_id}.{segment_id}."
                        "continuous_with_previous must be boolean"
                    )
                if segment_index == 0 and continuous:
                    raise QuickstartError(
                        f"{system_id}.{replica_id}.{segment_id} is the first segment "
                        "and cannot be continuous_with_previous"
                    )
                try:
                    probe = probe_trajectory(trajectory)
                except (FileProbeError, OSError) as exc:
                    raise QuickstartError(str(exc)) from exc
                if int(probe["atom_count"]) != atom_count:
                    raise QuickstartError(
                        f"{system_id} trajectory atom count differs from PDB/PSF: "
                        f"{trajectory}"
                    )
                count = int(probe["declared_frame_count"])
                if count < 1:
                    raise QuickstartError(
                        f"{system_id} trajectory declares no frames: {trajectory}"
                    )
                trajectories.append(trajectory)
                replica_frame_count += count
                segment = {
                    "segment_id": segment_id,
                    "trajectory": str(trajectory),
                    "timing": {
                        "first_frame_time": first,
                        "frame_interval": interval,
                        "unit": "ps",
                    },
                }
                if segment_index > 0:
                    segment["continuous_with_previous"] = continuous
                segments.append(segment)
            if replica_frame_count < 1:
                raise QuickstartError(
                    f"{system_id}.{replica_id} must contain at least one frame"
                )
            frame_counts.append(replica_frame_count)
            system_frame_counts.append(replica_frame_count)
            replica_row = {
                "replica_id": replica_id,
                "topology": str(pdb),
                "connectivity": str(connectivity),
                "segments": segments,
            }
            if force_field_parameters is not None:
                replica_row["force_field_parameters"] = deepcopy(
                    force_field_parameters
                )
            replicas.append(replica_row)
        composition = _composition(pdb)
        try:
            atoms = read_pdb_atoms(pdb)
            coordinates = next(
                iter_coordinate_frames(pdb, "angstrom")
            ).coordinates_angstrom
        except (AtomMappingError, CoordinateReadError, StopIteration) as exc:
            raise QuickstartError(
                f"could not read comparative reference {system_id}: {exc}"
            ) from exc
        compositions.append(composition)
        references.append((system_id, atoms, coordinates))
        parameter_paths = (
            [Path(str(value)) for value in force_field_parameters["files"]]
            if force_field_parameters is not None else []
        )
        protected_paths.extend([
            pdb, connectivity, *parameter_paths, *trajectories,
        ])
        system_rows.append({
            "system_id": system_id,
            "metadata": {
                "prepared_by": "salsbury-md-analysis prepare-comparison",
                "timing_source": "request-declared frame_interval_ps",
            },
            "replicas": replicas,
        })
        system_inputs.append({
            "system_id": system_id,
            "pdb": pdb,
            "connectivity": connectivity,
            "force_field_parameters": force_field_parameters,
            "trajectories": trajectories,
            "frame_counts": system_frame_counts,
            "composition": composition,
        })
    if len(set(system_ids)) != len(system_ids):
        raise QuickstartError("comparative system_id values must be unique")
    for system_id, system_input in zip(system_ids, system_inputs):
        if sum(int(value) for value in system_input["frame_counts"]) < 2:
            raise QuickstartError(
                f"{system_id} requires at least two pooled physical frames"
            )

    try:
        comparative_plan = plan_comparative_conformational_views(
            references, common_atom_policy="position"
        )
    except (ValueError, AtomMappingError) as exc:
        raise QuickstartError(
            "could not construct outcome-independent comparative views: " + str(exc)
        ) from exc
    composition = deepcopy(compositions[0])
    composition["conformational_view_plan"] = comparative_plan
    composition["atom_count"] = max(int(row["atom_count"]) for row in compositions)
    composition["residue_count"] = max(int(row["residue_count"]) for row in compositions)
    composition["has_protein"] = any(bool(row["has_protein"]) for row in compositions)
    composition["has_nucleic_acid"] = any(
        bool(row["has_nucleic_acid"]) for row in compositions
    )
    common_residue_names = {
        str(name).upper() for name in compositions[0].get("residue_names", [])
    }
    for row in compositions[1:]:
        common_residue_names.intersection_update(
            str(name).upper() for name in row.get("residue_names", [])
        )
    composition["residue_names"] = sorted(common_residue_names)
    water_counts = [int(row["water_residue_count"]) for row in compositions]
    composition["water_residue_count"] = max(water_counts) if all(water_counts) else 0
    composition["ion_atom_indices"] = (
        [0] if all(bool(row.get("ion_atom_indices")) for row in compositions) else []
    )
    composition["solute_heavy_atom_count"] = max(
        int(row["solute_heavy_atom_count"]) for row in compositions
    )
    selections = deepcopy(composition["selections"])
    assert isinstance(selections, dict)
    for view in comparative_plan["views"]:
        if isinstance(view, dict) and isinstance(view.get("selection"), dict):
            selections[str(view["selection_id"])] = deepcopy(view["selection"])
    trace_view = next(
        row for row in comparative_plan["views"]
        if isinstance(row, dict) and row.get("view_id") == "macromolecular_trace"
    )
    composition["trace_atom_count"] = int(trace_view["common_atom_count"])
    composition["selections"] = selections

    root = _require_new_directory(output_directory)
    system = {"systems": system_rows}
    system_path = root / "system.json"
    _json_write(system_path, system)
    validate_system(system, source_path=system_path, check_paths=True)
    registry_ids = [module.module_id for module in list_modules()]
    view_ids_for_config = {
        str(view["view_id"]) for view in comparative_plan["views"]
        if isinstance(view, dict)
    }
    for system_composition in compositions:
        system_view_plan = system_composition["conformational_view_plan"]
        assert isinstance(system_view_plan, dict)
        for view in system_view_plan["views"]:
            if isinstance(view, dict):
                view_ids_for_config.add(str(view["view_id"]))
    try:
        analysis_config = load_analysis_config(
            config_path, registry_ids, sorted(view_ids_for_config),
            additional_protected_modules=("integrated_comparison",),
            module_selection_override=(
                "protected_core_only" if protected_core_only else None
            ),
            stride_mode_override=(
                "uniform_cache_stride" if uniform_cache_stride else None
            ),
        )
    except (AnalysisConfigError, OSError) as exc:
        raise QuickstartError(str(exc)) from exc
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
    if target_hours is not None:
        execution = analysis_config["execution"]
        assert isinstance(execution, dict)
        execution["maximum_hours_per_cpu"] = target_hours
        execution["maximum_total_cpu_hours"] = (
            int(execution["maximum_parallel_cpus"]) * target_hours
        )
    comparison_config = analysis_config["comparisons"]
    assert isinstance(comparison_config, dict)
    if (
        comparison_config.get("mode") == "reference_vs_all"
        and comparison_config.get("reference_system_id") not in system_ids
    ):
        raise QuickstartError(
            "comparisons.reference_system_id must name one declared comparative system"
        )
    dssp = _discover_dssp_executable(dssp_executable)
    dssr = _discover_dssr_executable(dssr_executable)
    energetic_parameter_details = []
    for inputs in system_inputs:
        probe = probe_energetic_parameter_source(
            Path(str(inputs["pdb"])),
            Path(str(inputs["connectivity"])),
            inputs.get("force_field_parameters"),
        )
        energetic_parameter_details.append({
            "system_id": str(inputs["system_id"]),
            **probe,
        })
    energetic_parameter_available = all(
        row.get("availability_status") == "available"
        for row in energetic_parameter_details
    )
    energetic_systems_have_protein = all(
        bool(row.get("has_protein")) for row in compositions
    )
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
        dssr_probe={
            "status": "not_available",
            "reason": "comparative_per_system_duplex_probe_required",
            "executable": dssr,
        },
    )
    requested = [
        "provenance_manifest", "preflight_inventory", "common_atom_mapping",
        "structural_integrity_qc", "replica_rmsd_rg", "pooled_rmsf", "dccm",
        "generalized_correlation_and_information", "information_dynamics",
        "perturbation_response_dynamics", "trajectory_reweighting",
        "correlation_networks", "allosteric_pathways",
        "energetic_network_embeddings", "multivalent_molecular_bridges",
        "hydration_density_channels", "ensemble_pocket_dynamics",
        "interaction_fingerprints", "spatial_interaction_ensembles",
        "interaction_persistence", "helical_mechanics",
        "individual_pca", "common_pca",
        "time_lagged_independent_component_analysis", "pca_fes_basins",
        "random_feature_koopman",
        "clustering_kmeans", "clustering_hdbscan", "clustering_imwkmeans",
        "alternative_clustering", "representative_frames", "markov_state_models",
        "reactive_path_ensembles",
        "grouped_ml", "dihedral_distributions", "hydrogen_bond_discovery",
        "solvent_accessible_surface_area", "convergence_uncertainty",
        "rmsf_permutation_inference", "integrated_comparison",
    ]
    if "water_mediated_hydrogen_bond_networks" in definitions:
        requested.append("water_mediated_hydrogen_bond_networks")
    if "secondary_structure" in definitions:
        requested.append("secondary_structure")
    try:
        definitions, commands, requested, disabled = apply_module_configuration(
            definitions, commands, requested, analysis_config
        )
    except AnalysisConfigError as exc:
        raise QuickstartError(str(exc)) from exc
    commands, requested = _apply_experimental_input_gates(
        definitions, commands, requested, exclusions, composition
    )
    # Helical mechanics is scheduled independently for every system whose
    # reference passes the duplex/DSSR applicability probe.
    commands = [command for command in commands if command != "helical-mechanics"]
    requested = [module for module in requested if module != "helical_mechanics"]
    energetic_availability_file: Optional[str] = None
    if (
        "energetic_network_embeddings" in requested
        and (
            not energetic_systems_have_protein
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
        if not energetic_systems_have_protein:
            protein_absent = [
                str(inputs["system_id"])
                for inputs in system_inputs
                if not bool(inputs["composition"].get("has_protein"))
            ]
            reason = (
                "not applicable: protein residues are absent from "
                + ", ".join(protein_absent)
            )
        else:
            unavailable = next(
                row for row in energetic_parameter_details
                if row.get("availability_status") != "available"
            )
            reason = "not available: " + str(
                unavailable.get("availability_reason", "no compatible parameters")
            )
        exclusions["energetic_network_embeddings"] = reason
        energetic_availability_file = (
            "energetic-network-embeddings-availability.json"
        )
        _json_write(root / energetic_availability_file, {
            "module_id": "energetic_network_embeddings",
            "technical_status": "complete",
            "scientific_status": "not evaluated",
            "availability_status": "not_available",
            "availability_reason": reason,
            "planner_task_created": False,
            "systems": energetic_parameter_details,
        })
    if "integrated_comparison" not in requested:
        raise QuickstartError(
            "integrated_comparison is mandatory for comparative campaigns and cannot be disabled"
        )
    commands, requested = _exclude_conformational_views_from_base_workflow(
        commands, requested
    )
    project = {
        "project_id": project_id,
        "analysis_profile": "standard_md_v1",
        "system_manifest": "system.json",
        "analysis_output_root": "results",
        "reference_system": system_ids[0],
        "temperature_kelvin": temperature,
        "sampling_mode": "UNBIASED_MD",
        "coordinate_unit": "angstrom",
        "time_unit": "ps",
        "periodic_coordinate_policy": "unwrap_continuous",
        "periodic_reconstruction": {
            "maximum_bond_length_angstrom": 4.0,
            "cycle_closure_tolerance_angstrom": 0.05,
            "maximum_anchor_displacement_angstrom": 100.0,
        },
        "reference_structure": str(protected_paths[0]),
        "reference_connectivity": str(protected_paths[1]),
        "common_atom_policy": "position",
        "selections": selections,
        "definitions": definitions,
        "requested_modules": requested,
        "protected_locations": list(dict.fromkeys(
            str(path) for path in protected_paths
        )),
    }
    project_path = root / "project.json"
    _json_write(project_path, project)
    validate_project(project, source_path=project_path, check_paths=True)
    _json_write(root / "sampling-plan.json", sampling_plan)
    _json_write(root / "analysis-config.json", analysis_config)
    view_ids: list[str] = []
    view_project_files: list[str] = []
    view_frame_counts_by_id: Dict[str, list[int]] = {}
    per_system_manifest_files: list[str] = []
    context_projects: list[Dict[str, object]] = []
    context_project_files: list[str] = []
    context_generated_files: list[str] = []
    context_frame_counts_by_id: Dict[str, list[int]] = {}
    context_not_applicable: Dict[str, list[str]] = {}
    if bool(comparison_config["run_shared_basis_comparisons"]):
        shared_ids, shared_files = _conformational_view_projects(
            root,
            project,
            composition,
            frame_counts_per_replica=frame_counts,
            analysis_config=analysis_config,
            workflow_scope="shared_basis_comparison",
        )
        view_ids.extend(shared_ids)
        view_project_files.extend(shared_files)
        for view_id in shared_ids:
            view_frame_counts_by_id[view_id] = list(frame_counts)
    if bool(comparison_config["run_per_system_analysis"]):
        for index, inputs in enumerate(system_inputs):
            system_id = str(inputs["system_id"])
            system_filename = f"system-{system_id}.json"
            single_system = {"systems": [deepcopy(system_rows[index])]}
            single_system_path = root / system_filename
            _json_write(single_system_path, single_system)
            validate_system(
                single_system, source_path=single_system_path, check_paths=True
            )
            per_system_manifest_files.append(system_filename)
            per_system_project = deepcopy(project)
            per_system_project.update({
                "project_id": f"{project_id}-{system_id}-per-system",
                "system_manifest": system_filename,
                "analysis_output_root": f"results/per-system/{system_id}",
                "reference_system": system_id,
                "reference_structure": str(inputs["pdb"]),
                "reference_connectivity": str(inputs["connectivity"]),
                "common_atom_policy": "strict",
                "selections": deepcopy(inputs["composition"]["selections"]),
                "protected_locations": [
                    str(inputs["pdb"]), str(inputs["connectivity"]),
                    *[str(path) for path in inputs["trajectories"]],
                ],
            })
            (
                context_filename,
                _context_inference,
                context_exclusions,
                generated_context,
            ) = _automatic_context_project(
                root=root,
                system_id=system_id,
                system_manifest_filename=system_filename,
                base_project=per_system_project,
                reference_structure=inputs["pdb"],
                reference_connectivity=inputs["connectivity"],
                source_frame_counts=inputs["frame_counts"],
                composition=inputs["composition"],
                analysis_config=analysis_config,
                dssr_executable=dssr,
            )
            context_generated_files.extend(generated_context)
            for module_id, reason in context_exclusions.items():
                context_not_applicable.setdefault(str(module_id), []).append(
                    f"{system_id}: {reason}"
                )
            if context_filename is not None:
                context_project = load_json(root / context_filename)
                context_requested = context_project.get("requested_modules", [])
                assert isinstance(context_requested, list)
                context_commands = [
                    _AUTOMATIC_CONTEXT_COMMANDS[str(module_id)]
                    for module_id in context_requested
                    if str(module_id) in _AUTOMATIC_CONTEXT_COMMANDS
                ]
                context_id = context_filename[len("project-") : -len(".json")]
                context_projects.append({
                    "context_id": context_id,
                    "system_id": system_id,
                    "project_filename": context_filename,
                    "commands": context_commands,
                })
                context_project_files.append(context_filename)
                context_frame_counts_by_id[context_id] = list(
                    inputs["frame_counts"]
                )
            prefix = f"system_{system_id}"
            system_ids_for_views, system_files = _conformational_view_projects(
                root,
                per_system_project,
                inputs["composition"],
                frame_counts_per_replica=inputs["frame_counts"],
                analysis_config=analysis_config,
                workflow_prefix=prefix,
                plan_filename=f"conformational-views-{system_id}.json",
                output_root_prefix=(
                    f"results/per-system/{system_id}/conformational-views"
                ),
                workflow_scope="per_system_conformational_analysis",
                workflow_system_id=system_id,
            )
            view_ids.extend(system_ids_for_views)
            view_project_files.extend(system_files)
            for view_id in system_ids_for_views:
                view_frame_counts_by_id[view_id] = list(inputs["frame_counts"])
    try:
        campaign_resource_plan = plan_and_apply_complete_campaign(
            root=root,
            sampling_plan=sampling_plan,
            analysis_config=analysis_config,
            view_project_files=view_project_files,
            base_project_path=project_path,
            view_frame_counts_by_id=view_frame_counts_by_id,
            context_project_files=context_project_files,
            context_frame_counts_by_id=context_frame_counts_by_id,
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
                _json_write(root / "memory-feasibility-report.json", {
                    "report_schema": "salsbury-memory-feasibility-report-v1",
                    "technical_status": "complete",
                    "planning_status": "insufficient_memory",
                    "automatic_changes_applied": False,
                    **deepcopy(dict(memory)),
                    "requested_config_path": str(root / "analysis-config.json"),
                    "proposed_action": (
                        "Increase execution.maximum_memory_gib or rerun preparation "
                        "with --auto-disable-to-fit-memory and a new output directory."
                    ),
                })
                raise QuickstartMemoryError(
                    str(exc), plan=exc.plan, analysis_config=analysis_config,
                    output_directory=root,
                ) from exc
            raise QuickstartPlanningError(
                str(exc), plan=exc.plan, analysis_config=analysis_config,
                output_directory=root,
            ) from exc
        raise QuickstartError(str(exc)) from exc
    planner_exclusions = dict(exclusions)
    for module_id, reasons in context_not_applicable.items():
        planner_exclusions.setdefault(
            module_id, "; ".join(sorted(set(reasons)))
        )
    _record_conformational_experimental_exclusions(
        root, view_project_files, planner_exclusions, analysis_config
    )
    # Preserve the complete plan even if the coverage audit itself detects an
    # integration defect. This makes the failure diagnosable without rerunning
    # input discovery.
    _json_write(root / "campaign-resource-plan.json", campaign_resource_plan)
    experimental_planner_coverage = _experimental_planner_coverage(
        analysis_config, campaign_resource_plan, planner_exclusions
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
        len(frame_counts),
    )
    deferred = _deferred_modules(exclusions, disabled)
    automatic_ids = set(requested) | {
        module_id
        for filename in view_project_files
        if filename.startswith("project-") and filename.endswith(".json")
        for module_id in load_json(root / filename).get("requested_modules", [])
    } | {
        module_id
        for filename in context_project_files
        for module_id in load_json(root / filename).get("requested_modules", [])
    }
    # Keep the deferred inventory mutually exclusive with the workflow that
    # was actually generated.  Some generic fallbacks in _deferred_modules
    # become obsolete once automatic chemistry supplies a valid definition.
    deferred = {
        module_id: reason
        for module_id, reason in deferred.items()
        if module_id not in automatic_ids
    }
    for module_id, reasons in context_not_applicable.items():
        if module_id not in automatic_ids:
            deferred[module_id] = "; ".join(sorted(set(reasons)))
    for module_id, reason in planner_exclusions.items():
        if module_id not in automatic_ids:
            deferred[module_id] = str(reason)
    unaccounted = sorted(set(registry_ids).difference(automatic_ids, deferred))
    if unaccounted:
        raise QuickstartError(
            "comparative quickstart module accounting is incomplete: "
            + ", ".join(unaccounted)
        )
    coverage = {
        "coverage_schema": "salsbury-self-service-module-coverage-v1",
        "registry_module_count": len(registry_ids),
        "automatically_requested_modules": requested,
        "automatically_executed_commands": commands,
        "deferred_or_context_specific": deferred,
        "comparison_system_ids": system_ids,
        "comparison_policy": analysis_config["comparisons"],
        "module_status": {
            module_id: (
                {"status": "automatic", "reason": "included in the generated workflow"}
                if module_id in automatic_ids
                else {"status": "deferred", "reason": deferred[module_id]}
            )
            for module_id in registry_ids
        },
        "experimental_planner_coverage": experimental_planner_coverage,
    }
    _json_write(root / "module-coverage.json", coverage)
    (root / "logs").mkdir()
    (root / "results").mkdir()
    context_slurm_files, context_stage_counts = _automatic_context_slurm_files(
        root,
        project_id,
        context_projects,
        target_wall_hours=float(
            analysis_config["execution"]["maximum_hours_per_cpu"]  # type: ignore[index]
        ),
        python_executable=_active_python_executable(),
        package_root=str(Path(__file__).resolve(strict=True).parents[1]),
    )
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
        root,
        project_id,
        commands,
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
        automatic_context_stage_counts=context_stage_counts,
        rmsf_permutation_enabled=("rmsf_permutation_inference" in requested),
        integrated_comparison_enabled=True,
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
    readme = f"""# {project_id}: generated comparative Salsbury MD analysis

This workflow references {len(system_ids)} systems and {len(frame_counts)} replicas in
place. It does not copy or modify PDB, PSF, or DCD inputs. Shared topology-derived
comparison views use one common atom mapping, PCA basis, and grid. Separate per-system
views use an independently fitted basis while pooling only that system's replicas.
Strictly equivalent oligomer members are aligned separately and retain member
provenance; they are not mislabeled as independent replicas.
Every conformational view reads the all-frame made-whole molecular-payload cache;
base water-dependent analyses retain the original solvated trajectories.

Review `planning-report.md` first. It lists every analysis family, its effective raw
stride over the original trajectory, and every explicitly disabled, deferred, or
inapplicable capability. Exact decisions remain in `planning-report.json`,
`campaign-resource-plan.json`, `sampling-plan.json`, and `analysis-config.json`.
Then run:

```bash
cd {root}
./{active_launcher}
```

The active execution adapter is the {adapter_description}. Local, Slurm, and custom
modes execute the same staged workers, outputs, hashes, and scientific definitions. Select
the mode with `execution.submission_adapter`; `custom` hands the scheduler-neutral
`launcher-contract.json` to a user-supplied launcher, and Slurm additionally requires
a validated `execution.slurm_profile`. The retained `execution-adapter.json` and
`local-execution-plan.json` make the launch decision and local dependency graph explicit.

The finding picker applies the configured all-pairs or reference-versus-all policy with
Benjamini-Hochberg correction. Topology-local ion, DNA, RDF, and scalar definitions
are inferred before outcome review and run per system against the original solvated
trajectories. Chemically inapplicable definitions and unavailable external software
remain visible in `module-coverage.json`.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    return {
        "technical_status": "complete",
        "project_id": project_id,
        "output_directory": str(root),
        "system_count": len(system_ids),
        "system_ids": system_ids,
        "replica_count": len(frame_counts),
        "total_source_frame_count": sum(frame_counts),
        "frame_counts_per_replica": frame_counts,
        "reference_connectivity_checks": connectivity_checks,
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
            *per_system_manifest_files, *view_project_files,
            *context_generated_files, *context_slurm_files,
            *([energetic_availability_file] if energetic_availability_file else []),
            *coordinate_cache_files, *view_slurm_files,
            "analysis-config.json", *slurm_files,
            *execution_artifacts["generated_files"], *planning_report_files,
            "README.md",
        ],
        "execution_adapter": execution_artifacts["adapter"],
        "slurm_profile_id": execution_artifacts["slurm_profile_id"],
        "next_command": execution_artifacts["next_command"],
    }


def prepare_comparative_analysis_resource_fit(
    *,
    request_path: Path,
    output_directory: Path,
    project_id: str,
    temperature_kelvin: float = 300.0,
    target_wall_hours: Optional[float] = None,
    dssp_executable: Optional[str] = None,
    config_path: Optional[Path] = None,
    protected_core_only: bool = False,
    uniform_cache_stride: bool = False,
) -> Dict[str, object]:
    """Prepare an opt-in dependency-closed comparison resource reduction."""

    destination = output_directory.expanduser().resolve(strict=False)
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise QuickstartError(
            f"output directory is not empty: {destination}; choose a new "
            "versioned directory"
        )
    common = {
        "request_path": request_path,
        "project_id": project_id,
        "temperature_kelvin": temperature_kelvin,
        "dssp_executable": dssp_executable,
        "protected_core_only": protected_core_only,
        "uniform_cache_stride": uniform_cache_stride,
    }
    with tempfile.TemporaryDirectory(
        prefix="salsbury-comparison-resource-fit-planning-"
    ) as temporary:
        temporary_root = Path(temporary)
        requested_output = temporary_root / "requested-plan"
        recommendation: Mapping[str, object] = {}
        try:
            prepare_comparative_analysis(
                **common,
                output_directory=requested_output,
                target_wall_hours=target_wall_hours,
                config_path=config_path,
            )
            requested_config = load_json(
                requested_output / "analysis-config.json"
            )
            requested_plan = load_json(
                requested_output / "campaign-resource-plan.json"
            )
            active_config = deepcopy(requested_config)
            direct_disabled: list[str] = []
            transitive_disabled: list[str] = []
        except QuickstartPlanningError as exc:
            requested_config = deepcopy(exc.analysis_config)
            requested_plan = deepcopy(exc.plan)
            raw_recommendation = exc.plan.get(
                "method_reduction_recommendation"
            )
            if not isinstance(raw_recommendation, Mapping):
                raise QuickstartError(
                    "comparison resource-fit planning produced no dependency-closed "
                    "reduction recommendation"
                ) from exc
            recommendation = raw_recommendation
            if recommendation.get("recommendation_status") != (
                "feasible_subset_found"
            ):
                raise
            patch = recommendation.get("configuration_patch")
            if not isinstance(patch, Mapping) or not patch:
                raise QuickstartError(
                    "comparison resource-fit recommendation contains no "
                    "configuration patch"
                ) from exc
            active_config, direct_disabled, transitive_disabled = (
                make_resource_fit_config(requested_config, list(patch))
            )

        resolved_path = temporary_root / "resolved-resource-fit-config.json"
        _json_write(resolved_path, active_config)
        report = prepare_comparative_analysis(
            **common,
            output_directory=destination,
            target_wall_hours=None,
            config_path=resolved_path,
        )
        final_plan = load_json(destination / "campaign-resource-plan.json")
        fit_report = {
            "report_schema": "salsbury-resource-fit-report-v1",
            "technical_status": "complete",
            "planning_status": (
                "replanned_with_dependency_closed_optional_reduction"
                if direct_disabled or transitive_disabled
                else "requested_plan_feasible"
            ),
            "automatic_changes_applied": bool(
                direct_disabled or transitive_disabled
            ),
            "protected_set_preserved": bool(
                recommendation.get("protected_set_preserved", True)
            ),
            "directly_disabled_configuration_switches": sorted(
                direct_disabled
            ),
            "transitively_disabled_modules": sorted(transitive_disabled),
            "reduction_decisions": deepcopy(
                recommendation.get("decisions", [])
            ),
            "requested_config": "analysis-config.requested.json",
            "resolved_config": "analysis-config.resource-fit.json",
            "requested_plan": "campaign-resource-plan.requested.json",
            "final_plan": "campaign-resource-plan.json",
            "final_feasibility_status": final_plan.get("feasibility_status"),
            "original_request_preserved": True,
            "failure_boundary": (
                "Preparation fails when no dependency-closed subset preserving "
                "every protected module can fit the requested resource envelope."
            ),
        }
        _json_write(
            destination / "analysis-config.requested.json", requested_config
        )
        _json_write(
            destination / "analysis-config.resource-fit.json", active_config
        )
        _json_write(
            destination / "campaign-resource-plan.requested.json", requested_plan
        )
        _json_write(destination / "resource-fit-report.json", fit_report)
        generated = report.get("generated_files")
        if isinstance(generated, list):
            generated.extend([
                "analysis-config.requested.json",
                "analysis-config.resource-fit.json",
                "campaign-resource-plan.requested.json",
                "resource-fit-report.json",
            ])
        report["resource_fit"] = fit_report
        return report


def prepare_comparative_analysis_memory_fit(
    *,
    request_path: Path,
    output_directory: Path,
    project_id: str,
    temperature_kelvin: float = 300.0,
    target_wall_hours: Optional[float] = None,
    dssp_executable: Optional[str] = None,
    dssr_executable: Optional[str] = None,
    config_path: Optional[Path] = None,
    protected_core_only: bool = False,
    uniform_cache_stride: bool = False,
) -> Dict[str, object]:
    """Prepare a comparison with an explicit, opt-in memory-fit reduction."""

    destination = output_directory.expanduser().resolve(strict=False)
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise QuickstartError(
            f"output directory is not empty: {destination}; choose a new "
            "versioned directory"
        )
    common = {
        "request_path": request_path,
        "project_id": project_id,
        "temperature_kelvin": temperature_kelvin,
        "dssp_executable": dssp_executable,
        "dssr_executable": dssr_executable,
        "protected_core_only": protected_core_only,
        "uniform_cache_stride": uniform_cache_stride,
    }
    with tempfile.TemporaryDirectory(
        prefix="salsbury-comparison-memory-fit-planning-"
    ) as temporary:
        temporary_root = Path(temporary)
        probe_output = temporary_root / "requested-plan"
        try:
            prepare_comparative_analysis(
                **common, output_directory=probe_output,
                target_wall_hours=target_wall_hours, config_path=config_path,
            )
            requested_config = load_json(probe_output / "analysis-config.json")
            active_config = deepcopy(requested_config)
            requested_plan = load_json(
                probe_output / "campaign-resource-plan.json"
            )
            current_plan = deepcopy(requested_plan)
            direct_disabled: list[str] = []
            transitive_disabled: list[str] = []
        except QuickstartMemoryError as exc:
            requested_config = deepcopy(exc.analysis_config)
            active_config = deepcopy(exc.analysis_config)
            requested_plan = deepcopy(exc.plan)
            current_plan = deepcopy(exc.plan)
            direct_disabled = []
            transitive_disabled = []
            for iteration in range(1, len(requested_config["modules"]) + 3):
                memory = current_plan.get("memory_feasibility")
                if not isinstance(memory, Mapping):
                    raise QuickstartError(
                        "comparison memory fallback received no feasibility report"
                    )
                switches = memory.get(
                    "configuration_switches_to_disable_to_fit_configured_memory",
                    memory.get("modules_to_disable_to_fit_configured_memory", []),
                )
                if not isinstance(switches, list) or not switches:
                    raise QuickstartError(
                        "comparison memory fallback found no configurable switch"
                    )
                active_config, direct, transitive = make_memory_fit_config(
                    active_config, [str(value) for value in switches]
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
                    prepare_comparative_analysis(
                        **common, output_directory=retry_output,
                        target_wall_hours=None, config_path=reduced_path,
                    )
                    break
                except QuickstartMemoryError as retry_exc:
                    current_plan = deepcopy(retry_exc.plan)
            else:
                raise QuickstartError(
                    "comparison memory fallback did not converge after disabling "
                    "every reported oversized switch"
                )

        resolved_path = temporary_root / "resolved-memory-fit-config.json"
        _json_write(resolved_path, active_config)
        report = prepare_comparative_analysis(
            **common, output_directory=destination,
            target_wall_hours=None, config_path=resolved_path,
        )
        final_plan = load_json(destination / "campaign-resource-plan.json")
        fallback_report = {
            "report_schema": "salsbury-memory-feasibility-report-v1",
            "technical_status": "complete",
            "planning_status": "replanned_with_explicit_reduced_config",
            "automatic_changes_applied": bool(
                direct_disabled or transitive_disabled
            ),
            "requested_memory": deepcopy(
                requested_plan.get("memory_feasibility", {})
            ),
            "final_memory": deepcopy(
                final_plan.get("memory_feasibility", {})
            ),
            "directly_disabled_configuration_switches": sorted(direct_disabled),
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
