"""Check downstream population limits without calculating a PCA or clustering."""

from pathlib import Path


def validate_projection_guards(project, selected_observations, *, tica_selected_observations=None):
    """Limits are rejection guards, not permission to change scientific sampling."""
    definitions = project.get("definitions", {})
    requested = project.get("requested_modules", list(definitions))
    for module, field in (("representative_frames", "maximum_candidates"),
                          ("grouped_ml", "maximum_observations")):
        settings = definitions.get(module)
        if module not in requested or not isinstance(settings, dict) or field not in settings:
            continue
        producer = "clustering_kmeans" if module == "grouped_ml" else settings.get("source")
        source = definitions.get(producer, {}).get("feature_source", "common_pca")
        count = selected_observations
        if source == "tica":
            tica = definitions.get("time_lagged_independent_component_analysis", {})
            if tica.get("short_segment_policy", "error") == "omit":
                if tica_selected_observations is None:
                    continue  # Verify after segment-level cache metadata exists.
                count = tica_selected_observations
        elif source != "common_pca":
            continue  # A trajectory-feature stream is not a PCA population.
        limit = settings[field]
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"definitions.{module}.{field} must be a positive integer")
        if count > limit:
            raise ValueError(
                f"Project {project.get('project_id')!r}: {module} receives "
                f"{count} selected pooled observations, exceeding "
                f"definitions.{module}.{field}={limit}. Review the explicit limit "
                "and replan resources before launch; no limit or sampling was changed."
            )


def validate_project_observation_guards(project, source_path):
    """Use the runtime frame selector; count headers, not fitted states.

    A derived cache may not exist during initial preparation. Its final view
    allocation is checked separately, and this check repeats before execution.
    """
    from .frame_sampling import normalize_frame_selection, plan_frame_selection
    from .manifests import load_json, resolve_manifest_path
    definitions = project.get("definitions", {})
    requested = project.get("requested_modules", list(definitions))
    if not any(key in requested and key in definitions for key in ("representative_frames", "grouped_ml")):
        return
    pca = definitions.get("common_pca")
    if not isinstance(pca, dict) or not source_path or not project.get("system_manifest"):
        return
    system_path = resolve_manifest_path(project["system_manifest"], Path(source_path))
    if not system_path.is_file():
        return
    system = load_json(system_path)
    paths = [resolve_manifest_path(segment["trajectory"], system_path)
             for item in system.get("systems", []) for replica in item.get("replicas", [])
             for segment in replica.get("segments", [])]
    if not paths or not all(path.is_file() for path in paths):
        return
    stride = pca.get("projection_frame_stride", pca.get("frame_stride", 1))
    selection = normalize_frame_selection(pca.get("projection_frame_selection"), stride)
    _, report = plan_frame_selection(system, system_path, project.get("coordinate_unit", "angstrom"),
                                     selection, frame_stride=stride)
    symmetry = pca.get("symmetry_expansion", {})
    multiplier = symmetry.get("member_count", 1)
    if isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier < 1:
        raise ValueError("common_pca symmetry expansion member_count must be a positive integer")
    count = report["selected_frame_count"] * multiplier
    tica_count = None
    tica = definitions.get("time_lagged_independent_component_analysis", {})
    if tica.get("short_segment_policy", "error") == "omit":
        lag, minimum = tica.get("lag_frames"), tica.get("minimum_pairs_per_segment")
        if all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in (lag, minimum)):
            tica_count = sum(segment["selected_frame_count"] for replica in report["replicas"]
                             for segment in replica["segments"]
                             if segment["selected_frame_count"] >= lag + minimum) * multiplier
    validate_projection_guards(project, count, tica_selected_observations=tica_count)
    return count
