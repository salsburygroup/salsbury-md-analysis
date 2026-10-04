"""Comparable memory axes; missing dimensions never authorize a lower request."""


def memory_workload(module_id, settings, *, maximum_atom_count, selected_frames, workers):
    result = dict(implementation_id=module_id + ":replica-reducer-v1",
                  maximum_atom_count=maximum_atom_count,
                  selected_frames=selected_frames, concurrent_workers=workers)
    fields = (("sphere_point_count", "output_detail") if module_id == "solvent_accessible_surface_area"
              else ("maximum_neighbor_pairs_per_frame", "maximum_bridge_paths_per_frame", "maximum_sparse_records"))
    for field in fields:
        # No guessed defaults: unknown axes make the record ineligible for
        # reducing memory, while task-scoped RSS remains a measured lower bound.
        result[field] = settings.get(field)
    return result


def report_memory_workload(report, selected_frames):
    module = report.get("module_id")
    if module not in {"solvent_accessible_surface_area", "water_mediated_hydrogen_bond_networks"}:
        return {}
    rows = report.get("replicas", []) if module == "solvent_accessible_surface_area" else report.get("chemistry_reports", [])
    counts = [row.get("source_atom_count") for row in rows]
    if not counts or any(not isinstance(value, int) or value <= 0 for value in counts):
        return {}
    parallel = report.get("replica_execution", {})
    workers = parallel.get("workers_used", 1) if len(rows) == 1 else parallel.get("workers_used")
    if workers is None:
        return {}
    return memory_workload(module, report.get("settings", {}), maximum_atom_count=max(counts),
                           selected_frames=selected_frames, workers=workers)
