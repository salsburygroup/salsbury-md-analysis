"""Named primary quantities for optional experimental reports.

Only declared result fields are plotted; configuration and resource values are
never substituted for an absent scientific quantity.
"""
from __future__ import annotations

SPECIFICATIONS = {
    "perturbation_response_dynamics": {"dfi": "Dynamic flexibility index", "dci": "Dynamic coupling index"},
    "allosteric_pathways": {"combined_allosteric_score": "Combined pathway score", "weighted_betweenness_centrality": "Weighted betweenness"},
    "energetic_network_embeddings": {"summed_wasserstein_distance": "Summed Wasserstein distance (embedding units)"},
    "multivalent_molecular_bridges": {"bridge_occupancy": "Bridge occupancy fraction", "mean_active_bridges_per_frame": "Mean active bridges per frame"},
    "interaction_fingerprints": {"occupancy_fraction": "Interaction occupancy fraction", "phi_coefficient": "Phi correlation"},
    "spatial_interaction_ensembles": {"centroid_displacement_angstrom": "Centroid displacement (Å)"},
    "ensemble_pocket_dynamics": {"occupancy_fraction": "Pocket occupancy fraction", "occupancy_difference": "Pocket occupancy difference", "volume_angstrom3": "Pocket volume (Å³)"},
    "hydration_density_channels": {"mean_voxel_frame_occupancy": "Mean voxel frame occupancy", "volume_angstrom3": "Hydration component volume (Å³)", "maximum_absolute_voxel_occupancy_difference": "Maximum absolute voxel occupancy difference"},
    "helical_mechanics": {"population_fraction": "Step-state population fraction", "mutual_information_bits": "Neighbor-step mutual information (bits)"},
    "reactive_path_ensembles": {"complete_path_count": "Complete reactive paths", "population_fraction": "Route population fraction"},
    "trajectory_reweighting": {"kish_effective_sample_size_ratio": "Kish effective-sample ratio", "kish_ratio": "Kish effective-sample ratio"},
    "interaction_persistence": {"occupancy_fraction": "Interaction occupancy fraction", "complete_event_count": "Complete events"},
}
SKIP = {"settings", "thresholds", "execution_resources", "planner_benchmark", "frame_records",
        "frames", "segments", "projections", "frame_feature_records", "frame_pocket_region_records",
        "voxel_frequency_map", "density_projections_xy", "finding_candidates", "issues", "limitations"}
IDENTITIES = ("system_id", "system_i", "system_j", "replica_id", "feature_id", "node_id",
              "state_id", "basin_id", "region_id", "mediator_type", "species", "step_id")


def generate(output_root, path, report, module_id, artifacts):
    from .presentation_artifacts import (_bar_svg, _finite, _register_pair, _slug,
                                         human_label)
    fields = SPECIFICATIONS[module_id]
    rows_by_quantity = {name: [] for name in fields}

    def visit(value, location="", identities=()):
        if isinstance(value, dict):
            identity = (*identities, *(f"{key}={value[key]}" for key in IDENTITIES if key in value))
            for key, child in value.items():
                if key in SKIP:
                    continue
                address = f"{location}.{key}" if location else key
                if key in fields:
                    values = child if isinstance(child, list) else [child]
                    for index, number in enumerate(values):
                        number = _finite(number)
                        if number is not None:
                            source_row = address + (f"[{index}]" if isinstance(child, list) else "")
                            rows_by_quantity[key].append({
                                "source_row": source_row,
                                "label": "; ".join(identity) + ("; " if identity else "") + source_row,
                                "value": number,
                            })
                elif isinstance(child, (dict, list)):
                    visit(child, address, identity)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, (dict, list)):
                    visit(child, f"{location}[{index}]", identities)
    visit(report)
    for quantity, rows in rows_by_quantity.items():
        if not rows:
            continue
        axis = fields[quantity]
        title = f"{human_label(module_id)}: {axis}"
        ordered = sorted(rows, key=lambda row: (-abs(row["value"]), row["source_row"]))
        _register_pair(output_root, path, artifacts, module_id=module_id, purpose=quantity,
            title=title, directory=output_root / _slug(module_id) / quantity,
            rows=ordered, fieldnames=("source_row", "label", "value"),
            svg=_bar_svg(ordered, title, "label", "value", axis, maximum_rows=60),
            context={"quantity": quantity, "figure_row_limit": 60,
                     "table_row_count": len(rows), "row_order": "absolute magnitude; all rows retained in CSV"})
