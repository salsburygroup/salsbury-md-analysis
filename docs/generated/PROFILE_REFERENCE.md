# Profile reference

> Generated from `profiles/**/*.json`.

## standard_md_v1

- Version: `1.0.0-draft`
- Status: **experimental**

Complete experimental comparative molecular-simulation analysis profile; HDBSCAN, DSSP, and x3dna-dssr require declared external dependencies.

| Order | Module | Current status |
|---:|---|---|
| 1 | `provenance_manifest` | experimental |
| 2 | `preflight_inventory` | experimental |
| 3 | `common_atom_mapping` | experimental |
| 4 | `structural_integrity_qc` | experimental |
| 5 | `replica_rmsd_rg` | experimental |
| 6 | `pooled_rmsf` | experimental |
| 7 | `dccm` | experimental |
| 8 | `generalized_correlation_and_information` | experimental |
| 9 | `information_dynamics` | experimental |
| 10 | `correlation_networks` | experimental |
| 11 | `individual_pca` | experimental |
| 12 | `common_pca` | experimental |
| 13 | `trajectory_features` | experimental |
| 14 | `scalar_feature_distributions` | experimental |
| 15 | `scalar_threshold_states` | experimental |
| 16 | `time_lagged_independent_component_analysis` | experimental |
| 17 | `pca_fes_basins` | experimental |
| 18 | `clustering_kmeans` | experimental |
| 19 | `clustering_hdbscan` | experimental |
| 20 | `clustering_imwkmeans` | experimental |
| 21 | `alternative_clustering` | experimental |
| 22 | `pald_community_analysis` | experimental |
| 23 | `representative_frames` | experimental |
| 24 | `state_coordinate_exports` | experimental |
| 25 | `representative_structures` | experimental |
| 26 | `markov_state_models` | experimental |
| 27 | `dihedral_distributions` | experimental |
| 28 | `hydrogen_bonds` | experimental |
| 29 | `hydrogen_bond_discovery` | experimental |
| 30 | `hydrogen_bond_comparison` | experimental |
| 31 | `water_mediated_hydrogen_bond_networks` | experimental |
| 32 | `hydrogen_bond_patterns` | experimental |
| 33 | `grouped_ml` | experimental |
| 34 | `grouped_regularized_classification` | experimental |
| 35 | `secondary_structure` | experimental |
| 36 | `nucleic_acid_structure` | experimental |
| 37 | `nucleic_acid_geometry` | experimental |
| 38 | `ion_coordination_geometry` | experimental |
| 39 | `ion_atmosphere` | experimental |
| 40 | `solvent_accessible_surface_area` | experimental |
| 41 | `radial_distribution_functions` | experimental |
| 42 | `optional_observables` | experimental |
| 43 | `convergence_uncertainty` | experimental |
| 44 | `rmsf_permutation_inference` | experimental |
| 45 | `integrated_comparison` | experimental |

## Execution configs

### deac-default

Source: `profiles/analysis/deac-default.json`

- Adapter: `slurm`
- Parallel CPU limit: `32`
- Campaign wall-hour limit: `24`
- Slurm profile: `../slurm/deac.json`

### local-default

Source: `profiles/analysis/local-default.json`

- Adapter: `local`
- Parallel CPU limit: `8`
- Campaign wall-hour limit: `24`
- Slurm profile: `none`

## Slurm cluster profiles

### wfu-deac-salsbury-group-v1

Source: `profiles/slurm/deac.json`

- Cluster: `deac`
- Account: `salsburygrp`
- Unix group: `salsburyGrp`
- QoS: `normal`
- Default partition: `small`
- Analysis partition: `small`
- Conformational partition: `small`
- Large-memory partition: `large`
- Large-memory threshold GiB: `96`
- Python executable: `/deac/phy/salsburyGrp/software/salsbury-md-analysis/environments/v76/bin/python3.12`

### replace-with-your-cluster-name

Source: `profiles/slurm/generic-template.json`

- Cluster: `replace-with-your-cluster-name`
- Account: `site/user default`
- Unix group: `site/user default`
- QoS: `site/user default`
- Default partition: `site/user default`
- Analysis partition: `default`
- Conformational partition: `default`
- Large-memory partition: `default`
- Large-memory threshold GiB: `96`
- Python executable: `site/user default`

## Measured resource calibration

Source: `profiles/apollo_measured_resource_calibrations_v2.json`

Measured entries: **109**.
Completed executions: **106**.
Right-censored timeouts: **3**.
