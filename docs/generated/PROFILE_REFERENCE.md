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
| 10 | `perturbation_response_dynamics` | experimental |
| 11 | `trajectory_reweighting` | experimental |
| 12 | `correlation_networks` | experimental |
| 13 | `allosteric_pathways` | experimental |
| 14 | `energetic_network_embeddings` | experimental |
| 15 | `multivalent_molecular_bridges` | experimental |
| 16 | `hydration_density_channels` | experimental |
| 17 | `ensemble_pocket_dynamics` | experimental |
| 18 | `interaction_fingerprints` | experimental |
| 19 | `spatial_interaction_ensembles` | experimental |
| 20 | `interaction_persistence` | experimental |
| 21 | `individual_pca` | experimental |
| 22 | `common_pca` | experimental |
| 23 | `trajectory_features` | experimental |
| 24 | `scalar_feature_distributions` | experimental |
| 25 | `scalar_threshold_states` | experimental |
| 26 | `time_lagged_independent_component_analysis` | experimental |
| 27 | `random_feature_koopman` | experimental |
| 28 | `pca_fes_basins` | experimental |
| 29 | `clustering_kmeans` | experimental |
| 30 | `clustering_hdbscan` | experimental |
| 31 | `clustering_imwkmeans` | experimental |
| 32 | `alternative_clustering` | experimental |
| 33 | `pald_community_analysis` | experimental |
| 34 | `representative_frames` | experimental |
| 35 | `state_coordinate_exports` | experimental |
| 36 | `representative_structures` | experimental |
| 37 | `markov_state_models` | experimental |
| 38 | `reactive_path_ensembles` | experimental |
| 39 | `dihedral_distributions` | experimental |
| 40 | `hydrogen_bonds` | experimental |
| 41 | `hydrogen_bond_discovery` | experimental |
| 42 | `hydrogen_bond_comparison` | experimental |
| 43 | `water_mediated_hydrogen_bond_networks` | experimental |
| 44 | `hydrogen_bond_patterns` | experimental |
| 45 | `grouped_ml` | experimental |
| 46 | `grouped_regularized_classification` | experimental |
| 47 | `secondary_structure` | experimental |
| 48 | `nucleic_acid_structure` | experimental |
| 49 | `helical_mechanics` | experimental |
| 50 | `nucleic_acid_geometry` | experimental |
| 51 | `ion_coordination_geometry` | experimental |
| 52 | `ion_atmosphere` | experimental |
| 53 | `solvent_accessible_surface_area` | experimental |
| 54 | `radial_distribution_functions` | experimental |
| 55 | `optional_observables` | experimental |
| 56 | `convergence_uncertainty` | experimental |
| 57 | `rmsf_permutation_inference` | experimental |
| 58 | `integrated_comparison` | experimental |

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
- Partition node limits: `{"large": 16, "small": 1}`
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
- Partition node limits: `site defaults`
- Large-memory threshold GiB: `96`
- Python executable: `site/user default`

## Measured resource calibration

Source: `profiles/apollo_measured_resource_calibrations_v5.json`

Measured entries: **311**.
Completed executions: **279**.
Right-censored timeouts: **32**.
