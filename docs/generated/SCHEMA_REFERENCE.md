# Schema reference

> Generated from `schemas/*.json`.

## Salsbury MD analysis preparation configuration

Source: `schemas/analysis-config.schema.json`

| Field | Required | Type or constraint |
|---|---|---|
| `config_schema` | yes | `structured` |
| `default_module_policy` | no | `structured` |
| `planning` | no | `object` |
| `module_groups` | no | `object` |
| `modules` | no | `object` |
| `views` | no | `object` |
| `reporting` | no | `object` |
| `comparisons` | no | `object` |
| `sampling` | no | `object` |
| `clustering` | no | `object` |
| `community_analysis` | no | `object` |
| `inference` | no | `object` |
| `oligomers` | no | `object` |
| `exports` | no | `object` |
| `execution` | no | `object` |

## Publication or Project Analysis Lock

Source: `schemas/analysis-lock.schema.json`

| Field | Required | Type or constraint |
|---|---|---|
| `project_id` | yes | `string` |
| `suite_repository` | no | `string` |
| `suite_version` | no | `string` |
| `suite_commit` | yes | `string` |
| `project_commit` | yes | `string` |
| `profile_id` | yes | `string` |
| `environment_identity` | yes | `string` |
| `input_manifest_sha256` | yes | `#/$defs/sha256` |
| `source_manifest_sha256` | yes | `#/$defs/sha256` |
| `output_manifest_sha256` | no | `#/$defs/sha256` |
| `authoritative_data_roots` | no | `array` |
| `external_dependencies` | no | `array` |
| `commands` | no | `array` |
| `random_seeds` | no | `array` |
| `owner` | yes | `string` |
| `reviewers` | no | `array` |
| `technical_status` | yes | `['complete', 'failed', 'blocked', 'skipped', 'withdrawn']` |
| `scientific_status` | yes | `string` |
| `frame_budget_sensitivity` | no | `object` |
| `replica_diagnostics` | no | `object` |
| `limitations` | no | `array` |

## Grouped hydrogen-bond comparison request

Source: `schemas/hydrogen-bond-comparison-request.schema.json`

| Field | Required | Type or constraint |
|---|---|---|
| `comparison_id` | no | `string` |
| `conditions` | yes | `array` |
| `cutoff_id` | no | `string` |
| `group_donor_hydrogens` | no | `structured` |
| `expected_interaction_scope` | no | `['all_solute', 'protein_protein', 'protein_ligand', 'protein_nucleic_acid', 'nucleic_acid_nucleic_acid', 'nucleic_acid_ligand', 'ligand_ligand']` |
| `homolog_mappings` | no | `array` |
| `top_n` | no | `integer` |

## Analysis Output Manifest

Source: `schemas/output-manifest.schema.json`

| Field | Required | Type or constraint |
|---|---|---|
| `run_id` | yes | `string` |
| `suite_commit` | yes | `string` |
| `profile_id` | yes | `string` |
| `modules` | yes | `array` |
| `technical_status` | yes | `['complete', 'failed', 'blocked', 'partial']` |
| `scientific_status` | yes | `string` |
| `limitations` | no | `array` |

## Salsbury MD Analysis Project Configuration

Source: `schemas/project.schema.json`

| Field | Required | Type or constraint |
|---|---|---|
| `project_id` | yes | `string` |
| `analysis_profile` | yes | `string` |
| `system_manifest` | yes | `string` |
| `analysis_output_root` | yes | `string` |
| `reference_system` | no | `['string', 'null']` |
| `production_interval` | no | `object` |
| `analysis_stride` | no | `['integer', 'string']` |
| `temperature_kelvin` | no | `number` |
| `coordinate_unit` | no | `['angstrom', 'nanometer']` |
| `time_unit` | no | `['fs', 'ps', 'ns', 'us']` |
| `periodic_coordinate_policy` | no | `['reject', 'allow_wrapped_diagnostic', 'make_whole', 'unwrap_continuous', 'preprocessed_make_whole']` |
| `preprocessed_coordinate_source` | no | `object` |
| `periodic_reconstruction` | no | `object` |
| `reference_connectivity` | no | `['string', 'null']` |
| `selections` | no | `object` |
| `sampling_mode` | yes | `['UNBIASED_MD', 'BIASED_MD', 'ENHANCED_SAMPLING', 'AI_ENSEMBLE']` |
| `statistical_weights` | no | `['string', 'null']` |
| `reference_structure` | no | `['string', 'null']` |
| `common_atom_policy` | no | `['string', 'null']` |
| `definitions` | no | `object` |
| `requested_modules` | no | `array` |
| `compute_environment` | no | `['string', 'null']` |
| `protected_locations` | yes | `array` |

## Hash-Pinned Analysis Regression Case

Source: `schemas/regression-case.schema.json`

| Field | Required | Type or constraint |
|---|---|---|
| `regression_id` | yes | `string` |
| `module_id` | yes | `['structural_integrity_qc', 'replica_rmsd_rg', 'pooled_rmsf', 'dccm', 'individual_pca', 'common_pca']` |
| `project_manifest` | yes | `string` |
| `expected_identity` | yes | `object` |
| `assertions` | yes | `array` |
| `approval` | yes | `object` |

## Scientific sampling minimums

Source: `schemas/scientific-minimums.schema.json`

| Field | Required | Type or constraint |
|---|---|---|
| `minimums_schema` | yes | `structured` |
| `base_policy_id` | yes | `structured` |
| `interpretation` | no | `object` |
| `override_policy` | no | `string` |
| `methods` | yes | `object` |

## System and Continuous-Segment Manifest

Source: `schemas/system-manifest.schema.json`

| Field | Required | Type or constraint |
|---|---|---|
| `systems` | yes | `array` |
