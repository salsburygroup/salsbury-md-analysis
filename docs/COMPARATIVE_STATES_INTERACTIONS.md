# Comparative states, ions, and hydrogen bonds

All capabilities on this page are **experimental**. They provide reproducible
definitions and technical evidence; they do not establish convergence,
affinity, metastability, mechanism, or biological validity.

## Shared-basis system comparisons

`pca_fes_basins` fits one common PCA basis across the declared systems. Its
pooled surface defines the grid bounds, bin counts, bin centers, and smoothing
settings. The module also calculates one conditional surface per system on
that exact grid. Each system surface is normalized independently, so positions
and shapes can be compared but free-energy/occupancy offsets cannot be compared
between systems. Basin identifiers are local to a surface and do not assert a
cross-system state correspondence.

The pooled FES and every clustering runner report `state_population_comparison`:
raw state counts, fractions of all evaluated frames, fractions of assigned
frames, assigned/noise coverage, replica breakdowns, and pairwise descriptive
system differences. These are frame summaries, not inferential confidence
intervals; replica- or time-block uncertainty remains required.

## Ion-distance clustering and threshold binding states

KMeans, iMWK-Means, HDBSCAN, and `alternative_clustering` accept either
`tica`, `common_pca`, or explicitly selected `trajectory_features` columns. Ion-site,
ion-pair, or loop-to-ion distances can therefore be clustered with the same
parameter grids, silhouette evidence, retained-coverage reporting, and exact
frame identities as PCA features. The selected feature IDs and one-based value
indices are part of the manifest contract.

Alternative-clustering families may fit distinct deterministic integer-stride
samples from every replica while assigning the resulting partition to every
source observation. This permits inexpensive mixture models to use more fit
observations than quadratic medoid, similarity, or neighborhood methods. PAM
and MWPAM use their fitted medoids; Gaussian and
variational mixtures, affinity propagation, and mean shift use the fitted
model's prediction. Reports distinguish exact assignment to a sampled fitted
model from an approximate extension that is not equivalent to a full refit.
Ward and quality-threshold are excluded from this extension policy.

Ward and quality-threshold are not run when the planner would require a
subsampled fit. A full exact Ward partition is primary-eligible; quality-
threshold is primary-eligible only when the exact full fit also assigns every
observation. No approximate Ward/QT all-frame extension or sampled MSM is
produced. HDBSCAN, when explicitly enabled, is treated separately on its
retained dense cores: no transition
crosses a noise gap, and every population or timescale is explicitly conditional
on the retained observations. These partial models can test whether sampled
cores exhibit internally consistent exchange across lag choices; it cannot
recover the kinetics or equilibrium population of omitted frames.

The default large-system quick start requests one fit budget, not repeated
fits. Optional B-versus-2B sensitivity declares two per-replica budgets and
reports full-partition adjusted Rand agreement. Multi-budget resource
calibration must instead run each budget as a separate scheduler job so wall
time and peak memory remain attributable to one fit size. PaLD is planned as a
separate cubic, strictly gated community analysis and is not modeled as a
quadratic conventional clustering method.

`scalar_threshold_states` converts one scalar feature, such as
`group_minimum_mean_distance` from a loop to candidate ions, into a declared
two-state definition. It reports the primary threshold, a required sensitivity
grid, state assignments, system and replica populations, within-segment
transition counts, and boundary-censored residence runs. A threshold-defined
"bound" state is operational only; it is not a binding free energy, affinity,
or electronic-structure determination.

For inferred ion questions, `trajectory_features` is the sole coordinate-pass
producer. Scalar distributions and threshold states are budgeted separately as
postprocessing tasks, run after that producer, and accept only a technically
complete report whose project, module contract, system manifest, and current
input-content signature match. Each downstream report records whether it reused
that validated upstream result or computed features directly in a standalone
invocation.

## Hydrogen bonds: explicit evaluation versus discovery

`hydrogen_bonds` remains the strict option when the scientific question names
specific donor–hydrogen–acceptor triples. It does not search for other bonds.

`hydrogen_bond_discovery` is the broad default. With
`automatic_topology_templates_v1`, it derives donor and acceptor roles from
standard protein and nucleic-acid chemical templates plus explicit PSF,
PRMTOP/PARM7, or portable bond-JSON connectivity. The user chooses a scientific
scope such as `all_solute`, `protein_ligand`, or `protein_nucleic_acid`; atom
index lists are not required. Standard macromolecular roles are templated.
Unknown covalently connected residues use a conservative N/O/S fallback and
are explicitly marked *provisional* so their protonation, charge, and bond order
can be reviewed.

The default `mdanalysis_compatible_v1` primary geometry is D--A <= 3.0 Å and
D--H--A >= 150°. It also records a prespecified 3.0/3.2/3.5 Å by
120/135/150° sensitivity grid. A `custom_v1` policy permits a different primary
rule and sensitivity grid, with a 24-combination resource gate. Geometry is
always labeled as donor--acceptor distance; it is never silently mixed with a
hydrogen--acceptor cutoff. In the default
`sparse_spatial_observed_union_v3` path, the complete topology-eligible universe
is defined by compact donor--hydrogen groups and acceptor endpoints. Its
Cartesian product remains implicit: it is neither stored nor scanned on every
frame. An exact periodic cell list finds only endpoints within the largest
declared distance cutoff, and angle evaluation is limited to those nearby
pairs. A stable candidate identity is materialized only if it satisfies at
least one prespecified cutoff in at least one selected frame. Candidates absent
from a frame are exact zeros; topology-eligible identities absent from the
pooled observed dictionary are exact global zeros. Reports retain exact
conceptual and observed counts, interaction-stratum counts, and geometry-work
avoidance. This avoids hidden outcome-dependent feature selection without a
million-entry Python dictionary.

Endpoint matching uses chain, residue number, insertion code, atom name, and
alternate location, so atom-index shifts do not corrupt comparisons. The
actual residue name and element are retained separately for every system. An
unchanged endpoint keeps its chemical residue label; a position occupied by
different homologous residues is labeled `POSITION_HARMONIZED` and retains a
per-system residue-name map. Element disagreement at a matched position, or
residue-name disagreement among replicas of one system, fails closed. These
rules are system-agnostic and require no package-coded residue substitutions.

The older `sparse_implicit_zero_v1` and `sparse_packed_v2` modes retain explicit
candidate dictionaries for compatibility. `dense_v1` is legacy-only for small
publication-locked dictionaries. Direct water contacts are excluded from this
contract; the separate water-mediated module evaluates one-water bridge paths.

`explicit_atoms_connectivity_v1` remains available for a publication lock or a
question about named donor--hydrogen--acceptor triples.

## Comparing direct hydrogen bonds across conditions

For homologous systems, prefer one multi-system discovery report and select the
two conditions by `system_id`. This guarantees that a bond observed in either
condition has the same stable candidate identity in both, while a missing
per-frame event remains an exact zero. Each comparison must use the same
interaction scope and cutoff definition. The `all_solute` scope includes and
labels protein--protein, protein--nucleic-acid, nucleic-acid--protein, and
nucleic-acid--nucleic-acid strata; it does not include water or ions.

`compare-hydrogen-bonds` consumes two condition selections from technically
complete sparse discovery reports, including
`sparse_spatial_observed_union_v3`. It collapses all explicit
hydrogens bonded to the same donor heavy atom before calculating occupancy.
Thus interchangeable amide or amine hydrogens can satisfy one chemical
donor--acceptor interaction in a frame, but never count it twice. Condition
occupancies are reported both as equal-replica means and as pooled-frame
descriptors, with per-replica evidence retained.

Nonidentical residues are never aligned implicitly. Optional homolog mappings
are explicit entries in the comparison request. Each entry selects a chemical
identity in one named condition and applies declared canonical identity
updates; unmatched, overlapping, or within-condition collision-producing
mappings fail closed. Project-specific mappings such as a native base and a
modified-base homolog belong in the project request or publication lock, not
in package source.

An aligned donor--acceptor group that exists in a condition's complete
candidate universe but is never present has zero occupancy. A group containing
a topology-specific atom has `null` occupancy in the chemically ineligible
condition and is excluded from ranked numeric differences; chemical absence is
never silently converted into an evaluated zero.

```json
{
  "comparison_id": "condition-a-vs-condition-b-protein-dna",
  "conditions": [
    {"condition_id": "condition-a", "report": "a.report.json"},
    {"condition_id": "condition-b", "report": "b.report.json"}
  ],
  "cutoff_id": "primary",
  "group_donor_hydrogens": true,
  "expected_interaction_scope": "protein_nucleic_acid",
  "homolog_mappings": [
    {
      "condition_id": "condition-a",
      "match": {"chain_id": "C", "residue_number": 4, "residue_name": "NATIVE"},
      "canonical_updates": {"residue_name": "TARGET_HOMOLOG"}
    },
    {
      "condition_id": "condition-b",
      "match": {"chain_id": "C", "residue_number": 4, "residue_name": "MODIFIED"},
      "canonical_updates": {"residue_name": "TARGET_HOMOLOG"}
    }
  ],
  "top_n": 100
}
```

Execute with:

```bash
salsbury-md-analysis compare-hydrogen-bonds comparison-request.json > comparison.report.json
```

What it currently cannot do:

- discover a donor–hydrogen bond when explicit connectivity is absent;
- represent reactive proton transfer or infer missing protonation;
- turn occupancy or a classifier coefficient into energetic or mechanistic
  evidence.

Automatic chemistry cannot repair a missing or incorrect protonation,
tautomer, formal charge, or ligand bond order. These conditions are reported as
provisional rather than inferred from coordinate distances.

## One-water-mediated networks

`water_mediated_hydrogen_bond_networks` is the separate scalable contract for
solvent bridges. The user selects an interaction scope and geometry policy but
does not provide donor, acceptor, hydrogen, or water atom lists. Solute roles
use the same topology templates as direct discovery. Waters are recognized by
standard residue identity and must have one oxygen plus connectivity-declared
O–H bonds.

For each frame, a nonperiodic or conservative fractional cell list finds only
solute endpoint–water oxygen neighbors within the largest declared cutoff.
Every retained edge is then checked with exact donor–acceptor distance,
donor–hydrogen–acceptor angle, and triclinic minimum-image geometry. A bridge is
one water hydrogen-bonded to two distinct in-scope solute endpoints. The module
retains donor–acceptor, donor–donor, and acceptor–acceptor endpoint relations,
all default or custom cutoff-sensitivity results, direct-versus-mediated
coincidence, water multiplicity, sparse frame paths, descriptive network
degrees, and deterministic representative-frame locators.

Residence is reported two ways: continuity of the bridge through any water and
continuity of the same water identity. Runs never cross segment boundaries and
record left/right censoring. They are sampled-frame descriptors, not kinetic
rates. Version 1 deliberately stops at one bridging water; multi-water wires,
Grotthuss proton hopping, water exchange energetics, affinity, and mechanistic
interpretation remain outside the contract. Its bridge dictionary is the union
of observed paths, so it must not be used as an outcome-independent predictive
feature filter.

The project-manifest definition is intentionally explicit about scale limits:

```json
{
  "water_mediated_hydrogen_bond_networks": {
    "chemistry_policy": "automatic_topology_templates_v1",
    "interaction_scope": "all_solute",
    "water_identity_policy": "standard_residue_names_connectivity_v1",
    "maximum_bridge_length": 1,
    "exclude_same_residue_endpoints": true,
    "frame_stride": 1,
    "frame_selection": {
      "mode": "integer_stride_per_replica_v1",
      "stride": 20
    },
    "cutoff_policy": {"preset": "mdanalysis_compatible_v1"},
    "maximum_reference_donor_hydrogen_bond_angstrom": 1.5,
    "neighbor_search": "cell_list_v1",
    "maximum_solute_endpoints": 5000,
    "maximum_waters": 30000,
    "maximum_evaluated_frames": 100000,
    "maximum_neighbor_pairs_per_frame": 100000,
    "maximum_bridge_paths_per_frame": 1000000,
    "maximum_sparse_records": 10000000
  }
}
```

The standard water aliases include the three-character legacy CHARMM PDB label
`TIP` as well as `TIP3`/`TIP3P`, `HOH`, `WAT`, `SOL`, and the other declared
aliases in the chemistry registry. This matters for large solvated PDB files:
`TIP3` can be truncated to `TIP`, and it must remain solvent rather than enter a
solute-heavy atom mapping.

Water identity does not rely on uniqueness of the PDB residue fields. Each
recognized oxygen and its explicit connectivity-declared hydrogens define one
water, and the oxygen atom index is part of the stable analysis identity. This
is required for large legacy PDB files in which fixed-width residue numbers may
wrap and repeat; repeated residue labels remain separate molecules.

These are example capacity limits, not universal defaults. Set them after a
read-only input inventory and retain the selected limits in a publication lock.
`fixed_stride_v1` preserves the historical per-segment stride behavior. The
recommended `integer_stride_per_replica_v1` applies one exact integer stride
over each replica's full concatenated segment history and reports exact
coverage by replica and segment. It retains frame zero without forcing the last
frame and requires `frame_stride: 1`; `maximum_evaluated_frames` remains a
fail-closed global cap. The legacy near-uniform budget mode is readable for
frozen-project reproduction but is not emitted by the generic workflow.
For non-continuous DCD analysis, skipped frames have their record envelopes
validated without decoding coordinates; continuous unwrapping still processes
every raw frame to preserve image history.
The 1.5 Å reference D–H gate accommodates standard sulfur–hydrogen bonds;
users should lower or raise it only with force-field and topology evidence.

## Grouped logistic and elastic-net analysis

`grouped_regularized_classification` uses the discovery matrix to classify
declared systems. It scans L2 logistic and elastic-net regularization grids
inside nested leave-one-group-out validation. The held-out outer group never
enters model fitting, hyperparameter selection, or top-feature selection.
Outputs include inner tuning records, outer held-out confusion/precision/recall/
F1 metrics, coefficients, and training-only top-feature/no-top-feature
ablations.

Replica holdout is preferred. Segment holdout is acceptable when segments are
independent. Time-block holdout reduces adjacent-frame leakage but cannot make
blocks equivalent to independently prepared or simulated replicas. Each target
class must have at least two declared groups, otherwise execution fails closed.
