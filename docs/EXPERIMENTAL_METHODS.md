# Default-off experimental methods

The `experimental` branch adds thirteen default-off methods on top of the
current core package. They are registered and shown in generated
`analysis-config.json` files, but all thirteen have `enabled: false` by default.
Turning one or all of them on is always explicit. DFI/DCI, reweighting, and allosteric
pathways also require scientific inputs that preparation cannot infer safely.
The allosteric module derives its physical residue network from the trajectory
by default; only its source and sink residue-node indices must be declared.
Reactive paths can instead discover a clearly labeled, non-biological endpoint
pair from the existing state assignments.

Relative `weights_path` values are resolved from the generated project manifest
that consumes them; absolute paths are also accepted. Keep the external file at
that resolved location before execution. `network_path` is used only by the
explicit `external_json` pathway-network override.

## Configuration switches

`enable_all_experimental_modules: true` activates all thirteen in one step. Module
entries are applied afterward, so an explicit `enabled: false` can leave any
one out. The switch does not invent required weights or site definitions and
does not bypass applicability gates. Replace the example paths and zero-based
node indices before use.

Declaring nonempty DFI/DCI `functional_site_node_indices` automatically enables
the required `macromolecular_trace` view. Its trajectory export remains off
unless enabled separately. Explicitly disabling that view while supplying DFI/DCI
functional-site nodes is a configuration error. With no functional-site nodes,
the module stays enabled in the resolved configuration but is reported as not
available; the workflow does not guess a biological site.

```json
{
  "config_schema": "salsbury-analysis-config-v1",
  "enable_all_experimental_modules": true,
  "modules": {
    "perturbation_response_dynamics": {
      "options": {
        "functional_site_node_indices": [12, 47, 83],
        "minimum_cumulative_explained_variance": 0.8
      }
    },
    "trajectory_reweighting": {
      "options": {
        "weights_path": "inputs/frame-log-weights.json"
      }
    },
    "allosteric_pathways": {
      "options": {
        "source_node_indices": [12, 47],
        "sink_node_indices": [101, 114]
      }
    },
    "energetic_network_embeddings": {
      "options": {}
    },
    "multivalent_molecular_bridges": {
      "options": {
        "include_supported_ions": true,
        "include_recognized_waters": true,
        "mediator_residue_names": ["NEO", "SPM"]
      }
    },
    "reactive_path_ensembles": {
      "options": {
        "endpoint_mode": "automatic_recurrent_pair",
        "source_state_ids": [],
        "sink_state_ids": []
      }
    }
  },
  "views": {
    "macromolecular_trace": {
      "enabled": true,
      "state_trajectory_exports_enabled": false
    }
  }
}
```

`perturbation_response_dynamics` and `trajectory_reweighting` inherit exact
frame identities from common PCA. DFI/DCI is restricted to the
`macromolecular_trace` project so its nodes are residue representatives rather
than an arbitrary mixture of heavy atoms. The functional-site indices refer to
the zero-based `analysis_nodes` order in the report. Review the trace mapping
before interpreting a node as a residue.

## DFI/DCI contract

For each system, the module calculates the population covariance of common-PCA
scores and maps it into the retained shared Cartesian subspace. Seeded random
unit forces are applied under linear response. The report contains the complete
target-by-source response matrix, DFI and inclusive percentile DFI, DCI to the
declared functional-site set and percentile DCI, and differences from the
reference system.

The defaults use 250 force directions and seed `20260824`. Force count, seed,
PCA dimensionality, explained-variance threshold, alignment, mapping, and
functional-site selection all require sensitivity review. Frame-pooled
covariance is not replica-level uncertainty. DCI is dynamic coupling under the
declared linear-response model; it is not a causal or mechanistic claim.

## Reweighting input

The weight file is strict JSON:

```json
{
  "weight_schema": "salsbury-frame-log-weights-v1",
  "weight_semantics": "log_unnormalized_target_over_source_probability",
  "rows": [
    {
      "system_id": "variant-a",
      "replica_id": "r1",
      "segment_id": "production",
      "source_frame_index": 0,
      "log_weight": -1.25
    }
  ]
}
```

Add `member_id` for symmetry-expanded projections. The weight identities must
match the common-PCA projections exactly: missing, duplicate, or extra rows
fail before normalization. Row order has no meaning.

Weights are normalized independently within each system with stable
log-sum-exp arithmetic. The report includes Kish and entropy effective sample
sizes, Kish ratio, relative entropy from uniform, maximum frame weight, top-one-
percent mass, and weighted common-PCA moments. The generated reliability gates
are Kish ESS at least 20, Kish ratio at least 0.05, and maximum individual
weight at most 0.25; override them only as a reviewed project decision. A gate
failure leaves execution technically complete but sets
`weighted_thermodynamic_interpretation_allowed` to false.

This module consumes weights. It does not derive MBAR free energies, bias
potentials, or maximum-entropy multipliers, and it does not silently enable the
biased/enhanced-sampling FES path.
If the module is enabled without a nonempty `weights_path`, preparation records
it as unavailable and creates no planner or scheduler task.

## Trajectory-derived allosteric network

The generated definition uses `network_source: "trajectory"` and an empty
`network_path`. It maps the declared `node_selection` to exactly one
representative atom per residue (normally protein C-alpha or nucleic-acid
C1-prime), reads the planner-selected frames, and calculates a separate contact
occupancy matrix for each system. A pair is in contact when its representative
atoms are within `contact_cutoff_angstrom`; within-chain pairs separated by
fewer than `minimum_sequence_separation` residue positions are excluded. The
defaults are 8.0 angstrom and two positions, respectively.

The campaign planner ties these physical-frame identities to the DCCM frame
allocation, so later resource reductions cannot silently make the pathway and
dependency networks use different trajectory samples. The pathway module still
recalculates its own fitted dependency matrix and never substitutes correlation
for a contact edge.

When `neighbor_correlation_factor_enabled` is true, the same selected frames
are fitted through the declared alignment selection and used to calculate a
representative-atom displacement DCCM. This dependency matrix contributes only
to the optional local dependency score. It never creates a physical edge.

The NEMO experimental fixture enables this mode in
`tutorials/nemo_zinc_finger/experimental-analysis-config.json`. Its source and
sink indices identify four fixture-specific cysteine nodes solely as a bounded
technical test; they are not a general pathway definition or scientific claim.
When source or sink nodes are absent, preparation records the method as
unavailable and does not guess biological endpoints.

## Explicit external-network override

Set `network_source` to `external_json` and provide `network_path` only when a
reviewed network must override trajectory derivation. The external file has
this form:

The physical-network file has this form:

```json
{
  "network_schema": "salsbury-residue-contact-network-v1",
  "nodes": [
    {"node_id": "A:10"},
    {"node_id": "A:11"}
  ],
  "contact_occupancy_matrix": [
    [0.0, 0.82],
    [0.82, 0.0]
  ],
  "dependency_matrix": [
    [1.0, 0.35],
    [0.35, 1.0]
  ]
}
```

The symmetric contact matrix must contain occupancies from 0 to 1. In override
mode it is the physical edge authority; a DCCM is not accepted as a substitute
for contact.
The optional symmetric dependency matrix is required when
`neighbor_correlation_factor_enabled` is true, as it is in the generated
definition. Set that option to false to run contact-path and betweenness
analysis without the local dependency score.

Edges below the declared occupancy threshold are removed. Retained occupancies
become path costs through `-ln(occupancy) + epsilon`. The module reports all
equal-shortest-path counts, a deterministic representative path, fractional
node and edge participation across tied paths, and normalized weighted
betweenness. When dependency data are supplied, it also reports the explicitly
defined contact-occupancy-weighted mean absolute dependency with local physical
neighbors and a combined score formed from separately min-max-normalized
betweenness and local dependency.

The local dependency factor is a code-defined experimental measure; equivalence
to an external SenseNet NCF implementation is not claimed. Network cutoff,
contact definition, endpoint sets, trajectory sampling, and dependency measure
all require sensitivity analysis. A connected high-ranking route does not by
itself establish allostery or mechanism.

## Multivalent molecular-bridge networks

This trajectory-native module retains a frame-level hyperedge whenever one
mediator residue simultaneously contacts at least two distinct solute
residues. Supported ions, recognized water residues, and explicitly declared
ligand or cosolvent residue names can be mediators. The native hyperedge is the
authoritative observation. A separately labeled pairwise projection expands a
bridge spanning `k` residues into all `k choose 2` residue pairs for ordinary
network tools; it does not replace the mediator identity or multiplicity.

Recognized water uses the water oxygen and a separate polar-solute cutoff. It
is therefore a geometric solvent-bridge screen. Use the water-mediated
hydrogen-bond module when donor, acceptor, and angle chemistry is required.
The report includes mediator and mediator-type occupancies, multiplicity
distributions, interchain flags, and segment-safe boundary-censored residence
events. Every observed bridge contributes to those summaries and to a compact
frame-feature matrix used by `interaction_fingerprints`. Because a detailed
nested hyperedge for every mediator-frame can become much larger than the
trajectory, the report retains a deterministic SHA-256 min-hash sample of
detailed hyperedges up to `maximum_bridge_records`. The report states the
observed and retained counts; changing that limit does not change the complete
occupancy, residence, projected-edge, or frame-feature calculations.
Subsampled runs describe consecutive selected observations, not continuous-time
lifetimes.
Every frame record retains system, replica, segment, and source-frame identity,
so bridge presence can be joined exactly to FES or clustering assignments for
a separately reviewed state-conditioned comparison. The bridge module itself
does not choose a conformational state model.

For a K-retained/K-absent aptamer comparison, use the per-system mediator
inventory and bridge summaries to distinguish:

- K-positive guanine coordination hyperedges in the retained system;
- Na or other-ion substitution in the K-absent system;
- water bridges occupying or reorganizing around the vacated channel; and
- ion- or water-mediated aptamer-thrombin interchain contacts.

A system with no configured mediator is retained as a zero-mediator result
when another comparison system supplies the configured class. A project with
no configured mediators anywhere still fails closed. Compare systems at the
replica level and repeat cutoff and atom-selection sensitivity checks. These
observations describe network reorganization; they do not estimate K binding
free energy, explain a Kd difference, or establish mechanism.

## Aligned hydration and ion-density channels

`hydration_density_channels` is a default-off coordinate analysis that
supplements RDF, bridge, and interaction-fingerprint modules. It makes the
solute whole under the declared periodic policy, aligns every selected frame
to the common reference, images each recognized water oxygen or supported ion
to the nearest solute image, and accumulates species-resolved frame occupancy
on one explicit three-dimensional grid. The report retains the full grid
contract, two-dimensional projections, exact frame identities, and pairwise
common-grid differences. Per-frame occupied voxels are written to
process-private temporary storage as sorted flat 32-bit indices and replayed
after aggregate density components are defined. This changes storage cost, not
voxel identity, occupancy, frame coverage, or frame-to-feature assignment.

`maximum_sparse_frame_voxels` bounds the compact voxel membership of any one
selected frame. The total temporary stream volume is bounded separately by
`maximum_particle_observations`; selected frames are not dropped to satisfy an
in-memory accumulation limit.

Voxels above the configured frame-occupancy threshold are grouped with
six-face connectivity. A component that reaches the bounded grid exterior and
also extends to the configured interior depth is labeled a *geometric channel
candidate*. That label does not imply diffusion, flux, permeability, free
energy, or a transport mechanism. Alignment, grid spacing and padding,
occupancy threshold, species identity, and frame sampling all require
sensitivity analysis.

The components can also enter `interaction_fingerprints` as typed water- or
ion-density features through exact frame joins. This does not replace direct
hydrogen-bond chemistry, multivalent bridge hyperedges, ion coordination, ion
shells, or RDF normalization. For K-retained/K-absent comparisons it adds the
spatial question: where persistent K, replacement-ion, or water density moves,
appears, or disappears around the aptamer and thrombin interface.

## Ensemble geometric pocket dynamics

`ensemble_pocket_dynamics` is a separate default-off trajectory module. Its
default `native_frequency_grid_v2` backend aligns the solute-heavy ensemble,
detects solvent-sized locally enclosed empty grid points in each selected
frame, and accumulates their occupancy on one reference grid. Recurrent
connected regions are defined only after all selected frames have been read.
The report contains the voxel frequency map, exact frame-to-region identities,
region volumes, lining-residue occurrence summaries, observed representative
frame identities, per-system occupancy, and pairwise occupancy differences.
This avoids assigning a persistent ID to every transient frame pocket.

The generated configuration exposes the region definition directly:

- `minimum_region_frequency_fraction` is the minimum within-system frame
  frequency for a voxel to enter recurrent-region discovery.
- `minimum_region_voxels` removes connected regions smaller than the declared
  grid size.
- `maximum_frequency_regions` is a fail-closed resource gate; regions are not
  silently discarded when it is exceeded.
- `representative_frames_per_region` controls how many observed frame
  identities are retained for structure export.
- `maximum_sparse_frame_voxels` bounds the compact frame-to-voxel accounting.

The older `native_grid_v1` backend remains available when discrete pocket
tracking is specifically requested. It selects clearance maxima, grows bounded
frame pockets, and joins them with lining-residue Jaccard overlap and a
centroid-distance gate. That identity assignment is more sensitive to pocket
splits, merges, and transient detections, so it is no longer the generated
default. The frequency-map backend follows the ensemble-map design used by
MDpocket, but its cavity detector is the package's native geometric grid screen,
not an fpocket/MDpocket alpha-sphere implementation.

These are geometric cavities or clefts, not druggability predictions. The
backend does not estimate ligand affinity, binding free energy, or opening
kinetics. Grid, clearance, angular burial, seed/growth, frequency-region, and
frame-selection settings require sensitivity tests. The legacy backend also
requires tracking-threshold sensitivity tests. Water and ions are not pocket
walls: their occupancy is retained separately by
`hydration_density_channels`, allowing an exact-frame comparison of pocket
opening with hydration or ion occupancy without conflating the definitions.

## Chemically typed interaction fingerprints

`interaction_fingerprints` is a default-off post-processing module. It reuses
complete reports from direct hydrogen-bond discovery, one-water hydrogen-bond
networks, ion coordination, ion-atmosphere shells, multivalent molecular
bridges, and aligned hydration/ion-density components. It does not reread
coordinates. Each feature retains its source module
and chemical type, while every sparse frame row retains the exact system,
replica, segment, and source-frame identity.

The join is `pairwise_complete_observations_v1`. A frame sampled by one source
but not another is explicitly missing for the latter; it is never encoded as an
absent interaction. Feature occupancies use their source module's evaluated-frame
denominator. Co-occurrence, conditional probabilities, Jaccard, and phi
correlation use only exact frame identities observed by both source modules.
The campaign planner creates a post-processing task only when the module is
explicitly enabled.

For K-retained/K-absent trajectories, this makes replacement questions directly
testable without collapsing chemical channels: for example, whether a water
bridge or Na shell appears on the same frames that a K coordination or
K-mediated residue bridge disappears. These remain descriptive frame
associations. They do not estimate K binding affinity, reweight one condition
into another, or establish causal compensation.

## Spatial interaction superfeatures

`spatial_interaction_ensembles` is a separate default-off consumer of the
fingerprint report and the original trajectories. It does not replace binary
fingerprints, RDFs, bridge hyperedges, hydration density, or persistence. It
adds the spatial question: after fitting the receptor to the declared common
reference, where is the other endpoint of an interaction found?

The dependency-free `endpoint_partner_coordinates_v1` construction currently
uses only fingerprint types that expose exact atom identities. Direct hydrogen
bonds produce donor-centered acceptor clouds and acceptor-centered donor
clouds. Ion coordination produces ion-site-centered ligand-atom clouds. Water
bridges, ion shells, multivalent bridges, and density components remain in an
explicit unsupported-feature inventory until their upstream reports expose an
unambiguous dynamic partner atom; the module never invents one.

Each point preserves its system, replica, segment, source-frame, source-feature,
and atom identity. Per-system superfeatures report centroid, covariance,
principal axes, and radius quantiles. Optional deterministic stratified-NANI
partitions are called *spatial mode candidates* only after minimum point and
frame coverage, cluster size and fraction, exact silhouette, centroid
separation, time-block recurrence, and replica-support gates pass. When a cloud
exceeds the exact-mode resource cap, mode inference is withheld rather than
estimated from a random sample; the unclustered spatial summary remains.

For K-retained/K-absent analysis, this can distinguish a coordination contact
that occurs in one compact geometry from one that occupies a broad or shifted
set of geometries, and can compare the aligned centroid and spread of the same
site across systems. It cannot by itself call a binding mode, free-energy
basin, metastable state, affinity change, kinetic state, or mechanism.

## Temporal persistence of interaction fingerprints

`interaction_persistence` is a default-off consumer of the exact-frame
fingerprint report. It measures how long each chemically typed feature remains
present across source-observed snapshots within one system, replica, and
segment. The primary definition permits no absent observation inside an event.
Configured positive gap tolerances are reported only as an intermittent-event
sensitivity and never replace that zero-gap primary result.

Frames not evaluated by a feature's source module remain missing and are never
converted to interaction-negative observations. Every source-observed series
must also have a regular physical-time interval within the declared tolerance;
otherwise that feature/segment series is labeled unavailable. Events touching
the beginning or end of a segment are left- or right-censored and are excluded
from complete-event duration summaries. A feature/system duration summary is
rankable only after its configured minimum number of complete events is met.

Durations span saved and evaluated snapshots. They are not continuous-time bond
lifetimes between frames. For K-retained/K-absent analysis, this supplements
fingerprint occupancy by distinguishing a feature that flickers frequently
from one that persists across many saved observations—for example K-shell,
replacement-ion, water-bridge, or interface features. It still does not
estimate affinity, residence kinetics at unsaved time resolution, mechanism,
or causality.

## Random-feature nonlinear Koopman sensitivity

`random_feature_koopman` is a default-off, dependency-free nonlinear kinetic
sensitivity built on the existing TICA projections. Selected linear TICA
coordinates are standardized and mapped into an isotropic-Gaussian random
Fourier dictionary. The module scans declared random-feature counts and kernel
bandwidth scales, fits a segment-safe reversible time-lagged model for every
prespecified feature-map seed, and evaluates each run with contiguous-block
held-out VAMP-E.

No candidate is selected unless both the held-out-score range and the recovered
slow-subspace similarity pass their configured gates across at least three
recorded seeds. The first prespecified seed supplies report projections only
after that all-seed gate passes; seeds are not searched for a favorable model.
The report retains every grid candidate, every seed run, pairwise subspace
similarities, deterministic bandwidth evidence, and the selected random
feature dictionary. If no candidate is stable, technical execution can still
complete while the analysis is explicitly `not_available` and no nonlinear
coordinates are promoted.

This module does not add PyTorch, SPIB, or another package. It asks whether a
bounded Gaussian-kernel approximation exposes reproducible slow coordinates
beyond linear TICA. It does not itself define discrete states, replace MSM
validation, prove metastability, or establish a mechanism. Kernel bandwidth,
feature count, lag, TICA inputs, validation blocks, and seed gates all require
sensitivity review.

## DSSR-gated duplex helical mechanics

`helical_mechanics` is also default off. Preparation discovers or accepts an
explicit `x3dna-dssr`, runs a read-only reference probe, and requires at least
one DSSR `stems` record. This stricter stem gate is used because a generic DSSR
helix can include nonduplex assemblies. The probe also discovers the installed
JSON object's actual shift, slide, rise, tilt, roll, and twist field path and
freezes six numeric queries into the project.

When any prerequisite is absent, preparation writes
`helical-mechanics-availability.json` with `availability_status:
"not_available"`, removes the command from the workflow, and creates no planner
task. When the module is enabled and the DSSR/duplex/descriptor probe passes, it
inherits the DSSR frame set, receives a planner task, and the runtime repeats
the duplex and executable-provenance checks before analysis.
Protein-only systems are reported as scientifically inapplicable before DSSR
availability is considered; a missing DSSR executable is relevant only when a
nucleic-acid system could support the method.

`prepare-comparison` applies the same gate independently to every system. An
enabled system with a verified duplex receives paired DSSR and helical-mechanics
planner tasks; an unavailable system receives a system-specific availability
report and no helical task. `--dssr-executable` may be used with either
preparation command.

For each stable DSSR step-order position and replica, the module converts the
three angular coordinates to radians, tests deterministic one- to three-state
partitions under minimum-population and silhouette gates, and estimates
`K = k_B T C_regularized^-1` separately inside each retained state. It also
reports neighboring-step joint state counts and mutual information. It refuses
to silently realign a changing number of DSSR steps across frames.

The matrix mixes angstrom translations and radian rotations, so its element
units are correspondingly mixed. It is a local harmonic fluctuation model, not
a free-energy calculation. Step identity/order, multimodality, regularization,
sampling, and replica sensitivity must be reviewed before interpretation. This
module is appropriate for duplex DNA/RNA or duplex stem regions, not a
G-quadruplex core merely because nucleic acid is present.

## Protein residue interaction-energy network embeddings

`energetic_network_embeddings` implements the Cowan/Thayer protein-only
energetic-network workflow without invoking cpptraj. It is optional and off by
default. Three parameter sources are supported without adding a Python
dependency:

- atom-order-matched Amber `.prmtop` or `.parm7` connectivity;
- CHARMM PSF connectivity plus matching `.prm`, `.par`, `.str`, or historical
  `.inp` files, with
  native NONBONDED mixing and NBFIX handling; or
- serialized OpenMM `System` XML containing exactly one standard
  `NonbondedForce`, paired with any supported explicit connectivity source.

OpenMM particle units are converted from nm and kJ/mol to angstrom and kcal/mol,
and declared exceptions are retained before the Cowan/Thayer bond-distance
exclusion is applied. Parameter offsets and `CustomNonbondedForce` expressions
fail closed because their active interaction expression cannot be inferred
safely. Raw GROMACS `.tpr` extraction is explicitly unavailable; a
GROMACS-derived system can use the OpenMM XML route when its nonbonded model is
representable by one standard `NonbondedForce`.

When the module is enabled without a complete supported source, preparation
writes `energetic-network-embeddings-availability.json`, records
`availability_status: "not_available"`, and creates no planner task. No missing
force-field parameter is inferred. Systems without protein residues are first
reported as scientifically inapplicable, rather than as missing force-field
parameters.

For a one-system CHARMM preparation, repeat the parameter flag in the same
order used to build the simulation:

```bash
salsbury-md-analysis prepare-analysis ... \
  --energetic-charmm-parameter toppar/par_all36m_prot.prm \
  --energetic-charmm-parameter toppar/stream/modified.str
```

Alternatively use `--energetic-openmm-system-xml system.xml`. Comparative
request systems use a `force_field_parameters` object with `format` equal to
`charmm_parameter_files_v1`, `openmm_system_xml_v1`, or
`gromacs_tpr_v1`, plus an ordered `files` array. The parameter files become
hashed base-system-manifest inputs. Solute coordinate-cache manifests do not
carry them because atom subsetting invalidates the original parameter ordering;
this energetic method always runs against the original base trajectories.

The atom-pair calculation matches cpptraj `pairwise`: bond-graph-derived 1-2,
1-3, and 1-4 exclusions,
direct nonperiodic `332.0522173 q_i q_j/r` electrostatics, and
`A/r^12-B/r^6` Lennard-Jones energies after the project's declared made-whole
coordinate processing. The cpptraj `cuteelec` and `cutevdw` values are retained
as provenance but do not filter energy-map values. The report records that
scaled 1-4 terms are excluded by the pairwise exclusion setup, the calculation
is nonperiodic, and the result is a residue-pair interaction-energy network—not
a unique decomposition of reciprocal-space PME or the simulation's complete
energy.

Only complete protein residues with identical residue and atom identities in
every compared topology are retained. Mutated or atom-incomplete residues,
solvent, ions, nucleic acids, and ligands are excluded. Atom-pair energies are
converted to absolute values and summed into symmetric residue pairs; no
solvent-inclusive or signed-energy extension is implemented. The module also
calculates the absolute VDW/electrostatic weight ratio and reports whether it
passes the configured Cowan/Thayer compatibility gate before interpretation.

For every selected frame, the electrostatic matrix is locally normalized as
`max(Eij/sum_i, Eij/sum_j)`, edges must be strictly above `0.003`, and a
normalized-Laplacian heat kernel is evaluated at the published default
diffusion time `t=6`. Three deterministic PCA coordinates are calculated from
each frame's heat-kernel rows. Pairwise systems are compared residue by residue
with the sum of the three one-dimensional Wasserstein distances, matching the
published supplementary workflow. That PCA/Wasserstein construction is
descriptive: near-degenerate axes and marginal rather than joint distances are
explicit limitations, and the values are not p-values, affinities, causal
pathways, or proof of allostery.

The planner treats this as a direct, quadratic-in-selected-atoms trajectory
estimator and applies hard atom-pair, pair-frame, kernel-element, and chunk-size
gates. Available runs receive a task only when the module is enabled. Completed
pairwise residue rankings feed the finding selector as experimental
descriptive results. The separate interactive companion can display the
completed core reports.

## Reactive-path ensembles from ordinary MD

This conformational-view module reuses the complete KMeans assignment table:
its state IDs, exact system/replica/segment/member/frame identities, physical
time, and input feature coordinates. It does not read or refit the trajectory
coordinates. In the default `automatic_recurrent_pair` mode, it inventories
every unordered pair of KMeans states, counts complete paths in both
directions, and selects a pair by the declared bidirectional recurrence gate,
then the smaller directional count, total path count, and centroid separation.
The selected labels have no inferred biological meaning and are not assumed to
be metastable.

Set `endpoint_mode` to `explicit_state_sets` to supply more than one source
state and/or more than one sink state. The two sets must be nonempty and
disjoint. In either mode, a reactive path begins at the last observation in
the source set before departure and ends at the first subsequent observation
in the sink set. Paths never cross system, replica, segment, or symmetry-member
boundaries.

Within each direction, paths are compared in selected, globally standardized
feature coordinates with multidimensional dynamic time warping and a declared
Sakoe-Chiba window. Bounded average-linkage clustering chooses a route-family
count by precomputed-distance silhouette and reports the observed medoid path
for each family. `maximum_paths_per_direction`, `maximum_path_frames`, and
`maximum_pairwise_dtw_cells` are hard reporting/computation gates and are
included in the campaign planner.

For shared-view comparisons, automatic recurrence and sufficiency are evaluated
in every system rather than after pooling conditions. Route clusters include
per-system path counts, so a K-retained/K-absent analysis can ask whether the
two systems populate the same observed route families. One trajectory from
each condition never becomes two replicas per condition.

The report always states one of four transition-sufficiency outcomes:
`not_observed`, `observed_but_insufficient`, `pathway_comparison_ready`, or
`kinetics_estimation_ready`. Per-system event-count, bidirectionality,
physical-replica, retained-path, and completed-DTW gates control the first
comparison threshold. The final readiness label additionally requires stricter
per-system counts and, by default, a validated KMeans Markov-state model. It is
only a prerequisite label: this module does not estimate a rate, committor, or
flux. Route clusters remain descriptive and ordinary MD cannot reveal a route
that it never sampled.

## Future work only: TS-DAR

TS-DAR transition-state candidate detection is intentionally not implemented,
registered, or included in the planner. It would require a time-lagged learned
state model, seed and feature sensitivity analysis, candidate export, and a
separately budgeted shooting-simulation/committor validation workflow. Keep it
out of routine analysis until a transition-rich benchmark and an engine-aware
validation design are available.
