# Salsbury MD Analysis documentation

This toolkit turns an atom-order-matched structure, bond topology, and one or
more trajectories into a planned, auditable analysis campaign. It can run on a
workstation or through Slurm. It is designed to choose reasonable routine
analyses automatically while making its choices, frame sampling, resource use,
and limitations visible.

The toolkit is still experimental. A report marked `technical_status: complete`
means that the software finished and its output contract passed. It does not
mean that the trajectory is converged, a state is metastable, an ion is
biologically bound, or a result is ready to publish. Those conclusions still
need scientific review.

## Where to begin

- Follow the [NEMO zinc-finger tutorial](../tutorials/nemo_zinc_finger/README.md)
  for a complete workstation example using a real published-simulation subset.
- Use the [local and cluster workflow](../tutorials/local_and_cluster/README.md)
  when preparing your own system or moving the same plan to Slurm.
- Start with [general biomolecular systems](GENERAL_BIOMOLECULAR_SYSTEMS.md) to
  see which inputs and chemical compositions the automatic workflow handles.
- Read [local and Slurm execution](EXECUTION_ADAPTERS.md) before choosing a
  workstation or cluster run.
- Use [configuration and final reporting](CONFIGURATION_AND_FINAL_REPORTING.md)
  to change modules, CPU/time limits, exports, clustering, or comparisons.
- Use [scientific validation scope](SCIENTIFIC_VALIDATION.md) to understand what
  has and has not been validated.

The pages under `generated/` are produced directly from the code registry,
profiles, schemas, and command parser. They are precise references rather than
tutorial prose. Run `python scripts/generate_docs.py --check` to verify that
they still agree with the source.

## Reference and method guides

- [Module reference](generated/MODULE_REFERENCE.md)
- [Analysis coverage](generated/ANALYSIS_COVERAGE.md)
- [Standard profile](generated/PROFILE_REFERENCE.md)
- [Schema reference](generated/SCHEMA_REFERENCE.md)
- [Command-line reference](generated/CLI_REFERENCE.md)
- [General biomolecular systems](GENERAL_BIOMOLECULAR_SYSTEMS.md)
- [Legacy review closure](LEGACY_REVIEW.md)
- [Manifest and provenance workflow](MANIFESTS.md)
- [FES, clustering, and state-coordinate exports](FES_CLUSTERING_EXPORTS.md)
- [Scientific reporting standard](REPORTING_STANDARD.md)
- [Methods and citations](METHODS_AND_CITATIONS.md)
- [Prioritized findings and complete module accounting](FINDING_PICKER.md)
- [Automatic and pooled frame sampling](FRAME_SAMPLING.md)
- [Comparative states, ions, and hydrogen bonds](COMPARATIVE_STATES_INTERACTIONS.md)
- [RDF and native-contact observables](RDF_NATIVE_CONTACTS.md)
- [Nucleic-acid and bound-ion geometry](NUCLEIC_ACID_ION_GEOMETRY.md)
- [Compiled analysis context](CONTEXTS.md)
- [Topology and trajectory preflight](PREFLIGHT.md)
- [Common-atom mapping](ATOM_MAPPING.md)
- [Periodic coordinate reconstruction](PERIODIC_COORDINATES.md)
- [Reusable coordinate caches](COORDINATE_CACHE.md)
- [Replica parallelism and pooled reducers](ENSEMBLE_PARALLELISM.md)
- [Structural-integrity QC](STRUCTURAL_QC.md)
- [RMSD and radius of gyration](RMSD_RG.md)
- [RMSF and uncertainty](RMSF.md)
- [Dynamic cross-correlation](DCCM.md)
- [Information correlation](INFORMATION_CORRELATION.md)
- [Individual and common-basis PCA](PCA.md)
- [Automatic conformational views](CONFORMATIONAL_VIEWS.md)
- [TICA](TICA.md)
- [Representative frames](REPRESENTATIVE_FRAMES.md)
- [SASA](SASA.md)
- [Secondary structure](SECONDARY_STRUCTURE.md)
- [Scientific validation scope](SCIENTIFIC_VALIDATION.md)
- [Hash-pinned regressions](REGRESSIONS.md)
