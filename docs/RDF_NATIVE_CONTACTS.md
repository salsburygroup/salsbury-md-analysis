# RDF and native-contact observables

`radial_distribution_functions` replaces the legacy AMD RDF wrapper with an
explicit, replica-resolved contract. Each feature declares two atom-index groups,
the radial interval, bin width, stride, resource gate, and scientific question.
Distances use the triclinic minimum image. Each shell reports raw observed pair
counts, the uniform fixed-pair expectation accumulated with each frame's actual
cell volume, and `g(r) = observed / expected`.

Every evaluated frame must carry a valid periodic cell. The maximum radius may
not exceed half the minimum triclinic face height, avoiding double counting and
ambiguous images. The radius span must be an integer multiple of the bin width.
Replica results remain separate; no frame-pooled uncertainty is implied.

`optional_observables` also supports `native_contact_fraction`. The feature
declares explicit atoms, a reference cutoff, an observation cutoff, and a minimum
atom-index separation. Native pairs are derived once from `reference_structure`;
target atom identities must match under the declared common-atom policy. Reports
retain per-frame contact fractions, native-pair distances, pair occupancies, and
the complete cutoff definition.

Both analyses apply the project's periodic-coordinate policy. For production
molecular observables, use connectivity-aware `make_whole` or
`unwrap_continuous`; wrapped-coordinate execution remains diagnostic. RDF peaks
and native-contact occupancies are descriptive and do not alone establish
binding, stability, energetics, or mechanism.

## Nucleic-acid structure

`nucleic_acid_structure` replaces AMD's machine-specific `dssr_ensemble.py`
wrapper. It runs a declared `x3dna-dssr` executable without a shell, records its
version and JSON command contract, applies frame/timeout gates, and removes local
temporary PDB/JSON files after execution. The project explicitly names the DSSR
JSON collections to count; missing requested collections count as zero, malformed
present collections fail closed, and every frame retains its observed JSON keys.
Motif counts are summarized separately by replica. As with protein DSSP,
production use on periodic trajectories requires connectivity-aware molecular
reconstruction.
