# Automatic conformational views

Routine preparation classifies the reference topology before reading trajectory
outcomes. It then writes separate analysis projects for complementary
conformational questions.

For a protein–nucleic-acid complex the default views are:

1. `global_common_heavy`: all common non-hydrogen solute atoms, used as the
   primary global conformational view.
2. `chemical_interface`: a modified nucleotide and its immediate sequence
   neighbors, or all detected nucleic acid when no modification is found, plus
   complete protein residues contacting that focus within 6 Å in the reference
   structure. This is the primary local chemical view.
3. `macromolecular_trace`: protein C-alpha and nucleic-acid C1-prime atoms, used
   as a lower-dimensional sensitivity view.
4. `oligomer_member_common_heavy`, when strict topology/reference detection
   finds equivalent protein-centered members: every protein-DNA member is
   independently reconstructed and aligned to one canonical member, then
   pooled on a shared common-heavy PCA basis with `member_id` retained.

Protein-only monomers receive the global common-heavy and macromolecular-trace
views. Strict protein homooligomers also receive the member-expanded view.
Other compositions retain explicit classifications and do not invent a
protein–DNA interface or equivalence relation.

The view plan records `macromolecular_trace`, but the generated analysis config
turns that view and its state-trajectory export off by default. Either switch can
be enabled explicitly. This keeps the low-dimensional trace available as a
sensitivity analysis without making it part of the routine primary result.

Every view has its own PCA axes, FES minima, cluster labels, populations, and
representatives. Conditions may be compared on one shared basis within a view.
Assignments can be cross-tabulated across views, but component numbers or state
labels are not assumed equivalent between views. Water and recognized ions are
excluded from Cartesian conformational PCA by default and analyzed through
water/ion modules unless a stable bound-ion identity is explicitly locked.
Ligands and cofactors remain part of the global solute-heavy view. Canonical DNA
and RNA names, including `RA/RC/RG/RU`, are not mislabeled as modifications;
nonstandard nucleic-acid names remain explicit modified candidates.

The view plan is outcome-independent: it uses only reference topology,
reference coordinates, residue chemistry, and the declared distance cutoff.
The generated `conformational-views.json` records every selected residue, atom
count, PCA resource decision, basis sample, all-frame projection policy, and
the associated project manifest. After FES construction, the generated workflow
writes observed representative PDBs and, when enabled, bounded immutable basin
trajectories. Feature and alignment selections do not restrict export content:
ordinary-view exports use the configured complete-solute payload (protein,
nucleic acid, hydrogens, ligands, cofactors, and ions) without the bulk water
box. For member-expanded views, each independently aligned canonical member
payload is written separately so no apparent time series alternates between
member identities. Nearby-water export is a distinct explicit option.

For a homodimer, 30,000 physical frames yield up to 60,000 member observations.
That doubles conformational observations, not simulation time, physical frames,
or independent replicas. PCA/FES/clustering can use the pooled member rows.
TICA, information dynamics, and MSM keep each member as a distinct time series.
Uncertainty remains based on original replicas and physical-time blocks.
Continuous PCA/TICA scores report within-frame member cross-correlations;
cluster and FES assignments report within-frame state concordance and Cramer's
V. These are descriptive coupling measures, not evidence of causality or new
independent replicates.

For comparative projects with nonidentical topologies, planning must use every
declared condition reference together. A modification present in any reference
is transferred by chain, residue number, and insertion code before coordinates
are evaluated. The chemical view then uses the same modified-site neighborhood
in all conditions, unions complete reference-contacting protein residues across
conditions, and retains only atom identities common under the declared mapping
policy. A control topology lacking the modified residue name therefore cannot
silently broaden the view to all DNA, and a lesion topology cannot define a
narrow condition-specific comparison after results are known.

Comparative batches (including many variants) receive a shared member-expanded
view only when every reference can be resolved against one equivalent-member
contract under the declared common-atom policy. Failure in any condition makes
that comparative view explicitly inapplicable; it does not silently drop the
condition or change chain assignments.
