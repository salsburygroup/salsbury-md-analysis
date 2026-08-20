# Nucleic-acid and bound-ion geometry

The reusable toolkit separates three questions that project scripts often mix:

- `nucleic_acid_geometry` calculates intrinsic ring and declared plane-pair
  geometry directly from coordinates;
- `nucleic_acid_structure` runs a declared x3dna-dssr executable and reads JSON
  motif or numeric descriptors; and
- `ion_coordination_geometry` handles arbitrary bound ions and ion pairs, not a
  hard-coded magnesium or TREX system.

## Intrinsic ring and stacking metrics

Every ring is an explicitly ordered cyclic list of zero-based topology atom
indices. The implementation verifies the strict atom identities across
replicas. Each evaluated frame retains:

- least-squares fitted-plane RMS and maximum absolute displacement;
- every cyclic consecutive-four-atom dihedral;
- signed departure of each torsion from its nearest planar value, 0 or 180
  degrees;
- RMS, maximum, and signed mean torsion departure;
- acute angle and centroid distance for every declared plane pair, labeled as
  either a fused-ring fold or base-stacking relationship.

Fused-ring fold, intrinsic puckering, and whole-base/stack reorientation are
not interchangeable. Plane normals are treated as unoriented, so their reported
pair angle lies between 0 and 90 degrees. Periodic centroid distances use the
exact triclinic minimum image. Production ring analysis requires declared
connectivity-aware reconstruction.

The report preserves frame identity, per-replica summaries, early-versus-late
mean shifts, declared blocks, and an independent scalar distribution for every
metric. Scott, Freedman-Diaconis, or Rice binning is applied separately per
metric. Histogram assignments and segment-boundary-censored residence runs are
retained. Constant metrics are labeled not estimable rather than given an
invented histogram.

## DSSR descriptors

The DSSR adapter invokes no shell, records the resolved executable and version,
uses temporary frame files, and preserves requested JSON collection counts.
Optional numeric queries use an explicit token path; `*` expands a list or a
deterministically ordered object. This allows a project lock to retain numeric
base-pair, base-step, groove, helical, backbone, or sugar descriptors exposed by
the installed DSSR JSON version. Every query declares whether a missing path is
skipped or fails the analysis. The exact query paths and DSSR version are part
of the output because JSON fields can differ by external-tool version.

## Bound ions and ion pairs

Each ion site declares one ion atom, candidate ligand atoms, a coordination
cutoff, and optional ideal templates. Every frame retains the bound ligand
identities and distances, coordination number, nearest candidate distance, and
ligand occupancy. Candidate atoms are unrestricted: a project may declare
protein, nucleic-acid, water, cofactor, sulfur, oxygen, nitrogen, or other
chemically justified ligands. Declared ion pairs are evaluated with the exact
triclinic minimum image.

Available rotation-free coordination templates are linear, trigonal planar,
tetrahedral, square planar, trigonal bipyramidal, square pyramidal, and
octahedral. A score is reported only when the observed coordination number
matches the template. One score is the RMS difference between sorted observed
and ideal ligand-pair angles. A complementary continuous shape score searches
all ligand labelings and the best proper rotation against the ideal unit-vector
polyhedron. Ligand-distance summaries and second-moment eigenvalues remain
separate, so angular shape is not confused with bond-length variation.

When both members of an ion pair are also declared sites, the report identifies
shared coordinating ligands and gives both ion-ligand distances and the
ion-ligand-ion bridge angle. These remain geometric descriptors, not an
electronic-structure, protonation, binding-affinity, or catalytic model.

TREX ring identities, Mg sites, ligand candidate lists, thresholds, and gates
belong in a TREX publication lock. Thrombin ion identities and coordination
questions belong in a thrombin project lock. Neither is embedded in the public
implementation.
