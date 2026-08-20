# Scientific validation scope

Validation snapshot: **2026-08-12**

The current validation substantially strengthens confidence in the numerical
kernels, file readers, periodic reconstruction, and reporting contracts. It
does not make the toolkit a supported scientific release and does not validate
a biological conclusion.

## What was exercised

A retained private control dataset contains three independent trajectory
replicas, 2,470 source frames per replica spanning 3.01–249.91 ns, 7,418 atoms,
and explicit 7,522-bond connectivity. Seventy-five long-range frames were used
for the expanded cross-method analysis. The source coordinates and private
storage paths are not part of the public distribution.

All 26 executable project runners in the standard validation harness completed
without a failure or crash. Two supplemental project runners exercised the new
intrinsic nucleic-acid and generic bound-ion geometry on the same long-range
frames. The remaining registry capabilities are contract/preflight or
reusable-kernel layers; the 38-module snapshot is mapped to at least one of 25
scoped validation cases. All 25 cases passed their predeclared numerical or
structural tolerances. A refreshed four-case comparison now validates automatic
topology-template hydrogen-bond discovery on the same 75 real frames, raising
the independent case total to 29. The two other later additions—scalar
threshold states and grouped regularized classification—currently have focused
contract and synthetic numerical tests but have not yet been added to a
refreshed long-trajectory independent reference comparison. Those additions
brought the non-water registry to 42 modules. The water-mediated module brought
the registry to 43 and has synthetic chemistry,
periodic-neighbor, bridge, water-exchange, residence,
and cutoff-sensitivity tests. A bounded ten-frame real solvated-trajectory
smoke test also completed, but it has not yet passed an independent reference
comparison and is not included in the 29-case claim. A second bounded
all-replica smoke test completed on the hash-pinned published-Miller TREX1
8-oxoG protein-DNA-Mg lesion package. It exercised a uniform per-replica frame
budget over all three aggregate trajectories, but is likewise not an
independent comparison or part of the 29-case claim.
The separate PaLD local-depth and strong-tie community module brings the current
registry to 44. It has focused synthetic contract and numerical tests only; no
real-trajectory scientific validation has yet passed.
The species-resolved ion-atmosphere module brings the current registry to 45.
It has focused numerical tests plus bounded software-execution evidence for K,
Mg, Zn, Na, Ca, Cl, and Fe. That evidence validates species routing, frame
accounting, and resource instrumentation only; it is not an independent
scientific reference comparison, does not infer iron oxidation state, and does
not classify an ion as biologically bound without the declared structural and
residence evidence.
After the balanced planner was shared across other expensive modules, a
read-only planner check against those same three aggregate trajectories again
counted 1,000 frames per replica and selected 100 per replica with source
endpoints 0 and 999. This verifies refactor coverage accounting only; no
scientific estimator or reference comparison was rerun.

## Authoritative 30,000-frame TREX validation trajectory

The project owner has designated the current published-Miller TREX1 lesion set
as the authoritative scientific trajectory for toolkit validation. It contains
three 10,000-frame replicas of the 85,199-atom solvated
protein-DNA-modified-DNA-Mg system. This authority decision supersedes the
earlier scaling-only label for this exact trajectory lineage. The generic
planner still does not infer authority from file presence or successful
execution; project authority remains an explicit owner decision.

Read-only one-CPU Apollo runs have now completed all 30,000 pooled frames for
DCCM, common PCA basis fitting and projection, automatic direct hydrogen-bond
discovery, RMSD/Rg, pooled RMSF, trajectory features, protein/DNA dihedrals,
intrinsic nucleic-acid geometry, ion geometry, and RDF. The completed common
PCA run used all 30,000 frames for both basis fitting and projection and took
5:42:16 with 1,658,588 KiB peak resident memory. DCCM completed in 2:45:34;
automatic hydrogen-bond discovery completed in 3:10:56 and evaluated 64,640
candidates per frame. These measurements justify 30,000 pooled starting
ceilings for those exact workload classes rather than automatically reducing
them to 10,000.

This is execution and scaling evidence on an authoritative input, not yet an
independent scientific comparison of every 30,000-frame result. The benchmark
reports therefore correctly retain `scientific_status: not evaluated`.
Full-frame structural QC, 960-point SASA, and one-water-network jobs remained
active at this snapshot; an earlier full-water attempt was cancelled and a
separate immediate configuration failure was retained rather than hidden.
Expensive-method ceilings remain conservative until the active jobs and
method-appropriate independent comparisons are complete.

Independent references included MDAnalysis 2.10.0, MDTraj 1.11.1, SciPy 1.18.0,
scikit-learn 1.9.0, NetworkX 3.6.1, and statsmodels 0.14.6. Comparisons covered:

- DCD coordinates and cell lengths, topology identities, named selections, and
  connectivity-aware make-whole reconstruction;
- RMSD, radius of gyration, RMSF, DCCM, common and individual PCA, TICA, and
  PCA free-energy-grid calculations;
- mutual information, transfer entropy, lagged correlation, coskewness,
  autocorrelation, and exact RMSF permutation probabilities;
- KMeans metrics, legacy clustering families, representative selection,
  transition counting, stationary distributions, correlation networks, and
  Jaccard pattern distances;
- protein torsions, hydrogen-bond geometry, distance features, SASA, and DSSP;
- automatic protein/DNA donor and acceptor chemistry, complete DNA–DNA
  candidate discovery, protein–DNA candidate sampling, all nine declared
  distance/angle rules, and replica-resolved occupancies;
- individual DNA-ring planes and cyclic torsions, fused-ring/stacking plane
  relationships, bound-ion coordination and ideal-polyhedron shape, and all
  declared ion-pair distances;
- grouped held-out accounting and the integrated report's explicit
  no-composite-score contract.

## Quantitative highlights

- DCD coordinates agreed to 0 Å with MDAnalysis and within 7.63×10⁻⁶ Å with
  MDTraj on sampled frames.
- Connectivity reconstruction agreed with MDAnalysis within 2.81×10⁻⁶ Å after
  removing the arbitrary whole-component translation; reconstructed bond
  lengths agreed within 1.04×10⁻⁷ Å.
- Maximum differences were 1.66×10⁻⁷ Å for RMSD, 2.96×10⁻⁸ Å for radius of
  gyration, 4.45×10⁻⁹ Å for RMSF, and 5.25×10⁻⁹ for DCCM.
- Common-PCA eigenvalues agreed within 1.12×10⁻⁸ Å² and the minimum absolute
  component overlap was greater than 0.999999999999.
- All 56,150 compared protein torsions agreed with MDTraj within 0.000777°.
- At 960 sphere points, total SASA differed from MDTraj by 0.065% and the
  per-atom correlation was 0.999981. The 24-, 240-, and 480-point totals
  differed from 960 by 3.09%, 0.96%, and 0.48%, respectively; this motivated
  the enforced floor and resolution warning.
- DSSP populations matched exactly after explicitly mapping DSSP 4.6's PPII
  `P` code to the older comparison alphabet's coil category. The toolkit keeps
  the original `P` code in its output.
- Across 1,050 DNA ring/frame comparisons, fitted-plane RMS values agreed with
  independent NumPy calculations within 1.96×10⁻⁶ Å and torsion RMS values
  within 0.000350°. Across 450 ion-pair distances and 300 nearest-ligand
  distances, MDTraj agreement was within 4.85×10⁻⁶ Å and 4.96×10⁻⁶ Å,
  respectively; all coordination numbers matched exactly.
- Automatic chemistry classified 7,414 protein/DNA atoms from standard
  templates with no provisional assignments, excluded four Mg ions, and found
  690 donor and 764 acceptor atoms. OpenMM CHARMM36 2024 contained all 822
  donor–hydrogen bonds; all 746 N/O acceptors carried partial charge at or below
  -0.49 e. MDTraj omitted two terminal DNA O5′–HO5′ bonds that explicit
  connectivity and OpenMM both retained; no unexplained donor mismatch remained.
- The integrated DNA–DNA run evaluated 1,220 candidates over 75 frames. All
  823,500 nine-cutoff decisions and every per-replica occupancy count matched
  MDTraj exactly; maximum geometry differences were 1.05×10⁻⁵ Å and
  5.79×10⁻⁴°. An outcome-independent 1,024-candidate protein–DNA sample added
  691,200 exact cutoff comparisons, with maximum differences of 1.21×10⁻⁵ Å
  and 8.34×10⁻⁴°. The independent full 63,228-candidate sensitivity scan was
  monotonic across the 3.0/3.2/3.5 Å and 120/135/150° grid.

The DSSR adapter's JSON parser and shell-free process contract were tested, but
no installed DSSR executable was available in this validation environment.
That external-tool path is therefore not claimed as a completed real-project
comparison.

The public machine-readable snapshot is
[`validation/scientific_validation_summary.json`](../validation/scientific_validation_summary.json).
The bounded hydrogen-bond evidence is
[`validation/hydrogen_bond_discovery_cross_validation.json`](../validation/hydrogen_bond_discovery_cross_validation.json).
Its private-evidence hash permits later integrity verification without
publishing restricted trajectories or storage paths.

The separate public-safe water-network smoke summary is
[`validation/water_mediated_hydrogen_bond_smoke_summary.json`](../validation/water_mediated_hydrogen_bond_smoke_summary.json).
It records a completed stride-10 evaluation of 10 frames from a 100-frame DCD
on 85,206 atoms and 25,882 waters. The cell-list engine evaluated 15,405 nearby
endpoint–water pairs and retained 11,546 sparse paths across all nine cutoff
rules. This demonstrates bounded execution and spatial filtering only; it is
not a convergence, sampling, or independent scientific validation.

The complementary all-replica lesion evidence is
[`validation/water_mediated_hydrogen_bond_trex1_published_miller_all_replicas_smoke_summary.json`](../validation/water_mediated_hydrogen_bond_trex1_published_miller_all_replicas_smoke_summary.json).
The authenticated package inventories 30,000 raw 10-ps frames across three
replicas and 3,000 aggregate 100-ps frames. The bounded run selected ten
uniformly distributed aggregate frames from each replica and completed all
three without warnings. It authenticated protein, DNA, two 8-oxoG residues,
four Mg ions, water, and force-field-derived connectivity. The 8-oxoG residues
are now template-classified as nucleic acid; Mg is intentionally excluded from
hydrogen-bond roles and belongs to the ion-geometry contract. This is a
lineage-authenticated execution check, not validation of lesion chemistry,
water kinetics, or a mechanistic conclusion.

## Required negative and cautionary findings

The validation deliberately preserves nonpassing scientific gates. At least
one declared convergence/population-validity criterion did not pass on the
selected long-range frames. Recurring omega outliers were independently
confirmed as input or scientific findings rather than hidden as an
implementation error. Clustering and MSM estimators agreed with their scoped
references, but their states and kinetics remain sampling-, feature-, lag-,
and parameter-sensitive.

These are important successes of fail-closed reporting, not reasons to infer a
scientific result. No test here approves a project-specific selection,
threshold, state model, comparison, mechanism, or publication claim.

## Promotion boundary

Every module remains `experimental`. Promotion still requires a named methods
owner to review the definition and parameter sensitivity, a project-appropriate
hash-pinned regression on adequate sampling, documented limitations and
resource bounds, and an independent software review. The full private evidence
matrix should be retained with the group's controlled validation records.

Automatic direct discovery also has an explicit scaling boundary: it retains
the full chemically eligible donor–hydrogen–acceptor dictionary, so evaluation
work grows with donor–hydrogen pairs × acceptors × frames. The opt-in chunked
sparse representation prevents dense frame/cutoff report growth but does not
remove that evaluation cost. The water-mediated module instead uses spatial
neighbor lists and sparse observed paths, with independent gates on waters,
neighbor pairs, per-frame paths, and total sparse records. Neither path is yet
approved for arbitrary ligands, unusually named water models, multi-water
wires, or a real solvated production comparison. Water-edge pairing is
partitioned by the requested interaction scope before within-water path
construction, preventing avoidable all-solute quadratic expansion for targeted
protein-nucleic-acid questions while retaining the same geometry and eligibility
contract.
