# Methods and citations

This page maps every registered analysis module to the method it implements and
to the reference that should accompany a scientific use of that method. Cite
the Salsbury MD Analysis version and commit in every use. Cite the listed
method papers for the modules used in a result, plus the simulation software,
force field, system preparation, and experimental sources that belong to the
study itself.

The references explain the method family. The executable contract in this
repository controls the actual calculation: selections, units, frame indices,
periodic treatment, estimators, random seeds, cutoffs, and failure gates are
recorded in each report. A cited method name does not make an experimental
module scientifically accepted.

## Core data and coordinate methods

| Registered modules | Implemented method and citation |
| --- | --- |
| `provenance_manifest`, `preflight_inventory`, `common_atom_mapping` | Repository-defined manifest, input-inventory, and atom-identity contracts. Cite the software and the file-format or simulation-engine documentation used to create the inputs. These modules do not claim a separate published estimator. |
| `structural_integrity_qc` | Repository-defined coordinate, topology, chirality, peptide-link, displacement, extent, and clash checks. The DCD reader follows the documented CHARMM/NAMD record layout. Thresholds are configuration, not literature-derived scientific acceptance criteria. |
| `replica_rmsd_rg` | Least-squares rigid-body alignment uses Horn's unit-quaternion solution: Horn, *JOSA A* 4, 629-642 (1987), [doi:10.1364/JOSAA.4.000629](https://doi.org/10.1364/JOSAA.4.000629). RMSD and mass-weighted radius of gyration are reported as defined in [RMSD and radius of gyration](RMSD_RG.md). |
| `pooled_rmsf` | Root-mean-square fluctuation after declared rigid-body fitting, with frame-pooled, replica-balanced, and block summaries kept distinct. The uncertainty rules are tied to the block and resampling references below, not to independent-frame assumptions. |
| `dccm` | Cartesian displacement cross-correlation follows the covariance analysis described by Ichiye and Karplus, *Proteins* 11, 205-217 (1991), [doi:10.1002/prot.340110305](https://doi.org/10.1002/prot.340110305). |
| `generalized_correlation_and_information` | Histogram mutual information uses Shannon entropy in nats and a scalar generalized-correlation transform. See Shannon, *Bell System Technical Journal* 27, 379-423 (1948), [doi:10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x), and Lange and Grubmuller, *Proteins* 62, 1053-1061 (2006), [doi:10.1002/prot.20784](https://doi.org/10.1002/prot.20784). The implementation uses quantile histograms, not a nearest-neighbor mutual-information estimator. |
| `information_dynamics` | Lagged mutual information and transfer-entropy-style directional summaries retain segment boundaries. For transfer entropy, see Schreiber, *Physical Review Letters* 85, 461-464 (2000), [doi:10.1103/PhysRevLett.85.461](https://doi.org/10.1103/PhysRevLett.85.461). Directional dependence is not proof of causality. |
| `correlation_networks` | Graphs are built from explicitly thresholded correlation or information matrices. Community labels and centralities inherit the source matrix and threshold; cite the matrix method above and Newman, *PNAS* 103, 8577-8582 (2006), [doi:10.1073/pnas.0601602103](https://doi.org/10.1073/pnas.0601602103), when modularity communities are reported. |

## Dimensional reduction, landscapes, and states

| Registered modules | Implemented method and citation |
| --- | --- |
| `individual_pca`, `common_pca` | Cartesian essential-dynamics PCA follows Amadei, Linssen, and Berendsen, *Proteins* 17, 412-425 (1993), [doi:10.1002/prot.340170408](https://doi.org/10.1002/prot.340170408). The bounded leading-subspace path follows randomized range-finding principles from Halko, Martinsson, and Tropp, *SIAM Review* 53, 217-288 (2011), [doi:10.1137/090771806](https://doi.org/10.1137/090771806). |
| `time_lagged_independent_component_analysis` | Reversible, covariance-whitened tICA uses segment-local lag pairs. See Perez-Hernandez et al., *Journal of Chemical Physics* 139, 015102 (2013), [doi:10.1063/1.4811489](https://doi.org/10.1063/1.4811489). |
| `random_feature_koopman` | Seeded Gaussian random Fourier dictionaries follow Rahimi and Recht, *Advances in Neural Information Processing Systems* 20 (2007), [paper](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines). The reversible slow-mode fit inherits the tICA contract above. The implementation requires cross-seed slow-subspace agreement; the random-feature approximation is not evidence that the chosen kernel is physically suitable. |
| `pca_fes_basins`, `scalar_feature_distributions`, `scalar_threshold_states` | Histogram choices include Scott and Freedman-Diaconis rules: Scott, *Biometrika* 66, 605-610 (1979), [doi:10.1093/biomet/66.3.605](https://doi.org/10.1093/biomet/66.3.605); Freedman and Diaconis, *Zeitschrift fur Wahrscheinlichkeitstheorie* 57, 453-476 (1981), [doi:10.1007/BF01025868](https://doi.org/10.1007/BF01025868). Relative free energy is emitted only for declared unweighted, unbiased MD. Basin catchments, threshold states, and censored residence runs follow the versioned repository definitions. |
| `trajectory_features` | Repository-defined extraction of named coordinate, distance, angle, dihedral, and upstream-report features with full frame identity. Cite the scientific definition for every selected feature. |
| `clustering_kmeans` | Lloyd iteration with k-means++ seeding. Cite Lloyd, *IEEE Transactions on Information Theory* 28, 129-137 (1982), [doi:10.1109/TIT.1982.1056489](https://doi.org/10.1109/TIT.1982.1056489), and Arthur and Vassilvitskii, SODA 2007, [doi:10.5555/1283383.1283494](https://doi.org/10.5555/1283383.1283494). |
| `clustering_hdbscan` | The optional `hdbscan` package implements hierarchical density clustering. Cite Campello, Moulavi, and Sander, PAKDD 2013, [doi:10.1007/978-3-642-37456-2_14](https://doi.org/10.1007/978-3-642-37456-2_14), and McInnes, Healy, and Astels, *JOSS* 2, 205 (2017), [doi:10.21105/joss.00205](https://doi.org/10.21105/joss.00205). |
| `clustering_imwkmeans` | Minkowski-weighted k-means with cluster-specific feature weights and deterministic initialization. Cite de Amorim and Mirkin, *Pattern Recognition* 45, 1061-1075 (2012), [doi:10.1016/j.patcog.2011.08.012](https://doi.org/10.1016/j.patcog.2011.08.012). |
| `alternative_clustering` | Each result keeps its algorithm name. Cite Kaufman and Rousseeuw, *Finding Groups in Data* (1990), [doi:10.1002/9780470316801](https://doi.org/10.1002/9780470316801), for PAM; Ward, *JASA* 58, 236-244 (1963), [doi:10.1080/01621459.1963.10500845](https://doi.org/10.1080/01621459.1963.10500845), for Ward linkage; Dempster, Laird, and Rubin, *JRSS B* 39, 1-22 (1977), [doi:10.1111/j.2517-6161.1977.tb01600.x](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x), for EM mixture fitting; Frey and Dueck, *Science* 315, 972-976 (2007), [doi:10.1126/science.1136800](https://doi.org/10.1126/science.1136800), for affinity propagation; and Comaniciu and Meer, *IEEE TPAMI* 24, 603-619 (2002), [doi:10.1109/34.1000236](https://doi.org/10.1109/34.1000236), for mean shift. The quality-threshold and weighted-PAM variants use the definitions recorded in their reports. |
| `pald_community_analysis` | Partitioned Local Depth cohesion, local depth, strong ties, and connected communities follow Berenhaut, Moore, and Melvin, *PNAS* 119, e2003634119 (2022), [doi:10.1073/pnas.2003634119](https://doi.org/10.1073/pnas.2003634119). |
| `representative_frames`, `representative_structures`, `state_coordinate_exports` | Observed nearest-center frames, coordinate-space medoids, average structures, and immutable state exports are separate repository contracts. Cite the state-definition method that supplied the assignments. An observed representative is not an average structure or proof of a physical state. |
| `markov_state_models` | Segment-safe transition counts, reversible estimation, implied timescales, Chapman-Kolmogorov checks, and validation follow Prinz et al., *Journal of Chemical Physics* 134, 174105 (2011), [doi:10.1063/1.3565032](https://doi.org/10.1063/1.3565032). Time-blocked VAMP-E scoring follows the variational approach summarized by Wu and Noe, *Journal of Nonlinear Science* 30, 23-66 (2020), [doi:10.1007/s00332-019-09567-y](https://doi.org/10.1007/s00332-019-09567-y). Passing software checks does not establish Markovianity or valid kinetics. |
| `reactive_path_ensembles` | The module extracts segment-local source-to-sink subsequences from pre-existing discrete state assignments, then clusters complete observed paths. Transition-path theory supplies the scientific context: E and Vanden-Eijnden, *Journal of Statistical Physics* 123, 503-523 (2006), [doi:10.1007/s10955-005-9003-9](https://doi.org/10.1007/s10955-005-9003-9). This implementation does not estimate a committor, reactive flux, or rate. |

Silhouette, adjusted Rand, Calinski-Harabasz, and Davies-Bouldin values are
diagnostics, not physical-state criteria. Cite Rousseeuw, *Journal of
Computational and Applied Mathematics* 20, 53-65 (1987),
[doi:10.1016/0377-0427(87)90125-7](https://doi.org/10.1016/0377-0427(87)90125-7),
and Hubert and Arabie, *Journal of Classification* 2, 193-218 (1985),
[doi:10.1007/BF01908075](https://doi.org/10.1007/BF01908075), when those
statistics support a reported comparison.

## Molecular structure and interactions

| Registered modules | Implemented method and citation |
| --- | --- |
| `dihedral_distributions` | Explicit four-atom torsions with circular summaries and residue-name templates. Nonstandard residues require an explicit adapter. Cite the force-field or structural convention that defines the atoms used in the study. |
| `hydrogen_bonds`, `hydrogen_bond_discovery`, `hydrogen_bond_comparison`, `hydrogen_bond_patterns`, `water_mediated_hydrogen_bond_networks` | Direct and one-water-mediated hydrogen bonds use explicit donor-hydrogen-acceptor distance and angle criteria. Baker and Hubbard give the standard structural-analysis context: *Progress in Biophysics and Molecular Biology* 44, 97-179 (1984), [doi:10.1016/0079-6107(84)90007-5](https://doi.org/10.1016/0079-6107(84)90007-5). Always report the actual cutoffs, chemistry templates, periodic policy, sparse implicit-zero contract, and comparison identity rules written by the tool. |
| `grouped_ml`, `grouped_regularized_classification` | Cross-validation groups whole replicas or declared independent units. Regularized logistic models use the scikit-learn implementation. Cite Tibshirani, *JRSS B* 58, 267-288 (1996), [doi:10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x), for lasso and Zou and Hastie, *JRSS B* 67, 301-320 (2005), [doi:10.1111/j.1467-9868.2005.00503.x](https://doi.org/10.1111/j.1467-9868.2005.00503.x), for elastic net. Predictive separation is not a mechanism or an independent scientific validation. |
| `secondary_structure` | Protein assignments come from the declared `mkdssp` executable. Cite Kabsch and Sander, *Biopolymers* 22, 2577-2637 (1983), [doi:10.1002/bip.360221211](https://doi.org/10.1002/bip.360221211), and Touw et al., *Nucleic Acids Research* 43, D364-D368 (2015), [doi:10.1093/nar/gku1028](https://doi.org/10.1093/nar/gku1028). Record the executable version and treatment of the `P` assignment. |
| `nucleic_acid_structure` | Motifs and requested descriptors come from the declared 3DNA-DSSR executable. Cite Lu and Olson, *Nucleic Acids Research* 31, 5108-5121 (2003), [doi:10.1093/nar/gkg680](https://doi.org/10.1093/nar/gkg680), and Lu, Bussemaker, and Olson, *Nucleic Acids Research* 43, e142 (2015), [doi:10.1093/nar/gkv107](https://doi.org/10.1093/nar/gkv107). Record the executable version and JSON query paths. |
| `helical_mechanics` | Segment-local means, covariances, inverse-covariance stiffness matrices, and propagated bend/twist summaries are calculated from accepted DSSR duplex descriptors. Cite the 3DNA/DSSR papers above, report the descriptor and unit basis, and treat the fitted stiffness as a sampling-dependent local model rather than a material constant. |
| `nucleic_acid_geometry` | Least-squares ring planes, cyclic torsion departure, plane-pair angles, and minimum-image centroid distances use the exact definitions in [Nucleic-acid and bound-ion geometry](NUCLEIC_ACID_ION_GEOMETRY.md). Cite the chemical atom and plane definitions used by the study; do not relabel these descriptors as DSSR output. |
| `ion_coordination_geometry` | Distance-defined first-shell coordination plus rotation-free ideal-polyhedron scores. Cite Pinsky and Avnir, *Inorganic Chemistry* 37, 5575-5582 (1998), [doi:10.1021/ic9804925](https://doi.org/10.1021/ic9804925), when continuous shape scores are reported. Geometry does not establish charge transfer, affinity, or catalysis. |
| `ion_atmosphere`, `radial_distribution_functions` | Triclinic minimum-image pair distances and shell-volume-normalized radial distributions. Cite the simulation package and force field that define the periodic ensemble, and Allen and Tildesley, *Computer Simulation of Liquids*, second edition (2017), [doi:10.1093/oso/9780198803195.001.0001](https://doi.org/10.1093/oso/9780198803195.001.0001), for the statistical-mechanics definition of radial distribution functions. |
| `solvent_accessible_surface_area` | Shrake-Rupley probe-center sphere sampling with explicitly declared radii, probe, atom selections, and point count. Cite Shrake and Rupley, in *Environment and Exposure to Environmental Agents* (1973), [doi:10.1016/B978-0-12-456080-1.50008-6](https://doi.org/10.1016/B978-0-12-456080-1.50008-6), and Bondi, *Journal of Physical Chemistry* 68, 441-451 (1964), [doi:10.1021/j100785a001](https://doi.org/10.1021/j100785a001), if the default radii are used. |
| `optional_observables` | Each optional observable carries its own definition. For native-contact fractions, cite Best, Hummer, and Eaton, *PNAS* 110, 17874-17879 (2013), [doi:10.1073/pnas.1311599110](https://doi.org/10.1073/pnas.1311599110), when that contact convention is selected. |
| `perturbation_response_dynamics` | DFI/DCI profiles use covariance-based linear response with reproducible isotropic unit-force directions. Cite Gerek, Kumar, and Ozkan, *Evolutionary Applications* 6, 423-433 (2013), [doi:10.1111/eva.12052](https://doi.org/10.1111/eva.12052). The selected residue nodes, functional-site set, retained PCA subspace, and force count are part of the result definition. Coupling is not causality. |
| `trajectory_reweighting` | Exact frame-key joins, stable log-sum-exp normalization, weighted moments, and Kish and entropy effective-sample diagnostics are applied to externally supplied log weights. Cite Kish, *Survey Sampling* (1965), for the Kish effective sample size. The module neither derives the weights nor performs MBAR; passing its weight-concentration gates does not establish phase-space overlap. |
| `allosteric_pathways` | Contact occupancies form the physical graph; retained occupancies become negative-log edge costs for shortest paths, and tied-path participation plus weighted betweenness are reported. Cite Dijkstra, *Numerische Mathematik* 1, 269-271 (1959), [doi:10.1007/BF01386390](https://doi.org/10.1007/BF01386390), and Brandes, *Journal of Mathematical Sociology* 25, 163-177 (2001), [doi:10.1080/0022250X.2001.9990249](https://doi.org/10.1080/0022250X.2001.9990249). The negative-log cost and optional dependency score are repository-defined; a graph path does not establish allosteric signaling. |
| `energetic_network_embeddings` | Protein-only residue interaction-energy networks, local edge normalization, heat kernels, per-frame PCA, and residue-wise Wasserstein comparisons follow Cowan, Beveridge, and Thayer, *Journal of Physical Chemistry B* 127, 623-633 (2023), [doi:10.1021/acs.jpcb.2c06546](https://doi.org/10.1021/acs.jpcb.2c06546). The report states the supported parameter source and exclusions; it is not a full PME energy decomposition. |
| `multivalent_molecular_bridges` | Simultaneous mediator-to-solute contacts are retained as frame-level hyperedges, with a separately labeled pairwise projection and segment-safe residence records. This is a repository-defined geometric contract. Cite the relevant ion, water, ligand, force-field, and contact-definition sources for the system, and do not substitute the pairwise projection for the native hyperedge. |
| `hydration_density_channels` | Reference-aligned water and ion voxel occupancies are accumulated on a bounded grid, then thresholded into connected geometric components. Grid-based hydration analysis provides context; see Nguyen, Young, and Gilson, *Journal of Chemical Physics* 137, 044101 (2012), [doi:10.1063/1.4733951](https://doi.org/10.1063/1.4733951). This module does not calculate GIST entropies or free energies, and a boundary-reaching component is not evidence of flux. |
| `ensemble_pocket_dynamics` | The primary backend defines recurrent connected regions from reference-aligned geometric pocket-voxel frequencies. Compare Schmidtke et al., *Bioinformatics* 27, 3276-3285 (2011), [doi:10.1093/bioinformatics/btr550](https://doi.org/10.1093/bioinformatics/btr550). The package uses its own enclosure screen, not the fpocket/MDpocket alpha-sphere detector; pocket occurrence does not establish druggability or affinity. |
| `interaction_fingerprints` | Exact frame keys join accepted hydrogen-bond, ion, bridge, and density features into a sparse typed fingerprint. Source missingness remains distinct from absence; co-occurrence, conditional probability, Jaccard, and phi summaries use pairwise-complete observations. This synthesis is repository-defined, so cite every contributing interaction method rather than treating the fingerprint as new physical evidence. |
| `spatial_interaction_ensembles` | Exact partner coordinates for supported fingerprint types are receptor-aligned and summarized in three dimensions. Optional partitions use NANI initialization from Chen et al., *Journal of Chemical Theory and Computation* 20, 5583-5597 (2024), [doi:10.1021/acs.jctc.4c00308](https://doi.org/10.1021/acs.jctc.4c00308), with the recorded spatial gates. A spatial mode candidate is not a binding state or kinetic state. |
| `interaction_persistence` | Exact-frame fingerprints become segment-safe zero-gap and explicitly gap-tolerant runs with left- and right-censor labels. Censoring terminology follows Kaplan and Meier, *Journal of the American Statistical Association* 53, 457-481 (1958), [doi:10.1080/01621459.1958.10501452](https://doi.org/10.1080/01621459.1958.10501452), but the module reports run distributions rather than a Kaplan-Meier estimator. Persistence across sampled frames is not an unobserved continuous-time lifetime. |

## Uncertainty, comparison, and reporting

| Registered modules | Implemented method and citation |
| --- | --- |
| `convergence_uncertainty` | Autocorrelation sequences, initial-positive-sequence effective sample sizes, split means, and contiguous block summaries are diagnostics. For blocking, cite Flyvbjerg and Petersen, *Journal of Chemical Physics* 91, 461-466 (1989), [doi:10.1063/1.457480](https://doi.org/10.1063/1.457480). For bootstrap intervals, cite Efron, *Annals of Statistics* 7, 1-26 (1979), [doi:10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552). |
| `rmsf_permutation_inference` | Replica-label permutation uses replicas, not frames or oligomer members, as exchangeable units. False-discovery-rate adjustment follows Benjamini and Hochberg, *JRSS B* 57, 289-300 (1995), [doi:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x). With too few replicas, the report states that inference is insufficient. |
| `integrated_comparison` | Non-aggregating collation of accepted upstream reports. Cite each source method used in the final comparison. The collation step adds no independent biological or statistical evidence. |

## Software references

The package depends on NumPy and SciPy and can call scikit-learn, HDBSCAN,
OpenMM, DSSP, and DSSR when those optional paths are selected. Reports retain
the available package or executable versions. Relevant software papers are:

- Harris et al., NumPy, *Nature* 585, 357-362 (2020),
  [doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
- Virtanen et al., SciPy, *Nature Methods* 17, 261-272 (2020),
  [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
- Pedregosa et al., scikit-learn, *JMLR* 12, 2825-2830 (2011),
  [article](https://jmlr.org/papers/v12/pedregosa11a.html).
- Eastman et al., OpenMM 7, *PLoS Computational Biology* 13, e1005659 (2017),
  [doi:10.1371/journal.pcbi.1005659](https://doi.org/10.1371/journal.pcbi.1005659).
- Michaud-Agrawal et al., MDAnalysis, *Journal of Computational Chemistry* 32,
  2319-2327 (2011),
  [doi:10.1002/jcc.21787](https://doi.org/10.1002/jcc.21787). MDAnalysis was an
  independent validation reference; Salsbury MD Analysis does not use it as its
  trajectory engine.

## Salsbury-group context

These papers are relevant applications or fixture provenance, not substitutes
for the primary method references above:

- Salsbury, *Current Opinion in Pharmacology* 10, 738-744 (2010),
  [doi:10.1016/j.coph.2010.09.016](https://doi.org/10.1016/j.coph.2010.09.016),
  reviews atomistic MD in drug-discovery research.
- Negureanu and Salsbury, *Journal of Biomolecular Structure and Dynamics* 30,
  347-361 (2012),
  [doi:10.1080/07391102.2012.680034](https://doi.org/10.1080/07391102.2012.680034),
  is a protein-DNA dynamics application relevant to comparative-complex work.
- Godwin, Gmeiner, and Salsbury, *Journal of Biomolecular Structure and
  Dynamics* 34, 125-134 (2016),
  [doi:10.1080/07391102.2015.1015168](https://doi.org/10.1080/07391102.2015.1015168),
  supplies the scientific provenance and sampling warning for the NEMO tutorial
  fixture.

## Citation-review record

The DOI links and bibliographic details above were checked against primary
publisher or archival records on 2026-09-02. The requested Scite review could
not be completed: the connected WFU Scite account reported that its 250-call
monthly MCP allowance was exhausted until 2026-10-01, and the alternate
signed-in browser route was denied by an administrator-enforced browser policy.
This is an open documentation gate. Do not describe the reference list as
Scite-checked until that audit is rerun successfully.
