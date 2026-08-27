# Command-line reference

> Generated from the actual argument parser.

```text
usage: salsbury-md-analysis [-h] [--version]
                            {list-modules,validate-manifest,inventory-system,preflight-system,map-common-atoms,compile-context,structural-qc,rmsd-rg,rmsf,dccm,information-correlation,individual-pca,common-pca,tica,pca-fes-basins,cluster-kmeans,representative-frames,cluster-imwkmeans,cluster-hdbscan,markov-models,dihedrals,hydrogen-bonds,observables,sasa,convergence,grouped-ml,integrate,integrate-comparison-results,secondary-structure,alternative-clustering,pald-community,information-dynamics,correlation-networks,trajectory-features,state-coordinate-exports,rdf,scalar-distributions,scalar-threshold-states,hydrogen-bond-discovery,water-mediated-hydrogen-bonds,grouped-regularized-classification,nucleic-acid-structure,nucleic-acid-geometry,ion-geometry,ion-atmosphere,compare-hydrogen-bonds,hydrogen-bond-patterns,representative-structures,rmsf-permutation,plan-frame-resources,plan-campaign-resources,plan-automatic-sampling,prepare-analysis,prepare-comparison,run-local-workflow,report-plan-matrix,write-planning-report,rmsf-permutation-from-report,advise-slurm-capacity,build-coordinate-cache,prepare-unwrapped-cache,write-scientific-minimums-template,summarize-timeseries,run-instrumented,run-coordinate-cache-instrumented,summarize-execution-resources,build-resource-calibration-catalog,prioritize-findings,export-rmsf-visualization,run-regression}
                            ...

positional arguments:
  {list-modules,validate-manifest,inventory-system,preflight-system,map-common-atoms,compile-context,structural-qc,rmsd-rg,rmsf,dccm,information-correlation,individual-pca,common-pca,tica,pca-fes-basins,cluster-kmeans,representative-frames,cluster-imwkmeans,cluster-hdbscan,markov-models,dihedrals,hydrogen-bonds,observables,sasa,convergence,grouped-ml,integrate,integrate-comparison-results,secondary-structure,alternative-clustering,pald-community,information-dynamics,correlation-networks,trajectory-features,state-coordinate-exports,rdf,scalar-distributions,scalar-threshold-states,hydrogen-bond-discovery,water-mediated-hydrogen-bonds,grouped-regularized-classification,nucleic-acid-structure,nucleic-acid-geometry,ion-geometry,ion-atmosphere,compare-hydrogen-bonds,hydrogen-bond-patterns,representative-structures,rmsf-permutation,plan-frame-resources,plan-campaign-resources,plan-automatic-sampling,prepare-analysis,prepare-comparison,run-local-workflow,report-plan-matrix,write-planning-report,rmsf-permutation-from-report,advise-slurm-capacity,build-coordinate-cache,prepare-unwrapped-cache,write-scientific-minimums-template,summarize-timeseries,run-instrumented,run-coordinate-cache-instrumented,summarize-execution-resources,build-resource-calibration-catalog,prioritize-findings,export-rmsf-visualization,run-regression}
    list-modules        List registered analyses and honest implementation
                        status.
    validate-manifest   Validate a project, system, output, or publication-
                        lock manifest.
    inventory-system    Inventory files named by a system manifest without
                        modifying them.
    preflight-system    Inspect supported topology/trajectory metadata and
                        segment consistency read-only.
    map-common-atoms    Create a deterministic common PDB/GRO atom map with
                        explicit coverage gates.
    compile-context     Compile explicit units, selections, and system
                        identities read-only.
    structural-qc       Stream supported coordinates and apply explicit
                        initial integrity gates.
    rmsd-rg             Fit declared selections and report replica-resolved
                        RMSD and radius of gyration.
    rmsf                Report frame-pooled, replica, and time-block atomic
                        RMSF estimates.
    dccm                Calculate common-basis replica and system dynamic
                        cross-correlation matrices.
    information-correlation
                        Estimate nonlinear mutual-information dependence
                        between declared features.
    individual-pca      Fit an independent Cartesian PCA basis for each
                        declared replica.
    common-pca          Fit one global common-atom Cartesian PCA basis across
                        replicas.
    tica                Fit segment-safe reversible TICA to declared common-
                        PCA features.
    pca-fes-basins      Build a mode-aware PCA landscape and deterministic
                        occupancy basins.
    cluster-kmeans      Scan a seeded KMeans grid over declared common-PCA
                        features.
    representative-frames
                        Select deterministic observed representatives for
                        clusters or PCA basins.
    cluster-imwkmeans   Scan a deterministic intelligent Minkowski weighted
                        KMeans grid.
    cluster-hdbscan     Run an optional reference-HDBSCAN parameter
                        sensitivity scan.
    markov-models       Build segment-safe transition models and lag/CK
                        validation diagnostics.
    dihedrals           Calculate declared backbone and chi1 circular
                        distributions.
    hydrogen-bonds      Evaluate explicitly indexed hydrogen-bond occupancy
                        features.
    observables         Evaluate question-linked explicit distance and contact
                        features.
    sasa                Calculate deterministic Shrake-Rupley solvent-
                        accessible surface area.
    convergence         Evaluate block, ESS, split-mean, and optional
                        exploratory replica diagnostics.
    grouped-ml          Run leakage-resistant grouped decision-tree
                        validation.
    integrate           Assemble prespecified module values without hidden
                        aggregation.
    integrate-comparison-results
                        Review and integrate every completed report in a
                        prepared comparative campaign.
    secondary-structure
                        Run the external mkdssp adapter with executable
                        provenance.
    alternative-clustering
                        Run distinctly labeled clustering families on common-
                        PCA features.
    pald-community      Calculate sampled PaLD cohesion, local depth, strong
                        ties, and communities.
    information-dynamics
                        Calculate segment-safe transfer entropy and higher-
                        order feature statistics.
    correlation-networks
                        Build thresholded signed networks from DCCM outputs.
    trajectory-features
                        Extract Cartesian, COM, distance, fluctuation, and
                        principal-axis features.
    state-coordinate-exports
                        Write immutable state trajectories and observed
                        representative structures.
    rdf                 Calculate periodic, volume-normalized radial
                        distribution functions.
    scalar-distributions
                        Build Scott/FD/Rice scalar histograms and segment-safe
                        residence runs.
    scalar-threshold-states
                        Build threshold-sensitive scalar states, transitions,
                        and residence runs.
    hydrogen-bond-discovery
                        Discover direct hydrogen bonds from topology-backed
                        chemistry.
    water-mediated-hydrogen-bonds
                        Discover scalable one-water hydrogen-bond networks.
    grouped-regularized-classification
                        Run nested grouped classification on hydrogen-bond
                        patterns.
    nucleic-acid-structure
                        Run the external x3dna-dssr JSON motif adapter.
    nucleic-acid-geometry
                        Calculate intrinsic ring, fused-fold, and base-
                        stacking geometry.
    ion-geometry        Calculate bound-ion coordination and ion-pair
                        geometry.
    ion-atmosphere      Calculate species-resolved ion atmospheres around
                        solute groups.
    compare-hydrogen-bonds
                        Compare two sparse discovery reports after grouping
                        equivalent donor hydrogens.
    hydrogen-bond-patterns
                        Cluster explicit frame-level bond patterns using
                        Jaccard distance.
    representative-structures
                        Select average, closest, medoid, and central
                        structures from aligned coordinates.
    rmsf-permutation    Run unit-level exact or seeded RMSF permutation
                        inference.
    plan-frame-resources
                        Estimate all-frame feasibility or balanced subsampling
                        from retained pilots.
    plan-campaign-resources
                        Plan sampling from a CPU, memory, and time envelope.
    plan-automatic-sampling
                        Inspect a system manifest, estimate per-method wall
                        time, and assign method-, size-, trajectory-, and
                        time-aware balanced sampling.
    prepare-analysis    Create a validated, time-budgeted local or Slurm
                        analysis from one PDB, supplied or explicitly
                        requested OpenMM-derived connectivity, and one or more
                        replica DCD trajectories.
    prepare-comparison  Create one shared-basis, common-grid analysis for two
                        or more systems declared in a salsbury-comparative-
                        analysis-input-v1 request.
    run-local-workflow  Execute a prepared workflow without Slurm while
                        enforcing its CPU cap and dependency order.
    report-plan-matrix  Combine prepared planning-report.json files into an
                        analysis-family by resource-envelope stride table.
    write-planning-report
                        Regenerate the user-facing report for an existing
                        prepared campaign.
    rmsf-permutation-from-report
                        Compare per-replica RMSF profiles between systems
                        using the prepared comparison policy.
    advise-slurm-capacity
                        Optionally inspect a prepared campaign and the live
                        Slurm queue without submitting or changing any job.
    build-coordinate-cache
                        Write an atomic made-whole, unaligned molecular-
                        payload DCD cache for non-water trajectory analyses.
    prepare-unwrapped-cache
                        Continuously unwrap every source frame once and write
                        a reusable lossless molecular-payload cache.
    write-scientific-minimums-template
                        Write the editable per-replica, pooled-per-system, and
                        ordered-method time-gap minima used by campaign
                        planning.
    summarize-timeseries
                        Apply Scott-histogram-first reporting to generic non-
                        RMSD scalar series.
    run-instrumented    Run one project analysis and attach measured CPU,
                        wall, memory, host, and Slurm evidence.
    run-coordinate-cache-instrumented
                        Build an atomic coordinate cache and attach measured
                        CPU, wall, memory, host, Slurm, and exact frame
                        evidence.
    summarize-execution-resources
                        Write consolidated CSV, JSON, and Markdown
                        resource/frame tables.
    build-resource-calibration-catalog
                        Build hash-bound measured CPU, memory, and frame-
                        coverage planner evidence.
    prioritize-findings
                        Rank transparent single- and multi-system findings
                        without an opaque score.
    export-rmsf-visualization
                        Export RMSF as PDB B factors and a VMD NewCartoon/Beta
                        script.
    run-regression      Run a hash-pinned project regression without changing
                        project data.

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
```

Inspection and experimental analysis commands are read-only and emit reports to standard output. Experimental does not mean production-supported.
