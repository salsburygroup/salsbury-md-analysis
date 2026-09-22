# Bounded direct-runtime applicability repair

## Scope and decision

Owner: Project_Salsbury_MD_Analysis_Toolkit. The canonical Obsidian note is the
owning record; this desktop stages publication in the existing durable outbox.
Methods/code live in this repository. Original data and frozen execution evidence
remain on DEAC; local supporting evidence is retained with this task.

Correct the inconsistent transfer of measured direct-module work to small systems.
Do not alter estimator algorithms, scientific sampling floors, replica reductions,
memory policy, old calibration evidence, or production MD files.

The candidate covers RMSD/Rg, RMSF, individual PCA, DCCM, and DSSP, with one replica,
423–21,478 topology atoms, and 1,000–20,000 source frames. It uses the existing
declared atom-work multiplier. Outside that envelope, unknown metadata, unsupported
methods, or censored-only evidence preserve the legacy model. This is a bounded
transfer model, not a newly fitted universal calibration.

Baseline startup terms remain separate from the transferred catalog affine
intercept. That intercept represents workload-dependent residual cost, not a
direct measurement of process startup. Timeout observations remain in the audit
record; their transferred values are estimates for the new workload, not measured
lower bounds. DSSP retains an unscaled completed per-frame wall-cost floor because
it launches an external process for every frame. Conservative runtime estimates
remain distinct from scheduler kill limits. No global safety factor is reduced.

## Premortem and acceptance gate

Assume the next deployment underestimates a large job despite a fast NEMO test.

| Failure path | Warning | Containment and falsifying evidence | Disposition |
| --- | --- | --- | --- |
| Large-system costs are scaled down | Large fallback differs from baseline | Exact fallback regressions and fresh 104,300-atom holdouts | Covered |
| NEMO tuning masquerades as validation | Timings enter coefficients or post-hoc thresholds | Freeze source hashes and predictions before fresh timings; no fitting on holdouts | Gate-strengthening |
| DSSP startup scales to zero | External process underprediction | Preserve completed per-frame wall envelope and fresh DSSP timings | Covered |
| Timing pass conceals wrong results | Incomplete reports or changed kernels | Require technical completion and identical scientific-kernel sources | Covered |
| Improvement changes scientific sampling rules | Floor, replica, or memory contracts change | Regression tests; only additional permitted coverage may result | Covered |

The most likely residual failure is an unmodeled execution or feature dimension;
the most dangerous failure is a cost reduction outside the tested envelope. False
success would be claiming general calibration accuracy from the original NEMO
diagnostic. No biological or convergence claim follows from these tests.

Merge requires all fresh runtime observations below frozen conservative estimates,
unchanged large-system fallback, unchanged estimator code, a passing full suite,
feasible NEMO planning without reduced methods/floors, and green repository CI.
Hold on any failure; do not tune to the failed holdout and relabel it independent.
The new Slurm batch has at most two concurrent CPUs and 16 GiB aggregate memory.
It reads existing source data and writes only isolated benchmark outputs.

Current phase: S4 implementation and holdout execution. Merge readiness: NOT READY
until the specified evidence is complete. Canonical publication remains pending
the designated publisher. No recurring task or Goal was created.
