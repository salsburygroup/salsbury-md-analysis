# Observable-specific validation

A simulation can reproduce a state population while misrepresenting its
switching rate. These opt-in tools keep the observable, its conditioning rule,
its temporal behavior, and the evidence for numerical sensitivity separate.
They extend the existing convergence module; the standard campaign continues
to use its existing RMSD/Rg input unless an observable input is explicitly chosen.

Tupper's equivalence between approximation in distribution and modified weak
shadowing motivates attention to selected observables and their path
statistics. It does not prove accuracy for a particular simulation or
integrator: Paul Tupper, *The Relation between Approximation in Distribution
and Shadowing in Molecular Dynamics*, SIAM Journal on Applied Dynamical Systems
8, 734–755 (2009), [doi:10.1137/08072526X](https://doi.org/10.1137/08072526X).

## Use existing observable series

The runnable [synthetic example](../examples/observable_validation/README.md)
contains two series with identical populations and different switching rates.
It includes the input files, literal source values, checksums, and settings.

```sh
salsbury-md-analysis convergence examples/observable_validation/project.json
```

For a standalone diagnostic invocation, the project file can contain this
`definitions` object. For an existing project, add the same definition to its
complete manifest. Paths are relative to the project file.

```json
{
  "definitions": {
    "convergence_uncertainty": {
      "source_module": "observable_timeseries",
      "timeseries_file": "fast.json",
      "metrics": ["contact"],
      "block_size_frames": 10,
      "minimum_blocks": 2,
      "include_partial_final_block": false,
      "effective_sample_size_reference": 20,
      "split_mean_difference_reference_in_sd": 1,
      "lag_frames": [1, 2, 5]
    }
  }
}
```

The [input schema](../schemas/observable-timeseries.schema.json) requires:

- A definition, unit, kind, and increasing histogram edges for every observable.
  Indicators use numeric 0/1 values and the unit `dimensionless`. Scalar values
  must be finite. Use meaningful scalar components, such as sine and cosine of
  periodic angles; the scalar methods do not infer circular geometry.
- A source description and SHA256 digest. The output also hashes the actual
  time-series input file. Declared source provenance is retained; it is not
  independently authenticated against an external trajectory.
- System, replica, segment, source-frame, and physical-time identities. Every
  segment must have its declared regular frame spacing. Split gaps into separate
  segments. Static or unordered ensembles must not be given invented times.
- Explicit `frame` or `replica_equal` weighting. The latter assigns equal total
  mass to each observed replica distribution within a system, using its full
  selected-frame denominator before conditioning. Oligomer members are not
  independent replicas; reduce them to the intended per-frame observable before
  using this interface.

A conditional observable names an unconditional indicator with
`condition_observable_id`. Supply the complete series, including observations
that fail the condition. The report preserves the fraction satisfying the
condition separately from the distribution within it. Lag pairs require every
intermediate observation to satisfy the condition. No lag pair crosses a
segment boundary or an excluded interval.

`joint_observables` optionally lists pairs of observable IDs. Their joint
histograms use the declared shared edges and the intersection of their
conditions. Out-of-range probability is retained. Scalar distributions retain
both tails separately, so finite plotting bounds cannot silently remove mass.

The report includes pooled distributions and means, replica counts, conditioning
fractions, physical observation windows, lagged correlations, indicator switch
counts, and contiguous-run uncertainty diagnostics. These switch counts describe
sampled lag pairs; they are not continuous-time rate estimates. Constant series
have undefined correlation/ESS where appropriate. Short retained runs remain
visible with a reason when uncertainty is not calculable.

ESS-adjusted intervals remain exploratory within-series quantities. No pooled
confidence interval, ensemble-adequacy verdict, or kinetic-validity conclusion
is inferred. Existing report consumers receive the compatible quantitative
summary plus observable-specific evidence.

## Compare integration steps

```sh
salsbury-md-analysis compare-numerical-protocols examples/observable_validation/comparison.json
```

The [comparison schema](../schemas/numerical-comparison.schema.json) declares
reference and candidate series files, each requested statistic, its tolerance,
and bootstrap settings. Supported statistics are `mean`, `distribution`,
`conditioning_fraction`, `lagged_correlation`, and `switch_probability`.
Temporal statistics also declare `lag_frames`.

Each segment carries `simulation_protocol`. The same optional contract is
accepted at replica level in the system manifest. It records integration step
in fs, integrator, thermostat and constraint names and parameters, a Hamiltonian
SHA256, ensemble, temperature, and initial-ensemble identity. Use `none` and an
empty parameter object when a thermostat or constraint method is absent.
Thermostat parameters should include their units in the parameter names. Include
all Hamiltonian-defining inputs and thermodynamic controls in the declared
fingerprint; the code cannot establish their physical correctness.

An integration-step comparison is eligible only when the two arms change the
integration step while retaining the other declared protocol fields, observable
definitions, weighting, observation windows, and saved-frame intervals. The
code checks declared provenance. Matching metadata does not prove that simulation
inputs were correct. Missing provenance, identical integration steps, changed
thermostats, or changed saving intervals leave numerical sensitivity unassessed.

The signed difference is candidate minus reference. Histogram distribution
distance is total variation across the declared bins and both tails; its unit
is dimensionless. It cannot detect differences within a bin. A conditional
comparison concerns only its named estimand: include a separate
`conditioning_fraction` comparison when the full mixture matters.

The moving-block bootstrap samples within each original segment, holds the
observed replica set fixed, and keeps sampled blocks separate for temporal
statistics. Blocks must exceed the requested lag, and each segment must contain
at least two complete blocks. Choose block length using the relevant correlation
scale and examine its sensitivity; the default is not a universal adequacy rule.
The workload is capped at 50 million observation-statistic evaluations. The
command rejects larger requests rather than silently changing their sampling.

Intervals are exploratory percentile intervals conditional on the observed
segments and chosen block length. They do not capture unvisited states,
between-replica population uncertainty, or force-field error, and simultaneous
coverage across comparisons is not asserted. An undefined bootstrap statistic
prevents an interval from being reported.

A result is within the declared tolerance only when the observed difference and
the entire reported interval lie within it. A resolved excess is reported when
the observed difference exceeds tolerance and the interval lies entirely beyond
it. Other calculable cases are inconclusive. All results retain
`scientific_status: not evaluated`; finite-window compatibility is not a
shadowing proof, physical validation, or evidence of kinetic accuracy for other
observables.

## VAMP assessment and generalization

VAMP validation folds now use complete held-out frame blocks. Lag windows that
cross a boundary are excluded. `vamp_temporal_buffer_frames`, default zero,
excludes additional neighboring training observations. Purging prevents shared
frames; it does not establish statistical independence. Increase the buffer
when the correlation scale warrants it. Folds with fewer than two remaining
training or test pairs are not calculable.

`validation_gates.vamp_scores_available` replaces the misleading
`cross_validated_vamp` gate name. Score availability is separate from
`predictive_quality_assessment`. The optional `minimum_heldout_vamp_e` criterion
is applied to the mean held-out score at every requested lag. It has no universal
default: it must be justified for the observable space, model rank, baselines,
and intended use. Passing a declared score reference does not prove physical
validity. Existing other kinetic diagnostics remain necessary.

The `kinetic_validation_status` field can now be `not assessed` when numerical
diagnostics pass but no predictive criterion was declared. Consumers must not
treat this value as success. Old configurations still run. Purged scores,
model rankings, and kinetic statuses can change; retained older reports remain
immutable and must not be relabeled as if produced by this version.

These VAMP scores condition on supplied state assignments. They do not refit
PCA, tICA, clustering, or a learned representation inside each fold. For claims
about generalization to new replicas, use the separate
`replica_holdout.replica_pipeline_holdout` API with the **complete fitting
pipeline** in its fit callback. That callback receives only training replicas;
the score callback receives the fitted pipeline and held-out replica. It is
called afresh for every outer fold. Any hyperparameter tuning also belongs
inside training data. Caller-supplied closures must not reuse a globally fitted
representation. This API is opt-in; it does not automatically refit every
existing clustering family or turn replica agreement into an acceptance rule.

## Tests and release boundary

The regression suite checks equal populations with different switching rates,
equal conditional distributions with different mixture weights, constant
projections, conditional gaps, frame purging, temporal buffers, protocol
attribution, explicit weighting, and training-only outer fitting. It also checks
the command-line routes and input hashes.

These tests establish implementation behavior on bounded synthetic inputs.
Method-appropriate private fixtures, review of tolerance and block-length
choices, and human scientific review remain necessary before a scientific
release. The methods contribution is explicit separation of observable
estimation, temporal evidence, numerical-protocol sensitivity, and predictive
assessment. No production trajectories are rerun by installing these changes.
