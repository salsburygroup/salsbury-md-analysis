# Apollo node-count planning, 2026-09-02

Status: technical resource planning complete; scientific validity not evaluated.

Evidence JSON SHA-256:
`3ab9f9657edd83c1d2827b7eaee078fae58ccbe24d75a90cf7361a1ee8f03472`

The calculations use 44-core, 185-GiB Apollo nodes. Each campaign is capped at
16 nodes. Every planner-backed analysis job requests the padded Slurm limit of
168 hours (`7-00:00:00`). The shorter values below are predicted execution times
after an allocation starts. They do not include queue delay.

No simulation or analysis job was submitted.

## Recommendation

The operational score gives equal weight to three goals: retain information,
finish sooner, and request fewer nodes. Each recommended point is on the
scenario's Pareto curve. A point above the maximum useful node count is never
eligible.

| Case | Request | Peak used | Predicted runtime | Information retained | Mean floor multiple | Median floor multiple |
|---|---:|---:|---:|---:|---:|---:|
| TOP1-EdU D, harmonized | 6 nodes | 6 | 121.09 h | 100.00% of shared-contract maximum | 11.27x | 7.50x |
| TOP1-EdU T, harmonized | 3 nodes | 3 | 123.00 h | 95.15% of shared-contract maximum | 14.38x | 14.99x |
| TBA current | 3 nodes | 3 | 100.33 h | 99.47% | 29.86x | 40.00x |
| TREX current, 250 ns | 3 nodes | 3 | 75.07 h | 100.00% | 65.98x | 73.95x |
| Thrombin current, 60 replicas at 100 ns | 4 nodes | 4 | 112.64 h | 100.00% | 29.62x | 8.00x |
| TREX projected, 1 us | 5 nodes | 5 | 118.02 h | 94.36% | 232.34x | 299.40x |
| Thrombin projected, 63 replicas at 1 us | 3 nodes | 3 | 117.87 h | 88.91% | 198.93x | 2.00x |

The TOP1 rows form one paired recommendation. If both allocations start at the
same time, both components are predicted to finish in 123.00 hours. The pair
requests nine nodes in total and retains 97.57% of the combined information
available under the shared-stride contract.

The independent TOP1 sweeps were rejected: they disagreed on 22 shared sampling
groups. The accepted pair fixes all 28 groups present in both components to the
same effective raw integer stride. Fixed strides fail closed if they conflict
within a group, violate a scientific floor, or exceed a frame ceiling.

## TOP1 paired alternatives

These are the useful neighboring choices after stride harmonization.

| D nodes | T nodes | Total nodes | Time until both finish | Combined information |
|---:|---:|---:|---:|---:|
| 6 | 3 | 9 | 123.00 h | 97.57% |
| 6 | 5 | 11 | 121.09 h | 97.66% |
| 6 | 7 | 13 | 121.09 h | 100.00% |

The 6/3 pair is the balanced choice. Moving T from three to five nodes saves
1.91 hours but adds two nodes and only 0.09 percentage points of combined
information. Moving from five to seven nodes does not reduce time-to-both;
it raises combined information by 2.34 percentage points.

## Pareto node curves

Only nondominated points are shown. Each entry is `requested nodes / predicted
hours / information retained`. The complete 1-through-16 curves, including raw
planner regressions and replayed dominated points, remain in the JSON evidence.

| Case | Pareto points |
|---|---|
| TOP1-EdU D, harmonized | `6 / 121.09 / 100.00%` |
| TOP1-EdU T, harmonized | `3 / 123.00 / 95.15%`; `5 / 115.66 / 95.33%`; `7 / 106.33 / 100.00%` |
| TBA current | `1 / 115.43 / 93.78%`; `3 / 100.33 / 99.47%`; `7 / 116.08 / 99.83%`; `9 / 106.61 / 100.00%` |
| TREX current, 250 ns | `1 / 123.52 / 85.89%`; `3 / 75.07 / 100.00%` |
| Thrombin current, 60 replicas | `1 / 124.49 / 89.70%`; `2 / 118.86 / 90.03%`; `3 / 122.76 / 92.66%`; `4 / 112.64 / 100.00%` |
| TREX projected, 1 us | `1 / 120.20 / 66.33%`; `3 / 122.24 / 72.74%`; `5 / 118.02 / 94.36%`; `7 / 122.02 / 100.00%` |
| Thrombin projected, 63 replicas | `1 / 124.20 / 81.01%`; `2 / 118.37 / 83.10%`; `3 / 117.87 / 88.91%`; `4 / 119.55 / 91.52%`; `5 / 124.06 / 92.14%`; `6 / 118.92 / 93.70%`; `9 / 124.15 / 94.30%`; `11 / 111.45 / 98.45%`; `13 / 115.30 / 100.00%` |

The nonmonotone runtimes are expected. A larger node cap lets the planner add
frames or enable more concurrent work; it does not hold the amount of work
fixed. The monotonic replay envelope prevents a larger cap from losing a better
smaller-node plan.

## Information-cutoff sensitivity

Each entry is `nodes / predicted hours`. The retained fraction can exceed the
named cutoff because the curve is discrete.

| Case | 75% | 80% | 90% | 95% | 99% | 100% |
|---|---|---|---|---|---|---|
| TOP1-EdU pair | `6+3 / 123.00` | `6+3 / 123.00` | `6+3 / 123.00` | `6+3 / 123.00` | `6+7 / 121.09` | `6+7 / 121.09` |
| TBA current | `1 / 115.43` | `1 / 115.43` | `1 / 115.43` | `3 / 100.33` | `3 / 100.33` | `9 / 106.61` |
| TREX current | `1 / 123.52` | `1 / 123.52` | `3 / 75.07` | `3 / 75.07` | `3 / 75.07` | `3 / 75.07` |
| Thrombin current | `1 / 124.49` | `1 / 124.49` | `2 / 118.86` | `4 / 112.64` | `4 / 112.64` | `4 / 112.64` |
| TREX projected | `5 / 118.02` | `5 / 118.02` | `5 / 118.02` | `7 / 122.02` | `7 / 122.02` | `7 / 122.02` |
| Thrombin projected | `1 / 124.20` | `1 / 124.20` | `4 / 119.55` | `11 / 111.45` | `13 / 115.30` | `13 / 115.30` |

## How the ceiling is found

The planner starts with the full 16-node allowance. It loads the enabled task
graph, replica-worker caps, execution bundles, dependency stages, and
safety-adjusted memory requests. The exact maximum-node lane schedule is checked
against an analytical full-inventory pack. The larger result sets the useful
node ceiling. More nodes cannot expose additional work for that fixed graph.

The information score is the priority-weighted mean square root of normalized
physical-frame coverage across analysis tasks. Coordinate-cache work is not
scored. Scientific-floor multiples compare selected physical frames with each
method's registered planning minimum.

## Boundaries

Current rows use available input metadata. Future TREX and thrombin rows are
capacity projections for the stated topology size, replica count, and length.
They are not permission to submit work.

The 168-hour value is a Slurm kill limit. It is not a runtime claim, and neither
it nor the planner runtime includes queue delay. Queue delay requires separate
scheduler-history evidence.

Scientific floors are planning minima, not proof of independent sampling or
adequate sampling for a specific claim. These results do not establish
equilibration, convergence, metastability, kinetics, mechanism, binding, or
biological importance. TREX retains its earlier scientific-QC hold and cannot
support a biological conclusion without explicit human acceptance.
