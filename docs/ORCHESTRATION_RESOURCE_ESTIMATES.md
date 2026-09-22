# Preflight and reporting resource estimates

The campaign planner budgets the base preflight, each distinct view preflight,
and final reporting as required jobs. Comparisons also budget their separate
integrated-reporting job. These jobs appear under **Required execution overhead**
in `planning-report.md`; they do not contribute frames or information coverage.

The scheduler uses the planner's estimates and memory reservations. A job's
timeout allowance remains separate from its estimated duration. Older prepared
workflows without overhead rows retain their legacy fallback requests; prepare
a new output directory to use the new model. A new-format plan missing an
expected preflight or final-reporting row is rejected.

## Workload model

The initial model is `orchestration-workload-v1`. Its coefficients are conservative
engineering allowances informed by completed cluster records, not a fitted or
independently validated prediction interval.

| Job | Unpadded elapsed-time estimate | Working-memory estimate |
| --- | --- | --- |
| Preflight | 60 seconds + bytes read / 64 MiB/s + 2 seconds per file read + 2 seconds per replica | 0.5 GiB + a large-file allowance capped at 3.5 GiB + 12 times the largest topology/connectivity file size |
| Final resource reporting | 60 seconds + 2 seconds per planned report bundle | 1 GiB |
| Presentation and findings, when enabled | Add 120 seconds + 15 seconds per report bundle + 30 seconds per view | 1 GiB + (0.25 GiB per report bundle + 0.5 GiB per view) times the atom-size factor |
| Integrated comparison | 60 seconds + 3 seconds per planned report bundle | 1 GiB + 0.05 GiB per report bundle times the atom-size factor |

The large-file allowance is one GiB per GiB of input reads, up to 3.5 GiB.
The atom-size factor is `max(0.1, sqrt(maximum_atom_count / 85206))`.
Sequential alternative-clustering fits that publish one report count as one
bundle, rather than one report per algorithm.
Counts include repeated reads when the same file appears more than once in a
manifest: the current preflight hashes each occurrence. Planning inspects file
sizes without hashing trajectory content. Missing original files are errors.

Future cache preflights use the larger of original input bytes and an all-source
cache-size bound: 16 bytes per source atom per source frame, plus 512 bytes per
source atom per replica for metadata. The topology bound is 256 bytes per source
atom. This intentionally overestimates a stripped or strided cache; it never
assumes that a cache not yet written is empty. Reused external caches use their
actual file sizes. No preflight work is removed by changing analysis strides.

Single-core occupied time is charged conservatively as CPU budget, including I/O
wait. The configured `time_safety_factor` applies once. The planner applies named
memory uncertainty and cluster padding once. The DEAC profile retains its 1.5×
task-memory factor and separate 1-GiB-per-node reserve; the adapter does not add
another memory factor. Additional configured campaign reserves still apply.

## Evidence and limits

Completed historical single-core base preflights took 19 seconds to 3 minutes
1 second in the inspected records. View preflights took 2–19 seconds. An older
resource-summary finalizer took 22 seconds. These records support budgeting
overhead separately from the former two-hour and 30-minute timeout defaults.
They do not establish a storage-bandwidth guarantee.

Historical full reporting rebuilds ranged from minutes to over an hour and
included different code and, in some cases, interactive output. Their timings
must not be assigned directly to the narrower generated finalizer. The current
model distinguishes resource-only reporting from figure/finding generation.
Interactive-repository work and scientific state-coordinate exports are outside
this overhead model.

Peak memory, file sizes, report counts, implementation version, and enabled
reporting components should be retained from subsequent executions. In particular,
accounting values near a job's memory limit need corroboration before fitting a
memory model. These estimates do not replace the existing scientific-analysis
calibration catalog or change sampling minima. Queue delay is not predicted.
