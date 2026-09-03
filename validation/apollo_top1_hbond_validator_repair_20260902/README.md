# TOP1 hydrogen-bond validator repair

Jobs `8298182` and `8298183` completed their analysis calculations and wrote
technically complete reports, then failed in the external installation wrapper.
The wrapper divided an estimated campaign total evenly across replicas. That
did not match the package's complete-interval index rule.

This repair does not read trajectories or repeat either calculation. It checks
the saved report and summary hashes, validates each replica with
`integer_stride_selected_count`, checks the sparse hydrogen-bond contract, and
hard-links the accepted files into new versioned storage and the originally
planned destinations. Existing, nonidentical destinations cause a closed
failure.

Technical validation does not establish scientific validity or support a
biological interpretation.

## Post-run reconciliation

Before the validator-only repair completed, independently submitted
complete-interval jobs `8298283` (T) and `8298284` (D) installed valid reports
at the originally planned destinations. That is why validator attempt
`8298341` encountered nonidentical destination files and failed closed. It did
not overwrite them.

Both recomputation jobs completed with exit code `0:0`. Their validation and
installation gates pass, and the reports' scientific payloads match the saved
validator-only reports exactly. Full-file hashes differ because each report
records its own provenance and resource measurements. No downstream path remap
is required. The executed batch script and its output are retained unchanged
as historical evidence of what the validator-only run reported at that time.
