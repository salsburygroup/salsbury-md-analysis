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
