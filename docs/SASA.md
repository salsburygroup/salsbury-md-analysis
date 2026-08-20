# Solvent-accessible surface area

Module ID: `solvent_accessible_surface_area`  
CLI: `salsbury-md-analysis sasa PROJECT.json`

The module implements Shrake--Rupley probe-center point sampling directly in
the suite. It does not shell out to a historical script or silently inherit MDTraj
defaults. Surface atoms, occluding atoms, probe radius, sphere-point count,
frame stride, and resource gates are all required project settings.

Areas are reported in square angstrom for every selected atom, aggregated by
topology residue identity, and summed for the complete surface selection. The
default element table uses explicit Bondi-style radii and fails on an unknown
element instead of guessing.

Periodic trajectories are accepted only under connectivity-aware `make_whole`
or `unwrap_continuous` preprocessing. Wrapped diagnostic execution is refused
because broken molecular images produce scientifically meaningless SASA.

Required definition:

```json
{
  "solvent_accessible_surface_area": {
    "surface_selection": "analysis",
    "occluder_selection": "analysis",
    "probe_radius_angstrom": 1.4,
    "sphere_point_count": 960,
    "frame_stride": 10,
    "maximum_surface_atoms": 10000,
    "maximum_observations": 10000000
  }
}
```

Sphere-point resolution, atom inclusion, protonation, radii, and disconnected
component imaging are scientific sensitivity dimensions. Technical completion
does not make a SASA difference mechanistic or statistically significant.

Project execution refuses fewer than 240 sphere points. Counts from 240 through
959 are accepted with a `SASA_RESOLUTION_SENSITIVITY_REQUIRED` warning; 960 is
the independently cross-validated setting, not a universal convergence proof.
On the retained 466-surface-atom trajectory frame, the 24-point total differed
from the 960-point result by about 3.1%, while the 960-point implementation
agreed with MDTraj within 0.065% in total and had a per-atom correlation above
0.99998. Every publication comparison must still repeat the calculation at a
higher resolution or otherwise demonstrate that its conclusion is stable to
the sphere-point setting.
