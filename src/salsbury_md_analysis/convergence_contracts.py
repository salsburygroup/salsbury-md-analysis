"""Shared preparation/runtime contract for time-series uncertainty blocks."""

from typing import Mapping, MutableMapping, Sequence


def minimum_observations_for_blocks(
    block_size: int, minimum_blocks: int, include_partial: bool
) -> int:
    if include_partial:
        return block_size * (minimum_blocks - 1) + 1
    return block_size * minimum_blocks


def configure_convergence_blocks(
    settings: MutableMapping[str, object],
    selected_series: Sequence[Mapping[str, object]],
    *,
    explicit_block_size: bool = False,
) -> dict:
    """Choose generated blocks or validate explicit blocks without changing sampling.

    RMSD/Rg convergence evaluates each segment separately.  A replica total is
    therefore not the denominator for this contract, even when its segments
    are continuous.  Series identities are retained for precise diagnostics.
    """
    for name in ("minimum_blocks", "block_size_frames"):
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"convergence_uncertainty.{name} must be a positive integer")
    partial = settings.get("include_partial_final_block")
    if not isinstance(partial, bool):
        raise ValueError("convergence_uncertainty.include_partial_final_block must be boolean")
    if not selected_series:
        raise ValueError("convergence preparation has no selected RMSD/Rg time series")
    for row in selected_series:
        count = row.get("selected_observation_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("convergence preparation has invalid selected observation counts")
    smallest = min(selected_series, key=lambda row: row["selected_observation_count"])
    count = int(smallest["selected_observation_count"])
    blocks = int(settings["minimum_blocks"])
    if not explicit_block_size:
        maximum_block = (
            (count - 1) // (blocks - 1)
            if partial and blocks > 1 else count // blocks
        )
        settings["block_size_frames"] = max(1, min(max(1, count // 10), maximum_block))
    size = int(settings["block_size_frames"])
    required = max(2, minimum_observations_for_blocks(size, blocks, partial))
    if count < required:
        identity = "/".join(str(smallest.get(key, "unknown")) for key in (
            "system_id", "replica_id", "segment_id"
        ))
        choice = "explicit" if explicit_block_size else "generated"
        raise ValueError(
            "convergence block contract is impossible during preparation for "
            f"{identity}: {count} selected observations cannot yield "
            f"{blocks} blocks of {size} observations (minimum required {required}); "
            f"{choice} block_size_frames={size}, "
            f"include_partial_final_block={partial}. "
            "Increase the selected observations or provide a compatible block size; "
            "sampling and minimum_blocks have not been changed."
        )
    return {
        "status": "validated",
        "block_size_source": "explicit_configuration" if explicit_block_size else "final_selected_segment_counts",
        "block_size_frames": size,
        "minimum_blocks": blocks,
        "minimum_required_observations": required,
        "minimum_selected_observations_per_segment": count,
        "series_count": len(selected_series),
        "selected_series": [dict(row) for row in selected_series],
    }
