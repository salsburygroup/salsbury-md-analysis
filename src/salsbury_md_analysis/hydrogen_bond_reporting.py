"""Frame-weighted occupancy accounting shared by full and compact reports."""
from collections import Counter


def occupancy_accounting(report):
    """Require a complete denominator, including frames with no observed bond.

    Sparse occupancy rows cannot establish missing zero-event segments. Legacy
    reports without a frame ledger or explicit system totals therefore abstain.
    """
    totals = Counter()
    ledger = {}
    frames = report.get("frame_bond_matrix")
    source = "frame_bond_matrix"
    if isinstance(frames, list):
        for frame in frames:
            key = tuple(str(frame.get(field, "")) for field in
                        ("system_id", "replica_id", "segment_id"))
            ledger[key] = ledger.get(key, 0) + 1
            totals[key[0]] += 1
    else:
        source = "evaluated_frame_count_by_system"
        declared = report.get("evaluated_frame_count_by_system")
        if declared is None:
            declared = report.get("observation_accounting", {}).get(
                "selected_physical_frame_count_by_system")
        if not isinstance(declared, dict):
            return {"status": "not_estimable", "reason":
                    "Complete evaluated-frame denominators are absent; sparse event rows cannot count zero-event segments.",
                    "counts": {}, "totals": {}, "segment_frame_counts": []}
        for system, count in declared.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("hydrogen-bond evaluated-frame count must be a nonnegative integer")
            totals[str(system)] = count
        for row in report.get("segment_frame_counts", []):
            key = tuple(str(row.get(field, "")) for field in
                        ("system_id", "replica_id", "segment_id"))
            if key in ledger:
                raise ValueError("duplicate hydrogen-bond segment denominator")
            count = row.get("evaluated_frame_count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("invalid hydrogen-bond segment denominator")
            ledger[key] = count
        if ledger:
            checked = Counter()
            for key, count in ledger.items():
                checked[key[0]] += count
            if dict(checked) != dict(totals):
                raise ValueError("hydrogen-bond segment and system denominators disagree")
    counts = {}
    seen = set()
    for row in report.get("occupancies", []):
        system = str(row.get("system_id"))
        bond = row.get("bond_id")
        key = tuple(str(row.get(field, "")) for field in
                    ("system_id", "replica_id", "segment_id"))
        evaluated, present = row.get("evaluated_frame_count"), row.get("present_frame_count")
        if (not isinstance(bond, str)
                or any(isinstance(v, bool) or not isinstance(v, int) for v in (evaluated, present))
                or evaluated < 1 or not 0 <= present <= evaluated):
            raise ValueError("invalid hydrogen-bond occupancy counts")
        if (key, bond) in seen:
            raise ValueError("duplicate hydrogen-bond segment/bond occupancy")
        seen.add((key, bond))
        if ledger and (key not in ledger or evaluated != ledger[key]):
            raise ValueError("hydrogen-bond occupancy denominator differs from frame ledger")
        if system not in totals or evaluated > totals[system]:
            raise ValueError("hydrogen-bond occupancy exceeds evaluated-frame coverage")
        values = counts.setdefault(system, {})
        values[bond] = values.get(bond, 0) + present
        if values[bond] > totals[system]:
            raise ValueError("pooled hydrogen-bond occupancy is outside [0, 1]")
    return {"status": "complete", "denominator_source": source,
            "counts": counts, "totals": dict(totals), "segment_frame_counts": [
                dict(zip(("system_id", "replica_id", "segment_id"), key),
                     evaluated_frame_count=count) for key, count in sorted(ledger.items())]}
