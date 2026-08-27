#!/usr/bin/env python3
"""Prepare and summarize the default-off experimental NEMO trial."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping


def _load(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _projections(pca: Mapping[str, object]) -> Iterable[Dict[str, object]]:
    systems = pca.get("systems")
    if not isinstance(systems, list):
        raise ValueError("common-PCA report contains no systems")
    for system in systems:
        if not isinstance(system, dict):
            raise ValueError("invalid common-PCA system row")
        for replica in system.get("replicas", []):
            if not isinstance(replica, dict):
                raise ValueError("invalid common-PCA replica row")
            for segment in replica.get("segments", []):
                if not isinstance(segment, dict):
                    raise ValueError("invalid common-PCA segment row")
                for projection in segment.get("projections", []):
                    if not isinstance(projection, dict):
                        raise ValueError("invalid common-PCA projection row")
                    row: Dict[str, object] = {
                        "system_id": str(system["system_id"]),
                        "replica_id": str(replica["replica_id"]),
                        "segment_id": str(segment["segment_id"]),
                        "source_frame_index": int(projection["source_frame_index"]),
                        "log_weight": 0.0,
                    }
                    if projection.get("member_id") is not None:
                        row["member_id"] = str(projection["member_id"])
                    yield row


def write_uniform_weights(root: Path) -> Dict[str, object]:
    pca_path = (
        root / "results/conformational-views/macromolecular_trace/common-pca/report.json"
    )
    rows = list(_projections(_load(pca_path)))
    if not rows:
        raise ValueError("common-PCA report contains no projection rows")
    output = root / "nemo-uniform-log-weights.json"
    payload = {
        "weight_schema": "salsbury-frame-log-weights-v1",
        "weight_semantics": "log_unnormalized_target_over_source_probability",
        "rows": rows,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return {
        "technical_status": "complete",
        "weights_path": str(output),
        "weights_sha256": _sha256(output),
        "row_count": len(rows),
        "control": "uniform log weights; exact-identity no-op reweighting control",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    print(json.dumps(write_uniform_weights(root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
