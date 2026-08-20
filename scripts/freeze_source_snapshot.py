#!/usr/bin/env python3
"""Hash and make a validation source snapshot read-only exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    source = args.source.expanduser().resolve(strict=True)
    manifest = args.manifest.expanduser().resolve(strict=False)
    if manifest.exists():
        raise FileExistsError(f"refusing to replace frozen source manifest: {manifest}")
    files = sorted(
        path for path in source.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    rows = [
        {
            "path": str(path.relative_to(source)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    payload = {
        "snapshot_schema": "salsbury-source-snapshot-v1",
        "source_root": str(source),
        "file_count": len(rows),
        "files": rows,
        "write_policy": "read_only_after_manifest",
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in files:
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    for directory in sorted(
        (path for path in source.rglob("*") if path.is_dir()), reverse=True
    ):
        directory.chmod(
            directory.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        )
    source.chmod(source.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print(json.dumps({"source": str(source), "file_count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
