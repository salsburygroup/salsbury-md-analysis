"""Hash-bound NumPy columnar artifacts for large identity-preserving tables."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterator, Mapping, Sequence

import numpy as np


SCHEMA = "salsbury-columnar-table-v1"


class ColumnarArtifactError(ValueError):
    """Raised when a columnar artifact is incomplete or fails validation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


class AtomicColumnarBundle:
    """Stage multiple tables and publish the bundle with one atomic rename."""

    def __init__(self, output_root: Path):
        self.output_root = Path(output_root).expanduser().resolve(strict=False)
        if self.output_root.exists():
            raise ColumnarArtifactError(
                f"columnar artifact output already exists: {self.output_root}"
            )
        self.output_root.parent.mkdir(parents=True, exist_ok=True)
        self.stage_root = Path(tempfile.mkdtemp(
            prefix=f".{self.output_root.name}.tmp-",
            dir=self.output_root.parent,
        ))
        self._published = False

    def write_table(
        self,
        table_id: str,
        columns: Mapping[str, object],
        *,
        constants: Mapping[str, object],
        provenance: Mapping[str, object],
    ) -> Dict[str, object]:
        if not table_id or table_id.startswith("/") or ".." in Path(table_id).parts:
            raise ColumnarArtifactError("table_id must be a safe relative path")
        table_root = self.stage_root / table_id
        table_root.mkdir(parents=True, exist_ok=False)
        arrays: Dict[str, Dict[str, object]] = {}
        row_count = None
        for index, (name, value) in enumerate(sorted(columns.items())):
            if not name or "/" in name or name in arrays:
                raise ColumnarArtifactError("column names must be unique path-safe strings")
            array = np.asarray(value)
            if array.ndim == 0:
                raise ColumnarArtifactError(f"column {name} must have a row dimension")
            if array.dtype.kind not in "biuf":
                raise ColumnarArtifactError(
                    f"column {name} must use a boolean, integer, or floating dtype"
                )
            if row_count is None:
                row_count = int(array.shape[0])
            elif int(array.shape[0]) != row_count:
                raise ColumnarArtifactError("column row counts do not match")
            filename = f"{index:04d}-{name}.npy"
            path = table_root / filename
            np.save(path, array, allow_pickle=False)
            arrays[name] = {
                "path": filename,
                "sha256": _sha256(path),
                "dtype": str(array.dtype),
                "shape": [int(value) for value in array.shape],
                "size_bytes": path.stat().st_size,
            }
        if row_count is None or row_count <= 0:
            raise ColumnarArtifactError("columnar table must contain at least one row")
        manifest = {
            "artifact_schema": SCHEMA,
            "technical_status": "complete",
            "table_id": table_id,
            "row_count": row_count,
            "columns": arrays,
            "constants": dict(constants),
            "provenance": dict(provenance),
        }
        manifest_path = table_root / "manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        final_manifest_path = self.output_root / table_id / "manifest.json"
        return {
            "artifact_schema": SCHEMA,
            "manifest_path": str(final_manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "row_count": row_count,
            "storage": "numpy_npy_memory_mapped",
        }

    def publish(self) -> None:
        if self._published:
            raise ColumnarArtifactError("columnar bundle was already published")
        os.replace(self.stage_root, self.output_root)
        self._published = True

    def abort(self) -> None:
        stage_root = getattr(self, "stage_root", None)
        if (
            not getattr(self, "_published", False)
            and isinstance(stage_root, Path)
            and stage_root.exists()
        ):
            shutil.rmtree(stage_root)

    def __del__(self) -> None:
        try:
            self.abort()
        except (AttributeError, OSError):
            pass


def load_columnar_table(
    reference: Mapping[str, object],
    *,
    verify_array_hashes: bool = True,
) -> Dict[str, object]:
    """Validate one table reference and open its arrays read-only via memmap."""

    if reference.get("artifact_schema") != SCHEMA:
        raise ColumnarArtifactError("unsupported columnar artifact schema")
    manifest_path = Path(str(reference.get("manifest_path", ""))).expanduser().resolve(
        strict=True
    )
    if reference.get("manifest_sha256") != _sha256(manifest_path):
        raise ColumnarArtifactError("columnar manifest hash mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ColumnarArtifactError(f"columnar manifest is unreadable: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("artifact_schema") != SCHEMA
        or manifest.get("technical_status") != "complete"
        or manifest.get("row_count") != reference.get("row_count")
        or not isinstance(manifest.get("columns"), dict)
    ):
        raise ColumnarArtifactError("columnar manifest contract is invalid")
    arrays = {}
    for name, raw in manifest["columns"].items():
        if not isinstance(raw, dict):
            raise ColumnarArtifactError("column descriptor must be an object")
        path = (manifest_path.parent / str(raw.get("path", ""))).resolve(strict=True)
        if path.parent != manifest_path.parent:
            raise ColumnarArtifactError("column path escapes its table directory")
        if verify_array_hashes and raw.get("sha256") != _sha256(path):
            raise ColumnarArtifactError(f"column {name} hash mismatch")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            str(array.dtype) != raw.get("dtype")
            or [int(value) for value in array.shape] != raw.get("shape")
            or int(array.shape[0]) != int(manifest["row_count"])
        ):
            raise ColumnarArtifactError(f"column {name} metadata mismatch")
        arrays[str(name)] = array
    return {
        "manifest": manifest,
        "arrays": arrays,
        "constants": manifest.get("constants", {}),
        "provenance": manifest.get("provenance", {}),
    }


def iter_columnar_records(
    reference: Mapping[str, object],
    *,
    verify_array_hashes: bool = True,
) -> Iterator[Dict[str, object]]:
    """Yield ordinary row dictionaries without materializing the full table."""

    table = load_columnar_table(
        reference, verify_array_hashes=verify_array_hashes
    )
    arrays = table["arrays"]
    constants = table["constants"]
    assert isinstance(arrays, dict) and isinstance(constants, dict)
    row_count = int(table["manifest"]["row_count"])
    for index in range(row_count):
        row = dict(constants)
        for name, array in arrays.items():
            value = array[index]
            row[name] = value.item() if value.ndim == 0 else value.tolist()
        yield row


def materialize_columnar_records(
    reference: Mapping[str, object],
) -> Sequence[Dict[str, object]]:
    """Compatibility helper for consumers that still require a list."""

    return list(iter_columnar_records(reference))
