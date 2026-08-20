"""Deterministic serialization helpers for provenance and regression identities."""

from __future__ import annotations

import hashlib
import json


def stable_json_sha256(payload: object) -> str:
    """Return the SHA-256 of one canonical, ASCII JSON serialization."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
