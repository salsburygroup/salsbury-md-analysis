"""Explicit simulation protocol provenance, distinct from saved-frame timing."""

from __future__ import annotations

import math
from collections.abc import Mapping


class SimulationProtocolError(ValueError):
    """A declared simulation protocol is incomplete or ambiguous."""


PROTOCOL_FIELDS = {
    "protocol_id",
    "integration_step_fs",
    "integrator",
    "thermostat",
    "constraints",
    "hamiltonian_sha256",
    "ensemble",
    "temperature_kelvin",
    "initial_ensemble_id",
}


def validate_simulation_protocol(protocol):
    """Validate explicit provenance without inferring it from trajectory output."""
    if not isinstance(protocol, Mapping) or set(protocol) != PROTOCOL_FIELDS:
        raise SimulationProtocolError(
            "simulation_protocol must contain exactly: "
            + ", ".join(sorted(PROTOCOL_FIELDS))
        )
    for field in PROTOCOL_FIELDS - {
        "integration_step_fs",
        "temperature_kelvin",
        "thermostat",
        "constraints",
    }:
        if not isinstance(protocol[field], str) or not protocol[field].strip():
            raise SimulationProtocolError(field + " must be a nonempty string")
    for field in ("integration_step_fs", "temperature_kelvin"):
        value = protocol[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise SimulationProtocolError(field + " must be finite and positive")
    digest = protocol["hamiltonian_sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise SimulationProtocolError(
            "hamiltonian_sha256 must be a lowercase SHA256 digest"
        )
    for field in ("thermostat", "constraints"):
        value = protocol[field]
        if not isinstance(value, Mapping) or set(value) != {"name", "parameters"}:
            raise SimulationProtocolError(field + " must contain name and parameters")
        if (
            not isinstance(value["name"], str)
            or not value["name"].strip()
            or not isinstance(value["parameters"], Mapping)
        ):
            raise SimulationProtocolError(
                field
                + " requires a name and parameter object (use name 'none' when absent)"
            )

        def check(item):
            if isinstance(item, float) and not math.isfinite(item):
                raise SimulationProtocolError(field + " parameters must be finite")
            if isinstance(item, Mapping):
                for child in item.values():
                    check(child)
            elif isinstance(item, list):
                for child in item:
                    check(child)

        check(value["parameters"])
    return dict(protocol)
