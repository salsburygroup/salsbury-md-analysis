"""Small shared validators for explicit analysis configuration gates."""

from __future__ import annotations

from typing import Type


def positive_integer(
    value: object,
    label: str,
    *,
    error_type: Type[ValueError] = ValueError,
) -> int:
    """Return a positive integer or raise the caller's analysis error type."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise error_type(f"{label} must be a positive integer")
    return value


def integer_at_least(
    value: object,
    label: str,
    minimum: int = 1,
    *,
    error_type: Type[ValueError] = ValueError,
) -> int:
    """Return an integer meeting an explicit inclusive lower bound."""

    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("minimum must be a positive integer")
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise error_type(f"{label} must be an integer of at least {minimum}")
    return value
