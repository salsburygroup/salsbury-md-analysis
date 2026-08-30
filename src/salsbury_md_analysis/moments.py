"""Streaming coordinate moments and small-sample summary helpers."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Coordinate = Tuple[float, float, float]


class MomentError(ValueError):
    """Raised when a streaming moment update is malformed."""


class CoordinateMoments:
    """Numerically stable per-atom Cartesian means and population fluctuations."""

    def __init__(self, atom_count: int):
        if isinstance(atom_count, bool) or not isinstance(atom_count, int) or atom_count <= 0:
            raise MomentError("atom_count must be a positive integer")
        self.atom_count = atom_count
        self.count = 0
        self._mean: List[List[float]] = [[0.0, 0.0, 0.0] for _ in range(atom_count)]
        self._m2: List[List[float]] = [[0.0, 0.0, 0.0] for _ in range(atom_count)]

    def update(self, coordinates: Iterable[Sequence[float]]) -> None:
        points = tuple(tuple(float(value) for value in point) for point in coordinates)
        if len(points) != self.atom_count:
            raise MomentError(
                f"coordinate atom count is {len(points)}; expected {self.atom_count}"
            )
        if any(len(point) != 3 for point in points):
            raise MomentError("each coordinate must contain three values")
        if not all(math.isfinite(value) for point in points for value in point):
            raise MomentError("coordinates contain a non-finite value")
        self.count += 1
        for atom_index, point in enumerate(points):
            for axis in range(3):
                delta = point[axis] - self._mean[atom_index][axis]
                self._mean[atom_index][axis] += delta / self.count
                delta_after = point[axis] - self._mean[atom_index][axis]
                self._m2[atom_index][axis] += delta * delta_after

    def merge(self, other: "CoordinateMoments") -> None:
        if self.atom_count != other.atom_count:
            raise MomentError("cannot merge coordinate moments with different atom counts")
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self._mean = [values[:] for values in other._mean]
            self._m2 = [values[:] for values in other._m2]
            return
        combined_count = self.count + other.count
        for atom_index in range(self.atom_count):
            for axis in range(3):
                delta = other._mean[atom_index][axis] - self._mean[atom_index][axis]
                self._mean[atom_index][axis] += delta * other.count / combined_count
                self._m2[atom_index][axis] += (
                    other._m2[atom_index][axis]
                    + delta * delta * self.count * other.count / combined_count
                )
        self.count = combined_count

    def mean_coordinate(self, atom_index: int) -> Coordinate:
        if self.count == 0:
            raise MomentError("coordinate moments contain no samples")
        values = self._mean[atom_index]
        return values[0], values[1], values[2]

    def rmsf(self, atom_index: int) -> float:
        """Return sqrt(<|r-<r>|^2>) using the population denominator N."""

        if self.count == 0:
            raise MomentError("coordinate moments contain no samples")
        return math.sqrt(sum(self._m2[atom_index]) / self.count)

    def rmsf_values(self) -> Tuple[float, ...]:
        return tuple(self.rmsf(index) for index in range(self.atom_count))

    def to_state(self) -> Dict[str, object]:
        """Return the exact mergeable sufficient state for one worker shard."""

        return {
            "state_schema": "coordinate-moments-v1",
            "atom_count": self.atom_count,
            "count": self.count,
            "mean": [values[:] for values in self._mean],
            "m2": [values[:] for values in self._m2],
        }

    @classmethod
    def from_state(cls, state: Dict[str, object]) -> "CoordinateMoments":
        """Restore a worker state after validating its complete shape."""

        if state.get("state_schema") != "coordinate-moments-v1":
            raise MomentError("coordinate moment state schema is unsupported")
        atom_count = state.get("atom_count")
        count = state.get("count")
        mean = state.get("mean")
        m2 = state.get("m2")
        if (
            isinstance(atom_count, bool) or not isinstance(atom_count, int)
            or isinstance(count, bool) or not isinstance(count, int) or count < 0
            or not isinstance(mean, list) or not isinstance(m2, list)
            or len(mean) != atom_count or len(m2) != atom_count
        ):
            raise MomentError("coordinate moment state is malformed")
        restored = cls(atom_count)
        restored.count = count
        restored._mean = _validated_vector_state(mean, atom_count, "mean")
        restored._m2 = _validated_vector_state(m2, atom_count, "m2")
        if count == 0 and any(value != 0.0 for row in restored._m2 for value in row):
            raise MomentError("empty coordinate moment state has nonzero m2")
        return restored


class DisplacementCovariance:
    """Streaming scalar dot-product covariance between atomic displacement vectors."""

    def __init__(self, atom_count: int):
        if isinstance(atom_count, bool) or not isinstance(atom_count, int) or atom_count <= 0:
            raise MomentError("atom_count must be a positive integer")
        self.atom_count = atom_count
        self.count = 0
        self._mean = np.zeros((atom_count, 3), dtype=float)
        self._co_m2 = np.zeros((atom_count, atom_count), dtype=float)

    def update(self, coordinates: Iterable[Sequence[float]]) -> None:
        points = np.asarray(tuple(tuple(float(value) for value in point) for point in coordinates), dtype=float)
        if points.shape[0] != self.atom_count:
            raise MomentError(
                f"coordinate atom count is {points.shape[0]}; expected {self.atom_count}"
            )
        if points.shape != (self.atom_count, 3):
            raise MomentError("each coordinate must contain three values")
        if not np.isfinite(points).all():
            raise MomentError("coordinates contain a non-finite value")
        new_count = self.count + 1
        deltas = points - self._mean
        self._mean += deltas / new_count
        residuals = points - self._mean
        increment = deltas @ residuals.T
        self._co_m2 += 0.5 * (increment + increment.T)
        self.count = new_count

    def merge(self, other: "DisplacementCovariance") -> None:
        if self.atom_count != other.atom_count:
            raise MomentError("cannot merge covariance states with different atom counts")
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self._mean = other._mean.copy()
            self._co_m2 = other._co_m2.copy()
            return
        combined_count = self.count + other.count
        deltas = other._mean - self._mean
        scale = self.count * other.count / combined_count
        self._co_m2 += other._co_m2 + scale * (deltas @ deltas.T)
        self._mean += deltas * other.count / combined_count
        self.count = combined_count

    def correlation_matrix(
        self, minimum_variance: float
    ) -> Tuple[Tuple[object, ...], ...]:
        """Return DCCM values, using None wherever either atom lacks variance."""

        if self.count == 0:
            raise MomentError("covariance state contains no samples")
        if not math.isfinite(minimum_variance) or minimum_variance <= 0.0:
            raise MomentError("minimum_variance must be finite and positive")
        diagonal = np.diag(self._co_m2) / self.count
        valid = diagonal >= minimum_variance
        denominator = np.sqrt(np.outer(np.diag(self._co_m2), np.diag(self._co_m2)))
        matrix = np.full((self.atom_count, self.atom_count), np.nan, dtype=float)
        valid_pairs = np.outer(valid, valid)
        matrix[valid_pairs] = self._co_m2[valid_pairs] / denominator[valid_pairs]
        matrix = np.clip(matrix, -1.0, 1.0)
        rows: List[Tuple[object, ...]] = []
        for left in range(self.atom_count):
            row: List[object] = []
            for right in range(self.atom_count):
                if not valid_pairs[left, right]:
                    row.append(None)
                    continue
                row.append(float(matrix[left, right]))
            rows.append(tuple(row))
        return tuple(rows)

    def to_state(self) -> Dict[str, object]:
        """Return the exact mergeable covariance state for one worker shard."""

        return {
            "state_schema": "displacement-covariance-v1",
            "atom_count": self.atom_count,
            "count": self.count,
            "mean": self._mean.tolist(),
            "co_m2": self._co_m2.tolist(),
        }

    @classmethod
    def from_state(cls, state: Dict[str, object]) -> "DisplacementCovariance":
        """Restore a covariance worker state with exact dimensional checks."""

        if state.get("state_schema") != "displacement-covariance-v1":
            raise MomentError("displacement covariance state schema is unsupported")
        atom_count = state.get("atom_count")
        count = state.get("count")
        if (
            isinstance(atom_count, bool) or not isinstance(atom_count, int)
            or isinstance(count, bool) or not isinstance(count, int) or count < 0
        ):
            raise MomentError("displacement covariance state is malformed")
        mean = np.asarray(state.get("mean"), dtype=float)
        co_m2 = np.asarray(state.get("co_m2"), dtype=float)
        if mean.shape != (atom_count, 3) or co_m2.shape != (atom_count, atom_count):
            raise MomentError("displacement covariance state has invalid dimensions")
        if not np.isfinite(mean).all() or not np.isfinite(co_m2).all():
            raise MomentError("displacement covariance state contains non-finite values")
        if not np.allclose(co_m2, co_m2.T, rtol=0.0, atol=1.0e-10):
            raise MomentError("displacement covariance state is not symmetric")
        restored = cls(atom_count)
        restored.count = count
        restored._mean = mean.copy()
        restored._co_m2 = co_m2.copy()
        return restored


def _validated_vector_state(
    rows: object, atom_count: int, label: str
) -> List[List[float]]:
    if not isinstance(rows, list) or len(rows) != atom_count:
        raise MomentError(f"coordinate moment {label} state is malformed")
    result: List[List[float]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 3:
            raise MomentError(f"coordinate moment {label} state is malformed")
        values = [float(value) for value in row]
        if not all(math.isfinite(value) for value in values):
            raise MomentError(
                f"coordinate moment {label} state contains non-finite values"
            )
        result.append(values)
    return result


def sample_summary(values: Iterable[float]) -> Dict[str, object]:
    """Return mean, sample SD, and SEM without inventing one-sample uncertainty."""

    data = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in data):
        raise MomentError("summary values must be finite")
    if not data:
        return {"count": 0, "mean": None, "sample_sd": None, "sem": None}
    mean = sum(data) / len(data)
    if len(data) == 1:
        return {"count": 1, "mean": mean, "sample_sd": None, "sem": None}
    variance = sum((value - mean) ** 2 for value in data) / (len(data) - 1)
    sample_sd = math.sqrt(variance)
    return {
        "count": len(data),
        "mean": mean,
        "sample_sd": sample_sd,
        "sem": sample_sd / math.sqrt(len(data)),
    }
