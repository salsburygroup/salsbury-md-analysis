"""Dependency-free geometry primitives used by trajectory analyses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


Coordinate = Tuple[float, float, float]
Matrix3 = Tuple[Coordinate, Coordinate, Coordinate]
IDENTITY_MATRIX: Matrix3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


class GeometryError(ValueError):
    """Raised when coordinates cannot support the requested calculation."""


@dataclass(frozen=True)
class RigidTransform:
    """Rotation about the mobile centroid followed by reference translation."""

    rotation: Matrix3
    mobile_centroid: Coordinate
    reference_centroid: Coordinate
    fitted_rmsd_angstrom: float


def _validated_coordinates(
    coordinates: Iterable[Sequence[float]], label: str
) -> Tuple[Coordinate, ...]:
    result = []
    for index, coordinate in enumerate(coordinates):
        if len(coordinate) != 3:
            raise GeometryError(f"{label}[{index}] must contain three coordinates")
        values = tuple(float(value) for value in coordinate)
        if not all(math.isfinite(value) for value in values):
            raise GeometryError(f"{label}[{index}] contains a non-finite value")
        result.append((values[0], values[1], values[2]))
    if not result:
        raise GeometryError(f"{label} must contain at least one atom")
    return tuple(result)


def _centroid(coordinates: Sequence[Coordinate]) -> Coordinate:
    count = len(coordinates)
    return tuple(
        sum(coordinate[axis] for coordinate in coordinates) / count
        for axis in range(3)
    )  # type: ignore[return-value]


def _largest_symmetric_eigenvector(matrix: Sequence[Sequence[float]]) -> Tuple[float, ...]:
    """Return the largest-eigenvalue eigenvector using symmetric Jacobi sweeps."""

    size = len(matrix)
    values = [list(row) for row in matrix]
    vectors = [[1.0 if row == column else 0.0 for column in range(size)] for row in range(size)]
    for _ in range(100):
        p, q = 0, 1
        maximum = abs(values[p][q])
        for row in range(size):
            for column in range(row + 1, size):
                candidate = abs(values[row][column])
                if candidate > maximum:
                    maximum = candidate
                    p, q = row, column
        if maximum <= 1.0e-15:
            break

        app = values[p][p]
        aqq = values[q][q]
        apq = values[p][q]
        tau = (aqq - app) / (2.0 * apq)
        tangent = (
            1.0 / (tau + math.sqrt(1.0 + tau * tau))
            if tau >= 0.0
            else -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
        )
        cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
        sine = tangent * cosine

        for index in range(size):
            if index in (p, q):
                continue
            aip = values[index][p]
            aiq = values[index][q]
            values[index][p] = values[p][index] = cosine * aip - sine * aiq
            values[index][q] = values[q][index] = sine * aip + cosine * aiq
        values[p][p] = (
            cosine * cosine * app
            - 2.0 * sine * cosine * apq
            + sine * sine * aqq
        )
        values[q][q] = (
            sine * sine * app
            + 2.0 * sine * cosine * apq
            + cosine * cosine * aqq
        )
        values[p][q] = values[q][p] = 0.0
        for index in range(size):
            vip = vectors[index][p]
            viq = vectors[index][q]
            vectors[index][p] = cosine * vip - sine * viq
            vectors[index][q] = sine * vip + cosine * viq

    largest = max(range(size), key=lambda index: values[index][index])
    result = tuple(vectors[row][largest] for row in range(size))
    norm = math.sqrt(sum(value * value for value in result))
    if norm == 0.0:
        raise GeometryError("optimal-rotation eigenvector has zero length")
    normalized = tuple(value / norm for value in result)
    if normalized[0] < 0.0:
        normalized = tuple(-value for value in normalized)
    return normalized


def _quaternion_rotation(quaternion: Sequence[float]) -> Matrix3:
    w, x, y, z = quaternion
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ),
        (
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ),
        (
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _rotate(rotation: Matrix3, coordinate: Coordinate) -> Coordinate:
    return tuple(
        sum(rotation[row][column] * coordinate[column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def rmsd(first: Iterable[Sequence[float]], second: Iterable[Sequence[float]]) -> float:
    """Return the unweighted positional RMSD for two ordered coordinate sets."""

    left = _validated_coordinates(first, "first")
    right = _validated_coordinates(second, "second")
    if len(left) != len(right):
        raise GeometryError("RMSD coordinate sets must have equal atom counts")
    squared = sum(
        sum((a[axis] - b[axis]) ** 2 for axis in range(3))
        for a, b in zip(left, right)
    )
    return math.sqrt(squared / len(left))


def distance3(first: Sequence[float], second: Sequence[float]) -> float:
    """Return Euclidean distance between two three-dimensional coordinates."""

    return math.sqrt(sum((first[index] - second[index]) ** 2 for index in range(3)))


def apply_transform(
    coordinates: Iterable[Sequence[float]], transform: RigidTransform
) -> Tuple[Coordinate, ...]:
    """Apply a fitted transform to coordinates in the mobile coordinate system."""

    points = _validated_coordinates(coordinates, "coordinates")
    result = []
    for point in points:
        centered = tuple(
            point[axis] - transform.mobile_centroid[axis] for axis in range(3)
        )
        rotated = _rotate(transform.rotation, centered)  # type: ignore[arg-type]
        result.append(tuple(
            rotated[axis] + transform.reference_centroid[axis] for axis in range(3)
        ))
    return tuple(result)  # type: ignore[return-value]


def best_fit_transform(
    mobile: Iterable[Sequence[float]], reference: Iterable[Sequence[float]]
) -> RigidTransform:
    """Fit mobile to reference with Horn's unit-quaternion least-squares rotation."""

    mobile_points = _validated_coordinates(mobile, "mobile")
    reference_points = _validated_coordinates(reference, "reference")
    if len(mobile_points) != len(reference_points):
        raise GeometryError("fit coordinate sets must have equal atom counts")
    mobile_centroid = _centroid(mobile_points)
    reference_centroid = _centroid(reference_points)
    rotation = IDENTITY_MATRIX
    if len(mobile_points) > 1:
        centered_mobile = [
            tuple(point[axis] - mobile_centroid[axis] for axis in range(3))
            for point in mobile_points
        ]
        centered_reference = [
            tuple(point[axis] - reference_centroid[axis] for axis in range(3))
            for point in reference_points
        ]
        covariance = [[0.0] * 3 for _ in range(3)]
        for left, right in zip(centered_mobile, centered_reference):
            for row in range(3):
                for column in range(3):
                    covariance[row][column] += left[row] * right[column]
        sxx, sxy, sxz = covariance[0]
        syx, syy, syz = covariance[1]
        szx, szy, szz = covariance[2]
        quaternion_matrix = (
            (sxx + syy + szz, syz - szy, szx - sxz, sxy - syx),
            (syz - szy, sxx - syy - szz, sxy + syx, szx + sxz),
            (szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy),
            (sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz),
        )
        rotation = _quaternion_rotation(
            _largest_symmetric_eigenvector(quaternion_matrix)
        )
    provisional = RigidTransform(
        rotation=rotation,
        mobile_centroid=mobile_centroid,
        reference_centroid=reference_centroid,
        fitted_rmsd_angstrom=0.0,
    )
    fitted_rmsd = rmsd(apply_transform(mobile_points, provisional), reference_points)
    return RigidTransform(
        rotation=rotation,
        mobile_centroid=mobile_centroid,
        reference_centroid=reference_centroid,
        fitted_rmsd_angstrom=fitted_rmsd,
    )


def mass_weighted_radius_of_gyration(
    coordinates: Iterable[Sequence[float]], masses: Iterable[float]
) -> float:
    """Return mass-weighted radius of gyration in the coordinate input unit."""

    points = _validated_coordinates(coordinates, "coordinates")
    mass_values = tuple(float(mass) for mass in masses)
    if len(points) != len(mass_values):
        raise GeometryError("coordinate and mass counts must match")
    if not all(math.isfinite(mass) and mass > 0.0 for mass in mass_values):
        raise GeometryError("every atomic mass must be finite and positive")
    total_mass = sum(mass_values)
    center = tuple(
        sum(mass * point[axis] for mass, point in zip(mass_values, points)) / total_mass
        for axis in range(3)
    )
    squared = sum(
        mass * sum((point[axis] - center[axis]) ** 2 for axis in range(3))
        for mass, point in zip(mass_values, points)
    )
    return math.sqrt(squared / total_mass)
