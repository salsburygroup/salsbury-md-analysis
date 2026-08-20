"""Dependency-free streaming covariance and deterministic PCA primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


class PCAError(ValueError):
    """Raised when a covariance state or PCA solution is not well defined."""


def _finite_vector(values: Iterable[float], expected: int) -> Tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != expected:
        raise PCAError(f"feature count is {len(result)}; expected {expected}")
    if not all(math.isfinite(value) for value in result):
        raise PCAError("feature vector contains a non-finite value")
    return result


class CartesianCovariance:
    """Streaming full Cartesian covariance using the population denominator N."""

    def __init__(self, feature_count: int):
        if (
            isinstance(feature_count, bool)
            or not isinstance(feature_count, int)
            or feature_count <= 0
        ):
            raise PCAError("feature_count must be a positive integer")
        self.feature_count = feature_count
        self.count = 0
        self._mean = np.zeros(feature_count, dtype=float)
        self._m2 = np.zeros((feature_count, feature_count), dtype=float)

    def update(self, values: Iterable[float]) -> None:
        vector = np.asarray(_finite_vector(values, self.feature_count), dtype=float)
        next_count = self.count + 1
        delta = vector - self._mean
        self._mean += delta / next_count
        residual = vector - self._mean
        self._m2 += np.outer(delta, residual)
        self.count = next_count

    def merge(self, other: "CartesianCovariance") -> None:
        if self.feature_count != other.feature_count:
            raise PCAError("cannot merge covariance states with different feature counts")
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self._mean = other._mean.copy()
            self._m2 = other._m2.copy()
            return
        combined_count = self.count + other.count
        delta = other._mean - self._mean
        correction_scale = self.count * other.count / combined_count
        self._m2 += other._m2 + correction_scale * np.outer(delta, delta)
        self._mean += delta * other.count / combined_count
        self.count = combined_count

    def mean(self) -> Tuple[float, ...]:
        if self.count == 0:
            raise PCAError("covariance state contains no samples")
        return tuple(float(value) for value in self._mean)

    def population_covariance(self) -> Tuple[Tuple[float, ...], ...]:
        if self.count == 0:
            raise PCAError("covariance state contains no samples")
        return tuple(
            tuple(float(value) for value in row)
            for row in self._m2 / self.count
        )


def mixture_covariance(
    states: Sequence[CartesianCovariance], weights: Sequence[float]
) -> Tuple[Tuple[float, ...], Tuple[Tuple[float, ...], ...]]:
    """Return a weighted mixture of population means and covariances.

    Each weight applies to an entire state population, not to each frame. This
    makes equal-replica PCA precise even when replicas contain unequal numbers
    of evaluated frames.
    """

    if not states:
        raise PCAError("at least one covariance state is required")
    if len(states) != len(weights):
        raise PCAError("state and weight counts must match")
    feature_count = states[0].feature_count
    if any(state.feature_count != feature_count for state in states):
        raise PCAError("mixture states have different feature counts")
    if any(state.count == 0 for state in states):
        raise PCAError("mixture states must each contain at least one sample")
    normalized = tuple(float(weight) for weight in weights)
    if not all(math.isfinite(weight) and weight > 0.0 for weight in normalized):
        raise PCAError("mixture weights must be finite and positive")
    total_weight = sum(normalized)
    normalized = tuple(weight / total_weight for weight in normalized)
    state_means = [np.asarray(state.mean(), dtype=float) for state in states]
    mean = sum(
        weight * state_mean
        for weight, state_mean in zip(normalized, state_means)
    )
    covariance = np.zeros((feature_count, feature_count), dtype=float)
    for weight, state_mean, state in zip(normalized, state_means, states):
        delta = state_mean - mean
        covariance += weight * (
            np.asarray(state.population_covariance(), dtype=float)
            + np.outer(delta, delta)
        )
    return (
        tuple(float(value) for value in mean),
        tuple(tuple(float(value) for value in row) for row in covariance),
    )


@dataclass(frozen=True)
class PrincipalComponent:
    component_index: int
    eigenvalue_angstrom2: float
    explained_variance_fraction: float
    cumulative_explained_variance_fraction: float
    vector: Tuple[float, ...]
    residual_norm_angstrom2: float
    iteration_count: int
    converged: bool


@dataclass(frozen=True)
class PCAResult:
    total_variance_angstrom2: float
    requested_component_count: int
    numerical_rank_lower_bound: int
    components: Tuple[PrincipalComponent, ...]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    return [_dot(row, vector) for row in matrix]


def _orient(vector: Sequence[float]) -> Tuple[float, ...]:
    pivot = max(range(len(vector)), key=lambda index: (abs(vector[index]), -index))
    sign = -1.0 if vector[pivot] < 0.0 else 1.0
    return tuple(sign * value for value in vector)


def principal_components(
    covariance: Sequence[Sequence[float]],
    component_count: int,
    *,
    eigenvalue_tolerance_angstrom2: float = 1.0e-12,
    solver_tolerance: float = 1.0e-10,
    maximum_relative_residual: float = 1.0e-8,
    maximum_iterations: int = 10_000,
) -> PCAResult:
    """Return leading components of a finite symmetric positive-semidefinite matrix.

    The symmetric LAPACK eigensolver exposed by NumPy replaces the former
    nested-Python power iteration. Every returned component retains a directly
    evaluated eigenpair residual and a deterministic sign orientation.
    """

    size = len(covariance)
    if size == 0 or any(len(row) != size for row in covariance):
        raise PCAError("covariance must be a nonempty square matrix")
    if (
        isinstance(component_count, bool)
        or not isinstance(component_count, int)
        or component_count <= 0
    ):
        raise PCAError("component_count must be a positive integer")
    if component_count > size:
        raise PCAError("component_count cannot exceed feature_count")
    if not all(math.isfinite(float(value)) for row in covariance for value in row):
        raise PCAError("covariance contains a non-finite value")
    if not math.isfinite(eigenvalue_tolerance_angstrom2) or eigenvalue_tolerance_angstrom2 <= 0:
        raise PCAError("eigenvalue tolerance must be finite and positive")
    if not math.isfinite(solver_tolerance) or solver_tolerance <= 0:
        raise PCAError("solver tolerance must be finite and positive")
    if not math.isfinite(maximum_relative_residual) or maximum_relative_residual <= 0:
        raise PCAError("maximum relative residual must be finite and positive")
    if maximum_iterations <= 0:
        raise PCAError("maximum_iterations must be positive")
    original = np.asarray(covariance, dtype=float)
    if np.any(np.diag(original) < -eigenvalue_tolerance_angstrom2):
        raise PCAError("covariance has a materially negative diagonal entry")
    scale = np.maximum(1.0, np.maximum(np.abs(original), np.abs(original.T)))
    if np.any(np.abs(original - original.T) > solver_tolerance * scale):
        raise PCAError("covariance is not symmetric within solver tolerance")
    original = 0.5 * (original + original.T)
    total_variance = float(np.maximum(0.0, np.diag(original)).sum())
    if total_variance <= eigenvalue_tolerance_angstrom2:
        raise PCAError("total Cartesian variance is below the numerical eigenvalue gate")
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(original)
    except np.linalg.LinAlgError as exc:
        raise PCAError(f"symmetric eigensolver failed: {exc}") from exc
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    raw_components = []
    for index in order[:component_count]:
        eigenvalue = float(eigenvalues[index])
        if eigenvalue < -eigenvalue_tolerance_angstrom2:
            raise PCAError("PCA solver produced a materially negative leading eigenvalue")
        if eigenvalue <= eigenvalue_tolerance_angstrom2:
            continue
        vector_tuple = _orient(eigenvectors[:, index].tolist())
        vector = np.asarray(vector_tuple, dtype=float)
        residual = float(np.linalg.norm(original @ vector - eigenvalue * vector))
        relative_residual = residual / max(eigenvalue, total_variance)
        if relative_residual > maximum_relative_residual:
            raise PCAError(
                f"eigenpair relative residual {relative_residual:.3e} "
                f"exceeds {maximum_relative_residual:.3e}"
            )
        raw_components.append((eigenvalue, vector_tuple, residual, 1))

    cumulative = 0.0
    components = []
    for index, (eigenvalue, vector, residual, iterations) in enumerate(raw_components, start=1):
        fraction = eigenvalue / total_variance
        cumulative += fraction
        components.append(PrincipalComponent(
            component_index=index,
            eigenvalue_angstrom2=eigenvalue,
            explained_variance_fraction=fraction,
            cumulative_explained_variance_fraction=min(1.0, cumulative),
            vector=vector,
            residual_norm_angstrom2=residual,
            iteration_count=iterations,
            converged=True,
        ))
    return PCAResult(
        total_variance_angstrom2=total_variance,
        requested_component_count=component_count,
        numerical_rank_lower_bound=len(components),
        components=tuple(components),
    )


def randomized_truncated_pca(
    samples: np.ndarray,
    sample_weights: Sequence[float],
    component_count: int,
    *,
    oversampling: int = 10,
    power_iterations: int = 4,
    power_iteration_schedule: Sequence[int] | None = None,
    random_seed: int = 0,
    eigenvalue_tolerance_angstrom2: float = 1.0e-12,
    maximum_relative_residual: float = 1.0e-4,
) -> Tuple[Tuple[float, ...], PCAResult, Dict[str, object]]:
    """Fit deterministic weighted truncated PCA without a dense covariance.

    The sample matrix is centered using normalized positive sample weights. A
    Halko-style randomized range finder with QR normalization estimates a small
    right-singular subspace. Exact covariance-action residuals, total variance,
    and orthonormality are evaluated before a result is returned.
    """

    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise PCAError("samples must be a two-dimensional matrix with at least two rows")
    if not np.isfinite(values).all():
        raise PCAError("samples contain a non-finite value")
    if (
        isinstance(component_count, bool)
        or not isinstance(component_count, int)
        or component_count <= 0
        or component_count > min(values.shape)
    ):
        raise PCAError("component_count must be positive and not exceed the sample rank bound")
    if isinstance(oversampling, bool) or not isinstance(oversampling, int) or oversampling < 2:
        raise PCAError("oversampling must be an integer of at least two")
    if (
        isinstance(power_iterations, bool)
        or not isinstance(power_iterations, int)
        or power_iterations < 0
    ):
        raise PCAError("power_iterations must be a nonnegative integer")
    if power_iteration_schedule is None:
        refinement_schedule = (power_iterations,)
    else:
        refinement_schedule = tuple(power_iteration_schedule)
        if (
            not refinement_schedule
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in refinement_schedule
            )
            or refinement_schedule[0] != power_iterations
            or any(
                later <= earlier
                for earlier, later in zip(
                    refinement_schedule, refinement_schedule[1:]
                )
            )
        ):
            raise PCAError(
                "power_iteration_schedule must begin with power_iterations and "
                "contain strictly increasing nonnegative integers"
            )
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise PCAError("random_seed must be a nonnegative integer")
    if (
        not math.isfinite(maximum_relative_residual)
        or maximum_relative_residual <= 0.0
    ):
        raise PCAError("maximum_relative_residual must be finite and positive")
    weights = np.asarray(tuple(float(value) for value in sample_weights), dtype=float)
    if weights.shape != (values.shape[0],):
        raise PCAError("sample_weights length must match the sample row count")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise PCAError("sample_weights must be finite and positive")
    weights /= float(weights.sum())
    mean_array = weights @ values
    weighted = (values - mean_array) * np.sqrt(weights)[:, np.newaxis]
    total_variance = float(np.sum(weighted * weighted))
    if total_variance <= eigenvalue_tolerance_angstrom2:
        raise PCAError("total Cartesian variance is below the numerical eigenvalue gate")

    subspace_size = min(
        component_count + oversampling, values.shape[0], values.shape[1]
    )
    generator = np.random.default_rng(random_seed)
    omega = generator.standard_normal((values.shape[1], subspace_size))
    image = weighted @ omega
    current_iterations = 0
    refinement_attempts: List[Dict[str, object]] = []
    accepted = None
    for target_iterations in refinement_schedule:
        while current_iterations < target_iterations:
            left, _ = np.linalg.qr(image, mode="reduced")
            right, _ = np.linalg.qr(weighted.T @ left, mode="reduced")
            image = weighted @ right
            current_iterations += 1
        left, _ = np.linalg.qr(image, mode="reduced")
        reduced = left.T @ weighted
        try:
            _, singular_values, right_vectors = np.linalg.svd(
                reduced, full_matrices=False
            )
        except np.linalg.LinAlgError as exc:
            raise PCAError(f"randomized truncated SVD failed: {exc}") from exc

        raw = []
        for index in range(min(component_count, len(singular_values))):
            eigenvalue = float(singular_values[index] ** 2)
            if eigenvalue <= eigenvalue_tolerance_angstrom2:
                continue
            vector = np.asarray(_orient(right_vectors[index].tolist()), dtype=float)
            raw.append((eigenvalue, vector))
        if not raw:
            raise PCAError(
                "randomized solver returned no component above the eigenvalue gate"
            )
        vectors = np.column_stack([row[1] for row in raw])
        eigenvalues = np.asarray([row[0] for row in raw], dtype=float)
        residual_matrix = weighted.T @ (weighted @ vectors) - vectors * eigenvalues
        residual_norms = np.linalg.norm(residual_matrix, axis=0)
        relative_residuals = residual_norms / np.maximum(
            eigenvalues, eigenvalue_tolerance_angstrom2
        )
        maximum_observed_residual = float(np.max(relative_residuals))
        gram = vectors.T @ vectors
        orthonormality_error = float(
            np.max(np.abs(gram - np.eye(len(raw))))
        )
        refinement_attempts.append({
            "power_iterations": target_iterations,
            "maximum_relative_residual": maximum_observed_residual,
            "relative_residual_gate_satisfied": (
                maximum_observed_residual <= maximum_relative_residual
            ),
            "orthonormality_maximum_absolute_error": orthonormality_error,
        })
        if orthonormality_error > 1.0e-10:
            raise PCAError(
                f"randomized PCA orthonormality error {orthonormality_error:.3e} "
                "exceeds 1e-10"
            )
        if maximum_observed_residual <= maximum_relative_residual:
            accepted = (
                raw, residual_norms, maximum_observed_residual,
                orthonormality_error, target_iterations,
            )
            break
    if accepted is None:
        evidence = ", ".join(
            f"q={attempt['power_iterations']}:"
            f"{float(attempt['maximum_relative_residual']):.3e}"
            for attempt in refinement_attempts
        )
        raise PCAError(
            "randomized PCA maximum relative residual remained above "
            f"{maximum_relative_residual:.3e} after bounded refinement "
            f"({evidence})"
        )
    (
        raw, residual_norms, maximum_observed_residual,
        orthonormality_error, applied_power_iterations,
    ) = accepted

    cumulative = 0.0
    components = []
    for index, ((eigenvalue, vector), residual) in enumerate(
        zip(raw, residual_norms), start=1
    ):
        fraction = eigenvalue / total_variance
        cumulative += fraction
        components.append(PrincipalComponent(
            component_index=index,
            eigenvalue_angstrom2=eigenvalue,
            explained_variance_fraction=fraction,
            cumulative_explained_variance_fraction=min(1.0, cumulative),
            vector=tuple(float(value) for value in vector),
            residual_norm_angstrom2=float(residual),
            iteration_count=applied_power_iterations,
            converged=True,
        ))
    result = PCAResult(
        total_variance_angstrom2=total_variance,
        requested_component_count=component_count,
        numerical_rank_lower_bound=len(components),
        components=tuple(components),
    )
    diagnostics = {
        "method": "randomized_truncated_svd_v1",
        "sample_count": int(values.shape[0]),
        "feature_count": int(values.shape[1]),
        "subspace_size": subspace_size,
        "oversampling": oversampling,
        "power_iterations": applied_power_iterations,
        "requested_power_iterations": power_iterations,
        "power_iteration_schedule": list(refinement_schedule),
        "refinement_attempts": refinement_attempts,
        "random_seed": random_seed,
        "maximum_relative_residual": maximum_observed_residual,
        "relative_residual_gate": maximum_relative_residual,
        "orthonormality_maximum_absolute_error": orthonormality_error,
        "weight_sum": float(weights.sum()),
    }
    return tuple(float(value) for value in mean_array), result, diagnostics


def project(vector: Iterable[float], mean: Sequence[float], components: Sequence[PrincipalComponent]) -> Tuple[float, ...]:
    """Project one Cartesian feature vector onto ordered PCA components."""

    values = _finite_vector(vector, len(mean))
    centered = [value - center for value, center in zip(values, mean)]
    return tuple(_dot(centered, component.vector) for component in components)
