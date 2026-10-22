"""Coordinate-only molecular Schoenflies point-group detection.

Only NumPy and the Python standard library are required.  Symmetry operations
are constructed from atom correspondences, rather than inferred from principal
moments of inertia; accidental degeneracies of those moments are harmless.

Public API
----------
detect_point_group(xyz_input, tol=1e-2)
point_group(xyz_input, tol=1e-2)

The ``main`` function is suitable for a ``pgmol = "pgmol:main"`` console-script
entry point.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["detect_point_group", "point_group", "main"]


def _read_xyz(xyz_input: str | Path) -> tuple[list[str], np.ndarray]:
    if isinstance(xyz_input, Path):
        text = xyz_input.read_text(encoding="utf-8")
    elif isinstance(xyz_input, str):
        if "\n" in xyz_input or "\r" in xyz_input:
            text = xyz_input
        else:
            text = Path(xyz_input).read_text(encoding="utf-8")
    else:
        raise TypeError("xyz_input must be a string or pathlib.Path")

    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ input must contain an atom count and comment line")

    try:
        count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("The first XYZ line must contain an integer") from exc

    if count < 1:
        raise ValueError("XYZ input must contain at least one atom")
    if len(lines) < count + 2:
        raise ValueError(f"XYZ input declares {count} atoms but has too few lines")

    symbols: list[str] = []
    coordinates = np.empty((count, 3), dtype=float)

    # The comment line is deliberately never interpreted.
    for i, line in enumerate(lines[2 : count + 2]):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid atom record on XYZ line {i + 3}")
        symbol = fields[0]
        if not symbol.isalpha():
            raise ValueError(f"Invalid element symbol on XYZ line {i + 3}")
        symbols.append(symbol.capitalize())
        try:
            coordinates[i] = [float(value) for value in fields[1:4]]
        except ValueError as exc:
            raise ValueError(f"Invalid coordinates on XYZ line {i + 3}") from exc

    if not np.isfinite(coordinates).all():
        raise ValueError("XYZ coordinates must be finite")

    return symbols, coordinates


def _element_groups(symbols: Sequence[str]) -> list[np.ndarray]:
    groups: dict[str, list[int]] = {}
    for i, symbol in enumerate(symbols):
        groups.setdefault(symbol, []).append(i)
    return [np.asarray(indices, dtype=int) for indices in groups.values()]


def _perfect_matching(adjacency: np.ndarray) -> np.ndarray | None:
    """Find a bijection in a bipartite graph without recursive calls."""
    size = len(adjacency)
    if size == 0:
        return np.empty(0, dtype=int)
    if not adjacency.any(axis=1).all() or not adjacency.any(axis=0).all():
        return None

    left_match = np.full(size, -1, dtype=int)
    right_match = np.full(size, -1, dtype=int)
    neighbors = [np.flatnonzero(row) for row in adjacency]

    # Start constrained vertices first; correctness does not depend on ordering.
    for root in sorted(range(size), key=lambda i: len(neighbors[i])):
        parents = np.full(size, -1, dtype=int)
        seen_left = np.zeros(size, dtype=bool)
        seen_left[root] = True
        queue = [root]
        head = 0
        endpoint = -1

        while head < len(queue) and endpoint < 0:
            left = queue[head]
            head += 1
            for right in neighbors[left]:
                if parents[right] >= 0:
                    continue
                parents[right] = left
                other_left = right_match[right]
                if other_left < 0:
                    endpoint = int(right)
                    break
                if not seen_left[other_left]:
                    seen_left[other_left] = True
                    queue.append(int(other_left))

        if endpoint < 0:
            return None

        right = endpoint
        while right >= 0:
            left = int(parents[right])
            previous_right = int(left_match[left])
            left_match[left] = right
            right_match[right] = left
            right = previous_right

    return left_match


def _permutation(
    operation: np.ndarray,
    coordinates: np.ndarray,
    groups: list[np.ndarray],
    tolerance: float,
) -> np.ndarray | None:
    """Return a species-preserving, distance-bounded atom bijection."""
    transformed = coordinates @ operation.T
    permutation = np.empty(len(coordinates), dtype=int)
    limit = tolerance * tolerance

    for indices in groups:
        source = transformed[indices]
        target = coordinates[indices]
        differences = source[:, None, :] - target[None, :, :]
        distances2 = np.einsum("ijk,ijk->ij", differences, differences)

        closest = distances2.argmin(axis=1)
        if (
            np.all(distances2[np.arange(len(indices)), closest] <= limit)
            and len(np.unique(closest)) == len(indices)
        ):
            permutation[indices] = indices[closest]
            continue

        matching = _perfect_matching(distances2 <= limit)
        if matching is None:
            return None
        permutation[indices] = indices[matching]

    return permutation


def _fit_operation(
    coordinates: np.ndarray, permutation: np.ndarray, determinant: int
) -> np.ndarray:
    """Orthogonal Procrustes fit, preserving the prescribed determinant."""
    covariance = coordinates.T @ coordinates[permutation]
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.ones(3)
    correction[-1] = determinant * np.linalg.det(left @ right_t)
    row_operation = (left * correction) @ right_t
    return row_operation.T


def _frame(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    e1 = first / np.linalg.norm(first)
    perpendicular = second - np.dot(second, e1) * e1
    e2 = perpendicular / np.linalg.norm(perpendicular)
    return np.column_stack((e1, e2, np.cross(e1, e2)))


def _enumerate_operations(
    symbols: Sequence[str],
    coordinates: np.ndarray,
    groups: list[np.ndarray],
    tolerance: float,
) -> list[np.ndarray]:
    """Enumerate both orthogonal extensions of every anchor correspondence.

    Two nonparallel centered atom vectors determine an orthogonal operation up
    to the sign of its action normal to their span.  Enumerating both signs also
    captures every planar symmetry, including reflection in the molecular
    plane.
    """
    radii = np.linalg.norm(coordinates, axis=1)
    radius = float(radii.max())
    species = np.asarray(symbols, dtype=object)

    pools: list[np.ndarray] = []
    for i in range(len(coordinates)):
        pools.append(
            np.flatnonzero(
                (species == species[i])
                & (np.abs(radii - radii[i]) <= 2.0 * tolerance)
            )
        )

    # Favor well-conditioned anchors with few possible images.
    first = max(
        range(len(coordinates)),
        key=lambda i: radii[i] / max(1, len(pools[i])),
    )
    first_direction = coordinates[first] / radii[first]
    projections = coordinates @ first_direction
    perpendicular_lengths = np.linalg.norm(
        coordinates - projections[:, None] * first_direction, axis=1
    )
    second = max(
        range(len(coordinates)),
        key=lambda i: perpendicular_lengths[i] / max(1, len(pools[i])),
    )

    source_frame = _frame(coordinates[first], coordinates[second])
    source_parallel = float(projections[second])
    source_perpendicular = float(perpendicular_lengths[second])

    operations = [np.eye(3)]

    # Equal operations are compared geometrically, not by their orientation in
    # the input coordinate system.
    duplicate_threshold = max(
        2e-8, min(1e-3, 0.2 * tolerance / radius)
    )

    def add_operation(operation: np.ndarray) -> None:
        if any(
            np.linalg.norm(operation - existing) <= duplicate_threshold
            for existing in operations
        ):
            return
        operations.append(operation)

    for image_first in pools[first]:
        vector_first = coordinates[image_first]
        length_first = float(radii[image_first])
        if length_first == 0.0:
            continue
        image_direction = vector_first / length_first

        for image_second in pools[second]:
            vector_second = coordinates[image_second]
            parallel = float(np.dot(vector_second, image_direction))
            perpendicular = vector_second - parallel * image_direction
            perpendicular_length = float(np.linalg.norm(perpendicular))

            if perpendicular_length <= np.finfo(float).eps * radius * 32:
                continue
            if abs(parallel - source_parallel) > 4.0 * tolerance:
                continue
            if abs(perpendicular_length - source_perpendicular) > 4.0 * tolerance:
                continue

            target_frame = _frame(vector_first, vector_second)

            # Anchored maps of slightly perturbed geometries may have amplified
            # residuals; a subsequent least-squares fit must still pass the
            # original Cartesian tolerance.
            loose_tolerance = tolerance * (
                4.0 + 4.0 * radius / source_perpendicular
            )

            for determinant in (1, -1):
                signs = np.diag([1.0, 1.0, float(determinant)])
                candidate = target_frame @ signs @ source_frame.T

                strict_permutation = _permutation(
                    candidate, coordinates, groups, tolerance
                )
                permutation = strict_permutation
                if permutation is None:
                    permutation = _permutation(
                        candidate, coordinates, groups, loose_tolerance
                    )
                if permutation is None:
                    continue

                fitted = _fit_operation(coordinates, permutation, determinant)
                fitted_permutation = _permutation(
                    fitted, coordinates, groups, tolerance
                )
                if fitted_permutation is not None:
                    add_operation(fitted)
                elif strict_permutation is not None:
                    add_operation(candidate)

    return operations


def _rotation_order(operation: np.ndarray, group_order: int) -> int:
    """Infer an operation's finite order using Lagrange's theorem."""
    cosine = float(np.clip((np.trace(operation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    fraction = angle / (2.0 * math.pi)

    best_order = 1
    best_error = abs(fraction)
    for order in range(2, group_order + 1):
        if group_order % order:
            continue
        numerator = int(round(fraction * order))
        if numerator < 1 or math.gcd(numerator, order) != 1:
            continue
        error = abs(fraction - numerator / order)
        if error < best_error:
            best_error = error
            best_order = order
    return best_order


def _rotation_axis(operation: np.ndarray) -> np.ndarray:
    _, _, right_t = np.linalg.svd(operation - np.eye(3))
    axis = right_t[-1]
    return axis / np.linalg.norm(axis)


def _classify_operations(
    operations: list[np.ndarray], radius: float, tolerance: float
) -> str:
    proper = [op for op in operations if np.linalg.det(op) > 0]
    improper = [op for op in operations if np.linalg.det(op) < 0]
    count = len(proper)
    orders = [_rotation_order(op, count) for op in proper]
    maximum_order = max(orders)

    angular_tolerance = max(2e-7, min(0.05, 3.0 * tolerance / radius))
    inversion = any(
        np.linalg.norm(op + np.eye(3)) < angular_tolerance
        for op in improper
    )
    mirrors = [
        op
        for op in improper
        if abs(float(np.trace(op)) - 1.0) < angular_tolerance
        and np.linalg.norm(op @ op - np.eye(3)) < angular_tolerance
    ]

    # These signatures distinguish polyhedral groups from accidental spherical
    # tops and from axial groups having the same number of operations.
    if count == 12 and maximum_order == 3:
        if not improper:
            return "T"
        return "Th" if inversion else "Td"
    if count == 24 and maximum_order == 4:
        return "Oh" if improper else "O"
    if count == 60 and maximum_order == 5:
        return "Ih" if improper else "I"

    if count == 1:
        if not improper:
            return "C1"
        return "Ci" if inversion else "Cs"

    n = maximum_order
    principal_axes = [
        _rotation_axis(op)
        for op, order in zip(proper, orders)
        if order == n
    ]

    # D2 has three equally legitimate principal axes.  Testing all of them
    # distinguishes D2h from D2d without an arbitrary inertia-axis choice.
    horizontal = any(
        np.linalg.norm(mirror @ axis + axis) < angular_tolerance
        for mirror in mirrors
        for axis in principal_axes
    )

    if count == 2 * n:
        if not improper:
            return f"D{n}"
        return f"D{n}h" if horizontal else f"D{n}d"

    if not improper:
        return f"C{n}"
    if horizontal:
        return f"C{n}h"
    if mirrors:
        return f"C{n}v"

    # An improper-only cyclic extension has generator S_(2n), with n proper
    # rotations.  S2 is Ci and was handled above.
    return f"S{2 * n}"


def detect_point_group(xyz_input: str | Path, tol: float = 1e-2) -> str:
    """Determine the Schoenflies point group from atomic coordinates.

    Parameters
    ----------
    xyz_input : str | pathlib.Path
        An XYZ file path or a raw multiline XYZ string.  Its comment line is
        ignored completely.
    tol : float, optional
        Maximum Cartesian atom-matching distance in angstroms; default 1e-2.

    Returns
    -------
    str
        ASCII Schoenflies symbol, including ``Kh``, ``Cinfv`` and ``Dinfh``.

    Notes
    -----
    Atom matching is element-preserving and bijective.  Coordinates are
    centered at their unweighted centroid.  No masses, molecular formulas,
    filenames, comment metadata, or inertia degeneracies enter classification.

    As with any Cartesian tolerance scheme, excessively large tolerances or
    geometries precisely on a tolerance boundary can make symmetry ambiguous.
    """
    try:
        tolerance = float(tol)
    except (TypeError, ValueError) as exc:
        raise ValueError("tol must be a positive finite number") from exc
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tol must be a positive finite number")

    symbols, coordinates = _read_xyz(xyz_input)
    if len(coordinates) == 1:
        return "Kh"

    # Subtracting an existing point first avoids losing centroid precision when
    # a small molecule is supplied with a large global translation.
    coordinates = coordinates - coordinates[0]
    coordinates = coordinates - coordinates.mean(axis=0)
    groups = _element_groups(symbols)
    radius = float(np.linalg.norm(coordinates, axis=1).max())

    numerical_slack = 64.0 * np.finfo(float).eps * max(radius, tolerance)
    tolerance += numerical_slack

    if radius <= numerical_slack:
        return "Kh"

    # Test linearity geometrically, not through eigenvalue degeneracy.
    _, _, right_t = np.linalg.svd(coordinates, full_matrices=False)
    axis = right_t[0]
    transverse = coordinates - (coordinates @ axis)[:, None] * axis
    if np.max(np.linalg.norm(transverse, axis=1)) <= tolerance:
        inversion = _permutation(
            -np.eye(3), coordinates, groups, tolerance
        )
        return "Dinfh" if inversion is not None else "Cinfv"

    operations = _enumerate_operations(
        symbols, coordinates, groups, tolerance
    )
    return _classify_operations(operations, radius, tolerance)


point_group = detect_point_group


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        prog="pgmol",
        description="Determine an XYZ molecule's Schoenflies point group.",
    )
    parser.add_argument("xyz", type=Path, help="path to an XYZ file")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="Cartesian matching tolerance in angstroms (default: 0.01)",
    )
    arguments = parser.parse_args(argv)
    try:
        symbol = detect_point_group(arguments.xyz, arguments.tol)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"pgmol: error: {exc}\n")
    print(symbol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
