"""Coordinate-based molecular Schoenflies point-group detection.

This module requires only NumPy and the Python standard library.
"""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Sequence

import numpy as np

__all__ = ["detect_point_group", "point_group", "main"]


def _read_xyz(xyz_input: str | pathlib.Path) -> tuple[list[str], np.ndarray]:
    if isinstance(xyz_input, pathlib.Path):
        text = xyz_input.read_text(encoding="utf-8")
    elif isinstance(xyz_input, str):
        if "\n" in xyz_input or "\r" in xyz_input:
            text = xyz_input
        else:
            try:
                path = pathlib.Path(xyz_input)
                text = (
                    path.read_text(encoding="utf-8")
                    if path.is_file()
                    else xyz_input
                )
            except (OSError, ValueError):
                text = xyz_input
    else:
        raise TypeError("xyz_input must be a string or pathlib.Path")

    lines = text.lstrip("\ufeff").splitlines()
    if not lines:
        raise ValueError("Empty XYZ input")

    try:
        atom_count = int(lines[0].strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("The first XYZ line must contain the atom count") from exc

    if atom_count < 1:
        raise ValueError("An XYZ structure must contain at least one atom")
    if len(lines) < atom_count + 2:
        raise ValueError(
            f"XYZ input declares {atom_count} atoms but contains too few lines"
        )

    symbols: list[str] = []
    coordinates: list[list[float]] = []

    # Line 2 is deliberately ignored in its entirety.
    for line_number, line in enumerate(lines[2 : atom_count + 2], start=3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid XYZ atom record on line {line_number}")

        symbol = fields[0]
        if not symbol:
            raise ValueError(f"Missing element symbol on line {line_number}")
        symbol = symbol[0].upper() + symbol[1:].lower()

        try:
            xyz = [float(fields[1]), float(fields[2]), float(fields[3])]
        except ValueError as exc:
            raise ValueError(
                f"Invalid Cartesian coordinate on line {line_number}"
            ) from exc

        if not np.all(np.isfinite(xyz)):
            raise ValueError(
                f"Non-finite Cartesian coordinate on line {line_number}"
            )

        symbols.append(symbol)
        coordinates.append(xyz)

    return symbols, np.asarray(coordinates, dtype=float)


def _perfect_matching(
    transformed: np.ndarray,
    target: np.ndarray,
    symbols: Sequence[str],
    tolerance: float,
) -> np.ndarray | None:
    """Find a symbol-preserving perfect matching within a distance threshold."""
    count = len(symbols)
    mapping = np.full(count, -1, dtype=int)

    symbol_groups: dict[str, list[int]] = {}
    for index, symbol in enumerate(symbols):
        symbol_groups.setdefault(symbol, []).append(index)

    tolerance_squared = tolerance * tolerance

    for indices in symbol_groups.values():
        source_indices = np.asarray(indices, dtype=int)
        target_indices = np.asarray(indices, dtype=int)

        difference = (
            transformed[source_indices, None, :]
            - target[target_indices][None, :, :]
        )
        distance_squared = np.einsum(
            "ijk,ijk->ij", difference, difference, optimize=True
        )

        adjacency = [
            list(np.flatnonzero(distance_squared[row] <= tolerance_squared))
            for row in range(len(indices))
        ]
        if any(not neighbors for neighbors in adjacency):
            return None

        # Match the most constrained atoms first.
        row_order = sorted(range(len(indices)), key=lambda row: len(adjacency[row]))
        matched_row_for_column = [-1] * len(indices)

        def augment(row: int, visited: list[bool]) -> bool:
            neighbors = sorted(
                adjacency[row],
                key=lambda column: distance_squared[row, column],
            )
            for column in neighbors:
                if visited[column]:
                    continue
                visited[column] = True
                previous = matched_row_for_column[column]
                if previous == -1 or augment(previous, visited):
                    matched_row_for_column[column] = row
                    return True
            return False

        for row in row_order:
            if not augment(row, [False] * len(indices)):
                return None

        for column, local_row in enumerate(matched_row_for_column):
            mapping[source_indices[local_row]] = target_indices[column]

    return mapping


def _orthogonal_fit(
    source: np.ndarray,
    target: np.ndarray,
    mapping: np.ndarray,
    determinant: int,
) -> np.ndarray:
    """Fit Q in Q @ source_i ~= target_mapping[i], fixing det(Q)."""
    covariance = source.T @ target[mapping]
    left, _, right_transpose = np.linalg.svd(covariance)

    row_transform = left @ right_transpose
    if np.linalg.det(row_transform) * determinant < 0.0:
        correction = np.eye(3)
        correction[-1, -1] = -1.0
        row_transform = left @ correction @ right_transpose

    return row_transform.T


def _refine_operation(
    initial: np.ndarray,
    coordinates: np.ndarray,
    symbols: Sequence[str],
    tolerance: float,
) -> np.ndarray | None:
    determinant = 1 if np.linalg.det(initial) >= 0.0 else -1
    operation = initial

    # A generous preliminary threshold allows the atom-defined frame to be
    # polished by an all-atom orthogonal Procrustes fit.
    preliminary_tolerance = max(4.0 * tolerance, 1.0e-9)

    for _ in range(3):
        transformed = coordinates @ operation.T
        mapping = _perfect_matching(
            transformed, coordinates, symbols, preliminary_tolerance
        )
        if mapping is None:
            return None
        operation = _orthogonal_fit(
            coordinates, coordinates, mapping, determinant
        )

    transformed = coordinates @ operation.T
    mapping = _perfect_matching(
        transformed,
        coordinates,
        symbols,
        tolerance * (1.0 + 1.0e-9) + 1.0e-10,
    )
    if mapping is None:
        return None

    operation = _orthogonal_fit(coordinates, coordinates, mapping, determinant)
    transformed = coordinates @ operation.T
    if (
        _perfect_matching(
            transformed,
            coordinates,
            symbols,
            tolerance * (1.0 + 1.0e-9) + 1.0e-10,
        )
        is None
    ):
        return None

    return operation


def _is_duplicate(
    operation: np.ndarray,
    operations: Sequence[np.ndarray],
    threshold: float,
) -> bool:
    return any(np.linalg.norm(operation - other, ord="fro") <= threshold
               for other in operations)


def _find_frame(coordinates: np.ndarray, tolerance: float) -> tuple[int, int] | None:
    """Choose a numerically stable ordered pair of non-collinear atom vectors."""
    count = len(coordinates)
    best_pair: tuple[int, int] | None = None
    best_score = -1.0

    for first in range(count):
        norm_first = float(np.linalg.norm(coordinates[first]))
        if norm_first <= 1.0e-14:
            continue
        for second in range(count):
            if first == second:
                continue
            norm_second = float(np.linalg.norm(coordinates[second]))
            if norm_second <= 1.0e-14:
                continue

            cross_norm = float(
                np.linalg.norm(np.cross(coordinates[first], coordinates[second]))
            )
            score = cross_norm / max(norm_first, norm_second)
            if score > best_score:
                best_score = score
                best_pair = (first, second)

    if best_pair is None or best_score <= max(tolerance, 1.0e-10):
        return None
    return best_pair


def _basis(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    norm_first = float(np.linalg.norm(first))
    if norm_first <= 1.0e-14:
        return None

    e1 = first / norm_first
    perpendicular = second - np.dot(second, e1) * e1
    perpendicular_norm = float(np.linalg.norm(perpendicular))
    if perpendicular_norm <= 1.0e-14:
        return None

    e2 = perpendicular / perpendicular_norm
    e3 = np.cross(e1, e2)
    return np.column_stack((e1, e2, e3))


def _enumerate_operations(
    coordinates: np.ndarray,
    symbols: Sequence[str],
    tolerance: float,
) -> list[np.ndarray]:
    identity = np.eye(3)
    operations: list[np.ndarray] = [identity]

    frame_indices = _find_frame(coordinates, tolerance)
    if frame_indices is None:
        return operations

    first, second = frame_indices
    source_basis = _basis(coordinates[first], coordinates[second])
    if source_basis is None:
        return operations

    scale = max(float(np.max(np.linalg.norm(coordinates, axis=1))), tolerance)
    duplicate_threshold = max(1.0e-6, min(2.0e-3, tolerance / scale * 0.05))

    first_targets = [
        index for index, symbol in enumerate(symbols)
        if symbol == symbols[first]
    ]
    second_targets = [
        index for index, symbol in enumerate(symbols)
        if symbol == symbols[second]
    ]

    for target_first in first_targets:
        for target_second in second_targets:
            if target_first == target_second:
                continue

            target_basis = _basis(
                coordinates[target_first], coordinates[target_second]
            )
            if target_basis is None:
                continue

            for determinant in (1, -1):
                handedness = np.diag([1.0, 1.0, float(determinant)])
                candidate = target_basis @ handedness @ source_basis.T
                refined = _refine_operation(
                    candidate, coordinates, symbols, tolerance
                )
                if refined is None:
                    continue
                if not _is_duplicate(refined, operations, duplicate_threshold):
                    operations.append(refined)

    return operations


def _operation_order(operation: np.ndarray, maximum: int = 240) -> int:
    identity = np.eye(3)
    power = np.eye(3)
    for order in range(1, maximum + 1):
        power = power @ operation
        if np.linalg.norm(power - identity, ord="fro") <= 5.0e-3:
            return order
    return maximum + 1


def _proper_axis(operation: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(operation)
    index = int(np.argmin(np.abs(values - 1.0)))
    axis = np.real(vectors[:, index])
    axis /= np.linalg.norm(axis)
    return axis


def _improper_axis(operation: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(operation)
    index = int(np.argmin(np.abs(values + 1.0)))
    axis = np.real(vectors[:, index])
    axis /= np.linalg.norm(axis)
    return axis


def _is_inversion(operation: np.ndarray) -> bool:
    return np.linalg.norm(operation + np.eye(3), ord="fro") <= 5.0e-3


def _is_reflection(operation: np.ndarray) -> bool:
    if np.linalg.det(operation) >= 0.0:
        return False
    return (
        _operation_order(operation) == 2
        and not _is_inversion(operation)
        and abs(float(np.trace(operation)) - 1.0) <= 1.0e-2
    )


def _reflection_normal(operation: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((operation + operation.T) * 0.5)
    index = int(np.argmin(values))
    normal = vectors[:, index]
    normal /= np.linalg.norm(normal)
    return normal


def _has_horizontal_reflection(
    improper: Sequence[np.ndarray], axis: np.ndarray
) -> bool:
    for operation in improper:
        if not _is_reflection(operation):
            continue
        alignment = abs(float(np.dot(_reflection_normal(operation), axis)))
        if alignment >= 0.9:
            return True
    return False


def _has_vertical_reflection(
    improper: Sequence[np.ndarray], axis: np.ndarray
) -> bool:
    for operation in improper:
        if not _is_reflection(operation):
            continue
        alignment = abs(float(np.dot(_reflection_normal(operation), axis)))
        if alignment <= 0.1:
            return True
    return False


def _classify_finite(operations: Sequence[np.ndarray]) -> str:
    proper = [
        operation for operation in operations
        if np.linalg.det(operation) >= 0.0
    ]
    improper = [
        operation for operation in operations
        if np.linalg.det(operation) < 0.0
    ]

    proper_orders = [_operation_order(operation) for operation in proper]
    proper_order_set = set(proper_orders)
    proper_count = len(proper)
    inversion = any(_is_inversion(operation) for operation in improper)

    # Identify polyhedral groups from the full proper-rotation subgroup rather
    # than from inertia degeneracies. This avoids accidental spherical tops.
    if (
        proper_count == 60
        and proper_order_set == {1, 2, 3, 5}
    ):
        return "Ih" if improper else "I"

    if (
        proper_count == 24
        and proper_order_set == {1, 2, 3, 4}
    ):
        return "Oh" if improper else "O"

    if (
        proper_count == 12
        and proper_order_set == {1, 2, 3}
    ):
        if not improper:
            return "T"
        return "Th" if inversion else "Td"

    nonidentity_proper = [
        (operation, order)
        for operation, order in zip(proper, proper_orders)
        if order > 1
    ]

    if not nonidentity_proper:
        if not improper:
            return "C1"
        if inversion:
            return "Ci"
        return "Cs"

    main_operation, n = max(nonidentity_proper, key=lambda item: item[1])
    axis = _proper_axis(main_operation)

    # In D2d, all proper C2 axes have the same order. The S4 axis selects the
    # conventional principal axis and prevents a spurious D2h assignment.
    if n == 2:
        higher_improper = [
            operation
            for operation in improper
            if _operation_order(operation) > 2
        ]
        if higher_improper:
            axis = _improper_axis(
                max(higher_improper, key=_operation_order)
            )

    is_dihedral = (
        proper_count >= 4 if n == 2 else proper_count >= 2 * n
    )

    horizontal = _has_horizontal_reflection(improper, axis)
    vertical = _has_vertical_reflection(improper, axis)

    if is_dihedral:
        if not improper:
            return f"D{n}"
        if horizontal:
            return f"D{n}h"
        return f"D{n}d"

    if not improper:
        return f"C{n}"
    if vertical:
        return f"C{n}v"
    if horizontal:
        return f"C{n}h"

    improper_orders = [_operation_order(operation) for operation in improper]
    s_order = max(improper_orders)
    return f"S{s_order}"


def _is_linear(coordinates: np.ndarray, tolerance: float) -> bool:
    radii = np.linalg.norm(coordinates, axis=1)
    farthest = int(np.argmax(radii))
    if radii[farthest] <= tolerance:
        return True

    axis = coordinates[farthest] / radii[farthest]
    projections = coordinates @ axis
    perpendicular = coordinates - projections[:, None] * axis
    maximum_deviation = float(np.max(np.linalg.norm(perpendicular, axis=1)))
    return maximum_deviation <= tolerance


def _has_linear_inversion(
    coordinates: np.ndarray,
    symbols: Sequence[str],
    tolerance: float,
) -> bool:
    return (
        _perfect_matching(
            -coordinates,
            coordinates,
            symbols,
            tolerance * (1.0 + 1.0e-9) + 1.0e-10,
        )
        is not None
    )


def detect_point_group(
    xyz_input: str | pathlib.Path,
    tol: float = 1e-2,
) -> str:
    """
    Determine the Schoenflies point group of a molecule.

    Parameters
    ----------
    xyz_input : str | pathlib.Path
        Path to an XYZ file OR a raw XYZ-formatted multiline string.
    tol : float, optional
        Cartesian distance tolerance in ångströms (default: 1e-2).

    Returns
    -------
    str
        Exact ASCII Schoenflies point-group symbol.
    """
    if not isinstance(tol, (int, float, np.integer, np.floating)):
        raise TypeError("tol must be a positive real number")
    tolerance = float(tol)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tol must be a positive finite number")

    symbols, coordinates = _read_xyz(xyz_input)

    if len(symbols) == 1:
        return "Kh"

    # Every symmetry of a finite point set fixes its unweighted centroid.
    # Using the geometric rather than mass-weighted centroid also avoids any
    # dependency on an external isotope or atomic-mass table.
    coordinates = coordinates - np.mean(coordinates, axis=0)

    if _is_linear(coordinates, tolerance):
        if _has_linear_inversion(coordinates, symbols, tolerance):
            return "Dinfh"
        return "Cinfv"

    operations = _enumerate_operations(coordinates, symbols, tolerance)
    return _classify_finite(operations)


point_group = detect_point_group


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``pgmol``."""
    parser = argparse.ArgumentParser(
        prog="pgmol",
        description="Determine the Schoenflies point group of an XYZ structure.",
    )
    parser.add_argument("xyz_file", type=pathlib.Path, help="Path to an XYZ file")
    parser.add_argument(
        "--tol",
        type=float,
        default=1.0e-2,
        help="Cartesian matching tolerance in ångströms (default: 1e-2)",
    )
    arguments = parser.parse_args(argv)
    print(detect_point_group(arguments.xyz_file, arguments.tol))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
