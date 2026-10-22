"""
pgmol: coordinate-only molecular Schoenflies point-group detection.

Only numpy and the Python standard library are required.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from collections import defaultdict
from typing import Iterable

import numpy as np

__all__ = ["detect_point_group", "point_group"]

_EYE = np.eye(3)
_INV = -np.eye(3)


def _read_xyz(xyz_input: str | pathlib.Path) -> tuple[list[str], np.ndarray]:
    """Read an XYZ path or raw XYZ-formatted string."""
    if isinstance(xyz_input, pathlib.Path):
        text = xyz_input.read_text()
    else:
        value = str(xyz_input)
        if "\n" in value or "\r" in value:
            text = value
        else:
            path = pathlib.Path(value)
            if path.exists():
                text = path.read_text()
            else:
                raise FileNotFoundError(
                    f"{value!r} is neither an existing XYZ path nor a multiline XYZ string"
                )

    lines = text.splitlines()
    if not lines:
        raise ValueError("Empty XYZ input")

    try:
        natoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("First XYZ line must contain the atom count") from exc

    if natoms < 1:
        raise ValueError("XYZ atom count must be positive")
    if len(lines) < natoms + 2:
        raise ValueError("XYZ input is shorter than its declared atom count")

    symbols: list[str] = []
    coordinates: list[list[float]] = []
    for line_number, line in enumerate(lines[2 : natoms + 2], start=3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid XYZ atom line {line_number}: {line!r}")
        try:
            xyz = [float(fields[1]), float(fields[2]), float(fields[3])]
        except ValueError as exc:
            raise ValueError(f"Invalid coordinates on XYZ line {line_number}") from exc
        symbols.append(fields[0])
        coordinates.append(xyz)

    return symbols, np.asarray(coordinates, dtype=float)


def _perfect_matching(distance_matrix: np.ndarray, threshold: float) -> bool:
    """
    Determine whether every row can be assigned to a distinct compatible column.

    This small bipartite matcher avoids failure of greedy nearest-neighbour
    matching for geometrically repeated atoms.
    """
    n = distance_matrix.shape[0]
    compatible = [
        np.flatnonzero(distance_matrix[i] <= threshold).tolist() for i in range(n)
    ]
    if any(not choices for choices in compatible):
        return False

    order = sorted(range(n), key=lambda i: len(compatible[i]))
    assigned = [-1] * n

    def augment(row: int, visited: set[int]) -> bool:
        for col in compatible[row]:
            if col in visited:
                continue
            visited.add(col)
            if assigned[col] < 0 or augment(assigned[col], visited):
                assigned[col] = row
                return True
        return False

    for row in order:
        if not augment(row, set()):
            return False
    return True


def _matrix_order(matrix: np.ndarray, atol: float, maximum: int = 256) -> int | None:
    power = np.eye(3)
    for order in range(1, maximum + 1):
        power = power @ matrix
        if np.allclose(power, _EYE, atol=atol, rtol=0.0):
            return order
    return None


def _rotation_axis(matrix: np.ndarray) -> np.ndarray | None:
    """Return a normalized +1 eigenvector of a proper rotation."""
    values, vectors = np.linalg.eig(matrix)
    index = int(np.argmin(np.abs(values - 1.0)))
    if abs(values[index] - 1.0) > 1.0e-5:
        return None
    axis = np.real(vectors[:, index])
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-12:
        return None
    return axis / norm


def _mirror_normal(matrix: np.ndarray) -> np.ndarray | None:
    """Return the normal of a reflection matrix."""
    values, vectors = np.linalg.eig(matrix)
    index = int(np.argmin(np.abs(values + 1.0)))
    if abs(values[index] + 1.0) > 1.0e-5:
        return None
    normal = np.real(vectors[:, index])
    norm = float(np.linalg.norm(normal))
    if norm < 1.0e-12:
        return None
    return normal / norm


def _is_identity(matrix: np.ndarray, atol: float) -> bool:
    return bool(np.allclose(matrix, _EYE, atol=atol, rtol=0.0))


def _is_inversion(matrix: np.ndarray, atol: float) -> bool:
    return bool(np.allclose(matrix, _INV, atol=atol, rtol=0.0))


def _independent_pair(coords: np.ndarray, rank_tol: float) -> tuple[int, int]:
    """
    Select two non-collinear centered coordinate vectors.

    Molecules of rank >= 2 always possess such a pair.  Using a fixed source
    pair is sufficient: every molecular symmetry maps that pair onto an
    equivalent target pair.
    """
    nonzero = [
        i for i, point in enumerate(coords) if np.linalg.norm(point) > rank_tol
    ]
    for i in nonzero:
        for j in nonzero:
            if i >= j:
                continue
            if np.linalg.norm(np.cross(coords[i], coords[j])) > rank_tol:
                return i, j
    raise ValueError("Could not construct a non-collinear coordinate pair")


def _orthonormal_frame(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Construct right-handed orthonormal frame whose first two directions span u,v."""
    e1 = u / np.linalg.norm(u)
    residual = v - np.dot(v, e1) * e1
    e2 = residual / np.linalg.norm(residual)
    e3 = np.cross(e1, e2)
    return np.column_stack((e1, e2, e3))


def _symmetry_operations(
    symbols: list[str],
    coords: np.ndarray,
    tol: float,
) -> list[np.ndarray]:
    """
    Enumerate all finite orthogonal operations preserving labeled coordinates.

    For rank >= 2 every operation is determined by images of two independent
    atom vectors.  Both possible images of the normal are examined, so planar
    structures and improper operations are handled without special alignment.
    """
    n = len(symbols)
    groups: dict[str, np.ndarray] = {}
    for symbol in set(symbols):
        groups[symbol] = np.asarray(
            [i for i, item in enumerate(symbols) if item == symbol], dtype=int
        )

    scale = max(1.0, float(np.max(np.linalg.norm(coords, axis=1))))
    rank_tol = max(1.0e-10, tol * 1.0e-5)
    source_i, source_j = _independent_pair(coords, rank_tol)
    source_u = coords[source_i]
    source_v = coords[source_j]
    source_frame = _orthonormal_frame(source_u, source_v)

    radius_u = float(np.linalg.norm(source_u))
    radius_v = float(np.linalg.norm(source_v))
    dot_uv = float(np.dot(source_u, source_v))
    invariant_tol = max(tol * scale * 8.0, 1.0e-10)

    candidate_matrices: list[np.ndarray] = [_EYE.copy()]
    candidates_u = groups[symbols[source_i]]
    candidates_v = groups[symbols[source_j]]

    for target_i in candidates_u:
        target_u = coords[target_i]
        if abs(np.linalg.norm(target_u) - radius_u) > invariant_tol:
            continue

        for target_j in candidates_v:
            target_v = coords[target_j]
            if abs(np.linalg.norm(target_v) - radius_v) > invariant_tol:
                continue
            if abs(float(np.dot(target_u, target_v)) - dot_uv) > invariant_tol:
                continue
            if np.linalg.norm(np.cross(target_u, target_v)) <= rank_tol:
                continue

            target_frame = _orthonormal_frame(target_u, target_v)

            for handedness in (1.0, -1.0):
                signed_target = target_frame.copy()
                signed_target[:, 2] *= handedness
                operation = signed_target @ source_frame.T
                candidate_matrices.append(operation)

    accepted: list[np.ndarray] = []
    matrix_tol = max(2.0e-7, tol / max(scale, 1.0) * 0.2)

    for operation in candidate_matrices:
        if not np.allclose(operation.T @ operation, _EYE, atol=1.0e-7, rtol=0.0):
            continue

        valid = True
        transformed = coords @ operation.T
        for symbol, indices in groups.items():
            source_points = transformed[indices]
            target_points = coords[indices]
            differences = source_points[:, None, :] - target_points[None, :, :]
            distances = np.linalg.norm(differences, axis=2)
            if not _perfect_matching(distances, tol):
                valid = False
                break

        if not valid:
            continue

        if not any(np.allclose(operation, old, atol=matrix_tol, rtol=0.0) for old in accepted):
            accepted.append(operation)

    if not any(_is_identity(op, matrix_tol) for op in accepted):
        accepted.append(_EYE.copy())
    return accepted


def _classify_operations(operations: Iterable[np.ndarray], tol: float) -> str:
    operations = list(operations)
    matrix_tol = max(2.0e-6, tol * 0.2)

    proper = [op for op in operations if np.linalg.det(op) > 0.0]
    improper = [op for op in operations if np.linalg.det(op) < 0.0]

    proper_orders: list[tuple[np.ndarray, int]] = []
    for op in proper:
        order = _matrix_order(op, matrix_tol)
        if order is not None:
            proper_orders.append((op, order))

    max_proper_order = max((order for _, order in proper_orders), default=1)
    proper_count = len(proper)
    inversion = any(_is_inversion(op, matrix_tol) for op in improper)

    mirrors: list[tuple[np.ndarray, np.ndarray]] = []
    for op in improper:
        if _matrix_order(op, matrix_tol) == 2 and not _is_inversion(op, matrix_tol):
            normal = _mirror_normal(op)
            if normal is not None:
                mirrors.append((op, normal))

    # Cubic / icosahedral groups are recognized from their finite proper
    # rotational subgroup, not from inertia tensor degeneracy.
    if proper_count == 60 and max_proper_order == 5:
        return "Ih" if inversion else "I"
    if proper_count == 24 and max_proper_order == 4:
        return "Oh" if inversion else "O"
    if proper_count == 12 and max_proper_order == 3:
        if not improper:
            return "T"
        return "Th" if inversion else "Td"

    if proper_count == 1:
        if not improper:
            return "C1"
        if mirrors:
            return "Cs"
        if inversion:
            return "Ci"
        improper_orders = [
            order
            for op in improper
            if (order := _matrix_order(op, matrix_tol)) is not None and order > 2
        ]
        return f"S{max(improper_orders)}" if improper_orders else "C1"

    # Main proper rotational axis.  For D2-like groups any of the equivalent
    # C2 axes works; horizontal/vertical mirror tests remain valid.
    principal_axis: np.ndarray | None = None
    for op, order in proper_orders:
        if order == max_proper_order and order > 1:
            principal_axis = _rotation_axis(op)
            if principal_axis is not None:
                break

    n = max_proper_order
    is_dihedral = proper_count == 2 * n and n >= 2

    if is_dihedral:
        if not improper:
            return f"D{n}"

        horizontal = False
        if principal_axis is not None:
            for _, normal in mirrors:
                if abs(float(np.dot(normal, principal_axis))) > 0.85:
                    horizontal = True
                    break

        if horizontal:
            return f"D{n}h"
        if mirrors:
            return f"D{n}d"

        # A full dihedral extension normally contains mirrors if it contains
        # improper operations.  This fallback is chiefly defensive for noisy
        # borderline inputs.
        return f"D{n}h" if inversion else f"D{n}d"

    # Cyclic groups and pure improper-rotation groups.
    if not improper:
        return f"C{n}"

    if mirrors:
        horizontal = False
        if principal_axis is not None:
            for _, normal in mirrors:
                if abs(float(np.dot(normal, principal_axis))) > 0.85:
                    horizontal = True
                    break
        return f"C{n}h" if horizontal else f"C{n}v"

    improper_orders = [
        order
        for op in improper
        if (order := _matrix_order(op, matrix_tol)) is not None and order > 2
    ]
    if improper_orders:
        return f"S{max(improper_orders)}"

    if inversion and n == 1:
        return "Ci"
    return f"C{n}"


def detect_point_group(xyz_input: str | pathlib.Path, tol: float = 1e-2) -> str:
    """
    Determine the Schoenflies point group of a molecule.

    Parameters
    ----------
    xyz_input : str | pathlib.Path
        Path to an XYZ file OR a raw XYZ-formatted multiline string.
    tol : float, optional
        Cartesian distance tolerance in angstroms (default: 1e-2).

    Returns
    -------
    str
        Exact ASCII Schoenflies point-group symbol.
    """
    if tol <= 0:
        raise ValueError("tol must be positive")

    symbols, coordinates = _read_xyz(xyz_input)
    natoms = len(symbols)

    if natoms == 1:
        return "Kh"

    # The centroid is invariant under every symmetry operation because any
    # allowed operation only permutes atoms.  Equal weighting is intentional:
    # no isotope or atomic-mass assumptions are needed.
    coords = coordinates - np.mean(coordinates, axis=0)

    singular_values = np.linalg.svd(coords, compute_uv=False)
    scale = max(1.0, float(singular_values[0]))
    rank_threshold = max(1.0e-10, tol * 1.0e-5 * scale)
    rank = int(np.count_nonzero(singular_values > rank_threshold))

    if rank <= 1:
        # Every finite linear molecular geometry has an infinite principal
        # rotational axis.  Inversion distinguishes Dinfh from Cinfv.
        by_symbol: defaultdict[str, list[np.ndarray]] = defaultdict(list)
        for symbol, point in zip(symbols, coords):
            by_symbol[symbol].append(point)

        inverted = True
        for points in by_symbol.values():
            array = np.asarray(points)
            distances = np.linalg.norm(
                (-array)[:, None, :] - array[None, :, :], axis=2
            )
            if not _perfect_matching(distances, tol):
                inverted = False
                break
        return "Dinfh" if inverted else "Cinfv"

    operations = _symmetry_operations(symbols, coords, tol)
    return _classify_operations(operations, tol)


point_group = detect_point_group


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``pgmol`` and ``python -m pgmol``."""
    parser = argparse.ArgumentParser(
        prog="pgmol",
        description="Determine a molecule's Schoenflies point group from an XYZ file.",
    )
    parser.add_argument("xyz", type=pathlib.Path, help="Path to an XYZ file")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="Cartesian matching tolerance in angstroms (default: 1e-2)",
    )
    args = parser.parse_args(argv)
    print(detect_point_group(args.xyz, tol=args.tol))
    return 0


if __name__ == "__main__":
    sys.exit(main())
