"""Determine molecular Schoenflies point groups from XYZ coordinates.

The implementation uses only the Python standard library and NumPy. Symmetry
operations are found by matching geometry-preserving atom mappings and
orthogonal transformations; element symbols are preserved in every match.
"""

from __future__ import annotations

import argparse
import itertools
import math
import pathlib
from typing import Iterable

import numpy as np


__all__ = ["detect_point_group", "point_group"]


def _read_xyz(xyz_input: str | pathlib.Path) -> tuple[list[str], np.ndarray]:
    if isinstance(xyz_input, pathlib.Path):
        text = xyz_input.read_text(encoding="utf-8")
    elif isinstance(xyz_input, str):
        if "\n" in xyz_input or "\r" in xyz_input:
            text = xyz_input
        else:
            path = pathlib.Path(xyz_input)
            if not path.exists():
                raise FileNotFoundError(xyz_input)
            text = path.read_text(encoding="utf-8")
    else:
        raise TypeError("xyz_input must be a file path or raw XYZ string")

    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ input must contain an atom count and comment line")

    try:
        count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("Invalid atom count in XYZ input") from exc
    if count < 1:
        raise ValueError("XYZ input must contain at least one atom")
    if len(lines) < count + 2:
        raise ValueError("XYZ input contains fewer atom rows than declared")

    elements: list[str] = []
    coords: list[list[float]] = []
    for line_number, line in enumerate(lines[2 : count + 2], start=3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid atom row on line {line_number}")
        elements.append(fields[0])
        try:
            coords.append([float(fields[1]), float(fields[2]), float(fields[3])])
        except ValueError as exc:
            raise ValueError(f"Invalid coordinates on line {line_number}") from exc

    xyz = np.asarray(coords, dtype=float)
    if not np.all(np.isfinite(xyz)):
        raise ValueError("XYZ coordinates must be finite")
    return elements, xyz


def _has_matching_permutation(
    transformed: np.ndarray,
    elements: list[str],
    tolerance: float,
) -> bool:
    """Test whether transformed atoms can be bijectively matched to the input."""
    n = len(elements)
    tolerance_sq = tolerance * tolerance
    adjacency: list[list[int]] = []

    for i in range(n):
        candidates = [
            j
            for j in range(n)
            if elements[i] == elements[j]
            and float(np.dot(transformed[i] - transformed[j],
                             transformed[i] - transformed[j]))
            <= tolerance_sq
        ]
        if not candidates:
            return False
        adjacency.append(candidates)

    # Assign the most constrained atoms first. This avoids the false negatives
    # that a simple nearest-neighbour greedy assignment can produce.
    order = sorted(range(n), key=lambda i: len(adjacency[i]))
    assigned_to = [-1] * n

    def augment(atom_index: int, seen: list[bool]) -> bool:
        for target in adjacency[atom_index]:
            if seen[target]:
                continue
            seen[target] = True
            if assigned_to[target] == -1 or augment(assigned_to[target], seen):
                assigned_to[target] = atom_index
                return True
        return False

    for atom_index in order:
        if not augment(atom_index, [False] * n):
            return False
    return True


def _orthogonalize(matrix: np.ndarray) -> np.ndarray:
    """Return the nearest orthogonal matrix to ``matrix``."""
    u, _, vt = np.linalg.svd(matrix)
    return u @ vt


def _add_operation(
    operations: list[np.ndarray],
    operation: np.ndarray,
    coords: np.ndarray,
    elements: list[str],
    tolerance: float,
) -> None:
    operation = _orthogonalize(operation)
    if not _has_matching_permutation(coords @ operation, elements, tolerance):
        return
    if not any(np.linalg.norm(operation - old, ord="fro") < 1.0e-5
               for old in operations):
        operations.append(operation)


def _best_rank_three_basis(coords: np.ndarray) -> tuple[int, int, int]:
    """Find a well-conditioned triple of atom position vectors."""
    n = len(coords)
    best_pair: tuple[int, int] | None = None
    best_area = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            area = float(np.linalg.norm(np.cross(coords[i], coords[j])))
            if area > best_area:
                best_area = area
                best_pair = (i, j)
    if best_pair is None or best_area == 0.0:
        raise ValueError("Could not find a rank-three coordinate basis")

    i, j = best_pair
    normal = np.cross(coords[i], coords[j])
    best_k = -1
    best_volume = 0.0
    for k in range(n):
        if k == i or k == j:
            continue
        volume = abs(float(np.dot(normal, coords[k])))
        if volume > best_volume:
            best_volume = volume
            best_k = k
    if best_k < 0 or best_volume == 0.0:
        raise ValueError("Could not find a rank-three coordinate basis")
    return i, j, best_k


def _enumerate_rank_three_operations(
    coords: np.ndarray,
    elements: list[str],
    tolerance: float,
) -> list[np.ndarray]:
    n = len(coords)
    basis_indices = _best_rank_three_basis(coords)
    source = coords[list(basis_indices)]
    gram = source @ source.T
    scale = max(float(np.max(np.linalg.norm(coords, axis=1))), 1.0)
    gram_tol = 4.0 * tolerance * scale + tolerance * tolerance

    candidates: list[list[int]] = []
    for source_index in basis_indices:
        candidates.append([
            j for j in range(n)
            if elements[j] == elements[source_index]
            and abs(float(np.dot(coords[j], coords[j]) - gram[
                len(candidates), len(candidates)
            ])) <= gram_tol
        ])

    operations: list[np.ndarray] = []
    for a in candidates[0]:
        va = coords[a]
        if abs(float(np.dot(va, va) - gram[0, 0])) > gram_tol:
            continue
        for b in candidates[1]:
            vb = coords[b]
            if abs(float(np.dot(va, vb) - gram[0, 1])) > gram_tol:
                continue
            if abs(float(np.dot(vb, vb) - gram[1, 1])) > gram_tol:
                continue
            for c in candidates[2]:
                vc = coords[c]
                if (
                    abs(float(np.dot(va, vc) - gram[0, 2])) > gram_tol
                    or abs(float(np.dot(vb, vc) - gram[1, 2])) > gram_tol
                    or abs(float(np.dot(vc, vc) - gram[2, 2])) > gram_tol
                ):
                    continue
                target = np.vstack((va, vb, vc))
                try:
                    row_transform = np.linalg.solve(source, target)
                except np.linalg.LinAlgError:
                    continue
                _add_operation(
                    operations, row_transform, coords, elements, tolerance
                )

    return operations


def _enumerate_rank_two_operations(
    coords: np.ndarray,
    elements: list[str],
    tolerance: float,
) -> list[np.ndarray]:
    n = len(coords)
    best_pair: tuple[int, int] | None = None
    best_area = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            area = float(np.linalg.norm(np.cross(coords[i], coords[j])))
            if area > best_area:
                best_area = area
                best_pair = (i, j)
    if best_pair is None:
        return [np.eye(3)]

    i, j = best_pair
    e1 = coords[i] / np.linalg.norm(coords[i])
    v = coords[j] - np.dot(coords[j], e1) * e1
    if np.linalg.norm(v) == 0.0:
        return [np.eye(3)]
    e2 = v / np.linalg.norm(v)
    normal = np.cross(e1, e2)
    basis = np.column_stack((e1, e2, normal))
    planar = coords @ basis[:, :2]

    source = planar[[i, j]]
    gram = source @ source.T
    scale = max(float(np.max(np.linalg.norm(coords, axis=1))), 1.0)
    gram_tol = 4.0 * tolerance * scale + tolerance * tolerance

    candidate_lists: list[list[int]] = []
    for source_index, diagonal in ((i, 0), (j, 1)):
        candidate_lists.append([
            k for k in range(n)
            if elements[k] == elements[source_index]
            and abs(float(np.dot(planar[k], planar[k]) - gram[diagonal, diagonal]))
            <= gram_tol
        ])

    operations: list[np.ndarray] = []
    for a in candidate_lists[0]:
        va = planar[a]
        for b in candidate_lists[1]:
            vb = planar[b]
            if (
                abs(float(np.dot(va, vb) - gram[0, 1])) > gram_tol
                or abs(float(np.dot(vb, vb) - gram[1, 1])) > gram_tol
            ):
                continue
            target = np.vstack((va, vb))
            try:
                transform_2d = _orthogonalize(np.linalg.solve(source, target))
            except np.linalg.LinAlgError:
                continue
            for normal_sign in (1.0, -1.0):
                local = np.eye(3)
                local[:2, :2] = transform_2d
                local[2, 2] = normal_sign
                operation = basis @ local @ basis.T
                _add_operation(
                    operations, operation, coords, elements, tolerance
                )
    return operations


def _rotation_order(operation: np.ndarray) -> int:
    """Return the order of a proper rotation matrix."""
    angle_cos = min(1.0, max(-1.0, (float(np.trace(operation)) - 1.0) / 2.0))
    angle = math.acos(angle_cos)
    if angle < 1.0e-5:
        return 1
    return max(2, int(round(2.0 * math.pi / angle)))


def _axis_of_rotation(operation: np.ndarray) -> np.ndarray | None:
    values, vectors = np.linalg.eig(operation.T)
    index = int(np.argmin(np.abs(values - 1.0)))
    vector = np.real(vectors[:, index])
    length = float(np.linalg.norm(vector))
    if length < 1.0e-8:
        return None
    return vector / length


def _is_mirror(operation: np.ndarray) -> bool:
    return float(np.linalg.det(operation)) < 0.0 and abs(
        float(np.trace(operation)) - 1.0
    ) < 1.0e-4


def _has_horizontal_mirror(
    operations: Iterable[np.ndarray],
    axis: np.ndarray | None,
) -> bool:
    if axis is None:
        return False
    for operation in operations:
        if not _is_mirror(operation):
            continue
        # A horizontal mirror reverses the principal axis.
        if np.linalg.norm(operation @ axis + axis) < 1.0e-4:
            return True
    return False


def _has_vertical_mirror(
    operations: Iterable[np.ndarray],
    axis: np.ndarray | None,
) -> bool:
    if axis is None:
        return False
    for operation in operations:
        if not _is_mirror(operation):
            continue
        # A vertical mirror leaves the principal axis in its plane.
        if np.linalg.norm(operation @ axis - axis) < 1.0e-4:
            return True
    return False


def _classify_operations(operations: list[np.ndarray]) -> str:
    proper = [
        operation for operation in operations
        if float(np.linalg.det(operation)) > 0.0
    ]
    improper = [
        operation for operation in operations
        if float(np.linalg.det(operation)) < 0.0
    ]
    proper_count = len(proper)
    total_count = len(operations)

    if proper_count == 0:
        return "C1"

    orders = [_rotation_order(operation) for operation in proper]
    max_order = max(orders)

    # The finite proper polyhedral rotation groups have characteristic orders.
    if proper_count == 12 and max_order == 3:
        if not improper:
            return "T"
        if any(np.linalg.norm(operation + np.eye(3), ord="fro") < 1.0e-4
               for operation in improper):
            return "Th"
        return "Td"
    if proper_count == 24 and max_order == 4:
        return "Oh" if improper else "O"
    if proper_count == 60 and max_order == 5:
        return "Ih" if improper else "I"

    # Determine whether the proper subgroup is cyclic or dihedral.
    cyclic_order = max_order
    is_cyclic = proper_count == cyclic_order
    is_dihedral = proper_count == 2 * cyclic_order

    if not is_cyclic and not is_dihedral:
        # This fallback is useful for degenerate numerical inputs; it avoids
        # mistaking a spherical moment tensor for a polyhedral symmetry group.
        is_cyclic = proper_count == 1
        is_dihedral = not is_cyclic and proper_count % 2 == 0

    # Select a highest-order axis. For D2 and similar groups all axes can be
    # equivalent; the classification below is independent of that choice.
    principal_axis = None
    for operation, order in zip(proper, orders):
        if order == max_order and order > 1:
            principal_axis = _axis_of_rotation(operation)
            if principal_axis is not None:
                break

    if not improper:
        if is_cyclic:
            return f"C{cyclic_order}"
        if is_dihedral:
            return f"D{cyclic_order}"
        return f"C{max(1, cyclic_order)}"

    has_horizontal = _has_horizontal_mirror(operations, principal_axis)
    has_vertical = _has_vertical_mirror(operations, principal_axis)

    if is_cyclic:
        if cyclic_order == 1:
            # With C1 as the proper subgroup, the only possible added operation
            # is inversion or a reflection.
            if any(np.linalg.norm(operation + np.eye(3), ord="fro") < 1.0e-4
                   for operation in improper):
                return "Ci"
            return "Cs"

        if has_horizontal and has_vertical:
            # This case occurs for the small, highly overlapping axial groups.
            return f"D{cyclic_order}h"
        if has_horizontal:
            return f"C{cyclic_order}h"
        if has_vertical:
            return f"C{cyclic_order}v"

        # A cyclic proper subgroup with an improper coset and no mirror planes
        # is an improper-rotation group S_(2n).
        if total_count == 2 * cyclic_order:
            return f"S{2 * cyclic_order}"
        return f"C{cyclic_order}"

    if is_dihedral:
        if has_horizontal:
            return f"D{cyclic_order}h"
        if has_vertical:
            return f"D{cyclic_order}d"
        # In the usual Schoenflies families a dihedral improper extension has
        # either horizontal or diagonal/vertical mirror planes.
        if total_count == 4 * cyclic_order:
            return f"D{cyclic_order}d"
        return f"D{cyclic_order}"

    return "C1"


def detect_point_group(
    xyz_input: str | pathlib.Path,
    tol: float = 1e-2,
) -> str:
    """Determine the Schoenflies point group of a molecule.

    Parameters
    ----------
    xyz_input:
        Path to an XYZ file or a raw multiline XYZ-formatted string.
    tol:
        Cartesian distance tolerance in ångströms.

    Returns
    -------
    str
        Schoenflies point group symbol.
    """
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be a positive finite distance")

    elements, xyz = _read_xyz(xyz_input)
    if len(elements) == 1:
        return "Kh"

    # Translation invariance: all symmetry operations are tested about the
    # geometric centre, not about the input coordinate origin.
    coords = xyz - np.mean(xyz, axis=0)
    singular_values = np.linalg.svd(coords, compute_uv=False)
    rank = int(np.count_nonzero(singular_values > tol))

    if rank == 0:
        return "Kh"

    if rank == 1:
        if _has_matching_permutation(-coords, elements, tol):
            return "Dinfh"
        return "Cinfv"

    if rank == 2:
        operations = _enumerate_rank_two_operations(coords, elements, tol)
    else:
        operations = _enumerate_rank_three_operations(coords, elements, tol)

    # The identity should always be present, including in borderline cases.
    _add_operation(operations, np.eye(3), coords, elements, tol)
    return _classify_operations(operations)


point_group = detect_point_group


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for ``pgmol``."""
    parser = argparse.ArgumentParser(
        description="Determine a molecule's Schoenflies point group from XYZ."
    )
    parser.add_argument("xyz_file", help="path to an XYZ file")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="Cartesian matching tolerance in angstroms (default: 0.01)",
    )
    args = parser.parse_args(argv)
    print(detect_point_group(args.xyz_file, tol=args.tol))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
