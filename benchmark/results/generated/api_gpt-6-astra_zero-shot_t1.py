"""Coordinate-only molecular Schoenflies point-group detection.

Only NumPy and the Python standard library are required.

Public API
----------
detect_point_group(xyz_input, tol=1e-2)
point_group(xyz_input, tol=1e-2)

The console-script entry point is ``pgmol:main``. A package's
``__main__.py`` can delegate to this entry point with::

    from . import main
    raise SystemExit(main())

Algorithm
---------
After centering, every orthogonal symmetry of a non-linear configuration
is determined by the images of two independent position vectors and its
determinant. Enumerating these images avoids reliance on principal
moments, including their accidental degeneracies. Candidate operations
are checked using element-preserving, bijective Cartesian matching.
Classification uses the resulting operations, not molecular names,
formulas, XYZ comments, or filenames.

Tolerance is an absolute Cartesian distance in the input coordinate units,
normally angstroms. As with any floating-point symmetry detector, structures
on tolerance boundaries can be numerically ambiguous; approximate
symmetries need not themselves form a mathematical group.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from collections import deque
from typing import Sequence

import numpy as np

__all__ = ["detect_point_group", "point_group", "main"]

_Color = tuple[tuple[str, int], ...]
_Permutation = tuple[int, ...]
_Operation = tuple[np.ndarray, _Permutation]


def _read_xyz(
    xyz_input: str | pathlib.Path,
) -> tuple[list[str], np.ndarray]:
    """Read the declared atom records, without inspecting the comment."""
    if isinstance(xyz_input, pathlib.Path):
        text = xyz_input.read_text(encoding="utf-8")
    elif isinstance(xyz_input, str):
        if "\n" in xyz_input or "\r" in xyz_input:
            text = xyz_input
        else:
            text = pathlib.Path(xyz_input).read_text(encoding="utf-8")
    else:
        raise TypeError("xyz_input must be a string or pathlib.Path")

    lines = text.splitlines()
    if not lines:
        raise ValueError("Empty XYZ input")

    try:
        count = int(lines[0].lstrip("\ufeff").strip())
    except ValueError as exc:
        raise ValueError("The first XYZ line must be an atom count") from exc

    if count < 1:
        raise ValueError("An XYZ molecule must contain at least one atom")
    if len(lines) < count + 2:
        raise ValueError(
            f"XYZ declares {count} atoms but has too few atom records"
        )

    symbols: list[str] = []
    coordinates = np.empty((count, 3), dtype=np.longdouble)

    # lines[1] is deliberately never interpreted.
    for i, line in enumerate(lines[2 : count + 2]):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid XYZ atom record on line {i + 3}")

        symbol = fields[0]
        symbols.append(symbol[:1].upper() + symbol[1:].lower())
        try:
            coordinates[i] = [np.longdouble(s) for s in fields[1:4]]
        except (ValueError, OverflowError) as exc:
            raise ValueError(
                f"Invalid Cartesian coordinates on line {i + 3}"
            ) from exc

    if not np.isfinite(coordinates).all():
        raise ValueError("All Cartesian coordinates must be finite")

    return symbols, coordinates


def _prepare_sites(
    symbols: list[str], coordinates: np.ndarray
) -> tuple[list[_Color], np.ndarray]:
    """Combine exactly coincident sites while retaining element multiplicity."""
    sites: dict[tuple, dict[str, int]] = {}
    for symbol, row in zip(symbols, coordinates):
        key = tuple(row)
        counts = sites.setdefault(key, {})
        counts[symbol] = counts.get(symbol, 0) + 1

    colors: list[_Color] = []
    positions = np.empty((len(sites), 3), dtype=np.longdouble)
    for i, (position, counts) in enumerate(sites.items()):
        positions[i] = position
        colors.append(tuple(sorted(counts.items())))

    # Center in extended precision. Sorting each summation makes the
    # centroid independent of input row ordering, up to arithmetic precision.
    offset = positions.min(axis=0)
    shifted = positions - offset
    center = np.sort(shifted, axis=0).mean(
        axis=0, dtype=np.longdouble
    )
    with np.errstate(over="ignore", invalid="ignore"):
        centered = np.asarray(shifted - center, dtype=float)

    if not np.isfinite(centered).all():
        raise ValueError("Coordinate range is too large for computation")

    return colors, centered


def _color_blocks(colors: list[_Color]) -> list[np.ndarray]:
    groups: dict[_Color, list[int]] = {}
    for i, color in enumerate(colors):
        groups.setdefault(color, []).append(i)
    return [
        np.asarray(indices, dtype=np.intp)
        for indices in groups.values()
    ]


def _perfect_matching(
    squared_distances: np.ndarray, cutoff_squared: float
) -> np.ndarray | None:
    """Find a distance-bounded bijection, rather than a nearest-neighbor test."""
    size = len(squared_distances)
    nearest = squared_distances.argmin(axis=1)
    rows = np.arange(size)

    if np.any(squared_distances[rows, nearest] > cutoff_squared):
        return None

    if np.unique(nearest).size == size:
        return nearest

    allowed = squared_distances <= cutoff_squared
    if not np.all(allowed.any(axis=0)):
        return None

    neighbors: list[np.ndarray] = []
    for row in range(size):
        choices = np.flatnonzero(allowed[row])
        order = np.argsort(
            squared_distances[row, choices], kind="stable"
        )
        neighbors.append(choices[order])

    left_match = np.full(size, -1, dtype=np.intp)
    right_match = np.full(size, -1, dtype=np.intp)

    # Iterative augmenting paths avoid recursion limits for large inputs.
    for root in range(size):
        parent_right = np.full(size, -1, dtype=np.intp)
        seen_left = np.zeros(size, dtype=bool)
        seen_left[root] = True
        queue = deque([root])
        free_right = -1

        while queue and free_right < 0:
            left = queue.popleft()
            for right_value in neighbors[left]:
                right = int(right_value)
                if parent_right[right] >= 0:
                    continue
                parent_right[right] = left
                owner = int(right_match[right])
                if owner < 0:
                    free_right = right
                    break
                if not seen_left[owner]:
                    seen_left[owner] = True
                    queue.append(owner)

        if free_right < 0:
            return None

        right = free_right
        while right >= 0:
            left = int(parent_right[right])
            old_right = int(left_match[left])
            left_match[left] = right
            right_match[right] = left
            right = old_right

    return left_match


class _Geometry:
    def __init__(
        self, coordinates: np.ndarray, colors: list[_Color], tol: float
    ) -> None:
        self.x = coordinates
        self.colors = colors
        self.blocks = _color_blocks(colors)
        self.count = len(coordinates)
        self.radii = np.linalg.norm(coordinates, axis=1)
        self.radius = float(self.radii.max())
        # Only compensate for floating-point arithmetic, not coordinate noise.
        self.tol = tol + 64.0 * np.finfo(float).eps * self.radius
        self.identity = tuple(range(self.count))

    def match(
        self, transformed: np.ndarray, tolerance: float | None = None
    ) -> _Permutation | None:
        cutoff = self.tol if tolerance is None else tolerance
        permutation = np.empty(self.count, dtype=np.intp)

        for indices in self.blocks:
            differences = (
                transformed[indices, None, :]
                - self.x[indices][None, :, :]
            )
            squared = np.einsum(
                "ijk,ijk->ij", differences, differences
            )
            local = _perfect_matching(squared, cutoff * cutoff)
            if local is None:
                return None
            permutation[indices] = indices[local]

        return tuple(int(i) for i in permutation)

    def fit(
        self, permutation: _Permutation, determinant: int
    ) -> np.ndarray:
        """Origin-preserving orthogonal Procrustes fit with fixed determinant."""
        target = self.x[np.asarray(permutation, dtype=np.intp)]
        u, _, vt = np.linalg.svd(self.x.T @ target)
        v = vt.T
        correction = np.eye(3)
        orientation = 1 if np.linalg.det(v @ u.T) >= 0 else -1
        correction[2, 2] = determinant * orientation
        return v @ correction @ u.T

    def validate(
        self, matrix: np.ndarray, determinant: int
    ) -> _Operation | None:
        """Validate directly, then refine imperfect anchor-derived candidates."""
        transformed = self.x @ matrix.T
        permutation = self.match(transformed)
        if permutation is not None:
            return matrix, permutation

        # Anchor coordinates may each contain noise. A wider preliminary
        # match is used only to initialize a fit; final acceptance always
        # uses the requested Cartesian tolerance.
        preliminary = self.match(transformed, 6.0 * self.tol)
        if preliminary is None:
            return None

        seen: set[_Permutation] = set()
        for _ in range(5):
            if preliminary in seen:
                break
            seen.add(preliminary)
            fitted = self.fit(preliminary, determinant)
            transformed = self.x @ fitted.T

            permutation = self.match(transformed)
            if permutation is not None:
                return fitted, permutation

            preliminary = self.match(transformed, 6.0 * self.tol)
            if preliminary is None:
                break

        return None


def _frame(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    """Construct a right-handed frame from two independent vectors."""
    length = float(np.linalg.norm(first))
    if length == 0.0:
        return None

    e1 = first / length
    perpendicular = second - np.dot(second, e1) * e1
    height = float(np.linalg.norm(perpendicular))
    if height <= 32.0 * np.finfo(float).eps * max(
        length, float(np.linalg.norm(second))
    ):
        return None

    e2 = perpendicular / height
    e3 = np.cross(e1, e2)
    e3 /= np.linalg.norm(e3)
    # Reorthogonalize to reduce error for elongated configurations.
    e2 = np.cross(e3, e1)
    return np.column_stack((e1, e2, e3))


def _enumerate_operations(
    geometry: _Geometry,
) -> tuple[list[_Operation], list[_Operation]]:
    """Enumerate all orthogonal self-maps through two-vector images."""
    x = geometry.x
    radii = geometry.radii
    tolerance = geometry.tol

    first = int(np.argmax(radii))
    direction = x[first] / radii[first]
    perpendicular = x - np.outer(x @ direction, direction)
    heights = np.linalg.norm(perpendicular, axis=1)
    second = int(np.argmax(heights))

    reference = _frame(x[first], x[second])
    if reference is None:
        raise ValueError("Cannot resolve independent molecular directions")

    first_images = np.asarray(
        [
            i
            for i, color in enumerate(geometry.colors)
            if color == geometry.colors[first]
            and abs(radii[i] - radii[first]) <= 2.0 * tolerance
        ],
        dtype=np.intp,
    )
    second_images = np.asarray(
        [
            i
            for i, color in enumerate(geometry.colors)
            if color == geometry.colors[second]
            and abs(radii[i] - radii[second]) <= 2.0 * tolerance
        ],
        dtype=np.intp,
    )

    separation = float(np.linalg.norm(x[first] - x[second]))
    proper: dict[_Permutation, np.ndarray] = {
        geometry.identity: np.eye(3)
    }
    improper: dict[_Permutation, np.ndarray] = {}

    for image_first in first_images:
        distances = np.linalg.norm(
            x[second_images] - x[image_first], axis=1
        )
        eligible = second_images[
            np.abs(distances - separation) <= 2.0 * tolerance
        ]

        for image_second in eligible:
            if image_first == image_second:
                continue
            target = _frame(x[image_first], x[image_second])
            if target is None:
                continue

            for determinant, operations in (
                (1, proper),
                (-1, improper),
            ):
                oriented_target = target.copy()
                oriented_target[:, 2] *= determinant
                candidate = oriented_target @ reference.T

                operation = geometry.validate(candidate, determinant)
                if operation is None:
                    continue

                matrix, permutation = operation
                # On a non-collinear set, a permutation and determinant
                # uniquely specify its exact orthogonal operation.
                operations.setdefault(permutation, matrix)

    return (
        [(matrix, permutation) for permutation, matrix in proper.items()],
        [(matrix, permutation) for permutation, matrix in improper.items()],
    )


def _permutation_order(permutation: _Permutation) -> int:
    visited = [False] * len(permutation)
    result = 1
    for start in range(len(permutation)):
        if visited[start]:
            continue
        length = 0
        current = start
        while not visited[current]:
            visited[current] = True
            current = permutation[current]
            length += 1
        result = math.lcm(result, length)
    return result


def _rotation_axis(matrix: np.ndarray) -> np.ndarray:
    _, _, vt = np.linalg.svd(matrix - np.eye(3))
    axis = vt[-1]
    return axis / np.linalg.norm(axis)


def _is_reflection(operation: _Operation) -> bool:
    matrix, permutation = operation
    # An improper involution is either a reflection or inversion.
    # The trace separates them (respectively +1 and -3). A plane
    # reflection fixing an entirely planar molecule has permutation order 1.
    return (
        _permutation_order(permutation) <= 2
        and float(np.trace(matrix)) > 0.0
    )


def _classify(
    geometry: _Geometry,
    proper: list[_Operation],
    improper: list[_Operation],
) -> str:
    rotational_size = len(proper)
    has_inversion = geometry.match(-geometry.x) is not None

    if rotational_size == 1:
        if has_inversion:
            return "Ci"
        return "Cs" if improper else "C1"

    orders = [_permutation_order(p) for _, p in proper]
    principal_index = int(np.argmax(orders))
    n = orders[principal_index]

    # In an exact finite group every operation order divides its size.
    # A noisy, tolerance-boundary configuration can violate this condition.
    # Use the smallest resolved rotation angle as a conservative fallback.
    if n > rotational_size or rotational_size % n:
        angles = []
        for matrix, _ in proper:
            cosine = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
            angles.append(math.acos(float(cosine)))
        eligible = [
            i for i, angle in enumerate(angles) if angle > 1e-7
        ]
        if eligible:
            principal_index = min(eligible, key=angles.__getitem__)
            n = max(
                2,
                int(round(2.0 * math.pi / angles[principal_index])),
            )
        else:
            return "Ci" if has_inversion else ("Cs" if improper else "C1")

    # These signatures distinguish polyhedral rotational groups from
    # cyclic/dihedral groups of the same cardinality.
    if rotational_size == 12 and n == 3:
        if not improper:
            return "T"
        return "Th" if has_inversion else "Td"

    if rotational_size == 24 and n == 4:
        return "Oh" if improper else "O"

    if rotational_size == 60 and n == 5:
        return "Ih" if improper else "I"

    axis = _rotation_axis(proper[principal_index][0])
    dihedral = rotational_size == 2 * n

    if rotational_size not in (n, 2 * n):
        # Exact finite axial groups never need this branch. It provides a
        # useful best-effort answer when tolerance acceptance is not closed
        # under composition.
        for (matrix, _), order in zip(proper, orders):
            if order == 2:
                other_axis = _rotation_axis(matrix)
                if abs(float(np.dot(axis, other_axis))) < 0.1:
                    dihedral = True
                    break

    if not improper:
        return f"D{n}" if dihedral else f"C{n}"

    reflections = [op for op in improper if _is_reflection(op)]
    horizontal = any(
        float(axis @ matrix @ axis) < -0.9
        for matrix, _ in reflections
    )

    # A direct check also covers small numerical shifts in an extracted axis.
    if not horizontal:
        horizontal_matrix = np.eye(3) - 2.0 * np.outer(axis, axis)
        horizontal = (
            geometry.match(geometry.x @ horizontal_matrix.T) is not None
        )

    if dihedral:
        return f"D{n}h" if horizontal else f"D{n}d"

    if horizontal:
        return f"C{n}h"
    if reflections:
        return f"C{n}v"
    return f"S{2 * n}"


def detect_point_group(
    xyz_input: str | pathlib.Path, tol: float = 1e-2
) -> str:
    """Determine a molecule's ASCII Schoenflies point-group symbol.

    Parameters
    ----------
    xyz_input : str | pathlib.Path
        Path to an XYZ file, or a raw multiline XYZ string. The second
        line is ignored completely. Only the declared atom records are
        interpreted.
    tol : float, optional
        Maximum atom-matching Cartesian distance in angstroms, default
        ``1e-2``. Must be finite and strictly positive.

    Returns
    -------
    str
        A Schoenflies symbol, including ``Kh``, ``Cinfv``, ``Dinfh``,
        polyhedral groups, and finite cyclic/dihedral/improper groups.

    Raises
    ------
    TypeError
        If the input is not a string or pathlib.Path.
    ValueError
        If the XYZ content or tolerance is invalid.
    OSError
        If an input file cannot be read.

    Notes
    -----
    Atom identity is determined by its element symbol. Exactly coincident
    atoms are retained as multiplicities of their coordinate site.
    Noncoincident atoms are never merged by the distance tolerance.

    No principal-inertia degeneracy is used to infer a symmetry family.
    Planar structures are handled by enumerating both possible extensions
    of each in-plane isometry into three dimensions.
    """
    try:
        tolerance = float(tol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("tol must be finite and strictly positive") from exc
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tol must be finite and strictly positive")

    symbols, coordinates = _read_xyz(xyz_input)
    colors, centered = _prepare_sites(symbols, coordinates)

    if len(centered) == 1:
        return "Kh"

    geometry = _Geometry(centered, colors, tolerance)
    if geometry.radius == 0.0:
        return "Kh"

    # SVD is used only to resolve linearity, never to classify spherical tops.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    residual = centered - np.outer(centered @ axis, axis)
    if float(np.linalg.norm(residual, axis=1).max()) <= geometry.tol:
        return (
            "Dinfh"
            if geometry.match(-centered) is not None
            else "Cinfv"
        )

    proper, improper = _enumerate_operations(geometry)
    return _classify(geometry, proper, improper)


point_group = detect_point_group


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point: print the point group of an XYZ file."""
    parser = argparse.ArgumentParser(
        prog="pgmol",
        description="Determine a molecule's Schoenflies point group.",
    )
    parser.add_argument("xyz", type=pathlib.Path, help="input XYZ file")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        metavar="ANGSTROM",
        help="Cartesian matching tolerance (default: 0.01)",
    )
    arguments = parser.parse_args(argv)

    try:
        symbol = detect_point_group(arguments.xyz, arguments.tol)
    except (OSError, ValueError, TypeError, np.linalg.LinAlgError) as exc:
        print(f"pgmol: {exc}", file=sys.stderr)
        return 2

    print(symbol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
