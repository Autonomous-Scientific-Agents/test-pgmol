"""Coordinate-based molecular Schoenflies point-group detection.

Requires only NumPy and the Python standard library.

The module can also be run directly::

    python -m pgmol molecule.xyz
"""

from __future__ import annotations

import argparse
import math
import pathlib
from collections import defaultdict

import numpy as np

__all__ = ["detect_point_group", "point_group"]


def _read_xyz(xyz_input: str | pathlib.Path) -> tuple[list[str], np.ndarray]:
    if isinstance(xyz_input, pathlib.Path):
        text = xyz_input.read_text(encoding="utf-8")
    elif isinstance(xyz_input, str):
        if "\n" in xyz_input or "\r" in xyz_input:
            text = xyz_input
        else:
            try:
                path = pathlib.Path(xyz_input)
                text = path.read_text(encoding="utf-8") if path.is_file() else xyz_input
            except OSError:
                text = xyz_input
    else:
        raise TypeError("xyz_input must be a string or pathlib.Path")

    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ input must contain an atom count and a comment line")

    try:
        count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("Invalid XYZ atom count") from exc

    if count < 1 or len(lines) < count + 2:
        raise ValueError("XYZ input has an invalid atom count or missing atom lines")

    symbols: list[str] = []
    coordinates: list[list[float]] = []
    for line in lines[2 : count + 2]:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid XYZ atom line: {line!r}")
        symbols.append(fields[0].capitalize())
        try:
            coordinates.append([float(value) for value in fields[1:4]])
        except ValueError as exc:
            raise ValueError(f"Invalid XYZ coordinates: {line!r}") from exc

    points = np.asarray(coordinates, dtype=float)
    if not np.isfinite(points).all():
        raise ValueError("XYZ coordinates must be finite")
    return symbols, points


def _unit(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        return None
    return vector / length


def _candidate_axes(
    points: np.ndarray, symbols: list[str], tol: float
) -> list[np.ndarray]:
    """Construct rotation-axis candidates from intrinsic coordinate vectors.

    In particular, equal-length edges meeting at a vertex of a rotational
    orbit supply its axis even when the inertia tensor is spherical.
    """
    axes: dict[tuple[int, int, int], np.ndarray] = {}

    def add(vector: np.ndarray) -> None:
        axis = _unit(vector)
        if axis is None:
            return
        for component in axis:
            if abs(component) > 1e-10:
                if component < 0:
                    axis = -axis
                break
        key = tuple(int(round(float(value) * 1_000_000)) for value in axis)
        axes.setdefault(key, axis)

    count = len(points)
    radii = np.linalg.norm(points, axis=1)

    for point in points:
        add(point)

    # Principal inertia directions are useful candidates, but are never
    # treated as evidence that a symmetry actually exists.
    _, eigenvectors = np.linalg.eigh(points.T @ points)
    for column in eigenvectors.T:
        add(column)

    for i in range(count):
        for j in range(i + 1, count):
            add(np.cross(points[i], points[j]))
            if symbols[i] == symbols[j] and abs(radii[i] - radii[j]) <= 2 * tol:
                add(points[i] + points[j])

    for i in range(count):
        groups: dict[str, list[int]] = defaultdict(list)
        for j in range(count):
            if i != j and symbols[i] == symbols[j]:
                if abs(radii[i] - radii[j]) <= 2 * tol:
                    groups[symbols[j]].append(j)

        for members in groups.values():
            for position, j in enumerate(members):
                first = points[j] - points[i]
                first_length = float(np.linalg.norm(first))
                for k in members[position + 1 :]:
                    second = points[k] - points[i]
                    if abs(first_length - np.linalg.norm(second)) <= 2 * tol:
                        add(np.cross(first, second))

    return list(axes.values())


class _Symmetry:
    def __init__(self, symbols: list[str], points: np.ndarray, tol: float):
        self.points = points
        self.tol = tol
        self.groups: list[np.ndarray] = []
        by_symbol: dict[str, list[int]] = defaultdict(list)
        for index, symbol in enumerate(symbols):
            by_symbol[symbol].append(index)
        self.groups = [np.asarray(indices) for indices in by_symbol.values()]

    def accepts(self, matrix: np.ndarray) -> bool:
        transformed = self.points @ matrix.T
        threshold = self.tol * self.tol * (1 + 1e-10)

        for indices in self.groups:
            targets = self.points[indices]
            mapped = transformed[indices]
            distances = np.sum(
                (mapped[:, None, :] - targets[None, :, :]) ** 2, axis=2
            )
            choices = [np.flatnonzero(row <= threshold) for row in distances]
            if any(len(options) == 0 for options in choices):
                return False

            # A bijection is required: independently matching each atom to
            # its nearest neighbor could accept a false symmetry.
            assignment = [-1] * len(indices)

            def augment(source: int, visited: set[int]) -> bool:
                for target in choices[source]:
                    target = int(target)
                    if target in visited:
                        continue
                    visited.add(target)
                    previous = assignment[target]
                    if previous == -1 or augment(previous, visited):
                        assignment[target] = source
                        return True
                return False

            for source in sorted(range(len(indices)), key=lambda x: len(choices[x])):
                if not augment(source, set()):
                    return False

        return True


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    sine = math.sin(angle)
    cosine = math.cos(angle)
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (
        cosine * np.eye(3)
        + (1 - cosine) * np.outer(axis, axis)
        + sine * cross
    )


def _max_possible_order(
    axis: np.ndarray, points: np.ndarray, symbols: list[str], tol: float
) -> int:
    """Upper-bound an axis order using atoms' heights and radial distances."""
    heights = points @ axis
    radial = np.linalg.norm(points - heights[:, None] * axis, axis=1)
    off_axis = np.flatnonzero(radial > tol)
    if len(off_axis) == 0:
        return 1

    largest = 1
    for i in off_axis:
        matching = sum(
            symbols[j] == symbols[i]
            and abs(heights[j] - heights[i]) <= 3 * tol
            and abs(radial[j] - radial[i]) <= 3 * tol
            for j in off_axis
        )
        largest = max(largest, matching)
    return largest


def _mirror_normals(
    points: np.ndarray, symbols: list[str], tol: float
) -> list[np.ndarray]:
    normals: dict[tuple[int, int, int], np.ndarray] = {}

    def add(vector: np.ndarray) -> None:
        normal = _unit(vector)
        if normal is None:
            return
        for component in normal:
            if abs(component) > 1e-10:
                if component < 0:
                    normal = -normal
                break
        key = tuple(int(round(float(value) * 1_000_000)) for value in normal)
        normals.setdefault(key, normal)

    radii = np.linalg.norm(points, axis=1)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            # A mirror either fixes atoms in its plane or exchanges pairs.
            add(np.cross(points[i], points[j]))
            if symbols[i] == symbols[j] and abs(radii[i] - radii[j]) <= 2 * tol:
                add(points[i] - points[j])

    _, eigenvectors = np.linalg.eigh(points.T @ points)
    for normal in eigenvectors.T:
        add(normal)

    return list(normals.values())


def detect_point_group(xyz_input: str | pathlib.Path, tol: float = 1e-2) -> str:
    """Determine the ASCII Schoenflies point group from XYZ coordinates.

    ``xyz_input`` may be a file path or a raw multiline XYZ string. ``tol``
    is a Cartesian matching tolerance in ångströms.
    """
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")

    symbols, points = _read_xyz(xyz_input)
    if len(points) == 1:
        return "Kh"

    # The geometric centroid is fixed by every permutation of the atoms,
    # irrespective of their elements or masses.
    points = points - points.mean(axis=0)
    symmetry = _Symmetry(symbols, points, tol)
    inversion = symmetry.accepts(-np.eye(3))

    singular_values = np.linalg.svd(points, compute_uv=False)
    if singular_values[1] <= tol:
        return "Dinfh" if inversion else "Cinfv"

    axes = _candidate_axes(points, symbols, tol)
    rotations: list[tuple[np.ndarray, int]] = []

    for axis in axes:
        upper = min(len(points), _max_possible_order(axis, points, symbols, tol))
        for order in range(upper, 1, -1):
            if symmetry.accepts(_rotation(axis, 2 * math.pi / order)):
                rotations.append((axis, order))
                break

    highest = max((order for _, order in rotations), default=1)

    normals = _mirror_normals(points, symbols, tol)
    mirrors = [
        normal
        for normal in normals
        if symmetry.accepts(np.eye(3) - 2 * np.outer(normal, normal))
    ]

    if highest >= 5:
        return "Ih" if inversion else "I"

    if highest == 4:
        fourth_axes = [axis for axis, order in rotations if order == 4]
        if len(fourth_axes) >= 2:
            return "Oh" if inversion else "O"

    if highest >= 3:
        third_axes = [axis for axis, order in rotations if order % 3 == 0]
        if len(third_axes) >= 2:
            if inversion:
                return "Th"
            return "Td" if mirrors else "T"

    if highest == 1:
        if inversion:
            return "Ci"
        return "Cs" if mirrors else "C1"

    principal = next(axis for axis, order in rotations if order == highest)
    perpendicular_twos = [
        axis
        for axis, order in rotations
        if order % 2 == 0 and abs(float(np.dot(axis, principal))) < 1e-4
    ]
    dihedral = bool(perpendicular_twos)

    horizontal = any(
        abs(float(np.dot(normal, principal))) > 1 - 1e-4
        for normal in mirrors
    )
    vertical = any(
        abs(float(np.dot(normal, principal))) < 1e-4
        for normal in mirrors
    )

    if dihedral:
        if horizontal:
            return f"D{highest}h"
        if vertical:
            return f"D{highest}d"
        return f"D{highest}"

    if horizontal:
        return f"C{highest}h"
    if vertical:
        return f"C{highest}v"

    horizontal_reflection = np.eye(3) - 2 * np.outer(principal, principal)
    improper = _rotation(principal, math.pi / highest) @ horizontal_reflection
    if symmetry.accepts(improper):
        return f"S{2 * highest}"
    return f"C{highest}"


point_group = detect_point_group


def main() -> None:
    parser = argparse.ArgumentParser(description="Determine an XYZ molecule's point group")
    parser.add_argument("xyz_file", type=pathlib.Path)
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="Cartesian matching tolerance in ångströms",
    )
    args = parser.parse_args()
    print(detect_point_group(args.xyz_file, tol=args.tol))


if __name__ == "__main__":
    main()
