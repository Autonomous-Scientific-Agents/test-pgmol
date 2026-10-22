"""Point-group determination from Cartesian XYZ coordinates.

The implementation uses only the Python standard library and NumPy.  Symmetry
operations are generated directly from the atomic coordinates and validated by
explicit atom-to-atom matching, making the result independent of orientation,
translation, and atom ordering.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


__all__ = ["detect_point_group", "point_group"]


_MASSES = {
    "H": 1.008,
    "D": 2.014,
    "T": 3.016,
    "He": 4.003,
    "Li": 6.941,
    "Be": 9.012,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.086,
    "P": 30.974,
    "S": 32.066,
    "Cl": 35.453,
    "Ar": 39.948,
    "K": 39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.39,
    "Ga": 69.723,
    "Ge": 72.64,
    "As": 74.922,
    "Se": 78.96,
    "Br": 79.904,
    "Kr": 83.80,
    "Rb": 85.468,
    "Sr": 87.62,
    "Y": 88.906,
    "Zr": 91.224,
    "Nb": 92.906,
    "Mo": 95.94,
    "Tc": 98.0,
    "Ru": 101.07,
    "Rh": 102.906,
    "Pd": 106.42,
    "Ag": 107.868,
    "Cd": 112.411,
    "In": 114.818,
    "Sn": 118.710,
    "Sb": 121.760,
    "Te": 127.60,
    "I": 126.904,
    "Xe": 131.293,
}


def _read_xyz(source: str | Path) -> tuple[list[str], np.ndarray]:
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
    else:
        text = str(source)
        if "\n" not in text and "\r" not in text:
            path = Path(text)
            if path.exists():
                text = path.read_text(encoding="utf-8")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("Invalid XYZ input")

    try:
        n = int(lines[0].split()[0])
    except Exception as exc:
        raise ValueError("The first XYZ line must contain the atom count") from exc

    if n < 1 or len(lines) < n + 2:
        raise ValueError("XYZ input contains fewer atoms than declared")

    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : n + 2]:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid XYZ atom line: {line!r}")
        symbols.append(fields[0])
        coords.append([float(fields[1]), float(fields[2]), float(fields[3])])

    return symbols, np.asarray(coords, dtype=float)


def _canonical_matrix(matrix: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(matrix, 8).ravel())


def _independent_basis(vectors: np.ndarray, tol: float) -> tuple[int, ...]:
    nonzero = [i for i, v in enumerate(vectors) if np.linalg.norm(v) > tol]
    if len(nonzero) < 2:
        return tuple(nonzero[:1])

    best_pair = None
    best_area = -1.0
    for a in range(len(nonzero)):
        for b in range(a + 1, len(nonzero)):
            i, j = nonzero[a], nonzero[b]
            area = np.linalg.norm(np.cross(vectors[i], vectors[j]))
            if area > best_area:
                best_area = area
                best_pair = (i, j)

    scale = max(np.linalg.norm(v) for v in vectors) or 1.0
    if best_pair is None or best_area <= tol * scale:
        return (nonzero[0],)

    i, j = best_pair
    best_triple = None
    best_volume = -1.0
    for k in nonzero:
        volume = abs(np.dot(vectors[i], np.cross(vectors[j], vectors[k])))
        if volume > best_volume:
            best_volume = volume
            best_triple = k

    if best_triple is not None and best_volume > tol * scale**3:
        return (i, j, best_triple)
    return (i, j)


def _is_symmetry(
    matrix: np.ndarray,
    vectors: np.ndarray,
    symbols: list[str],
    buckets: dict[str, list[int]],
    tol: float,
) -> bool:
    transformed = vectors @ matrix.T
    tol2 = tol * tol
    for i, point in enumerate(transformed):
        found = False
        for j in buckets[symbols[i]]:
            if np.sum((point - vectors[j]) ** 2) <= tol2:
                found = True
                break
        if not found:
            return False
    return True


def _generate_operations(
    vectors: np.ndarray,
    symbols: list[str],
    tol: float,
) -> list[np.ndarray]:
    n = len(vectors)
    buckets: dict[str, list[int]] = {}
    for i, symbol in enumerate(symbols):
        buckets.setdefault(symbol, []).append(i)

    basis = _independent_basis(vectors, tol)
    if not basis:
        return [np.eye(3)]

    operations: dict[tuple[float, ...], np.ndarray] = {}

    if len(basis) == 1:
        # This case is handled as linear by the caller.  Include the obvious
        # axial operations needed for inversion testing.
        operations[_canonical_matrix(np.eye(3))] = np.eye(3)
        operations[_canonical_matrix(-np.eye(3))] = -np.eye(3)
        return list(operations.values())

    source = vectors[list(basis)].T
    source_rank = np.linalg.matrix_rank(source, tol=tol)
    target_lists = [buckets[symbols[i]] for i in basis]

    if len(basis) == 3 and source_rank == 3:
        source_inv = np.linalg.inv(source)
        for i in target_lists[0]:
            for j in target_lists[1]:
                for k in target_lists[2]:
                    target = vectors[[i, j, k]].T
                    matrix = target @ source_inv
                    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=5 * tol):
                        continue
                    if _is_symmetry(matrix, vectors, symbols, buckets, tol):
                        operations[_canonical_matrix(matrix)] = matrix
    else:
        i0, i1 = basis[:2]
        a = vectors[i0]
        b = vectors[i1]
        ea = a / np.linalg.norm(a)
        bb = b - np.dot(b, ea) * ea
        eb = bb / np.linalg.norm(bb)
        en = np.cross(ea, eb)
        source_frame = np.column_stack((ea, eb, en))

        for j0 in target_lists[0]:
            for j1 in target_lists[1]:
                c = vectors[j0]
                d = vectors[j1]
                ec = c / np.linalg.norm(c)
                dd = d - np.dot(d, ec) * ec
                if np.linalg.norm(dd) <= tol:
                    continue
                ed = dd / np.linalg.norm(dd)
                ef = np.cross(ec, ed)
                target_frame = np.column_stack((ec, ed, ef))
                for sign in (1.0, -1.0):
                    matrix = (
                        target_frame
                        @ np.diag([1.0, 1.0, sign])
                        @ source_frame.T
                    )
                    if _is_symmetry(matrix, vectors, symbols, buckets, tol):
                        operations[_canonical_matrix(matrix)] = matrix

    if not operations:
        operations[_canonical_matrix(np.eye(3))] = np.eye(3)
    return list(operations.values())


def _proper_operations(operations: Iterable[np.ndarray]) -> list[np.ndarray]:
    return [m for m in operations if np.linalg.det(m) > 0.0]


def _matrix_order(matrix: np.ndarray, limit: int = 120) -> int:
    power = np.eye(3)
    for order in range(1, limit + 1):
        power = power @ matrix
        if np.allclose(power, np.eye(3), atol=2e-5):
            return order
    return limit


def _rotation_axis(matrix: np.ndarray) -> np.ndarray | None:
    vals, vecs = np.linalg.eig(matrix)
    candidates = [vecs[:, i].real for i, value in enumerate(vals) if abs(value - 1) < 1e-5]
    if not candidates:
        return None
    axis = candidates[0]
    norm = np.linalg.norm(axis)
    return axis / norm if norm else None


def _distinct_axes(operations: Iterable[np.ndarray]) -> list[np.ndarray]:
    axes: list[np.ndarray] = []
    for matrix in operations:
        if abs(np.trace(matrix) + 1.0) < 2e-5:
            axis = _rotation_axis(matrix)
            if axis is None:
                continue
            if not any(abs(np.dot(axis, other)) > 1 - 1e-4 for other in axes):
                axes.append(axis)
    return axes


def _classify(
    operations: list[np.ndarray],
    proper: list[np.ndarray],
    vectors: np.ndarray,
    tol: float,
) -> str:
    total = len(operations)
    pcount = len(proper)

    if len(vectors) == 1:
        return "Kh"

    scale = max(float(np.max(np.linalg.norm(vectors, axis=1))), 1.0)
    rank = np.linalg.matrix_rank(vectors, tol=max(tol, 1e-7) * scale)

    if rank <= 1:
        has_inversion = any(np.allclose(m, -np.eye(3), atol=5 * tol) for m in operations)
        return "Dinfh" if has_inversion else "Cinfv"

    proper_nonidentity = [
        m for m in proper if not np.allclose(m, np.eye(3), atol=5 * tol)
    ]
    orders = [_matrix_order(m) for m in proper_nonidentity]
    max_order = max(orders, default=1)

    # Polyhedral rotational subgroups.
    if pcount == 12:
        if total == 12:
            return "T"
        if any(np.allclose(m, -np.eye(3), atol=5 * tol) for m in operations):
            return "Th"
        return "Td"
    if pcount == 24:
        return "O" if total == 24 else "Oh"
    if pcount == 60:
        return "I" if total == 60 else "Ih"

    # Determine whether the proper subgroup is Cn or Dn.
    c2_axes = _distinct_axes(
        m for m in proper_nonidentity if _matrix_order(m) == 2
    )
    is_dihedral = pcount > 2 and (
        pcount == 2 * max_order or len(c2_axes) >= 3
    )

    principal = None
    if proper_nonidentity:
        candidates = [
            m for m in proper_nonidentity if _matrix_order(m) == max_order
        ]
        principal = _rotation_axis(candidates[0]) if candidates else None

    def is_reflection(m: np.ndarray) -> bool:
        return np.linalg.det(m) < 0 and np.allclose(m @ m, np.eye(3), atol=5 * tol)

    inversion = any(np.allclose(m, -np.eye(3), atol=5 * tol) for m in operations)
    reflections = [m for m in operations if is_reflection(m)]

    if not is_dihedral:
        n = max_order
        if pcount == 1:
            if total == 1:
                return "C1"
            if inversion:
                return "Ci"
            return "Cs"

        if total == n:
            return f"C{n}"

        if principal is not None:
            vertical = []
            horizontal = []
            for m in reflections:
                image = m @ principal
                if np.dot(image, principal) > 1 - 1e-4:
                    vertical.append(m)
                elif np.dot(image, principal) < -1 + 1e-4:
                    horizontal.append(m)
            if vertical:
                return f"C{n}v"
            if horizontal:
                return f"C{n}h"

        # Remaining cyclic groups are roto-reflection groups S_n.
        if n == 1:
            return "Cs" if reflections else "Ci"
        return f"S{2 * n}"

    n = max_order
    if total == pcount:
        return f"D{n}"

    horizontal = False
    if principal is not None:
        for m in reflections:
            if np.dot(m @ principal, principal) < -1 + 1e-4:
                horizontal = True
                break

    return f"D{n}h" if horizontal else f"D{n}d"


def detect_point_group(xyz_input: str | Path, tol: float = 1e-2) -> str:
    """Determine the Schoenflies point group of an XYZ molecule."""
    if tol <= 0:
        raise ValueError("tol must be positive")

    symbols, coordinates = _read_xyz(xyz_input)
    if len(symbols) == 1:
        return "Kh"

    masses = np.asarray([_MASSES.get(s, 1.0) for s in symbols], dtype=float)
    center = np.sum(coordinates * masses[:, None], axis=0) / np.sum(masses)
    vectors = coordinates - center

    operations = _generate_operations(vectors, symbols, tol)
    return _classify(operations, _proper_operations(operations), vectors, tol)


point_group = detect_point_group


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pgmol",
        description="Determine the Schoenflies point group of an XYZ molecule.",
    )
    parser.add_argument("xyz_file", help="Path to an XYZ file")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="Cartesian coordinate tolerance in angstroms",
    )
    args = parser.parse_args(argv)

    try:
        print(detect_point_group(Path(args.xyz_file), tol=args.tol))
    except Exception as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
