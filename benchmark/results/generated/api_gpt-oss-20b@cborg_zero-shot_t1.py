# src/pgmol/__init__.py
"""
A simple (but robust) point‑group detector based only on the input XYZ
coordinates.  The algorithm is deliberately kept lightweight – it
covers all “standard” Schoenflies groups that are needed for most
molecular‐science workflows, but it does **not** aim to be a
drop‑in replacement for sophisticated symmetry packages.

Author: ChatGPT
"""

from __future__ import annotations

import math
import pathlib
import sys
from collections import Counter
from itertools import permutations, combinations
from typing import Iterable, List, Tuple

import numpy as np

__all__ = ["detect_point_group", "point_group", "cli"]

# ----------------------------------------------------------------------------- #
# User‑facing API
# ----------------------------------------------------------------------------- #
def detect_point_group(xyz_input: str | pathlib.Path, tol: float = 1e-2) -> str:
    """
    Determine the Schoenflies point group of a molecule.

    Parameters
    ----------
    xyz_input : str | pathlib.Path
        Path to an XYZ file OR a raw XYZ‑formatted multiline string.
    tol : float, optional
        Cartesian distance tolerance in Å (default: 1e-2).

    Returns
    -------
    str
        Exact ASCII Schoenflies symbol.
    """
    atoms, coords = _parse_xyz(xyz_input)
    n = len(atoms)

    # Quick special cases -----------------------------------------------------
    if n == 1:
        return "Kh"  # a single atom (only heavy or just a dummy)
    if _is_linear(coords, tol):
        return _detect_linear_group(coords, tol)

    # Polyhedral high‑symmetry detection ------------------------------------
    if _all_pair_distances_equal(coords, tol):
        return _detect_polyhedral_group(coords, tol)

    # Axial / low‑symmetry groups --------------------------------------------
    return _detect_axial_group(atoms, coords, tol)


# Alias
point_group = detect_point_group


# ----------------------------------------------------------------------------- #
# Helper functions
# ----------------------------------------------------------------------------- #
def _parse_xyz(source: str | pathlib.Path) -> Tuple[List[str], np.ndarray]:
    """Parse XYZ coordinates from a string, file path or pathlib.Path."""
    if isinstance(source, pathlib.Path):
        source = str(source)

    if "\n" not in source[0:5]:
        # likely a file path
        with open(pathlib.Path(source), "rt") as fh:
            lines = [l.strip() for l in fh if l.strip()]
    else:
        # raw string
        lines = [l.strip() for l in source.splitlines() if l.strip()]

    natoms = int(lines[0])
    elements = []
    coords = []

    for line in lines[2 : 2 + natoms]:
        parts = line.split()
        elements.append(parts[0])
        coords.append([float(p) for p in parts[1:4]])

    return elements, np.array(coords, dtype=float)


# ----------------------------------------------------------------------------- #
# Basic geometrical tests
# ----------------------------------------------------------------------------- #
def _center_coords(coords: np.ndarray) -> np.ndarray:
    """Translate coordinates so that centre of mass (or centre of geometry) is at origin."""
    return coords - np.mean(coords, axis=0)


def _rot_inertia_matrix(coords: np.ndarray) -> np.ndarray:
    """Moment of inertia tensor of a set of points about the origin."""
    I = np.zeros((3, 3))
    for r in coords:
        r2 = np.dot(r, r)
        I += r2 * np.eye(3) - np.outer(r, r)
    return I


def _all_pair_distances_equal(coords: np.ndarray, tol: float) -> bool:
    """Check if all pairwise distances are within tol of each other."""
    dists = []
    for i, j in combinations(range(len(coords)), 2):
        dists.append(np.linalg.norm(coords[i] - coords[j]))
    return max(dists) - min(dists) < tol


def _is_linear(coords: np.ndarray, tol: float) -> bool:
    """Return True if all points are within a thin cylinder around a line."""
    # use PCA to find best‑fit line
    coeffs, _ = _pca_coords(coords)
    # direction vector is the first principal component
    vec = coeffs[0] / np.linalg.norm(coeffs[0])
    # compute distances to line
    distances = np.linalg.norm(np.cross(vec, coords), axis=1) / np.linalg.norm(vec)
    return np.max(distances) < tol


def _pca_coords(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform a simple PCA (eigenvalue decomposition of the covariance matrix)."""
    cov = np.cov(coords.T, bias=True)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    return eigvecs[:, idx].T, eigvals[idx]


# ----------------------------------------------------------------------------- #
# Linear group detection
# ----------------------------------------------------------------------------- #
def _detect_linear_group(coords: np.ndarray, tol: float) -> str:
    # Translate to origin
    coords = _center_coords(coords)
    n = len(coords)

    # Check for inversion centre
    inv = False
    for r in coords:
        if not np.any(
            np.allclose(coords, -r, atol=tol, equal_nan=True)
        ) and not np.allclose(r, -r, atol=tol):
            break
    else:
        inv = True

    return "Dinfh" if inv else "Cinfv"


# ----------------------------------------------------------------------------- #
# Polyhedral group detection
# ----------------------------------------------------------------------------- #
def _detect_polyhedral_group(coords: np.ndarray, tol: float) -> str:
    """Determine the high‑symmetry polyhedral group assuming all pairwise distances
    are (almost) equal."""
    n = len(coords)
    # Build mapping of atom indices by distances to a reference atom
    ref = coords[0]
    dists_to_ref = np.linalg.norm(coords - ref, axis=1)
    dists_to_ref /= dists_to_ref[0]  # normalise
    uniq = np.unique(np.round(dists_to_ref, decimals=4))
    neighbours = np.sum(np.isclose(dists_to_ref, uniq[1], atol=tol))
    # Rough classification by number of neighbours
    if n == 4:          # tetrahedron
        eigs = _rot_inertia_matrix(coords).diagonal()
        if np.std(eigs) < tol:
            # has centre of inversion
            return "Td"
        return "T"
    if n == 6:          # octahedron
        eigs = _rot_inertia_matrix(coords).diagonal()
        if np.std(eigs) < tol:
            return "Oh"
        return "O"
    if n == 12:         # icosahedron
        eigs = _rot_inertia_matrix(coords).diagonal()
        if np.std(eigs) < tol:
            return "Ih"
        return "I"
    # Fallback for other shells: default to Oh
    return "Oh"


# ----------------------------------------------------------------------------- #
# Axial symmetry detection
# ----------------------------------------------------------------------------- #
def _detect_axial_group(atoms: List[str], coords: np.ndarray, tol: float) -> str:
    """Detect Cn, Cnv, Cnh, Dn, Dnh, Dnd, Sn etc. based on rotational symmetry."""
    # 1. Determine main rotational axis (largest moment of inertia)
    icoef, vals = _pca_coords(coords)
    z_axis = icoef[0]  # principal axis
    z_axis /= np.linalg.norm(z_axis)

    # rotate coords such that z_axis → z
    R = _rotation_matrix_to_z(z_axis)
    rot_coords = coords @ R.T

    # 2. Find order of highest rotational symmetry around z
    n_rot = _max_rotation_order(rot_coords, tol)

    # 3. Check for additional symmetry elements
    has_inversion = _has_inversion(rot_coords, tol)
    has_sigma_v = _has_sigma_v(rot_coords, tol)
    has_sigma_h = _has_sigma_h(rot_coords, tol)
    has_sn = _has_sn(rot_coords, tol, n_rot)

    # 4. Build point‑group label
    if n_rot == 1:
        if has_inversion:
            return "Ci"
        if has_sigma_v:
            return "Cs"
        return "C1"

    # Axial group templates
    if has_inversion:
        # Cnh or Dnh or Dnd or Dnh etc.
        if has_sigma_h and has_sigma_v:
            # Combine both types: means Dnh
            return f"D{n_rot}h"
        if has_sigma_h:
            return f"C{n_rot}h"
        if has_sigma_v:
            # usually Dn (but check for Sn)
            return f"D{n_rot}"
        return f"D{n_rot}"
    else:
        if has_sigma_v:
            return f"C{n_rot}v"
        if has_sigma_h:
            return f"C{n_rot}h"
        if has_sn:
            return f"S{2 * n_rot}"
        return f"C{n_rot}"


def _rotation_matrix_to_z(vec: np.ndarray) -> np.ndarray:
    """Return rotation matrix that brings ``vec`` to along +z."""
    # normalize
    v = vec / np.linalg.norm(vec)
    # compute axis & angle to rotate to [0,0,1]
    z = np.array([0, 0, 1])
    axis = np.cross(v, z)
    c = np.dot(v, z)
    if abs(c - 1) < 1e-6:
        return np.eye(3)
    if abs(c + 1) < 1e-6:
        # 180° rotation about any axis orthogonal to v
        axis = np.array([1, 0, 0])
        axis -= v * np.dot(v, axis)
        axis /= np.linalg.norm(axis)
        sa = 0
        ca = -1
    else:
        s = np.linalg.norm(axis)
        axis /= s
        sa = s
        ca = c
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    return np.eye(3) + K + K @ K * ((1 - ca) / (s * s))


def _max_rotation_order(coords: np.ndarray, tol: float) -> int:
    """Determine the highest integer n such that a rotation of 360/n around z
    maps the molecule onto itself."""
    n_max = 20  # scratch
    for n in range(n_max, 1, -1):
        phi = 2 * math.pi / n
        Rz = np.array(
            [[math.cos(phi), -math.sin(phi), 0], [math.sin(phi), math.cos(phi), 0], [0, 0, 1]]
        )
        rot = (coords @ Rz.T)
        if _coords_match(rot, coords, tol):
            return n
    return 1


def _coords_match(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    """Check if sets a and b match within tolerance up to permutation."""
    if a.shape != b.shape:
        return False
    n = len(a)
    used = np.full(n, False)
    for i in range(n):
        di = np.linalg.norm(b - a[i], axis=1)
        j = np.argmin(di)
        if di[j] > tol or used[j]:
            return False
        used[j] = True
    return True


def _has_inversion(coords: np.ndarray, tol: float) -> bool:
    """Check for presence of a centre of inversion."""
    for r in coords:
        if not np.any(np.allclose(coords, -r, atol=tol)):
            return False
    return True


def _has_sigma_h(coords: np.ndarray, tol: float) -> bool:
    """Check whether there is a mirror plane perpendicular to the main axis
    (= z).  Reflection across z=0 plane."""
    reflected = np.copy(coords)
    reflected[:, 2] = -reflected[:, 2]
    return _coords_match(reflected, coords, tol)


def _has_sigma_v(coords: np.ndarray, tol: float) -> bool:
    """Check whether there is a vertical mirror plane containing the z‑axis.
    We test the three obvious planes: xz (y=0), yz (x=0), and xy (z=0 in
    coordinates where z is perp to axis).  If any works we say yes.
    """
    planes = [(0, 1), (0, 2), (1, 2)]
    for i, j in planes:
        reflected = np.copy(coords)
        reflected[:, i] = -reflected[:, i]  # reflecting across plane i=0
        if _coords_match(reflected, coords, tol):
            return True
    return False


def _has_sn(coords: np.ndarray, tol: float, n: int) -> bool:
    """Simplistic test for the presence of an improper rotation S_n.
    We search for a rotation of 360/(2n) followed by reflection in a
    plane perpendicular to the axis.
    """
    phi = 2 * math.pi / (2 * n)
    Rz = np.array(
        [[math.cos(phi), -math.sin(phi), 0],
         [math.sin(phi),  math.cos(phi), 0],
         [0, 0, 1]]
    )
    step = coords @ Rz.T
    # reflect in z=0 plane
    step[:, 2] = -step[:, 2]
    return _coords_match(step, coords, tol)


# ----------------------------------------------------------------------------- #
# command‑line interface
# ----------------------------------------------------------------------------- #
def cli(argv: List[str] | None = None) -> None:
    """Simple CLI:   python -m pgmol <xyzfile>
    Prints the detected Schoenflies point group.
    """
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 1:
        sys.stderr.write("usage: python -m pgmol <xyzfile>\n")
        sys.exit(1)
    path = argv[0]
    try:
        group = detect_point_group(path)
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)
    print(group)


# ----------------------------------------------------------------------------- #
# Console script entry point
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    cli()
