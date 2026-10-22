# pgmol/__init__.py
"""
pgmol – minimal point‑group detection from XYZ coordinates.

Only the Python standard library and NumPy are used.
"""

from __future__ import annotations
import pathlib
import itertools
import math
from typing import Tuple, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Helper data
# ----------------------------------------------------------------------
# Approximate atomic masses (u); sufficient for inertia‑tensor based
# classification – the exact values are not crucial.
_ATOMIC_MASSES: Dict[str, float] = {
    "H": 1.0079, "He": 4.0026,
    "Li": 6.941, "Be": 9.0122, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085,
    "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078,
    # Add a few more for convenience
    "Fe": 55.845, "Cu": 63.546, "Zn": 65.38,
    "Br": 79.904, "I": 126.90,
}

# ----------------------------------------------------------------------
# Core utilities
# ----------------------------------------------------------------------
def _read_xyz(xyz_input: str | pathlib.Path) -> Tuple[List[str], np.ndarray]:
    """Parse an XYZ file or raw string.

    Returns a list of element symbols and an (N,3) array of coordinates.
    """
    if isinstance(xyz_input, pathlib.Path):
        txt = xyz_input.read_text()
    elif isinstance(xyz_input, str) and pathlib.Path(xyz_input).exists():
        txt = pathlib.Path(xyz_input).read_text()
    else:
        txt = xyz_input

    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("XYZ input is too short.")

    try:
        n_atoms = int(lines[0])
    except Exception as exc:
        raise ValueError("First line of XYZ must be the atom count.") from exc

    if n_atoms != len(lines) - 2:
        raise ValueError(f"XYZ header says {n_atoms} atoms, but {len(lines)-2} lines found.")

    symbols = []
    coords = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Bad XYZ line: {line!r}")
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return symbols, np.asarray(coords, dtype=float)


def _center(coords: np.ndarray) -> np.ndarray:
    """Translate coordinates to the centroid."""
    return coords - coords.mean(axis=0)


def _inertia_tensor(symbols: List[str], coords: np.ndarray) -> np.ndarray:
    """Mass‑weighted inertia tensor about the origin."""
    masses = np.array([_ATOMIC_MASSES.get(s, 12.0) for s in symbols])  # default to carbon
    I = np.zeros((3, 3))
    for m, r in zip(masses, coords):
        r2 = np.dot(r, r)
        I += m * (r2 * np.eye(3) - np.outer(r, r))
    return I


def _eigen_decomposition(I: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (sorted) and eigenvectors of inertia tensor."""
    vals, vecs = np.linalg.eigh(I)
    order = np.argsort(vals)
    return vals[order], vecs[:, order]  # eigenvectors are columns


def _are_equivalent(
    orig_sym: List[str],
    orig_coords: np.ndarray,
    cand_coords: np.ndarray,
    tol: float,
) -> bool:
    """Check whether *cand_coords* is a permutation of *orig_coords* with the same elements."""
    used = np.full(len(orig_coords), False, dtype=bool)
    for i, (sym_i, pt_i) in enumerate(zip(orig_sym, cand_coords)):
        # Find a matching original atom
        diffs = np.linalg.norm(orig_coords[~used] - pt_i, axis=1)
        mask = (diffs <= tol) & np.array([s == sym_i for s in np.array(orig_sym)[~used]])
        if not np.any(mask):
            return False
        # mark the first match as used
        idx_rel = np.where(mask)[0][0]
        idx_abs = np.flatnonzero(~used)[idx_rel]
        used[idx_abs] = True
    return True


def _apply_symmetry(
    coords: np.ndarray,
    sym: np.ndarray,
    symbols: List[str],
    tol: float,
) -> bool:
    """Transform *coords* by *sym* and test equality to original (up to permutation)."""
    transformed = coords @ sym.T
    # Re‑center because rotations may shift due to rounding errors
    transformed = _center(transformed)
    return _are_equivalent(symbols, coords, transformed, tol)


def _has_inversion_center(
    symbols: List[str], coords: np.ndarray, tol: float
) -> bool:
    """Check for a center of inversion at the origin."""
    inv = -np.eye(3)
    return _apply_symmetry(coords, inv, symbols, tol)


def _has_reflection(
    normal: np.ndarray,
    symbols: List[str],
    coords: np.ndarray,
    tol: float,
) -> bool:
    """Check for a mirror plane with given normal (must be unit)."""
    refl = np.eye(3) - 2.0 * np.outer(normal, normal)
    return _apply_symmetry(coords, refl, symbols, tol)


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix for *axis* (unit) and *angle* (rad)."""
    ux, uy, uz = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1 - c
    return np.array(
        [
            [c + ux * ux * C, ux * uy * C - uz * s, ux * uz * C + uy * s],
            [uy * ux * C + uz * s, c + uy * uy * C, uy * uz * C - ux * s],
            [uz * ux * C - uy * s, uz * uy * C + ux * s, c + uz * uz * C],
        ]
    )


def _has_proper_rotation(
    axis: np.ndarray,
    n: int,
    symbols: List[str],
    coords: np.ndarray,
    tol: float,
) -> bool:
    """True iff rotation by 2π/n about *axis* leaves molecule unchanged."""
    if n < 2:
        return False
    R = _rotation_matrix(axis, 2.0 * math.pi / n)
    return _apply_symmetry(coords, R, symbols, tol)


def _has_improper_rotation(
    axis: np.ndarray,
    n: int,
    symbols: List[str],
    coords: np.ndarray,
    tol: float,
) -> bool:
    """Sn axis: rotation by 2π/n followed by reflection through plane ⟂ axis."""
    if n < 2:
        return False
    R = _rotation_matrix(axis, 2.0 * math.pi / n)
    # reflection through plane perpendicular to axis
    refl = np.eye(3) - 2.0 * np.outer(axis, axis)
    S = refl @ R
    return _apply_symmetry(coords, S, symbols, tol)


def _candidate_axes(coords: np.ndarray) -> List[np.ndarray]:
    """Return a set of unique unit vectors defined by atom pairs (for symmetry testing)."""
    axes = []
    N = len(coords)
    for i, j in itertools.combinations(range(N), 2):
        vec = coords[i] - coords[j]
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            continue
        unit = vec / norm
        # keep only one direction for each line (±)
        if not any(np.allclose(unit, a) or np.allclose(-unit, a) for a in axes):
            axes.append(unit)
    return axes


# ----------------------------------------------------------------------
# Main detection routine
# ----------------------------------------------------------------------
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
    symbols, coords = _read_xyz(xyz_input)
    N = len(symbols)

    # Single atom -------------------------------------------------------
    if N == 1:
        return "Kh"

    # Center and analyse inertia tensor ---------------------------------
    coords = _center(coords)
    I, eigvecs = _eigen_decomposition(_inertia_tensor(symbols, coords))
    # sort eigenvalues (already sorted) – smallest corresponds to axis of highest symmetry
    # Align coordinates with principal axes for stable testing
    coords = coords @ eigvecs

    # Linear detection ---------------------------------------------------
    # Molecule is linear if all atoms lie on a single line (within tol)
    # Check variance orthogonal to the first principal axis
    if np.min(I) < 1e-6 * np.max(I):
        # Determine if there is an inversion centre
        if _has_inversion_center(symbols, coords, tol):
            return "Dinfh"
        else:
            return "Cinfv"

    # Basic symmetry elements -------------------------------------------
    has_i = _has_inversion_center(symbols, coords, tol)

    # Detection of proper rotations about principal axes
    # principal axis = eigenvector with smallest moment (largest symmetry)
    primary_axis = np.array([0.0, 0.0, 1.0])  # after alignment, z‑axis is principal
    max_order = 1
    for n in range(6, 1, -1):  # look for highest order up to 6 first
        if _has_proper_rotation(primary_axis, n, symbols, coords, tol):
            max_order = n
            break

    # Search for any higher order (5‑fold) – necessary for I groups
    for n in (5, 4):
        if _has_proper_rotation(primary_axis, n, symbols, coords, tol):
            max_order = max(max_order, n)

    # Determine presence of C2 axes perpendicular to primary
    perp_axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]  # x and y after alignment
    has_perp_c2 = any(_has_proper_rotation(ax, 2, symbols, coords, tol) for ax in perp_axes)

    # Horizontal mirror (σ_h) – plane perpendicular to primary axis (xy‑plane)
    sigma_h = _has_reflection(np.array([0.0, 0.0, 1.0]), symbols, coords, tol)

    # Vertical mirrors (σ_v) – planes containing primary axis
    sigma_v = any(
        _has_reflection(ax, symbols, coords, tol) for ax in perp_axes
    )

    # Dihedral mirrors (σ_d) – reflect through plane bisecting C2 axes
    # For a C2‑perpendicular set, σ_d exists iff an S4 operation is present
    sigma_d = False
    if has_perp_c2:
        # test S4 about primary axis
        sigma_d = _has_improper_rotation(primary_axis, 4, symbols, coords, tol)

    # Improper rotations Sn (primary axis only)
    sn_order = None
    for n in range(2, 7):
        if _has_improper_rotation(primary_axis, n, symbols, coords, tol):
            sn_order = n
            break

    # -------------------------------------------------------------------
    # 1. Polyhedral groups – detect based on existence of multiple high‑order axes
    # -------------------------------------------------------------------
    axes = _candidate_axes(coords)
    found_3fold = any(
        _has_proper_rotation(ax, 3, symbols, coords, tol) for ax in axes
    )
    found_4fold = any(
        _has_proper_rotation(ax, 4, symbols, coords, tol) for ax in axes
    )
    found_5fold = any(
        _has_proper_rotation(ax, 5, symbols, coords, tol) for ax in axes
    )

    if found_5fold:
        # Icosahedral groups
        return "Ih" if has_i else "I"
    if found_4fold and found_3fold:
        # Octahedral / tetrahedral mixture
        # Distinguish Td/Th vs O/Oh by presence of σ_h and inversion
        if has_i:
            return "Oh"
        if sigma_h:
            return "Oh"
        # check for S4 (characteristic of Td)
        if _has_improper_rotation(primary_axis, 4, symbols, coords, tol):
            return "Td"
        return "O"
    if found_3fold:
        # Tetrahedral families
        if has_i:
            return "Th"
        if _has_improper_rotation(primary_axis, 4, symbols, coords, tol):
            return "Td"
        return "T"

    # -------------------------------------------------------------------
    # 2. Cyclic and dihedral families
    # -------------------------------------------------------------------
    if max_order == 1:
        # No proper rotation axis
        if sn_order is not None:
            return f"S{sn_order}"
        if sigma_h or sigma_v:
            return "Cs"
        if has_i:
            return "Ci"
        return "C1"

    # At this point we have a Cn principal axis with order >=2
    n = max_order

    if not has_perp_c2:
        # Pure cyclic families
        if sigma_v:
            return f"C{n}v"
        if sigma_h:
            return f"C{n}h"
        return f"C{n}"
    else:
        # Dihedral families
        if sigma_h:
            return f"D{n}h"
        if sigma_d:
            return f"D{n}d"
        return f"D{n}"


# Alias required by specification
point_group = detect_point_group


# ----------------------------------------------------------------------
# Simple CLI entry point
# ----------------------------------------------------------------------
def _cli() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="pgmol",
        description="Detect the Schoenflies point group of a molecule from an XYZ file.",
    )
    parser.add_argument(
        "xyz",
        type=pathlib.Path,
        help="Path to an XYZ file (or raw XYZ string if prefixed with '@').",
    )
    args = parser.parse_args()

    try:
        pg = detect_point_group(args.xyz)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(pg)


if __name__ == "__main__":
    _cli()
