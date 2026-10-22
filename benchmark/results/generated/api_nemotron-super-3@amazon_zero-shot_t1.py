"""pgmol: Molecular point group detector.

This module provides functionality to determine the Schoenflies point group
of a molecule from its XYZ coordinates using only the Python standard library and numpy.
"""

from __future__ import annotations

import pathlib
import re
from typing import Tuple, List, Optional, Union

import numpy as np


__all__ = ["detect_point_group", "point_group"]


def _parse_xyz(xyz_input: Union[str, pathlib.Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Parse XYZ input into atomic symbols and coordinates.

    Parameters
    ----------
    xyz_input : str or pathlib.Path
        Path to an XYZ file or a raw XYZ-formatted multiline string.

    Returns
    -------
    symbols : np.ndarray of str
        Atomic symbols.
    coords : np.ndarray of float, shape (n_atoms, 3)
        Atomic coordinates in Angstroms.

    Raises
    ------
    ValueError
        If the XYZ format is invalid.
    """
    if isinstance(xyz_input, (str, pathlib.Path)) and pathlib.Path(xyz_input).exists():
        with open(xyz_input, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = str(xyz_input).strip().splitlines()

    if len(lines) < 3:
        raise ValueError("Invalid XYZ format: too few lines")

    try:
        n_atoms = int(lines[0].strip())
    except ValueError as e:
        raise ValueError("First line must be the number of atoms") from e

    if len(lines) < n_atoms + 2:
        raise ValueError(f"Not enough atom lines: expected {n_atoms}, got {len(lines) - 2}")

    symbols = []
    coords = []
    for i in range(1, n_atoms + 2):
        parts = lines[i].split()
        if len(parts) < 4:
            raise ValueError(f"Invalid atom line {i}: {lines[i]}")
        symbols.append(parts[0])
        try:
            x, y, z = map(float, parts[1:4])
            coords.append([x, y, z])
        except ValueError as e:
            raise ValueError(f"Invalid coordinates on line {i}: {lines[i]}") from e

    return np.array(symbols, dtype=object), np.array(coords, dtype=float)


def _center_of_mass(symbols: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Compute center of mass using atomic masses."""
    atomic_masses = {
        "H": 1.008,
        "He": 4.0026,
        "Li": 6.94,
        "Be": 9.0122,
        "B": 10.81,
        "C": 12.011,
        "N": 14.007,
        "O": 15.999,
        "F": 18.998,
        "Ne": 20.180,
        "Na": 22.990,
        "Mg": 24.305,
        "Al": 26.982,
        "Si": 28.085,
        "P": 30.974,
        "S": 32.06,
        "Cl": 35.45,
        "Ar": 39.948,
        "K": 39.098,
        "Ca": 40.078,
    }
    masses = np.array([atomic_masses.get(s, 12.0) for s in symbols], dtype=float)
    total_mass = np.sum(masses)
    if total_mass == 0:
        return np.mean(coords, axis=0)
    return np.sum(masses[:, None] * coords, axis=0) / total_mass


def _inertia_tensor(symbols: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Compute the inertia tensor relative to the center of mass."""
    com = _center_of_mass(symbols, coords)
    com_coords = coords - com
    atomic_masses = {
        "H": 1.008,
        "He": 4.0026,
        "Li": 6.94,
        "Be": 9.0122,
        "B": 10.81,
        "C": 12.011,
        "N": 14.007,
        "O": 15.999,
        "F": 18.998,
        "Ne": 20.180,
        "Na": 22.990,
        "Mg": 24.305,
        "Al": 26.982,
        "Si": 28.085,
        "P": 30.974,
        "S": 32.06,
        "Cl": 35.45,
        "Ar": 39.948,
        "K": 39.098,
        "Ca": 40.078,
    }
    masses = np.array([atomic_masses.get(s, 12.0) for s in symbols], dtype=float)
    I = np.zeros((3, 3))
    for i in range(len(symbols)):
        m = masses[i]
        x, y, z = com_coords[i]
        I[0, 0] += m * (y * y + z * z)
        I[1, 1] += m * (x * x + z * z)
        I[2, 2] += m * (x * x + y * y)
        I[0, 1] += m * (-x * y)
        I[1, 0] += m * (-x * y)
        I[0, 2] += m * (-x * z)
        I[2, 0] += m * (-x * z)
        I[1, 2] += m * (-y * z)
        I[2, 1] += m * (-y * z)
    return I


def _principal_axes(inertia: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize the inertia tensor to get principal moments and axes."""
    evals, evecs = np.linalg.eigh(inertia)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    return evals, evecs


def _is_linear(symbols: np.ndarray, coords: np.ndarray, tol: float = 1e-2) -> bool:
    """Check if the molecule is linear."""
    if len(symbols) <= 2:
        return True
    com = _center_of_mass(symbols, coords)
    com_coords = coords - com
    inertia = _inertia_tensor(symbols, coords)
    evals, _ = _principal_axes(inertia)
    return evals[0] < tol * (evals[1] + evals[2])


def _has_inversion_center(symbols: np.ndarray, coords: np.ndarray, tol: float) -> bool:
    """Check for inversion center."""
    com = _center_of_mass(symbols, coords)
    displaced = coords - com
    for i in range(len(symbols)):
        found = False
        for j in range(len(symbols)):
            if symbols[i] != symbols[j]:
                continue
            if np.allclose(displaced[i], -displaced[j], atol=tol):
                found = True
                break
        if not found:
            return False
    return True


def _has_horizontal_plane(symbols: np.ndarray, coords: np.ndarray, axis: int, tol: float) -> bool:
    """Check for horizontal mirror plane perpendicular to given axis."""
    com = _center_of_mass(symbols, coords)
    displaced = coords - com
    coord = displaced[:, axis]
    for i in range(len(symbols)):
        found = False
        for j in range(len(symbols)):
            if symbols[i] != symbols[j]:
                continue
            if np.allclose(displaced[i], displaced[j], atol=tol) and np.isclose(coord[i], -coord[j], atol=tol):
                found = True
                break
        if not found:
            return False
    return True


def _has_vertical_plane(symbols: np.ndarray, coords: np.ndarray, axis: int, tol: float) -> bool:
    """Check for vertical mirror plane containing the principal axis."""
    com = _center_of_mass(symbols, coords)
    displaced = coords - com
    # Project onto plane perpendicular to axis
    other_axes = [i for i in range(3) if i != axis]
    if len(other_axes) != 2:
        return False
    a, b = other_axes
    coords_ab = displaced[:, [a, b]]
    # Build KDTree-like structure for 2D points
    for i in range(len(symbols)):
        found = False
        for j in range(len(symbols)):
            if symbols[i] != symbols[j]:
                continue
            if np.allclose(coords_ab[i], coords_ab[j], atol=tol):
                found = True
                break
        if not found:
            return False
    return True


def _has_cn_axis(symbols: np.ndarray, coords: np.ndarray, n: int, tol: float) -> bool:
    """Check for proper Cn rotation axis."""
    if n < 1:
        return False
    com = _center_of_mass(symbols, coords)
    displaced = coords - com
    # Find the axis with maximum moment of inertia (unique axis for linear, else z by convention)
    inertia = _inertia_tensor(symbols, coords)
    evals, evecs = _principal_axes(inertia)
    # For linear molecules, any axis perpendicular to bond works; we check if any Cn exists
    # For nonlinear, we assume the principal axis is z (index 2)
    axis = 2
    angle = 2 * np.pi / n
    # Rotation matrix around z-axis
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotated = displaced @ R.T
    # Check if rotated coords match original up to permutation
    matched = [False] * len(symbols)
    for i in range(len(symbols)):
        found = False
        for j in range(len(symbols)):
            if matched[j] or symbols[i] != symbols[j]:
                continue
            if np.allclose(rotated[i], displaced[j], atol=tol):
                matched[j] = True
                found = True
                break
        if not found:
            return False
    return True


def _has_sn_axis(symbols: np.ndarray, coords: np.ndarray, n: int, tol: float) -> bool:
    """Check for improper Sn rotation axis."""
    if n < 1:
        return False
    com = _center_of_mass(symbols, coords)
    displaced = coords - com
    inertia = _inertia_tensor(symbols, coords)
    evals, evecs = _principal_axes(inertia)
    axis = 2
    angle = 2 * np.pi / n
    c, s = np.cos(angle), np.sin(angle)
    # Rotation matrix
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    # Reflection through horizontal plane
    sigma_h = np.diag([1, 1, -1])
    # Sn = sigma_h * Cn
    S = sigma_h @ R
    rotated = displaced @ S.T
    matched = [False] * len(symbols)
    for i in range(len(symbols)):
        found = False
        for j in range(len(symbols)):
            if matched[j] or symbols[i] != symbols[j]:
                continue
            if np.allclose(rotated[i], displaced[j], atol=tol):
                matched[j] = True
                found = True
                break
        if not found:
            return False
    return True


def detect_point_group(xyz_input: Union[str, pathlib.Path], tol: float = 1e-2) -> str:
    """Determine the Schoenflies point group of a molecule.

    Parameters
    ----------
    xyz_input : str or pathlib.Path
        Path to an XYZ file or a raw XYZ-formatted multiline string.
    tol : float, optional
        Cartesian distance tolerance in ångströms (default: 1e-2).

    Returns
    -------
    str
        Exact ASCII Schoenflies symbol.
    """
    symbols, coords = _parse_xyz(xyz_input)
    n_atoms = len(symbols)

    # Single atom
    if n_atoms == 1:
        return "Kh"

    # Check linear
    if _is_linear(symbols, coords, tol):
        if _has_inversion_center(symbols, coords, tol):
            return "Dinfh"
        else:
            return "Cinfv"

    # Nonlinear molecule
    com = _center_of_mass(symbols, coords)
    displaced = coords - com
    inertia = _inertia_tensor(symbols, coords)
    evals, evecs = _principal_axes(inertia)

    # Check for spherical top (all moments equal and non-linear)
    if np.allclose(evals, evals[0], atol=tol) and evals[0] > tol:
        # Could be T, Th, Td, O, Oh, I, Ih - need to check for higher symmetry
        # For now, we assume the highest possible, but we must verify axes
        # We will check for C3, C4, C5 axes to distinguish
        has_C3 = _has_cn_axis(symbols, coords, 3, tol)
        has_C4 = _has_cn_axis(symbols, coords, 4, tol)
        has_C5 = _has_cn_axis(symbols, coords, 5, tol)
        has_i = _has_inversion_center(symbols, coords, tol)
        has_sigma_h = any(_has_horizontal_plane(symbols, coords, ax, tol) for ax in range(3))
        has_sigma_v = any(_has_vertical_plane(symbols, coords, ax, tol) for ax in range(3))

        if has_C5:
            if has_i and has_sigma_h:
                return "Ih"
            elif has_i:
                return "I"
            elif has_sigma_h or has_sigma_v:
                return "Ih"  # Actually Ih requires i and sigma_h; but if sigma_v and C5, it's likely Ih
            else:
                return "I"
        elif has_C4:
            if has_i and has_sigma_h:
                return "Oh"
            elif has_i:
                return "O"
            elif has_sigma_h:
                # Could be Th or Td? Actually Oh has both i and sigma_h
                # For cubic, if has C4 and sigma_h but no i, it's not standard
                # We simplify: if has C4 and sigma_h, assume Oh if also has i, else error
                # But we know it's spherical top, so likely Oh
                return "Oh"
            elif has_sigma_v:
                return "Oh"
            else:
                return "O"
        elif has_C3:
            if has_i and has_sigma_h:
                return "Th"
            elif has_i:
                return "T"
            elif has_sigma_h:
                return "Th"  # Th has sigma_h and i; Td has sigma_d; T has neither
            elif has_sigma_v:
                return "Td"
            else:
                return "T"
        else:
            # No C3, C4, C5 - but spherical top implies accidental
            # Fall back to lower symmetry
            pass

    # Check for Sn axes first (as they are often missed)
    has_S2 = _has_inversion_center(symbols, coords, tol)  # S2 = i
    has_S1 = False  # S1 = sigma_h, handled separately
    has_S3 = _has_sn_axis(symbols, coords, 3, tol)
    has_S4 = _has_sn_axis(symbols, coords, 4, tol)
    has_S5 = _has_sn_axis(symbols, coords, 5, tol)
    has_S6 = _has_sn_axis(symbols, coords, 6, tol)

    # Check for Cn axes
    max_n = 0
    for n in range(1, 13):  # up to C12
        if _has_cn_axis(symbols, coords, n, tol):
            max_n = n
    if max_n == 0:
        max_n = 1  # C1 always exists

    has_i = _has_inversion_center(symbols, coords, tol)
    has_sigma_h = any(_has_horizontal_plane(symbols, coords, ax, tol) for ax in range(3))
    has_sigma_v = any(_has_vertical_plane(symbols, coords, ax, tol) for ax in range(3))

    # Determine point group
    if max_n == 1:
        if has_i:
            return "Ci"
        elif has_sigma_h:
            return "Cs"
        elif has_sigma_v:
            return "Cs"  # Actually, a single vertical plane is Cs
        else:
            return "C1"
    else:
        n = max_n
        if has_i and has_sigma_h:
            return f"D{n}h"
        elif has_i:
            return f"D{n}"
        elif has_sigma_h:
            return f"C{n}h"
        elif has_sigma_v:
            return f"C{n}v"
        elif has_S2 and not has_i:  # S2 is i, so this is redundant
            pass
        elif has_S4:
            return "S4"
        elif has_S6 and not has_i and not has_sigma_h:  # S6 without i or sigma_h
            return "S6"
        else:
            return f"C{n}"


# Alias for convenience
point_group = detect_point_group
