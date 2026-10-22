"""pgmol: Exact Schoenflies point group detection for molecular structures."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import List, Sequence, Tuple, Union

import numpy as np

# Standard atomic weights (IUPAC) for center-of-mass calculation
_ATOMIC_WEIGHTS = {
    "H": 1.008, "HE": 4.0026, "LI": 6.94, "BE": 9.0122, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "NE": 20.180,
    "NA": 22.990, "MG": 24.305, "AL": 26.982, "SI": 28.085, "P": 30.974,
    "S": 32.06, "CL": 35.45, "AR": 39.95, "K": 39.098, "CA": 40.078,
    "SC": 44.956, "TI": 47.867, "V": 50.942, "CR": 51.996, "MN": 54.938,
    "FE": 55.845, "CO": 58.933, "NI": 58.693, "CU": 63.546, "ZN": 65.38,
    "GA": 69.723, "GE": 72.630, "AS": 74.922, "SE": 78.971, "BR": 79.904,
    "KR": 83.798, "RB": 85.468, "SR": 87.62, "Y": 88.906, "ZR": 91.224,
    "NB": 92.906, "MO": 95.95, "TC": 97.0, "RU": 101.07, "RH": 102.91,
    "PD": 106.42, "AG": 107.87, "CD": 112.41, "IN": 114.82, "SN": 118.71,
    "SB": 121.76, "TE": 127.60, "I": 126.90, "XE": 131.29, "CS": 132.91,
    "BA": 137.33, "LA": 138.91, "CE": 140.12, "PR": 140.91, "ND": 144.24,
    "PM": 145.0, "SM": 150.36, "EU": 151.96, "GD": 157.25, "TB": 158.93,
    "DY": 162.50, "HO": 164.93, "ER": 167.26, "TM": 168.93, "YB": 173.05,
    "LU": 174.97, "HF": 178.49, "TA": 180.95, "W": 183.84, "RE": 186.21,
    "OS": 190.23, "IR": 192.22, "PT": 195.08, "AU": 196.97, "HG": 200.59,
    "TL": 204.38, "PB": 207.2, "BI": 208.98, "TH": 232.04, "PA": 231.04,
    "U": 238.03,
}


def _get_mass(element: str) -> float:
    return _ATOMIC_WEIGHTS.get(element.upper(), 12.0)


def _parse_xyz(xyz_input: Union[str, pathlib.Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Parse XYZ input into atomic symbols and Cartesian coordinates."""
    if isinstance(xyz_input, pathlib.Path):
        content = xyz_input.read_text(encoding="utf-8")
    elif isinstance(xyz_input, str):
        if "\n" not in xyz_input and (os.path.exists(xyz_input) or os.path.isfile(xyz_input)):
            content = pathlib.Path(xyz_input).read_text(encoding="utf-8")
        else:
            content = xyz_input
    else:
        raise TypeError(f"Expected str or pathlib.Path, got {type(xyz_input).__name__}")

    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    if not lines:
        return np.array([], dtype=str), np.empty((0, 3), dtype=float)

    try:
        n_atoms = int(lines[0])
        atom_lines = lines[2 : 2 + n_atoms] if len(lines) >= 2 + n_atoms else lines[2:]
    except ValueError:
        atom_lines = lines

    elements: List[str] = []
    coords: List[List[float]] = []

    for line in atom_lines:
        tokens = line.split()
        if len(tokens) >= 4:
            elements.append(tokens[0].capitalize())
            coords.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])

    return np.array(elements, dtype=str), np.array(coords, dtype=float).reshape(-1, 3)


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Generate a 3x3 rotation matrix around a unit axis by angle radians."""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    c1 = 1.0 - c
    return np.array([
        [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
        [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
        [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
    ], dtype=float)


def _reflection_matrix(normal: np.ndarray) -> np.ndarray:
    """Generate a 3x3 reflection matrix across a plane with given normal."""
    n = normal / np.linalg.norm(normal)
    return np.eye(3) - 2.0 * np.outer(n, n)


def _is_symmetry_op(matrix: np.ndarray, coords: np.ndarray, elements: np.ndarray, tol: float) -> bool:
    """Check if an orthogonal 3x3 transformation matrix is a valid molecular symmetry."""
    transformed = coords @ matrix.T
    tol_sq = tol * tol

    for el in np.unique(elements):
        idx = np.where(elements == el)[0]
        orig_el = coords[idx]
        trans_el = transformed[idx]

        diff = trans_el[:, np.newaxis, :] - orig_el[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)

        if np.any(np.min(dist_sq, axis=1) > tol_sq):
            return False
        if np.any(np.min(dist_sq, axis=0) > tol_sq):
            return False

    return True


def _canonical_axis(v: np.ndarray) -> np.ndarray:
    """Canonicalize vector orientation so the first non-zero component is positive."""
    u = v / np.linalg.norm(v)
    for val in u:
        if abs(val) > 1e-5:
            if val < 0:
                u = -u
            break
    return u


def _unique_axes(vectors: Sequence[np.ndarray], tol: float = 1e-3) -> List[np.ndarray]:
    """Filter candidate vectors into unique unoriented axes."""
    unique: List[np.ndarray] = []
    for v in vectors:
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            continue
        u = _canonical_axis(v)
        if not any(
            np.linalg.norm(u - prev) < tol or np.linalg.norm(u + prev) < tol
            for prev in unique
        ):
            unique.append(u)
    return unique


def _find_candidate_axes(coords: np.ndarray, elements: np.ndarray, tol: float) -> List[np.ndarray]:
    """Generate potential symmetry axis directions from molecular geometry and inertia tensor."""
    candidates: List[np.ndarray] = []

    # 1. Standard basis
    candidates.extend([np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])])

    # 2. Principal axes of inertia
    masses = np.array([_get_mass(el) for el in elements])
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    i_xx = np.sum(masses * (y * y + z * z))
    i_yy = np.sum(masses * (x * x + z * z))
    i_zz = np.sum(masses * (x * x + y * y))
    i_xy = -np.sum(masses * x * y)
    i_xz = -np.sum(masses * x * z)
    i_yz = -np.sum(masses * y * z)
    inertia_tensor = np.array([[i_xx, i_xy, i_xz], [i_xy, i_yy, i_yz], [i_xz, i_yz, i_zz]])

    evals, evecs = np.linalg.eigh(inertia_tensor)
    for i in range(3):
        candidates.append(evecs[:, i])

    # 3. Vectors from origin to atoms
    norms = np.linalg.norm(coords, axis=1)
    non_origin = np.where(norms > tol)[0]
    for idx in non_origin:
        candidates.append(coords[idx])

    # 4. Pair combinations (within same element / radial shells)
    unique_els = np.unique(elements)
    for el in unique_els:
        el_idx = np.where((elements == el) & (norms > tol))[0]
        n_el = len(el_idx)
        for i in range(n_el):
            idx_i = el_idx[i]
            r_i = coords[idx_i]
            for j in range(i + 1, n_el):
                idx_j = el_idx[j]
                r_j = coords[idx_j]
                candidates.append(r_i + r_j)
                candidates.append(r_i - r_j)
                cross = np.cross(r_i, r_j)
                if np.linalg.norm(cross) > 1e-4:
                    candidates.append(cross)

                # Triplet normal / centers
                for k in range(j + 1, min(j + 12, n_el)):
                    idx_k = el_idx[k]
                    r_k = coords[idx_k]
                    tri_norm = np.cross(r_j - r_i, r_k - r_i)
                    if np.linalg.norm(tri_norm) > 1e-4:
                        candidates.append(tri_norm)
                    candidates.append(r_i + r_j + r_k)

    # 5. Cross products between candidate axes
    base_axes = _unique_axes(candidates)
    cross_candidates: List[np.ndarray] = []
    n_base = len(base_axes)
    for i in range(min(n_base, 30)):
        for j in range(i + 1, min(n_base, 30)):
            cp = np.cross(base_axes[i], base_axes[j])
            if np.linalg.norm(cp) > 1e-4:
                cross_candidates.append(cp)

    return _unique_axes(candidates + cross_candidates)


def detect_point_group(xyz_input: Union[str, pathlib.Path], tol: float = 1e-2) -> str:
    """
    Determine the Schoenflies point group of a molecule from XYZ coordinates.

    Parameters
    ----------
    xyz_input : str | pathlib.Path
        Path to an XYZ file OR a raw XYZ-formatted multiline string.
    tol : float, optional
        Cartesian distance tolerance in ångströms (default: 1e-2).

    Returns
    -------
    str
        Exact ASCII Schoenflies symbol (e.g. 'C1', 'Cs', 'Ci', 'C2v', 'D3h',
        'Oh', 'Td', 'Ih', 'Dinfh', 'Cinfv', 'Kh', etc.).
    """
    elements, coords = _parse_xyz(xyz_input)
    n_atoms = len(elements)

    if n_atoms == 0:
        return "C1"
    if n_atoms == 1:
        return "Kh"

    # Translate center of mass to origin
    masses = np.array([_get_mass(el) for el in elements], dtype=float)
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / np.sum(masses)
    coords = coords - com

    # Check for linear molecule
    norms = np.linalg.norm(coords, axis=1)
    non_center = coords[norms > tol]
    if len(non_center) <= 1:
        is_linear = True
        linear_axis = np.array([0.0, 0.0, 1.0]) if len(non_center) == 0 else non_center[0] / np.linalg.norm(non_center[0])
    else:
        u0 = non_center[0] / np.linalg.norm(non_center[0])
        cross_mags = np.linalg.norm(np.cross(non_center, u0), axis=1)
        is_linear = np.all(cross_mags < tol)
        linear_axis = u0

    if is_linear:
        # Check inversion symmetry for Dinfh vs Cinfv
        has_inv = _is_symmetry_op(-np.eye(3), coords, elements, tol)
        return "Dinfh" if has_inv else "Cinfv"

    # Inversion center
    has_inversion = _is_symmetry_op(-np.eye(3), coords, elements, tol)

    # Search for all proper rotation axes
    candidate_axes = _find_candidate_axes(coords, elements, tol)
    max_order_tested = max(8, n_atoms)

    # Map each valid axis to its maximal rotational order n >= 2
    rotation_axes: List[Tuple[np.ndarray, int]] = []
    for axis in candidate_axes:
        highest_n = 1
        for n in range(max_order_tested, 1, -1):
            angle = 2.0 * np.pi / n
            rot = _rotation_matrix(axis, angle)
            if _is_symmetry_op(rot, coords, elements, tol):
                highest_n = n
                break
        if highest_n >= 2:
            rotation_axes.append((axis, highest_n))

    # Tally rotation axes by order
    c5_axes = [ax for ax, n in rotation_axes if n == 5]
    c4_axes = [ax for ax, n in rotation_axes if n == 4]
    c3_axes = [ax for ax, n in rotation_axes if n == 3]

    # --- Polyhedral Groups ---
    # Icosahedral (Ih, I): 6 C5 axes
    if len(c5_axes) >= 6:
        return "Ih" if has_inversion else "I"

    # Octahedral (Oh, O): 3 C4 axes
    if len(c4_axes) >= 3:
        return "Oh" if has_inversion else "O"

    # Tetrahedral (Th, Td, T): 4 C3 axes (and no C4/C5)
    if len(c3_axes) >= 4:
        if has_inversion:
            return "Th"
        # Distinguish Td from T via reflection planes
        # Td has 6 dihedral mirror planes; T has none
        has_mirror = False
        for ax in candidate_axes:
            refl = _reflection_matrix(ax)
            if _is_symmetry_op(refl, coords, elements, tol):
                has_mirror = True
                break
        return "Td" if has_mirror else "T"

    # --- Non-Polyhedral Groups ---
    if rotation_axes:
        # Find principal axis (highest order n)
        rotation_axes.sort(key=lambda item: item[1], reverse=True)
        principal_axis, n_prin = rotation_axes[0]

        # Count C2 axes perpendicular to principal axis
        perp_c2_count = 0
        for axis, n in rotation_axes:
            if n >= 2 and abs(np.dot(axis, principal_axis)) < 1e-2:
                # Count distinct perpendicular C2 axes
                perp_c2_count += 1

        is_dihedral = (perp_c2_count >= n_prin) or (n_prin == 2 and perp_c2_count >= 2)

        # Horizontal mirror plane (perpendicular to principal axis)
        sigma_h = _reflection_matrix(principal_axis)
        has_sigma_h = _is_symmetry_op(sigma_h, coords, elements, tol)

        if is_dihedral:
            if has_sigma_h:
                return f"D{n_prin}h"
            # Check for dihedral/vertical mirror planes or S_2n improper rotation
            # D_nd has improper rotation S_2n along principal axis
            s_2n = _reflection_matrix(principal_axis) @ _rotation_matrix(principal_axis, np.pi / n_prin)
            has_s_2n = _is_symmetry_op(s_2n, coords, elements, tol)
            if has_s_2n:
                return f"D{n_prin}d"
            return f"D{n_prin}"
        else:
            if has_sigma_h:
                return f"C{n_prin}h"
            # Check for vertical mirror planes containing principal axis
            has_sigma_v = False
            for axis in candidate_axes:
                if abs(np.dot(axis, principal_axis)) < 1e-2:  # Plane normal perp to principal axis
                    refl = _reflection_matrix(axis)
                    if _is_symmetry_op(refl, coords, elements, tol):
                        has_sigma_v = True
                        break
            if has_sigma_v:
                return f"C{n_prin}v"

            # Check improper axis S_2n along principal axis
            s_2n = _reflection_matrix(principal_axis) @ _rotation_matrix(principal_axis, np.pi / n_prin)
            if _is_symmetry_op(s_2n, coords, elements, tol):
                return f"S{2 * n_prin}"

            return f"C{n_prin}"

    # --- Low Symmetry Groups (No Cn axes with n >= 2) ---
    # Check for S4 without proper C4 (principal order would be C2, handled above, but verify)
    for axis in candidate_axes:
        refl = _reflection_matrix(axis)
        if _is_symmetry_op(refl, coords, elements, tol):
            return "Cs"

    if has_inversion:
        return "Ci"

    return "C1"


# Top-level API alias
point_group = detect_point_group


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for pgmol."""
    parser = argparse.ArgumentParser(
        prog="pgmol",
        description="Detect the Schoenflies point group of a molecule from an XYZ file.",
    )
    parser.add_argument("xyz_file", help="Path to XYZ file")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="Symmetry detection tolerance in Å (default: 1e-2)",
    )
    args = parser.parse_args(argv)

    try:
        pg = detect_point_group(args.xyz_file, tol=args.tol)
        print(pg)
        return 0
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
