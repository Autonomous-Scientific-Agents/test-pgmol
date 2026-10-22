"""
pgmol: Pure Python + NumPy Schoenflies Point Group Detection for Molecular Geometries.
"""

from __future__ import annotations

import argparse
import itertools
import math
import pathlib
import sys
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np


# -----------------------------------------------------------------------------
# XYZ Parsing & Geometry Utilities
# -----------------------------------------------------------------------------

def parse_xyz(xyz_input: Union[str, pathlib.Path]) -> Tuple[List[str], np.ndarray]:
    """
    Parse an XYZ string or file into elemental symbols and Cartesian coordinates.
    """
    lines = None
    if isinstance(xyz_input, pathlib.Path):
        with open(xyz_input, "r", encoding="utf-8") as f:
            lines = f.readlines()
    elif isinstance(xyz_input, str):
        if "\n" not in xyz_input:
            path_candidate = pathlib.Path(xyz_input)
            if path_candidate.is_file():
                with open(path_candidate, "r", encoding="utf-8") as f:
                    lines = f.readlines()
        if lines is None:
            lines = xyz_input.strip().splitlines()
    else:
        raise TypeError(f"Expected str or pathlib.Path, got {type(xyz_input)}")

    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines:
        raise ValueError("Empty XYZ input.")

    # Try reading standard XYZ header (N, comment, coords...)
    try:
        num_atoms = int(lines[0])
        coord_lines = lines[2 : 2 + num_atoms]
    except ValueError:
        # Fallback: maybe raw lines without atom count
        coord_lines = lines

    symbols: List[str] = []
    coords_list: List[List[float]] = []

    for ln in coord_lines:
        parts = ln.split()
        if len(parts) < 4:
            continue
        sym = parts[0].capitalize()
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        symbols.append(sym)
        coords_list.append([x, y, z])

    if not symbols:
        raise ValueError("Could not parse valid atomic coordinates from XYZ input.")

    return symbols, np.array(coords_list, dtype=np.float64)


def center_coordinates(symbols: List[str], coords: np.ndarray) -> np.ndarray:
    """Center geometry at the unweighted centroid."""
    return coords - np.mean(coords, axis=0)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < eps:
        return np.zeros_like(v)
    return v / norm


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix for rotation about unit axis by angle in radians."""
    u = normalize(axis)
    ux, uy, uz = u[0], u[1], u[2]
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    k = 1.0 - cos_a
    return np.array([
        [cos_a + ux * ux * k, ux * uy * k - uz * sin_a, ux * uz * k + uy * sin_a],
        [uy * ux * k + uz * sin_a, cos_a + uy * uy * k, uy * uz * k - ux * sin_a],
        [uz * ux * k - uy * sin_a, uz * uy * k + ux * sin_a, cos_a + uz * uz * k]
    ], dtype=np.float64)


def reflection_matrix(normal: np.ndarray) -> np.ndarray:
    """Householder reflection matrix across plane perpendicular to unit normal."""
    n = normalize(normal).reshape((3, 1))
    return np.eye(3, dtype=np.float64) - 2.0 * (n @ n.T)


def inversion_matrix() -> np.ndarray:
    return -np.eye(3, dtype=np.float64)


def is_symmetry_operation(
    matrix: np.ndarray,
    coords: np.ndarray,
    sym_groups: dict[str, List[int]],
    tol: float
) -> bool:
    """
    Check if a 3x3 orthogonal operation transforms the molecule onto itself,
    preserving chemical species.
    """
    transformed = coords @ matrix.T  # (N, 3)
    tol_sq = tol * tol

    for sym, indices in sym_groups.items():
        sub_orig = coords[indices]
        sub_trans = transformed[indices]
        m = len(indices)

        # Distance matrix between original and transformed atoms of this element
        # (m, 1, 3) - (1, m, 3) -> (m, m, 3)
        diff = sub_orig[:, None, :] - sub_trans[None, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)

        # Check that each transformed atom is within `tol` of at least one original atom
        min_dist_to_orig = np.min(dist_sq, axis=0)
        if np.any(min_dist_to_orig > tol_sq):
            return False

        # Check bijective 1-to-1 match
        min_dist_from_orig = np.min(dist_sq, axis=1)
        if np.any(min_dist_from_orig > tol_sq):
            return False

    return True


# -----------------------------------------------------------------------------
# Symmetry Element Probing
# -----------------------------------------------------------------------------

def has_inversion_center(coords: np.ndarray, sym_groups: dict, tol: float) -> bool:
    return is_symmetry_operation(inversion_matrix(), coords, sym_groups, tol)


def has_rotation_axis(axis: np.ndarray, n: int, coords: np.ndarray, sym_groups: dict, tol: float) -> bool:
    """Check if axis is a Cn rotation axis."""
    if n <= 1:
        return True
    u = normalize(axis)
    if np.linalg.norm(u) < 0.5:
        return False
    angle = 2.0 * math.pi / n
    R = rotation_matrix(u, angle)
    return is_symmetry_operation(R, coords, sym_groups, tol)


def has_sn_axis(axis: np.ndarray, n: int, coords: np.ndarray, sym_groups: dict, tol: float) -> bool:
    """Check if axis is an Sn improper rotation axis."""
    if n < 3:
        return False
    u = normalize(axis)
    if np.linalg.norm(u) < 0.5:
        return False
    angle = 2.0 * math.pi / n
    R = rotation_matrix(u, angle)
    sigma = reflection_matrix(u)
    S = sigma @ R
    return is_symmetry_operation(S, coords, sym_groups, tol)


def has_mirror_plane(normal: np.ndarray, coords: np.ndarray, sym_groups: dict, tol: float) -> bool:
    u = normalize(normal)
    if np.linalg.norm(u) < 0.5:
        return False
    sigma = reflection_matrix(u)
    return is_symmetry_operation(sigma, coords, sym_groups, tol)


def find_candidate_axes(coords: np.ndarray, tol: float) -> List[np.ndarray]:
    """
    Generate candidate directions for symmetry elements from:
    1. Principal axes of inertia (and cross products).
    2. Position vectors of atoms and centroids of atom pairs/triplets.
    3. Normal vectors to atom triplets (planes).
    4. Midpoints and bisectors.
    """
    candidates = []

    # 1. Inertia tensor eigenvectors
    # Center is (0,0,0)
    I = np.zeros((3, 3), dtype=np.float64)
    for x, y, z in coords:
        I[0, 0] += y * y + z * z
        I[1, 1] += x * x + z * z
        I[2, 2] += x * x + y * y
        I[0, 1] -= x * y
        I[0, 2] -= x * z
        I[1, 2] -= y * z
    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]

    eigvals, eigvecs = np.linalg.eigh(I)
    for col in range(3):
        candidates.append(eigvecs[:, col])

    # Also pairwise cross-products of principal axes
    candidates.append(np.cross(eigvecs[:, 0], eigvecs[:, 1]))
    candidates.append(np.cross(eigvecs[:, 1], eigvecs[:, 2]))
    candidates.append(np.cross(eigvecs[:, 2], eigvecs[:, 0]))

    # Standard Cartesian directions
    candidates.extend([np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])])

    # 2. Vectors from origin to atoms
    atom_vecs = [v for v in coords if np.linalg.norm(v) > tol]
    for v in atom_vecs:
        candidates.append(v)

    # 3. Sums, differences, and cross products of atom pairs
    n_at = len(atom_vecs)
    # If molecule is large, sample pairwise to avoid combinatorial explosion
    max_pairs = 120
    pair_indices = list(itertools.combinations(range(n_at), 2))
    if len(pair_indices) > max_pairs:
        # Uniform stride sampling
        step = len(pair_indices) // max_pairs
        pair_indices = pair_indices[::step]

    for i, j in pair_indices:
        v1 = atom_vecs[i]
        v2 = atom_vecs[j]
        candidates.append(v1 + v2)
        candidates.append(v1 - v2)
        cp = np.cross(v1, v2)
        if np.linalg.norm(cp) > tol:
            candidates.append(cp)

    # Filter and deduplicate unit vectors (modulo sign)
    unique_candidates: List[np.ndarray] = []
    for vec in candidates:
        u = normalize(vec)
        if np.linalg.norm(u) < 0.5:
            continue
        # Check if already present up to sign
        is_dup = False
        for ex in unique_candidates:
            dot = abs(float(np.dot(u, ex)))
            if dot > 1.0 - 1e-4:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(u)

    return unique_candidates


def find_all_cn_axes(
    order: int,
    candidate_dirs: List[np.ndarray],
    coords: np.ndarray,
    sym_groups: dict,
    tol: float
) -> List[np.ndarray]:
    """Find unique Cn rotation axes of a given order among candidates."""
    found: List[np.ndarray] = []
    for d in candidate_dirs:
        if has_rotation_axis(d, order, coords, sym_groups, tol):
            # Check uniqueness up to sign
            is_dup = False
            for ex in found:
                if abs(float(np.dot(d, ex))) > 1.0 - 1e-3:
                    is_dup = True
                    break
            if not is_dup:
                found.append(d)
    return found


def find_all_mirror_planes(
    candidate_dirs: List[np.ndarray],
    coords: np.ndarray,
    sym_groups: dict,
    tol: float
) -> List[np.ndarray]:
    """Find unique mirror plane normals."""
    found: List[np.ndarray] = []
    for d in candidate_dirs:
        if has_mirror_plane(d, coords, sym_groups, tol):
            is_dup = False
            for ex in found:
                if abs(float(np.dot(d, ex))) > 1.0 - 1e-3:
                    is_dup = True
                    break
            if not is_dup:
                found.append(d)
    return found


# -----------------------------------------------------------------------------
# Point Group Classification Algorithm
# -----------------------------------------------------------------------------

def detect_point_group(xyz_input: Union[str, pathlib.Path], tol: float = 1e-2) -> str:
    """
    Determine the Schoenflies point group of a molecule from XYZ coordinates.
    """
    symbols, coords = parse_xyz(xyz_input)
    n_atoms = len(symbols)

    if n_atoms == 1:
        return "Kh"

    coords = center_coordinates(symbols, coords)

    # Build symbol index mapping for fast permutation invariant symmetry checks
    sym_groups: dict[str, List[int]] = defaultdict(list)
    for idx, sym in enumerate(symbols):
        sym_groups[sym].append(idx)

    # Check for collinearity (linear molecules)
    # SVD or Inertia eigenvalues: linear if rank of coords is 1
    _, svals, _ = np.linalg.svd(coords)
    # In 3D coords, if 2 singular values are close to 0, it is linear
    if svals[1] < tol:
        # Linear molecule: either Dinfh or Cinfv
        if has_inversion_center(coords, sym_groups, tol):
            return "Dinfh"
        return "Cinfv"

    # Generate rich set of candidate symmetry directions
    candidates = find_candidate_axes(coords, tol)

    # Check for cubic / icosahedral groups: multiple high-order axes (C3, C4, C5)
    c5_axes = find_all_cn_axes(5, candidates, coords, sym_groups, tol)
    c4_axes = find_all_cn_axes(4, candidates, coords, sym_groups, tol)
    c3_axes = find_all_cn_axes(3, candidates, coords, sym_groups, tol)

    inv = has_inversion_center(coords, sym_groups, tol)

    # Icosahedral groups: 6 C5 axes (or 10 C3 axes)
    if len(c5_axes) >= 6 or (len(c5_axes) >= 1 and len(c3_axes) >= 10):
        if inv:
            return "Ih"
        return "I"

    # Octahedral / Tetrahedral groups: 4 C3 axes
    if len(c3_axes) >= 4:
        if len(c4_axes) >= 3:
            # Octahedral
            if inv:
                return "Oh"
            return "O"
        else:
            # Tetrahedral: T, Th, Td
            if inv:
                return "Th"
            # Check mirror planes: Td has 6 sigma_d planes, T has none
            planes = find_all_mirror_planes(candidates, coords, sym_groups, tol)
            if len(planes) >= 6:
                return "Td"
            return "T"

    # Not cubic/polyhedral. Find the principal rotation axis C_n (highest n).
    # Check possible n from 8 down to 2
    max_n = 1
    principal_axis: np.ndarray | None = None
    all_principal_axes: List[np.ndarray] = []

    for n in range(8, 1, -1):
        axes = find_all_cn_axes(n, candidates, coords, sym_groups, tol)
        if axes:
            max_n = n
            principal_axis = axes[0]
            all_principal_axes = axes
            break

    # If no Cn rotation axis (n >= 2), check C1, Cs, Ci
    if max_n == 1 or principal_axis is None:
        # Check for S4 (could be an S4 group without C4, but S4 implies C2 so handled below)
        # Check mirror plane
        planes = find_all_mirror_planes(candidates, coords, sym_groups, tol)
        if len(planes) > 0:
            return "Cs"
        if inv:
            return "Ci"
        return "C1"

    # We have a principal axis of order n = max_n.
    # Check if dihedral (Dn, Dnh, Dnd): exists n perpendicular C2 axes
    is_dihedral = False
    perp_c2_count = 0

    c2_axes = find_all_cn_axes(2, candidates, coords, sym_groups, tol)
    for c2 in c2_axes:
        # Check perpendicularity to principal axis
        dot = abs(float(np.dot(c2, principal_axis)))
        if dot < 0.1:  # perpendicular
            perp_c2_count += 1

    # In Dn, there are n perpendicular C2 axes (at least 1 unique axis found)
    if (max_n == 2 and len(c2_axes) >= 3) or perp_c2_count >= 1:
        is_dihedral = True

    # Check horizontal mirror plane: sigma_h perpendicular to principal_axis
    has_sigma_h = has_mirror_plane(principal_axis, coords, sym_groups, tol)

    # Check vertical / dihedral mirror planes: containing principal_axis
    all_planes = find_all_mirror_planes(candidates, coords, sym_groups, tol)
    vert_planes = []
    for p in all_planes:
        dot = abs(float(np.dot(p, principal_axis)))
        if dot < 0.1:  # normal perpendicular to principal axis => plane contains axis
            vert_planes.append(p)

    has_vert_planes = len(vert_planes) >= 1

    if is_dihedral:
        if has_sigma_h:
            return f"D{max_n}h"
        if has_vert_planes:
            return f"D{max_n}d"
        return f"D{max_n}"
    else:
        if has_sigma_h:
            return f"C{max_n}h"
        if has_vert_planes:
            return f"C{max_n}v"
        # Check S_2n axis collinear with principal axis
        if has_sn_axis(principal_axis, 2 * max_n, coords, sym_groups, tol):
            return f"S{2 * max_n}"
        return f"C{max_n}"


point_group = detect_point_group


# -----------------------------------------------------------------------------
# Command Line Interface
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect Schoenflies point group from XYZ molecular geometry."
    )
    parser.add_argument("xyz_file", help="Path to XYZ file")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="Tolerance in ångströms (default: 0.01)",
    )
    args = parser.parse_args()

    try:
        pg = detect_point_group(args.xyz_file, tol=args.tol)
        print(pg)
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
