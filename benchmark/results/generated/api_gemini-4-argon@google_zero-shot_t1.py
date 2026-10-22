"""
pgmol: Molecular Schoenflies Point Group Detection.

Determines the exact Schoenflies point group symbol of a 3D molecule from its
XYZ atomic coordinates using only the Python standard library and NumPy.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Sequence

import numpy as np

__all__ = ["detect_point_group", "point_group", "parse_xyz"]

# Standard atomic weights (for inertia tensor and center of mass calculation)
_ATOMIC_MASSES: dict[str, float] = {
    "H": 1.00794, "He": 4.002602, "Li": 6.941, "Be": 9.012182, "B": 10.811,
    "C": 12.0107, "N": 14.0067, "O": 15.9994, "F": 18.9984032, "Ne": 20.1797,
    "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815386, "Si": 28.0855,
    "P": 30.973762, "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.0983,
    "Ca": 40.078, "Sc": 44.955912, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961,
    "Mn": 54.938045, "Fe": 55.845, "Co": 58.933195, "Ni": 58.6934, "Cu": 63.546,
    "Zn": 65.38, "Ga": 69.723, "Ge": 72.64, "As": 74.9216, "Se": 78.96,
    "Br": 79.904, "Kr": 83.798, "Rb": 85.4678, "Sr": 87.62, "Y": 88.90585,
    "Zr": 91.224, "Nb": 92.90638, "Mo": 95.96, "Tc": 98.0, "Ru": 101.07,
    "Rh": 102.9055, "Pd": 106.42, "Ag": 107.8682, "Cd": 112.411, "In": 114.818,
    "Sn": 118.71, "Sb": 121.76, "Te": 127.6, "I": 126.90447, "Xe": 131.293,
    "Cs": 132.9054519, "Ba": 137.327, "La": 138.90547, "Ce": 140.116,
    "Pr": 140.90765, "Nd": 144.242, "Pm": 145.0, "Sm": 150.36, "Eu": 151.964,
    "Gd": 157.25, "Tb": 158.92535, "Dy": 162.5, "Ho": 164.93032, "Er": 167.259,
    "Tm": 168.93421, "Yb": 173.054, "Lu": 174.9668, "Hf": 178.49, "Ta": 180.94788,
    "W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217, "Pt": 195.084,
    "Au": 196.966569, "Hg": 200.59, "Tl": 204.3833, "Pb": 207.2, "Bi": 208.9804,
    "Po": 209.0, "At": 210.0, "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0,
    "Th": 232.03806, "Pa": 231.03588, "U": 238.02891, "Np": 237.0, "Pu": 244.0,
    "Am": 243.0, "Cm": 247.0, "Bk": 247.0, "Cf": 251.0, "Es": 252.0, "Fm": 257.0,
    "Md": 258.0, "No": 259.0, "Lr": 262.0, "Rf": 267.0, "Db": 268.0, "Sg": 271.0,
    "Bh": 272.0, "Hs": 270.0, "Mt": 276.0, "Ds": 281.0, "Rg": 280.0, "Cn": 285.0,
    "Nh": 284.0, "Fl": 289.0, "Mc": 288.0, "Lv": 293.0, "Ts": 294.0, "Og": 294.0,
}


def _normalize_symbol(sym: str) -> str:
    s = sym.strip()
    if not s:
        raise ValueError("Empty element symbol in XYZ input.")
    return s[0].upper() + s[1:].lower()


def _get_mass(sym: str) -> float:
    if sym in _ATOMIC_MASSES:
        return _ATOMIC_MASSES[sym]
    # Deterministic fallback mass for custom/isotopic symbols
    return float(sum((idx + 1) * ord(ch) for idx, ch in enumerate(sym)))


def parse_xyz(xyz_input: str | pathlib.Path) -> tuple[list[str], np.ndarray]:
    """
    Parse an XYZ file or raw XYZ multiline string.

    Parameters
    ----------
    xyz_input : str | pathlib.Path
        Path to an XYZ file or a raw XYZ-formatted string.

    Returns
    -------
    tuple[list[str], np.ndarray]
        Element symbols (normalized) and an (N, 3) float64 array of coordinates.
    """
    content: str | None = None
    if isinstance(xyz_input, pathlib.Path):
        content = xyz_input.read_text(encoding="utf-8")
    elif isinstance(xyz_input, str):
        if "\n" not in xyz_input and "\r" not in xyz_input:
            path = pathlib.Path(xyz_input)
            if path.is_file():
                content = path.read_text(encoding="utf-8")
        if content is None:
            content = xyz_input
    else:
        raise TypeError(f"Unsupported xyz_input type: {type(xyz_input)!r}")

    lines = [line.strip() for line in content.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    if len(lines) < 2:
        raise ValueError("Invalid XYZ format: fewer than 2 lines.")

    try:
        n_atoms = int(lines[0].split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid atom count on line 1: {lines[0]!r}") from exc

    if n_atoms <= 0:
        raise ValueError(f"Atom count must be positive, got {n_atoms}.")

    # Line 2 (lines[1]) is the comment line and is ignored completely.
    atom_lines = [ln for ln in lines[2:] if ln]
    if len(atom_lines) < n_atoms:
        raise ValueError(
            f"Expected {n_atoms} atom lines, found {len(atom_lines)}."
        )

    symbols: list[str] = []
    coords = np.empty((n_atoms, 3), dtype=np.float64)
    for i in range(n_atoms):
        parts = atom_lines[i].split()
        if len(parts) < 4:
            raise ValueError(f"Malformed atom line {i + 3}: {atom_lines[i]!r}")
        symbols.append(_normalize_symbol(parts[0]))
        try:
            coords[i, 0] = float(parts[1])
            coords[i, 1] = float(parts[2])
            coords[i, 2] = float(parts[3])
        except ValueError as exc:
            raise ValueError(
                f"Invalid coordinates on atom line {i + 3}: {atom_lines[i]!r}"
            ) from exc

    return symbols, coords


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return 3x3 proper rotation matrix around unit `axis` by `angle` radians."""
    u = axis / np.linalg.norm(axis)
    ux, uy, uz = u[0], u[1], u[2]
    c = math.cos(angle)
    s = math.sin(angle)
    oc = 1.0 - c
    return np.array(
        [
            [c + ux * ux * oc, ux * uy * oc - uz * s, ux * uz * oc + uy * s],
            [uy * ux * oc + uz * s, c + uy * uy * oc, uy * uz * oc - ux * s],
            [uz * ux * oc - uy * s, uz * uy * oc + ux * s, c + uz * uz * oc],
        ],
        dtype=np.float64,
    )


def _reflection_matrix(normal: np.ndarray) -> np.ndarray:
    """Return 3x3 reflection matrix across plane through origin with unit `normal`."""
    n = normal / np.linalg.norm(normal)
    return np.eye(3, dtype=np.float64) - 2.0 * np.outer(n, n)


def _improper_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return 3x3 improper rotation matrix S_n = sigma_h @ C_n around `axis`."""
    return _reflection_matrix(axis) @ _rotation_matrix(axis, angle)


class _SymmetryAnalyzer:
    """Internal helper encapsulating molecular coordinates and symmetry checks."""

    def __init__(self, symbols: Sequence[str], coords: np.ndarray, tol: float) -> None:
        self.tol = float(tol)
        self.tol_sq = self.tol * self.tol
        self.n_atoms = len(symbols)
        masses = np.array([_get_mass(s) for s in symbols], dtype=np.float64)
        total_mass = float(np.sum(masses))
        com = np.sum(coords * masses[:, None], axis=0) / total_mass
        self.coords = coords - com
        self.symbols = list(symbols)
        self.norms = np.linalg.norm(self.coords, axis=1)
        self.max_radius = float(np.max(self.norms)) if self.n_atoms > 0 else 1.0
        # Angular tolerance for deduplicating unit axis/normal vectors
        self.ang_tol = max(1e-2, min(0.12, self.tol / max(self.max_radius, 0.25)))

        # Group atom indices by element symbol for fast operation checking
        by_elem: dict[str, list[int]] = {}
        for idx, sym in enumerate(self.symbols):
            by_elem.setdefault(sym, []).append(idx)
        self.elem_coords: list[np.ndarray] = [
            self.coords[idxs] for idxs in by_elem.values()
        ]

        # Build distance-based shells of equivalent-element atoms not at the origin
        self.shells: list[np.ndarray] = []
        for grp in self.elem_coords:
            r = np.linalg.norm(grp, axis=1)
            non_origin = grp[r > self.tol]
            if len(non_origin) == 0:
                continue
            r_no = np.linalg.norm(non_origin, axis=1)
            order = np.argsort(r_no)
            sorted_pts = non_origin[order]
            sorted_r = r_no[order]
            start = 0
            for i in range(1, len(sorted_r) + 1):
                if i == len(sorted_r) or (sorted_r[i] - sorted_r[i - 1]) > 2.0 * self.tol:
                    self.shells.append(sorted_pts[start:i])
                    start = i
        self.shells.sort(key=len)

    def check_op(self, mat: np.ndarray) -> bool:
        """Check whether 3x3 orthogonal matrix `mat` is a valid symmetry operation."""
        tol_sq = self.tol_sq
        for grp in self.elem_coords:
            transformed = grp @ mat.T
            # Pairwise squared distances between transformed and original group
            diff = transformed[:, None, :] - grp[None, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            if np.any(np.min(dist_sq, axis=1) > tol_sq):
                return False
            if np.any(np.min(dist_sq, axis=0) > tol_sq):
                return False
        return True

    def has_inversion(self) -> bool:
        return self.check_op(-np.eye(3, dtype=np.float64))

    def has_cn(self, axis: np.ndarray, n: int) -> bool:
        return self.check_op(_rotation_matrix(axis, 2.0 * math.pi / n))

    def has_sigma(self, normal: np.ndarray) -> bool:
        return self.check_op(_reflection_matrix(normal))

    def has_sn(self, axis: np.ndarray, n: int) -> bool:
        return self.check_op(_improper_rotation_matrix(axis, 2.0 * math.pi / n))

    def refine_c2_or_cn_axis(self, axis: np.ndarray, n: int) -> np.ndarray:
        """Refine an approximate rotation axis using exact atom correspondence."""
        u = axis / np.linalg.norm(axis)
        mat = _rotation_matrix(u, 2.0 * math.pi / n)
        # Average u with sum of cycle orbits or v + R v to lock onto exact geometry
        acc = np.zeros(3, dtype=np.float64)
        for grp in self.elem_coords:
            if len(grp) == 0:
                continue
            if n == 2:
                transformed = grp @ mat.T
                diff = transformed[:, None, :] - grp[None, :, :]
                nearest = np.argmin(np.sum(diff * diff, axis=2), axis=1)
                sums = grp + grp[nearest]
                dots = sums @ u
                signs = np.where(dots >= 0.0, 1.0, -1.0)
                acc += np.sum(sums * signs[:, None], axis=0)
            else:
                # Sum vectors along the full n-fold cycle and cross products of steps
                orbit_sum = grp.copy()
                rot_k = np.eye(3, dtype=np.float64)
                for _ in range(1, n):
                    rot_k = rot_k @ mat
                    transformed = grp @ rot_k.T
                    diff = transformed[:, None, :] - grp[None, :, :]
                    nearest = np.argmin(np.sum(diff * diff, axis=2), axis=1)
                    orbit_sum = orbit_sum + grp[nearest]
                dots = orbit_sum @ u
                signs = np.where(dots >= 0.0, 1.0, -1.0)
                acc += np.sum(orbit_sum * signs[:, None], axis=0)

                transformed1 = grp @ mat.T
                diff1 = transformed1[:, None, :] - grp[None, :, :]
                idx1 = np.argmin(np.sum(diff1 * diff1, axis=2), axis=1)
                v0 = grp
                v1 = grp[idx1]
                v2 = grp[idx1[idx1]]
                crosses = np.cross(v1 - v0, v2 - v1)
                c_dots = crosses @ u
                c_signs = np.where(c_dots >= 0.0, 1.0, -1.0)
                acc += np.sum(crosses * c_signs[:, None], axis=0)

        norm_acc = float(np.linalg.norm(acc))
        if norm_acc > 1e-6:
            u_refined = acc / norm_acc
            if self.has_cn(u_refined, n):
                return u_refined
        return u

    def refine_sigma_normal(self, normal: np.ndarray) -> np.ndarray:
        """Refine an approximate mirror plane normal using matched atom pairs."""
        n = normal / np.linalg.norm(normal)
        mat = _reflection_matrix(n)
        acc = np.zeros(3, dtype=np.float64)
        for grp in self.elem_coords:
            transformed = grp @ mat.T
            diff = transformed[:, None, :] - grp[None, :, :]
            nearest = np.argmin(np.sum(diff * diff, axis=2), axis=1)
            deltas = grp - grp[nearest]
            dots = deltas @ n
            signs = np.where(dots >= 0.0, 1.0, -1.0)
            acc += np.sum(deltas * signs[:, None], axis=0)
        norm_acc = float(np.linalg.norm(acc))
        if norm_acc > 1e-6:
            n_refined = acc / norm_acc
            if self.has_sigma(n_refined):
                return n_refined
        return n


class _VectorPool:
    """Deduplicated pool of unit 3D direction vectors (identifying +v and -v)."""

    def __init__(self, ang_tol: float = 0.05) -> None:
        self.cos_thresh = math.cos(ang_tol)
        self.vectors: list[np.ndarray] = []

    def add(self, vec: np.ndarray) -> bool:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-7:
            return False
        u = vec / norm
        # Canonicalize sign
        for comp in u:
            if abs(comp) > 1e-6:
                if comp < 0:
                    u = -u
                break
        for existing in self.vectors:
            if abs(float(np.dot(u, existing))) >= self.cos_thresh:
                return False
        self.vectors.append(u)
        return True


def _inertia_axes(symbols: Sequence[str], coords: np.ndarray) -> np.ndarray:
    """Compute orthonormal eigenvectors of the molecular inertia tensor."""
    masses = np.array([_get_mass(s) for s in symbols], dtype=np.float64)
    r2 = np.sum(coords * coords, axis=1)
    inertia = np.sum(masses[:, None, None] * (r2[:, None, None] * np.eye(3) - coords[:, :, None] @ coords[:, None, :]), axis=0)
    _, eigvecs = np.linalg.eigh(inertia)
    return eigvecs.T


def _is_linear(coords: np.ndarray, tol: float) -> bool:
    """Return True if all atoms lie on a single line through the origin within `tol`."""
    if len(coords) <= 2:
        return True
    norms = np.linalg.norm(coords, axis=1)
    max_idx = int(np.argmax(norms))
    if norms[max_idx] <= tol:
        return True
    # Use SVD to find best-fit line direction
    _, _, vt = np.linalg.svd(coords, full_matrices=False)
    axis = vt[0]
    proj = (coords @ axis)[:, None] * axis[None, :]
    perp_dist = np.linalg.norm(coords - proj, axis=1)
    return bool(np.all(perp_dist <= tol))


def _plane_normal_if_planar(coords: np.ndarray, tol: float) -> np.ndarray | None:
    """If molecule is planar within `tol`, return unit normal to plane, else None."""
    if len(coords) < 3:
        return None
    _, s, vt = np.linalg.svd(coords, full_matrices=False)
    normal = vt[2]
    dists = np.abs(coords @ normal)
    if np.all(dists <= tol):
        return normal / np.linalg.norm(normal)
    return None


def _generate_candidate_directions(analyzer: _SymmetryAnalyzer) -> list[np.ndarray]:
    """
    Generate a complete set of candidate symmetry directions (for axes and plane normals)
    guaranteed to include all C_n axes (n >= 2), S_n axes, and mirror plane normals.
    """
    pool = _VectorPool(ang_tol=analyzer.ang_tol * 0.5)

    # 1. Principal axes of inertia
    for vec in _inertia_axes(analyzer.symbols, analyzer.coords):
        pool.add(vec)

    # 2. Normal to molecular plane if planar
    plane_normal = _plane_normal_if_planar(analyzer.coords, analyzer.tol)
    if plane_normal is not None:
        pool.add(plane_normal)

    # 3. Select representative shells that together span 3D (or 2D if planar)
    selected_shells: list[np.ndarray] = []
    accumulated_pts: list[np.ndarray] = []
    for sh in analyzer.shells:
        selected_shells.append(sh)
        accumulated_pts.append(sh)
        stacked = np.vstack(accumulated_pts)
        if len(stacked) >= 3:
            _, s, _ = np.linalg.svd(stacked, full_matrices=False)
            # Stop adding shells once we have full rank (3D, or 2D for planar molecules)
            if s[1] > analyzer.tol and (plane_normal is not None or (len(s) > 2 and s[2] > analyzer.tol)):
                if len(selected_shells) >= 2:
                    break

    # 4. From selected shells, add atom vectors, pair midpoints, pair differences,
    #    pair cross products, and equidistant triplet normals.
    for sh in selected_shells:
        m = len(sh)
        for i in range(m):
            vi = sh[i]
            pool.add(vi)
            for j in range(i + 1, m):
                vj = sh[j]
                pool.add(vi + vj)
                pool.add(vi - vj)
                pool.add(np.cross(vi, vj))

        # Equidistant triplets in the shell: for any C_n (n >= 3), consecutive images
        # (vi, vj, vk) have |vi - vj| == |vj - vk| and normal (vj - vi) x (vk - vj).
        if m >= 3:
            diff_mat = sh[:, None, :] - sh[None, :, :]
            dists = np.linalg.norm(diff_mat, axis=2)
            # Limit triplet search per vertex to nearest few distinct distances
            for j in range(m):
                dj = dists[j]
                # Sort neighbors by distance
                order = np.argsort(dj)
                for a_idx in range(1, m):
                    i = int(order[a_idx])
                    d_ij = dj[i]
                    if d_ij <= analyzer.tol:
                        continue
                    for b_idx in range(a_idx + 1, m):
                        k = int(order[b_idx])
                        if abs(dj[k] - d_ij) > 2.5 * analyzer.tol:
                            break
                        pool.add(np.cross(sh[j] - sh[i], sh[k] - sh[j]))
                        pool.add(sh[i] + sh[j] + sh[k])

    return pool.vectors


def detect_point_group(xyz_input: str | pathlib.Path, tol: float = 1e-2) -> str:
    """
    Determine the Schoenflies point group of a molecule.

    Parameters
    ----------
    xyz_input : str | pathlib.Path
        Path to an XYZ file OR a raw XYZ-formatted multiline string.
    tol : float, optional
        Cartesian distance tolerance in ångströms (default: 1e-2).

    Returns
    -------
    str
        Exact ASCII Schoenflies symbol, e.g. 'C1', 'Cs', 'Ci', 'C2', 'C2v',
        'C3h', 'D2h', 'D3d', 'S4', 'T', 'Td', 'Th', 'O', 'Oh', 'I', 'Ih',
        'Cinfv', 'Dinfh', or 'Kh'.
    """
    symbols, raw_coords = parse_xyz(xyz_input)
    n_atoms = len(symbols)

    # 1. Single atom -> Kh
    if n_atoms == 1:
        return "Kh"

    analyzer = _SymmetryAnalyzer(symbols, raw_coords, tol)

    # All atoms coincident at origin within tol
    if analyzer.max_radius <= tol:
        return "Kh"

    # 2. Linear molecules -> Dinfh or Cinfv
    if _is_linear(analyzer.coords, tol):
        return "Dinfh" if analyzer.has_inversion() else "Cinfv"

    # 3. Generate candidate directions and discover mirror planes and C2 axes first
    candidates = _generate_candidate_directions(analyzer)

    sigma_pool = _VectorPool(ang_tol=analyzer.ang_tol)
    c2_pool = _VectorPool(ang_tol=analyzer.ang_tol)

    for vec in candidates:
        if analyzer.has_sigma(vec):
            sigma_pool.add(analyzer.refine_sigma_normal(vec))
        if analyzer.has_cn(vec, 2):
            c2_pool.add(analyzer.refine_c2_or_cn_axis(vec, 2))

    # Cross products of found C2 axes and mirror normals always include the
    # principal axis (for D_n, C_nv, D_nh, D_nd) and cubic/icosahedral axes.
    extra_Pool = _VectorPool(ang_tol=analyzer.ang_tol * 0.5)
    for v in candidates:
        extra_Pool.add(v)

    base_vecs = sigma_pool.vectors + c2_pool.vectors
    for i in range(len(base_vecs)):
        for j in range(i + 1, len(base_vecs)):
            cross = np.cross(base_vecs[i], base_vecs[j])
            if np.linalg.norm(cross) > 1e-5:
                extra_Pool.add(cross)
            mid = base_vecs[i] + base_vecs[j]
            if np.linalg.norm(mid) > 1e-5:
                extra_Pool.add(mid)
            diff = base_vecs[i] - base_vecs[j]
            if np.linalg.norm(diff) > 1e-5:
                extra_Pool.add(diff)

    # Re-check extra directions for any newly revealed C2 axes or mirror planes
    all_candidates = extra_Pool.vectors
    for vec in all_candidates:
        if analyzer.has_sigma(vec):
            sigma_pool.add(analyzer.refine_sigma_normal(vec))
        if analyzer.has_cn(vec, 2):
            c2_pool.add(analyzer.refine_c2_or_cn_axis(vec, 2))

    # Determine max possible rotation order from shell sizes (at least 8)
    max_shell = max((len(sh) for sh in analyzer.shells), default=2)
    max_n = max(8, min(max_shell, 24))

    # Discover all proper rotation axes C_n (n >= 2)
    # Map each unique axis direction to its maximum proper rotation order n
    cn_axes: list[tuple[np.ndarray, int]] = []
    axis_pool = _VectorPool(ang_tol=analyzer.ang_tol)

    for vec in all_candidates:
        highest_n = 1
        best_vec = vec
        for n in range(max_n, 1, -1):
            if analyzer.has_cn(vec, n):
                highest_n = n
                best_vec = analyzer.refine_c2_or_cn_axis(vec, n)
                break
        if highest_n >= 2:
            if axis_pool.add(best_vec):
                cn_axes.append((axis_pool.vectors[-1], highest_n))
            else:
                # Update existing axis order if higher
                for idx, (ex_vec, ex_n) in enumerate(cn_axes):
                    if abs(float(np.dot(best_vec, ex_vec))) >= axis_pool.cos_thresh:
                        if highest_n > ex_n:
                            cn_axes[idx] = (ex_vec, highest_n)
                        break

    has_inv = analyzer.has_inversion()
    n_sigmas = len(sigma_pool.vectors)

    # 4. Check for Polyhedral / Cubic / Icosahedral groups (T, Th, Td, O, Oh, I, Ih)
    # Count axes with order >= 3 (note: a C_6 axis also supports C_3)
    c5_axes = [ax for ax, order in cn_axes if order % 5 == 0 or analyzer.has_cn(ax, 5)]
    c4_axes = [ax for ax, order in cn_axes if order % 4 == 0 or analyzer.has_cn(ax, 4)]
    c3_axes = [ax for ax, order in cn_axes if order % 3 == 0 or analyzer.has_cn(ax, 3)]

    if len(c5_axes) >= 2:
        # Icosahedral family (I or Ih)
        return "Ih" if (has_inv or n_sigmas > 0) else "I"

    if len(c3_axes) >= 2:
        if len(c4_axes) >= 2:
            # Octahedral family (O or Oh)
            return "Oh" if (has_inv or n_sigmas > 0) else "O"
        # Tetrahedral family (T, Th, Td)
        if has_inv:
            return "Th"
        if n_sigmas > 0:
            return "Td"
        return "T"

    # 5. Low-symmetry groups without any C_n (n >= 2) axis
    if not cn_axes:
        if n_sigmas > 0:
            return "Cs"
        if has_inv:
            return "Ci"
        return "C1"

    # 6. Axial groups (Cn, Cnv, Cnh, Dn, Dnh, Dnd, Sn)
    # Select the principal axis: highest order n.
    # Tie-breaker (e.g. for D2 / D2d / D2h where multiple C2 axes exist):
    #   1) Axis supporting S_{2n} (for D2d -> S4 axis is the principal C2 axis)
    #   2) Axis perpendicular to a mirror plane (sigma_h)
    #   3) Axis contained in the most vertical mirror planes (sigma_v)
    max_order = max(order for _, order in cn_axes)
    top_axes = [ax for ax, order in cn_axes if order == max_order]

    def _axis_priority(ax: np.ndarray) -> tuple[int, int, int]:
        has_s2n = 1 if analyzer.has_sn(ax, 2 * max_order) else 0
        has_sh = 1 if analyzer.has_sigma(ax) else 0
        n_sv = sum(
            1
            for sig in sigma_pool.vectors
            if abs(float(np.dot(ax, sig))) <= math.sin(analyzer.ang_tol)
        )
        return (has_s2n, has_sh, n_sv)

    principal_axis = max(top_axes, key=_axis_priority)
    n = max_order

    # Check for perpendicular C2 axes
    perp_c2_count = 0
    for ax, order in cn_axes:
        if abs(float(np.dot(ax, principal_axis))) <= math.sin(analyzer.ang_tol):
            if order % 2 == 0 or analyzer.has_cn(ax, 2):
                perp_c2_count += 1

    # Also explicitly search for a perpendicular C2 if none found yet in cn_axes
    if perp_c2_count == 0:
        for vec in c2_pool.vectors:
            if abs(float(np.dot(vec, principal_axis))) <= math.sin(analyzer.ang_tol):
                perp_c2_count += 1

    has_sigma_h = analyzer.has_sigma(principal_axis)
    n_sigma_v = sum(
        1
        for sig in sigma_pool.vectors
        if abs(float(np.dot(principal_axis, sig))) <= math.sin(analyzer.ang_tol)
    )
    has_s2n = analyzer.has_sn(principal_axis, 2 * n)

    if perp_c2_count >= 1:
        # Dihedral family: Dn, Dnh, Dnd
        if has_sigma_h:
            return f"D{n}h"
        if has_s2n or n_sigma_v >= 1:
            return f"D{n}d"
        return f"D{n}"
    else:
        # Cyclic / Improper family: Cn, Cnv, Cnh, Sn
        if has_sigma_h:
            return f"C{n}h"
        if n_sigma_v >= 1:
            return f"C{n}v"
        if has_s2n:
            return f"S{2 * n}"
        return f"C{n}"


# Alias required by specification
point_group = detect_point_group


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for `pgmol`."""
    parser = argparse.ArgumentParser(
        prog="pgmol",
        description="Detect the Schoenflies point group of an XYZ molecular structure.",
    )
    parser.add_argument(
        "xyz_file",
        type=pathlib.Path,
        help="Path to input .xyz file.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="Cartesian distance tolerance in Angstroms (default: 1e-2).",
    )
    args = parser.parse_args(argv)
    pg = detect_point_group(args.xyz_file, tol=args.tol)
    print(pg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
