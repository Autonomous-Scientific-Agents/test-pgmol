from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Molecule:
    elements: tuple[str, ...]
    coords: np.ndarray

    @classmethod
    def from_xyz(cls, path_or_text: str | Path) -> Molecule:
        if isinstance(path_or_text, Path):
            text = path_or_text.read_text(encoding="utf-8")
        else:
            p = Path(path_or_text)
            if "\n" not in path_or_text and p.exists():
                text = p.read_text(encoding="utf-8")
            else:
                text = path_or_text

        raw_lines = text.splitlines()
        while raw_lines and not raw_lines[0].strip():
            raw_lines.pop(0)
        if not raw_lines:
            raise ValueError("Empty XYZ input")
        n_atoms = int(raw_lines[0].strip())
        atom_lines = [line.strip() for line in raw_lines[2:] if line.strip()]
        if len(atom_lines) < n_atoms:
            raise ValueError(f"Expected {n_atoms} atoms, found {len(atom_lines)}")
        atom_lines = atom_lines[:n_atoms]

        elements: list[str] = []
        coords = np.zeros((n_atoms, 3), dtype=np.float64)
        for i, line in enumerate(atom_lines):
            parts = line.split()
            elements.append(parts[0])
            coords[i, 0] = float(parts[1])
            coords[i, 1] = float(parts[2])
            coords[i, 2] = float(parts[3])

        return cls(elements=tuple(elements), coords=coords)


def _deduplicate_axes(axes: list[np.ndarray] | np.ndarray, angle_tol: float = 1e-3) -> list[np.ndarray]:
    if len(axes) == 0:
        return []
    arr = np.asarray(axes, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    norms = np.linalg.norm(arr, axis=1)
    valid_mask = norms > 1e-6
    if not np.any(valid_mask):
        return []
    arr = arr[valid_mask] / norms[valid_mask, None]

    # Canonical sign: make the largest-magnitude component positive (or first non-zero)
    for i in range(len(arr)):
        for c in arr[i]:
            if abs(c) > 1e-4:
                if c < 0:
                    arr[i] = -arr[i]
                break

    # Fast approximate bucketing by rounding to 2 decimal places first, then exact dot-product check
    unique: list[np.ndarray] = []
    unique_mat = np.empty((0, 3), dtype=np.float64)
    cos_thresh = 1.0 - 0.5 * (angle_tol * angle_tol)

    for u in arr:
        if len(unique) == 0:
            unique.append(u)
            unique_mat = u[None, :]
        else:
            dots = np.abs(unique_mat @ u)
            if np.max(dots) < cos_thresh:
                unique.append(u)
                unique_mat = np.vstack((unique_mat, u))
    return unique


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    u = axis / np.linalg.norm(axis)
    x, y, z = u
    c = np.cos(angle)
    s = np.sin(angle)
    oc = 1.0 - c
    return np.array(
        [
            [c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s],
            [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s],
            [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc],
        ],
        dtype=np.float64,
    )


def reflection_matrix(normal: np.ndarray) -> np.ndarray:
    n = normal / np.linalg.norm(normal)
    return np.eye(3, dtype=np.float64) - 2.0 * np.outer(n, n)


def improper_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    return reflection_matrix(axis) @ rotation_matrix(axis, angle)


class SymmetryAnalyzer:
    def __init__(self, molecule: Molecule, tol: float = 1e-2) -> None:
        self.molecule = molecule
        self.tol = tol
        self.tol_sq = tol * tol
        self.n_atoms = len(molecule.elements)

        # Center at unweighted geometric centroid (invariant to translation, rotation, permutation)
        if self.n_atoms > 0:
            self.centroid = np.mean(molecule.coords, axis=0)
            self.coords = molecule.coords - self.centroid
        else:
            self.centroid = np.zeros(3, dtype=np.float64)
            self.coords = molecule.coords.copy()

        # Group atom indices by element and distance shell from origin, sorted smallest shell first for fast rejection
        self.groups: list[np.ndarray] = []
        if self.n_atoms > 0:
            radii = np.linalg.norm(self.coords, axis=1)
            by_elem: dict[str, list[int]] = {}
            for i, el in enumerate(molecule.elements):
                by_elem.setdefault(el, []).append(i)

            raw_groups: list[np.ndarray] = []
            for el, idxs in by_elem.items():
                idx_arr = np.array(idxs, dtype=int)
                r_arr = radii[idx_arr]
                order = np.argsort(r_arr)
                idx_sorted = idx_arr[order]
                r_sorted = r_arr[order]

                start = 0
                for k in range(1, len(idx_sorted)):
                    if r_sorted[k] - r_sorted[k - 1] > self.tol:
                        raw_groups.append(self.coords[idx_sorted[start:k]])
                        start = k
                raw_groups.append(self.coords[idx_sorted[start:]])

            # Sort shells so small, non-origin shells are checked first
            raw_groups.sort(key=lambda g: (0 if np.linalg.norm(g[0]) > self.tol else 1, len(g)))
            self.groups = raw_groups

    def is_symmetry_op(self, op_matrix: np.ndarray) -> bool:
        for grp in self.groups:
            transformed = grp @ op_matrix.T
            diff = transformed[:, None, :] - grp[None, :, :]
            dist_sq = np.sum(diff * diff, axis=-1)
            if np.any(np.min(dist_sq, axis=1) > self.tol_sq):
                return False
            if np.any(np.min(dist_sq, axis=0) > self.tol_sq):
                return False
        return True

    def has_inversion(self) -> bool:
        return self.is_symmetry_op(-np.eye(3, dtype=np.float64))

    def has_reflection(self, normal: np.ndarray) -> bool:
        return self.is_symmetry_op(reflection_matrix(normal))

    def has_rotation(self, axis: np.ndarray, n: int) -> bool:
        return self.is_symmetry_op(rotation_matrix(axis, 2.0 * np.pi / n))

    def has_improper_rotation(self, axis: np.ndarray, n: int) -> bool:
        return self.is_symmetry_op(improper_rotation_matrix(axis, 2.0 * np.pi / n))

    def _candidate_axes(self) -> list[np.ndarray]:
        candidates: list[np.ndarray] = []

        # 1. Principal axes of covariance matrix of all atoms and of each shell
        cov_all = self.coords.T @ self.coords
        evals, evecs = np.linalg.eigh(cov_all)
        for j in range(3):
            candidates.append(evecs[:, j])

        for grp in self.groups:
            if len(grp) >= 2 and np.linalg.norm(grp[0]) > self.tol:
                cov_g = grp.T @ grp
                _, evecs_g = np.linalg.eigh(cov_g)
                for j in range(3):
                    candidates.append(evecs_g[:, j])

        # 2. Atom position vectors from non-origin shells (prefer smaller shells if huge)
        non_origin_groups = [g for g in self.groups if np.linalg.norm(g[0]) > self.tol]
        # Use up to first 2 smallest shells (or any shell with <= 24 atoms) for exhaustive pair/triplet generation
        rep_groups = [g for g in non_origin_groups if len(g) <= 24]
        if not rep_groups and non_origin_groups:
            rep_groups = [non_origin_groups[0]]

        for grp in rep_groups:
            for r in grp:
                candidates.append(r)

            m = len(grp)
            # Pairwise sums, differences, and cross products
            for i in range(m):
                ri = grp[i]
                for j in range(i + 1, m):
                    rj = grp[j]
                    candidates.append(ri + rj)
                    candidates.append(ri - rj)
                    cr = np.cross(ri, rj)
                    if np.linalg.norm(cr) > 1e-4:
                        candidates.append(cr)

        # 3. For larger shells (>24 atoms, e.g. C60), only use nearest-neighbor pairs and triplets
        for grp in non_origin_groups:
            m = len(grp)
            if m > 24:
                for r in grp:
                    candidates.append(r)
                dists = np.linalg.norm(grp[:, None, :] - grp[None, :, :], axis=-1)
                for i in range(m):
                    order = np.argsort(dists[i])
                    # 6 nearest neighbors
                    for j in order[1:7]:
                        if j > i:
                            candidates.append(grp[i] + grp[j])
                            candidates.append(grp[i] - grp[j])

            if m >= 3:
                dists = np.linalg.norm(grp[:, None, :] - grp[None, :, :], axis=-1)
                for i in range(min(m, 5)):
                    d_row = dists[i]
                    order = np.argsort(d_row)
                    neighbors = [int(idx) for idx in order[1:8] if d_row[idx] > self.tol]
                    for a_idx in range(len(neighbors)):
                        j = neighbors[a_idx]
                        for b_idx in range(a_idx + 1, len(neighbors)):
                            k = neighbors[b_idx]
                            candidates.append(grp[i] + grp[j] + grp[k])
                            n_vec = np.cross(grp[j] - grp[i], grp[k] - grp[i])
                            if np.linalg.norm(n_vec) > 1e-4:
                                candidates.append(n_vec)

        return _deduplicate_axes(candidates)

    def classify(self) -> str:
        if self.n_atoms <= 1:
            return "Kh"

        radii = np.linalg.norm(self.coords, axis=1)
        max_idx = int(np.argmax(radii))
        ref_vec = self.coords[max_idx]
        ref_norm = radii[max_idx]
        if ref_norm < self.tol:
            return "Kh"

        # Linearity check
        axis = ref_vec / ref_norm
        cross_norms = np.linalg.norm(np.cross(self.coords, axis), axis=1)
        if np.all(cross_norms < self.tol):
            return "Dinfh" if self.has_inversion() else "Cinfv"

        candidates = self._candidate_axes()

        axes_by_order: dict[int, list[np.ndarray]] = {}
        c2_axes: list[np.ndarray] = []

        # Possible rotation orders must divide the size of at least one non-collinear shell, or be <= 8
        shell_sizes = [len(g) for g in self.groups if np.linalg.norm(g[0]) > self.tol]
        max_shell = max(shell_sizes, default=1)
        possible_orders = sorted(
            {
                n
                for n in range(3, min(max_shell, 12) + 1)
                if any(s % n == 0 or (s - 1) % n == 0 or (s - 2) % n == 0 for s in shell_sizes)
            },
            reverse=True,
        )

        for ax in candidates:
            if self.has_rotation(ax, 2):
                c2_axes.append(ax)
            for n in possible_orders:
                if self.has_rotation(ax, n):
                    axes_by_order.setdefault(n, []).append(ax)
                    break

        # Check cross products and bisectors of discovered axes for additional symmetry axes
        extra_candidates: list[np.ndarray] = []
        high_axes = [ax for n, ax_list in axes_by_order.items() if n >= 3 for ax in ax_list]
        base_axes = high_axes[:12] if high_axes else c2_axes[:12]
        for i in range(len(base_axes)):
            for j in range(i + 1, len(base_axes)):
                cr = np.cross(base_axes[i], base_axes[j])
                if np.linalg.norm(cr) > 1e-5:
                    extra_candidates.append(cr)
                extra_candidates.append(base_axes[i] + base_axes[j])
                extra_candidates.append(base_axes[i] - base_axes[j])

        for ax in _deduplicate_axes(extra_candidates):
            if self.has_rotation(ax, 2):
                c2_axes.append(ax)
            for n in possible_orders:
                if self.has_rotation(ax, n):
                    axes_by_order.setdefault(n, []).append(ax)
                    break

        c2_axes = _deduplicate_axes(c2_axes)
        for n in list(axes_by_order.keys()):
            axes_by_order[n] = _deduplicate_axes(axes_by_order[n])

        inv = self.has_inversion()

        # Check for multiple high-order axes (Cubic / Icosahedral: T, Td, Th, O, Oh, I, Ih)
        c5_axes = axes_by_order.get(5, [])
        c4_axes = axes_by_order.get(4, [])
        c3_axes = axes_by_order.get(3, [])

        if len(c5_axes) >= 2 or (
            len(c5_axes) >= 1 and len(c3_axes) >= 1 and abs(np.dot(c5_axes[0], c3_axes[0])) < 0.99
        ):
            return "Ih" if inv else "I"

        if len(c4_axes) >= 2 or (len(c4_axes) >= 1 and len(c3_axes) >= 2):
            return "Oh" if inv else "O"

        if len(c3_axes) >= 2:
            if inv:
                return "Th"
            for c2 in c2_axes:
                if self.has_improper_rotation(c2, 4):
                    return "Td"
            for i in range(len(c3_axes)):
                for j in range(i + 1, len(c3_axes)):
                    n_plane = np.cross(c3_axes[i], c3_axes[j])
                    if np.linalg.norm(n_plane) > 1e-5 and self.has_reflection(n_plane):
                        return "Td"
            return "T"

        # Axial or low symmetry groups
        all_orders = sorted(axes_by_order.keys(), reverse=True)
        if not all_orders and not c2_axes:
            if inv:
                return "Ci"
            for cand in candidates:
                if self.has_reflection(cand):
                    return "Cs"
            return "C1"

        if all_orders:
            n = all_orders[0]
            principal_candidates = axes_by_order[n]
        else:
            n = 2
            principal_candidates = c2_axes

        principal_axis = principal_candidates[0]
        if n == 2 and len(principal_candidates) > 1:
            for ax in principal_candidates:
                if self.has_improper_rotation(ax, 4):
                    principal_axis = ax
                    break

        has_s2n = self.has_improper_rotation(principal_axis, 2 * n)

        perp_c2s = [c2 for c2 in c2_axes if abs(np.dot(c2, principal_axis)) < 1e-2]
        if not perp_c2s:
            perp_c2 = self._find_perpendicular_c2(principal_axis)
            if perp_c2 is not None:
                perp_c2s.append(perp_c2)

        has_perp_c2 = len(perp_c2s) > 0
        has_sigma_h = self.has_reflection(principal_axis)
        has_sigma_v = self._has_vertical_mirror(principal_axis, perp_c2s, candidates)

        if has_perp_c2:
            if has_sigma_h:
                return f"D{n}h"
            if has_s2n or has_sigma_v:
                return f"D{n}d"
            return f"D{n}"
        else:
            if has_sigma_h:
                return f"C{n}h"
            if has_sigma_v:
                return f"C{n}v"
            if has_s2n:
                return f"S{2 * n}"
            return f"C{n}"

    def _find_perpendicular_c2(self, principal_axis: np.ndarray) -> np.ndarray | None:
        proj_vecs: list[np.ndarray] = []
        for grp in self.groups[:3]:
            proj = grp - np.outer(grp @ principal_axis, principal_axis)
            norms = np.linalg.norm(proj, axis=1)
            valid = proj[norms > self.tol]
            for i in range(len(valid)):
                u_i = valid[i] / np.linalg.norm(valid[i])
                proj_vecs.append(u_i)
                for j in range(i + 1, len(valid)):
                    u_j = valid[j] / np.linalg.norm(valid[j])
                    proj_vecs.append(u_i + u_j)
                    proj_vecs.append(u_i - u_j)

        for v in _deduplicate_axes(proj_vecs):
            if abs(np.dot(v, principal_axis)) < 1e-2 and self.has_rotation(v, 2):
                return v
        return None

    def _has_vertical_mirror(
        self,
        principal_axis: np.ndarray,
        perp_c2s: list[np.ndarray],
        candidates: list[np.ndarray],
    ) -> bool:
        normals: list[np.ndarray] = []
        for cand in candidates:
            if abs(np.dot(cand, principal_axis)) < 1e-2:
                normals.append(cand)
            cr = np.cross(principal_axis, cand)
            if np.linalg.norm(cr) > 1e-5:
                normals.append(cr)

        for i in range(len(perp_c2s)):
            c1 = perp_c2s[i]
            normals.append(c1)
            normals.append(np.cross(principal_axis, c1))
            for j in range(i + 1, len(perp_c2s)):
                c2 = perp_c2s[j]
                normals.append(c1 + c2)
                normals.append(c1 - c2)

        for grp in self.groups[:3]:
            proj = grp - np.outer(grp @ principal_axis, principal_axis)
            norms = np.linalg.norm(proj, axis=1)
            valid = proj[norms > self.tol]
            for i in range(len(valid)):
                u_i = valid[i] / np.linalg.norm(valid[i])
                normals.append(u_i)
                normals.append(np.cross(principal_axis, u_i))
                for j in range(i + 1, len(valid)):
                    u_j = valid[j] / np.linalg.norm(valid[j])
                    if np.linalg.norm(u_i + u_j) > 1e-5:
                        normals.append(u_i + u_j)
                        normals.append(np.cross(principal_axis, u_i + u_j))
                    if np.linalg.norm(u_i - u_j) > 1e-5:
                        normals.append(u_i - u_j)
                        normals.append(np.cross(principal_axis, u_i - u_j))

        for n_vec in _deduplicate_axes(normals):
            if abs(np.dot(n_vec, principal_axis)) < 1e-2 and self.has_reflection(n_vec):
                return True
        return False


def detect_point_group(path_or_molecule: str | Path | Molecule, tol: float = 1e-2) -> str:
    if isinstance(path_or_molecule, Molecule):
        mol = path_or_molecule
    else:
        mol = Molecule.from_xyz(path_or_molecule)
    return SymmetryAnalyzer(mol, tol=tol).classify()
