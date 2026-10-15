from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import numpy.typing as npt

__all__ = [
    "Molecule",
    "SymmetryAnalyzer",
    "detect_point_group",
    "rotation_matrix",
    "reflection_matrix",
    "improper_rotation_matrix",
]


def _cross3(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class Molecule:
    elements: tuple[str, ...]
    coords: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        elems = tuple(str(e) for e in self.elements)
        arr = np.asarray(self.coords, dtype=np.float64)
        if arr.ndim == 1 and arr.size == 0:
            arr = arr.reshape((0, 3))
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"coords must have shape (N, 3), got {arr.shape}")
        if len(elems) != arr.shape[0]:
            raise ValueError(f"Length of elements ({len(elems)}) does not match coords ({arr.shape[0]})")
        object.__setattr__(self, "elements", elems)
        object.__setattr__(self, "coords", arr)

    def __len__(self) -> int:
        return len(self.elements)

    @classmethod
    def from_xyz(cls, path_or_text: str | Path) -> Molecule:
        if isinstance(path_or_text, Path):
            if not path_or_text.is_file():
                raise FileNotFoundError(f"XYZ file not found: {path_or_text}")
            text = path_or_text.read_text(encoding="utf-8")
        elif isinstance(path_or_text, str):
            stripped = path_or_text.strip()
            if not stripped:
                raise ValueError("Empty XYZ input")
            if "\n" not in path_or_text:
                p = Path(path_or_text)
                try:
                    is_file = p.is_file()
                except OSError:
                    is_file = False
                if is_file:
                    text = p.read_text(encoding="utf-8")
                elif path_or_text.endswith(".xyz"):
                    raise FileNotFoundError(f"XYZ file not found: {path_or_text}")
                else:
                    text = path_or_text
            else:
                text = path_or_text
        else:
            raise TypeError(f"Expected str or Path, got {type(path_or_text).__name__}")

        raw_lines = text.splitlines()
        while raw_lines and not raw_lines[0].strip():
            raw_lines.pop(0)
        if not raw_lines:
            raise ValueError("Empty XYZ input")
        try:
            n_atoms = int(raw_lines[0].strip())
        except ValueError as exc:
            raise ValueError(f"Invalid atom count on line 1: {raw_lines[0]!r}") from exc
        if n_atoms < 0:
            raise ValueError(f"Negative atom count: {n_atoms}")

        atom_lines = [line.strip() for line in raw_lines[2:] if line.strip()]
        if len(atom_lines) < n_atoms:
            raise ValueError(f"Expected {n_atoms} atoms, found {len(atom_lines)}")
        atom_lines = atom_lines[:n_atoms]

        elements: list[str] = []
        coords = np.zeros((n_atoms, 3), dtype=np.float64)
        for i, line in enumerate(atom_lines):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Malformed XYZ atom line {i + 1}: {line!r}")
            elements.append(parts[0])
            try:
                coords[i, 0] = float(parts[1])
                coords[i, 1] = float(parts[2])
                coords[i, 2] = float(parts[3])
            except ValueError as exc:
                raise ValueError(f"Invalid coordinates on atom line {i + 1}: {line!r}") from exc

        return cls(elements=tuple(elements), coords=coords)


def _deduplicate_axes(
    axes: Sequence[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
    angle_tol: float = 5e-2,
) -> list[npt.NDArray[np.float64]]:
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

    n_vecs = arr.shape[0]
    unique_buf = np.empty((n_vecs, 3), dtype=np.float64)
    unique_list: list[npt.NDArray[np.float64]] = []
    count = 0
    cos_thresh = float(np.cos(angle_tol))

    for i in range(n_vecs):
        u = arr[i]
        # Canonical sign: make the first non-zero component positive
        if u[0] < -1e-4 or (abs(u[0]) <= 1e-4 and (u[1] < -1e-4 or (abs(u[1]) <= 1e-4 and u[2] < 0))):
            u = -u
        if count == 0:
            unique_buf[0] = u
            unique_list.append(u)
            count = 1
        else:
            dots = np.abs(unique_buf[:count] @ u)
            if float(np.max(dots)) < cos_thresh:
                unique_buf[count] = u
                unique_list.append(u)
                count += 1
    return unique_list


def _has_distinct_pair(axes: Sequence[npt.NDArray[np.float64]], cos_max: float = 0.95) -> bool:
    m = len(axes)
    for i in range(m):
        for j in range(i + 1, m):
            if abs(float(np.dot(axes[i], axes[j]))) < cos_max:
                return True
    return False


def rotation_matrix(axis: npt.NDArray[np.float64], angle: float) -> npt.NDArray[np.float64]:
    u = axis / np.linalg.norm(axis)
    x, y, z = float(u[0]), float(u[1]), float(u[2])
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    oc = 1.0 - c
    return np.array(
        [
            [c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s],
            [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s],
            [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc],
        ],
        dtype=np.float64,
    )


def reflection_matrix(normal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    n = normal / np.linalg.norm(normal)
    return np.eye(3, dtype=np.float64) - 2.0 * np.outer(n, n)


def improper_rotation_matrix(axis: npt.NDArray[np.float64], angle: float) -> npt.NDArray[np.float64]:
    return reflection_matrix(axis) @ rotation_matrix(axis, angle)


class SymmetryAnalyzer:
    def __init__(self, molecule: Molecule, tol: float = 1e-2) -> None:
        self.molecule = molecule
        self.tol = tol
        self.tol_sq = tol * tol
        self.n_atoms = len(molecule.elements)
        self._cached_pg: str | None = None

        # Center at unweighted geometric centroid (invariant to translation, rotation, permutation)
        if self.n_atoms > 0:
            self.centroid = np.mean(molecule.coords, axis=0)
            self.coords = molecule.coords - self.centroid
        else:
            self.centroid = np.zeros(3, dtype=np.float64)
            self.coords = molecule.coords.copy()

        # Group atom indices by element and distance shell from origin, sorted smallest shell first for fast rejection
        self.groups: list[npt.NDArray[np.float64]] = []
        if self.n_atoms > 0:
            radii = np.linalg.norm(self.coords, axis=1)
            by_elem: dict[str, list[int]] = {}
            for i, el in enumerate(molecule.elements):
                by_elem.setdefault(el, []).append(i)

            raw_groups: list[npt.NDArray[np.float64]] = []
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

            raw_groups.sort(key=lambda g: (0 if np.linalg.norm(g[0]) > self.tol else 1, len(g)))
            self.groups = raw_groups

    @property
    def point_group(self) -> str:
        if self._cached_pg is None:
            self._cached_pg = self.classify()
        return self._cached_pg

    def is_symmetry_op(self, op_matrix: npt.NDArray[np.float64]) -> bool:
        tol_sq = self.tol_sq
        for grp in self.groups:
            # Fast single-atom check for larger shells before full matrix distance calculation
            if grp.shape[0] > 8:
                first_t = grp[0] @ op_matrix.T
                d0 = grp - first_t
                if float(np.min(np.sum(d0 * d0, axis=1))) > tol_sq:
                    return False

            transformed = grp @ op_matrix.T
            diff = transformed[:, None, :] - grp[None, :, :]
            dist_sq = np.sum(diff * diff, axis=-1)
            if float(np.max(np.min(dist_sq, axis=1))) > tol_sq:
                return False
            if float(np.max(np.min(dist_sq, axis=0))) > tol_sq:
                return False
        return True

    def has_inversion(self) -> bool:
        return self.is_symmetry_op(-np.eye(3, dtype=np.float64))

    def has_reflection(self, normal: npt.NDArray[np.float64]) -> bool:
        return self.is_symmetry_op(reflection_matrix(normal))

    def has_rotation(self, axis: npt.NDArray[np.float64], n: int) -> bool:
        return self.is_symmetry_op(rotation_matrix(axis, 2.0 * np.pi / n))

    def has_improper_rotation(self, axis: npt.NDArray[np.float64], n: int) -> bool:
        return self.is_symmetry_op(improper_rotation_matrix(axis, 2.0 * np.pi / n))

    def _candidate_axes(self) -> list[npt.NDArray[np.float64]]:
        candidates: list[npt.NDArray[np.float64]] = []

        # 1. Principal axes of covariance matrix of all atoms and of each shell
        cov_all = self.coords.T @ self.coords
        _, evecs = np.linalg.eigh(cov_all)
        for j in range(3):
            candidates.append(evecs[:, j])

        for grp in self.groups:
            if len(grp) >= 2 and np.linalg.norm(grp[0]) > self.tol:
                cov_g = grp.T @ grp
                _, evecs_g = np.linalg.eigh(cov_g)
                for j in range(3):
                    candidates.append(evecs_g[:, j])

        # 2. Atom position vectors from non-origin shells
        non_origin_groups = [g for g in self.groups if np.linalg.norm(g[0]) > self.tol]
        rep_groups = [g for g in non_origin_groups if len(g) <= 24]
        if not rep_groups and non_origin_groups:
            rep_groups = [non_origin_groups[0]]

        for grp in rep_groups:
            for r in grp:
                candidates.append(r)

            m = len(grp)
            for i in range(m):
                ri = grp[i]
                for j in range(i + 1, m):
                    rj = grp[j]
                    candidates.append(ri + rj)
                    candidates.append(ri - rj)
                    cr = _cross3(ri, rj)
                    if cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2] > 1e-8:
                        candidates.append(cr)

        # 3. For larger shells (>24 atoms, e.g. C60), use nearest-neighbor pairs and triplets
        for grp in non_origin_groups:
            m = len(grp)
            if m > 24:
                for r in grp:
                    candidates.append(r)
                dists = np.linalg.norm(grp[:, None, :] - grp[None, :, :], axis=-1)
                for i in range(m):
                    order = np.argsort(dists[i])
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
                            n_vec = _cross3(grp[j] - grp[i], grp[k] - grp[i])
                            if n_vec[0] * n_vec[0] + n_vec[1] * n_vec[1] + n_vec[2] * n_vec[2] > 1e-8:
                                candidates.append(n_vec)

        return _deduplicate_axes(candidates)

    def classify(self) -> str:
        if self.n_atoms <= 1:
            return "Kh"

        radii = np.linalg.norm(self.coords, axis=1)
        max_idx = int(np.argmax(radii))
        ref_norm = float(radii[max_idx])
        if ref_norm < self.tol:
            return "Kh"

        # Linearity check using principal eigenvector of covariance matrix (with furthest atom fallback)
        cov_all = self.coords.T @ self.coords
        _, evecs = np.linalg.eigh(cov_all)
        axis = evecs[:, 2]
        if np.linalg.norm(axis) < 1e-8:
            axis = self.coords[max_idx] / ref_norm
        cross_norms = np.linalg.norm(np.cross(self.coords, axis), axis=1)
        if np.all(cross_norms < self.tol):
            return "Dinfh" if self.has_inversion() else "Cinfv"

        candidates = self._candidate_axes()

        axes_by_order: dict[int, list[npt.NDArray[np.float64]]] = {}
        c2_axes: list[npt.NDArray[np.float64]] = []

        shell_sizes = [len(g) for g in self.groups if np.linalg.norm(g[0]) > self.tol]
        max_shell = max(shell_sizes, default=1)
        possible_orders = sorted(
            {
                n
                for n in range(3, max_shell + 1)
                if any(s % n == 0 or (s - 1) % n == 0 or (s - 2) % n == 0 for s in shell_sizes)
            },
            reverse=True,
        )

        for ax in candidates:
            is_c2 = self.has_rotation(ax, 2)
            if is_c2:
                c2_axes.append(ax)
            for n in possible_orders:
                if n % 2 == 0 and not is_c2:
                    continue
                if self.has_rotation(ax, n):
                    axes_by_order.setdefault(n, []).append(ax)
                    break

        # Check cross products and bisectors of discovered axes for additional symmetry axes
        extra_candidates: list[npt.NDArray[np.float64]] = []
        high_axes = [ax for n, ax_list in axes_by_order.items() if n >= 3 for ax in ax_list]
        base_axes = high_axes[:12] if high_axes else c2_axes[:12]
        for i in range(len(base_axes)):
            for j in range(i + 1, len(base_axes)):
                cr = _cross3(base_axes[i], base_axes[j])
                if cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2] > 1e-8:
                    extra_candidates.append(cr)
                extra_candidates.append(base_axes[i] + base_axes[j])
                extra_candidates.append(base_axes[i] - base_axes[j])

        for ax in _deduplicate_axes(extra_candidates):
            is_c2 = self.has_rotation(ax, 2)
            if is_c2:
                c2_axes.append(ax)
            for n in possible_orders:
                if n % 2 == 0 and not is_c2:
                    continue
                if self.has_rotation(ax, n):
                    axes_by_order.setdefault(n, []).append(ax)
                    break

        c2_axes = _deduplicate_axes(c2_axes)
        for n in list(axes_by_order.keys()):
            axes_by_order[n] = _deduplicate_axes(axes_by_order[n])

        inv = self.has_inversion()

        # Check for multiple distinct high-order axes (Cubic / Icosahedral: T, Td, Th, O, Oh, I, Ih)
        c5_axes = axes_by_order.get(5, [])
        c4_axes = axes_by_order.get(4, [])
        c3_axes = axes_by_order.get(3, [])

        if _has_distinct_pair(c5_axes) or (
            len(c5_axes) >= 1
            and len(c3_axes) >= 1
            and any(abs(float(np.dot(c5_axes[0], c3))) < 0.95 for c3 in c3_axes)
        ):
            return "Ih" if inv else "I"

        if _has_distinct_pair(c4_axes) or (len(c4_axes) >= 1 and _has_distinct_pair(c3_axes)):
            return "Oh" if inv else "O"

        if _has_distinct_pair(c3_axes):
            if inv:
                return "Th"
            for c2 in c2_axes:
                if self.has_improper_rotation(c2, 4):
                    return "Td"
            for i in range(len(c3_axes)):
                for j in range(i + 1, len(c3_axes)):
                    n_plane = _cross3(c3_axes[i], c3_axes[j])
                    if np.linalg.norm(n_plane) > 1e-2 and self.has_reflection(n_plane):
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

        perp_c2s = [c2 for c2 in c2_axes if abs(float(np.dot(c2, principal_axis))) < 5e-2]
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

    def _find_perpendicular_c2(self, principal_axis: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | None:
        proj_vecs: list[npt.NDArray[np.float64]] = []
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
            if abs(float(np.dot(v, principal_axis))) < 5e-2 and self.has_rotation(v, 2):
                return v
        return None

    def _has_vertical_mirror(
        self,
        principal_axis: npt.NDArray[np.float64],
        perp_c2s: list[npt.NDArray[np.float64]],
        candidates: list[npt.NDArray[np.float64]],
    ) -> bool:
        normals: list[npt.NDArray[np.float64]] = []
        for cand in candidates:
            if abs(float(np.dot(cand, principal_axis))) < 5e-2:
                normals.append(cand)
            cr = _cross3(principal_axis, cand)
            if cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2] > 1e-8:
                normals.append(cr)

        for i in range(len(perp_c2s)):
            c1 = perp_c2s[i]
            normals.append(c1)
            normals.append(_cross3(principal_axis, c1))
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
                normals.append(_cross3(principal_axis, u_i))
                for j in range(i + 1, len(valid)):
                    u_j = valid[j] / np.linalg.norm(valid[j])
                    s_ij = u_i + u_j
                    if s_ij[0] * s_ij[0] + s_ij[1] * s_ij[1] + s_ij[2] * s_ij[2] > 1e-8:
                        normals.append(s_ij)
                        normals.append(_cross3(principal_axis, s_ij))
                    d_ij = u_i - u_j
                    if d_ij[0] * d_ij[0] + d_ij[1] * d_ij[1] + d_ij[2] * d_ij[2] > 1e-8:
                        normals.append(d_ij)
                        normals.append(_cross3(principal_axis, d_ij))

        for n_vec in _deduplicate_axes(normals):
            if abs(float(np.dot(n_vec, principal_axis))) < 5e-2 and self.has_reflection(n_vec):
                return True
        return False


def detect_point_group(path_or_xyz_or_molecule: str | Path | Molecule, tol: float = 1e-2) -> str:
    if isinstance(path_or_xyz_or_molecule, Molecule):
        mol = path_or_xyz_or_molecule
    else:
        mol = Molecule.from_xyz(path_or_xyz_or_molecule)
    return SymmetryAnalyzer(mol, tol=tol).point_group
