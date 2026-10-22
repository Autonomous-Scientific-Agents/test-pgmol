# pgmol/__init__.py

import itertools
import math
import pathlib
from collections import Counter

import numpy as np

# This is a self-contained module. All code is in this file.

# Public API
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
    symbols, coords = _parse_xyz(xyz_input)

    if not symbols:
        raise ValueError("No atoms found in input.")

    if len(symbols) == 1:
        return "Kh"

    finder = _PointGroupFinder(symbols, coords, tol)
    return finder.run()

point_group = detect_point_group


# Helper functions
def _parse_xyz(xyz_input: str | pathlib.Path) -> tuple[list[str], np.ndarray]:
    """Parses XYZ file format from a path or a raw string."""
    if isinstance(xyz_input, pathlib.Path):
        lines = xyz_input.read_text().strip().splitlines()
    elif isinstance(xyz_input, str) and ("\n" in xyz_input or "\r" in xyz_input):
        lines = xyz_input.strip().splitlines()
    else:
        with open(xyz_input, 'r') as f:
            lines = f.read().strip().splitlines()

    if not lines or len(lines) < 1:
        return [], np.array([])

    try:
        num_atoms = int(lines[0].strip())
    except (ValueError, IndexError):
        raise ValueError("Invalid XYZ format: Cannot parse number of atoms.")

    if len(lines) < num_atoms + 2:
        raise ValueError(f"Invalid XYZ format: Expected {num_atoms} atoms, but file has fewer lines.")

    symbols = []
    coords = []
    for line in lines[2:num_atoms + 2]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            symbols.append(parts[0].capitalize())
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except (ValueError, IndexError):
            raise ValueError(f"Invalid XYZ format: Could not parse line: '{line}'")

    if len(symbols) != num_atoms:
        raise ValueError(f"XYZ format error: Header says {num_atoms} atoms, but {len(symbols)} were found.")

    return symbols, np.array(coords)


def _get_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Generates a rotation matrix using Rodrigues' formula."""
    axis = axis / np.linalg.norm(axis)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return (
        cos_a * np.identity(3)
        + sin_a * np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        + (1 - cos_a) * np.outer(axis, axis)
    )


class _PointGroupFinder:
    """Main class to find the point group of a molecule."""

    def __init__(self, symbols: list[str], coords: np.ndarray, tol: float):
        self.symbols = np.array(symbols)
        self.coords = coords.copy()
        self.tol = tol
        self.n_atoms = len(symbols)
        self._cache = {}

        # 1. Center molecule at geometric center
        self.coords -= np.mean(self.coords, axis=0)

        # 2. Group atoms by element for efficient comparison
        self.unique_symbols = sorted(list(set(self.symbols)))
        self.atom_groups = {s: np.where(self.symbols == s)[0] for s in self.unique_symbols}

    def _geom_is_superimposable(self, coords1: np.ndarray, coords2: np.ndarray) -> bool:
        """Checks if two geometries are superimposable, respecting atom types."""
        if coords1.shape != coords2.shape:
            return False
        for s in self.unique_symbols:
            indices = self.atom_groups[s]
            if not indices.size:
                continue

            group1 = coords1[indices, :]
            group2 = coords2[indices, :]

            # A canonical representation for a set of points is to sort them.
            # This is invariant to permutation.
            sorted1 = group1[np.lexsort(group1.T)]
            sorted2 = group2[np.lexsort(group2.T)]

            if not np.allclose(sorted1, sorted2, atol=self.tol, rtol=0):
                return False
        return True

    def _check_op(self, op_matrix: np.ndarray) -> bool:
        """Checks if a 3x3 matrix represents a valid symmetry operation."""
        new_coords = self.coords @ op_matrix.T
        return self._geom_is_superimposable(self.coords, new_coords)

    def _check_inversion(self) -> bool:
        if "inversion" not in self._cache:
            self._cache["inversion"] = self._geom_is_superimposable(self.coords, -self.coords)
        return self._cache["inversion"]

    def _analyze_inertia(self) -> tuple[str, np.ndarray]:
        """Determines molecular shape and principal axes from inertia tensor."""
        if self.n_atoms == 0:
            return "empty", np.identity(3)
        
        inertia_tensor = np.sum(
            [(np.dot(c, c) * np.identity(3) - np.outer(c, c)) for c in self.coords], axis=0
        )

        moments, axes = np.linalg.eigh(inertia_tensor)
        sorted_indices = np.argsort(moments)
        moments = moments[sorted_indices]
        axes = axes[:, sorted_indices].T

        norm_moments = moments / np.max(moments) if np.max(moments) > self.tol else moments

        is_linear = norm_moments[0] < self.tol
        is_spherical = np.allclose(norm_moments, norm_moments[0], atol=self.tol * 10, rtol=0)
        
        shape = 'asymmetric_top'
        if is_linear: shape = 'linear'
        elif is_spherical: shape = 'spherical'
        elif np.isclose(norm_moments[0], norm_moments[1]) or np.isclose(norm_moments[1], norm_moments[2]):
            shape = 'symmetric_top'
            
        return shape, axes

    def _get_candidate_axes(self, principal_axes: np.ndarray) -> list[np.ndarray]:
        """Generates a set of unique candidate axes for symmetry detection."""
        axes = list(principal_axes)
        axes.extend([c for c in self.coords if np.linalg.norm(c) > self.tol])
        if self.n_atoms < 35:  # Performance limit for pairwise vectors
            for i, j in itertools.combinations(range(self.n_atoms), 2):
                vec = self.coords[i] - self.coords[j]
                if np.linalg.norm(vec) > self.tol:
                    axes.append(vec)

        unique_axes = []
        for axis in axes:
            norm_axis = axis / np.linalg.norm(axis)
            is_duplicate = any(np.allclose(norm_axis, ua, atol=1e-2) or np.allclose(-norm_axis, ua, atol=1e-2) for ua in unique_axes)
            if not is_duplicate:
                unique_axes.append(norm_axis)
        return unique_axes

    def _find_rotations(self, principal_axes: np.ndarray) -> tuple[int, np.ndarray | None, dict]:
        """Finds all rotation axes and the principal axis."""
        if "rotations" in self._cache:
            return self._cache["rotations"]

        candidate_axes = self._get_candidate_axes(principal_axes)
        rotations = {}
        max_n = 1
        principal_axis = np.array([0.0, 0.0, 1.0]) if self.n_atoms > 0 else None

        for axis in candidate_axes:
            for n in range(2, max(8, self.n_atoms) + 1):
                mat = _get_rotation_matrix(axis, 2 * math.pi / n)
                if self._check_op(mat):
                    if n not in rotations: rotations[n] = []
                    if not any(np.allclose(axis, ra, atol=1e-2) or np.allclose(-axis, ra, atol=1e-2) for ra in rotations[n]):
                        rotations[n].append(axis)
                    if n > max_n:
                        max_n = n
                        principal_axis = axis
                elif n == 2 and 4 in rotations and any(np.allclose(axis, ra, atol=1e-2) or np.allclose(-axis, ra, atol=1e-2) for ra in rotations[4]):
                    break  # C4 implies C2, no need to check

        self._cache["rotations"] = (max_n, principal_axis, rotations)
        return max_n, principal_axis, rotations

    def run(self) -> str:
        """Executes the point group determination flowchart."""
        shape, principal_axes = self._analyze_inertia()
        
        # 1. Linear molecules
        if shape == 'linear':
            return 'Dinfh' if self._check_inversion() else 'Cinfv'
        
        max_n, principal_axis, rotations = self._find_rotations(principal_axes)

        # 2. High-symmetry groups (I, O, T)
        n_counts = {n: len(axes) for n, axes in rotations.items()}
        if n_counts.get(5, 0) >= 6:
            return 'Ih' if self._check_inversion() else 'I'
        if n_counts.get(4, 0) >= 3 and n_counts.get(3, 0) >= 4:
            return 'Oh' if self._check_inversion() else 'O'
        if n_counts.get(3, 0) >= 4:
            has_reflection = False
            for axis in self._get_candidate_axes(principal_axes):
                refl_op = np.identity(3) - 2 * np.outer(axis, axis)
                if self._check_op(refl_op):
                    has_reflection = True
                    break
            if has_reflection: return 'Td'
            return 'Th' if self._check_inversion() else 'T'

        # 3. Low-symmetry groups (C1, Cs, Ci)
        if max_n == 1:
            if self._check_inversion(): return 'Ci'
            for axis in self._get_candidate_axes(principal_axes):
                if self._check_op(np.identity(3) - 2 * np.outer(axis, axis)): return 'Cs'
            return 'C1'

        # 4. Axial groups (D, C, S)
        z_axis = np.array([0., 0., 1.])
        if principal_axis is not None and not np.allclose(principal_axis, z_axis) and not np.allclose(principal_axis, -z_axis):
            align_R = _get_rotation_matrix(np.cross(principal_axis, z_axis), np.arccos(np.dot(principal_axis, z_axis)))
            self.coords = self.coords @ align_R.T
            self._cache.pop("rotations", None)
            max_n, principal_axis, rotations = self._find_rotations(np.identity(3))
        
        perp_c2_count = sum(1 for c2 in rotations.get(2, []) if abs(np.dot(c2, z_axis)) < self.tol)
        has_perp_c2s = perp_c2_count >= max_n
        
        has_sigma_h = self._check_op(np.diag([1., 1., -1.]))
        
        if has_perp_c2s:  # D groups
            if has_sigma_h: return f'D{max_n}h'
            # Check for sigma_d by checking for any vertical reflection plane
            for angle in np.linspace(0, math.pi, max(12, 2*max_n), endpoint=False):
                normal = np.array([math.cos(angle), math.sin(angle), 0])
                if self._check_op(np.identity(3) - 2 * np.outer(normal, normal)):
                    return f'D{max_n}d'
            return f'D{max_n}'
        else:  # C and S groups
            if has_sigma_h: return f'C{max_n}h'
            
            has_sigma_v = False
            for angle in np.linspace(0, math.pi, max(12, 2*max_n), endpoint=False):
                 normal = np.array([math.cos(angle), math.sin(angle), 0])
                 if self._check_op(np.identity(3) - 2 * np.outer(normal, normal)):
                     has_sigma_v = True
                     break
            if has_sigma_v: return f'C{max_n}v'

            if 2 * max_n > 2:
                refl_op = np.diag([1, 1, -1])
                rot_op = _get_rotation_matrix(z_axis, math.pi / max_n)
                if self._check_op(rot_op @ refl_op):
                    return f'S{2 * max_n}'

            return f'C{max_n}'
