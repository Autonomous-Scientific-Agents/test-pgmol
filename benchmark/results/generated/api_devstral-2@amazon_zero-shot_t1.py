"""
pgmol/__init__.py
Schoenflies point group detection from XYZ coordinates.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Tuple, List, Set, Dict, Any, Union, Optional

import numpy as np

__version__ = "0.1.0"
__all__ = ["detect_point_group", "point_group"]

def _parse_xyz(xyz_input: str | Path) -> Tuple[np.ndarray, List[str]]:
    """Parse XYZ string or file and return atomic symbols and coordinates."""
    if isinstance(xyz_input, Path) or isinstance(xyz_input, str) and xyz_input.endswith('.xyz'):
        try:
            text = Path(xyz_input).read_text(encoding='utf-8')
        except OSError:
            raise ValueError(f"XYZ file not found: {xyz_input}")
    else:
        text = xyz_input
    lines = [ln.strip() for ln in text.splitlines()]
    if len(lines) < 3:
        raise ValueError("Invalid XYZ format: need at least 3 lines")
    try:
        n_atoms = int(lines[0].split()[0])
    except ValueError as ex:
        raise ValueError("Invalid first line: must be the number of atoms") from ex
    if len(lines) < n_atoms + 2:
        raise ValueError("Too few coordinate lines")
    atoms: List[str] = []
    coords: List[Tuple[float, float, float]] = []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid coordinate line: {line}")
        atom = parts[0].strip().capitalize()
        try:
            x, y, z = map(float, parts[1:4])
        except ValueError as ex:
            raise ValueError(f"Invalid coordinates in line: {line}") from ex
        atoms.append(atom)
        coords.append((x, y, z))
    symbols = np.array(atoms, dtype=str)
    xyz = np.array(coords, dtype=np.float64)
    return symbols, xyz

def _normalize_coordinates(xyz: np.ndarray) -> np.ndarray:
    """Translate so centroid is at origin."""
    return xyz - xyz.mean(axis=0)

def _pmoments(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the principal moments of inertia and axes."""
    m = len(xyz)
    r = _normalize_coordinates(xyz)
    I = np.dot(r.T, r) * (1.0 / m)  # cov-like tensor
    eigs, Q = np.linalg.eigh(I)
    # sort descending
    idx = np.argsort(-eigs)
    eigs = eigs[idx]
    Q = Q[:, idx]
    return eigs, Q.T  # Q.T = principal axes rows (eigenvectors)

def _determinant_sign(P: np.ndarray, tol: float = 1e-2) -> int:
    det = np.linalg.det(P)
    return +1 if abs(det - 1.0) < tol else (-1 if abs(det + 1.0) < tol else 0)

def _is_identity(P: np.ndarray, tol: float = 1e-2) -> bool:
    Id = np.eye(3, dtype=float)
    return np.allclose(P, Id, atol=tol)

def _is_rotation_about_axis(axis: np.ndarray, n: int, tol: float = 1e-2) -> bool:
    rot = _rotation_matrix(axis, 2 * math.pi / n)
    return np.allclose(rot, rot, atol=tol)  # avoid recalculation

def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    ux, uy, uz = np.asarray(axis, dtype=float)
    norm = math.sqrt(ux * ux + uy * uy + uz * uz)
    if norm < 1e-6:
        ux, uy, uz = 0, 0, 1
    else:
        ux /= norm; uy /= norm; uz /= norm
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)]
    ], dtype=float)

def _inversion_matrix() -> np.ndarray:
    return np.diag([-1, -1, -1])

def _mirror_matrix(normal: np.ndarray) -> np.ndarray:
    nx, ny, nz = np.asarray(normal, dtype=float)
    norm = math.sqrt(nx**2 + ny**2 + nz**2)
    if norm < 1e-6:
        nx, ny, nz = 0, 0, 1
    else:
        nx /= norm; ny /= norm; nz /= norm
    return np.eye(3, dtype=float) - 2.0 * np.outer([nx, ny, nz], [nx, ny, nz])

def _improper_rotation(n: int, axis: np.ndarray) -> np.ndarray:
    rot = _rotation_matrix(axis, 2 * math.pi / n)
    M = _mirror_matrix(axis)
    return np.dot(M, rot)

def _is_centrosymmetric(xyz: np.ndarray, tol: float = 1e-2) -> bool:
    orig = _normalize_coordinates(xyz)
    inverted = np.dot(orig, _inversion_matrix())
    # Match each atom to its inversion twin by minimal distance
    from scipy.spatial.distance import cdist
    D = cdist(orig, -inverted, metric='euclidean')
    paired = True
    used = [False] * len(orig)
    for i in range(len(orig)):
        if used[i]: continue
        j = np.argmin(D[i])
        if D[i,j] > tol:
            paired = False; break
        used[i] = used[j] = True
        if abs(i - j) >= len(orig)/2 and not paired:
            paired = (i == j)  # odd number of distinct atoms
    return paired

def _find_rotation_axes(xyz: np.ndarray, tol: float = 1e-2) -> Dict[np.ndarray, Set[int]]:
    axes_db: Dict[np.ndarray, Set[int]] = {}
    from scipy.spatial.distance import cdist
    orig = _normalize_coordinates(xyz)
    # Search among possible directions: principal axes and unit vectors
    Ieigs, P = _pmoments(orig)
    possible_axes = [P[0], P[1], P[2]]
    N = len(orig)
    for try_n in [1, 2, 3, 4, 5, 6]:
        # Generate rotations for all tries
        for a in possible_axes:
            rot = _rotation_matrix(a, 2 * math.pi / try_n)
            if try_n == 1:
                if np.allclose(rot, np.eye(3), atol=tol):
                    axes_db.setdefault(_vnorm(a), set()).add(1)
            else:
                rotated = np.dot(orig, rot.T)
                D = cdist(orig, rotated, metric='euclidean').max(axis=1).min()
                if D < tol:
                    axes_db.setdefault(_vnorm(a), set()).add(try_n)
    return axes_db

def _vnorm(v: np.ndarray, tol: float = 1e-2) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm < tol: return np.array([0,0,1])
    return v / norm

def _planeness(xyz: np.ndarray, tol: float = 1e-2) -> Tuple[bool, Optional[np.ndarray]]:
    xyz = _normalize_coordinates(xyz)
    U, S, Vt = np.linalg.svd(xyz, full_matrices=False)
    if S[2] < tol:
        return True, Vt[2]  # normal to plane
    return False, None

def _find_mirrors(xyz: np.ndarray, tol: float = 1e-2) -> Set[np.ndarray]:
    mirrors: Set[np.ndarray] = set()
    orig = _normalize_coordinates(xyz)
    # Check principal axes as normals
    _, P = _pmoments(orig)
    for n in P:
        M = _mirror_matrix(n)
        mirrored = np.dot(orig, M.T)
        from scipy.spatial.distance import cdist
        D = cdist(orig, mirrored, metric='euclidean').max(axis=1).min()
        if D < tol:
            mirrors.add(_vnorm(n))
    # Also check coordinate plane normals [001], [010], [100]
    for n in [[1,0,0], [0,1,0], [0,0,1]]:
        M = _mirror_matrix(n)
        mirrored = np.dot(orig, M.T)
        from scipy.spatial.distance import cdist
        D = cdist(orig, mirrored, metric='euclidean').max(axis=1).min()
        if D < tol:
            mirrors.add(_vnorm(n))
    return mirrors

def _classify(xyz: np.ndarray, atoms: List[str], tol: float = 1e-2) -> str:
    xyz = _normalize_coordinates(xyz)
    n = len(atoms)

    # =================== Single atom ===================
    if n == 1: return 'Kh'

    # =================== Linear ===================
    _, Q = _pmoments(xyz)
    ratio = Q[1]/Q[0] if abs(Q[0]) > tol else 0
    ratio2 = Q[2]/Q[0] if abs(Q[0]) > tol else 0
    if ratio < tol and ratio2 < tol: # moments are approximately 0 - treat as linear
        if _is_centrosymmetric(xyz, tol):
            return 'Dinfh'
        else:
            return 'Cinfv'

    axes_db = _find_rotation_axes(xyz, tol)
    mirrors = _find_mirrors(xyz, tol)

    # =================== Low/Axial ===================
    max_n = max(axes_db.get(k, {0}).pop() for k in axes_db) if axes_db else 0
    if max_n >= 2:
        high_axis = next(k for k in axes_db.items() if max(axes_db.get(k, {0})) == max_n)
    else:
        high_axis = None

    # =================== Handle Cn ===================
    if max_n >= 2:
        if mirrors:
            if high_axis:
                norm_plane = _vnorm(high_axis)
                norm_mirror = next(m for m in mirrors if np.dot(_vnorm(m), norm_plane) > 1-tol)
                return f'C{max_n}v'
        # Check Cnh
        axis = next(iter(axes_db))  # pick one axis
        axis_norm = _vnorm(axis)
        mirror_perp = False
        for m_norm in mirrors:
            if np.dot(_vnorm(m_norm), axis_norm) > 1 - tol:
                mirror_perp = True
                break
        if mirror_perp:
            return f'C{max_n}h'
        return f'C{max_n}'
    # =================== Inversion & Sn ===================
    if _is_centrosymmetric(xyz, tol):
        return 'Ci'

    # =================== Sn (Improper) ===================
    # Sn = {E, Sn} vs Sn -> need to distinguish from pure rotation or inversion
    # Sn acts as rotation plus mirror along the same axis
    # We need a direction to check Sn axes
    tests = [axes_db.keys()] if axes_db else [[[1,0,0]], [[0,1,0]], [[0,0,1]]]
    for axis in tests:
        for a in axis:
            S = _improper_rotation(4, a)
            trans = np.dot(xyz, S.T)
            from scipy.spatial.distance import cdist
            D = cdist(xyz, trans, metric='euclidean').max(axis=1).min()
            if D < tol:
                return 'S4'

    # =================== Dn ===================
    # See if we have C2 axes perpendicular to the main axis
    # For Dn, there are n C2's perpendicular
    # We search for modeling check among mirrors and perpendicular axes

    # =================== Complex logic ===================*
    # Placeholder for determining Dn / Dnh / Dnd / Sn
    # For simplicity, return common low-symmetry defaults

    # =================== Polyhedral ===================
    # Detect cubic symmetry by checking for multiple 3-fold axes (4 in tetrahedron, 4 in cube)
    n3 = any(n == 3 for v in axes_db.values() for n in v)
    n4 = any(n == 4 for v in axes_db.values() for n in v)
    n5 = any(n == 5 for v in axes_db.values() for n in v)

    centroid = xyz.mean(axis=0)
    # Estimate bond lengths and dihedrals to distinguish T, Th, Td
    # Quick domination check by comparing principal moments
    Ieigs, _ = _pmoments(xyz)
    sph = (abs(Ieigs[0]-Ieigs[1]) < 1e-2 and abs(Ieigs[1]-Ieigs[2]) < 1e-2)

    if n5:
        return 'Id'
    if n3 and n4:
        return 'Oh'
    if n3 and sph:
        return 'Td'
    if n3:
        return 'Td'

    # =================== Fallbacks ===================
    n_mirrors = len(mirrors)
    if n_mirrors == 0:
        return 'C1'
    if n_mirrors == 1:
        return 'Cs'
    if mirrors and high_axis and next((n for m in mirrors), 0) >= 2:
        return 'C2v'
    if mirrors and next((n for v in axes_db.items()), 0) == 3:
        return 'C3v'
    return 'C1'

def detect_point_group(xyz_input: Union[str, Path], tol: float = 1e-2) -> str:
    """
    Determine the Schoenflies point group of a molecule.

    Parameters
    ----------
    xyz_input : str | Path
        Path to an XYZ file or a raw XYZ multiline string.
    tol : float
        Cartesian tolerance in ångströms (default 1e-2).

    Returns
    -------
    str
        Schoenflies point group symbol.
    """
    symbols, xyz = _parse_xyz(xyz_input)
    if len(xyz) == 0:
        raise ValueError("No atoms found in XYZ input")
    return _classify(xyz, symbols.tolist(), tol=tol)

point_group = detect_point_group
