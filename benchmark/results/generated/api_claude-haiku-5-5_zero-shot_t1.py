"""pgmol: Schoenflies point-group detection from XYZ coordinates.

Only the Python standard library and numpy are used.  Classification depends
solely on element labels and Cartesian coordinates.  The method:

1. Centre the molecule on its unweighted centroid (fixed by every symmetry
   operation that permutes like-labelled atoms).
2. Detect linear molecules directly (Dinfh / Cinfv).
3. Otherwise enumerate *all* orthogonal operations that map the molecule onto
   itself.  Two non-collinear reference atoms i, j are fixed; every candidate
   image pair (a, b) with matching labels and norms yields exactly one proper
   and one improper orthogonal map.  Each map is orthogonalised (polar
   decomposition) and verified against every atom within ``tol``.
4. Classify the resulting finite group from element orders, the number of
   proper rotations of each order, and the presence of inversion, reflections
   and rotation-reflection elements.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np

__all__ = ["detect_point_group", "point_group", "main"]

# Tolerance for dimensionless matrix comparisons (orders, identities, axes).
_MAT_ATOL = 0.1
_I3 = np.eye(3)
_MAX_ORDER = 120


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def _read_text(xyz_input: Union[str, pathlib.Path]) -> str:
    if isinstance(xyz_input, pathlib.Path):
        return xyz_input.read_text()
    text = str(xyz_input)
    if "\n" not in text and text.strip():
        candidate = pathlib.Path(text.strip())
        try:
            if candidate.is_file():
                return candidate.read_text()
        except (OSError, ValueError):
            pass
    return text


def _normalise_symbol(sym: str) -> str:
    sym = sym.strip()
    if sym.isdigit():
        return sym
    return sym[0].upper() + sym[1:].lower()


def _parse_xyz(text: str) -> Tuple[List[str], np.ndarray]:
    lines = text.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        raise ValueError("empty XYZ input")
    try:
        n = int(lines[i].split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError("first XYZ line must contain the number of atoms") from exc
    if n < 1:
        raise ValueError("XYZ input must contain at least one atom")

    symbols: List[str] = []
    coords: List[List[float]] = []
    k = i + 2  # line i+1 is the comment line and is ignored
    while len(symbols) < n and k < len(lines):
        parts = lines[k].split()
        k += 1
        if not parts:
            continue
        if len(parts) < 4:
            raise ValueError(f"malformed atom line: {lines[k - 1]!r}")
        symbols.append(_normalise_symbol(parts[0]))
        coords.append([float(x) for x in parts[1:4]])
    if len(symbols) != n:
        raise ValueError(f"expected {n} atoms, found {len(symbols)}")
    return symbols, np.asarray(coords, dtype=float)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _maps_onto(O: np.ndarray, r: np.ndarray, types: np.ndarray, tol: float) -> bool:
    """True if the orthogonal map O sends every atom onto a like-labelled atom."""
    p = r @ O.T
    d = np.linalg.norm(p[:, None, :] - r[None, :, :], axis=2)
    same = types[:, None] == types[None, :]
    d = np.where(same, d, np.inf)
    return bool(np.all(d.min(axis=1) < tol))


def _symmetry_operations(r: np.ndarray, types: np.ndarray, tol: float) -> List[np.ndarray]:
    """Return every orthogonal matrix that leaves the (centred) molecule invariant."""
    n = len(r)
    norms = np.linalg.norm(r, axis=1)
    i = int(np.argmax(norms))
    cross_norms = np.linalg.norm(np.cross(r[i], r), axis=1)
    j = int(np.argmax(cross_norms))
    ni = np.cross(r[i], r[j])
    F = np.column_stack([r[i], r[j], ni])
    Finv = np.linalg.inv(F)

    ops: List[np.ndarray] = []
    seen = set()
    for a in range(n):
        if types[a] != types[i] or abs(norms[a] - norms[i]) > tol:
            continue
        for b in range(n):
            if types[b] != types[j] or abs(norms[b] - norms[j]) > tol:
                continue
            na = np.cross(r[a], r[b])
            for s in (1.0, -1.0):
                T = np.column_stack([r[a], r[b], s * na])
                U, _, Vt = np.linalg.svd(T @ Finv)
                O = U @ Vt
                key = tuple(np.round(O, 4).ravel())
                if key in seen:
                    continue
                seen.add(key)
                if _maps_onto(O, r, types, tol):
                    ops.append(O)
    return ops


# ---------------------------------------------------------------------------
# Group classification
# ---------------------------------------------------------------------------
def _order(M: np.ndarray) -> int:
    P = M.copy()
    for k in range(1, _MAX_ORDER + 1):
        if np.max(np.abs(P - _I3)) < _MAT_ATOL:
            return k
        P = P @ M
    return 0


def _rotation_axis(R: np.ndarray) -> np.ndarray:
    _, _, vt = np.linalg.svd(R - _I3)
    return vt[-1]


def _classify(ops: Sequence[np.ndarray]) -> str:
    entries = []
    for M in ops:
        d = float(np.linalg.det(M))
        entries.append((M, d > 0, _order(M)))

    proper = [(M, o) for M, p, o in entries if p]
    improper = [(M, o) for M, p, o in entries if not p]
    has_inv = any(np.max(np.abs(M + _I3)) < _MAT_ATOL for M, _ in improper)
    has_improper = len(improper) > 0

    count3 = sum(1 for _, o in proper if o == 3)
    count4 = sum(1 for _, o in proper if o == 4)
    count5 = sum(1 for _, o in proper if o == 5)

    # --- polyhedral groups -------------------------------------------------
    if count3 >= 8 and count5 > 0:
        return "Ih" if has_improper else "I"
    if count3 >= 8:
        if count4 > 0:
            return "Oh" if has_improper else "O"
        if has_improper:
            return "Th" if has_inv else "Td"
        return "T"

    # --- axial and low-symmetry groups ------------------------------------
    nmax = max((o for _, o in proper), default=1)
    if nmax >= 3:
        axis_elem = next(M for M, o in proper if o == nmax)
        k = _rotation_axis(axis_elem)
        return _classify_high(nmax, k, proper, improper, has_inv)
    return _classify_low(len(ops), proper, improper, has_inv)


def _classify_high(n: int, k: np.ndarray, proper, improper, has_inv: bool) -> str:
    c2perp = any(
        o == 2 and np.linalg.norm(M @ k + k) < _MAT_ATOL for M, o in proper
    )
    sh = sv = False
    s_type = False
    for M, o in improper:
        if np.max(np.abs(M + _I3)) < _MAT_ATOL:
            continue  # inversion, handled by has_inv
        if o == 2:
            if np.linalg.norm(M @ k + k) < _MAT_ATOL:
                sh = True
            elif np.linalg.norm(M @ k - k) < _MAT_ATOL:
                sv = True
        else:
            s_type = True

    if not improper:
        return f"D{n}" if c2perp else f"C{n}"

    if c2perp:
        if sh and sv:
            return f"D{n}h"
        if sh:
            return f"D{n}h"
        return f"D{n}d"

    if sh:
        return f"C{n}h"
    if sv:
        return f"C{n}v"
    # Only rotation-reflection (S_2m) elements remain.
    return f"S{2 * n}"


def _classify_low(h: int, proper, improper, has_inv: bool) -> str:
    nC2 = sum(1 for _, o in proper if o == 2)
    refl = sum(1 for M, o in improper
               if o == 2 and np.max(np.abs(M + _I3)) >= _MAT_ATOL)
    nS = sum(1 for M, o in improper
             if o != 2)

    if h == 1:
        return "C1"
    if h == 2:
        if nC2:
            return "C2"
        if has_inv:
            return "Ci"
        return "Cs"
    if h == 4:
        if nC2 == 3:
            return "D2"
        if has_inv:
            return "C2h"
        if refl == 2:
            return "C2v"
        if nS > 0:
            return "S4"
        return "C2v"
    if h == 8:
        return "D2h" if has_inv else "D2d"
    return "C1"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def detect_point_group(xyz_input: Union[str, pathlib.Path], tol: float = 1e-2) -> str:
    """
    Determine the Schoenflies point group of a molecule.

    Parameters
    ----------
    xyz_input : str | pathlib.Path
        Path to an XYZ file OR a raw XYZ-formatted multiline string.
    tol : float, optional
        Cartesian distance tolerance in angstroms (default: 1e-2).

    Returns
    -------
    str
        Exact ASCII Schoenflies symbol, e.g. 'C1', 'Cs', 'Ci', 'C2', 'C2v',
        'C3h', 'D2h', 'D3d', 'S4', 'T', 'Td', 'Th', 'O', 'Oh', 'I', 'Ih',
        'Cinfv', 'Dinfh', or 'Kh'.
    """
    symbols, coords = _parse_xyz(_read_text(xyz_input))
    tol = float(tol)
    if len(symbols) == 1:
        return "Kh"

    types = np.array(symbols)
    r = coords - coords.mean(axis=0)
    if np.linalg.norm(r, axis=1).max() < tol:
        return "Kh"

    # Linear molecules
    _, _, vt = np.linalg.svd(r, full_matrices=False)
    u = vt[0]
    perp = r - np.outer(r @ u, u)
    if np.linalg.norm(perp, axis=1).max() < tol:
        return "Dinfh" if _maps_onto(-_I3, r, types, tol) else "Cinfv"

    ops = _symmetry_operations(r, types, tol)
    if not ops:
        return "C1"
    return _classify(ops)


point_group = detect_point_group


def main(argv: Iterable[str] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] in ("-h", "--help"):
        print("usage: pgmol <file.xyz>", file=sys.stderr)
        return 0 if args and args[0] in ("-h", "--help") else 2
    print(detect_point_group(pathlib.Path(args[0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
