"""
pgmol - Schoenflies point-group detection for molecules given as XYZ data.

Only the Python standard library and numpy are used.

Public API
----------
detect_point_group(xyz_input, tol=1e-2) -> str
point_group  (alias of detect_point_group)
main(argv=None) -> int   (CLI entry point; wire it up as console script ``pgmol``)

For ``python -m pgmol`` add a ``__main__.py`` next to this file containing::

    from pgmol import main
    raise SystemExit(main())

Algorithm overview
------------------
1.  Parse the XYZ data (element labels + Cartesian coordinates).
2.  Translate to the unweighted centroid.  Every symmetry operation permutes
    atoms of the same element, so the centroid is a fixed point of the whole
    point group.  No masses or inertia degeneracies are involved, so
    accidental spherical tops cannot be mistaken for cubic molecules.
3.  Detect single atoms (Kh) and linear molecules (Dinfh / Cinfv).
4.  Build an exhaustive set of candidate rotation axes from the geometry:
      * atom position vectors (axes passing through atoms),
      * midpoints of equivalent atom pairs (C2 axes between atoms),
      * normals of triangles drawn in one "shell" of equivalent atoms
        (any Cn axis, n >= 3, not passing through atoms),
      * the principal axes of the coordinate covariance (plane normal of
        planar molecules).
    Each candidate is verified explicitly; its maximal order is found by
    testing only divisors of the gcd of off-axis atom counts per element.
5.  Candidate mirror normals are the difference vectors of equivalent atom
    pairs plus the axis/plane candidates; each is verified explicitly.
6.  Classify with the standard Schoenflies decision tree (polyhedral groups
    when >= 2 distinct axes of order >= 3 exist; otherwise axial groups).

All decisions are based exclusively on element symbols and coordinates, so
the result is invariant to rotations, translations and atom permutations.
"""

from __future__ import annotations

import math
import os
import pathlib
import re
import sys
from functools import reduce
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["detect_point_group", "point_group", "main"]
__version__ = "1.0.0"


# --------------------------------------------------------------------------- #
# XYZ parsing
# --------------------------------------------------------------------------- #
def _read_text(xyz_input: Union[str, os.PathLike, bytes]) -> str:
    if isinstance(xyz_input, bytes):
        xyz_input = xyz_input.decode("utf-8", errors="replace")
    if isinstance(xyz_input, os.PathLike):
        return pathlib.Path(xyz_input).read_text()
    if not isinstance(xyz_input, str):
        raise TypeError("xyz_input must be a str or a pathlib.Path")
    s = xyz_input
    if "\n" not in s.strip():
        candidate = s.strip()
        try:
            p = pathlib.Path(candidate)
            if candidate and p.is_file():
                return p.read_text()
        except (OSError, ValueError):
            pass
        raise FileNotFoundError(f"XYZ file not found: {candidate!r}")
    return s


def _normalize_symbol(token: str) -> str:
    m = re.match(r"[A-Za-z]+", token)
    if m:
        letters = m.group(0)
        return letters[0].upper() + letters[1:].lower()
    try:
        return str(int(round(float(token))))  # atomic number given
    except ValueError:
        return token


def _to_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _parse_xyz(text: str) -> Tuple[List[str], np.ndarray]:
    lines = text.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        raise ValueError("Empty XYZ input")
    try:
        natoms = int(lines[i].split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError("First line of XYZ data must contain the atom count") from exc
    if natoms < 0:
        raise ValueError("Negative atom count")
    body = lines[i + 2:]  # skip the comment line entirely
    symbols: List[str] = []
    coords: List[List[float]] = []
    for ln in body:
        if len(symbols) == natoms:
            break
        parts = ln.split()
        if not parts:
            continue
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ atom line: {ln!r}")
        try:
            xyz = [_to_float(p) for p in parts[1:4]]
        except ValueError as exc:
            raise ValueError(f"Malformed coordinates in line: {ln!r}") from exc
        symbols.append(_normalize_symbol(parts[0]))
        coords.append(xyz)
    if len(symbols) != natoms:
        raise ValueError(f"Expected {natoms} atoms, found {len(symbols)}")
    if natoms == 0:
        raise ValueError("XYZ data contains no atoms")
    return symbols, np.asarray(coords, dtype=float)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def _reflection(normal: np.ndarray) -> np.ndarray:
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    return np.eye(3) - 2.0 * np.outer(n, n)


def _improper(axis: np.ndarray, order: int) -> np.ndarray:
    return _rotation(axis, 2.0 * math.pi / order) @ _reflection(axis)


def _divisors_desc(n: int) -> List[int]:
    ds = set()
    for k in range(1, int(math.isqrt(n)) + 1):
        if n % k == 0:
            ds.add(k)
            ds.add(n // k)
    return sorted(ds, reverse=True)


def _unique_directions(vectors: np.ndarray, cos_tol: float) -> np.ndarray:
    """Normalize, sign-canonicalize and deduplicate direction vectors."""
    if vectors.size == 0:
        return np.zeros((0, 3))
    V = np.asarray(vectors, dtype=float).reshape(-1, 3)
    norms = np.linalg.norm(V, axis=1)
    V = V[norms > 1e-10]
    if len(V) == 0:
        return np.zeros((0, 3))
    V = V / np.linalg.norm(V, axis=1)[:, None]
    idx = np.argmax(np.abs(V), axis=1)
    signs = np.sign(V[np.arange(len(V)), idx])
    signs[signs == 0] = 1.0
    V = V * signs[:, None]
    # cheap first pass: hash on rounded components
    _, first = np.unique(np.round(V, 5), axis=0, return_index=True)
    V = V[np.sort(first)]
    # exact greedy pass (|dot| because directions are unsigned)
    U = np.empty_like(V)
    k = 0
    for v in V:
        if k and np.max(np.abs(U[:k] @ v)) > cos_tol:
            continue
        U[k] = v
        k += 1
    return U[:k].copy()


# --------------------------------------------------------------------------- #
# Symmetry analysis
# --------------------------------------------------------------------------- #
class _Molecule:
    def __init__(self, symbols: Sequence[str], coords: np.ndarray, tol: float):
        self.tol = float(tol)
        self.X = coords - coords.mean(axis=0)
        labels = sorted(set(symbols))
        self.groups: List[np.ndarray] = []
        for lab in labels:
            idx = [i for i, s in enumerate(symbols) if s == lab]
            self.groups.append(self.X[idx])
        self._gsq = [np.einsum("ij,ij->i", G, G) for G in self.groups]
        self.tol2 = self.tol * self.tol

    # ---- operation test --------------------------------------------------- #
    def is_symmetry(self, R: np.ndarray) -> bool:
        for G, gsq in zip(self.groups, self._gsq):
            Y = G @ R.T
            d2 = np.einsum("ij,ij->i", Y, Y)[:, None] + gsq[None, :] - 2.0 * (Y @ G.T)
            if np.max(np.min(d2, axis=1)) > self.tol2:
                return False
        return True

    # ---- shells of potentially equivalent atoms --------------------------- #
    def shells(self) -> List[np.ndarray]:
        out = []
        for G in self.groups:
            r = np.linalg.norm(G, axis=1)
            order = np.argsort(r)
            start = 0
            for k in range(1, len(order) + 1):
                if k == len(order) or r[order[k]] - r[order[k - 1]] > self.tol:
                    out.append(G[order[start:k]])
                    start = k
        return out

    # ---- maximal rotation order about an axis ----------------------------- #
    def axis_order(self, axis: np.ndarray) -> int:
        counts = []
        for G in self.groups:
            h = G @ axis
            perp = np.linalg.norm(G - np.outer(h, axis), axis=1)
            c = int(np.sum(perp > self.tol))
            if c:
                counts.append(c)
        if not counts:
            return 1
        g = reduce(math.gcd, counts)
        for d in _divisors_desc(g):
            if d < 2:
                break
            if self.is_symmetry(_rotation(axis, 2.0 * math.pi / d)):
                return d
        return 1

    # ---- candidate generation --------------------------------------------- #
    def candidates(self) -> Tuple[np.ndarray, np.ndarray]:
        tol = self.tol
        axis_c: List[np.ndarray] = []
        mirror_c: List[np.ndarray] = []

        # principal axes of the (unweighted) coordinate covariance
        _, evecs = np.linalg.eigh(self.X.T @ self.X)
        axis_c.append(evecs.T)
        mirror_c.append(evecs.T)

        # atom position vectors
        r = np.linalg.norm(self.X, axis=1)
        axis_c.append(self.X[r > tol])

        shells = self.shells()
        for S in shells:
            m = len(S)
            if m < 2:
                continue
            i, j = np.triu_indices(m, k=1)
            mid = 0.5 * (S[i] + S[j])
            mid = mid[np.linalg.norm(mid, axis=1) > tol]
            axis_c.append(mid)
            diff = S[i] - S[j]
            diff = diff[np.linalg.norm(diff, axis=1) > tol]
            mirror_c.append(diff)

        # triangle normals from the smallest off-centre shell with >= 3 atoms
        good = [S for S in shells if len(S) >= 3 and np.linalg.norm(S[0]) > tol]
        if good:
            S = min(good, key=len)
            a = S[0]
            B = S[1:] - a
            if len(B) >= 2:
                i, j = np.triu_indices(len(B), k=1)
                nrm = np.cross(B[i], B[j])
                axis_c.append(nrm)

        axes = _unique_directions(np.vstack(axis_c), 1.0 - 1e-7)
        mirrors = np.vstack(mirror_c + [axes]) if len(axes) else np.vstack(mirror_c)
        mirrors = _unique_directions(mirrors, 1.0 - 1e-7)
        return axes, mirrors


def _dedup_verified(items: List[Tuple[np.ndarray, int]], cos_tol: float = 0.999):
    items = sorted(items, key=lambda t: -t[1])
    out: List[Tuple[np.ndarray, int]] = []
    for v, o in items:
        if any(abs(float(np.dot(v, w))) > cos_tol for w, _ in out):
            continue
        out.append((v, o))
    return out


def _classify(symbols: Sequence[str], coords: np.ndarray, tol: float) -> str:
    mol = _Molecule(symbols, coords, tol)
    X = mol.X
    n_atoms = len(X)
    radii = np.linalg.norm(X, axis=1)

    if n_atoms == 1 or float(np.max(radii)) < tol:
        return "Kh"

    has_inversion = mol.is_symmetry(-np.eye(3))

    # ---- linear molecules ---------------------------------------------- #
    _, evecs = np.linalg.eigh(X.T @ X)
    main_dir = evecs[:, -1]
    perp = X - np.outer(X @ main_dir, main_dir)
    if float(np.max(np.linalg.norm(perp, axis=1))) < tol:
        return "Dinfh" if has_inversion else "Cinfv"

    # ---- symmetry elements --------------------------------------------- #
    axis_cands, mirror_cands = mol.candidates()

    axes: List[Tuple[np.ndarray, int]] = []
    for a in axis_cands:
        o = mol.axis_order(a)
        if o >= 2:
            axes.append((a, o))
    axes = _dedup_verified(axes)

    mirrors: List[np.ndarray] = []
    for nvec in mirror_cands:
        if mol.is_symmetry(_reflection(nvec)):
            mirrors.append(nvec)
    mirrors = [v for v, _ in _dedup_verified([(m, 1) for m in mirrors])]

    # ---- polyhedral groups --------------------------------------------- #
    high = [(a, o) for a, o in axes if o >= 3]
    if len(high) >= 2:
        orders = [o for _, o in high]
        if any(o % 5 == 0 for o in orders):
            return "Ih" if has_inversion else "I"
        if any(o % 4 == 0 for o in orders):
            return "Oh" if has_inversion else "O"
        if has_inversion:
            return "Th"
        return "Td" if mirrors else "T"

    # ---- groups without proper rotation axes --------------------------- #
    if not axes:
        if mirrors:
            return "Cs"
        if has_inversion:
            return "Ci"
        return "C1"

    # ---- axial groups -------------------------------------------------- #
    n = max(o for _, o in axes)
    principals = [a for a, o in axes if o == n]
    z = principals[0]
    has_s2n = False
    for a in principals:
        if mol.is_symmetry(_improper(a, 2 * n)):
            z = a
            has_s2n = True
            break

    perp_tol = 0.05
    perp_c2 = [a for a, o in axes if o >= 2 and abs(float(np.dot(a, z))) < perp_tol]
    sigma_h = mol.is_symmetry(_reflection(z))
    sigma_v = any(abs(float(np.dot(m, z))) < perp_tol for m in mirrors)

    if perp_c2:
        if sigma_h:
            return f"D{n}h"
        if sigma_v:
            return f"D{n}d"
        return f"D{n}"
    if sigma_h:
        return f"C{n}h"
    if sigma_v:
        return f"C{n}v"
    if has_s2n:
        return f"S{2 * n}"
    return f"C{n}"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
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
    tol = float(tol)
    if not (tol > 0.0) or not math.isfinite(tol):
        raise ValueError("tol must be a positive finite number")
    text = _read_text(xyz_input)
    symbols, coords = _parse_xyz(text)
    if not np.all(np.isfinite(coords)):
        raise ValueError("Non-finite coordinates in XYZ data")
    return _classify(symbols, coords, tol)


point_group = detect_point_group


# --------------------------------------------------------------------------- #
# Command-line interface
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="pgmol",
        description="Print the Schoenflies point group of a molecule in XYZ format.",
    )
    parser.add_argument("xyz", help="path to an .xyz file ('-' reads from stdin)")
    parser.add_argument(
        "--tol", type=float, default=1e-2,
        help="Cartesian tolerance in angstroms (default: 0.01)",
    )
    args = parser.parse_args(argv)
    try:
        if args.xyz == "-":
            data: Union[str, pathlib.Path] = sys.stdin.read()
        else:
            data = pathlib.Path(args.xyz)
            if not data.is_file():
                print(f"pgmol: error: file not found: {args.xyz}", file=sys.stderr)
                return 2
        print(detect_point_group(data, tol=args.tol))
    except (ValueError, OSError, TypeError) as exc:
        print(f"pgmol: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
