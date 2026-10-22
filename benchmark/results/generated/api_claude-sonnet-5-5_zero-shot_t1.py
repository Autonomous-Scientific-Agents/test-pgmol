"""pgmol: Schoenflies point group detection from XYZ coordinates (numpy only)."""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

__all__ = ["detect_point_group", "point_group", "main"]


def _parse(xyz_input):
    if isinstance(xyz_input, pathlib.Path):
        text = xyz_input.read_text()
    else:
        s = str(xyz_input)
        text = s
        if "\n" not in s.strip():
            p = pathlib.Path(s.strip())
            if p.exists():
                text = p.read_text()
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    n = int(lines[0].split()[0])
    els, pts = [], []
    for ln in lines[2:2 + n]:
        t = ln.split()
        els.append(t[0].strip().capitalize())
        pts.append([float(t[1]), float(t[2]), float(t[3])])
    return els, np.array(pts, dtype=float).reshape(-1, 3)


def _rot(axis, theta):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def _mirror(n):
    n = n / np.linalg.norm(n)
    return np.eye(3) - 2 * np.outer(n, n)


def _canon(v):
    v = v / np.linalg.norm(v)
    for c in v:
        if abs(c) > 1e-6:
            if c < 0:
                v = -v
            break
    return v


def _unique(vecs, dec=3):
    seen = {}
    for v in vecs:
        nv = np.linalg.norm(v)
        if nv < 1e-6:
            continue
        c = _canon(v)
        key = tuple(np.round(c, dec) + 0.0)
        if key not in seen:
            seen[key] = c
    return list(seen.values())


def detect_point_group(xyz_input, tol: float = 1e-2) -> str:
    els, X = _parse(xyz_input)
    N = len(els)
    if N == 1:
        return "Kh"
    X = X - X.mean(axis=0)
    elarr = np.array(els)
    mism = elarr[:, None] != elarr[None, :]

    def sym(R):
        Y = X @ R.T
        D = np.linalg.norm(Y[:, None, :] - X[None, :, :], axis=2)
        D[mism] = np.inf
        return D.min(1).max() < tol and D.min(0).max() < tol

    has_inv = sym(-np.eye(3))

    # linear
    sv = np.linalg.svd(X, compute_uv=False)
    if sv[1] < tol:
        return "Dinfh" if has_inv else "Cinfv"

    norms = np.linalg.norm(X, axis=1)
    # inertia eigenvectors
    I = np.zeros((3, 3))
    for r in X:
        I += np.dot(r, r) * np.eye(3) - np.outer(r, r)
    _, ev = np.linalg.eigh(I)
    inertia = [ev[:, k] for k in range(3)]

    # ---- candidate rotation axes
    cand = list(inertia)
    for i in range(N):
        if norms[i] > 1e-6:
            cand.append(X[i])
    for i in range(N):
        for j in range(i + 1, N):
            if elarr[i] == elarr[j]:
                cand.append(X[i] + X[j])
    for i in range(N):
        if norms[i] < 1e-6:
            continue
        grp = np.where((elarr == elarr[i]) & (np.abs(norms - norms[i]) < 3 * tol))[0]
        if len(grp) > 12 and i not in grp[:6]:
            continue
        for j in grp:
            if j == i:
                continue
            d1 = X[i] - X[j]
            ks = grp[grp != j]
            if len(ks) == 0:
                continue
            cr = np.cross(d1[None, :], X[j][None, :] - X[ks])
            cand.extend(list(cr))
    cand = _unique(cand)

    maxn = min(max(N, 2), 12)
    axes = []
    for a in cand:
        for n in range(maxn, 1, -1):
            if sym(_rot(a, 2 * math.pi / n)):
                axes.append((a, n))
                break
    # dedupe validated axes
    axes.sort(key=lambda t: -t[1])
    ded = []
    for a, n in axes:
        if not any(abs(np.dot(a, b)) > 0.999 for b, _ in ded):
            ded.append((a, n))
    axes = ded

    # ---- mirror candidates
    mc = list(inertia) + [a for a, _ in axes]
    for i in range(N):
        if norms[i] > 1e-6:
            mc.append(X[i])
        for j in range(i + 1, N):
            if elarr[i] == elarr[j]:
                mc.append(X[i] - X[j])
            mc.append(np.cross(X[i], X[j]))
    mc = _unique(mc)
    mirrors = [m for m in mc if sym(_mirror(m))]
    has_mirror = bool(mirrors)

    hi = [(a, n) for a, n in axes if n >= 3]
    if len(hi) >= 2:
        mx = max(n for _, n in hi)
        if mx >= 5:
            return "Ih" if has_inv else "I"
        if mx == 4:
            return "Oh" if has_inv else "O"
        if has_inv:
            return "Th"
        return "Td" if has_mirror else "T"

    if not axes:
        if has_inv:
            return "Ci"
        return "Cs" if has_mirror else "C1"

    pa, n = axes[0]
    perp = 0
    for a, m in axes:
        if m % 2 == 0 and abs(np.dot(a, pa)) < 0.02:
            perp += 1
    sigma_h = any(abs(np.dot(m, pa)) > 0.999 for m in mirrors)
    if perp >= 1:
        if sigma_h:
            return f"D{n}h"
        if has_mirror:
            return f"D{n}d"
        return f"D{n}"
    if sigma_h:
        return f"C{n}h"
    if has_mirror:
        return f"C{n}v"
    if sym(_mirror(pa) @ _rot(pa, math.pi / n)):
        return f"S{2 * n}"
    return f"C{n}"


point_group = detect_point_group


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: pgmol FILE.xyz", file=sys.stderr)
        return 2
    print(detect_point_group(pathlib.Path(argv[0])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
