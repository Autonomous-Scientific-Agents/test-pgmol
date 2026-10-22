"""
pgmol - Schoenflies point-group detection from XYZ coordinates.
Only depends on NumPy and the Python standard library.
"""
from __future__ import annotations

import pathlib
import sys
import argparse
import numpy as np

__all__ = ["detect_point_group", "point_group", "main"]

# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _parse_input(xyz_input: str | pathlib.Path) -> tuple[list[str], np.ndarray]:
    """Parse raw XYZ text or file path into (symbols, Nx3 array)."""
    path = pathlib.Path(str(xyz_input))
    if path.is_file():
        text = path.read_text()
    else:
        text = str(xyz_input)

    lines = text.splitlines()
    idx = 0
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx >= len(lines):
        raise ValueError("XYZ input is empty")
    try:
        n_atoms = int(lines[idx].strip())
    except Exception as exc:
        raise ValueError("First line must be number of atoms") from exc

    symbols: list[str] = []
    coords: list[list[float]] = []
    # Comment line is at idx+1; data starts at idx+2
    for line in lines[idx + 2:]:
        if len(symbols) >= n_atoms:
            break
        line = line.strip()
        if line == "":
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(symbols) == 0:
        raise ValueError("No atomic coordinates found")
    return symbols, np.array(coords, dtype=float)


def _is_symmetry(p: np.ndarray, elements: list[str], R: np.ndarray, t: np.ndarray, tol: float) -> bool:
    """True iff R @ p[i] + t matches p[j] for all i with same element (permutation)."""
    # Centered or not - works with arbitrary t.
    p_trans = (R @ p.T).T + t
    n = len(elements)
    match_r = [-1] * n  # target -> source

    def dfs(u: int, seen: list[bool]) -> bool:
        for v in range(n):
            if elements[v] != elements[u]:
                continue
            if seen[v]:
                continue
            if np.linalg.norm(p_trans[u] - p[v]) < tol:
                seen[v] = True
                if match_r[v] == -1 or dfs(match_r[v], seen):
                    match_r[v] = u
                    return True
        return False

    for u in range(n):
        seen = [False] * n
        if not dfs(u, seen):
            return False
    return True


def _find_symmetries(p: np.ndarray, elements: list[str], tol: float = 1e-2) -> list[tuple[np.ndarray, np.ndarray]]:
    """Brute-force search for discrete orthogonal symmetries (R, t)."""
    p = np.array(p, dtype=float)
    p = p - p.mean(axis=0)
    ops: list[tuple[np.ndarray, np.ndarray]] = [(np.eye(3, dtype=float), np.zeros(3, dtype=float))]
    n = len(elements)
    if n <= 1:
        return ops

    # Find a non-collinear reference triple
    refs = None
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = p[j] - p[i]
                v2 = p[k] - p[i]
                if np.linalg.norm(np.cross(v1, v2)) > 1e-4:
                    refs = (i, j, k)
                    break
            if refs is not None:
                break
        if refs is not None:
            break

    if refs is None:
        # Essentially linear / very small; discrete search will not find non-identity ops
        # reliably via triple method. Return identity only; linear classification
        # relies on geometry check.
        return ops

    i0, i1, i2 = refs
    d01 = float(np.linalg.norm(p[i1] - p[i0]))
    d02 = float(np.linalg.norm(p[i2] - p[i0]))
    d12 = float(np.linalg.norm(p[i2] - p[i1]))

    for j0 in range(n):
        if elements[j0] != elements[i0]:
            continue
        for j1 in range(n):
            if elements[j1] != elements[i1]:
                continue
            if abs(np.linalg.norm(p[j1] - p[j0]) - d01) > tol:
                continue
            for j2 in range(n):
                if elements[j2] != elements[i2]:
                    continue
                if abs(np.linalg.norm(p[j2] - p[j0]) - d02) > tol:
                    continue
                if abs(np.linalg.norm(p[j2] - p[j1]) - d12) > tol:
                    continue

                a = p[i0]
                b = p[i1]
                c = p[i2]
                a_ = p[j0]
                b_ = p[j1]
                c_ = p[j2]

                u1 = b - a
                u2 = c - a
                u3 = np.cross(u1, u2)

                v1 = b_ - a_
                v2 = c_ - a_

                for det_sign in (1, -1):
                    v3 = det_sign * np.cross(v1, v2)
                    U = np.column_stack((u1, u2, u3))
                    if abs(np.linalg.det(U)) < 1e-6:
                        continue
                    M = np.column_stack((v1, v2, v3))
                    try:
                        R = M @ np.linalg.inv(U)
                    except np.linalg.LinAlgError:
                        continue
                    if np.linalg.norm(R.T @ R - np.eye(3)) > 1e-2:
                        continue
                    det_R = float(np.linalg.det(R))
                    if abs(det_R - det_sign) > 0.2:
                        continue
                    t = a_ - R @ a
                    if _is_symmetry(p, elements, R, t, tol):
                        # deduplicate
                        is_new = True
                        for R_old, t_old in ops:
                            if np.linalg.norm(R - R_old) < 1e-2 and np.linalg.norm(t - t_old) < 1e-2:
                                is_new = False
                                break
                        if is_new:
                            ops.append((R.copy(), t.copy()))
    return ops


def _classify(op_list: list[tuple[np.ndarray, np.ndarray]],
               p_raw: np.ndarray,
               elements: list[str],
               tol: float = 1e-2) -> str:
    """Return Schoenflies symbol from discovered symmetry operations."""
    p = np.array(p_raw, dtype=float)
    p -= p.mean(axis=0)
    n = len(elements)

    # Single atom
    if n == 1:
        return "Kh"

    # Linear geometry detection (all atoms essentially on a line)
    cov = np.cov(p.T) if n > 1 else np.eye(3) * 1e-6
    try:
        eigvals = np.sort(np.linalg.eigvalsh(cov))
    except Exception:
        eigvals = np.zeros(3)
    linear_geo = (eigvals[-2] < 1e-3) and (eigvals[-3] < 1e-3)

    # For linear molecules the only discrete groups are Dinfh / Cinfv.
    # Inversion check from operations
    inv_op = False
    for R, t in op_list:
        if np.linalg.norm(R + np.eye(3)) < 1e-2 and np.linalg.norm(t) < tol * 3:
            inv_op = True
            break
    if linear_geo or n == 2:
        return "Dinfh" if inv_op else "Cinfv"

    # Extract unique rotation matrices (ignore small translations)
    unique_R: list[np.ndarray] = []
    for R, t in op_list:
        if np.linalg.norm(t) > tol * 5:
            # For centered geometries true symmetries have t~0.
            pass
        is_new = True
        for R2 in unique_R:
            if np.linalg.norm(R - R2) < 1e-2:
                is_new = False
                break
        if is_new:
            unique_R.append(R.copy())

    # Inversion presence
    inv_present = any(np.linalg.norm(R + np.eye(3)) < 1e-2 for R in unique_R)

    # Count rotation axes (det=+1, not identity)
    axes: dict[int, list[np.ndarray]] = {}
    for R in unique_R:
        det = float(np.linalg.det(R))
        if abs(det - 1) > 0.3:
            continue
        tr = float(np.trace(R))
        if abs(tr - 3) < 0.3:
            continue  # identity
        ang = float(np.arccos(np.clip((tr - 1) / 2, -1.0, 1.0)))
        if ang < 1e-3:
            continue
        n_est = max(2, int(round(2 * np.pi / ang)) if ang > 1e-2 else 2)
        # Verify angle matches expected
        if abs(ang - 2 * np.pi / n_est) > 0.3:
            # Try closest integer
            n_est = max(2, int(round(2 * np.pi / ang)))
            if abs(ang - 2 * np.pi / n_est) > 0.3:
                n_est = 2
        # Axis from nullspace of (R - I)
        M = R - np.eye(3)
        _, _, Vt = np.linalg.svd(M)
        axis = Vt[-1, :]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        # Hemisphere normalization
        for i in range(3):
            if abs(axis[i]) > 1e-3:
                if axis[i] < 0:
                    axis = -axis
                break
        if n_est not in axes:
            axes[n_est] = []
        dup = False
        for a in axes[n_est]:
            if np.linalg.norm(axis - a) < 1e-2 or np.linalg.norm(axis + a) < 1e-2:
                dup = True
                break
        if not dup:
            axes[n_est].append(axis.copy())

    # ----- Polyhedral families -----
    # T family
    if (3 in axes and len(axes.get(3, [])) == 4 and
        2 in axes and len(axes.get(2, [])) == 3):
        if inv_present:
            return "Th"
        # Td has S4 operations (improper rotation, trace ~ -1, det=-1, axis along C2)
        s4 = False
        for R in unique_R:
            if abs(np.linalg.det(R) + 1) < 0.3:
                if abs(np.trace(R) + 1) < 0.6:
                    s4 = True
                    break
        if s4:
            return "Td"
        return "T"

    # O family
    if (4 in axes and len(axes.get(4, [])) == 3 and
        3 in axes and len(axes.get(3, [])) == 4):
        return "Oh" if inv_present else "O"

    # I family
    if (5 in axes and len(axes.get(5, [])) == 6 and
        3 in axes and len(axes.get(3, [])) == 10 and
        2 in axes and len(axes.get(2, [])) == 15):
        return "Ih" if inv_present else "I"

    # ----- Axial families -----
    # Principal axis = highest n found
    main_n = 0
    main_axis = None
    for k in sorted(axes.keys(), reverse=True):
        if k > main_n:
            main_n = k
            main_axis = axes[k][0] if axes[k] else None

    if main_n >= 2:
        # Horizontal mirror check (normal = principal axis)
        sigma_h = False
        for R in unique_R:
            det = float(np.linalg.det(R))
            if abs(det + 1) < 0.3 and abs(np.trace(R) - 1) < 0.5:
                M = R + np.eye(3)
                _, _, Vt = np.linalg.svd(M)
                normal = Vt[-1, :]
                normal /= np.linalg.norm(normal) + 1e-12
                if np.dot(normal, main_axis) < 0:
                    normal = -normal
                # Plane is perp to axis => normal aligns with axis
                if np.linalg.norm(np.abs(normal) - np.abs(main_axis)) < 0.25:
                    sigma_h = True

        # Perpendicular C2 axes
        c2_axes = axes.get(2, [])
        perp_c2 = sum(1 for a in c2_axes if abs(np.dot(a, main_axis)) < 0.25)
        c2_perp = (perp_c2 == main_n)

        if c2_perp:
            if sigma_h:
                return f"D{main_n}h"
            if inv_present:
                return f"D{main_n}d"
            return f"D{main_n}"
        else:
            if sigma_h:
                return f"C{main_n}h"
            # Count vertical planes (reflection, normal perp to axis)
            sigma_v = 0
            for R in unique_R:
                if abs(np.linalg.det(R) + 1) < 0.3 and abs(np.trace(R) - 1) < 0.5:
                    M = R + np.eye(3)
                    _, _, Vt = np.linalg.svd(M)
                    normal = Vt[-1, :]
                    normal /= np.linalg.norm(normal) + 1e-12
                    if abs(np.dot(normal, main_axis)) < 0.25:
                        sigma_v += 1
            if sigma_v >= main_n:
                return f"C{main_n}v"
            # Improper rotation Sn check
            sn = False
            for R in unique_R:
                if abs(np.linalg.det(R) + 1) < 0.3:
                    M = R + np.eye(3)
                    _, _, Vt = np.linalg.svd(M)
                    axis_s = Vt[-1, :]
                    axis_s /= np.linalg.norm(axis_s) + 1e-12
                    if abs(np.dot(axis_s, main_axis)) > 0.9:
                        tr = float(np.trace(R))
                        cos_th = (tr + 1) / 2
                        if -1 <= cos_th <= 1:
                            th = float(np.arccos(np.clip(cos_th, -1, 1)))
                            n_est = max(2, int(round(2 * np.pi / th)) if th > 1e-3 else 2)
                            if abs(th - 2 * np.pi / n_est) < 0.2 and n_est == main_n:
                                sn = True
                                break
            if sn:
                return f"S{main_n}"
            return f"C{main_n}"

    # No rotation axis
    refl = 0
    for R in unique_R:
        if abs(np.linalg.det(R) + 1) < 0.3 and abs(np.trace(R) - 1) < 0.5:
            refl += 1
    if inv_present:
        return "Ci" if refl == 0 else "Ci"  # with only identity+inv; if refl also -> still report Ci as closest standard
    if refl > 0:
        return "Cs"
    return "C1"


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

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
        Exact ASCII Schoenflies symbol.
    """
    symbols, coords = _parse_input(xyz_input)
    ops = _find_symmetries(coords, symbols, tol)
    return _classify(ops, coords, symbols, tol)


point_group = detect_point_group


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Detect Schoenflies point group from an XYZ file.")
    parser.add_argument("file", nargs="?", help="XYZ file (default: stdin)", default="-")
    args = parser.parse_args(argv)

    if args.file == "-":
        text = sys.stdin.read()
    else:
        text = pathlib.Path(args.file).read_text()
    print(detect_point_group(text, tol=1e-2))


if __name__ == "__main__":
    main()
