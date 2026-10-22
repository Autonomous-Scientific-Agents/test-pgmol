import sys
import math
import pathlib
import re
import itertools
import numpy as np

# Package metadata
__version__ = "0.1.0"
__all__ = ["detect_point_group", "point_group"]

# Dictionary of standard atomic masses for center of mass calculation
ELEMENT_MASSES = {
    'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81, 'C': 12.011,
    'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
    'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'Ar': 39.948,
    'K': 39.098, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996,
    'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
    'Ga': 69.723, 'Ge': 72.63, 'As': 74.922, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798,
    'I': 126.90, 'Xe': 131.29,
}


def clean_element(sym: str) -> str:
    """Clean chemical element symbols from labels like C1, H_a, etc."""
    m = re.match(r'^([a-zA-Z]{1,2})', sym)
    if m:
        return m.group(1).capitalize()
    return sym.capitalize()


def parse_xyz(xyz_input: str | pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse raw multiline XYZ text or an XYZ file path."""
    raw_str = None
    if isinstance(xyz_input, pathlib.Path):
        with open(xyz_input, 'r', encoding='utf-8') as f:
            raw_str = f.read()
    elif isinstance(xyz_input, str):
        try:
            p = pathlib.Path(xyz_input)
            if p.is_file():
                with open(p, 'r', encoding='utf-8') as f:
                    raw_str = f.read()
        except Exception:
            pass
        if raw_str is None:
            raw_str = xyz_input

    if not raw_str:
        raise ValueError("Empty or invalid XYZ input.")

    lines = [line.strip() for line in raw_str.strip().split('\n') if line.strip()]
    if not lines:
        raise ValueError("No content found in XYZ input.")

    try:
        num_atoms = int(lines[0])
    except ValueError:
        raise ValueError(f"Invalid atom count in the first line: {lines[0]}")

    atom_lines = lines[2:2+num_atoms]
    elements = []
    coords = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed coordinate line: {line}")
        elements.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return np.array(elements), np.array(coords)


def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Construct a 3D rotation matrix around a given axis by theta radians."""
    axis = axis / np.linalg.norm(axis)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    ux, uy, uz = axis
    K = np.array([
        [0.0, -uz, uy],
        [uz, 0.0, -ux],
        [-uy, ux, 0.0]
    ])
    R = np.eye(3) + sin_t * K + (1.0 - cos_t) * (K @ K)
    return R


def is_symmetry_operation(coords: np.ndarray, elements: np.ndarray, R: np.ndarray, tol: float) -> bool:
    """Check if the orthogonal transformation matrix R is a valid symmetry operation."""
    unique_elems = np.unique(elements)
    for elem in unique_elems:
        mask = (elements == elem)
        Y = coords[mask]
        Y_prime = Y @ R.T
        
        # Vectorized pairwise squared distance check
        dist2 = np.sum((Y_prime[:, None, :] - Y[None, :, :])**2, axis=-1)
        
        # Every transformed point must have a counterpart within the tolerance
        if np.any(np.min(dist2, axis=1) > tol**2) or np.any(np.min(dist2, axis=0) > tol**2):
            return False
    return True


def get_candidate_axes(coords: np.ndarray, elements: np.ndarray, tol: float) -> list[np.ndarray]:
    """Generate a robust, unique set of unit-vector axes based on molecular geometry."""
    norms = np.linalg.norm(coords, axis=1)
    non_origin_mask = norms > tol
    coords_nonzero = coords[non_origin_mask]
    elements_nonzero = elements[non_origin_mask]

    candidates = []

    # 1. Inertia tensor eigenvectors
    masses = np.array([ELEMENT_MASSES.get(el, 12.0) for el in elements])
    I = np.zeros((3, 3))
    for m, r in zip(masses, coords):
        I += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    try:
        evals, evecs = np.linalg.eigh(I)
        for i in range(3):
            candidates.append(evecs[:, i])
    except np.linalg.LinAlgError:
        pass

    # 2. Standard Cartesian directions
    candidates.extend([
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0])
    ])

    # 3. Geometric vectors (bonds, cross products, angle bisectors)
    if len(coords_nonzero) > 0:
        unique_el, counts = np.unique(elements_nonzero, return_counts=True)
        sorted_indices = np.argsort(counts)
        best_elem = unique_el[sorted_indices[0]]
        pts = coords_nonzero[elements_nonzero == best_elem]

        idx = 1
        while len(pts) < 6 and idx < len(unique_el):
            next_elem = unique_el[sorted_indices[idx]]
            pts = np.vstack([pts, coords_nonzero[elements_nonzero == next_elem]])
            idx += 1

        # Truncate to prevent combinatorial growth on large systems
        if len(pts) > 24:
            pts = pts[:24]

        for p in pts:
            candidates.append(p / np.linalg.norm(p))

        n_pts = len(pts)
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                p1, p2 = pts[i], pts[j]
                
                s = p1 + p2
                ns = np.linalg.norm(s)
                if ns > 1e-4:
                    candidates.append(s / ns)

                d = p1 - p2
                nd = np.linalg.norm(d)
                if nd > 1e-4:
                    candidates.append(d / nd)

                cp = np.cross(p1, p2)
                ncp = np.linalg.norm(cp)
                if ncp > 1e-4:
                    candidates.append(cp / ncp)

    # Clean and filter to a single hemisphere, removing duplicates
    unique_candidates = []
    for v in candidates:
        v = v / np.linalg.norm(v)
        if v[2] < -1e-5:
            v = -v
        elif abs(v[2]) < 1e-5:
            if v[1] < -1e-5:
                v = -v
            elif abs(v[1]) < 1e-5:
                if v[0] < -1e-5:
                    v = -v

        is_dup = False
        for u in unique_candidates:
            if np.dot(u, v) > 1.0 - 1e-4:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(v)

    return unique_candidates


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
    elements_raw, coords = parse_xyz(xyz_input)
    elements = np.array([clean_element(el) for el in elements_raw])

    # 1. Handle single atom
    if len(coords) == 1:
        return 'Kh'

    # Centering coordinates relative to the Center of Mass
    masses = np.array([ELEMENT_MASSES.get(el, 12.0) for el in elements])
    com = np.sum(coords * masses[:, None], axis=0) / np.sum(masses)
    coords = coords - com

    # 2. Check for linear molecules
    is_linear = False
    if len(coords) >= 2:
        cov = np.cov(coords.T)
        try:
            evals, evecs = np.linalg.eigh(cov)
            primary_axis = evecs[:, 2]
            dists = [np.linalg.norm(r - np.dot(r, primary_axis) * primary_axis) for r in coords]
            if np.max(dists) < tol:
                is_linear = True
        except Exception:
            pass

    # Check inversion symmetry operation
    has_inversion = is_symmetry_operation(coords, elements, -np.eye(3), tol)

    if is_linear:
        return 'Dinfh' if has_inversion else 'Cinfv'

    # Find candidate symmetry axes
    candidate_axes = get_candidate_axes(coords, elements, tol)

    proper_axes = []        # List of (axis, k)
    reflection_planes = []  # List of plane normal vectors
    improper_axes = []      # List of (axis, k)

    # Analyze symmetry operations along candidate axes
    orders_to_check = [12, 10, 8, 7, 6, 5, 4, 3, 2]
    for n in candidate_axes:
        # Proper Rotations
        found_k = 1
        for k in orders_to_check:
            R = rotation_matrix(n, 2.0 * np.pi / k)
            if is_symmetry_operation(coords, elements, R, tol):
                found_k = k
                break
        if found_k > 1:
            proper_axes.append((n, found_k))

        # Reflection Plane (normal parallel to n)
        R_ref = np.eye(3) - 2.0 * np.outer(n, n)
        if is_symmetry_operation(coords, elements, R_ref, tol):
            reflection_planes.append(n)

        # Improper Rotations
        found_s = 1
        for k in orders_to_check[:-1]:  # S_2 is Ci (inversion), S_1 is Cs (reflection)
            R_s = (np.eye(3) - 2.0 * np.outer(n, n)) @ rotation_matrix(n, 2.0 * np.pi / k)
            if is_symmetry_operation(coords, elements, R_s, tol):
                found_s = k
                break
        if found_s > 1:
            improper_axes.append((n, found_s))

    # Identify maximum proper rotation order
    n_max = max((k for _, k in proper_axes), default=1)

    # 3. High-symmetry (Polyhedral) Groups
    high_axes = [(n, k) for n, k in proper_axes if k >= 3]
    if len(high_axes) > 1:
        poly_n_max = max(k for _, k in high_axes)
        if poly_n_max == 5:
            return 'Ih' if (has_inversion or len(reflection_planes) > 0) else 'I'
        elif poly_n_max == 4:
            return 'Oh' if (has_inversion or len(reflection_planes) > 0) else 'O'
        elif poly_n_max == 3:
            if has_inversion:
                return 'Th'
            elif len(reflection_planes) > 0:
                return 'Td'
            else:
                return 'T'

    # 4. Axial / Low Symmetry Groups
    if n_max >= 2:
        candidate_principal_axes = [n for n, k in proper_axes if k == n_max]
        best_group = None
        best_score = -1

        for np_axis in candidate_principal_axes:
            # Perpendicular C2 proper axes
            has_perp_C2 = False
            for n, k in proper_axes:
                if k == 2 and abs(np.dot(n, np_axis)) < 1e-3:
                    has_perp_C2 = True
                    break

            # Horizontal plane (normal parallel to np_axis)
            has_horizontal_plane = False
            for m in reflection_planes:
                if 1.0 - abs(np.dot(m, np_axis)) < 1e-3:
                    has_horizontal_plane = True
                    break

            # Vertical plane (normal perpendicular to np_axis)
            has_vertical_plane = False
            for m in reflection_planes:
                if abs(np.dot(m, np_axis)) < 1e-3:
                    has_vertical_plane = True
                    break

            # Improper S_{2n} axis collinear with np_axis
            has_S2n = False
            for n, k in improper_axes:
                if k == 2 * n_max and (1.0 - abs(np.dot(n, np_axis)) < 1e-3):
                    has_S2n = True
                    break

            if has_perp_C2:
                if has_horizontal_plane:
                    grp, score = f'D{n_max}h', 6
                elif has_vertical_plane:
                    grp, score = f'D{n_max}d', 5
                else:
                    grp, score = f'D{n_max}', 4
            else:
                if has_horizontal_plane:
                    grp, score = f'C{n_max}h', 3
                elif has_vertical_plane:
                    grp, score = f'C{n_max}v', 3
                elif has_S2n:
                    grp, score = f'S{2 * n_max}', 2
                else:
                    grp, score = f'C{n_max}', 1

            if score > best_score:
                best_score = score
                best_group = grp

        return best_group if best_group is not None else f'C{n_max}'

    # 5. Very Low Symmetry Groups (n_max == 1)
    if len(reflection_planes) > 0:
        return 'Cs'
    if has_inversion:
        return 'Ci'
    return 'C1'


# Alias required by the API contract
point_group = detect_point_group


def main():
    """Command Line Interface entrypoint."""
    import argparse
    parser = argparse.ArgumentParser(description="Identify Schoenflies point group of a molecule from XYZ coordinates.")
    parser.add_argument("xyz_file", help="Path to the XYZ coordinate file.")
    parser.add_argument("--tol", type=float, default=1e-2, help="Cartesian distance matching tolerance in Å (default: 1e-2).")
    args = parser.parse_args()

    try:
        pg = detect_point_group(args.xyz_file, tol=args.tol)
        print(pg)
    except Exception as
