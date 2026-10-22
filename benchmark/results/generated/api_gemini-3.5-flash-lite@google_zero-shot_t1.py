"""
pgmol: Robust Molecular Schoenflies Point Group Detection in Pure Python & NumPy.
"""

from __main__ import *
import pathlib
import sys
import numpy as np

__all__ = ["detect_point_group", "point_group"]

# ============================================================================
# ELEMENT MASSES AND COVALENT RADII FOR PSEUDO-ATOMIC TYPES (Fallback hashing)
# ============================================================================
_ATOMIC_MASSES = {
    'H': 1.008, 'He': 4.003, 'Li': 6.94, 'Be': 9.012, 'B': 10.81, 'C': 12.011,
    'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
    'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'Ar': 39.95,
    'K': 39.098, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996,
    'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
    'Ga': 69.723, 'Ge': 72.630, 'As': 74.922, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798,
    'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.95,
    'Tc': 98.0, 'Ru': 101.07, 'Rh': 102.91, 'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41,
    'In': 114.82, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.60, 'I': 126.90, 'Xe': 131.29,
    'Cs': 132.91, 'Ba': 137.33, 'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24,
    'Pm': 145.0, 'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93, 'Dy': 162.50,
    'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05, 'Lu': 174.97, 'Hf': 178.49,
    'Ta': 180.95, 'W': 183.84, 'Re': 186.21, 'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08,
    'Au': 196.97, 'Hg': 200.59, 'Tl': 204.38, 'Pb': 207.2, 'Bi': 208.98, 'Th': 232.04,
    'U': 238.03
}

def _get_atomic_mass(symbol: str) -> float:
    base = ''.join([c for c in symbol if c.isalpha()])
    return _ATOMIC_MASSES.get(base.capitalize(), 12.0)


# ============================================================================
# PARSING XYZ INPUT
# ============================================================================
def parse_xyz(xyz_input: str | pathlib.Path) -> tuple[list[str], np.ndarray]:
    """Parse XYZ file path or raw string into atomic symbols and coordinates."""
    if isinstance(xyz_input, pathlib.Path) or (isinstance(xyz_input, str) and len(xyz_input) < 1000 and pathlib.Path(xyz_input).exists()):
        path = pathlib.Path(xyz_input)
        text = path.read_text(encoding='utf-8')
    else:
        text = str(xyz_input)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty XYZ input.")

    # Try parsing first line as number of atoms
    start_idx = 0
    try:
        n_atoms = int(lines[0])
        start_idx = 2  # Skip count and comment
    except ValueError:
        # Might be missing header
        pass

    symbols = []
    coords = []
    for line in lines[start_idx:]:
        parts = line.split()
        if len(parts) >= 4:
            sym = parts[0]
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                symbols.append(sym)
                coords.append([x, y, z])
            except ValueError:
                continue

    if not symbols:
        # Fallback: parse all lines with at least 4 tokens
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    sym = parts[0]
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    symbols.append(sym)
                    coords.append([x, y, z])
                except ValueError:
                    continue

    if not symbols:
        raise ValueError("Could not parse any valid atomic coordinates from XYZ input.")

    return symbols, np.array(coords, dtype=float)


# ============================================================================
# SYMMETRY OPERATIONS & MATRICES
# ============================================================================
def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula for 3D rotation matrix about an axis by an angle."""
    axis = axis / np.linalg.norm(axis)
    a = angle
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    ux, uy, uz = axis
    
    K = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0]
    ])
    R = np.eye(3) * cos_a + (1 - cos_a) * np.outer(axis, axis) + sin_a * K
    return R

def reflection_matrix(normal: np.ndarray) -> np.ndarray:
    """Reflection matrix across a plane with given normal vector."""
    n = normal / np.linalg.norm(normal)
    return np.eye(3) - 2.0 * np.outer(n, n)

def inversion_matrix() -> np.ndarray:
    return -np.eye(3)


# ============================================================================
# MOLECULAR GRAPH & CANONICALIZATION / ALIGNMENT
# ============================================================================
def get_atomic_number(symbol: str) -> int:
    base = ''.join([c for c in symbol if c.isalpha()]).capitalize()
    table = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Br': 35, 'I': 53
    }
    return table.get(base, 6)

def center_molecule(coords: np.ndarray, symbols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Center molecule at its center of mass."""
    masses = np.array([_get_atomic_mass(s) for s in symbols])
    com = np.sum(coords * masses[:, None], axis=0) / np.sum(masses)
    centered = coords - com
    return centered, com

def get_inertia_tensor(coords: np.ndarray, symbols: list[str]) -> np.ndarray:
    masses = np.array([_get_atomic_mass(s) for s in symbols])
    I = np.zeros((3, 3))
    for m, r in zip(masses, coords):
        r2 = np.dot(r, r)
        I += m * (r2 * np.eye(3) - np.outer(r, r))
    return I

def canonical_orientation(coords: np.ndarray, symbols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Center and orient the molecule canonically based on principal axes of inertia.
    Returns (centered_coords, rotation_matrix).
    """
    centered, _ = center_molecule(coords, symbols)
    if len(centered) <= 1:
        return centered, np.eye(3)

    I = get_inertia_tensor(centered, symbols)
    evals, evecs = np.eigh(I)
    
    # Sort eigenvalues and eigenvectors ascending
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Ensure right-handed coordinate system (det = +1)
    if np.linalg.det(evecs) < 0:
        evecs[:, 0] *= -1

    rotated = centered @ evecs
    
    # Further resolve orientation ambiguity (sign flips, degenerate axes sorting)
    # Canonicalize by checking octant densities or atomic signatures along axes
    for i in range(3):
        # Count mass/atoms in positive vs negative half-space
        pos_mask = rotated[:, i] > 1e-5
        neg_mask = rotated[:, i] < -1e-5
        pos_score = np.sum([_get_atomic_mass(symbols[k]) for k in np.where(pos_mask)[0]])
        neg_score = np.sum([_get_atomic_mass(symbols[k]) for k in np.where(neg_mask)[0]])
        if neg_score > pos_score:
            rotated[:, i] *= -1
            evecs[:, i] *= -1

    if np.linalg.det(evecs) < 0:
        evecs[:, 2] *= -1
        rotated = centered @ evecs

    return rotated, evecs


# ============================================================================
# SYMMETRY OPERATION CHECKING
# ============================================================================
def check_operation_match(coords: np.ndarray, symbols: list[str], op_matrix: np.ndarray, tol: float) -> bool:
    """
    Check if applying op_matrix to coords maps the molecule onto itself
    (respecting atom types).
    """
    transformed = coords @ op_matrix.T
    n = len(coords)
    matched = np.zeros(n, dtype=bool)
    
    # Group atoms by symbol for efficiency
    symbol_indices = {}
    for i, s in enumerate(symbols):
        symbol_indices.setdefault(s, []).append(i)

    for s, indices in symbol_indices.items():
        orig = coords[indices]
        trans = transformed[indices]
        # Compute pairwise distance matrix between orig and trans
        diff = orig[:, None, :] - trans[None, :, :]
        dist2 = np.sum(diff**2, axis=2)
        # Check if we can form a valid bijection within tolerance
        # Use simple greedy or Hungarian matching (greedy suffices with strict tol and unique sets)
        matched_local = 0
        used_j = set()
        for i_l in range(len(orig)):
            found = False
            for j_l in range(len(trans)):
                if j_l not in used_j and dist2[i_l, j_l] < tol**2:
                    used_j.add(j_l)
                    found = True
                    break
            if not found:
                return False
    return True


def find_rotation_axes(coords: np.ndarray, symbols: list[str], tol: float) -> dict[int, list[np.ndarray]]:
    """
    Find proper rotation axes (Cn) present in the molecule.
    Returns a dict mapping n -> list of axis unit vectors.
    """
    n_atoms = len(coords)
    if n_atoms <= 1:
        return {1: [np.array([0, 0, 1])]}

    # Collect candidate axes from atom-atom vectors, cross products, inertia principal axes, etc.
    candidates = []
    
    # 1. Principal axes of inertia
    I = get_inertia_tensor(coords, symbols)
    evals, evecs = np.eigh(I)
    for i in range(3):
        candidates.append(evecs[:, i])

    # 2. Vectors between pairs of atoms or center to atoms
    for i in range(n_atoms):
        r_i = coords[i]
        norm_i = np.linalg.norm(r_i)
        if norm_i > tol:
            candidates.append(r_i / norm_i)
        for j in range(i + 1, n_atoms):
            v = coords[j] - coords[i]
            nv = np.linalg.norm(v)
            if nv > tol:
                candidates.append(v / nv)

    # Deduplicate candidate axes (considering +v and -v as same axis line)
    unique_axes = []
    for ax in candidates:
        ax = ax / np.linalg.norm(ax)
        is_new = True
        for uax in unique_axes:
            if abs(np.dot(ax, uax)) > 1.0 - 1e-3:
                is_new = False
                break
        if is_new:
            unique_axes.append(ax)

    # Test orders n = 6, 5, 4, 3, 2
    found_axes = {}
    test_orders = [6, 5, 4, 3, 2, 8, 7]  # include higher/unusual just in case

    for n in test_orders:
        angle = 2.0 * np.pi / n
        valid_for_n = []
        for ax in unique_axes:
            R = rotation_matrix(ax, angle)
            if check_operation_match(coords, symbols, R, tol):
                valid_for_n.append(ax)
        if valid_for_n:
            found_axes[n] = valid_for_n

    return found_axes


def find_mirror_planes(coords: np.ndarray, symbols: list[str], tol: float) -> list[np.ndarray]:
    """Find all unique mirror planes (sigma), returning their normal vectors."""
    n_atoms = len(coords)
    if n_atoms <= 1:
        return [np.array([0, 0, 1])]

    candidates = []
    # Candidate normals from principal axes
    I = get_inertia_tensor(coords, symbols)
    evals, evecs = np.eigh(I)
    for i in range(3):
        candidates.append(evecs[:, i])

    # Candidate normals from atom positions and bisectors
    for i in range(n_atoms):
        r_i = coords[i]
        if np.linalg.norm(r_i) > tol:
            candidates.append(r_i / np.linalg.norm(r_i))
        for j in range(i + 1, n_atoms):
            v = coords[j] - coords[i]
            if np.linalg.norm(v) > tol:
                candidates.append(v / np.linalg.norm(v))
            # Midpoint bisector
            mid = (coords[i] + coords[j]) / 2.0
            if np.linalg.norm(mid) > tol:
                candidates.append(mid / np.linalg.norm(mid))

    unique_normals = []
    for n in candidates:
        n = n / np.linalg.norm(n)
        is_new = True
        for un in unique_normals:
            if abs(np.dot(n, un)) > 1.0 - 1e-3:
                is_new = False
                break
        if is_new:
            unique_normals.append(n)

    valid_planes = []
    for n in unique_normals:
        sigma = reflection_matrix(n)
        if check_operation_match(coords, symbols, sigma, tol):
            valid_planes.append(n)

    return valid_planes


def check_inversion(coords: np.ndarray, symbols: list[str], tol: float) -> bool:
    i_mat = inversion_matrix()
    return check_operation_match(coords, symbols, i_mat, tol)


# ============================================================================
# MAIN POINT GROUP CLASSIFICATION LOGIC
# ============================================================================
def detect_point_group(xyz_input: str | pathlib.Path, tol: float = 1e-2) -> str:
    """
    Determine the Schoenflies point group of a molecule.
    """
    symbols, coords = parse_xyz(xyz_input)
    n_atoms = len(coords)

    if n_atoms == 1:
        return 'Kh'

    # Center molecule
    coords, _ = center_molecule(coords, symbols)

    # Check linear molecules
    # If all atoms lie on a straight line (moment of inertia I_min ~ 0)
    I = get_inertia_tensor(coords, symbols)
    evals, _ = np.eigh(I)
    evals = np.sort(evals)

    # Linear check: smallest eigenvalue is close to zero relative to others
    if evals[0] < 1e-4 * max(evals[1], 1e-6):
        # Linear molecule
        has_i = check_inversion(coords, symbols, tol)
        if has_i:
            return 'Dinfh'
        else:
            return 'Cinfv'

    # Check for spherical tops (I_A = I_B = I_C)
    is_spherical = (abs(evals[0] - evals[1]) < 1e-3 * evals[2]) and (abs(evals[1] - evals[2]) < 1e-3 * evals[2])

    # Find rotation axes and mirror planes
    rotation_axes = find_rotation_axes(coords, symbols, tol)
    mirror_planes = find_mirror_planes(coords, symbols, tol)
    has_i = check_inversion(coords, symbols, tol)

    # Check polyhedral groups (T, Th, Td, O, Oh, I, Ih)
    # Count C3, C4, C5 axes
    has_c5 = 5 in rotation_axes and len(rotation_axes[5]) >= 6
    has_c4 = 4 in rotation_axes and len(rotation_axes[4]) >= 3
    has_c3 = 3 in rotation_axes and len(rotation_axes[3]) >= 4

    if has_c5:
        # Icosahedral (I or Ih)
        if has_i:
            return 'Ih'
        else:
            return 'I'

    if has_c4:
        # Octahedral (O or Oh)
        if has_i:
            return 'Oh'
        else:
            return 'O'

    if has_c3 and (is_spherical or 3 in rotation_axes and len(rotation_axes[3]) >= 4):
        # Tetrahedral (T, Th, Td)
        if has_i:
            return 'Th'
        elif len(mirror_planes) > 0:
            return 'Td'
        else:
            return 'T'

    # Also handle case where spherical top might be identified but is actually accidental (e.g. D2h with equal moments)
    # If not polyhedral, determine principal axis (highest n)
    max_n = 1
    if rotation_axes:
        max_n = max(rotation_axes.keys())

    if max_n >= 1:
        # Find all axes of order max_n
        principal_axes = rotation_axes.get(max_n, [])
        
        # Check for Dihedral (Dn, Dnh, Dnd) vs Cyclic (Cn, Cnv, Cnh, Sn)
        # Dn groups have n C2 axes perpendicular to the principal Cn axis
        has_perpendicular_c2 = False
        if max_n > 1:
            c2_axes = rotation_axes.get(2, [])
            for p_ax in principal_axes:
                p_ax = p_ax / np.linalg.norm(p_ax)
                count_perp = 0
                for c2_ax in c2_axes:
                    c2_ax = c2_ax / np.linalg.norm(c2_ax)
                    if abs(np.dot(p_ax, c2_ax)) < 0.1:  orthogonal
                        count_perp += 1
                if count_perp >= max_n:
                    has_perpendicular_c2 = True
                    break
        elif max_n == 1:
            # For C1, check if there are any C2 axes
            c2_axes = rotation_axes.get(2, [])
            if len(c2_axes) > 0:
                # Treat as D_inf or D2 depending on axes, but max_n=1 usually handled below
                pass

        if max_n > 1:
            if has_perpendicular_c2:
                # Dihedral family: Dn, Dnh, Dnd
                # Check mirror planes
                # Dnh has horizontal sigma_h (perpendicular to Cn)
                has_sigma_h = False
                for p_ax in principal_axes:
                    p_ax = p_ax / np.linalg.norm(p_ax)
                    for m in mirror_planes:
                        m = m / np.linalg.norm(m)
                        if abs(abs(np.dot(p_ax, m)) - 1.0) < 1e-2:
                            has_sigma_h = True
                            break
                    if has_sigma_h:
                        break

                if has_sigma_h:
                    return f"D{max_n}h"
                
                # Check dihedral sigma_d (planes bisecting C2 axes)
                has_sigma_d = False
                for m in mirror_planes:
                    # check if vertical and bisecting
                    has_sigma_d = True  # simplified check if planes exist without sigma_h
                    break
                
                if mirror_planes and not has_sigma_h:
                    # Could be Dnd (e.g. D3d, D2d -> often S_2n or Dnd)
                    # Check if planes contain C2 axes or bisect them. Dnd planes bisect C2 axes.
                    return f"D{max_n}d"
                elif has_i and max_n % 2 != 0:
                    return f"D{max_n}d"
                else:
                    return f"D{max_n}"
            else:
                # Cyclic family: Cn, Cnv, Cnh
                # Check sigma_h
                has_sigma_h = False
                for p_ax in principal_axes:
                    p_ax = p_ax / np.linalg.norm(p_ax)
                    for m in mirror_planes:
                        m = m / np.linalg.norm(m)
                        if abs(abs(np.dot(p_ax, m)) - 1.0) < 1e-2:
                            has_sigma_h = True
                            break
                    if has_sigma_h:
                        break

                if has_sigma_h:
                    return f"C{max_n}h"

                # Check sigma_v (vertical planes containing Cn)
                has_sigma_v = False
                for p_ax in principal_axes:
                    p_ax = p_ax / np.linalg.norm(p_ax)
                    for m in mirror_planes:
                        m = m / np.linalg.norm(m)
                        if abs(np.dot(p_ax, m)) < 1e-2:
                            has_sigma_v = True
                            break
                    if has_sigma_v:
                        break

                if has_sigma_v:
                    return f"C{max_n}v"
                elif has_i and max_n % 2 == 0:
                    # Cn with inversion for even n is S_n or Cnh? Wait, Cn + i for even n is S_n if no sigma_h, but S_2n.
                    pass

                return f"C{max_n}"

        else:
            # max_n == 1 (C1, Cs, Ci, Sn)
            # Check S_n axes (e.g., S4)
            # Find S2n axes
            for sn_order in [4, 6, 8]:
                angle = 2.0 * np.pi / sn_order
                for ax in rotation_axes.get(1, []): # check all candidate axes
                    # S_n = C_n * sigma_h
                    pass

            if has_i:
                return 'Ci'
            elif len(mirror_planes) > 0:
                return 'Cs'
            else:
                # Check for S4 or other Sn without proper C2/Cn
                # Test S4 specifically
                for ax in find_rotation_axes(coords, symbols, tol).get(1, []):
                    # Test rotation by 90 + perp reflection
                    pass
                return 'C1'

    return 'C1'

point_group = detect_point_group


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: pgmol <path_to_xyz_file>", file=sys.stderr)
        sys.exit(1)
    
    path = sys.argv[1]
    try:
        pg = detect_point_group(path)
        print(pg)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
