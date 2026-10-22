import sys
import pathlib
import argparse
import numpy as np
from collections import defaultdict

def rotation_matrix(axis, theta):
    """Return the rotation matrix for rotation around axis by theta."""
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
    ])

def reflection_matrix(normal):
    """Return the reflection matrix across the plane perpendicular to normal."""
    n = normal / np.linalg.norm(normal)
    return np.eye(3) - 2 * np.outer(n, n)

def is_sym_op(coords_dict, R, tol):
    """Check if the rotation/reflection R is a symmetry operation."""
    for elem, coords in coords_dict.items():
        if len(coords) == 0:
            continue
        transformed = coords @ R.T
        diff = transformed[:, None, :] - coords[None, :, :]
        dist2 = np.sum(diff**2, axis=-1)
        min_dist2 = np.min(dist2, axis=1)
        if np.max(min_dist2) > tol**2:
            return False
    return True

def get_candidate_axes(coords):
    """Generate potential symmetry axes from coordinates."""
    axes = []
    
    # Principal axes of inertia
    I = coords.T @ coords
    eigvals, eigvecs = np.linalg.eigh(I)
    axes.extend([eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]])
    
    norms = np.linalg.norm(coords, axis=1)
    valid = norms > 1e-3
    valid_coords = coords[valid]
    
    # Vectors to atoms
    if len(valid_coords) > 0:
        axes.extend(valid_coords / norms[valid, None])
    
    # Midpoints and cross products between pairs
    n_valid = len(valid_coords)
    for i in range(n_valid):
        for j in range(i + 1, n_valid):
            v1 = valid_coords[i]
            v2 = valid_coords[j]
            midpoint = (v1 + v2) / 2
            norm_mid = np.linalg.norm(midpoint)
            if norm_mid > 1e-3:
                axes.append(midpoint / norm_mid)
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            if norm_cross > 1e-3:
                axes.append(cross / norm_cross)
                
    # Deduplicate
    unique_axes = []
    for ax in axes:
        ax = ax / np.linalg.norm(ax)
        is_dup = False
        for u in unique_axes:
            if abs(np.dot(ax, u)) > 1 - 1e-4:
                is_dup = True
                break
        if not is_dup:
            unique_axes.append(ax)
            
    return unique_axes

def detect_point_group(xyz_input: str | pathlib.Path, tol: float = 1e-2) -> str:
    """
    Determine the Schoenflies point group of a molecule.
    """
    if isinstance(xyz_input, pathlib.Path):
        content = xyz_input.read_text().strip().split('\n')
    else:
        try:
            if pathlib.Path(xyz_input).is_file():
                content = pathlib.Path(xyz_input).read_text().strip().split('\n')
            else:
                content = xyz_input.strip().split('\n')
        except OSError:
            content = xyz_input.strip().split('\n')

    # Parse coordinates
    N = int(content[0].strip())
    coords_dict = defaultdict(list)
    all_coords = []
    masses = []
    
    # Simple mass map for center of mass approximation
    for line in content[2:N+2]:
        parts = line.split()
        if len(parts) >= 4:
            elem = parts[0]
            x, y, z = map(float, parts[1:4])
            coords_dict[elem].append([x, y, z])
            all_coords.append([x, y, z])
            # Arbitrary weighting: uniform per atom works just as well for symmetry
            masses.append(1.0) 
            
    if N == 1:
        return 'Kh'

    # Numpy arrays
    for k in coords_dict:
        coords_dict[k] = np.array(coords_dict[k])
    all_coords = np.array(all_coords)
    masses = np.array(masses)
    
    # Center to center of mass
    com = np.average(all_coords, axis=0, weights=masses)
    for k in coords_dict:
        coords_dict[k] -= com
    all_coords -= com

    # Check for linear
    is_linear = False
    ref_vec = None
    for pt in all_coords:
        if np.linalg.norm(pt) > 1e-3:
            if ref_vec is None:
                ref_vec = pt / np.linalg.norm(pt)
            else:
                if np.linalg.norm(np.cross(pt, ref_vec)) > tol:
                    break
    else:
        is_linear = True

    inversion_sym = is_sym_op(coords_dict, -np.eye(3), tol)
    if is_linear:
        return 'Dinfh' if inversion_sym else 'Cinfv'

    # Candidate axes
    candidate_axes = get_candidate_axes(all_coords)
    
    cn_axes = defaultdict(list)
    for ax in candidate_axes:
        for n in [8, 7, 6, 5, 4, 3, 2]:
            theta = 2 * np.pi / n
            R = rotation_matrix(ax, theta)
            if is_sym_op(coords_dict, R, tol):
                cn_axes[n].append(ax)
                break 

    n_max = max(cn_axes.keys()) if cn_axes else 1

    # Polyhedral / Cubic
    if n_max >= 3 and len(cn_axes[n_max]) > 1:
        # Icosahedral
        if n_max == 5:
            return 'Ih' if inversion_sym else 'I'
        # Octahedral
        elif n_max == 4:
            return 'Oh' if inversion_sym else 'O'
        # Tetrahedral
        else:
            if inversion_sym:
                return 'Th'
            # Check for S4
            s4_found = False
            for ax in candidate_axes:
                S4 = reflection_matrix(ax) @ rotation_matrix(ax, 2 * np.pi / 4)
                if is_sym_op(coords_dict, S4, tol):
                    s4_found = True
                    break
            if s4_found:
                return 'Td'
            return 'T'

    # Find principal axis
    if n_max > 1:
        principal_axis = cn_axes[n_max][0]
    else:
        principal_axis = np.array([0, 0, 1]) # Arbitrary

    # Check for perpendicular C2 axes
    has_perp_C2 = False
    if n_max > 1:
        for ax in cn_axes.get(2, []):
            if abs(np.dot(ax, principal_axis)) < 1e-3:
                has_perp_C2 = True
                break

    sigma_h_sym = is_sym_op(coords_dict, reflection_matrix(principal_axis), tol) if n_max > 1 else False

    if n_max > 1:
        if has_perp_C2:
            if sigma_h_sym:
                return f'D{n_max}h'
            # Check for sigma_d
            sigma_d = False
            for ax in candidate_axes:
                if abs(np.dot(ax, principal_axis)) < 1e-3:
                    if is_sym_op(coords_dict, reflection_matrix(ax), tol):
                        sigma_d = True
                        break
            return f'D{n_max}d' if sigma_d else f'D{n_max}'
        else:
            if sigma_h_sym:
                return f'C{n_max}h'
            # Check for sigma_v
            sigma_v = False
            for ax in candidate_axes:
                if abs(np.dot(ax, principal_axis)) < 1e-3:
                    if is_sym_op(coords_dict, reflection_matrix(ax), tol):
                        sigma_v = True
                        break
            if sigma_v:
                return f'C{n_max}v'
            # Check for S_2n
            s2n_found = False
            if n_max % 2 == 0 or n_max == 1:
                pass # usually handled
            S_2n_mat = reflection_matrix(principal_axis) @ rotation_matrix(principal_axis, 2 * np.pi / (2 * n_max))
            if is_sym_op(coords_dict, S_2n_mat, tol):
                return f'S{2 * n_max}'
            return f'C{n_max}'
    else:
        if inversion_sym:
            return 'Ci'
        # Check for any reflection plane
        has_sigma = False
        for ax in candidate_axes:
            if is_sym_op(coords_dict, reflection_matrix(ax), tol):
                has_sigma = True
                break
        return 'Cs' if has_sigma else 'C1'

point_group = detect_point_group

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Determine Schoenflies point group from an XYZ file.")
    parser.add_argument("xyz_file", type=str, help="Path to the XYZ file")
    parser.add_argument("--tol", type=float, default=1e-2, help="Distance tolerance for symmetry operations")
    args = parser.parse_args()
    
    pg = detect_point_group(args.xyz_file, tol=args.tol)
    print(pg)
