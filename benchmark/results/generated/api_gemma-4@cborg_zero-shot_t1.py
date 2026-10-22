import numpy as np
import pathlib
from typing import Union

def detect_point_group(xyz_input: Union[str, pathlib.Path], tol: float = 1e-2) -> str:
    """
    Determine the Schoenflies point group of a molecule from XYZ coordinates.
    
    Invariance is achieved by calculating the inertia tensor to find principal axes
    and checking for symmetry operations based on coordinate transformations.
    """
    # 1. Parse Input
    if isinstance(xyz_input, pathlib.Path) or (isinstance(xyz_input, str) and pathlib.Path(xyz_input).exists()):
        with open(xyz_input, 'r') as f:
            lines = f.readlines()
    else:
        lines = xyz_input.strip().splitlines()

    if not lines:
        raise ValueError("Empty input")

    try:
        num_atoms = int(lines[0].strip())
    except (ValueError, IndexError):
        raise ValueError("Invalid XYZ format: First line must be number of atoms")

    atoms = []
    coords = []
    for i in range(2, 2 + num_atoms):
        parts = lines[i].split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])

    atoms = np.array(atoms)
    coords = np.array(coords)

    # 2. Center and Normalize
    if num_atoms == 0:
        return "C1"
    if num_atoms == 1:
        return "Kh"

    center = np.mean(coords, axis=0)
    coords -= center

    # 3. Find Principal Axes (Inertia Tensor)
    # To handle accidental spherical tops, we use the coordinates directly to find symmetry
    # but use the Inertia Tensor to suggest candidate axes.
    masses = np.ones(num_atoms) # Assume unit mass for geometry symmetry
    inertia = np.zeros((3, 3))
    for i in range(num_atoms):
        r = coords[i]
        inertia += masses[i] * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

    evals, evecs = np.linalg.eigh(inertia)
    # evecs columns are principal axes. 
    # If evals are degenerate, any orthogonal set in that subspace works.
    
    def is_symmetric(matrix: np.ndarray) -> bool:
        """Check if the current coordinates are invariant under the transformation matrix."""
        transformed = coords @ matrix.T
        # We must find a permutation of atoms that maps the original to transformed
        # Sort atoms by element and then coordinates to compare
        def get_sorted_coords(pts):
            # Combine atoms and coords for sorting
            combined = []
            for i in range(len(pts)):
                combined.append((atoms[i], tuple(np.round(pts[i] / tol) * tol)))
            return sorted(combined)

        orig_sorted = get_sorted_coords(coords)
        trans_sorted = get_sorted_coords(transformed)
        
        for (a1, c1), (a2, c2) in zip(orig_sorted, trans_sorted):
            if a1 != a2 or np.linalg.norm(np.array(c1) - np.array(c2)) > tol:
                return False
        return True

    # Helper to generate rotation matrices
    def rot_z(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    def rot_x(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    # Transform coordinates to principal axis system
    coords_local = coords @ evecs
    
    # To check symmetry in local frame, we need to rotate back or just test in local frame
    # Let's redefine is_symmetric to work on coords_local
    def check_op(op_matrix):
        transformed = coords_local @ op_matrix.T
        combined_orig = sorted([(atoms[i], tuple(np.round(coords_local[i]/tol)*tol)) for i in range(num_atoms)])
        combined_trans = sorted([(atoms[i], tuple(np.round(transformed[i]/tol)*tol)) for i in range(num_atoms)])
        for (a1, c1), (a2, c2) in zip(combined_orig, combined_trans):
            if a1 != a2 or np.linalg.norm(np.array(c1) - np.array(c2)) > tol:
                return False
        return True

    # 4. Symmetry Detection Logic
    # Linear check
    dist_from_axis = np.linalg.norm(np.cross(coords_local, [0, 0, 1]), axis=1)
    if np.all(dist_from_axis < tol):
        # Linear: Check for inversion center
        if check_op(-np.eye(3)):
            return "Dinfh"
        return "Cinfv"

    # Find the highest Cn axis (Search for rotations around x, y, z)
    def get_max_n(axis_idx):
        # Try n from 6 down to 2
        for n in range(8, 1, -1): # Search up to C8
            theta = 2 * np.pi / n
            if axis_idx == 0: op = rot_x(theta)
            elif axis_idx == 1: 
                # Rot Y: [[c, 0, s], [0, 1, 0], [-s, 0, c]]
                c, s = np.cos(theta), np.sin(theta)
                op = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            else: op = rot_z(theta)
            if check_op(op): return n
        return 1

    nx = get_max_n(0)
    ny = get_max_n(1)
    nz = get_max_n(2)
    n = max(nx, ny, nz)

    # Center of inversion
    has_i = check_op(-np.eye(3))
    # Mirror planes
    # Sigma_h: plane perpendicular to the principal axis
    def has_sigma_h(axis_idx):
        op = np.eye(3)
        op[axis_idx, axis_idx] = -1
        return check_op(op)
    
    # Sigma_v: planes containing the principal axis
    def has_sigma_v(axis_idx):
        # Check several angles for mirror planes containing the axis
        # Specifically, reflections across planes containing axis
        for angle in np.linspace(0, np.pi, 10):
            # Reflection across plane at angle 'angle'
            # For Z axis, reflection is x' = x cos(2a) + y sin(2a), y' = x sin(2a) - y cos(2a), z' = z
            c, s = np.cos(2*angle), np.sin(2*angle)
            if axis_idx == 2:
                op = np.array([[c, s, 0], [s, -c, 0], [0, 0, 1]])
            elif axis_idx == 0:
                op = np.array([[1, 0, 0], [0, c, s], [0, s, -c]])
            else: # axis 1
                op = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
            if check_op(op): return True
        return False

    # Dihedral axes: Cn axes perpendicular to the main axis
    def has_dn(main_axis_idx):
        # Check for C2 axes perpendicular to main
        # If main is Z, look for C2 in X and Y
        others = [i for i in range(3) if i != main_axis_idx]
        for axis_idx in others:
            theta = np.pi
            if axis_idx == 0: op = rot_x(theta)
            elif axis_idx == 1: 
                c, s = np.cos(theta), np.sin(theta)
                op = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            else: op = rot_z(theta)
            if check_op(op): return True
        return False

    # 5. Polyhedral Logic (T, O, I families)
    # Check for multiple C3 axes (T family) or C4 (O family)
    c3_count = 0
    for i in range(3):
        # This is a simplification; real T/O/I check requires testing more directions
        # But we can check if it satisfies the base requirements
        pass
    
    # Simplified High Symmetry Check: 
    # O_h is the most symmetric (C4 axes, C3 axes, inversion, mirror)
    # We check if it satisfies O_h then O, etc.
    # Given the constraints, we verify specific characteristic operations.
    
    # Check for T, O, I using the number of symmetry operations or specific axes
    # To keep the module self-contained and robust:
    def count_sym_ops():
        # This is an expensive way, but a way to distinguish T/O/I
        # We'll check for C3 axes in non-principal directions
        # For a quick approximation, check for C3 and C4 on multiple axes
        c3 = 0
        c4 = 0
        # Try some directions
        dirs = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]), 
                np.array([1,1,1]), np.array([1,1,-1]), np.array([1,-1,1]), np.array([-1,1,1])]
        for d in dirs:
            d = d / np.linalg.norm(d)
            # Rotation matrix for angle theta around vector d
            def rot_vec(v, theta):
                c, s = np.cos(theta), np.sin(theta)
                return (c * np.eye(3) + s * np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) + (1-c) * np.outer(v,v))
            
            if check_op(rot_vec(d, 2*np.pi/3)): c3 += 1
            if check_op(rot_vec(d, 2*np.pi/4)): c4 += 1
        return c3, c4

    c3_axes, c4_axes = count_sym_ops()
    
    if c3_axes >= 4:
        # Icosahedral or Octahedral
        # Icosahedral has C5 axes
        has_c5 = False
        # Check a few C5 directions (golden ratio)
        phi = (1 + 5**0.5) / 2
        c5_dirs = [np.array([0, 1, phi]), np.array([phi, 0, 1]), np.array([1, phi, 0])]
        for d in c5_dirs:
            d = d / np.linalg.norm(d)
            def rot_vec(v, theta):
                c, s = np.cos(theta), np.sin(theta)
                return (c * np.eye(3) + s * np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) + (1-c) * np.outer(v,v))
            if check_op(rot_vec(d, 2*np.pi/5)):
                has_c5 = True; break
        
        if has_c5:
            return "Ih" if has_i else "I"
        if c4_axes >= 3:
            return "Oh" if has_i else "O"
        if c3_axes >= 4:
            if has_i: return "Th"
            # Check for Td (has C2v mirrors)
            if has_sigma_v(2): return "Td"
            return "T"

    # 6. Axial/Low Symmetry Logic
    # Principal axis is determined by the highest n
    # We need to re-orient the molecule so that the highest n is on Z
    if n > 1:
        # Find which axis had the max n
        if nz == n: main_axis = 2
        elif ny == n: main_axis = 1
        else: main_axis = 0
        
        # Since we check axes independently, let's just use 'n' and the properties
        # based on the axis that provided 'n'.
        s_h = has_sigma_h(main_axis)
        s_v = has_sigma_v(main_axis)
        d_n = has_dn(main_axis)
        
        if d_n:
            if s_h: return f"D{n}h"
            if s_v: return f"D{n}d"
            return f"D{n}"
        if s_h: return f"C{n}h"
        if s_v: return f"C{n}v"
        return f"C{n}"

    # n == 1
    if has_i: return "Ci"
    # Check for any mirror plane
    for i in range(3):
        op = np.eye(3)
        op[i, i] = -1
        if check_op(op): return "Cs"
    
    # Check for Sn (specifically S4, S6 etc)
    for sn in [6, 4]:
        # S_n is rotation by 2pi/n then reflection across plane perp to axis
        # Try Z axis
        op = rot_z(2 * np.pi / sn) @ np.array([[1,0,0],[0,1,0],[0,0,-1]])
        if check_op(op): return f"S{sn}"

    return "C1"

point_group = detect_point_group

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(detect_point_group(sys.argv[1]))
    else:
        print("Usage: pgmol <file.xyz>")
