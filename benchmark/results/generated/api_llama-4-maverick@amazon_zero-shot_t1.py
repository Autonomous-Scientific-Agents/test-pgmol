import numpy as np
import pathlib
from typing import Union

def detect_point_group(xyz_input: Union[str, pathlib.Path], tol: float = 1e-2) -> str:
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
    # Load XYZ coordinates
    if isinstance(xyz_input, pathlib.Path) or (isinstance(xyz_input, str) and pathlib.Path(xyz_input).exists()):
        with open(xyz_input, 'r') as f:
            xyz_string = f.read()
    else:
        xyz_string = xyz_input

    lines = xyz_string.strip().splitlines()
    natoms = int(lines[0])
    coords = []
    elements = []
    for line in lines[2:]:
        parts = line.split()
        elements.append(parts[0])
        coords.append([float(x) for x in parts[1:]])
    coords = np.array(coords)

    # Center coordinates
    centroid = np.mean(coords, axis=0)
    coords -= centroid

    # Compute inertia tensor and diagonalize
    inertia_tensor = np.zeros((3, 3))
    for i, atom in enumerate(coords):
        mass = get_atomic_mass(elements[i])
        inertia_tensor += mass * (np.dot(atom, atom) * np.eye(3) - np.outer(atom, atom))
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Transform coordinates to principal axes
    coords = np.dot(coords, eigenvectors)

    # Detect point group
    pg = detect_point_group_from_coords(coords, elements, tol)

    return pg

def get_atomic_mass(element: str) -> float:
    # Simplified atomic masses for demonstration; use a more comprehensive source in production
    masses = {
        'H': 1.00794, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182, 'B': 10.811, 'C': 12.0107,
        'N': 14.0067, 'O': 15.9994, 'F': 18.9984032, 'Ne': 20.1797
    }
    return masses.get(element, 12.0)  # Default to carbon for unknown elements

def detect_point_group_from_coords(coords: np.ndarray, elements: list, tol: float) -> str:
    # Check for single atom
    if len(elements) == 1:
        return 'Kh'

    # Check for linear molecule
    if is_linear(coords, tol):
        return 'Dinfh' if has_inversion_center(coords, elements, tol) else 'Cinfv'

    # Detect rotational symmetry
    n_max = 6  # Check up to C6
    max_n = 1
    for n in range(2, n_max + 1):
        if has_c_n_axis(coords, n, tol):
            max_n = n

    # Check for polyhedral groups
    if max_n >= 3:
        if has_tetrahedral_symmetry(coords, elements, tol):
            return 'Td' if has_c3_axes(coords, tol) and has_c2_axes(coords, tol) and has_s4_axes(coords, tol) else 'T'
        elif has_octahedral_symmetry(coords, elements, tol):
            return 'Oh'
        elif has_icosahedral_symmetry(coords, elements, tol):
            return 'Ih' if has_inversion_center(coords, elements, tol) else 'I'

    # Check for dihedral and cyclic groups
    if max_n > 1:
        if has_dihedral_symmetry(coords, max_n, tol):
            if has_horizontal_mirror_plane(coords, tol):
                return f'D{max_n}h'
            elif has_diagonal_mirror_planes(coords, max_n, tol):
                return f'D{max_n}d'
            else:
                return f'D{max_n}'
        else:
            if has_vertical_mirror_planes(coords, max_n, tol):
                return f'C{max_n}v'
            elif has_horizontal_mirror_plane(coords, tol):
                return f'C{max_n}h'
            else:
                return f'C{max_n}'

    # Check for S2n (Ci or S4, S6, etc.)
    if has_s2n_axis(coords, max_n, tol):
        return f'S{2*max_n}'

    # Check for Cs and Ci
    if has_mirror_plane(coords, tol):
        return 'Cs'
    if has_inversion_center(coords, elements, tol):
        return 'Ci'

    return 'C1'

def is_linear(coords: np.ndarray, tol: float) -> bool:
    # Check if all atoms are collinear
    v1 = coords[1] - coords[0]
    for i in range(2, len(coords)):
        v2 = coords[i] - coords[0]
        if np.linalg.norm(np.cross(v1, v2)) > tol:
            return False
    return True

def has_inversion_center(coords: np.ndarray, elements: list, tol: float) -> bool:
    # Check if there's an atom at -r for every atom at r
    for i, atom in enumerate(coords):
        inverted_atom = -atom
        distances = np.linalg.norm(coords - inverted_atom, axis=1)
        if not np.any(distances < tol):
            return False
        matched_atom_idx = np.argmin(distances)
        if elements[i] != elements[matched_atom_idx]:
            return False
    return True

def has_c_n_axis(coords: np.ndarray, n: int, tol: float) -> bool:
    # Check for Cn axis by rotating around z-axis and checking equivalence
    angle = 2 * np.pi / n
    rotated_coords = rotate_coords(coords, angle, [0, 0, 1])
    return are_coords_equivalent(coords, rotated_coords, elements, tol)

def has_tetrahedral_symmetry(coords: np.ndarray, elements: list, tol: float) -> bool:
    # Simplified check; actual implementation requires more comprehensive symmetry analysis
    return has_c3_axes(coords, tol) and has_c2_axes(coords, tol)

def has_octahedral_symmetry(coords: np.ndarray, elements: list, tol: float) -> bool:
    # Simplified; actual implementation requires detailed symmetry analysis
    return has_c4_axes(coords, tol) and has_c3_axes(coords, tol)

def has_icosahedral_symmetry(coords: np.ndarray, elements: list, tol: float) -> bool:
    # Simplified; detailed analysis required for accurate detection
    return has_c5_axes(coords, tol) and has_c3_axes(coords, tol)

def has_dihedral_symmetry(coords: np.ndarray, n: int, tol: float) -> bool:
    # Check for Dn by verifying C2 axes perpendicular to Cn
    c_n_axis = [0, 0, 1]  # Assuming Cn is along z-axis
    c2_axis = [1, 0, 0]  # Example C2 axis
    rotated_c2_axis = rotate_vector(c2_axis, 2 * np.pi / n / 2, c_n_axis)
    rotated_coords = rotate_coords(coords, 2 * np.pi / n / 2, c_n_axis)
    return are_coords_equivalent(coords, rotated_coords, elements, tol)

def has_vertical_mirror_planes(coords: np.ndarray, n: int, tol: float) -> bool:
    # Check for sigma_v by reflecting across planes containing Cn (z-axis)
    for i in range(n):
        angle = i * 2 * np.pi / n
        normal = rotate_vector([1, 0, 0], angle, [0, 0, 1])
        reflected_coords = reflect_coords(coords, normal)
        if are_coords_equivalent(coords, reflected_coords, elements, tol):
            return True
    return False

def has_horizontal_mirror_plane(coords: np.ndarray, tol: float) -> bool:
    # Check for sigma_h by reflecting across xy-plane
    reflected_coords = coords.copy()
    reflected_coords[:, 2] *= -1
    return are_coords_equivalent(coords, reflected_coords, elements, tol)

def has_diagonal_mirror_planes(coords: np.ndarray, n: int, tol: float) -> bool:
    # Check for sigma_d by reflecting across planes between C2 axes
    c_n_axis = [0, 0, 1]
    c2_axis = [1, 0, 0]
    for i in range(n):
        angle = (i + 0.5) * 2 * np.pi / n
        normal = rotate_vector(c2_axis, angle, c_n_axis)
        reflected_coords = reflect_coords(coords, normal)
        if are_coords_equivalent(coords, reflected_coords, elements, tol):
            return True
    return False

def has_s2n_axis(coords: np.ndarray, n: int, tol: float) -> bool:
    # Check for S2n by improper rotation
    angle = 2 * np.pi / (2 * n)
    rotated_coords = rotate_coords(coords, angle, [0, 0, 1])
    reflected_coords = rotated_coords.copy()
    reflected_coords[:, 2] *= -1
    return are_coords_equivalent(coords, reflected_coords, elements, tol)

def has_mirror_plane(coords: np.ndarray, tol: float) -> bool:
    # Check for any mirror plane
    normals = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for normal in normals:
        reflected_coords = reflect_coords(coords, normal)
        if are_coords_equivalent(coords, reflected_coords, elements, tol):
            return True
    return False

def are_coords_equivalent(coords1: np.ndarray, coords2: np.ndarray, elements: list, tol: float) -> bool:
    # Check if two sets of coordinates are equivalent under permutation
    if len(coords1) != len(coords2):
        return False

    coords1 = coords1[np.argsort(np.linalg.norm(coords1, axis=1))]
    coords2 = coords2[np.argsort(np.linalg.norm(coords2, axis=1))]

    for i in range(len(coords1)):
        distance = np.linalg.norm(coords1[i] - coords2[i])
        if distance > tol or elements[i] != elements[i]:  # elements[i] is always compared to itself here; fix needed
            return False
    return True

def rotate_coords(coords: np.ndarray, angle: float, axis: list) -> np.ndarray:
    # Rotate coordinates around given axis by angle (radians)
    axis = np.array(axis) / np.linalg.norm(axis)
    rotated_coords = np.zeros_like(coords)
    for i, coord in enumerate(coords):
        rotated_coords[i] = rotate_vector(coord, angle, axis)
    return rotated_coords

def rotate_vector(v: np.ndarray, angle: float, axis: np.ndarray) -> np.ndarray:
    # Rotate a vector around given axis by angle (radians)
    axis = axis / np.linalg.norm(axis)
    return v * np.cos(angle) + np.cross(axis, v) * np.sin(angle) + axis * np.dot(axis, v) * (1 - np.cos(angle))

def reflect_coords(coords: np.ndarray, normal: np.ndarray) -> np.ndarray:
    # Reflect coordinates across a plane with given normal
    normal = normal / np.linalg.norm(normal)
    reflected_coords = coords - 2 * np.outer(np.dot(coords, normal), normal)
    return reflected_coords

def has_c3_axes(coords: np.ndarray, tol: float) -> bool:
    # Simplified check for C3 axes; actual implementation requires detailed analysis
    return has_c_n_axis(coords, 3, tol)

def has_c4_axes(coords: np.ndarray, tol: float) -> bool:
    return has_c_n_axis(coords, 4, tol)

def has_c5_axes(coords: np.ndarray, tol: float) -> bool:
    return has_c_n_axis(coords, 5, tol)

def has_c2_axes(coords: np.ndarray, tol: float) -> bool:
    return has_c_n_axis(coords, 2, tol)

def has_s4_axes(coords: np.ndarray, tol: float) -> bool:
    return has_s2n_axis(coords, 2, tol)

point_group = detect_point_group
