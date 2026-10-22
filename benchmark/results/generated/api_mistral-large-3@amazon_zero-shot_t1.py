"""
pgmol - Determine Schoenflies point group from XYZ coordinates.

API:
    detect_point_group(xyz_input: str | pathlib.Path, tol: float = 1e-2) -> str
    point_group = detect_point_group
"""

import math
import pathlib
import typing
from collections import defaultdict
from itertools import combinations

import numpy as np

__all__ = ["detect_point_group", "point_group"]
point_group = detect_point_group

# Constants for angle comparisons
TWO_PI = 2 * math.pi
HALF_PI = 0.5 * math.pi

def _parse_xyz(xyz_input: typing.Union[str, pathlib.Path]) -> tuple[np.ndarray, np.ndarray]:
    """Parse XYZ file or string into atomic numbers and coordinates."""
    if isinstance(xyz_input, pathlib.Path):
        with open(xyz_input, "r") as f:
            lines = f.readlines()
    elif isinstance(xyz_input, str):
        if "\n" in xyz_input.strip():
            lines = xyz_input.strip().split("\n")
        else:
            with open(xyz_input, "r") as f:
                lines = f.readlines()
    else:
        raise TypeError("xyz_input must be str or pathlib.Path")

    try:
        num_atoms = int(lines[0].strip())
    except ValueError as e:
        raise ValueError("First line must be number of atoms") from e

    if len(lines) < num_atoms + 2:
        raise ValueError("XYZ file too short")

    # Ignore comment line
    atom_data = [line.strip().split() for line in lines[2:2+num_atoms] if line.strip()]

    elements = []
    coords = []
    for data in atom_data:
        if len(data) < 4:
            raise ValueError(f"Invalid atom line: {' '.join(data)}")
        elements.append(data[0])
        coords.append([float(x) for x in data[1:4]])

    # Convert element symbols to atomic numbers
    element_to_num = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
        'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18
    }
    atomic_numbers = np.array([element_to_num.get(e, 0) for e in elements])
    if (atomic_numbers == 0).any():
        missing = set(elements) - set(element_to_num.keys())
        raise ValueError(f"Unknown element(s): {missing}")

    return atomic_numbers, np.array(coords)

def _translate_to_com(coords: np.ndarray) -> np.ndarray:
    """Translate molecule to center of mass."""
    return coords - np.mean(coords, axis=0)

def _inertia_tensor(coords: np.ndarray, atomic_numbers: np.ndarray) -> np.ndarray:
    """Compute inertia tensor."""
    masses = atomic_numbers.astype(float)
    total_mass = np.sum(masses)

    if total_mass < 1e-6:
        raise ValueError("Total mass is zero")

    # Center coordinates
    centered = coords - np.average(coords, axis=0, weights=masses)

    # Compute inertia tensor
    I = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                I[i,j] = np.sum(masses * (centered[:, (j+1)%3]**2 + centered[:, (j+2)%3]**2))
            else:
                I[i,j] = -np.sum(masses * centered[:, i] * centered[:, j])

    return I

def _principal_axes(I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize inertia tensor to get principal axes and moments."""
    eigvals, eigvecs = np.linalg.eigh(I)
    # Sort by eigenvalue (moment of inertia)
    order = np.argsort(eigvals)
    moments = eigvals[order]
    axes = eigvecs[:, order]
    return moments, axes

def _normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm

def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two vectors in radians."""
    v1 = _normalize_vector(v1)
    v2 = _normalize_vector(v2)
    dot = np.dot(v1, v2)
    # Handle floating point errors
    dot = np.clip(dot, -1.0, 1.0)
    return math.acos(dot)

def _are_vectors_parallel(v1: np.ndarray, v2: np.ndarray, tol: float) -> bool:
    """Check if two vectors are parallel within tolerance."""
    angle = _angle_between(v1, v2)
    return angle < tol or abs(angle - math.pi) < tol

def _are_vectors_perpendicular(v1: np.ndarray, v2: np.ndarray, tol: float) -> bool:
    """Check if two vectors are perpendicular within tolerance."""
    angle = _angle_between(v1, v2)
    return abs(angle - HALF_PI) < tol

def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return rotation matrix around axis by angle."""
    axis = _normalize_vector(axis)
    ux, uy, uz = axis
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c

    R = np.array([
        [t*ux*ux + c,    t*ux*uy - s*uz, t*ux*uz + s*uy],
        [t*ux*uy + s*uz, t*uy*uy + c,    t*uy*uz - s*ux],
        [t*ux*uz - s*uy, t*uy*uz + s*ux, t*uz*uz + c   ]
    ])
    return R

def _is_rotation_symmetry(molecule: tuple[np.ndarray, np.ndarray],
                         axis: np.ndarray, order: int, tol: float) -> bool:
    """Check if molecule has n-fold rotation symmetry about axis."""
    if order < 2:
        return True

    angle = TWO_PI / order
    R = _rotation_matrix(axis, angle)

    atomic_numbers, coords = molecule
    rotated_coords = np.dot(coords, R.T)

    # Check if rotated coordinates match original (accounting for atom permutations)
    # We'll find the best permutation by minimizing the sum of distances
    distances = np.linalg.norm(rotated_coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)
    # Hungarian algorithm for minimum weight matching
    # Since we're just checking for symmetry, we can use a simpler approach
    # For each atom in rotated, find closest in original
    matched = np.zeros(len(coords), dtype=bool)
    total_distance = 0.0

    for i in range(len(coords)):
        min_dist = float('inf')
        best_j = -1
        for j in range(len(coords)):
            if not matched[j] and distances[i, j] < min_dist and atomic_numbers[i] == atomic_numbers[j]:
                min_dist = distances[i, j]
                best_j = j
        if best_j != -1:
            matched[best_j] = True
            total_distance += min_dist

    return total_distance < tol * len(coords)

def _find_rotation_axes(molecule: tuple[np.ndarray, np.ndarray],
                       tol: float, max_order: int = 6) -> dict[int, list[np.ndarray]]:
    """Find all rotation axes up to given order."""
    atomic_numbers, coords = molecule
    axes = defaultdict(list)

    # Check for C2 axes first (easiest to find)
    # Between every pair of atoms
    for i, j in combinations(range(len(coords)), 2):
        if atomic_numbers[i] != atomic_numbers[j]:
            continue
        axis = coords[j] - coords[i]
        if np.linalg.norm(axis) < 1e-6:
            continue
        axis = _normalize_vector(axis)

        # Check if C2 symmetry exists
        if _is_rotation_symmetry(molecule, axis, 2, tol):
            # Avoid duplicates
            is_duplicate = False
            for existing in axes[2]:
                if _are_vectors_parallel(existing, axis, tol):
                    is_duplicate = True
                    break
            if not is_duplicate:
                axes[2].append(axis)

    # Check for higher-order axes
    # We'll try various directions including midpoints between existing C2 axes
    test_directions = []
    # Add coordinate axes
    test_directions.extend([np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])])
    # Add directions from center to atoms
    for i in range(len(coords)):
        dir_ = _normalize_vector(coords[i])
        test_directions.append(dir_)
    # Add directions from C2 axes
    for c2_axis in axes[2]:
        test_directions.append(c2_axis)

    # Remove duplicates
    unique_directions = []
    for dir_ in test_directions:
        is_duplicate = False
        for existing in unique_directions:
            if _are_vectors_parallel(dir_, existing, tol):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_directions.append(dir_)

    for order in range(3, max_order + 1):
        for dir_ in unique_directions:
            if _is_rotation_symmetry(molecule, dir_, order, tol):
                # Check if we already have this axis
                is_duplicate = False
                for existing in axes[order]:
                    if _are_vectors_parallel(existing, dir_, tol):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    axes[order].append(dir_)

    return axes

def _find_high_order_axes(molecule: tuple[np.ndarray, np.ndarray],
                         tol: float) -> dict[int, list[np.ndarray]]:
    """Find high-order rotation axes (n > 6)."""
    axes = _find_rotation_axes(molecule, tol, max_order=6)

    # If we found any axes with order > 2, check for higher orders
    if any(order > 2 for order in axes):
        # For each existing axis, check higher orders
        for order in list(axes.keys()):
            for axis in axes[order]:
                # Check if it might be infinity (linear molecule)
                if _is_rotation_symmetry(molecule, axis, 10, tol):  # High order test
                    axes[10].append(axis)
                # Check for icosahedral/higher orders
                if _is_rotation_symmetry(molecule, axis, 5, tol):
                    axes[5].append(axis)

    # Remove duplicate axes
    unique_axes = {}
    for order, axis_list in axes.items():
        unique_axes[order] = []
        for axis in axis_list:
            is_duplicate = False
            for existing_order in unique_axes:
                for existing_axis in unique_axes[existing_order]:
                    if _are_vectors_parallel(axis, existing_axis, tol):
                        is_duplicate = True
                        break
                if is_duplicate:
                    break
            if not is_duplicate:
                unique_axes[order].append(axis)

    return unique_axes

def _has_horizontal_plane(molecule: tuple[np.ndarray, np.ndarray],
                         principal_axes: np.ndarray, tol: float) -> bool:
    """Check for horizontal mirror plane (perpendicular to highest order axis)."""
    atomic_numbers, coords = molecule

    # Find highest order axis
    highest_order = 1
    main_axis = None
    for order in [6, 5, 4, 3, 2]:
        if order in _find_rotation_axes(molecule, tol):
            main_axis = _find_rotation_axes(molecule, tol)[order][0]
            highest_order = order
            break

    if highest_order < 2:
        return False

    # Normal to the putative mirror plane is the main axis
    normal = main_axis

    # Check reflection symmetry
    # Reflection matrix through plane with normal n is: I - 2*outer(n, n)
    reflection = np.eye(3) - 2 * np.outer(normal, normal)
    reflected_coords = np.dot(coords, reflection.T)

    # Check if reflected coordinates match original
    distances = np.linalg.norm(reflected_coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)
    matched = np.zeros(len(coords), dtype=bool)
    total_distance = 0.0

    for i in range(len(coords)):
        min_dist = float('inf')
        best_j = -1
        for j in range(len(coords)):
            if not matched[j] and distances[i, j] < min_dist and atomic_numbers[i] == atomic_numbers[j]:
                min_dist = distances[i, j]
                best_j = j
        if best_j != -1:
            matched[best_j] = True
            total_distance += min_dist

    return total_distance < tol * len(coords)

def _has_vertical_plane(molecule: tuple[np.ndarray, np.ndarray],
                       principal_axes: np.ndarray, tol: float) -> bool:
    """Check for vertical mirror planes (containing highest order axis)."""
    atomic_numbers, coords = molecule

    # Find highest order axis
    highest_order = 1
    main_axis = None
    axes = _find_rotation_axes(molecule, tol)
    for order in [6, 5, 4, 3, 2]:
        if order in axes:
            main_axis = axes[order][0]
            highest_order = order
            break

    if highest_order < 2:
        return False

    # Try various planes containing the main axis
    # We'll use vectors perpendicular to the main axis as plane normals
    perpendiculars = []
    # Coordinate axes
    for axis in [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]:
        if not _are_vectors_parallel(main_axis, axis, tol):
            perp = _normalize_vector(np.cross(main_axis, axis))
            perpendiculars.append(perp)
    # Vectors to atoms
    for i in range(len(coords)):
        vec = coords[i] - np.dot(coords[i], main_axis) * main_axis
        if np.linalg.norm(vec) > tol:
            perp = _normalize_vector(vec)
            perpendiculars.append(perp)

    # Remove duplicates
    unique_perps = []
    for perp in perpendiculars:
        is_duplicate = False
        for existing in unique_perps:
            if _are_vectors_parallel(perp, existing, tol):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_perps.append(perp)

    # Check each potential mirror plane
    for perp in unique_perps:
        # Plane normal is perpendicular to main_axis and perp
        normal = _normalize_vector(np.cross(main_axis, perp))
        reflection = np.eye(3) - 2 * np.outer(normal, normal)
        reflected_coords = np.dot(coords, reflection.T)

        distances = np.linalg.norm(reflected_coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)
        matched = np.zeros(len(coords), dtype=bool)
        total_distance = 0.0

        for i in range(len(coords)):
            min_dist = float('inf')
            best_j = -1
            for j in range(len(coords)):
                if not matched[j] and distances[i, j] < min_dist and atomic_numbers[i] == atomic_numbers[j]:
                    min_dist = distances[i, j]
                    best_j = j
            if best_j != -1:
                matched[best_j] = True
                total_distance += min_dist

        if total_distance < tol * len(coords):
            return True

    return False

def _has_inversion_center(molecule: tuple[np.ndarray, np.ndarray], tol: float) -> bool:
    """Check for inversion center."""
    atomic_numbers, coords = molecule
    center = np.mean(coords, axis=0)
    inverted_coords = 2 * center - coords

    distances = np.linalg.norm(inverted_coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)
    matched = np.zeros(len(coords), dtype=bool)
    total_distance = 0.0

    for i in range(len(coords)):
        min_dist = float('inf')
        best_j = -1
        for j in range(len(coords)):
            if not matched[j] and distances[i, j] < min_dist and atomic_numbers[i] == atomic_numbers[j]:
                min_dist = distances[i, j]
                best_j = j
        if best_j != -1:
            matched[best_j] = True
            total_distance += min_dist

    return total_distance < tol * len(coords)

def _is_linear(molecule: tuple[np.ndarray, np.ndarray], tol: float) -> bool:
    """Check if molecule is linear."""
    atomic_numbers, coords = molecule

    if len(atomic_numbers) <= 2:
        return True

    # Check if all atoms are colinear
    # Take first two atoms as reference
    ref1, ref2 = coords[0], coords[1]
    axis = _normalize_vector(ref2 - ref1)

    for i in range(2, len(coords)):
        vec = coords[i] - ref1
        proj = np.dot(vec, axis) * axis
        if np.linalg.norm(vec - proj) > tol:
            return False

    # Check for rotation symmetry around the axis (should be infinite)
    return _is_rotation_symmetry(molecule, axis, 10, tol)

def _determine_linear_point_group(molecule: tuple[np.ndarray, np.ndarray], tol: float) -> str:
    """Determine point group for linear molecules."""
    if _has_inversion_center(molecule, tol):
        return "Dinfh"
    else:
        return "Cinfv"

def _determine_cubic_point_group(molecule: tuple[np.ndarray, np.ndarray], tol: float) -> str:
    """Determine cubic/icosahedral point group."""
    axes = _find_high_order_axes(molecule, tol)

    # Check for icosahedral groups
    if 5 in axes and len(axes[5]) >= 6:  # Icosahedron has 6 C5 axes
        if _has_inversion_center(molecule, tol):
            return "Ih"
        else:
            return "I"

    # Check for octahedral groups
    if 4 in axes and len(axes[4]) >= 3:  # Octahedron has 3 C4 axes
        if _has_inversion_center(molecule, tol):
            return "Oh"
        else:
            return "O"

    # Check for tetrahedral groups
    if 3 in axes and len(axes[3]) >= 4:  # Tetrahedron has 4 C3 axes
        if _has_inversion_center(molecule, tol):
            return "Th"
        elif _has_horizontal_plane(molecule, None, tol):
            return "Td"
        else:
            return "T"

    # If we get here, it might be a lower symmetry group
    return None

def _determine_axial_point_group(molecule: tuple[np.ndarray, np.ndarray],
                                principal_axes: np.ndarray, tol: float) -> str:
    """Determine point group for axial molecules."""
    axes = _find_high_order_axes(molecule, tol)
    highest_order = 1
    main_axis = None

    # Find highest order rotation axis
    for order in [6, 5, 4, 3, 2]:
        if order in axes:
            highest_order = order
            main_axis = axes[order][0]
            break

    if highest_order < 2:
        # No proper rotation axes other than C1
        if _has_inversion_center(molecule, tol):
            return "Ci"
        elif _has_horizontal_plane(molecule, principal_axes, tol):
            return "Cs"
        else:
            return "C1"

    # Check for S2n axes (improper rotations)
    n = highest_order
    has_s2n = False
    if n % 2 == 0:
        s2n_order = n
        # S2n = rotation by 2π/(2n) followed by reflection
        angle = TWO_PI / (2 * s2n_order)
        R_rot = _rotation_matrix(main_axis, angle)
        R_refl = np.eye(3) - 2 * np.outer(main_axis, main_axis)
        R_s2n = np.dot(R_refl, R_rot)
        rotated_coords = np.dot(molecule[1], R_s2n.T)
        distances = np.linalg.norm(rotated_coords[:, np.newaxis, :] - molecule[1][np.newaxis, :, :], axis=2)
        matched = np.zeros(len(molecule[1]), dtype=bool)
        total_distance = 0.0

        for i in range(len(rotated_coords)):
            min_dist = float('inf')
            best_j = -1
            for j in range(len(molecule[1])):
                if not matched[j] and distances[i, j] < min_dist and molecule[0][i] == molecule[0][j]:
                    min_dist = distances[i, j]
                    best_j = j
            if best_j != -1:
                matched[best_j] = True
                total_distance += min_dist

        if total_distance < tol * len(molecule[1]):
            has_s2n = True

    # Check for Cn group
    if len(axes.get(highest_order, [])) == 1:
        if has_s2n:
            return f"S{2*n}"
        # Check for horizontal mirror plane
        if _has_horizontal_plane(molecule, principal_axes, tol):
            return f"C{highest_order}h"
        else:
            return f"C{highest_order}"

    # Check for Dn group (n C2 axes perpendicular to main axis)
    c2_axes = axes.get(2, [])
    perpendicular_c2 = []
    for axis in c2_axes:
        if _are_vectors_perpendicular(axis, main_axis, tol):
            perpendicular_c2.append(axis)

    if len(perpendicular_c2) >= highest_order:
        # Check for horizontal mirror plane
        if _has_horizontal_plane(molecule, principal_axes, tol):
            return f"D{highest_order}h"
        # Check for vertical mirror planes
        elif _has_vertical_plane(molecule, principal_axes, tol):
            return f"D{highest_order}d"
        else:
            return f"D{highest_order}"

    # Check for Cnv group (vertical mirror planes)
    if _has_vertical_plane(molecule, principal_axes, tol):
        return f"C{highest_order}v"

    # Fall back to Cn if we didn't find other symmetry
    return f"C{highest_order}"

def detect_point_group(xyz_input: typing.Union[str, pathlib.Path], tol: float = 1e-2) -> str:
    """Determine the Schoenflies point group of a molecule."""
    # Parse XYZ file
    atomic_numbers, coords = _parse_xyz(xyz_input)

    # Handle single atom case
    if len(atomic_numbers) == 1:
        return "Kh"

    # Translate to center of mass
    coords = _translate_to_com(coords)

    # Form molecule tuple
    molecule = (atomic_numbers, coords)

    # Check for linear molecules first
    if _is_linear(molecule, tol):
        return _determine_linear_point_group(molecule, tol)

    # Check for cubic/icosahedral groups
    cubic_group = _determine_cubic_point_group(molecule, tol)
    if cubic_group is not None:
        return cubic_group

    # Determine principal axes
    I = _inertia_tensor(coords, atomic_numbers)
    moments, principal_axes = _principal_axes(I)

    # Determine axial point group
    return _determine_axial_point_group(molecule, principal_axes, tol)
