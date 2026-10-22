import numpy as np
from pathlib import Path

def detect_point_group(xyz_input: str | Path, tol: float = 1e-2) -> str:
    point_group = detect_point_group
    if isinstance(xyz_input, str):
        if Path(xyz_input).is_file():
            with open(xyz_input, 'r') as file:
                lines = file.readlines()
        else:
            lines = xyz_input.splitlines()
    elif isinstance(xyz_input, Path):
        with open(xyz_input, 'r') as file:
            lines = file.readlines()
    else:
        raise ValueError("xyz_input must be a string or pathlib.Path")

    n_atoms = int(lines[0].strip())
    if n_atoms == 1:
        return 'Kh'

    coordinates = np.array([[float(x) for x in line.split()[1:]] for line in lines[2:]])
    centroid = np.mean(coordinates, axis=0)
    coordinates -= centroid

    # Check for linearity
    vectors = coordinates[1:] - coordinates[:-1]
    cross_products = np.cross(vectors[:-1], vectors[1:])
    if np.allclose(cross_products, 0, atol=tol):
        # Linear molecule
        if np.linalg.norm(coordinates.sum(axis=0)) < tol:
            return 'Dinfh'
        else:
            return 'Cinfv'

    # Check for polyhedral groups
    distances = np.linalg.norm(coordinates, axis=1)
    unique_distances = np.unique(distances)
    if len(unique_distances) == 1:
        # All distances are equal, check for polyhedral groups
        if n_atoms == 4:
            return 'T'
        elif n_atoms == 6:
            return 'Oh'
        elif n_atoms == 12:
            return 'I'

    # Check for axial groups
    moments_of_inertia = np.array([np.sum(np.linalg.norm(coordinates, axis=1)**2) for _ in range(3)])
    sorted_moments = np.sort(moments_of_inertia)
    if np.isclose(sorted_moments[0], sorted_moments[1], atol=tol):
        # Planar molecule
        if np.isclose(sorted_moments[1], sorted_moments[2], atol=tol):
            return 'Dnh'
        else:
            return 'Cnh'
    elif np.isclose(sorted_moments[0], sorted_moments[2], atol=tol):
        return 'Dn'
    else:
        return 'Cn'

# Example usage:
# print(detect_point_group("path_to_xyz_file.xyz"))
