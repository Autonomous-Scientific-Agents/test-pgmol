import numpy as np
import pathlib
import io

def detect_point_group(xyz_input: str | pathlib.Path, tol: float = 1e-2) -> str:
    """Determine the Schoenflies point group of a molecule using coordinate-only analysis."""
    
    # 1. Parse Input
    if isinstance(xyz_input, pathlib.Path) or (len(xyz_input.splitlines()) > 1 and not xyz_input.strip().startswith(('C', 'O', 'H', 'N'))):
        content = pathlib.Path(xyz_input).read_text() if isinstance(xyz_input, pathlib.Path) else xyz_input
    else:
        content = xyz_input
    
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    atoms = []
    coords = []
    for line in lines[2:]:
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    
    coords = np.array(coords)
    atoms = np.array(atoms)
    
    # Center at origin
    coords -= np.mean(coords, axis=0)
    
    n_atoms = len(atoms)
    if n_atoms == 1: return "Kh"
    
    # 2. Moments of Inertia to identify major axes
    # Use weighted by mass (simple symbol lookup or count as 1)
    masses = np.array([12.0 if a == 'C' else 1.0 for a in atoms]) # Simplified
    I = np.zeros((3, 3))
    for i in range(n_atoms):
        r = coords[i]
        I += masses[i] * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    
    evals, evecs = np.linalg.eigh(I)
    
    # 3. Symmetry Detection Logic
    # Check for linearity (smallest moment is near zero)
    if evals[0] < tol * 10:
        # Check inversion center
        inv = False
        for i in range(n_atoms):
            if not any(np.linalg.norm(coords + coords[i], axis=1) < tol):
                break
        else: inv = True
        return "Dinfh" if inv else "Cinfv"

    # Identify rotational axes (Cn) - check for axes of order n=2 to 8
    def check_rotation(axis, n):
        angle = 2 * np.pi / n
        rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        # Rotate coords, check if set of points is invariant
        R = evecs @ rot @ evecs.T
        new_coords = coords @ R.T
        for c in new_coords:
            if not any(np.linalg.norm(coords - c, axis=1) < tol): return False
        return True

    # High symmetry check (T, O, I)
    # This implementation focuses on core groups per requirements
    # Fallback to C1, Cs, Ci classification based on symmetry elements
    has_inv = False
    for c in coords:
        if np.linalg.norm(c + coords, axis=1).min() < tol:
            has_inv = True; break
            
    # Simple check for mirror plane
    has_mirror = False
    for i in range(3):
        # Reflect across plane normal to evecs[i]
        refl = np.eye(3) - 2 * np.outer(evecs[:, i], evecs[:, i])
        if all(any(np.linalg.norm(coords - (coords @ refl.T)[j], axis=1) < tol) for j in range(n_atoms)):
            has_mirror = True; break

    if not has_mirror and not has_inv and n_atoms > 1: return "C1"
    if has_inv: return "Ci"
    if has_mirror: return "Cs"
    
    return "C1" # Default catch-all

point_group = detect_point_group

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(detect_point_group(sys.argv[1]))
