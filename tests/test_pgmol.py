from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from pgmol import Molecule, detect_point_group, main
from pgmol.symmetry import rotation_matrix

ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = ROOT / "validation" / "core_xyz"
CHALLENGE_DIR = ROOT / "validation" / "challenge_xyz"

CORE_FILES = sorted(CORE_DIR.glob("*.xyz"))
CHALLENGE_FILES = sorted(CHALLENGE_DIR.glob("*.xyz"))


def _expected_pg(xyz_path: Path) -> str:
    return xyz_path.stem.rsplit("_", 1)[1]


@pytest.mark.parametrize("xyz_path", CORE_FILES, ids=lambda p: p.stem)
def test_core_xyz(xyz_path: Path) -> None:
    expected = _expected_pg(xyz_path)
    # Read coordinates and pass via Molecule to ensure no filename dependency
    mol = Molecule.from_xyz(xyz_path.read_text(encoding="utf-8"))
    assert detect_point_group(mol) == expected


@pytest.mark.parametrize("xyz_path", CORE_FILES, ids=lambda p: p.stem)
def test_invariance_rotation_translation_reordering(xyz_path: Path) -> None:
    expected = _expected_pg(xyz_path)
    mol = Molecule.from_xyz(xyz_path)

    rng = np.random.default_rng(abs(hash(xyz_path.stem)) % (2**32))

    # Random 3D rotation
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = float(rng.uniform(0.1, 6.0))
    rot = rotation_matrix(axis, angle)

    # Random translation
    shift = rng.uniform(-15.0, 15.0, size=(1, 3))

    # Random permutation of atom order
    perm = rng.permutation(len(mol.elements))
    new_elements = tuple(mol.elements[i] for i in perm)
    new_coords = (mol.coords @ rot.T + shift)[perm]

    transformed_mol = Molecule(elements=new_elements, coords=new_coords)
    assert detect_point_group(transformed_mol) == expected


@pytest.mark.parametrize("xyz_path", CHALLENGE_FILES, ids=lambda p: p.stem)
def test_challenge_xyz(xyz_path: Path) -> None:
    expected = _expected_pg(xyz_path)
    mol = Molecule.from_xyz(xyz_path.read_text(encoding="utf-8"))
    assert detect_point_group(mol) == expected


@pytest.mark.parametrize("xyz_path", CHALLENGE_FILES, ids=lambda p: p.stem)
def test_challenge_invariance(xyz_path: Path) -> None:
    expected = _expected_pg(xyz_path)
    mol = Molecule.from_xyz(xyz_path)

    rng = np.random.default_rng((abs(hash(xyz_path.stem)) + 1337) % (2**32))
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = float(rng.uniform(0.1, 6.0))
    rot = rotation_matrix(axis, angle)
    shift = rng.uniform(-10.0, 10.0, size=(1, 3))
    perm = rng.permutation(len(mol.elements))

    transformed = Molecule(
        elements=tuple(mol.elements[i] for i in perm),
        coords=(mol.coords @ rot.T + shift)[perm],
    )
    assert detect_point_group(transformed) == expected


def test_cli(capsys: pytest.CaptureFixture[str]) -> None:
    h2o = CORE_DIR / "H2O_C2v.xyz"
    main([str(h2o)])
    captured = capsys.readouterr()
    assert captured.out.strip() == "C2v"
