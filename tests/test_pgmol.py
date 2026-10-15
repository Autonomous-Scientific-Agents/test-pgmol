from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from pgmol import Molecule, SymmetryAnalyzer, detect_point_group, main, point_group
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


@pytest.mark.parametrize(
    "xyz_path",
    [
        CORE_DIR / "BrF5_C4v.xyz",
        CORE_DIR / "CCl3H_C3v.xyz",
        CORE_DIR / "C10FeH10_D5h.xyz",
        CORE_DIR / "C2H6_D3d.xyz",
        CHALLENGE_DIR / "C8H8_D4.xyz",
        CHALLENGE_DIR / "C6H6_S6.xyz",
        CHALLENGE_DIR / "C5H12_T.xyz",
        CHALLENGE_DIR / "C5H12_Td.xyz",
    ],
    ids=lambda p: p.stem,
)
def test_small_coordinate_noise_robustness(xyz_path: Path) -> None:
    expected = _expected_pg(xyz_path)
    mol = Molecule.from_xyz(xyz_path)
    rng = np.random.default_rng(42)
    for _ in range(5):
        noisy_coords = mol.coords + rng.normal(scale=1e-4, size=mol.coords.shape)
        noisy_mol = Molecule(elements=mol.elements, coords=noisy_coords)
        assert detect_point_group(noisy_mol, tol=1e-2) == expected


def test_parser_and_validation_edge_cases() -> None:
    with pytest.raises(ValueError, match="Empty XYZ"):
        Molecule.from_xyz("")
    with pytest.raises(ValueError, match="Empty XYZ"):
        Molecule.from_xyz("   \n  \n ")
    with pytest.raises(FileNotFoundError):
        Molecule.from_xyz("nonexistent_file_12345.xyz")
    with pytest.raises(FileNotFoundError):
        Molecule.from_xyz(Path("nonexistent_file_12345.xyz"))
    with pytest.raises(ValueError, match="Malformed XYZ atom line"):
        Molecule.from_xyz("1\ncomment\nH 0.0 0.0\n")
    with pytest.raises(ValueError, match="Expected 2 atoms"):
        Molecule.from_xyz("2\ncomment\nH 0.0 0.0 0.0\n")
    with pytest.raises(ValueError, match="shape"):
        Molecule(elements=("H",), coords=np.zeros((1, 2)))
    with pytest.raises(ValueError, match="Length of elements"):
        Molecule(elements=("H", "O"), coords=np.zeros((1, 3)))

    empty_mol = Molecule(elements=(), coords=np.zeros((0, 3)))
    assert len(empty_mol) == 0
    assert point_group(empty_mol) == "Kh"


def test_cli_single_and_multi(capsys: pytest.CaptureFixture[str]) -> None:
    h2o = CORE_DIR / "H2O_C2v.xyz"
    nh3 = CORE_DIR / "H3N_C3v.xyz"

    main([str(h2o)])
    captured = capsys.readouterr()
    assert captured.out.strip() == "C2v"

    main([str(h2o), str(nh3), "--tol", "0.01"])
    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().splitlines()]
    assert lines == [f"{h2o}: C2v", f"{nh3}: C3v"]

    analyzer = SymmetryAnalyzer(Molecule.from_xyz(h2o))
    assert analyzer.point_group == "C2v"


@pytest.mark.parametrize("n", [13, 14, 15, 16])
def test_high_rotation_orders_above_12(n: int) -> None:
    angles = 2.0 * np.pi * np.arange(n) / n
    coords = np.column_stack((3.0 * np.cos(angles), 3.0 * np.sin(angles), np.zeros(n)))
    mol = Molecule(elements=("C",) * n, coords=coords)
    assert detect_point_group(mol) == f"D{n}h"


def test_long_string_path_above_255_chars(tmp_path: Path) -> None:
    nested = tmp_path
    while len(str(nested / "H2O_C2v.xyz")) < 270:
        nested = nested / ("a" * 40)
    nested.mkdir(parents=True, exist_ok=True)
    target = nested / "H2O_C2v.xyz"
    target.write_text((CORE_DIR / "H2O_C2v.xyz").read_text(encoding="utf-8"), encoding="utf-8")

    assert len(str(target)) >= 270
    assert detect_point_group(str(target)) == "C2v"
