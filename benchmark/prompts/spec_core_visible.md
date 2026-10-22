# Task: Molecular Schoenflies Point Group Detection (`pgmol`)

Create an installable Python package named `pgmol` (`src/pgmol/` layout, managed with `uv`) that reads a molecule's 3D atomic coordinates from an XYZ file and returns its Schoenflies point group symbol.

## Required Python API & CLI Contract

1. **Python API (`pgmol`):**
   Expose `detect_point_group` (and alias `point_group`) at the top level of `pgmol` (`from pgmol import detect_point_group, point_group`):
   ```python
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
           Exact ASCII Schoenflies symbol, e.g. 'C1', 'Cs', 'Ci', 'C2', 'C2v',
           'C3h', 'D2h', 'D3d', 'S4', 'T', 'Td', 'Th', 'O', 'Oh', 'I', 'Ih',
           'Cinfv', 'Dinfh', or 'Kh'.
       """
   ```

2. **Command-Line Interface:**
   Provide a CLI entry point `pgmol` (and `python -m pgmol`) that accepts a path to an `.xyz` file and prints the Schoenflies symbol to stdout.

3. **Schoenflies Notation Conventions:**
   - Single atom: `Kh`
   - Linear molecules: `Dinfh` (centrosymmetric) or `Cinfv` (non-centrosymmetric)
   - Polyhedral / cubic groups: `T`, `Th`, `Td`, `O`, `Oh`, `I`, `Ih`
   - Axial groups: `Cn`, `Cnv`, `Cnh`, `Dn`, `Dnh`, `Dnd`, `Sn` (e.g., `C2v`, `D3h`, `S4`), and `C1`, `Cs`, `Ci`.

4. **Strict Constraints & Invariance Requirements:**
   - **Dependencies:** Use only the Python standard library and `numpy` (plus `pytest` for testing). Do **not** use external chemistry/symmetry libraries (`pymatgen`, `pyscf`, `scipy`, `ase`, `molsym`, `pointgroup`, `spglib`, etc.).
   - **Coordinate-Only Detection:** Classification must depend strictly on atomic symbols and 3D Cartesian coordinates. Never inspect filenames, comment lines, or hardcode molecular formulas.
   - **3D Rigid Transformation & Permutation Invariance:** The returned point group must be invariant under arbitrary 3D rotations, translations, and atom row permutations.

## Validation Set (`validation/core_xyz/`)

The directory `validation/core_xyz/` contains 36 reference molecules named `<formula>_<pointgroup>.xyz`.
Run your tests using `uv run pytest` and verify that:
1. Every molecule in `validation/core_xyz/` is classified with 100% accuracy.
2. Classification remains 100% accurate when molecules are randomly rotated in 3D, translated, and atom-permuted.
