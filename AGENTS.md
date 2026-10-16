# AGENTS.md

## Commands & Package Structure
- Use `uv` (not conda/pip).
- Run all tests: `uv run pytest`
- Run a single test or subset: `uv run pytest -k "core_xyz"` or `uv run pytest tests/test_pgmol.py -k "H2O"`
- CLI entrypoint: `uv run pgmol validation/core_xyz/H2O_C2v.xyz` (outputs `C2v`)
- Library entrypoints (`src/pgmol/`): `pgmol.detect_point_group(path_or_xyz_or_molecule, tol=1e-2)` (aliased as `pgmol.point_group`), `pgmol.Molecule.from_xyz(...)`, and `pgmol.SymmetryAnalyzer`.

## Validation data (the oracle)
- `validation/core_xyz/` (36 files) and `validation/challenge_xyz/` (55 files, high-symmetry and near-symmetric).
- Expected answer is the filename suffix: `<formula>_<pointgroup>.xyz`. Parse it only in tests; the implementation must classify from coordinates alone (no hard-coded filenames/formulas).
- Labels are ASCII: `Cinfv`, `Dinfh`, single atom `Kh`. Other groups use Schoenflies case as in filenames (`C2v`, `D3h`, `Td`, `Ih`, `S4`).
- Coordinates are in angstrom. XYZ format: line 1 atom count, line 2 free-text comment, then `El x y z`. Don't parse the comment line.
- Files hold idealized coordinates, and an atom may be off-origin (e.g. `Ar_Kh.xyz`), so the code must handle translation.
- Required invariance: result must not change under rotation, translation, or atom reordering. Tested explicitly in `tests/test_pgmol.py`.
- Some challenge files share a formula with different groups (e.g. `C5H12_T` vs `C5H12_Td`, `C6H6_S6`), so tolerances must distinguish near-symmetric geometries.

## Workflow
- Commit early and often on a branch (`my-try`); one commit per green step. Sign off commit messages with the model name.
- Run `uv run pytest` after each step.
