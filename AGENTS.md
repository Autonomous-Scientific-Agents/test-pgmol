# AGENTS.md

## State of the repo
- Starter repo only: no code, manifest, tests, or CI yet. The only content is `README.md` and `validation/*.xyz`. Don't assume any build/test command exists; create them.
- Goal: Python package that reads an XYZ file and returns the Schoenflies point group (`H2O_C2v.xyz` -> `C2v`).
- Use `uv` (not conda/pip) with a `src/` layout when scaffolding the package.

## Validation data (the oracle)
- `validation/core_xyz/` (36 files) must all pass first; `validation/challenge_xyz/` (55 files, high-symmetry and near-symmetric) only after that.
- Expected answer is the filename suffix: `<formula>_<pointgroup>.xyz`. Parse it only in tests; the implementation must classify from coordinates alone (no hard-coded filenames/formulas).
- Labels are ASCII: `Cinfv`, `Dinfh`, single atom `Kh`. Other groups use Schoenflies case as in filenames (`C2v`, `D3h`, `Td`, `Ih`, `S4`).
- Coordinates are in angstrom. XYZ format: line 1 atom count, line 2 free-text comment, then `El x y z`. Don't parse the comment line.
- Files hold idealized coordinates, and an atom may be off-origin (e.g. `Ar_Kh.xyz`), so the code must handle translation.
- Required invariance: result must not change under rotation, translation, or atom reordering. Test this explicitly.
- Some challenge files share a formula with different groups (e.g. `C5H12_T` vs `C5H12_Td`, `C6H6_S6`), so tolerances must distinguish near-symmetric geometries.

## Workflow
- Commit early and often on a branch (README suggests `git switch -c my-try`); one commit per green step. Sign off commit messages with the model name.
- Run the validation tests after each step.
- The fuller reference spec is the `AGENTS.md` in `keceli/claude-pgmol` at commit `a6ff35f`; not copied here. Don't invent its contents.
