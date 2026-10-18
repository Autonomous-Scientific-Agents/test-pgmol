# test-pgmol

A starting point for trying coding agents on a small scientific problem: write a Python package that reads a
molecule's XYZ file and returns its Schoenflies point group (`H2O_C2v.xyz` → `C2v`).

Clone it, pick any agent harness (Claude Code, Codex, OpenCode, Cursor, Kilo Code, ...) and any model, and compare
what you get with the reference repositories below. This repo is part of the
[Agentic Coding Tutorial](https://autonomous-scientific-agents.github.io/AgenticCodingTutorial/).

## What is here

| Folder | Contents |
|---|---|
| `validation/core_xyz/` | 36 molecules, from H₂ to ferrocene and cubane. The expected point group is in the filename: `<formula>_<pointgroup>.xyz`. |
| `validation/challenge_xyz/` | 55 harder cases (high symmetry groups, near-symmetric geometries) for after the core set passes. |

Infinite groups use the ASCII labels `Cinfv` and `Dinfh`; a single atom is `Kh`. Coordinates are in ångströms.

## Try it

```bash
git clone https://github.com/Autonomous-Scientific-Agents/test-pgmol.git
cd test-pgmol
git switch -c my-try            # keep your attempt on its own branch
claude                          # or: codex, opencode, ...
```

Then give the agent the goal and the check, for example:

> Create an installable Python package (uv, `src/` layout) that reads an XYZ file and returns its Schoenflies
> point group. Success means every file in `validation/core_xyz/` is classified correctly from its coordinates
> alone, with no hard-coded filenames or formulas, and the result does not change under rotation, translation or
> reordering of atoms. Propose a plan first and wait for my OK. Then work in small steps, run the tests after
> each one, and commit each green step.

When the core set passes, point it at `validation/challenge_xyz/` and see what breaks. The full specification
used for the reference repositories is the
[`AGENTS.md` in claude-pgmol](https://github.com/keceli/claude-pgmol/blob/a6ff35f/AGENTS.md); copy it into this repo
if you want the same contract.

Running the agent inside the
[tutorial container](https://autonomous-scientific-agents.github.io/AgenticCodingTutorial/container/) keeps it
away from the rest of your machine. Model access for Genesis Mission teams:
[AmSC Model Access Gateway](https://docs.amsc.energy.gov/model-access-gateway/) or the
[ALCF Inference Endpoints](https://docs.alcf.anl.gov/services/inference-endpoints/).

## Other pgmol repositories

The same problem solved with different harnesses and models. Compare the code, the tests, and how long it took.

| Repository | Harness · model | Web app | Agent session replay |
|---|---|---|---|
| [keceli/claude-pgmol](https://github.com/keceli/claude-pgmol) | Claude Code · Claude Opus 5 | [app](https://keceli.github.io/claude-pgmol/) | [traces](https://keceli.github.io/claude-pgmol/traces/) |
| [Autonomous-Scientific-Agents/PGMol](https://github.com/Autonomous-Scientific-Agents/PGMol) | OpenCode · GPT-5.6 Sol | [app](https://autonomous-scientific-agents.github.io/PGMol/) | [traces](https://autonomous-scientific-agents.github.io/PGMol/agent-traces/) |
| [keceli/pgmol-inkling](https://github.com/keceli/pgmol-inkling) | Claude Code · Inkling | — | — |
| [Autonomous-Scientific-Agents/test-pgmol (`my-try`)](https://github.com/Autonomous-Scientific-Agents/test-pgmol/tree/my-try) | OpenCode · Gemini 4 Argon | — | — |

`claude-pgmol` started from commit [`a6ff35f`](https://github.com/keceli/claude-pgmol/tree/a6ff35f), which has only
the specification and the core molecules; check it out to start from exactly the same place.

## Benchmark results

Evaluated on all 36 `validation/core_xyz/` and 55 `validation/challenge_xyz/` molecules, both on the original coordinates and under random 3D rigid rotations, translations, and atom permutations. For external libraries, equivalent infinite/single-atom Schoenflies notations (`C*v`/`C0v`/`Coov` → `Cinfv`, `D*h`/`D0h`/`Dooh` → `Dinfh`, `SO3`/`K` → `Kh`) are normalized in the **Normalized** column.

### Agent `pgmol` repositories

| Repository | Harness · model | Core (36) | Challenge (55) | Rotated + Permuted (91) | Core time (avg / total) | Challenge time (avg / total) |
|---|---|---:|---:|---:|---:|---:|
| [`test-pgmol` (`my-try`)](https://github.com/Autonomous-Scientific-Agents/test-pgmol/tree/my-try) | OpenCode · Gemini 4 Argon | **36/36** (100.0%) | **55/55** (100.0%) | **91/91** (100.0%) | 2.61 ms / 0.094 s | 13.33 ms / 0.733 s |
| [Autonomous-Scientific-Agents/PGMol](https://github.com/Autonomous-Scientific-Agents/PGMol) | OpenCode · GPT-5.6 Sol | **36/36** (100.0%) | **55/55** (100.0%) | **91/91** (100.0%) | 3.57 ms / 0.129 s | 48.28 ms / 2.656 s |
| [keceli/claude-pgmol](https://github.com/keceli/claude-pgmol) | Claude Code · Claude Opus 5 | **36/36** (100.0%) | **55/55** (100.0%) | **91/91** (100.0%) | 10.67 ms / 0.384 s | 118.46 ms / 6.515 s |
| [keceli/pgmol-inkling](https://github.com/keceli/pgmol-inkling) | Claude Code · Inkling | **36/36** (100.0%) | **55/55** (100.0%) | 42/91 (46.2%)¹ | 8.88 ms / 0.320 s | 56.84 ms / 3.126 s |

¹ `pgmol-inkling` tests a fixed grid of canonical/icosahedral axes and a capped sample of atom-pair directions, so it matches all 91 canonical orientations but misses high-symmetry axes under arbitrary 3D rotations.

### External Python libraries

| Library | Version | Core Exact (36) | Core Normalized (36) | Challenge Exact (55) | Challenge Normalized (55) | Rotated + Permuted (91) | Core time (avg / total) | Challenge time (avg / total) | Notes / Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| [`pymatgen`](https://pymatgen.org/) (`PointGroupAnalyzer`) | 2026.9.24 | 30/36 (83.3%) | **36/36** (100.0%) | 50/55 (90.9%) | 54/55 (98.2%) | 90/91 (98.9%) | 1.43 ms / 0.051 s | 3.69 ms / 0.203 s | Emits `C*v`/`D*h` for linear; raises `ValueError` on `H3_D3h.xyz` |
| [`pyscf`](https://pyscf.org/) (`pyscf.symm.detect_symm`) | 2.14.0 | 29/36 (80.6%) | **36/36** (100.0%) | 49/55 (89.1%) | 53/55 (96.4%) | 89/91 (97.8%) | **0.68 ms / 0.024 s** | **1.34 ms / 0.074 s** | Emits `Coov`/`Dooh`/`SO3`; misclassifies `C4N4Pt_C4h` (`C2h`) and `C5H5P_C5` (`C1`) |
| [`pointgroup`](https://pypi.org/project/pointgroup/) (`PointGroup`) | 0.4.4 | 35/36 (97.2%) | 35/36 (97.2%) | **54/55** (98.2%) | 54/55 (98.2%) | 88/91 (96.7%) | 110.43 ms / 3.976 s | 1143.13 ms / 62.872 s | Misclassifies `Ar_Kh` (`Dinfh`) and `C2H2O2_D2h` (`Th`); raises on rotated `B12H60_I` |
| [`molsym`](https://github.com/NASymmetry/MolSym) (`find_point_group`) | 1.2.0 | 29/36 (80.6%) | 35/36 (97.2%) | 50/55 (90.9%) | 54/55 (98.2%) | 89/91 (97.8%) | 90.73 ms / 3.266 s | 3341.46 ms / 183.780 s | Emits `C0v`/`D0h`; misclassifies `Ar_Kh` (`D0h`) and `C5H12_T` (`Td`); slow on `B12H60_I` (83 s) and `C60_Ih` (67 s) |

## Share your result

Tried another harness or model? Open an issue or a pull request in this repo with a link to your repository, the
harness and model you used, and how many `core_xyz` and `challenge_xyz` files you got right. We will add it to
the table above.
