# test-pgmol

A starting point for trying coding agents on a small scientific problem: write a Python package that reads a
molecule's XYZ file and returns its Schoenflies point group (`H2O_C2v.xyz` → `C2v`).

Clone it, pick any agent harness (Claude Code, Codex, OpenCode, Cursor, Kilo Code, ...) and any model, and compare
what you get with the reference repositories below. This repo is part of the
[Agentic Coding Tutorial](https://autonomous-scientific-agents.github.io/AgenticCodingTutorial/).

👉 **[Live Benchmark & Interactive 3D Failure-Case Explorer](https://autonomous-scientific-agents.github.io/test-pgmol/)** — inspect the benchmark tables and rotate the 3D structures of the molecules that break external symmetry libraries.

## What is here

| Folder | Contents |
|---|---|
| `validation/core_xyz/` | 36 molecules, from H₂ to ferrocene and cubane. The expected point group is in the filename: `<formula>_<pointgroup>.xyz`. |
| `validation/challenge_xyz/` | 55 harder cases (high symmetry groups, near-symmetric geometries) for after the core set passes. |
| `benchmark/` | Standardized `pgmol-bench` runner for evaluating both LLM APIs (1-shot or multi-turn) and headless CLI agents (`claude-code`, `opencode`, `codex`, `aider`) with strict workspace isolation and anonymized evaluation. |

Infinite groups use the ASCII labels `Cinfv` and `Dinfh`; a single atom is `Kh`. Coordinates are in ångströms.

## Standardized Benchmark Runner (`benchmark/`)

`benchmark/` provides an isolated evaluation harness supporting two visibility modes (`--mode core-visible` where only the 36 core molecules are mounted in the workspace, and `--mode zero-shot` where all XYZ files are held out) and evaluates against anonymized filenames (`mol_0001.xyz`) across **Core (36)**, **Challenge (55)**, and **3D Rotated + Translated + Permuted (91)**:

```bash
# 1. Score an existing repository or directory
PYTHONPATH=. uv run --isolated --project benchmark python -m benchmark score \
  --target /path/to/repo --harness opencode --model gemini-4-argon

# 2. Benchmark an LLM API (1-shot or multi-turn self-repair loop)
PYTHONPATH=. uv run --isolated --project benchmark python -m benchmark api \
  --model gpt-5 --mode core-visible --turns 3

# 3. Benchmark a headless CLI coding agent in an isolated workspace
PYTHONPATH=. uv run --isolated --project benchmark python -m benchmark agent \
  --harness claude-code --model claude-opus-4-6 --mode core-visible

# 4. Generate a Markdown leaderboard table from JSON reports
PYTHONPATH=. uv run --isolated --project benchmark python -m benchmark leaderboard
```

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

### Root-cause analysis of failure cases

Wait—are the external library and agent failures simply tolerance mismatches, or structural bugs in their algorithms? Inspecting the source code of each library against the failing fixtures shows a mix of **1 tolerance threshold issue** and **7 genuine algorithmic bugs**:

#### 1. `pymatgen` (`PointGroupAnalyzer`) — `H3_D3h.xyz` (`ValueError: min() arg is an empty sequence`)
* **Verdict:** **Tolerance threshold issue combined with an unhandled compact-molecule edge case.**
* **Code trace (`pymatgen/symmetry/analyzer.py`):**
  * `PointGroupAnalyzer.__init__` defaults to `tolerance = 0.3` Å.
  * When checking rotational symmetry (`_check_rot_sym`), it calls `_get_smallest_set_not_on_axis(axis)`, which first clusters sites by radial distance from the center of mass using `cluster_sites(self.centered_mol, self.tol)` (`analyzer.py:1290`).
  * Inside `cluster_sites` (`analyzer.py:1653`), any cluster whose average distance from the origin is `< tol` (`0.3` Å) is treated as the single **origin atom** (`origin_site = site`) and excluded from `clustered_sites`!
  * However, `cluster_sites` passes `dists = [[norm(site.coords), 0] for site in mol]` (a 2D list with a dummy zero) to `np.mean(val)` (`analyzer.py:1649`), which computes the mean over both elements `(r, 0.0)`, yielding `avg_dist = r / 2`!
  * In `H3_D3h.xyz` ($\text{H}_3^+$ equilateral triangle with $\text{H–H} \approx 0.866$ Å), each H atom sits at $r = 0.50$ Å from the origin. Because of the `np.mean([0.50, 0.0]) = 0.25` bug in `cluster_sites`, `avg_dist = 0.25 < 0.30`, so all 3 hydrogen atoms are swallowed into `origin_site`, leaving `clustered_sites` empty and causing `min(valid_sets, key=len)` (`analyzer.py:1296`) to crash with `ValueError: min() arg is an empty sequence`.
  * Lowering `tolerance <= 0.24` (e.g. `PointGroupAnalyzer(mol, tolerance=0.1)`) avoids the `0.25 < tol` trap and returns `'D3h'`.

#### 2. `pyscf` (`pyscf.symm.detect_symm`) — `C4N4Pt_C4h.xyz` (returns `C2h`) and `C5H5P_C5.xyz` (returns `C1`)
* **Verdict:** **1 algorithmic bug (`C4N4Pt_C4h.xyz`) + 1 tolerance/rounding sensitivity (`C5H5P_C5.xyz`).**
* **Code trace (`pyscf/symm/geom.py`):**
  * **`C4N4Pt_C4h.xyz` (Algorithmic bug — fails at every tolerance from `1e-7` to `0.1`):**
    * `detect_symm` (`geom.py:243`) checks whether the molecule has a higher-order principal axis ($n \ge 3$) by testing the eigenvalue degeneracies of rank-1 and rank-2 Cartesian multipole tensors (`rawsys.cartesian_tensor(1)` and `cartesian_tensor(2)`).
    * For a planar $C_{4h}$ pinwheel lying in the $xy$-plane, the rank-1 tensor has eigenvalues `[0, 1.147, 1.147]` (degeneracy 2), and the rank-2 tensor has 6 eigenvalues `[0, 0, 0, 0.648, 6.633, 7.282]`. Why aren't the non-zero rank-2 eigenvalues degenerate? Because under $C_4$, the quadratic functions $(x^2 - y^2)$ and $(xy)$ both transform as the 1D irreducible representation $B$ (not a 2D $E$ irrep!), so they mix and split into two distinct non-degenerate eigenvalues (`0.648` and `6.633`), while the three out-of-plane components $(xz, yz, z^2)$ are all strictly `0` (`degen_values == 0`).
    * Line 244 checks `elif (2 in w1_degeneracy and numpy.any(w2_degeneracy[w2_degen_values > 0] >= 2)):`. Because the only degenerate eigenvalues in `w2` are `0` (`w2_degen_values == 0`), this condition evaluates to `False`!
    * Execution falls into the `else:` branch at line 255: `n = -1 # tag as D2h and subgroup`, which completely skips `search_c_highest()` and only tests 2-fold rotations (`C2x, C2y, C2z`), misclassifying the $C_{4h}$ molecule as `C2h` regardless of tolerance.
  * **`C5H5P_C5.xyz` (Tolerance & candidate-angle sensitivity):**
    * `geom.TOLERANCE` defaults to `1e-5`. In `search_possible_rotations` (`geom.py:705–713`), `pyscf` estimates candidate rotation orders from interatomic chord angles: `nfrac = 2*pi / (pi - ang); n = int(round(nfrac)); if abs(nfrac - n) < TOLERANCE:`.
    * The coordinates in `C5H5P_C5.xyz` are given to 12 decimal places, and due to floating-point rounding in $\arccos$, `abs(nfrac - 5)` slightly exceeds `1e-5`, rejecting the $C_5$ candidate axis and returning `C1`. Increasing `pyscf.symm.geom.TOLERANCE = 1e-4` (or `1e-3`) restores `'C5'`.

#### 3. `pointgroup` (`PointGroup`) — `C2H2O2_D2h.xyz`, `Ar_Kh.xyz`, and rotated `B12H60_I.xyz`
* **Verdict:** **3 algorithmic bugs in `pointgroup/__init__.py`.**
* **Code trace (`pointgroup/__init__.py`):**
  * **`C2H2O2_D2h.xyz` (Accidental spherical top — fails regardless of tolerance):**
    * `C2H2O2_D2h.xyz` is a $D_{2h}$ molecule constructed so its three principal moments of inertia are equal ($I_x = I_y = I_z = 2/3$).
    * In `PointGroup.__init__` (`__init__.py:77`), `eig_degeneracy == 3` dispatches unconditionally to `self._spherical()`, which assumes the molecule **must** belong to a cubic/icosahedral group ($T, T_h, T_d, O, O_h, I, I_h$) and has no fallback to $D_{2h}$ or lower symmetry (`__init__.py:209–236`)!
    * Inside `self._spherical()`, `while main_axis is None:` loops over a cubed-sphere grid looking for a $C_5$, $C_4$, or $C_3$ axis. Because $D_{2h}$ has none, it prints `increase tolerance` and multiplies `self._tolerance_ang *= 1.01` in an infinite loop (217 iterations!) until the angular tolerance balloons past $\sim 35^\circ$, at which point a false $C_3$ operation is hallucinated and combined with inversion to return `'Th'`.
  * **`Ar_Kh.xyz` (Single atom missing `Kh` branch):**
    * For 1 atom at the origin, the inertia tensor is $\mathbf{0}$ (`eigenvalues = [0, 0, 0]`). Line 64 checks `if np.min(abs(self._eigenvalues)) < self._tolerance_eig: self._lineal()`, routing a single atom to the linear molecule handler (`Dinfh`) because `Kh` is never implemented.
  * **Rotated `B12H60_I.xyz` (`Exception: Error orientation I group`):**
    * In `determine_orientation_I` (`__init__.py:242–257`), after finding one $C_5$ axis on the cubed-sphere grid, `pointgroup` tilts by the theoretical icosahedral angle $\arcsin((\sqrt{5}+1)/(2\sqrt{3}))$ and scans in steps of `tolerance_ang` ($4^\circ$) for a second $C_5$ axis. Under a general 3D rotation, grid discretization error on the first $C_5$ axis plus the coarse $4^\circ$ azimuthal step misses the second $C_5$ axis and raises `Exception('Error orientation I group')`.

#### 4. `molsym` (`find_point_group`) — `C5H12_T.xyz`, `Ar_Kh.xyz`, and $O(N^4)$ runtime
* **Verdict:** **3 algorithmic/implementation bugs in `molsym/pgdetect/`.**
* **Code trace (`molsym/pgdetect/flowchart.py` & `flowchart_helper.py`):**
  * **`C5H12_T.xyz` (Unimplemented $T$ group — explicitly marked `TODO` in source):**
    * In `molsym/pgdetect/flowchart.py:67–74`, the spherical-top tetrahedral branch literally reads:
      ```python
      # Tetrahedral, TODO: No path for T
      elif n == 3:
          paxis = axes[0]
          saxis = axes[1]
          if invertable:
              pg = "Th"
          else:
              pg = "Td"
      ```
      Because `molsym` never checks for $\sigma_d$ or $S_4$ in the `n == 3` spherical-top branch, **every chiral tetrahedral ($T$) molecule is unconditionally misclassified as `Td`** at any tolerance.
  * **`Ar_Kh.xyz` (Single atom routed to linear branch):**
    * `flowchart.py:24` checks `if np.isclose(Ia_mol, 0.0, atol=mol.tol):` before checking atom count, classifying a single atom as `'D0h'` (`Dinfh`).
  * **Extreme slowness on `B12H60_I.xyz` (82.8 s) and `C60_Ih.xyz` (67.3 s):**
    * In `flowchart_helper.py:426–443` (`find_C3s_for_Ih`) and `458–485` (`find_C4s_for_Oh`), `molsym` uses triple ($O(N^3)$) and quadruple ($O(N^4)$) nested Python loops over all atoms (`for i in range(mol.natoms): for j ... for k ...`), and inside the inner loop calls `isequivalent(mol, molB)` (`symtools.py:78`), which is another $O(N^2)$ Python loop—resulting in $O(N^5)$–$O(N^6)$ scaling for $I/I_h$ and $O/O_h$ molecules!

#### 5. `pgmol-inkling` (`symmetry_finder.py`) — 42/91 (46.2%) on Rotated + Permuted
* **Verdict:** **Algorithmic orientation dependence (not a tolerance issue).**
* **Code trace (`symmetry_finder.py:116–299`):**
  * `symmetry_finder.py` seeds its axis search with hardcoded coordinate-frame directions (`[1,0,0]`, `[0,1,0]`, `[0,0,1]`, `[1,1,1]`, and golden-ratio icosahedral vectors `[0, 1, phi]`, etc. at lines 116–127), caps atom-pair direction sampling at `if count_pairs > 30: break` (line 164) crossed only with `[1,0,0]` and `[0,1,0]` (lines 179, 186), tests improper rotations **only** along `[0,0,1]`, `[0,1,0]`, `[1,0,0]`, and `[1,1,1]` (line 283), and checks cubic $C_4$/$C_3$ axes by comparing their components against `[1,0,0]` and `|x| ≈ |y| ≈ |z|` (`lines 376, 415`).
  * Consequently, it scores 91/91 when molecules are aligned with the Cartesian axes, but fails on 49/91 molecules once rotated away from the standard coordinate axes.

---

### How ASA `PGMol` (`GPT-5.6 Sol`) differs from `test-pgmol` (`my-try` / `Gemini 4 Argon`)

Both **ASA `PGMol`** and **`test-pgmol` (`my-try`)** achieve **91/91 (100%)** across `core_xyz`, `challenge_xyz`, and randomly rotated/permuted geometries using only Python and NumPy, and neither suffers from the inertia-tensor or orientation pitfalls above. However, they use fundamentally different mathematical formulations:

| Dimension | [Autonomous-Scientific-Agents/PGMol](https://github.com/Autonomous-Scientific-Agents/PGMol) (OpenCode · GPT-5.6 Sol) | [`test-pgmol` (`my-try`)](https://github.com/Autonomous-Scientific-Agents/test-pgmol/tree/my-try) (OpenCode · Gemini 4 Argon) |
|---|---|---|
| **Core formulation** | **Exhaustive $O(3)$ group discovery via orthonormal frames:** picks one non-collinear atom pair $(a_0, b_0)$ to form a source frame $F_{\text{src}}$, loops over all same-element pairs $(a_i, b_j)$ at matching radii to form target frames $F_{\text{tgt}}$, and tests $O = F_{\text{tgt}}\operatorname{diag}(1,1,\pm 1)F_{\text{src}}^T$ to build the complete matrix group. | **Shell-partitioned geometric axis generation & Schoenflies tree:** partitions atoms into radial shells per element, generates candidate axes from shell covariance eigenvectors, atom vectors, pair midpoints/differences/cross-products, and (for shells $>24$ atoms) nearest-neighbor pairs/triplets. |
| **Classification rule** | **Group order & trace/eigenvectors:** counts proper/improper matrices (`>=60` $\to I/I_h$, `>=24` $\to O/O_h$, `>=12` $\to T/T_h/T_d$); extracts rotation orders via $\arccos((\operatorname{tr}(O)-1)/2)$ and axes via `np.linalg.eig`. | **Direct geometric probing:** filters candidate orders $n$ by shell-size divisibility ($s \equiv 0, 1, 2 \pmod n$), detects distinct high-order axes for $I/O/T$, and checks perpendicular $C_2$, $\sigma_h$, $\sigma_v/\sigma_d$, and $S_{2n}$. |
| **Atom matching** | Vectorized `argmin` when nearest neighbors are unique; falls back to an exact DFS **bipartite perfect matching** (`_perfect_matching`) when tolerance spheres overlap. | Shell-by-shell bidirectional squared-distance check with a **1-atom fast-rejection probe** (`grp[0] @ op_matrix.T`) before computing the full distance matrix on shells with $>8$ atoms. |
| **Scaling & speed** | **Core:** 0.129 s (3.57 ms/mol) · **Challenge:** 2.656 s (48.28 ms/mol). On single-shell cages like $\text{C}_{60}$ or $\text{B}_{12}\text{H}_{60}$, radial filtering cannot prune the 60-atom shell, testing $60 \times 59 \times 2 = 7{,}080$ candidate frames. | **Core:** 0.094 s (2.61 ms/mol) · **Challenge:** 0.733 s (13.33 ms/mol) (**~3.6× faster** on Challenge). Large shells ($>24$ atoms) restrict pair/triplet candidates to nearest neighbors and exit early on the 1-atom probe. |
| **Default tolerance** | `0.1` Å | `0.01` Å (`1e-2`) |
| **XYZ parser & extras** | Strict 4-column parser validating against 118 IUPAC symbols; includes a static 3D WebGL browser app (`docs/`) and session trace (`agent-traces/`). | Flexible parser accepting paths, raw XYZ strings, or `Molecule` objects; includes CLI (`pgmol`) and `pytest` suite (`tests/`). |

## Share your result

Tried another harness or model? Open an issue or a pull request in this repo with a link to your repository, the
harness and model you used, and how many `core_xyz` and `challenge_xyz` files you got right. We will add it to
the table above.
