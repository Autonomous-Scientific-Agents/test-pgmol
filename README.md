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
| [keceli/pgmol-argon-oc](https://github.com/keceli/pgmol-argon-oc) | OpenCode · Gemini 4 Argon | — | — |

`claude-pgmol` started from commit [`a6ff35f`](https://github.com/keceli/claude-pgmol/tree/a6ff35f), which has only
the specification and the core molecules; check it out to start from exactly the same place.

## Share your result

Tried another harness or model? Open an issue or a pull request in this repo with a link to your repository, the
harness and model you used, and how many `core_xyz` and `challenge_xyz` files you got right. We will add it to
the table above.
