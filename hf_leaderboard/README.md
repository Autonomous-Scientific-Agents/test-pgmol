---
title: pgmol Symmetry Benchmark Leaderboard
emoji: ⬢
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: true
license: mit
short_description: Benchmark of LLMs, coding agents & symmetry libraries on 3D molecular point groups
---

# ⬢ `pgmol` Molecular Point-Group Benchmark Leaderboard

Interactive Hugging Face Space leaderboard for [`Autonomous-Scientific-Agents/test-pgmol`](https://github.com/Autonomous-Scientific-Agents/test-pgmol): evaluating **27 LLMs (zero-shot 1-shot via AmSC MAG)**, **autonomous coding agents** (`OpenCode`, `Claude Code`), and **external Python quantum-chemistry libraries** (`pymatgen`, `pyscf`, `pointgroup`, `molsym`) on 3D Schoenflies point-group determination across **182 test cases**:

- **Core (`36` molecules):** From $\text{H}_2$ ($D_{\infty h}$) and $\text{H}_2\text{O}$ ($C_{2v}$) to ferrocene ($D_{5h}$), cubane ($O_h$), and $\text{B}_{12}\text{H}_{12}^{2-}$ ($I_h$).
- **Challenge (`55` molecules):** High-order proper/improper axes ($S_4, S_6, S_8, C_{3h}, C_{4h}, D_{4d}, D_{5d}, D_{6d}, D_{8d}$), chiral polyhedra ($T, O, I$), $\text{C}_{60}$ ($I_h$), and accidental spherical tops ($\text{C}_2\text{H}_2\text{O}_2$, $D_{2h}$ with $I_x = I_y = I_z$).
- **Rotated + Permuted (`91` molecules):** Every Core and Challenge geometry subjected to a random 3D proper rotation in $\text{SO}(3)$, a random center-of-mass translation, a random atom-index permutation, and anonymized filenames (`mol_0001.xyz`).

## Running Locally

```bash
uv run --with gradio --with pandas python hf_leaderboard/app.py
```

## Deploying to Hugging Face Spaces

```bash
# Option 1: Using huggingface-cli
uvx --from huggingface_hub hf upload <your-org>/pgmol-leaderboard hf_leaderboard/ . --repo-type space

# Option 2: Using git subtree push from the repository root
git subtree push --prefix=hf_leaderboard https://huggingface.co/spaces/<your-org>/pgmol-leaderboard main
```
