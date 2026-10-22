You are an expert computational chemist and Python developer.
Your task is to write a complete, self-contained Python module (`pgmol/__init__.py`) that determines the Schoenflies point group of any molecule from its XYZ coordinates using only the Python standard library and `numpy`.

Requirements:
1. Implement `detect_point_group(xyz_input: str | pathlib.Path, tol: float = 1e-2) -> str` and alias `point_group = detect_point_group`.
2. `xyz_input` may be a `pathlib.Path`, a file path string, or a raw multiline XYZ string:
   Line 1: number of atoms N
   Line 2: comment line (ignore completely)
   Lines 3..N+2: `<element_symbol> <x> <y> <z>`
3. Return the exact ASCII Schoenflies point group symbol:
   - Single atom: `'Kh'`
   - Linear: `'Dinfh'` or `'Cinfv'`
   - Polyhedral: `'T'`, `'Th'`, `'Td'`, `'O'`, `'Oh'`, `'I'`, `'Ih'`
   - Low/Axial: `'C1'`, `'Cs'`, `'Ci'`, `'Cn'`, `'Cnv'`, `'Cnh'`, `'Dn'`, `'Dnh'`, `'Dnd'`, `'Sn'`
4. Must be 100% invariant to arbitrary 3D rotations, translations, and atom permutations, and must handle accidental spherical tops (e.g., $D_{2h}$ with $I_x = I_y = I_z$), chiral vs achiral polyhedral groups ($T$ vs $T_d$/$T_h$, $I$ vs $I_h$), and planar $C_{nh}$ pinwheels.
5. Output ONLY a single fenced Python code block (```python ... ```) containing the complete module implementation.
