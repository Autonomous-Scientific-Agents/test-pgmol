from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pgmol.symmetry import Molecule, SymmetryAnalyzer, detect_point_group

point_group = detect_point_group

__all__ = [
    "Molecule",
    "SymmetryAnalyzer",
    "detect_point_group",
    "point_group",
    "main",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pgmol",
        description="Determine the Schoenflies point group of a molecule from an XYZ file.",
    )
    parser.add_argument("xyz_files", nargs="+", type=Path, help="Path(s) to XYZ file(s)")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="Symmetry distance tolerance in angstroms (default: 1e-2)",
    )
    args = parser.parse_args(argv)

    multi = len(args.xyz_files) > 1
    for path in args.xyz_files:
        pg = detect_point_group(path, tol=args.tol)
        if multi:
            print(f"{path}: {pg}")
        else:
            print(pg)


if __name__ == "__main__":
    main(sys.argv[1:])
