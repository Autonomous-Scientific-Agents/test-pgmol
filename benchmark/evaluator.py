"""Standardized evaluator and scoring engine for pgmol benchmark runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_DIR = REPO_ROOT / "validation"

NORMALIZE_MAP: dict[str, str] = {
    "C*v": "Cinfv",
    "C0v": "Cinfv",
    "Coov": "Cinfv",
    "C∞v": "Cinfv",
    "D*h": "Dinfh",
    "D0h": "Dinfh",
    "Dooh": "Dinfh",
    "D∞h": "Dinfh",
    "SO3": "Kh",
    "SO(3)": "Kh",
    "K": "Kh",
    "R3": "Kh",
}


def normalize_point_group(symbol: str | None) -> str:
    """Normalize equivalent Schoenflies point group symbols."""
    if not symbol:
        return ""
    cleaned = symbol.strip()
    return NORMALIZE_MAP.get(cleaned, cleaned)


def expected_point_group_from_filename(path: Path) -> str:
    """Extract expected point group from `<formula>_<pointgroup>.xyz`."""
    stem = path.stem
    if "_" not in stem:
        raise ValueError(f"Cannot extract expected point group from filename: {path.name}")
    return stem.rsplit("_", 1)[1]


def parse_xyz_text(text: str) -> tuple[list[str], np.ndarray]:
    """Parse an XYZ string into element symbols and (N, 3) float coordinates."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty XYZ content")
    natoms = int(lines[0])
    # In standard XYZ, line 1 is atom count, line 2 is comment (which might be blank in raw text)
    raw_lines = text.splitlines()
    atom_lines = [line.strip() for line in raw_lines[2 : 2 + natoms] if line.strip()]
    if len(atom_lines) != natoms:
        atom_lines = lines[2 : 2 + natoms] if len(lines) >= 2 + natoms else lines[1 : 1 + natoms]
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in atom_lines:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.asarray(coords, dtype=float)


def format_xyz_text(symbols: list[str], coords: np.ndarray, comment: str = "anonymized") -> str:
    """Serialize symbols and (N, 3) coordinates into a clean XYZ text string."""
    out = [str(len(symbols)), comment]
    for sym, (x, y, z) in zip(symbols, coords, strict=True):
        out.append(f"{sym:<4s} {x:18.12f} {y:18.12f} {z:18.12f}")
    return "\n".join(out) + "\n"


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Generate a uniform random 3D proper rotation matrix in SO(3) (Haar measure)."""
    u1, u2, u3 = rng.uniform(0.0, 1.0, size=3)
    q0 = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    q1 = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    q2 = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    q3 = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    return np.array(
        [
            [1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2)],
        ],
        dtype=float,
    )


def transform_xyz_text(text: str, rng: np.random.Generator) -> str:
    """Apply a random 3D rigid rotation, translation, and atom row permutation."""
    symbols, coords = parse_xyz_text(text)
    n = len(symbols)
    rot = random_rotation_matrix(rng)
    translation = rng.uniform(-5.0, 5.0, size=(1, 3))
    transformed = coords @ rot.T + translation
    perm = rng.permutation(n)
    perm_symbols = [symbols[i] for i in perm]
    perm_coords = transformed[perm]
    return format_xyz_text(perm_symbols, perm_coords, comment="rotated_translated_permuted")


@dataclass
class MoleculeCaseResult:
    name: str
    split: str
    expected: str
    predicted: str | None
    normalized_predicted: str
    exact_match: bool
    normalized_match: bool
    elapsed_ms: float
    error: str | None = None


@dataclass
class SplitMetrics:
    split: str
    total: int
    exact_correct: int
    normalized_correct: int
    exact_accuracy: float
    normalized_accuracy: float
    avg_ms: float
    p95_ms: float
    total_s: float
    failures: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CodeStats:
    python_files: int = 0
    lines_of_code: int = 0
    forbidden_imports: list[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    target: str
    core: SplitMetrics
    challenge: SplitMetrics
    rotated_permuted: SplitMetrics
    overall_exact_accuracy: float
    overall_normalized_accuracy: float
    code_stats: CodeStats
    cases: list[MoleculeCaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FORBIDDEN_LIBRARIES = {
    "pymatgen",
    "pyscf",
    "scipy",
    "ase",
    "molsym",
    "pointgroup",
    "spglib",
    "rdkit",
    "openbabel",
    "sympy",
}


def analyze_code_directory(target_dir: Path) -> CodeStats:
    """Count Python LOC (excluding tests/ and .venv/) and check for forbidden imports."""
    py_files: list[Path] = []
    for base_name in ("src", "pgmol"):
        candidate = target_dir / base_name
        if candidate.is_dir():
            py_files.extend(sorted(candidate.rglob("*.py")))
    if not py_files:
        for p in sorted(target_dir.glob("*.py")):
            if not p.name.startswith("test_"):
                py_files.append(p)

    loc = 0
    forbidden: set[str] = set()
    for path in py_files:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in content.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                loc += 1
            if stripped.startswith(("import ", "from ")):
                tokens = stripped.replace(",", " ").replace(".", " ").split()
                for lib in FORBIDDEN_LIBRARIES:
                    if lib in tokens:
                        forbidden.add(lib)

    return CodeStats(
        python_files=len(py_files),
        lines_of_code=loc,
        forbidden_imports=sorted(forbidden),
    )


_WORKER_SCRIPT = r"""
import importlib
import json
from pathlib import Path
import sys
import time

def load_detector(target_dir: Path):
    for candidate in (target_dir / "src", target_dir):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    mod = importlib.import_module("pgmol")
    for fn_name in ("detect_point_group", "point_group", "get_point_group", "find_point_group"):
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            return fn

    for sub_name in ("pgmol.symmetry", "pgmol.pointgroup", "pgmol.detector", "pgmol.core"):
        try:
            sub = importlib.import_module(sub_name)
        except Exception:
            continue
        for fn_name in ("detect_point_group", "point_group", "get_point_group", "find_point_group"):
            fn = getattr(sub, fn_name, None)
            if callable(fn):
                return fn

    raise AttributeError("Could not find detect_point_group or point_group in pgmol")

def main():
    payload = json.loads(sys.stdin.read())
    target_dir = Path(payload["target_dir"])
    cases = payload["cases"]
    try:
        detector = load_detector(target_dir)
    except Exception as exc:
        print(json.dumps({"fatal_error": f"{type(exc).__name__}: {exc}"}))
        return

    results = []
    for item in cases:
        xyz_path = Path(item["xyz_path"])
        t0 = time.perf_counter()
        pred = None
        err = None
        try:
            res = detector(xyz_path)
            if isinstance(res, str):
                pred = res
            elif hasattr(res, "point_group"):
                pred = str(res.point_group)
            elif hasattr(res, "symbol"):
                pred = str(res.symbol)
            else:
                pred = str(res)
        except Exception as exc:
            try:
                res = detector(str(xyz_path))
                pred = res if isinstance(res, str) else str(res)
            except Exception as exc2:
                err = f"{type(exc2).__name__}: {exc2}"
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        results.append({
            "id": item["id"],
            "predicted": pred,
            "elapsed_ms": elapsed_ms,
            "error": err,
        })
    print(json.dumps({"results": results}))

if __name__ == "__main__":
    main()
"""


class BenchmarkEvaluator:
    """Evaluates a pgmol implementation directory on Core, Challenge, and Rotated+Permuted splits."""

    def __init__(
        self,
        validation_dir: Path = VALIDATION_DIR,
        seed: int = 20261022,
        timeout_s: float = 180.0,
    ) -> None:
        self.validation_dir = validation_dir
        self.seed = seed
        self.timeout_s = timeout_s

    def evaluate_directory(
        self,
        target_dir: Path,
        splits: tuple[str, ...] = ("core", "challenge", "rotated_permuted"),
    ) -> EvaluationResult:
        """
        Evaluate a package directory against anonymized XYZ files in an isolated subprocess.

        Note: All XYZ files passed to the target code are renamed to `mol_0001.xyz`,
        `mol_0002.xyz`, etc. and have their comment lines stripped so that solutions
        cannot cheat by inspecting filenames or XYZ headers.
        """
        target_dir = target_dir.resolve()
        code_stats = analyze_code_directory(target_dir)

        core_files = sorted((self.validation_dir / "core_xyz").glob("*.xyz"))
        challenge_files = sorted((self.validation_dir / "challenge_xyz").glob("*.xyz"))

        rng = np.random.default_rng(self.seed)

        with tempfile.TemporaryDirectory(prefix="pgmol_eval_") as tmp:
            tmp_path = Path(tmp)
            worker_inputs: list[dict[str, str]] = []
            case_meta: list[tuple[str, str, str]] = []  # (name, split, expected)

            idx = 0
            if "core" in splits:
                for path in core_files:
                    idx += 1
                    expected = expected_point_group_from_filename(path)
                    symbols, coords = parse_xyz_text(path.read_text(encoding="utf-8"))
                    anon_path = tmp_path / f"mol_{idx:04d}.xyz"
                    anon_path.write_text(
                        format_xyz_text(symbols, coords, comment=f"molecule_{idx:04d}"),
                        encoding="utf-8",
                    )
                    worker_inputs.append({"id": str(idx), "xyz_path": str(anon_path)})
                    case_meta.append((path.name, "core", expected))

            if "challenge" in splits:
                for path in challenge_files:
                    idx += 1
                    expected = expected_point_group_from_filename(path)
                    symbols, coords = parse_xyz_text(path.read_text(encoding="utf-8"))
                    anon_path = tmp_path / f"mol_{idx:04d}.xyz"
                    anon_path.write_text(
                        format_xyz_text(symbols, coords, comment=f"molecule_{idx:04d}"),
                        encoding="utf-8",
                    )
                    worker_inputs.append({"id": str(idx), "xyz_path": str(anon_path)})
                    case_meta.append((path.name, "challenge", expected))

            if "rotated_permuted" in splits:
                for path in core_files + challenge_files:
                    idx += 1
                    expected = expected_point_group_from_filename(path)
                    transformed = transform_xyz_text(path.read_text(encoding="utf-8"), rng)
                    anon_path = tmp_path / f"mol_{idx:04d}.xyz"
                    anon_path.write_text(transformed, encoding="utf-8")
                    worker_inputs.append({"id": str(idx), "xyz_path": str(anon_path)})
                    case_meta.append((path.name, "rotated_permuted", expected))

            payload = json.dumps({"target_dir": str(target_dir), "cases": worker_inputs})
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", _WORKER_SCRIPT],
                    input=payload,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                proc = None

        raw_by_id: dict[str, dict[str, Any]] = {}
        fatal_error: str | None = None
        if proc is None:
            fatal_error = f"TimeoutExpired: evaluation exceeded {self.timeout_s}s"
        elif proc.returncode != 0:
            stderr_tail = (proc.stderr or "").strip().splitlines()[-5:]
            fatal_error = "SubprocessError: " + " | ".join(stderr_tail)
        else:
            try:
                stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
                parsed = json.loads(stdout_lines[-1])
                if "fatal_error" in parsed:
                    fatal_error = parsed["fatal_error"]
                else:
                    for item in parsed.get("results", []):
                        raw_by_id[item["id"]] = item
            except Exception as exc:
                fatal_error = f"OutputParseError: {exc}"

        all_cases: list[MoleculeCaseResult] = []
        for i, (name, split, expected) in enumerate(case_meta, start=1):
            item = raw_by_id.get(str(i))
            if item is None:
                pred = None
                elapsed_ms = 0.0
                err = fatal_error or "MissingResult"
            else:
                pred = item.get("predicted")
                elapsed_ms = float(item.get("elapsed_ms", 0.0))
                err = item.get("error")

            norm_pred = normalize_point_group(pred)
            exact_ok = pred is not None and pred.strip() == expected
            norm_ok = norm_pred == expected
            all_cases.append(
                MoleculeCaseResult(
                    name=name,
                    split=split,
                    expected=expected,
                    predicted=pred.strip() if isinstance(pred, str) else None,
                    normalized_predicted=norm_pred,
                    exact_match=exact_ok,
                    normalized_match=norm_ok,
                    elapsed_ms=elapsed_ms,
                    error=err,
                )
            )

        core_metrics = self._summarize_split("core", all_cases)
        challenge_metrics = self._summarize_split("challenge", all_cases)
        rot_metrics = self._summarize_split("rotated_permuted", all_cases)

        total_cases = len(all_cases)
        overall_exact = (
            sum(1 for c in all_cases if c.exact_match) / total_cases if total_cases else 0.0
        )
        overall_norm = (
            sum(1 for c in all_cases if c.normalized_match) / total_cases if total_cases else 0.0
        )

        return EvaluationResult(
            target=str(target_dir),
            core=core_metrics,
            challenge=challenge_metrics,
            rotated_permuted=rot_metrics,
            overall_exact_accuracy=overall_exact,
            overall_normalized_accuracy=overall_norm,
            code_stats=code_stats,
            cases=all_cases,
        )

    @staticmethod
    def _summarize_split(split: str, cases: list[MoleculeCaseResult]) -> SplitMetrics:
        split_cases = [c for c in cases if c.split == split]
        total = len(split_cases)
        if total == 0:
            return SplitMetrics(
                split=split,
                total=0,
                exact_correct=0,
                normalized_correct=0,
                exact_accuracy=0.0,
                normalized_accuracy=0.0,
                avg_ms=0.0,
                p95_ms=0.0,
                total_s=0.0,
                failures=[],
            )

        exact_correct = sum(1 for c in split_cases if c.exact_match)
        norm_correct = sum(1 for c in split_cases if c.normalized_match)
        times = [c.elapsed_ms for c in split_cases]
        total_s = sum(times) / 1000.0
        avg_ms = sum(times) / total
        p95_ms = float(np.percentile(times, 95))

        failures = [
            {
                "name": c.name,
                "expected": c.expected,
                "predicted": c.predicted,
                "error": c.error,
            }
            for c in split_cases
            if not c.exact_match
        ]

        return SplitMetrics(
            split=split,
            total=total,
            exact_correct=exact_correct,
            normalized_correct=norm_correct,
            exact_accuracy=exact_correct / total,
            normalized_accuracy=norm_correct / total,
            avg_ms=avg_ms,
            p95_ms=p95_ms,
            total_s=total_s,
            failures=failures,
        )
