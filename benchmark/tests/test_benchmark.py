"""Unit and integration tests for the pgmol benchmark suite."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from benchmark.evaluator import (
    BenchmarkEvaluator,
    analyze_code_directory,
    expected_point_group_from_filename,
    normalize_point_group,
    parse_xyz_text,
    random_rotation_matrix,
    transform_xyz_text,
)
from benchmark.report import format_single_report_summary, generate_markdown_leaderboard, save_run_report
from benchmark.runners.agent_runner import build_agent_command, parse_agent_telemetry
from benchmark.runners.api_runner import LLMAPIRunner, extract_python_code
from benchmark.workspace import WorkspaceManager


def test_workspace_isolation_core_visible(tmp_path: Path) -> None:
    wm = WorkspaceManager(base_temp_dir=tmp_path)
    ctx = wm.create_workspace(mode="core-visible", run_id="test_core", keep_workdir=False)
    try:
        assert (ctx.workdir / "AGENTS.md").is_file()
        assert (ctx.workdir / "validation" / "core_xyz").is_dir()
        core_count = len(list((ctx.workdir / "validation" / "core_xyz").glob("*.xyz")))
        assert core_count == 36
        # Must NEVER expose challenge_xyz to the agent/LLM during generation
        assert not (ctx.workdir / "validation" / "challenge_xyz").exists()
    finally:
        ctx.cleanup()
    assert not ctx.workdir.exists()


def test_workspace_isolation_zero_shot(tmp_path: Path) -> None:
    wm = WorkspaceManager(base_temp_dir=tmp_path)
    ctx = wm.create_workspace(mode="zero-shot", run_id="test_zs", keep_workdir=False)
    try:
        assert (ctx.workdir / "AGENTS.md").is_file()
        assert not (ctx.workdir / "validation").exists()
    finally:
        ctx.cleanup()


def test_rotation_and_transformation_invariants() -> None:
    rng = np.random.default_rng(42)
    rot = random_rotation_matrix(rng)
    # R^T R == I and det(R) == +1
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-12)

    xyz = "3\nwater\nO 0.0 0.0 0.1\nH 0.0 0.7 -0.4\nH 0.0 -0.7 -0.4\n"
    syms1, coords1 = parse_xyz_text(xyz)
    transformed = transform_xyz_text(xyz, rng)
    syms2, coords2 = parse_xyz_text(transformed)
    assert sorted(syms1) == sorted(syms2)
    # Pairwise distances must be preserved under rigid 3D rotation + translation + permutation
    d1 = np.sort(np.linalg.norm(coords1[:, None, :] - coords1[None, :, :], axis=-1).ravel())
    d2 = np.sort(np.linalg.norm(coords2[:, None, :] - coords2[None, :, :], axis=-1).ravel())
    np.testing.assert_allclose(d1, d2, atol=1e-10)


def test_normalize_and_filename_parsing() -> None:
    assert normalize_point_group("C*v") == "Cinfv"
    assert normalize_point_group("Dooh") == "Dinfh"
    assert normalize_point_group("SO3") == "Kh"
    assert normalize_point_group("Td") == "Td"
    assert expected_point_group_from_filename(Path("H2O_C2v.xyz")) == "C2v"


def test_anti_cheat_filename_anonymization(tmp_path: Path) -> None:
    """Verify that an implementation trying to read the point group from the filename gets 0/36."""
    src_dir = tmp_path / "src" / "pgmol"
    src_dir.mkdir(parents=True)
    # Cheating implementation that inspects filename stem `formula_pg.xyz`
    (src_dir / "__init__.py").write_text(
        "from pathlib import Path\n"
        "def detect_point_group(p, tol=1e-2):\n"
        "    return Path(p).stem.split('_')[-1]\n",
        encoding="utf-8",
    )
    evaluator = BenchmarkEvaluator()
    res = evaluator.evaluate_directory(tmp_path, splits=("core",))
    assert res.core.total == 36
    assert res.core.exact_correct == 0


def test_forbidden_import_detection(tmp_path: Path) -> None:
    src_dir = tmp_path / "src" / "pgmol"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text(
        "import numpy as np\nfrom pymatgen.symmetry.analyzer import PointGroupAnalyzer\n",
        encoding="utf-8",
    )
    stats = analyze_code_directory(tmp_path)
    assert "pymatgen" in stats.forbidden_imports
    assert stats.lines_of_code == 2


def test_api_runner_multiturn_self_repair(tmp_path: Path) -> None:
    """Verify that LLMAPIRunner feeds back failures on turn 1 and accepts repaired code on turn 2."""
    wm = WorkspaceManager(base_temp_dir=tmp_path)
    evaluator = BenchmarkEvaluator()
    runner = LLMAPIRunner(workspace_manager=wm, evaluator=evaluator)

    call_count = 0

    def fake_chat_fn(messages: list[dict[str, str]]) -> tuple[str, dict[str, int]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Turn 1: buggy module returning C1 for everything
            code = "```python\ndef detect_point_group(p, tol=1e-2):\n    return 'C1'\n```"
            return code, {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        # Turn 2: confirm feedback was appended, return code that detects Kh for 1-atom
        assert any("failed the validation check" in m["content"] for m in messages)
        code = (
            "```python\n"
            "from pathlib import Path\n"
            "def detect_point_group(p, tol=1e-2):\n"
            "    lines = Path(p).read_text().strip().splitlines()\n"
            "    return 'Kh' if int(lines[0].strip()) == 1 else 'C1'\n"
            "```"
        )
        return code, {"prompt_tokens": 200, "completion_tokens": 40, "total_tokens": 240}

    report = runner.run(
        model="mock-model",
        mode="core-visible",
        max_turns=2,
        chat_fn=fake_chat_fn,
    )
    assert report.metadata.turns_used == 2
    assert report.metadata.total_tokens == 360
    # Ar_Kh.xyz + CHFClBr_C1.xyz in core_xyz = 2 matches
    assert report.evaluation.core.exact_correct == 2

    out_json = tmp_path / "results" / "mock.json"
    save_run_report(report, out_json)
    summary = format_single_report_summary(report)
    assert "mock-model" in summary
    md = generate_markdown_leaderboard([out_json])
    assert "| `api_mock-model_core-visible_t2` |" in md


def test_agent_command_and_telemetry() -> None:
    cmd = build_agent_command("opencode", "Solve pgmol", model="gemini-4-argon")
    assert cmd[:3] == ["opencode", "run", "Solve pgmol"]
    assert "--model" in cmd

    sample_stdout = (
        '{"type": "step_finish", "part": {"tokens": {"input": 1200, "output": 350}, "cost": 0.025}}\n'
    )
    tel = parse_agent_telemetry(sample_stdout)
    assert tel["input_tokens"] == 1200
    assert tel["output_tokens"] == 350
    assert tel["total_tokens"] == 1550
    assert tel["cost_usd"] == 0.025


def test_extract_python_code() -> None:
    raw = "Here is the code:\n```python\nx = 1\n```\nDone."
    assert extract_python_code(raw) == "x = 1\n"


def test_cli_score_on_my_try_branch(tmp_path: Path) -> None:
    """End-to-end evaluation of the `my-try` branch implementation across all 182 benchmark cases."""
    import subprocess
    from benchmark.cli import main as cli_main
    from benchmark.workspace import REPO_ROOT

    pkg_dir = tmp_path / "src" / "pgmol"
    pkg_dir.mkdir(parents=True)
    for fname in ("__init__.py", "symmetry.py"):
        out = subprocess.check_output(
            ["git", "show", f"my-try:src/pgmol/{fname}"],
            cwd=REPO_ROOT,
            text=True,
        )
        (pkg_dir / fname).write_text(out, encoding="utf-8")

    out_json = tmp_path / "my_try_result.json"
    rc = cli_main(
        [
            "score",
            "--target",
            str(tmp_path),
            "--harness",
            "opencode",
            "--model",
            "gemini-4-argon",
            "--run-id",
            "my-try",
            "--output",
            str(out_json),
        ]
    )
    assert rc == 0
    assert out_json.is_file()

    evaluator = BenchmarkEvaluator()
    res = evaluator.evaluate_directory(tmp_path)
    assert res.core.exact_correct == 36
    assert res.challenge.exact_correct == 55
    assert res.rotated_permuted.exact_correct == 91
    assert res.code_stats.forbidden_imports == []

