"""CLI entrypoint for the pgmol benchmark framework."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from benchmark.evaluator import BenchmarkEvaluator
from benchmark.report import (
    format_single_report_summary,
    generate_markdown_leaderboard,
    save_run_report,
)
from benchmark.runners import BenchmarkRunReport, RunMetadata
from benchmark.runners.agent_runner import AGENT_PRESETS, HeadlessAgentRunner
from benchmark.runners.api_runner import LLMAPIRunner
from benchmark.workspace import REPO_ROOT

DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmark" / "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pgmol-bench",
        description="Standardized benchmark runner for LLM APIs and headless coding agents on pgmol.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # 1. `score`: Evaluate an existing directory / repository
    p_score = sub.add_parser(
        "score",
        help="Evaluate an existing pgmol directory/repository against Core, Challenge, and Rotated+Permuted suites.",
    )
    p_score.add_argument(
        "--target",
        type=Path,
        default=REPO_ROOT,
        help="Path to the repository or package directory to evaluate (default: current repo).",
    )
    p_score.add_argument("--harness", default="existing", help="Harness label for report metadata.")
    p_score.add_argument("--model", default="unknown", help="Model label for report metadata.")
    p_score.add_argument(
        "--mode",
        choices=("core-visible", "zero-shot"),
        default="core-visible",
        help="Visibility mode label for report metadata.",
    )
    p_score.add_argument("--run-id", default=None, help="Custom run identifier.")
    p_score.add_argument("--seed", type=int, default=20261022, help="RNG seed for 3D rotations/permutations.")
    p_score.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file path to save the evaluation report.",
    )

    # 2. `api`: Benchmark an LLM via API (1-shot or multi-turn loop)
    p_api = sub.add_parser(
        "api",
        help="Benchmark an LLM via OpenAI-compatible chat completions API (1-shot or multi-turn).",
    )
    p_api.add_argument("--model", required=True, help="Model ID (e.g., gpt-5, claude-opus-4-6, gemini-4-argon).")
    p_api.add_argument(
        "--mode",
        choices=("core-visible", "zero-shot"),
        default="core-visible",
        help="Whether core_xyz is visible for multi-turn feedback or held out (zero-shot).",
    )
    p_api.add_argument(
        "--turns",
        type=int,
        default=1,
        help="Max turns: 1 for 1-shot synthesis, >1 for multi-turn self-repair loop (default: 1).",
    )
    p_api.add_argument("--base-url", default=None, help="OpenAI-compatible base URL.")
    p_api.add_argument("--api-key", default=None, help="API key (defaults to OPENAI_API_KEY / AMSC_API_KEY).")
    p_api.add_argument("--temperature", type=float, default=None, help="Sampling temperature (optional).")
    p_api.add_argument("--keep-workdir", action="store_true", help="Preserve temporary workspace after run.")
    p_api.add_argument("--run-id", default=None, help="Custom run identifier.")
    p_api.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (defaults to benchmark/results/<run_id>.json).",
    )

    # 3. `agent`: Benchmark a headless CLI coding agent
    p_agent = sub.add_parser(
        "agent",
        help="Benchmark a headless coding agent (claude-code, opencode, codex, aider, or custom CLI).",
    )
    p_agent.add_argument(
        "--harness",
        default="opencode",
        help=f"Agent harness preset ({', '.join(sorted(AGENT_PRESETS))}) or custom label.",
    )
    p_agent.add_argument("--model", required=True, help="Model identifier passed to the agent.")
    p_agent.add_argument(
        "--mode",
        choices=("core-visible", "zero-shot"),
        default="core-visible",
        help="Workspace visibility mode: core-visible (copies only core_xyz) or zero-shot (no XYZ files).",
    )
    p_agent.add_argument(
        "--cmd-template",
        default=None,
        help="Optional custom shell command template with {prompt} and {model} placeholders.",
    )
    p_agent.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Max agent execution time in seconds (default: 1800).",
    )
    p_agent.add_argument("--keep-workdir", action="store_true", help="Preserve temporary workspace after run.")
    p_agent.add_argument("--run-id", default=None, help="Custom run identifier.")
    p_agent.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (defaults to benchmark/results/<run_id>.json).",
    )

    # 4. `leaderboard`: Render Markdown table from result JSON files
    p_board = sub.add_parser(
        "leaderboard",
        help="Generate a Markdown comparison table from benchmark result JSON files.",
    )
    p_board.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Result JSON files (defaults to all *.json in benchmark/results/).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "score":
        evaluator = BenchmarkEvaluator(seed=args.seed)
        eval_res = evaluator.evaluate_directory(args.target)
        run_id = args.run_id or f"score_{args.target.resolve().name}"
        meta = RunMetadata(
            run_id=run_id,
            runner_type="existing",
            harness=args.harness,
            model=args.model,
            mode=args.mode,
            max_turns=0,
            turns_used=0,
            wall_time_s=0.0,
            workdir=str(args.target.resolve()),
        )
        report = BenchmarkRunReport(metadata=meta, evaluation=eval_res)
        print(format_single_report_summary(report))
        if args.output:
            save_run_report(report, args.output)
            print(f"\nSaved JSON report to {args.output}")
        return 0

    if args.subcommand == "api":
        runner = LLMAPIRunner()
        report = runner.run(
            model=args.model,
            mode=args.mode,
            max_turns=args.turns,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            keep_workdir=args.keep_workdir,
            run_id=args.run_id,
        )
        print(format_single_report_summary(report))
        out_path = args.output or (DEFAULT_RESULTS_DIR / f"{report.metadata.run_id}.json")
        save_run_report(report, out_path)
        print(f"\nSaved JSON report to {out_path}")
        return 0

    if args.subcommand == "agent":
        runner = HeadlessAgentRunner()
        report = runner.run(
            harness=args.harness,
            model=args.model,
            mode=args.mode,
            cmd_template=args.cmd_template,
            timeout_s=args.timeout,
            keep_workdir=args.keep_workdir,
            run_id=args.run_id,
        )
        print(format_single_report_summary(report))
        out_path = args.output or (DEFAULT_RESULTS_DIR / f"{report.metadata.run_id}.json")
        save_run_report(report, out_path)
        print(f"\nSaved JSON report to {out_path}")
        return 0

    if args.subcommand == "leaderboard":
        files = list(args.files) if args.files else sorted(DEFAULT_RESULTS_DIR.glob("*.json"))
        if not files:
            print("No result JSON files found.", file=sys.stderr)
            return 1
        print(generate_markdown_leaderboard(files))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
