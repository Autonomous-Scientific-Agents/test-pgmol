"""Headless CLI coding agent runner (Claude Code, OpenCode, Codex, Aider, Custom)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any

from benchmark.evaluator import BenchmarkEvaluator
from benchmark.runners import BenchmarkRunReport, RunMetadata
from benchmark.workspace import VisibilityMode, WorkspaceManager

AGENT_PRESETS: dict[str, list[str]] = {
    "claude-code": [
        "claude",
        "-p",
        "{prompt}",
        "--dangerously-skip-permissions",
        "--output-format",
        "json",
    ],
    "opencode": [
        "opencode",
        "run",
        "{prompt}",
        "--format",
        "json",
    ],
    "codex": [
        "codex",
        "exec",
        "--full-auto",
        "{prompt}",
    ],
    "aider": [
        "aider",
        "--yes-always",
        "--no-auto-commits",
        "--message",
        "{prompt}",
    ],
}


def build_agent_command(
    harness: str,
    prompt: str,
    model: str | None = None,
    cmd_template: str | None = None,
) -> list[str]:
    """Build the subprocess command list for a headless CLI agent."""
    if cmd_template:
        parts = shlex.split(cmd_template)
        return [
            p.format(prompt=prompt, model=model or "")
            for p in parts
        ]

    if harness not in AGENT_PRESETS:
        raise ValueError(
            f"Unknown agent harness '{harness}'. Choose from {sorted(AGENT_PRESETS)} "
            "or pass --cmd-template."
        )

    cmd = [
        arg.format(prompt=prompt, model=model or "")
        for arg in AGENT_PRESETS[harness]
    ]

    if model:
        if harness in ("claude-code", "opencode", "codex", "aider"):
            cmd.extend(["--model", model])

    return cmd


def parse_agent_telemetry(stdout: str) -> dict[str, Any]:
    """Best-effort extraction of token usage and cost from JSON/JSONL CLI outputs."""
    input_tokens = 0
    output_tokens = 0
    cost_usd: float | None = None
    turns = 1

    for line in stdout.splitlines():
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        # Claude Code JSON output format
        if "usage" in obj and isinstance(obj["usage"], dict):
            u = obj["usage"]
            input_tokens += int(u.get("input_tokens", u.get("prompt_tokens", 0)))
            output_tokens += int(u.get("output_tokens", u.get("completion_tokens", 0)))
        if "total_cost_usd" in obj and isinstance(obj["total_cost_usd"], (int, float)):
            cost_usd = float(obj["total_cost_usd"])
        if "num_turns" in obj and isinstance(obj["num_turns"], int):
            turns = int(obj["num_turns"])

        # OpenCode JSON event format
        if obj.get("type") == "step_finish" and isinstance(obj.get("part"), dict):
            tokens = obj["part"].get("tokens", {})
            if isinstance(tokens, dict):
                input_tokens += int(tokens.get("input", 0))
                output_tokens += int(tokens.get("output", 0))
            if isinstance(obj["part"].get("cost"), (int, float)):
                cost_usd = (cost_usd or 0.0) + float(obj["part"]["cost"])

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": cost_usd,
        "turns_used": turns,
    }


class HeadlessAgentRunner:
    """Runs a headless coding agent in an isolated workspace and evaluates the result."""

    def __init__(
        self,
        workspace_manager: WorkspaceManager | None = None,
        evaluator: BenchmarkEvaluator | None = None,
    ) -> None:
        self.workspace_manager = workspace_manager or WorkspaceManager()
        self.evaluator = evaluator or BenchmarkEvaluator()

    def run(
        self,
        harness: str,
        model: str,
        mode: VisibilityMode = "core-visible",
        cmd_template: str | None = None,
        timeout_s: float = 1800.0,
        keep_workdir: bool = False,
        run_id: str | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> BenchmarkRunReport:
        safe_model = model.replace("/", "_").replace(":", "_")
        run_id = run_id or f"agent_{harness}_{safe_model}_{mode}"
        ctx = self.workspace_manager.create_workspace(
            mode=mode,
            run_id=run_id,
            keep_workdir=keep_workdir,
            scaffold_pyproject=False,
        )

        cmd = build_agent_command(
            harness=harness,
            prompt=ctx.prompt_text,
            model=model,
            cmd_template=cmd_template,
        )

        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)

        t0 = time.perf_counter()
        exit_code = 0
        stdout_text = ""
        notes = ""

        try:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=ctx.workdir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    check=False,
                )
                exit_code = proc.returncode
                stdout_text = proc.stdout or ""
                if exit_code != 0 and proc.stderr:
                    notes = f"Agent exited with {exit_code}: {proc.stderr.strip()[:200]}"
            except subprocess.TimeoutExpired as exc:
                exit_code = 124
                stdout_text = (exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""))
                notes = f"Agent timed out after {timeout_s}s"

            wall_time_s = time.perf_counter() - t0
            telemetry = parse_agent_telemetry(stdout_text)

            eval_result = self.evaluator.evaluate_directory(ctx.workdir)
            meta = RunMetadata(
                run_id=run_id,
                runner_type="agent",
                harness=harness,
                model=model,
                mode=mode,
                max_turns=0,
                turns_used=telemetry["turns_used"],
                wall_time_s=wall_time_s,
                input_tokens=telemetry["input_tokens"],
                output_tokens=telemetry["output_tokens"],
                total_tokens=telemetry["total_tokens"],
                cost_usd=telemetry["cost_usd"],
                workdir=str(ctx.workdir) if keep_workdir else None,
                exit_code=exit_code,
                notes=notes,
            )
            return BenchmarkRunReport(metadata=meta, evaluation=eval_result)
        finally:
            ctx.cleanup()
