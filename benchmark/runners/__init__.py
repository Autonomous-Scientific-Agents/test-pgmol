"""Runners for raw LLM APIs and headless coding agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.evaluator import EvaluationResult
from benchmark.workspace import VisibilityMode


@dataclass
class RunMetadata:
    run_id: str
    runner_type: str  # "api" | "agent" | "existing"
    harness: str  # e.g., "openai-api", "claude-code", "opencode", "codex"
    model: str
    mode: VisibilityMode
    max_turns: int
    turns_used: int
    wall_time_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    workdir: str | None = None
    exit_code: int = 0
    notes: str = ""


@dataclass
class BenchmarkRunReport:
    metadata: RunMetadata
    evaluation: EvaluationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": asdict(self.metadata),
            "evaluation": self.evaluation.to_dict(),
        }
