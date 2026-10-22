"""LLM API Runner supporting 1-shot and multi-turn self-repair loops."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request

from benchmark.evaluator import BenchmarkEvaluator
from benchmark.runners import BenchmarkRunReport, RunMetadata
from benchmark.workspace import VisibilityMode, WorkspaceManager

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_python_code(response_text: str) -> str:
    """Extract Python source code from a markdown-fenced LLM response."""
    matches = CODE_BLOCK_RE.findall(response_text)
    if matches:
        # Pick the longest python block if multiple exist
        return max(matches, key=len).strip() + "\n"
    return response_text.strip() + "\n"


class OpenAICompatibleClient:
    """Zero-dependency HTTP client for OpenAI-compatible chat completions endpoints."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        timeout_s: float = 300.0,
    ) -> None:
        self.model = model
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("AMSC_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or ""
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def chat_completion(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, dict[str, int]]:
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API HTTP {exc.code}: {err_body}") from exc

        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        token_counts = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
        return content, token_counts


class LLMAPIRunner:
    """
    Runs a direct LLM API benchmark in either:
    - 1-shot mode (`max_turns=1`)
    - Multi-turn self-repair mode (`max_turns > 1`):
      - In `core-visible` mode: evaluates against `core_xyz` (never `challenge_xyz`!)
        and feeds back any failing molecules/tracebacks to the model.
      - In `zero-shot` mode: runs a syntax + synthetic smoke check (H2O, CH4, Ar, CO2)
        and feeds back any runtime exceptions.
    """

    def __init__(
        self,
        workspace_manager: WorkspaceManager | None = None,
        evaluator: BenchmarkEvaluator | None = None,
    ) -> None:
        self.workspace_manager = workspace_manager or WorkspaceManager()
        self.evaluator = evaluator or BenchmarkEvaluator()

    def run(
        self,
        model: str,
        mode: VisibilityMode = "core-visible",
        max_turns: int = 1,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        keep_workdir: bool = False,
        run_id: str | None = None,
        chat_fn: Any | None = None,
    ) -> BenchmarkRunReport:
        run_id = run_id or f"api_{model.replace('/', '_')}_{mode}_t{max_turns}"
        ctx = self.workspace_manager.create_workspace(
            mode=mode,
            run_id=run_id,
            keep_workdir=keep_workdir,
            scaffold_pyproject=True,
        )

        client = (
            None
            if chat_fn is not None
            else OpenAICompatibleClient(
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
            )
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": ctx.system_prompt_text},
            {"role": "user", "content": ctx.prompt_text},
        ]

        total_in = 0
        total_out = 0
        total_tok = 0
        turns_used = 0
        t0 = time.perf_counter()

        try:
            for turn in range(1, max_turns + 1):
                turns_used = turn
                if chat_fn is not None:
                    reply_text, usage = chat_fn(messages)
                else:
                    assert client is not None
                    reply_text, usage = client.chat_completion(messages)

                total_in += int(usage.get("prompt_tokens", 0))
                total_out += int(usage.get("completion_tokens", 0))
                total_tok += int(usage.get("total_tokens", 0))

                code = extract_python_code(reply_text)
                init_path = ctx.workdir / "src" / "pgmol" / "__init__.py"
                init_path.write_text(code, encoding="utf-8")

                if turn >= max_turns:
                    break

                # Self-repair feedback check (strictly respecting visibility mode!)
                passed, feedback = self._build_turn_feedback(ctx.workdir, mode)
                if passed:
                    break

                messages.append({"role": "assistant", "content": reply_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your implementation failed the validation check:\n\n"
                            f"{feedback}\n\n"
                            "Please fix the bugs and return the complete updated Python module "
                            "in a single ```python ... ``` block."
                        ),
                    }
                )

            wall_time_s = time.perf_counter() - t0
            eval_result = self.evaluator.evaluate_directory(ctx.workdir)
            meta = RunMetadata(
                run_id=run_id,
                runner_type="api",
                harness=f"llm-api ({max_turns}-turn)" if max_turns > 1 else "llm-api (1-shot)",
                model=model,
                mode=mode,
                max_turns=max_turns,
                turns_used=turns_used,
                wall_time_s=wall_time_s,
                input_tokens=total_in,
                output_tokens=total_out,
                total_tokens=total_tok,
                workdir=str(ctx.workdir) if keep_workdir else None,
            )
            return BenchmarkRunReport(metadata=meta, evaluation=eval_result)
        finally:
            ctx.cleanup()

    def _build_turn_feedback(
        self, workdir: Path, mode: VisibilityMode
    ) -> tuple[bool, str]:
        """
        Generate feedback for multi-turn LLM repair without leaking held-out data.
        - `core-visible`: evaluates ONLY on `core` split.
        - `zero-shot`: runs syntax + basic synthetic molecules check (Ar, CO2, H2O, CH4).
        """
        if mode == "core-visible":
            res = self.evaluator.evaluate_directory(workdir, splits=("core",))
            core = res.core
            if core.exact_correct == core.total and core.total > 0:
                return True, "All 36 core_xyz molecules passed!"
            lines = [
                f"Core validation score: {core.exact_correct}/{core.total} ({core.exact_accuracy * 100:.1f}%)."
            ]
            for fail in core.failures[:15]:
                lines.append(
                    f"- {fail['name']}: expected={fail['expected']}, "
                    f"got={fail['predicted']}, error={fail['error']}"
                )
            return False, "\n".join(lines)

        # Zero-shot mode: run a synthetic smoke check with no validation files
        smoke_script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
import pgmol
fn = getattr(pgmol, "detect_point_group", None) or getattr(pgmol, "point_group", None)
if fn is None:
    raise AttributeError("pgmol must define detect_point_group")

samples = [
    ("1\\nAr\\nAr 0.0 0.0 0.0\\n", "Kh"),
    ("3\\nCO2\\nC 0.0 0.0 0.0\\nO 0.0 0.0 1.16\\nO 0.0 0.0 -1.16\\n", "Dinfh"),
    ("2\\nCO\\nC 0.0 0.0 0.0\\nO 0.0 0.0 1.128\\n", "Cinfv"),
    ("3\\nH2O\\nO 0.0 0.0 0.1173\\nH 0.0 0.7572 -0.4692\\nH 0.0 -0.7572 -0.4692\\n", "C2v"),
]
failures = []
for idx, (xyz_text, exp) in enumerate(samples):
    p = Path(sys.argv[1]) / f"_smoke_{idx}.xyz"
    p.write_text(xyz_text)
    got = fn(p)
    p.unlink(missing_ok=True)
    if got != exp:
        failures.append(f"Sample {idx} (expected {exp}, got {got})")
if failures:
    raise AssertionError("; ".join(failures))
print("OK")
"""
        proc = subprocess.run(
            [sys.executable, "-c", smoke_script, str(workdir)],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        if proc.returncode == 0 and "OK" in proc.stdout:
            return True, "Synthetic smoke check passed."
        err_msg = (proc.stderr or proc.stdout).strip()
        return False, f"Smoke test failed:\n{err_msg}"
