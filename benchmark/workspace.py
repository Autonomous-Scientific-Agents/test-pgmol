"""Workspace isolation and data-leakage prevention for pgmol benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Literal

VisibilityMode = Literal["core-visible", "zero-shot"]

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
VALIDATION_DIR = REPO_ROOT / "validation"


@dataclass
class WorkspaceContext:
    """Represents an isolated workspace prepared for an LLM or agent run."""

    workdir: Path
    mode: VisibilityMode
    prompt_text: str
    system_prompt_text: str
    keep_workdir: bool = False

    def cleanup(self) -> None:
        if not self.keep_workdir and self.workdir.exists():
            shutil.rmtree(self.workdir, ignore_errors=True)


class WorkspaceManager:
    """Creates isolated workspaces that strictly enforce test-set visibility rules."""

    def __init__(
        self,
        repo_root: Path = REPO_ROOT,
        base_temp_dir: Path | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.base_temp_dir = base_temp_dir

    def get_prompt(self, mode: VisibilityMode) -> str:
        prompt_file = (
            PROMPTS_DIR / "spec_core_visible.md"
            if mode == "core-visible"
            else PROMPTS_DIR / "spec_zero_shot.md"
        )
        return prompt_file.read_text(encoding="utf-8")

    def get_api_system_prompt(self) -> str:
        return (PROMPTS_DIR / "api_system.md").read_text(encoding="utf-8")

    def create_workspace(
        self,
        mode: VisibilityMode = "core-visible",
        run_id: str = "run",
        keep_workdir: bool = False,
        scaffold_pyproject: bool = False,
    ) -> WorkspaceContext:
        """
        Create an isolated git-initialized workspace.

        - In `core-visible` mode: copies `validation/core_xyz/` (36 molecules) ONLY.
          `validation/challenge_xyz/` is NEVER copied into the workspace.
        - In `zero-shot` mode: no `validation/` files are copied at all.
        """
        if self.base_temp_dir is not None:
            self.base_temp_dir.mkdir(parents=True, exist_ok=True)
            workdir = Path(
                tempfile.mkdtemp(prefix=f"pgmol_bench_{run_id}_", dir=self.base_temp_dir)
            )
        else:
            workdir = Path(tempfile.mkdtemp(prefix=f"pgmol_bench_{run_id}_"))

        prompt_text = self.get_prompt(mode)
        system_prompt_text = self.get_api_system_prompt()

        # Write task specification as AGENTS.md and PROMPT.md in the workspace
        (workdir / "AGENTS.md").write_text(prompt_text, encoding="utf-8")
        (workdir / "README.md").write_text(
            "# pgmol\n\nSchoenflies point group detection package.\n",
            encoding="utf-8",
        )

        if mode == "core-visible":
            src_core = self.repo_root / "validation" / "core_xyz"
            dst_core = workdir / "validation" / "core_xyz"
            dst_core.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_core, dst_core)

        if scaffold_pyproject:
            self._write_default_pyproject(workdir)

        # Initialize a clean git repository so agents (Claude Code, OpenCode, Codex)
        # have a clean working tree without history containing answers or challenge_xyz.
        subprocess.run(
            ["git", "init", "-q"],
            cwd=workdir,
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "benchmark@pgmol.local"],
            cwd=workdir,
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "pgmol-bench"],
            cwd=workdir,
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["git", "add", "."],
            cwd=workdir,
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-q", "-m", "chore: initial benchmark workspace"],
            cwd=workdir,
            check=False,
            capture_output=True,
        )

        return WorkspaceContext(
            workdir=workdir,
            mode=mode,
            prompt_text=prompt_text,
            system_prompt_text=system_prompt_text,
            keep_workdir=keep_workdir,
        )

    @staticmethod
    def _write_default_pyproject(workdir: Path) -> None:
        pyproject = """[project]
name = "pgmol"
version = "0.1.0"
description = "Schoenflies point group determination for molecules from 3D coordinates"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
]

[project.scripts]
pgmol = "pgmol:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[dependency-groups]
dev = [
    "pytest>=8.0",
]
"""
        (workdir / "pyproject.toml").write_text(pyproject, encoding="utf-8")
        src_dir = workdir / "src" / "pgmol"
        src_dir.mkdir(parents=True, exist_ok=True)
        init_file = src_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")
