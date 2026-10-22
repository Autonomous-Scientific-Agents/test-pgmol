"""Hugging Face Space (Gradio) Leaderboard for the pgmol Molecular Symmetry Benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import gradio as gr
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
BUNDLED_DATA_FILE = HERE / "data" / "leaderboard_data.json"
MONO_RESULTS_DIR = REPO_ROOT / "benchmark" / "results"


AGENT_REPO_ROWS: list[dict[str, Any]] = [
    {
        "Repository / Branch": "Autonomous-Scientific-Agents/test-pgmol (my-try)",
        "Harness · Model": "OpenCode · Gemini 4 Argon",
        "Mode": "core-visible (interactive)",
        "Core (36)": "36/36 (100.0%)",
        "Challenge (55)": "55/55 (100.0%)",
        "Rotated + Permuted (91)": "91/91 (100.0%)",
        "Overall (182)": "182/182 (100.0%)",
        "Core Avg (ms)": 2.61,
        "Challenge Avg (ms)": 13.33,
        "Notes": "Shell-partitioned axis discovery + 1-atom fast rejection probe",
    },
    {
        "Repository / Branch": "Autonomous-Scientific-Agents/PGMol",
        "Harness · Model": "OpenCode · GPT-5.6 Sol",
        "Mode": "core-visible (interactive)",
        "Core (36)": "36/36 (100.0%)",
        "Challenge (55)": "55/55 (100.0%)",
        "Rotated + Permuted (91)": "91/91 (100.0%)",
        "Overall (182)": "182/182 (100.0%)",
        "Core Avg (ms)": 3.57,
        "Challenge Avg (ms)": 48.28,
        "Notes": "Exhaustive O(3) orthonormal frame discovery + DFS perfect matching",
    },
    {
        "Repository / Branch": "keceli/claude-pgmol",
        "Harness · Model": "Claude Code · Claude Opus 5",
        "Mode": "core-visible (interactive)",
        "Core (36)": "36/36 (100.0%)",
        "Challenge (55)": "55/55 (100.0%)",
        "Rotated + Permuted (91)": "91/91 (100.0%)",
        "Overall (182)": "182/182 (100.0%)",
        "Core Avg (ms)": 10.67,
        "Challenge Avg (ms)": 118.46,
        "Notes": "Inertia-eigenstate classification + geometric candidate axis search",
    },
    {
        "Repository / Branch": "keceli/pgmol-inkling",
        "Harness · Model": "Claude Code · Inkling",
        "Mode": "core-visible (interactive)",
        "Core (36)": "36/36 (100.0%)",
        "Challenge (55)": "55/55 (100.0%)",
        "Rotated + Permuted (91)": "42/91 (46.2%)",
        "Overall (182)": "133/182 (73.1%)",
        "Core Avg (ms)": 8.88,
        "Challenge Avg (ms)": 56.84,
        "Notes": "Hardcoded coordinate/golden-ratio axis grid fails under random 3D rotations",
    },
]

EXTERNAL_LIBRARY_ROWS: list[dict[str, Any]] = [
    {
        "Library": "pymatgen (PointGroupAnalyzer)",
        "Version": "2026.9.24",
        "Core Exact (36)": "30/36 (83.3%)",
        "Core Normalized (36)": "36/36 (100.0%)",
        "Challenge Exact (55)": "50/55 (90.9%)",
        "Challenge Normalized (55)": "54/55 (98.2%)",
        "Rotated + Permuted (91)": "90/91 (98.9%)",
        "Core Avg (ms)": 1.43,
        "Challenge Avg (ms)": 3.69,
        "Failure Root Cause": "Crashes with ValueError on H3_D3h.xyz due to r/2 bug in cluster_sites with default tol=0.3",
    },
    {
        "Library": "pyscf (pyscf.symm.detect_symm)",
        "Version": "2.14.0",
        "Core Exact (36)": "29/36 (80.6%)",
        "Core Normalized (36)": "36/36 (100.0%)",
        "Challenge Exact (55)": "49/55 (89.1%)",
        "Challenge Normalized (55)": "53/55 (96.4%)",
        "Rotated + Permuted (91)": "89/91 (97.8%)",
        "Core Avg (ms)": 0.68,
        "Challenge Avg (ms)": 1.34,
        "Failure Root Cause": "Planar C4h rank-2 tensor degeneracy bug on C4N4Pt_C4h (returns C2h); tol=1e-5 miss on C5H5P_C5",
    },
    {
        "Library": "pointgroup (PointGroup)",
        "Version": "0.4.4",
        "Core Exact (36)": "35/36 (97.2%)",
        "Core Normalized (36)": "35/36 (97.2%)",
        "Challenge Exact (55)": "54/55 (98.2%)",
        "Challenge Normalized (55)": "54/55 (98.2%)",
        "Rotated + Permuted (91)": "88/91 (96.7%)",
        "Core Avg (ms)": 110.43,
        "Challenge Avg (ms)": 1143.13,
        "Failure Root Cause": "Accidental spherical top C2H2O2_D2h loops 217x into Th; Ar_Kh -> Dinfh; rotated B12H60_I raises",
    },
    {
        "Library": "molsym (find_point_group)",
        "Version": "1.2.0",
        "Core Exact (36)": "29/36 (80.6%)",
        "Core Normalized (36)": "35/36 (97.2%)",
        "Challenge Exact (55)": "50/55 (90.9%)",
        "Challenge Normalized (55)": "54/55 (98.2%)",
        "Rotated + Permuted (91)": "89/91 (97.8%)",
        "Core Avg (ms)": 90.73,
        "Challenge Avg (ms)": 3341.46,
        "Failure Root Cause": "Unimplemented T branch ('# TODO: No path for T') returns Td on C5H12_T; O(N^5) loop on B12H60_I (83s)",
    },
]


def load_leaderboard_entries() -> list[dict[str, Any]]:
    """Load the 27-model AmSC MAG benchmark entries from bundled JSON or monorepo."""
    if BUNDLED_DATA_FILE.exists():
        return json.loads(BUNDLED_DATA_FILE.read_text(encoding="utf-8"))
    if MONO_RESULTS_DIR.exists():
        import sys

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from benchmark.report import build_web_leaderboard_entries

        return build_web_leaderboard_entries(
            sorted(MONO_RESULTS_DIR.glob("*.json")), include_code=True
        )
    return []


def filter_and_format_dataframe(
    family_filter: str = "All",
    search_query: str = "",
    sort_by: str = "Overall Accuracy (High → Low)",
    hide_zero_runs: bool = False,
) -> pd.DataFrame:
    """Filter, sort, and format the zero-shot LLM leaderboard into a pandas DataFrame."""
    entries = load_leaderboard_entries()
    filtered = []
    q = (search_query or "").strip().lower()

    for r in entries:
        if hide_zero_runs and r["overall_ok"] == 0:
            continue
        if family_filter == "100% Club (182/182)":
            if r["overall_ok"] != r["overall_tot"]:
                continue
        elif family_filter != "All" and r["family"] != family_filter:
            continue
        if q and q not in r["model"].lower() and q not in r["family"].lower():
            continue
        filtered.append(r)

    if sort_by == "Challenge Speed (Fast → Slow)":
        filtered.sort(
            key=lambda r: (
                r["chal_ms"] if r["chal_ok"] > 0 else 1e9,
                -r["overall_ok"],
            )
        )
    elif sort_by == "Core Speed (Fast → Slow)":
        filtered.sort(
            key=lambda r: (
                r["core_ms"] if r["core_ok"] > 0 else 1e9,
                -r["overall_ok"],
            )
        )
    elif sort_by == "Conciseness / LOC (Low → High)":
        filtered.sort(
            key=lambda r: (
                r["loc"] if r["loc"] > 0 else 1e9,
                -r["overall_ok"],
            )
        )
    elif sort_by == "Generation Time (Fast → Slow)":
        filtered.sort(key=lambda r: (r["wall_s"], -r["overall_ok"]))
    else:
        filtered.sort(
            key=lambda r: (
                -r["overall_ok"],
                r["chal_s"] if r["chal_ok"] > 0 else 1e9,
                r["loc"] if r["loc"] > 0 else 1e9,
            )
        )

    rows = []
    for r in filtered:
        rows.append(
            {
                "Rank": r["rank"],
                "Model (AmSC MAG)": r["model"],
                "Family": r["family"],
                "Overall (182)": f"{r['overall_ok']}/{r['overall_tot']} ({r['overall_pct']:.1f}%)",
                "Core (36)": f"{r['core_ok']}/{r['core_tot']} ({r['core_pct']:.1f}%)",
                "Challenge (55)": f"{r['chal_ok']}/{r['chal_tot']} ({r['chal_pct']:.1f}%)",
                "Rotated + Permuted (91)": f"{r['rot_ok']}/{r['rot_tot']} ({r['rot_pct']:.1f}%)",
                "Core Avg (ms)": r["core_ms"] if r["core_ok"] > 0 else None,
                "Challenge Avg (ms)": r["chal_ms"] if r["chal_ok"] > 0 else None,
                "Gen Time (s)": r["wall_s"],
                "LOC": r["loc"],
                "Forbidden Imports": ", ".join(r.get("forbidden_imports", [])) or "—",
            }
        )
    return pd.DataFrame(rows)


def inspect_model_run(model_name: str) -> tuple[str, str]:
    """Return Markdown failure summary and generated Python code for a selected model."""
    entries = load_leaderboard_entries()
    selected = next((e for e in entries if e["model"] == model_name), None)
    if selected is None and entries:
        selected = entries[0]
    if selected is None:
        return "No model data found.", "# No code available"

    lines = [
        f"### `{selected['model']}` (`{selected['run_id']}`)",
        f"- **Rank:** #{selected['rank']} · **Family:** {selected['family']} · **Mode:** `{selected['mode']}` (1-shot)",
        f"- **Overall Score:** **{selected['overall_ok']}/{selected['overall_tot']} ({selected['overall_pct']:.1f}%)**",
        f"- **Core (`36`):** `{selected['core_ok']}/{selected['core_tot']}` ({selected['core_pct']:.1f}%) — avg `{selected['core_ms']:.2f} ms` / total `{selected['core_s']:.3f} s`",
        f"- **Challenge (`55`):** `{selected['chal_ok']}/{selected['chal_tot']}` ({selected['chal_pct']:.1f}%) — avg `{selected['chal_ms']:.2f} ms` / total `{selected['chal_s']:.3f} s`",
        f"- **Rotated + Permuted (`91`):** `{selected['rot_ok']}/{selected['rot_tot']}` ({selected['rot_pct']:.1f}%) — avg `{selected['rot_ms']:.2f} ms` / total `{selected['rot_s']:.3f} s`",
        f"- **Code Size:** `{selected['loc']} LOC` · **Generation Time:** `{selected['wall_s']:.1f} s`",
    ]
    if selected.get("forbidden_imports"):
        lines.append(
            f"- ⚠️ **Forbidden Imports Detected:** `{', '.join(selected['forbidden_imports'])}`"
        )

    all_fails = (
        [f"- **[Core]** `{f}`" for f in selected.get("core_failures", [])]
        + [f"- **[Challenge]** `{f}`" for f in selected.get("chal_failures", [])]
        + [f"- **[Rotated+Permuted]** `{f}`" for f in selected.get("rot_failures", [])]
    )
    if not all_fails:
        lines.append(
            "\n✅ **Zero failures across all 182 test cases (Core, Challenge, and Rotated+Permuted)!**"
        )
    else:
        lines.append("\n#### Sample Failure Cases")
        lines.extend(all_fails)

    code = selected.get("generated_code", "")
    if not code:
        gen_file = MONO_RESULTS_DIR / "generated" / f"{selected['run_id']}.py"
        if gen_file.exists():
            code = gen_file.read_text(encoding="utf-8")
        else:
            code = "# Generated Python file not found."
    return "\n".join(lines), code


def evaluate_uploaded_file(uploaded_file: str | None) -> str:
    """Score either an uploaded benchmark JSON report or a Python __init__.py module."""
    if not uploaded_file:
        return "Please upload either a `.py` implementation of `detect_point_group(xyz_path)` or a `.json` report generated by `python -m benchmark`."

    path = Path(uploaded_file)
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            m = data["metadata"]
            ev = data["evaluation"]
            c, ch, rot = ev["core"], ev["challenge"], ev["rotated_permuted"]
            return (
                f"### Uploaded Report: `{m.get('run_id', path.name)}`\n"
                f"- **Harness · Model:** `{m.get('harness')}` · `{m.get('model')}` (`{m.get('mode')}`)\n"
                f"- **Core (36):** `{c['exact_correct']}/{c['total']}` ({c['exact_accuracy']*100:.1f}%) — avg `{c['avg_ms']:.2f} ms`\n"
                f"- **Challenge (55):** `{ch['exact_correct']}/{ch['total']}` ({ch['exact_accuracy']*100:.1f}%) — avg `{ch['avg_ms']:.2f} ms`\n"
                f"- **Rotated + Permuted (91):** `{rot['exact_correct']}/{rot['total']}` ({rot['exact_accuracy']*100:.1f}%) — avg `{rot['avg_ms']:.2f} ms`\n"
                f"- **Overall Exact Accuracy:** **{ev.get('overall_exact_accuracy', 0.0)*100:.1f}%**"
            )
        except Exception as exc:
            return f"Failed to parse JSON report: `{type(exc).__name__}: {exc}`"

    if path.suffix.lower() == ".py":
        validation_dir = REPO_ROOT / "validation"
        if not validation_dir.exists():
            return "Live `.py` evaluation requires the `validation/` dataset directory."
        import sys

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from benchmark.evaluator import BenchmarkEvaluator

        tmpdir = Path(tempfile.mkdtemp(prefix="pgmol_hf_eval_"))
        try:
            pkg_dir = tmpdir / "src" / "pgmol"
            pkg_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, pkg_dir / "__init__.py")
            evaluator = BenchmarkEvaluator(validation_dir=validation_dir)
            res = evaluator.evaluate_directory(tmpdir)
            c, ch, rot = res.core, res.challenge, res.rotated_permuted
            return (
                f"### Live Evaluation of `{path.name}` ({res.code_stats.lines_of_code} LOC)\n"
                f"- **Core (36):** `{c.exact_correct}/{c.total}` ({c.exact_accuracy*100:.1f}%) — avg `{c.avg_ms:.2f} ms` / total `{c.total_s:.3f} s`\n"
                f"- **Challenge (55):** `{ch.exact_correct}/{ch.total}` ({ch.exact_accuracy*100:.1f}%) — avg `{ch.avg_ms:.2f} ms` / total `{ch.total_s:.3f} s`\n"
                f"- **Rotated + Permuted (91):** `{rot.exact_correct}/{rot.total}` ({rot.exact_accuracy*100:.1f}%) — avg `{rot.avg_ms:.2f} ms` / total `{rot.total_s:.3f} s`\n"
                f"- **Overall Score:** **{c.exact_correct + ch.exact_correct + rot.exact_correct}/182 ({res.overall_exact_accuracy*100:.1f}%)**"
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return f"Unsupported file extension `{path.suffix}`. Upload a `.py` or `.json` file."


def build_app() -> gr.Blocks:
    """Construct the Hugging Face Space Gradio Blocks interface."""
    entries = load_leaderboard_entries()
    model_names = [e["model"] for e in entries]
    default_model = model_names[0] if model_names else ""

    with gr.Blocks(
        title="pgmol Molecular Symmetry Benchmark Leaderboard",
    ) as demo:
        gr.Markdown(
            """
            # ⬢ `pgmol` Molecular Symmetry Benchmark Leaderboard
            Evaluating **27 LLMs (1-shot zero-shot via AmSC Model Access Gateway)**, **autonomous coding agents** (`OpenCode`, `Claude Code`), and **external Python quantum-chemistry libraries** (`pymatgen`, `pyscf`, `pointgroup`, `molsym`) on 3D Schoenflies point-group determination across **182 molecular geometries** (`36 Core` + `55 Challenge` + `91 Randomly 3D-Rotated, Translated & Atom-Permuted`).

            🔗 **[Interactive 3D Failure Explorer & Website](https://autonomous-scientific-agents.github.io/test-pgmol/)** · **[GitHub Repository (`Autonomous-Scientific-Agents/test-pgmol`)](https://github.com/Autonomous-Scientific-Agents/test-pgmol)**
            """
        )

        with gr.Tabs():
            with gr.TabItem("🏆 Zero-Shot LLM Leaderboard (27 Models)"):
                gr.Markdown(
                    """
                    ### 1-Shot Zero-Shot Evaluation on AmSC Model Access Gateway (`--mode zero-shot --turns 1`)
                    Each model receives only the specification prompt with **no access to any XYZ files or test feedback**, and must write a complete, pure-Python + NumPy Schoenflies point-group detector evaluated on **182 anonymized molecules** (`mol_0001.xyz`).
                    - **6 models achieved a perfect `182/182 (100.0%)` in a single shot:** `gpt-6-astra`, `claude-sonnet-5-5` (most concise at **171 LOC** & **1.26 ms** avg Core), `gpt-6.1-sol`, `gpt-5.6-terra`, `claude-opus-5-5-medium@amazon`, and `claude-haiku-low`.
                    """
                )
                with gr.Row():
                    family_dd = gr.Dropdown(
                        choices=[
                            "All",
                            "100% Club (182/182)",
                            "OpenAI GPT",
                            "Anthropic Claude",
                            "Google Gemini",
                            "CBorg / Open-Weights",
                            "Amazon / Mistral / NVIDIA",
                        ],
                        value="All",
                        label="Filter by Model Family",
                    )
                    search_tb = gr.Textbox(
                        label="Search Model Name",
                        placeholder="e.g. claude, gpt-6, gemini-4, inkling...",
                    )
                    sort_dd = gr.Dropdown(
                        choices=[
                            "Overall Accuracy (High → Low)",
                            "Challenge Speed (Fast → Slow)",
                            "Core Speed (Fast → Slow)",
                            "Conciseness / LOC (Low → High)",
                            "Generation Time (Fast → Slow)",
                        ],
                        value="Overall Accuracy (High → Low)",
                        label="Sort By",
                    )
                    hide_zero_cb = gr.Checkbox(
                        value=False,
                        label="Hide 0% syntax/import error runs",
                    )

                lb_table = gr.Dataframe(
                    value=filter_and_format_dataframe(),
                    interactive=False,
                )

                for ctrl in (family_dd, search_tb, sort_dd, hide_zero_cb):
                    ctrl.change(
                        fn=filter_and_format_dataframe,
                        inputs=[family_dd, search_tb, sort_dd, hide_zero_cb],
                        outputs=lb_table,
                    )

            with gr.TabItem("🤖 Agentic Repos & External Libraries"):
                gr.Markdown("### Autonomous Coding Agent Repositories (`core-visible` interactive sessions)")
                gr.Dataframe(
                    value=pd.DataFrame(AGENT_REPO_ROWS),
                    interactive=False,
                )
                gr.Markdown("### External Python Quantum-Chemistry & Symmetry Libraries")
                gr.Dataframe(
                    value=pd.DataFrame(EXTERNAL_LIBRARY_ROWS),
                    interactive=False,
                )

            with gr.TabItem("🔍 Model Code & Failure Inspector"):
                gr.Markdown(
                    "Select any of the 27 AmSC MAG zero-shot runs below to inspect its per-split accuracy, failure cases, and complete generated Python source code (`src/pgmol/__init__.py`)."
                )
                model_selector = gr.Dropdown(
                    choices=model_names,
                    value=default_model,
                    label="Select Model Run",
                )
                init_summary, init_code = inspect_model_run(default_model)
                summary_md = gr.Markdown(value=init_summary)
                code_view = gr.Code(
                    value=init_code,
                    language="python",
                    label="Generated src/pgmol/__init__.py",
                )
                model_selector.change(
                    fn=inspect_model_run,
                    inputs=[model_selector],
                    outputs=[summary_md, code_view],
                )

            with gr.TabItem("🧪 Live Evaluator / Submit Run"):
                gr.Markdown(
                    """
                    ### Evaluate a `pgmol` Implementation or Inspect a Result JSON
                    Upload either:
                    1. A Python module (`__init__.py`) defining `detect_point_group(xyz_path)` (pure Python + NumPy) to run the live 182-molecule evaluator (`Core` + `Challenge` + `Rotated + Permuted` with anonymized filenames), **or**
                    2. A benchmark `.json` report produced by `PYTHONPATH=. uv run --project benchmark python -m benchmark ...`.
                    """
                )
                upload_btn = gr.File(
                    label="Upload Python Module (.py) or Benchmark Report (.json)",
                    file_types=[".py", ".json"],
                    type="filepath",
                )
                eval_output = gr.Markdown()
                upload_btn.change(
                    fn=evaluate_uploaded_file,
                    inputs=[upload_btn],
                    outputs=[eval_output],
                )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
