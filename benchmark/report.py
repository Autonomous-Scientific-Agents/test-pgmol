"""Report and Markdown leaderboard formatting for pgmol benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.runners import BenchmarkRunReport


def save_run_report(report: BenchmarkRunReport, out_path: Path) -> Path:
    """Save a BenchmarkRunReport as a JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return out_path


def format_single_report_summary(report: BenchmarkRunReport) -> str:
    """Format a concise console summary of a single benchmark run."""
    m = report.metadata
    ev = report.evaluation
    core = ev.core
    chal = ev.challenge
    rot = ev.rotated_permuted

    lines = [
        f"=== pgmol-bench Run: {m.run_id} ===",
        f"Harness / Model : {m.harness} · {m.model}",
        f"Visibility Mode : {m.mode} (turns: {m.turns_used}, wall time: {m.wall_time_s:.2f}s)",
        f"Code Stats      : {ev.code_stats.lines_of_code} LOC across {ev.code_stats.python_files} file(s)",
    ]
    if ev.code_stats.forbidden_imports:
        lines.append(
            f"WARNING         : Forbidden imports detected: {', '.join(ev.code_stats.forbidden_imports)}"
        )
    lines.extend(
        [
            "",
            f"Core (36)               : {core.exact_correct}/{core.total} ({core.exact_accuracy * 100:.1f}%) "
            f"[norm: {core.normalized_correct}/{core.total}] "
            f"| avg {core.avg_ms:.2f} ms / total {core.total_s:.3f} s",
            f"Challenge (55)          : {chal.exact_correct}/{chal.total} ({chal.exact_accuracy * 100:.1f}%) "
            f"[norm: {chal.normalized_correct}/{chal.total}] "
            f"| avg {chal.avg_ms:.2f} ms / total {chal.total_s:.3f} s",
            f"Rotated + Permuted (91) : {rot.exact_correct}/{rot.total} ({rot.exact_accuracy * 100:.1f}%) "
            f"[norm: {rot.normalized_correct}/{rot.total}] "
            f"| avg {rot.avg_ms:.2f} ms / total {rot.total_s:.3f} s",
        ]
    )
    for split_obj in (core, chal, rot):
        if split_obj.failures:
            lines.append(f"\nFailures in {split_obj.split} (up to 10):")
            for f in split_obj.failures[:10]:
                lines.append(
                    f"  - {f['name']}: expected={f['expected']}, got={f['predicted']}"
                    + (f" ({f['error']})" if f.get("error") else "")
                )
    return "\n".join(lines)


def generate_markdown_leaderboard(json_files: list[Path]) -> str:
    """Generate a standardized Markdown comparison table from result JSON files."""
    rows: list[dict[str, Any]] = []
    for path in json_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(data)

    # Sort by overall exact accuracy descending, then challenge total_s ascending
    rows.sort(
        key=lambda r: (
            -float(r["evaluation"].get("overall_exact_accuracy", 0.0)),
            float(r["evaluation"]["challenge"].get("total_s", 1e9)),
        )
    )

    header = (
        "| Run ID | Harness · Model | Mode | Core (36) | Challenge (55) | "
        "Rotated + Permuted (91) | Core time (avg / total) | Challenge time (avg / total) | "
        "Gen Wall Time | LOC |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    table_lines = [header]

    for r in rows:
        m = r["metadata"]
        ev = r["evaluation"]
        c = ev["core"]
        ch = ev["challenge"]
        rot = ev["rotated_permuted"]
        loc = ev.get("code_stats", {}).get("lines_of_code", 0)

        def _fmt_score(s: dict[str, Any]) -> str:
            corr = s["exact_correct"]
            tot = s["total"]
            pct = s["exact_accuracy"] * 100.0
            cell = f"{corr}/{tot} ({pct:.1f}%)"
            return f"**{cell}**" if corr == tot and tot > 0 else cell

        table_lines.append(
            f"| `{m['run_id']}` | {m['harness']} · {m['model']} | `{m['mode']}` | "
            f"{_fmt_score(c)} | {_fmt_score(ch)} | {_fmt_score(rot)} | "
            f"{c['avg_ms']:.2f} ms / {c['total_s']:.3f} s | "
            f"{ch['avg_ms']:.2f} ms / {ch['total_s']:.3f} s | "
            f"{m['wall_time_s']:.1f} s | {loc} |"
        )

    return "\n".join(table_lines)


def classify_model_family(model: str) -> str:
    """Group model names into high-level families for leaderboard filtering."""
    m = model.lower()
    if "claude" in m:
        return "Anthropic Claude"
    if "gpt-oss" in m or "@cborg" in m:
        return "CBorg / Open-Weights"
    if "gpt-" in m:
        return "OpenAI GPT"
    if "gemini" in m or "gemma" in m:
        return "Google Gemini"
    return "Amazon / Mistral / NVIDIA"


def build_web_leaderboard_entries(
    json_files: list[Path],
    include_code: bool = False,
) -> list[dict[str, Any]]:
    """Build structured leaderboard entries from benchmark JSON report files."""
    entries: list[dict[str, Any]] = []
    for path in json_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        m = data["metadata"]
        ev = data["evaluation"]
        c = ev["core"]
        ch = ev["challenge"]
        rot = ev["rotated_permuted"]
        code_stats = ev.get("code_stats", {})

        def _fmt_failures(split_dict: dict[str, Any], limit: int = 8) -> list[str]:
            items = []
            for f in split_dict.get("failures", [])[:limit]:
                err = f.get("error")
                suffix = f" ({err})" if err else ""
                items.append(
                    f"{f['name']}: expected={f['expected']}, got={f['predicted']}{suffix}"
                )
            return items

        core_ok = int(c["exact_correct"])
        chal_ok = int(ch["exact_correct"])
        rot_ok = int(rot["exact_correct"])
        overall_ok = core_ok + chal_ok + rot_ok
        overall_tot = int(c["total"]) + int(ch["total"]) + int(rot["total"])

        entry: dict[str, Any] = {
            "run_id": m["run_id"],
            "harness": m["harness"],
            "model": m["model"],
            "family": classify_model_family(m["model"]),
            "mode": m["mode"],
            "turns_used": int(m.get("turns_used", 1)),
            "wall_s": round(float(m.get("wall_time_s", 0.0)), 1),
            "loc": int(code_stats.get("lines_of_code", 0)),
            "forbidden_imports": list(code_stats.get("forbidden_imports", [])),
            "core_ok": core_ok,
            "core_tot": int(c["total"]),
            "core_pct": round(float(c["exact_accuracy"]) * 100.0, 1),
            "core_ms": round(float(c["avg_ms"]), 2),
            "core_s": round(float(c["total_s"]), 3),
            "chal_ok": chal_ok,
            "chal_tot": int(ch["total"]),
            "chal_pct": round(float(ch["exact_accuracy"]) * 100.0, 1),
            "chal_ms": round(float(ch["avg_ms"]), 2),
            "chal_s": round(float(ch["total_s"]), 3),
            "rot_ok": rot_ok,
            "rot_tot": int(rot["total"]),
            "rot_pct": round(float(rot["exact_accuracy"]) * 100.0, 1),
            "rot_ms": round(float(rot["avg_ms"]), 2),
            "rot_s": round(float(rot["total_s"]), 3),
            "overall_ok": overall_ok,
            "overall_tot": overall_tot,
            "overall_pct": round(float(ev.get("overall_exact_accuracy", 0.0)) * 100.0, 1),
            "core_failures": _fmt_failures(c),
            "chal_failures": _fmt_failures(ch),
            "rot_failures": _fmt_failures(rot),
        }
        if include_code:
            gen_path = path.parent / "generated" / f"{m['run_id']}.py"
            entry["generated_code"] = (
                gen_path.read_text(encoding="utf-8") if gen_path.exists() else ""
            )
        entries.append(entry)

    entries.sort(
        key=lambda r: (
            -r["overall_ok"],
            r["chal_s"] if r["chal_ok"] > 0 else 1e9,
            r["loc"] if r["loc"] > 0 else 1e9,
        )
    )
    for idx, entry in enumerate(entries, start=1):
        entry["rank"] = idx
    return entries

