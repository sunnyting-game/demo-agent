"""
Entry point.

Usage:
    python main.py <project_path>

Example:
    python main.py C:/Users/me/my-project
"""

from __future__ import annotations

import asyncio
import datetime
import sys
from pathlib import Path

import display
from agent import build_agent

# Result logs are written to this folder (relative to main.py)
_RESULT_DIR = Path(__file__).parent / "result"


async def main() -> None:
    if len(sys.argv) < 2:
        display.print_usage()
        sys.exit(1)

    project_path = sys.argv[1]
    if not Path(project_path).exists():
        display.print_error(f"path does not exist: {project_path}")
        sys.exit(1)

    # Generate unique session ID from current timestamp
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    display.print_scanning(project_path)

    agent = build_agent()
    final_state = await agent.ainvoke(
        {
            "project_path": project_path,
            "project_understanding": None,
            "user_summary": "",
            "approved": False,
            "conversation_history": [],
            "opportunities": None,
        }
    )

    _save_log(session_id, project_path, final_state)
    display.print_done()
    display.console.print(
        f"[dim]Session log saved → result/log_{session_id}.md[/dim]\n"
    )


def _save_log(session_id: str, project_path: str, state: dict) -> None:
    """Write a markdown session log to result/log_<session_id>.md."""
    _RESULT_DIR.mkdir(exist_ok=True)
    log_path = _RESULT_DIR / f"log_{session_id}.md"

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────
    lines += [
        f"# Session Log: {session_id}",
        "",
        f"**Date:** {now}  ",
        f"**Project:** {project_path}",
        "",
        "---",
        "",
    ]

    # ── Project Understanding ─────────────────────────────────────────────
    understanding = state.get("project_understanding")
    user_summary = state.get("user_summary", "")

    lines += ["## Project Understanding", ""]
    if understanding:
        lines += [
            f"**Name:** {understanding.project_name}  ",
            f"**Purpose:** {understanding.purpose}",
            "",
        ]
        if understanding.tech_stack:
            lines += ["**Tech Stack:**", ""]
            for tech in understanding.tech_stack:
                lines.append(f"- {tech}")
            lines.append("")

        if understanding.modules:
            lines += ["**Modules:**", ""]
            for m in understanding.modules:
                lines.append(f"- **{m.name}** `{m.status}` — {m.description}")
            lines.append("")

        if understanding.notable_observations:
            lines += ["**Notable Observations:**", ""]
            for obs in understanding.notable_observations:
                lines.append(f"- {obs}")
            lines.append("")

        if user_summary:
            lines += ["**Summary:**", "", user_summary, ""]
    else:
        lines += ["*(not captured)*", ""]

    lines += ["---", ""]

    # ── Project Goal ──────────────────────────────────────────────────────
    goal = state.get("project_goal")
    lines += ["## Project Goal", ""]
    if goal:
        lines += [goal.project_goal, ""]
        if goal.success_criteria:
            lines += ["**Success Criteria:**", ""]
            for c in goal.success_criteria:
                lines.append(f"- {c}")
            lines.append("")
        if goal.value_drivers:
            lines += ["**Value Drivers:**", ""]
            for v in goal.value_drivers:
                lines.append(f"- {v}")
            lines.append("")
    else:
        lines += ["*(not captured)*", ""]

    lines += ["---", ""]

    # ── Opportunities ─────────────────────────────────────────────────────
    opportunities = state.get("opportunities") or []
    lines += ["## Opportunities", ""]
    if opportunities:
        for i, opp in enumerate(opportunities, 1):
            lines += [
                f"### {i}. {opp.title}",
                "",
                f"**What:** {opp.what}  ",
                f"**Why:** {opp.why}  ",
                f"**Value:** {' + '.join(opp.value)}",
                "",
            ]
    else:
        lines += ["*(not captured)*", ""]

    lines += ["---", ""]

    # ── Approved Proposals ────────────────────────────────────────────────
    proposals = state.get("proposals") or []
    lines += ["## Approved Proposals", ""]
    if proposals:
        for i, p in enumerate(proposals, 1):
            lines += [
                f"### {i}. {p.title}",
                "",
                f"**What:** {p.what}  ",
                f"**Approach:** {p.approach}  ",
                f"**Effort:** {p.effort}",
                "",
            ]
    else:
        lines += ["*(none approved)*", ""]

    log_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
