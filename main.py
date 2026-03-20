"""
Entry point.

Usage:
    python main.py <project_path>

Example:
    python main.py C:/Users/me/my-project
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import display
from agent import build_agent


async def main() -> None:
    if len(sys.argv) < 2:
        display.print_usage()
        sys.exit(1)

    project_path = sys.argv[1]
    if not Path(project_path).exists():
        display.print_error(f"path does not exist: {project_path}")
        sys.exit(1)

    display.print_scanning(project_path)

    agent = build_agent()
    await agent.ainvoke(
        {
            "project_path": project_path,
            "project_understanding": None,
            "user_summary": "",
            "approved": False,
            "conversation_history": [],
            "opportunities": None,
        }
    )

    display.print_done()


if __name__ == "__main__":
    asyncio.run(main())
