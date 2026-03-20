"""
Goal node — infers the real-world purpose and success criteria of the project
from the post-confirmation project understanding.

Single-turn Gemini call (no tool loop).
Output is printed for user transparency, then parsed into a structured
ProjectGoal object for the opportunities node to consume.
"""

from __future__ import annotations

from google import genai
from google.genai import types

import display
from display import print_goal
from models import ProjectGoal, ProjectUnderstanding
from prompts import GOAL_SYSTEM_PROMPT, GOAL_USER_PROMPT_TEMPLATE
from scan_node import _extract_json


# ──────────────────────────────────────────────
# Node entry point
# ──────────────────────────────────────────────

async def goal_node(state: dict) -> dict:
    """
    LangGraph node.

    Reads  ``state["project_understanding"]``, ``state["user_summary"]``
    Writes ``state["project_goal"]``
    """
    understanding: ProjectUnderstanding = state["project_understanding"]
    user_summary: str = state["user_summary"]

    prompt = GOAL_USER_PROMPT_TEMPLATE.format(
        user_summary=user_summary,
        understanding_json=understanding.model_dump_json(indent=2),
    )

    client = genai.Client()
    with display.thinking("Identifying project goal..."):
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(system_instruction=GOAL_SYSTEM_PROMPT),
        )
    raw_text = (response.text or "").strip()

    project_goal = _parse_goal(raw_text)

    print_goal(project_goal)

    return {**state, "project_goal": project_goal}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _parse_goal(text: str) -> ProjectGoal:
    """
    Parse the JSON block from the goal node response into a ProjectGoal object.
    Falls back to sensible defaults if parsing fails.
    """
    data = _extract_json(text)
    if data:
        try:
            return ProjectGoal(
                project_goal=data.get("project_goal", ""),
                success_criteria=data.get("success_criteria", []),
                value_drivers=data.get("value_drivers", []),
            )
        except Exception:  # noqa: BLE001
            pass

    # Fallback — return empty goal so the pipeline can continue
    return ProjectGoal(project_goal=text[:300], success_criteria=[], value_drivers=[])
