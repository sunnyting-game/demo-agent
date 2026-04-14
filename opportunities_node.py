"""
Opportunities node — identifies 5 high-value opportunities from the
post-confirmation project understanding.

Single-turn Claude call (no tool loop).
Output is printed for user transparency, then parsed into structured
Opportunity objects for the proposal node to consume.
"""

from __future__ import annotations

import re

import anthropic

import display
from display import print_opportunities
from models import Opportunity, ProjectGoal, ProjectUnderstanding
from prompts import OPPORTUNITIES_SYSTEM_PROMPT, OPPORTUNITIES_USER_PROMPT_TEMPLATE


# ──────────────────────────────────────────────
# Node entry point
# ──────────────────────────────────────────────

async def opportunities_node(state: dict) -> dict:
    """
    LangGraph node.

    Reads  ``state["project_understanding"]``, ``state["user_summary"]``,
            ``state["project_goal"]``
    Writes ``state["opportunities"]``
    """
    understanding: ProjectUnderstanding = state["project_understanding"]
    user_summary: str = state["user_summary"]
    project_goal: ProjectGoal = state["project_goal"]

    prompt = OPPORTUNITIES_USER_PROMPT_TEMPLATE.format(
        goal_json=project_goal.model_dump_json(indent=2),
        user_summary=user_summary,
        understanding_json=understanding.model_dump_json(indent=2),
    )

    client = anthropic.AsyncAnthropic()
    with display.thinking("Finding opportunities..."):
        response = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=8096,
            system=OPPORTUNITIES_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    raw_text = next((b.text for b in response.content if b.type == "text"), "").strip()

    opportunities = _parse_opportunities(raw_text)

    print_opportunities(opportunities)

    return {**state, "opportunities": opportunities}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _parse_opportunities(text: str) -> list[Opportunity]:
    """
    Parse the structured markdown response into Opportunity objects.

    Expected format per opportunity:
        **Opportunity N: Title**
        - What: ...
        - Why: ...
        - Value: Execution + Quality
    """
    # Split on each "**Opportunity N:" header, keeping the title
    blocks = re.split(r"\*\*Opportunity\s+\d+:\s*", text)
    # First element is text before the first opportunity — discard it
    blocks = [b.strip() for b in blocks[1:] if b.strip()]

    opportunities: list[Opportunity] = []
    for block in blocks:
        try:
            opportunities.append(_parse_block(block))
        except Exception:  # noqa: BLE001
            continue  # Skip malformed blocks rather than failing entirely

    return opportunities


def _parse_block(block: str) -> Opportunity:
    """Parse a single opportunity block starting after 'Opportunity N: '."""
    # First line is the title (ends at the closing **)
    first_line, _, rest = block.partition("\n")
    title = first_line.rstrip("*").strip()

    what = _extract_field(rest, "What")
    why = _extract_field(rest, "Why")
    value_raw = _extract_field(rest, "Value")

    # Parse "Execution + Quality + Leverage" → ["Execution", "Quality", "Leverage"]
    value_tags = [v.strip() for v in re.split(r"\s*\+\s*", value_raw) if v.strip()]

    return Opportunity(title=title, what=what, why=why, value=value_tags)


def _extract_field(text: str, field: str) -> str:
    """Extract the value of a '- Field: ...' line from a block."""
    match = re.search(rf"[-*]\s*{field}:\s*(.+?)(?=\n[-*]|\Z)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""
