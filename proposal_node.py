"""
Proposal node — filters opportunities down to 1–3 that are achievable
with the agent's available tools and outputs them directly.
"""

from __future__ import annotations

import json
import re

import anthropic

import display
from display import print_proposals
from models import Opportunity, Proposal, ProjectGoal, ProjectUnderstanding
from prompts import (
    PROPOSAL_SYSTEM_PROMPT,
    PROPOSAL_USER_PROMPT_TEMPLATE,
)


def _extract_json_array(text: str) -> list | None:
    """Extract a JSON array from a fenced ```json block or bare [...] in text."""
    # 1. Fenced ```json ... ``` block (array variant)
    fenced = re.search(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Last bare [ ... ] array in the text
    last_bracket = text.rfind("[")
    if last_bracket != -1:
        try:
            return json.loads(text[last_bracket:])
        except json.JSONDecodeError:
            pass

    return None


# ──────────────────────────────────────────────
# Node entry point
# ──────────────────────────────────────────────

async def proposal_node(state: dict) -> dict:
    """
    LangGraph node.

    Reads  ``state["opportunities"]``, ``state["project_goal"]``,
            ``state["project_understanding"]``
    Writes ``state["proposals"]``
    """
    opportunities: list[Opportunity] = state["opportunities"]
    project_goal: ProjectGoal = state["project_goal"]
    understanding: ProjectUnderstanding = state["project_understanding"]

    with display.thinking("Preparing proposals..."):
        _, proposals = await _present(opportunities, project_goal, understanding)

    print_proposals(proposals)

    return {**state, "proposals": proposals}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

async def _present(
    opportunities: list[Opportunity],
    project_goal: ProjectGoal,
    understanding: ProjectUnderstanding,
) -> tuple[str, list[Proposal]]:
    """Initial Claude call to select and present 1–3 proposals."""
    opportunities_json = json.dumps(
        [o.model_dump() for o in opportunities], indent=2
    )
    prompt = PROPOSAL_USER_PROMPT_TEMPLATE.format(
        goal_json=project_goal.model_dump_json(indent=2),
        opportunities_json=opportunities_json,
        understanding_json=understanding.model_dump_json(indent=2),
    )

    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=8096,
        system=PROPOSAL_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "").strip()

    return _split_response(text, fallback_proposals=[])



def _split_response(
    text: str,
    fallback_proposals: list[Proposal],
) -> tuple[str, list[Proposal]]:
    """
    Split Claude response into (presentation_text, list[Proposal]).

    The response format is:
        <presentation text>

        ```json
        [{"title": ..., "what": ..., "approach": ..., "effort": ...}, ...]
        ```
    """
    json_start = text.find("```json")
    if json_start > 0:
        presentation_text = text[:json_start].strip()
        json_text = text[json_start:]
    else:
        presentation_text = text
        json_text = text

    data = _extract_json_array(json_text)
    if data:
        try:
            proposals = [Proposal(**item) for item in data]
            return presentation_text, proposals
        except Exception:  # noqa: BLE001
            pass

    # Parsing failed — keep existing proposals, show full text
    return presentation_text, fallback_proposals
