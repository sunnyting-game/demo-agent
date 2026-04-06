"""
Proposal node — filters the 5 opportunities down to 1–3 that are achievable
with the agent's available tools, presents them to the user, and loops until
the user explicitly approves.

Interactive loop (same pattern as confirm_node).
Output is printed for transparency, then parsed into structured Proposal objects.
"""

from __future__ import annotations

import json
import re

import anthropic

import display
from display import print_proposals
from models import Opportunity, Proposal, ProjectGoal, ProjectUnderstanding
from prompts import (
    PROPOSAL_REFINE_PROMPT_TEMPLATE,
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
            ``state["project_understanding"]``,
            ``state["proposal_conversation_history"]``
    Writes ``state["proposals"]``, ``state["proposals_approved"]``,
            ``state["proposal_conversation_history"]``
    """
    opportunities: list[Opportunity] = state["opportunities"]
    project_goal: ProjectGoal = state["project_goal"]
    understanding: ProjectUnderstanding = state["project_understanding"]
    history: list[dict] = list(state.get("proposal_conversation_history", []))

    # ── Initial presentation on first entry (no history yet) ────────────────
    if not history:
        with display.thinking("Preparing proposals..."):
            presentation_text, proposals = await _present(
                opportunities, project_goal, understanding
            )
    else:
        # Re-entering after a refine cycle — restore from state
        proposals = state.get("proposals") or []
        presentation_text = history[-1].get("content", "") if history else ""

    print_proposals(proposals)
    display.print_proposal_prompt()

    user_input = display.ask_user()

    _APPROVAL_WORDS = {
        "approve", "yes", "y", "correct", "looks good", "good", "ok", "okay",
        "yep", "yup", "move on", "proceed", "confirmed", "confirm", "right",
    }
    if user_input.lower() in _APPROVAL_WORDS or user_input.lower().startswith("yes"):
        return {**state, "proposals": proposals, "proposals_approved": True}

    # ── Refine based on user feedback ──────────────────────────────────────
    history.append({"role": "user", "content": user_input})

    with display.thinking("Updating proposals..."):
        new_proposals, new_presentation = await _refine(proposals, history)

    history.append({"role": "assistant", "content": new_presentation})

    return {
        **state,
        "proposals": new_proposals,
        "proposals_approved": False,
        "proposal_conversation_history": history,
    }


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


async def _refine(
    proposals: list[Proposal],
    history: list[dict],
) -> tuple[list[Proposal], str]:
    """Refine proposals based on the accumulated conversation history."""
    conversation_str = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Agent'}: {m['content']}"
        for m in history
    )
    proposals_json = json.dumps([p.model_dump() for p in proposals], indent=2)
    prompt = PROPOSAL_REFINE_PROMPT_TEMPLATE.format(
        proposals_json=proposals_json,
        conversation=conversation_str,
    )

    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=8096,
        system=PROPOSAL_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "").strip()

    return _split_response(text, fallback_proposals=proposals)


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
