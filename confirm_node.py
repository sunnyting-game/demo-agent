"""
Confirm node — interactive loop that shows the agent's understanding to the
developer and waits for approval.

Loop:
    1. Print the current ~100-word user_summary.
    2. Read developer input.
    3. If input is "approve" (case-insensitive) → mark approved and exit.
    4. Otherwise → call Claude to enhance the internal ProjectUnderstanding
       using the conversation history, extract the new summary, and loop back.
"""

from __future__ import annotations

import anthropic

import display
from models import Gap, Module, ProjectUnderstanding
from prompts import ENHANCE_SYSTEM_PROMPT, ENHANCE_USER_PROMPT_TEMPLATE
from scan_node import _extract_json, _parse_project_context


# ──────────────────────────────────────────────
# Node entry point
# ──────────────────────────────────────────────

async def confirm_node(state: dict) -> dict:
    """
    LangGraph node.

    Reads  ``state["user_summary"]``, ``state["project_understanding"]``,
            ``state["conversation_history"]``
    Writes ``state["approved"]``, and on each enhancement loop:
            ``state["project_understanding"]``, ``state["user_summary"]``,
            ``state["conversation_history"]``
    """
    user_summary: str = state["user_summary"]
    understanding: ProjectUnderstanding = state["project_understanding"]
    history: list[dict] = list(state.get("conversation_history", []))

    display.print_summary(user_summary)
    display.print_confirm_prompt()

    user_input = display.ask_user()

    _APPROVAL_WORDS = {"approve", "yes", "y", "correct", "looks good", "good", "ok", "okay", "yep", "yup", "move on", "proceed", "confirmed", "confirm", "right"}
    if user_input.lower() in _APPROVAL_WORDS or user_input.lower().startswith("yes"):
        return {**state, "approved": True}

    # ── Enhance internal understanding ─────────────────────────────────────
    history.append({"role": "user", "content": user_input})

    with display.thinking("Updating my understanding..."):
        new_understanding, new_summary = await _enhance(understanding, history)

    history.append({"role": "assistant", "content": new_summary})

    return {
        **state,
        "project_understanding": new_understanding,
        "user_summary": new_summary,
        "conversation_history": history,
        "approved": False,
    }


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

async def _enhance(
    understanding: ProjectUnderstanding,
    history: list[dict],
) -> tuple[ProjectUnderstanding, str]:
    conversation_str = "\n".join(
        f"{'Developer' if m['role'] == 'user' else 'Agent'}: {m['content']}"
        for m in history
    )
    prompt = ENHANCE_USER_PROMPT_TEMPLATE.format(
        understanding_json=understanding.model_dump_json(indent=2),
        conversation=conversation_str,
    )

    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=8096,
        system=ENHANCE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "").strip()

    # Split: summary is everything before the ```json block
    json_start = text.find("```json")
    if json_start > 0:
        new_summary = text[:json_start].strip()
        json_text = text[json_start:]
    else:
        new_summary = text[:500].strip()
        json_text = text

    # Parse updated understanding; fall back to existing if parsing fails
    data = _extract_json(json_text)
    if data:
        try:
            new_understanding = ProjectUnderstanding(
                project_name=data.get("project_name", understanding.project_name),
                purpose=data.get("purpose", understanding.purpose),
                tech_stack=data.get("tech_stack", understanding.tech_stack),
                modules=[Module(**m) for m in data.get("modules", [])],
                gaps=[Gap(**g) for g in data.get("gaps", [])],
                recent_focus=data.get("recent_focus", understanding.recent_focus),
                notable_observations=data.get(
                    "notable_observations", understanding.notable_observations
                ),
                raw_summary=text,
            )
            return new_understanding, new_summary
        except Exception:  # noqa: BLE001
            pass

    # Parsing failed — keep existing understanding, just update raw_summary
    updated = understanding.model_copy(update={"raw_summary": text})
    return updated, new_summary
