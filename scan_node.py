"""
Scan node — workspace assistant agentic loop, powered by Claude (claude-haiku-4-5).

Loop:
    1. Send the project-analysis prompt with tool declarations.
    2. While Claude returns tool_use blocks, execute them and feed results back.
    3. When Claude stops calling tools, parse the final text into a
       ProjectUnderstanding and return it via the LangGraph state.
"""

from __future__ import annotations

import json
import re

import anthropic

from models import Gap, Module, ProjectUnderstanding
from prompts import (
    SCAN_SYSTEM_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT,
    SUMMARIZE_USER_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE,
)
from tools import ALL_TOOLS, dispatch_tool


# ──────────────────────────────────────────────
# Node entry point
# ──────────────────────────────────────────────

async def scan_node(state: dict) -> dict:
    """
    LangGraph node.

    Reads  ``state["project_path"]``
    Writes ``state["project_understanding"]``
    """
    project_path: str = state["project_path"]

    messages: list[dict] = [
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(project_path=project_path),
        }
    ]

    client = anthropic.AsyncAnthropic()
    response = None

    # ── Agentic loop ───────────────────────────────────────────────────────
    _MAX_ITERATIONS = 15
    for _iteration in range(_MAX_ITERATIONS):
        response = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=8096,
            system=SCAN_SYSTEM_PROMPT,
            tools=ALL_TOOLS,
            messages=messages,
        )

        # Natural stopping point — Claude is done
        if response.stop_reason == "end_turn":
            break

        # Extract tool_use blocks
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            break  # No tool calls, Claude finished

        # Append the assistant turn (includes tool_use blocks)
        messages.append({"role": "assistant", "content": response.content})

        # Execute every requested tool and collect tool_result blocks
        tool_results: list[dict] = []
        for tu in tool_uses:
            result = dispatch_tool(tu.name, tu.input, project_path)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result,
                }
            )

        # Feed all results back in a single user turn
        messages.append({"role": "user", "content": tool_results})
    else:
        # Hit iteration cap — ask Claude to wrap up with what it has
        messages.append(
            {
                "role": "user",
                "content": "You have reached the exploration limit. Output your ProjectUnderstanding JSON now based on what you have explored so far.",
            }
        )
        response = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=8096,
            system=SCAN_SYSTEM_PROMPT,
            tools=ALL_TOOLS,
            messages=messages,
        )

    # ── Parse final response ───────────────────────────────────────────────
    assert response is not None
    final_text = next((b.text for b in response.content if b.type == "text"), "")
    project_understanding = _parse_project_context(final_text, project_path)

    user_summary = await _generate_summary(project_understanding, client)

    return {
        **state,
        "project_understanding": project_understanding,
        "user_summary": user_summary,
        "approved": False,
        "conversation_history": [],
    }


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

async def _generate_summary(
    understanding: ProjectUnderstanding,
    client: anthropic.AsyncAnthropic,
) -> str:
    prompt = SUMMARIZE_USER_PROMPT_TEMPLATE.format(
        understanding_json=understanding.model_dump_json(indent=2),
    )
    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=8096,
        system=SUMMARIZE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return next((b.text for b in response.content if b.type == "text"), "").strip()


def _parse_project_context(text: str, project_path: str) -> ProjectUnderstanding:
    data = _extract_json(text)
    if data:
        try:
            return ProjectUnderstanding(
                project_name=data.get("project_name") or _infer_name(project_path),
                purpose=data.get("purpose", ""),
                tech_stack=data.get("tech_stack", []),
                modules=[Module(**m) for m in data.get("modules", [])],
                gaps=[Gap(**g) for g in data.get("gaps", [])],
                recent_focus=data.get("recent_focus", ""),
                notable_observations=data.get("notable_observations", []),
                raw_summary=text,
            )
        except Exception:  # noqa: BLE001
            pass

    return ProjectUnderstanding(
        project_name=_infer_name(project_path),
        purpose="(parsing failed — see raw_summary)",
        raw_summary=text,
    )


def _extract_json(text: str) -> dict | None:
    # 1. Fenced ```json ... ``` block
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Last bare { ... } object in the text
    last_brace = text.rfind("{")
    if last_brace != -1:
        try:
            return json.loads(text[last_brace:])
        except json.JSONDecodeError:
            pass

    return None


def _infer_name(project_path: str) -> str:
    from pathlib import Path
    return Path(project_path).name or "unknown-project"
