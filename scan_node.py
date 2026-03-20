"""
Scan node — workspace assistant agentic loop, powered by Gemini 2.5 Flash.

Loop:
    1. Send the project-analysis prompt with tool declarations.
    2. While Gemini returns function calls, execute them and feed results back.
    3. When Gemini stops calling tools, parse the final text into a
       ProjectUnderstanding and return it via the LangGraph state.
"""

from __future__ import annotations

import json
import re

from google import genai
from google.genai import types

from models import Gap, Module, ProjectUnderstanding
from prompts import (
    SCAN_SYSTEM_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT,
    SUMMARIZE_USER_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE,
)
from tools import ALL_TOOLS, dispatch_tool


# ──────────────────────────────────────────────
# Build Gemini tool declarations once at import time
# ALL_TOOLS is a list of Anthropic-style dicts; Gemini accepts the same
# JSON-Schema dict for `parameters`, so we just rewrap them.
# ──────────────────────────────────────────────

_GEMINI_TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=t["input_schema"],
            )
            for t in ALL_TOOLS
        ]
    )
]

_GEMINI_CONFIG = types.GenerateContentConfig(
    system_instruction=SCAN_SYSTEM_PROMPT,
    tools=_GEMINI_TOOLS,
)


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

    contents: list[types.Content] = [
        types.Content(
            role="user",
            parts=[types.Part(text=USER_PROMPT_TEMPLATE.format(project_path=project_path))],
        )
    ]

    client = genai.Client()
    response = None

    # ── Agentic loop ───────────────────────────────────────────────────────
    _MAX_ITERATIONS = 15
    for _iteration in range(_MAX_ITERATIONS):
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=_GEMINI_CONFIG,
        )

        # response.function_calls is [] when Gemini is done
        fn_calls = response.function_calls or []
        if not fn_calls:
            break  # Natural stopping point — no more tool requests

        # Append the assistant turn (includes the function_call parts)
        contents.append(response.candidates[0].content)

        # Execute every requested tool and collect function responses
        fn_response_parts: list[types.Part] = []
        for fc in fn_calls:
            result = dispatch_tool(fc.name, dict(fc.args), project_path)
            fn_response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": result},
                    )
                )
            )

        # Feed all results back in a single user turn
        contents.append(types.Content(role="user", parts=fn_response_parts))
    else:
        # Hit iteration cap — ask Gemini to wrap up with what it has
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text="You have reached the exploration limit. Output your ProjectUnderstanding JSON now based on what you have explored so far.")],
            )
        )
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=_GEMINI_CONFIG,
        )

    # ── Parse final response ───────────────────────────────────────────────
    assert response is not None
    final_text = response.text or ""
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

async def _generate_summary(understanding: ProjectUnderstanding, client: genai.Client) -> str:
    prompt = SUMMARIZE_USER_PROMPT_TEMPLATE.format(
        understanding_json=understanding.model_dump_json(indent=2),
    )
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=types.GenerateContentConfig(system_instruction=SUMMARIZE_SYSTEM_PROMPT),
    )
    return (response.text or "").strip()


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
