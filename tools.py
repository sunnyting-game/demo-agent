"""
Tool definitions (Anthropic format) and their executors.

Four tools are provided:
  glob  — find files by pattern
  grep  — search file content by regex
  read  — read a file
  bash  — run a shell command
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

# ──────────────────────────────────────────────
# Anthropic tool definitions
# ──────────────────────────────────────────────

GLOB_TOOL: dict = {
    "name": "glob",
    "description": (
        "Find files matching a glob pattern inside the project. "
        "Use ** for recursive search, e.g. '**/*.py' or 'src/**/*.ts'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern relative to the project root.",
            },
            "root": {
                "type": "string",
                "description": (
                    "Sub-directory to search from, relative to the project root. "
                    "Omit to search from the project root."
                ),
            },
        },
        "required": ["pattern"],
    },
}

GREP_TOOL: dict = {
    "name": "grep",
    "description": (
        "Search for a regex pattern in file contents. "
        "Returns matching lines with file paths and line numbers."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex (or plain string) to search for.",
            },
            "path": {
                "type": "string",
                "description": "File or directory path (relative to project root) to search in.",
            },
            "include": {
                "type": "string",
                "description": "Optional file glob filter, e.g. '*.py' or '*.{ts,tsx}'.",
            },
        },
        "required": ["pattern", "path"],
    },
}

READ_TOOL: dict = {
    "name": "read",
    "description": "Read the contents of a single file.",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file (relative to the project root).",
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum lines to return (default: 200).",
            },
        },
        "required": ["file_path"],
    },
}

BASH_TOOL: dict = {
    "name": "bash",
    "description": (
        "Run a bash/shell command inside the project directory. "
        "Useful for 'git log', 'ls -la', 'find', etc. "
        "Timeout: 30 seconds."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute.",
            },
        },
        "required": ["command"],
    },
}

ALL_TOOLS: list[dict] = [GLOB_TOOL, GREP_TOOL, READ_TOOL, BASH_TOOL]


# ──────────────────────────────────────────────
# Tool executor
# ──────────────────────────────────────────────

def execute_tools(content_blocks, project_root: str) -> list[dict]:
    """
    Execute every tool_use block found in *content_blocks* and return a list
    of tool_result dicts ready to be appended as a user message.
    """
    results: list[dict] = []
    for block in content_blocks:
        if block.type != "tool_use":
            continue
        output = _dispatch(block.name, block.input, project_root)
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            }
        )
    return results


_MAX_TOOL_OUTPUT_CHARS = 4000


def dispatch_tool(name: str, inputs: dict, project_root: str) -> str:
    """Public entry point for executing a single tool by name."""
    result = _dispatch(name, inputs, project_root)
    if len(result) > _MAX_TOOL_OUTPUT_CHARS:
        omitted = len(result) - _MAX_TOOL_OUTPUT_CHARS
        result = result[:_MAX_TOOL_OUTPUT_CHARS] + f"\n... ({omitted} chars omitted)"
    return result


def _dispatch(name: str, inputs: dict, project_root: str) -> str:
    try:
        if name == "glob":
            return _run_glob(inputs, project_root)
        if name == "grep":
            return _run_grep(inputs, project_root)
        if name == "read":
            return _run_read(inputs, project_root)
        if name == "bash":
            return _run_bash(inputs, project_root)
        return f"[error] Unknown tool: {name}"
    except Exception as exc:  # noqa: BLE001
        return f"[error] {name} failed: {exc}"


# ──────────────────────────────────────────────
# Individual runners
# ──────────────────────────────────────────────

def _resolve(path: str, project_root: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(project_root) / p


def _run_glob(inputs: dict, project_root: str) -> str:
    pattern: str = inputs["pattern"]
    root_str: str = inputs.get("root", "")
    root = _resolve(root_str, project_root) if root_str else Path(project_root)

    matches = sorted(root.glob(pattern))
    if not matches:
        return f"No files found matching: {pattern}"

    cap = 200
    lines = [str(m.relative_to(root)) for m in matches[:cap]]
    result = f"Found {len(matches)} file(s):\n" + "\n".join(lines)
    if len(matches) > cap:
        result += f"\n... and {len(matches) - cap} more"
    return result


def _run_grep(inputs: dict, project_root: str) -> str:
    pattern: str = inputs["pattern"]
    path_str: str = inputs["path"]
    include: str = inputs.get("include", "")

    abs_path = _resolve(path_str, project_root)

    # Compile regex (fall back to literal match on bad pattern)
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)

    # Gather files to search
    if abs_path.is_file():
        files = [abs_path]
    elif abs_path.is_dir():
        glob_pat = f"**/{include}" if include else "**/*"
        files = sorted(f for f in abs_path.glob(glob_pat) if f.is_file())[:100]
    else:
        return f"[error] Path not found: {path_str}"

    hits: list[str] = []
    for file_path in files:
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as fh:
                for lineno, line in enumerate(fh, 1):
                    if regex.search(line):
                        rel = file_path.relative_to(
                            abs_path if abs_path.is_dir() else abs_path.parent
                        )
                        hits.append(f"{rel}:{lineno}: {line.rstrip()}")
                        if len(hits) >= 100:
                            break
        except OSError:
            continue
        if len(hits) >= 100:
            break

    if not hits:
        return f"No matches found for: {pattern}"
    result = "\n".join(hits)
    if len(hits) == 100:
        result += "\n... (output capped at 100 matches)"
    return result


def _run_read(inputs: dict, project_root: str) -> str:
    file_path_str: str = inputs["file_path"]
    max_lines: int = int(inputs.get("max_lines", 200))

    abs_path = _resolve(file_path_str, project_root)
    if not abs_path.exists():
        return f"[error] File not found: {file_path_str}"
    if abs_path.is_dir():
        return f"[error] Path is a directory, not a file: {file_path_str}"

    with open(abs_path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    total = len(lines)
    content = "".join(lines[:max_lines])
    if total > max_lines:
        content += f"\n... ({total - max_lines} more lines, use max_lines to read more)"
    return content


def _run_bash(inputs: dict, project_root: str) -> str:
    command: str = inputs["command"]
    proc = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=project_root,
    )
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()

    if stdout and stderr:
        return f"{stdout}\n[stderr] {stderr}"
    return stdout or stderr or "(no output)"
