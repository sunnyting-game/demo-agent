"""
Tool definitions (Anthropic format) and executors for the proactive workspace agent.

Seven tools:
  web_search      — search the web and return titled results
  web_fetch       — fetch and extract main text from a URL
  read            — read a file's contents
  write           — write content to a file
  execute_command — run a shell command with safety checks
  glob            — find files by glob pattern
  grep            — search file contents by pattern

Each tool has two interfaces:
  - A direct Python function returning a ToolResult (ok, output, error) for agent chaining.
  - A dispatch_tool() entry point that serialises to a plain string for LLM consumption.
"""

from __future__ import annotations

import re
import subprocess
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import TypedDict


# ──────────────────────────────────────────────
# Structured result type
# ──────────────────────────────────────────────

class ToolResult(TypedDict):
    ok: bool
    output: str  # non-empty on success
    error: str   # non-empty on failure


def _ok(output: str) -> ToolResult:
    return {"ok": True, "output": output, "error": ""}


def _err(message: str) -> ToolResult:
    return {"ok": False, "output": "", "error": message}


# ──────────────────────────────────────────────
# Anthropic tool definitions
# ──────────────────────────────────────────────

WEB_SEARCH_TOOL: dict = {
    "name": "web_search",
    "description": (
        "Search the web and return a list of results (title, url, snippet). "
        "Use for finding documentation, articles, or current information online."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default: 5, max: 20).",
            },
        },
        "required": ["query"],
    },
}

WEB_FETCH_TOOL: dict = {
    "name": "web_fetch",
    "description": (
        "Fetch a webpage and return its main text content with HTML stripped. "
        "Useful for reading documentation pages or articles."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Full URL to fetch (must start with http:// or https://).",
            },
        },
        "required": ["url"],
    },
}

READ_TOOL: dict = {
    "name": "read",
    "description": "Read and return the contents of a file.",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file (absolute or relative to the project root).",
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of lines to return (default: 200).",
            },
        },
        "required": ["file_path"],
    },
}

WRITE_TOOL: dict = {
    "name": "write",
    "description": (
        "Write content to a file, creating the file (and any parent directories) "
        "if they do not exist. Overwrites the file if it already exists."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Destination file path (absolute or relative to the project root).",
            },
            "content": {
                "type": "string",
                "description": "Text content to write to the file.",
            },
        },
        "required": ["file_path", "content"],
    },
}

EXECUTE_COMMAND_TOOL: dict = {
    "name": "execute_command",
    "description": (
        "Run a shell command and return stdout, stderr, and exit code. "
        "Destructive operations (rm -r, del /f, format, DROP TABLE, etc.) are "
        "blocked unless allow_destructive=true is set. Timeout: 30 seconds."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Executable name or shell command.",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Arguments to pass to the command.",
            },
            "working_dir": {
                "type": "string",
                "description": "Directory in which to run the command (default: project root).",
            },
            "allow_destructive": {
                "type": "boolean",
                "description": "Set true to permit destructive operations (default: false).",
            },
        },
        "required": ["command"],
    },
}

GLOB_TOOL: dict = {
    "name": "glob",
    "description": (
        "Return a list of file paths matching a glob pattern within a base directory. "
        "Use ** for recursive search, e.g. '**/*.py' or 'src/**/*.ts'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern, e.g. '**/*.py' or 'tests/**/*.json'.",
            },
            "base_dir": {
                "type": "string",
                "description": (
                    "Directory to search from (absolute or relative to project root). "
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
        "Search for a string or regex pattern across files at a given path. "
        "Returns matching file paths and the matching lines with line numbers."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Pattern to search for.",
            },
            "path": {
                "type": "string",
                "description": "File or directory path to search in (relative to project root).",
            },
            "is_regex": {
                "type": "boolean",
                "description": (
                    "Treat pattern as a regular expression (default: false — plain string match)."
                ),
            },
        },
        "required": ["pattern", "path"],
    },
}

ALL_TOOLS: list[dict] = [
    WEB_SEARCH_TOOL,
    WEB_FETCH_TOOL,
    READ_TOOL,
    WRITE_TOOL,
    EXECUTE_COMMAND_TOOL,
    GLOB_TOOL,
    GREP_TOOL,
]


# ──────────────────────────────────────────────
# Dispatch layer  (plain-string output for LLM)
# ──────────────────────────────────────────────

_MAX_OUTPUT_CHARS = 4000


def dispatch_tool(name: str, inputs: dict, project_root: str) -> str:
    """Execute a tool by name and return a plain-string result for LLM consumption."""
    result = _dispatch(name, inputs, project_root)
    text = result["output"] if result["ok"] else f"[error] {result['error']}"
    if len(text) > _MAX_OUTPUT_CHARS:
        omitted = len(text) - _MAX_OUTPUT_CHARS
        text = text[:_MAX_OUTPUT_CHARS] + f"\n... ({omitted} chars omitted)"
    return text


def _dispatch(name: str, inputs: dict, project_root: str) -> ToolResult:
    try:
        if name == "web_search":
            return web_search(inputs["query"], int(inputs.get("num_results", 5)))

        if name == "web_fetch":
            return web_fetch(inputs["url"])

        if name == "read":
            path = _resolve(inputs["file_path"], project_root)
            return read(path, int(inputs.get("max_lines", 200)))

        if name == "write":
            path = _resolve(inputs["file_path"], project_root)
            return write(path, inputs["content"])

        if name == "execute_command":
            return execute_command(
                inputs["command"],
                inputs.get("args", []),
                inputs.get("working_dir") or project_root,
                bool(inputs.get("allow_destructive", False)),
            )

        if name == "glob":
            base = (
                str(_resolve(inputs["base_dir"], project_root))
                if inputs.get("base_dir")
                else project_root
            )
            return glob(inputs["pattern"], base)

        if name == "grep":
            abs_path = str(_resolve(inputs["path"], project_root))
            return grep(inputs["pattern"], abs_path, bool(inputs.get("is_regex", False)))

        return _err(f"Unknown tool: {name}")

    except KeyError as exc:
        return _err(f"Missing required input: {exc}")
    except Exception as exc:  # noqa: BLE001
        return _err(f"{name} failed unexpectedly: {exc}")


# ──────────────────────────────────────────────
# Path helper
# ──────────────────────────────────────────────

def _resolve(path: str, root: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(root) / p


# ──────────────────────────────────────────────
# Tool implementations  (return ToolResult)
# ──────────────────────────────────────────────

def web_search(query: str, num_results: int = 5) -> ToolResult:
    """Search the web and return results as title / url / snippet triples."""
    num_results = min(max(1, num_results), 20)
    try:
        from duckduckgo_search import DDGS  # optional dependency
    except ImportError:
        return _err(
            "duckduckgo_search package is not installed — "
            "run: pip install duckduckgo-search"
        )

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=num_results))
    except Exception as exc:  # noqa: BLE001
        return _err(f"Search failed: {exc}")

    if not raw:
        return _ok(f"No results found for: {query}")

    lines: list[str] = [f"Search results for: {query}\n"]
    for i, item in enumerate(raw, 1):
        title = item.get("title", "(no title)")
        url = item.get("href", "")
        snippet = item.get("body", "")
        lines.append(f"{i}. {title}\n   URL: {url}\n   {snippet}")
    return _ok("\n\n".join(lines))


# ── HTML → plain text ────────────────────────

class _TextExtractor(HTMLParser):
    """Minimal HTML-to-plain-text converter."""

    _SKIP_TAGS = {"script", "style", "head", "noscript", "svg", "meta", "link"}
    _BLOCK_TAGS = {
        "p", "div", "li", "br", "tr",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "article", "section", "header", "footer", "blockquote", "pre",
    }

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def web_fetch(url: str) -> ToolResult:
    """Fetch a URL and return its main text content with HTML stripped."""
    if not url.startswith(("http://", "https://")):
        return _err(f"Invalid URL (must start with http:// or https://): {url}")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; workspace-agent/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get_content_type()
            if content_type and "html" not in content_type and "text" not in content_type:
                return _err(
                    f"URL does not return HTML/text content "
                    f"(content-type: {content_type})"
                )
            raw_bytes = resp.read(1_000_000)  # cap at ~1 MB
    except urllib.error.HTTPError as exc:
        return _err(f"HTTP {exc.code}: {exc.reason} — {url}")
    except urllib.error.URLError as exc:
        return _err(f"Failed to reach {url}: {exc.reason}")
    except Exception as exc:  # noqa: BLE001
        return _err(f"Fetch failed: {exc}")

    html = raw_bytes.decode("utf-8", errors="replace")
    extractor = _TextExtractor()
    extractor.feed(html)
    text = extractor.text()

    if not text:
        return _err("Page returned no readable text content")
    return _ok(f"[{url}]\n\n{text}")


def read(file_path: Path | str, max_lines: int = 200) -> ToolResult:
    """Read and return the contents of a file."""
    p = Path(file_path)
    if not p.exists():
        return _err(f"File not found: {file_path}")
    if p.is_dir():
        return _err(f"Path is a directory, not a file: {file_path}")

    try:
        with open(p, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        return _err(f"Cannot read file: {exc}")

    total = len(lines)
    content = "".join(lines[:max_lines])
    if total > max_lines:
        content += f"\n... ({total - max_lines} more lines)"
    return _ok(content)


def write(file_path: Path | str, content: str) -> ToolResult:
    """Write content to a file, creating parent directories as needed."""
    p = Path(file_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except OSError as exc:
        return _err(f"Cannot write file: {exc}")
    return _ok(f"Written {len(content)} characters to {p}")


# ── Destructive-command guard ────────────────

_DESTRUCTIVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\brm\s+-[^ ]*r", re.IGNORECASE),             # rm -r / rm -rf
    re.compile(r"\brmdir\b", re.IGNORECASE),
    re.compile(r"\bdel\s+/[sf]", re.IGNORECASE),              # del /s /f (Windows)
    re.compile(r"\bformat\b", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bdd\b.*\bof=", re.IGNORECASE),
    re.compile(r"\bdrop\s+(table|database|schema)\b", re.IGNORECASE),
    re.compile(r"\btruncate\s+table\b", re.IGNORECASE),
    re.compile(r">\s*/dev/sd[a-z]", re.IGNORECASE),           # redirect to block device
    re.compile(r"\bshred\b", re.IGNORECASE),
]


def _is_destructive(command: str, args: list[str]) -> bool:
    full = " ".join([command] + args)
    return any(pat.search(full) for pat in _DESTRUCTIVE_PATTERNS)


def execute_command(
    command: str,
    args: list[str] | None = None,
    working_dir: str | None = None,
    allow_destructive: bool = False,
) -> ToolResult:
    """Run a shell command and return stdout, stderr, and exit code."""
    args = args or []
    if not allow_destructive and _is_destructive(command, args):
        return _err(
            "Command matches a destructive pattern. "
            "Set allow_destructive=true to proceed."
        )

    try:
        proc = subprocess.run(
            [command] + args,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=working_dir,
        )
    except FileNotFoundError:
        return _err(f"Command not found: {command}")
    except subprocess.TimeoutExpired:
        return _err(f"Command timed out after 30 seconds: {command}")
    except Exception as exc:  # noqa: BLE001
        return _err(f"Command failed: {exc}")

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    parts: list[str] = [f"exit_code: {proc.returncode}"]
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    output = "\n".join(parts)

    if proc.returncode != 0:
        return _err(output)
    return _ok(output)


def glob(pattern: str, base_dir: str | Path = ".") -> ToolResult:
    """Return a list of file paths matching a glob pattern within base_dir."""
    root = Path(base_dir)
    if not root.exists():
        return _err(f"Base directory not found: {base_dir}")

    matches = sorted(root.glob(pattern))
    if not matches:
        return _ok(f"No files found matching: {pattern}")

    cap = 200
    lines = [str(m.relative_to(root)) for m in matches[:cap]]
    result = f"Found {len(matches)} file(s):\n" + "\n".join(lines)
    if len(matches) > cap:
        result += f"\n... and {len(matches) - cap} more"
    return _ok(result)


def grep(pattern: str, path: str | Path, is_regex: bool = False) -> ToolResult:
    """Search for a pattern across files, returning matching paths and lines."""
    abs_path = Path(path)
    if not abs_path.exists():
        return _err(f"Path not found: {path}")

    try:
        regex = re.compile(pattern if is_regex else re.escape(pattern), re.IGNORECASE)
    except re.error as exc:
        return _err(f"Invalid regex pattern: {exc}")

    if abs_path.is_file():
        files = [abs_path]
    else:
        files = sorted(f for f in abs_path.glob("**/*") if f.is_file())[:100]

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
        return _ok(f"No matches found for: {pattern}")

    result = "\n".join(hits)
    if len(hits) == 100:
        result += "\n... (output capped at 100 matches)"
    return _ok(result)
