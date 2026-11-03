"""Parse model output into a tool call and execute it in the sandbox."""

from __future__ import annotations

import json
from dataclasses import dataclass

from afo.environments.base import SandboxWorld, ToolResult


@dataclass
class ParsedAction:
    tool: str | None
    args: dict
    thought: str
    error: str | None = None


def _first_json_object(text: str) -> str | None:
    """Return the first balanced {...} substring, respecting JSON strings."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def parse_action(text: str) -> ParsedAction:
    blob = _first_json_object(text or "")
    if blob is None:
        return ParsedAction(None, {}, "", error="no JSON object found")
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError as e:
        return ParsedAction(None, {}, "", error=f"invalid JSON: {e.msg}")
    tool = obj.get("tool") or obj.get("name") or obj.get("action")
    args = obj.get("args") or obj.get("arguments") or obj.get("parameters") or {}
    if not isinstance(tool, str):
        return ParsedAction(None, {}, str(obj.get("thought", "")), error="missing 'tool' field")
    if not isinstance(args, dict):
        return ParsedAction(tool, {}, str(obj.get("thought", "")), error="'args' must be an object")
    return ParsedAction(tool, args, str(obj.get("thought", "")))


class ToolExecutor:
    """Executes parsed actions against a sandbox world; never raises on bad agent input."""

    def __init__(self, world: SandboxWorld):
        self.world = world

    def execute(self, action: ParsedAction) -> ToolResult:
        if action.error:
            return ToolResult(
                f"Error: could not parse your action ({action.error}). Reply with one JSON object like "
                '{"thought": "...", "tool": "read_file", "args": {"path": "..."}}', ok=False)
        try:
            return self.world.execute(action.tool, action.args)
        except Exception as e:  # defensive: malformed args must not crash a run
            return ToolResult(f"Error: tool execution failed: {type(e).__name__}: {e}", ok=False)
