#!/usr/bin/env python3
"""Shared helpers for converting benchmark data into src_requests rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_REQUEST_SUFFIX = (
    "<user> Based on our conversation above, please only make one tool call to solve my need.</user>"
)


def build_system_prompt(schemas: list[dict[str, Any]]) -> str:
    """Create the standard single-tool-call system prompt."""
    tools_text = []
    for idx, schema in enumerate(schemas, start=1):
        params_repr = json.dumps(schema.get("parameters", {}), ensure_ascii=False)
        tools_text.append(
            f"{idx}. Name: {schema.get('name', '')}\n"
            f"Description: {schema.get('description', '')}\n"
            f"Parameters: {params_repr}"
        )
    tools_block = "\n".join(tools_text)
    return (
        "You are a helpful assistant that selects the best available tool and calls it.\n\n"
        "**Available Tools**\n"
        f"{tools_block}\n\n"
        "**Output Format**\n"
        "Output ONLY a single JSON object with exactly two keys:\n"
        '  "name": the name of the tool to call\n'
        '  "parameters": a JSON object of parameter names to values\n'
        "Do not wrap the JSON in markdown, code fences, or any explanation."
    )


def build_user_prompt(user_content: str) -> str:
    """Wrap a task in the src_requests dialogue-history format."""
    return (
        "**Dialogue Records History**\n"
        f"<user>{user_content}</user>\n\n"
        f"{SRC_REQUEST_SUFFIX}"
    )


def normalize_tool_schema(
    tool: dict[str, Any],
    *,
    name_key: str = "name",
    description_key: str = "description",
    parameters_key: str = "parameters",
    required_key: str = "required",
) -> dict[str, Any]:
    """Normalize common JSON-schema-like tool definitions.

    The supported inputs cover OpenAI-style function specs, BFCL specs, and
    dataset-specific records with custom name/description keys.
    """
    fn = tool.get("function", tool)
    params = fn.get(parameters_key, {})
    if not isinstance(params, dict):
        params = {}

    if "properties" in params:
        properties = params.get("properties", {})
        required = set(params.get(required_key, []))
    else:
        properties = params
        required = set(fn.get(required_key, tool.get(required_key, [])))

    normalized: dict[str, dict[str, Any]] = {}
    for param_name, param_def in properties.items():
        if isinstance(param_def, dict):
            normalized[param_name] = {
                "type": param_def.get("type", "any"),
                "description": param_def.get("description", ""),
                "required": param_name in required,
            }
        else:
            normalized[param_name] = {
                "type": "any",
                "description": str(param_def),
                "required": param_name in required,
            }

    return {
        "name": fn.get(name_key, ""),
        "description": fn.get(description_key, ""),
        "parameters": normalized,
    }


def build_src_request(
    *,
    request_id: str,
    source: str,
    user_content: str,
    schemas: list[dict[str, Any]],
    answer: Any = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete src_requests row."""
    system = build_system_prompt(schemas)
    user = build_user_prompt(user_content)
    row = {
        "request_id": request_id,
        "source": source,
        "system": system,
        "user": user,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "schemas": schemas,
        "answer": answer,
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def latest_user_content(question: Any) -> str:
    """Extract the final user message from BFCL's nested question format."""
    if not isinstance(question, list) or not question:
        return ""
    turn = question[-1]
    if not isinstance(turn, list) or not turn:
        return ""
    message = turn[-1]
    if not isinstance(message, dict):
        return ""
    return message.get("content", "")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def path_arg(value: str) -> Path:
    return Path(value).expanduser()


def add_output_arg(parser: argparse.ArgumentParser, default: Path) -> None:
    parser.add_argument("--output-path", type=path_arg, default=default)
