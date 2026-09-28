"""Helpers for parsing JSON-like responses from LLMs."""

import ast
import json
import re
from typing import Any

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


# Parses a JSON object even when the model wraps it or uses Python dict syntax.
def parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response."""

    raw_json = _extract_object_text(content)
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        payload = _parse_json_fallback(raw_json)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object from the model response.")
    return payload


# Parses relaxed JSON variants returned by some chat model wrappers.
def _parse_json_fallback(raw_json: str) -> Any:
    """Parse JSON after unescaping literal newlines or Python dict syntax."""

    try:
        return json.loads(raw_json.replace("\\n", "\n").replace("\\t", "\t"))
    except json.JSONDecodeError:
        return ast.literal_eval(raw_json)


# Extracts the most likely object body from raw model text.
def _extract_object_text(content: str) -> str:
    """Extract fenced or inline object text from a model response."""

    match = JSON_FENCE_RE.search(content)
    raw_json = match.group(1) if match else content
    raw_json = raw_json.strip()
    start = raw_json.find("{")
    end = raw_json.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw_json[start : end + 1]
    return raw_json
