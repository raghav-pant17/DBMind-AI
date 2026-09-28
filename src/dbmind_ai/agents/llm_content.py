"""Helpers for extracting text from LLM response content."""

from typing import Any


# Converts string or structured content blocks into plain text.
def content_to_text(content: Any) -> str:
    """Return readable text from a LangChain chat response content payload."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(text for block in content if (text := _block_to_text(block)))
    return str(content)


# Extracts text from one structured content block.
def _block_to_text(block: Any) -> str:
    """Return the text field from one model content block."""

    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        value = block.get("text") or block.get("content")
        return value if isinstance(value, str) else ""
    value = getattr(block, "text", None) or getattr(block, "content", None)
    return value if isinstance(value, str) else ""
