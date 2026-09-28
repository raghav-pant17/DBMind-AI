"""Tests for LLM content extraction helpers."""

from dbmind_ai.agents.llm_content import content_to_text


# Verifies string content passes through unchanged.
def test_content_to_text_accepts_string() -> None:
    assert content_to_text("select 1") == "select 1"


# Verifies provider-specific structured content blocks return only text.
def test_content_to_text_extracts_structured_text_blocks() -> None:
    content = [
        {
            "type": "text",
            "text": "SELECT * FROM film;",
            "extras": {"signature": "ignore-me"},
        }
    ]

    assert content_to_text(content) == "SELECT * FROM film;"
