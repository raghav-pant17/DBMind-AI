"""Tests for LLM JSON parsing helpers."""

from dbmind_ai.agents.json_parser import parse_json_object


# Verifies strict JSON is parsed directly.
def test_parse_json_object_accepts_strict_json() -> None:
    assert parse_json_object('{"status": "pass", "score": 100}') == {
        "status": "pass",
        "score": 100,
    }


# Verifies fenced JSON is parsed after removing markdown fences.
def test_parse_json_object_accepts_fenced_json() -> None:
    assert parse_json_object('```json\n{"metric": "revenue"}\n```') == {"metric": "revenue"}


# Verifies Python-style dict output is tolerated.
def test_parse_json_object_accepts_python_dict_text() -> None:
    assert parse_json_object("{'status': 'warn', 'messages': ['check joins']}") == {
        "status": "warn",
        "messages": ["check joins"],
    }


# Verifies extra prose around the object is ignored.
def test_parse_json_object_extracts_inline_object() -> None:
    assert parse_json_object('Here is the JSON:\n{"operation": "sum"}\nThanks') == {
        "operation": "sum"
    }


# Verifies escaped newline text from model wrappers is parsed.
def test_parse_json_object_accepts_escaped_newlines() -> None:
    assert parse_json_object('{\\n  "metric": "revenue"\\n}') == {"metric": "revenue"}
