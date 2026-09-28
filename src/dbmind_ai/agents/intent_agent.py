"""LLM question-to-intent extraction agent."""

from typing import Any

from langchain_core.language_models import BaseChatModel

from dbmind_ai.agents.json_parser import parse_json_object
from dbmind_ai.agents.llm_content import content_to_text
from dbmind_ai.domain import Intent


# Extracts structured intent from a business question using the configured LLM.
class IntentAgent:
    """Extract a lightweight structured intent from a business question."""

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    # Converts a raw question into an Intent model.
    def run(self, question: str) -> Intent:
        response = self.llm.invoke(self._build_prompt(question))
        content = content_to_text(response.content)
        payload = self._extract_json(content)
        return Intent.model_validate(payload)

    # Builds the prompt that asks the model for strict intent JSON.
    @staticmethod
    def _build_prompt(question: str) -> str:
        return f"""
You are DBMind-AI's intent extraction agent.

Extract structured intent from the user's business question.

Return only valid JSON with this exact shape:
{{
  "metric": "string or null",
  "dimensions": ["string"],
  "filters": {{}},
  "comparison": "string or null",
  "operation": "string or null",
  "period": "string or null"
}}

Guidelines:
- metric should be the main business measure, such as revenue, orders, customers, refunds.
- dimensions should be grouping fields requested by the user, such as region,
  customer, product, month, year.
- filters should contain explicit constraints from the question.
- comparison should describe comparison intent, such as year_over_year.
- operation should describe ranking or calculation intent, such as highest,
  lowest, count, sum, average.
- period should capture an explicit year/date period when present.
- Use null when a field is unknown.

User question:
{question}
""".strip()

    # Extracts and parses JSON from model output.
    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        return parse_json_object(content)
