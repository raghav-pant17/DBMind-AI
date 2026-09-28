"""LLM aggregation validation agent."""

from typing import Any

from langchain_core.language_models import BaseChatModel

from dbmind_ai.agents.json_parser import parse_json_object
from dbmind_ai.agents.llm_content import content_to_text
from dbmind_ai.domain import (
    BusinessRule,
    Intent,
    SchemaContext,
    ValidationResult,
    ValidationStatus,
)


# Validates aggregate SQL semantics using the configured LLM.
class AggregationValidator:
    """Validate aggregation correctness with the configured LLM."""

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    # Asks the model whether the SQL aggregation matches the user request.
    def run(
        self,
        question: str,
        sql: str,
        intent: Intent,
        schema: SchemaContext,
        rules: list[BusinessRule],
    ) -> ValidationResult:
        try:
            response = self.llm.invoke(self._build_prompt(question, sql, intent, schema, rules))
            content = content_to_text(response.content)
            payload = self._extract_json(content)
            return ValidationResult(
                name="aggregation_validator",
                status=ValidationStatus(payload["status"]),
                score=int(payload["score"]),
                messages=list(payload.get("messages", [])),
            )
        except Exception as exc:
            return ValidationResult(
                name="aggregation_validator",
                status=ValidationStatus.FAIL,
                score=0,
                messages=[f"LLM aggregation validation failed: {exc}"],
            )

    # Builds the semantic aggregation validation prompt for the model.
    def _build_prompt(
        self,
        question: str,
        sql: str,
        intent: Intent,
        schema: SchemaContext,
        rules: list[BusinessRule],
    ) -> str:
        return f"""
You are DBMind-AI's aggregation validation agent.

Evaluate whether the generated SQL uses the correct aggregation logic for the
user's request. Judge semantic correctness, not keyword presence.

Check whether the SQL correctly handles grouping, counting, sums, averages,
distinct counts, time periods, comparisons, percentages, window functions, and
ranking based on what the user asked.

Return only valid JSON with this exact shape:
{{
  "status": "pass | warn | fail",
  "score": 0,
  "messages": ["short explanation"]
}}

Scoring:
- 90-100: aggregation logic clearly matches the request.
- 75-89: aggregation logic is mostly correct, with minor ambiguity.
- 0-74: aggregation logic is missing, incorrect, or misleading.

User question:
{question}

Structured intent:
{intent.model_dump_json(indent=2)}

Generated SQL:
{sql}

Schema:
{self._format_schema(schema)}

Relevant business rules:
{self._format_rules(rules)}
""".strip()

    # Formats schema context for the validation prompt.
    @staticmethod
    def _format_schema(schema: SchemaContext) -> str:
        lines: list[str] = []
        for table in schema.tables:
            columns = ", ".join(column.name for column in table.columns)
            lines.append(f"- {table.name}: {columns}")
        return "\n".join(lines)

    # Formats business rules for aggregation context.
    @staticmethod
    def _format_rules(rules: list[BusinessRule]) -> str:
        if not rules:
            return "No business rules were retrieved."
        return "\n\n".join(f"{rule.name}: {rule.description}" for rule in rules)

    # Extracts and parses JSON from model output.
    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        return parse_json_object(content)
