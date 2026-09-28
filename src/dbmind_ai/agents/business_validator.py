"""LLM business-rule validation agent."""

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


# Checks generated SQL against retrieved business rules using the configured LLM.
class BusinessRuleValidator:
    """Validate semantic business-rule compliance with the configured LLM."""

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    # Asks the model whether SQL satisfies retrieved business rules.
    def run(
        self,
        question: str,
        sql: str,
        intent: Intent,
        rules: list[BusinessRule],
        schema: SchemaContext,
    ) -> ValidationResult:
        try:
            response = self.llm.invoke(self._build_prompt(question, sql, intent, rules, schema))
            content = content_to_text(response.content)
            payload = self._extract_json(content)
            return ValidationResult(
                name="business_rule_validator",
                status=ValidationStatus(payload["status"]),
                score=int(payload["score"]),
                messages=list(payload.get("messages", [])),
            )
        except Exception as exc:
            return ValidationResult(
                name="business_rule_validator",
                status=ValidationStatus.FAIL,
                score=0,
                messages=[f"LLM business-rule validation failed: {exc}"],
            )

    # Builds the semantic validation prompt for the model.
    def _build_prompt(
        self,
        question: str,
        sql: str,
        intent: Intent,
        rules: list[BusinessRule],
        schema: SchemaContext,
    ) -> str:
        return f"""
You are DBMind-AI's business-rule validation agent.

Evaluate whether the generated SQL satisfies the retrieved business rules for
the user's request. Judge semantic correctness, not just keyword presence.

Return only valid JSON with this exact shape:
{{
  "status": "pass | warn | fail",
  "score": 0,
  "messages": ["short explanation"]
}}

Scoring:
- 90-100: SQL clearly satisfies the relevant business rules.
- 75-89: SQL mostly satisfies rules, with minor ambiguity.
- 0-74: SQL misses or violates important business rules.

User question:
{question}

Structured intent:
{intent.model_dump_json(indent=2)}

Generated SQL:
{sql}

Schema:
{self._format_schema(schema)}

Retrieved business rules:
{self._format_rules(rules)}
""".strip()

    # Formats schema context for the validation prompt.
    @staticmethod
    def _format_schema(schema: SchemaContext) -> str:
        lines: list[str] = []
        for table in schema.tables:
            columns = ", ".join(column.name for column in table.columns)
            lines.append(f"- {table.name}: {columns}")
        if schema.relationships:
            lines.append("Relationships:")
            for relationship in schema.relationships:
                lines.append(
                    "- "
                    f"{relationship.from_table}.{relationship.from_column} -> "
                    f"{relationship.to_table}.{relationship.to_column}"
                )
        return "\n".join(lines)

    # Formats retrieved rules for the validation prompt.
    @staticmethod
    def _format_rules(rules: list[BusinessRule]) -> str:
        if not rules:
            return "No business rules were retrieved."
        return "\n\n".join(f"{rule.name}: {rule.description}" for rule in rules)

    # Extracts and parses JSON from model output.
    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        return parse_json_object(content)
