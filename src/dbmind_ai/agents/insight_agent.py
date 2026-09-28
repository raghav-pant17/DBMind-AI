"""LLM business insight generation agent."""

import json

from langchain_core.language_models import BaseChatModel

from dbmind_ai.agents.llm_content import content_to_text
from dbmind_ai.domain import BusinessRule, QueryResult, ValidationResult


# Converts query results into a business-readable answer using the configured LLM.
class InsightAgent:
    """Generate final business insights with the configured LLM."""

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    # Builds the final natural-language response from rows, rules, and score.
    def run(
        self,
        question: str,
        sql: str,
        result: QueryResult,
        trust_score: int,
        validations: list[ValidationResult],
        business_rules: list[BusinessRule],
    ) -> str:
        try:
            response = self.llm.invoke(
                self._build_prompt(
                    question=question,
                    sql=sql,
                    result=result,
                    trust_score=trust_score,
                    validations=validations,
                    business_rules=business_rules,
                )
            )
            return content_to_text(response.content).strip()
        except Exception as exc:
            return f"The SQL executed successfully, but LLM insight generation failed: {exc}"

    # Builds the prompt that asks the model for a concise business answer.
    @staticmethod
    def _build_prompt(
        question: str,
        sql: str,
        result: QueryResult,
        trust_score: int,
        validations: list[ValidationResult],
        business_rules: list[BusinessRule],
    ) -> str:
        return f"""
You are DBMind-AI's insight agent.

Write a concise business-friendly answer based only on the SQL result.

Rules:
- Answer the user's question directly.
- Mention important business-rule assumptions when relevant.
- Mention the trust score.
- Do not invent facts that are not in the result.
- Keep the answer short.

User question:
{question}

Trusted SQL:
{sql}

Query result:
{json.dumps(result.model_dump(), indent=2, default=str)}

Validation summary:
{json.dumps([validation.model_dump() for validation in validations], indent=2, default=str)}

Business rules used:
{json.dumps([rule.model_dump() for rule in business_rules], indent=2, default=str)}

Trust score:
{trust_score}/100
""".strip()
