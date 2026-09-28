"""SQL correction agent."""

from dbmind_ai.agents.sql_generator import SQLGenerator
from dbmind_ai.domain import BusinessRule, Intent, SchemaContext, ValidationResult


# Produces a corrected SQL candidate from validation feedback.
class CorrectionAgent:
    """Regenerate SQL using validation feedback."""

    # Stores the generator used for correction attempts.
    def __init__(self, generator: SQLGenerator) -> None:
        self.generator = generator

    # Regenerates SQL using the original context plus failure messages.
    def run(
        self,
        question: str,
        intent: Intent,
        schema: SchemaContext,
        rules: list[BusinessRule],
        failures: list[ValidationResult],
    ) -> str:
        feedback = " ".join(" ".join(result.messages) for result in failures).lower()
        corrected_question = question
        if "cancelled" in feedback and "revenue" in (intent.metric or ""):
            corrected_question = f"{question} excluding cancelled and refunded orders"
        return self.generator.run(corrected_question, intent, schema, rules)
