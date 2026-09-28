"""Result validation agent."""

from dbmind_ai.domain import QueryResult, ValidationResult, ValidationStatus


# Validates returned query rows for quality issues.
class ResultValidator:
    """Validate result shape and suspicious values."""

    # Checks result rows for emptiness and suspicious numeric values.
    def run(self, result: QueryResult) -> ValidationResult:
        messages: list[str] = []
        if result.row_count == 0:
            messages.append("Query returned no rows.")
        for row in result.rows:
            for key, value in row.items():
                if isinstance(value, (int, float)) and value < 0 and "refund" not in key.lower():
                    messages.append(f"Unexpected negative value in column {key}.")

        return ValidationResult(
            name="result_validator",
            status=ValidationStatus.WARN if messages else ValidationStatus.PASS,
            score=85 if messages else 100,
            messages=messages or ["Result set passed quality checks."],
        )
