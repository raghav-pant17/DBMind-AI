"""Deterministic SQL guardrail based on SQLGlot."""

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from dbmind_ai.domain import SchemaContext, ValidationResult, ValidationStatus
from dbmind_ai.guardrails.safety_rules import FORBIDDEN_EXPRESSIONS, FORBIDDEN_KEYWORDS
from dbmind_ai.guardrails.schema_validator import unknown_columns, unknown_tables


# Performs deterministic SQL safety and schema validation.
class SQLGlotValidator:
    """Validate generated SQL before any database execution."""

    # Stores SQL dialect and fallback row limit settings.
    def __init__(self, dialect: str = "sqlite", default_limit: int = 100) -> None:
        self.dialect = dialect
        self.default_limit = default_limit

    # Validates SQL syntax, read-only safety, schema references, and limits.
    def validate(self, sql: str, schema: SchemaContext) -> ValidationResult:
        try:
            tree = parse_one(sql, read=self.dialect)
        except ParseError as exc:
            return ValidationResult(
                name="deterministic_sql_guardrail",
                status=ValidationStatus.FAIL,
                score=0,
                messages=[f"SQL parse failed: {exc}"],
            )

        messages: list[str] = []
        lowered = sql.lower()
        if any(keyword in lowered.split() for keyword in FORBIDDEN_KEYWORDS):
            messages.append("Unsafe SQL keyword detected.")

        if not isinstance(tree, exp.Select):
            messages.append("Only SELECT statements are allowed.")

        forbidden_nodes = [
            node.__class__.__name__
            for node in tree.walk()
            if node.__class__.__name__ in FORBIDDEN_EXPRESSIONS
        ]
        if forbidden_nodes:
            messages.append(
                f"Forbidden SQL operation detected: {', '.join(sorted(set(forbidden_nodes)))}."
            )

        tables = unknown_tables(tree, schema)
        if tables:
            messages.append(f"Unknown tables: {', '.join(tables)}.")

        columns = unknown_columns(tree, schema)
        if columns:
            messages.append(f"Unknown columns: {', '.join(columns)}.")

        if not tree.args.get("limit"):
            messages.append(
                f"Query has no LIMIT; a default limit of {self.default_limit} will be applied."
            )

        fatal = [message for message in messages if not message.startswith("Query has no LIMIT")]
        return ValidationResult(
            name="deterministic_sql_guardrail",
            status=ValidationStatus.FAIL if fatal else ValidationStatus.PASS,
            score=0 if fatal else 95 if messages else 100,
            messages=messages or ["SQL is syntactically valid, read-only, and schema-safe."],
        )
