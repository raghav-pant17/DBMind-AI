"""Join validation agent."""

from sqlglot import exp, parse_one

from dbmind_ai.domain import SchemaContext, ValidationResult, ValidationStatus
from dbmind_ai.guardrails.schema_validator import extract_table_aliases


# Validates SQL join conditions against known schema relationships.
class JoinValidator:
    """Validate joins against known schema relationships."""

    # Stores the SQL dialect used by SQLGlot.
    def __init__(self, dialect: str = "sqlite") -> None:
        self.dialect = dialect

    # Checks whether each join condition follows a known relationship.
    def run(self, sql: str, schema: SchemaContext) -> ValidationResult:
        tree = parse_one(sql, read=self.dialect)
        aliases = extract_table_aliases(tree)
        joins = list(tree.find_all(exp.Join))
        if not joins:
            return ValidationResult(
                name="join_validator",
                status=ValidationStatus.PASS,
                score=100,
                messages=["No joins required for this query."],
            )

        relationship_pairs = {
            frozenset(
                {
                    f"{rel.from_table}.{rel.from_column}",
                    f"{rel.to_table}.{rel.to_column}",
                }
            )
            for rel in schema.relationships
        }
        messages: list[str] = []
        for join in joins:
            for eq in join.find_all(exp.EQ):
                left = self._qualified_column(eq.left, aliases)
                right = self._qualified_column(eq.right, aliases)
                pair = frozenset({left, right})
                if pair not in relationship_pairs:
                    messages.append(
                        f"Join condition is not a known relationship: {left} = {right}."
                    )

        return ValidationResult(
            name="join_validator",
            status=ValidationStatus.FAIL if messages else ValidationStatus.PASS,
            score=70 if messages else 100,
            messages=messages or ["Join paths match known schema relationships."],
        )

    # Converts a SQLGlot column expression into a fully qualified table.column name.
    def _qualified_column(self, expression: exp.Expression, aliases: dict[str, str]) -> str:
        if isinstance(expression, exp.Column) and expression.table:
            table = aliases.get(expression.table, expression.table)
            return f"{table}.{expression.name}"
        return expression.sql(dialect=self.dialect)
