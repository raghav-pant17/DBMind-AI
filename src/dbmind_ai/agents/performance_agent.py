"""Hybrid SQL performance validation agent."""

from sqlalchemy import Engine, text
from sqlglot import exp, parse_one

from dbmind_ai.domain import ValidationResult, ValidationStatus


# Reviews SQL performance with static checks and database EXPLAIN output.
class PerformanceAgent:
    """Validate query performance using deterministic database signals."""

    # Stores database and SQL dialect settings for performance review.
    def __init__(self, engine: Engine, dialect: str = "sqlite") -> None:
        self.engine = engine
        self.dialect = dialect

    # Reviews SQL using static warnings and database EXPLAIN details.
    def run(self, sql: str) -> ValidationResult:
        findings = self._static_findings(sql)
        explain_findings = self._explain_findings(sql)
        messages = [*findings, *explain_findings]
        risk_count = sum(1 for message in messages if not message.startswith("OK:"))
        score = max(70, 100 - (10 * risk_count))
        status = ValidationStatus.PASS if risk_count == 0 else ValidationStatus.WARN

        return ValidationResult(
            name="performance_agent",
            status=status,
            score=score,
            messages=messages or ["OK: No performance risks detected."],
        )

    # Finds objective performance risks from the SQL AST.
    def _static_findings(self, sql: str) -> list[str]:
        tree = parse_one(sql, read=self.dialect)
        findings: list[str] = []
        join_count = len(list(tree.find_all(exp.Join)))

        if not tree.args.get("where"):
            findings.append("Query has no WHERE filter.")
        if not tree.args.get("limit"):
            findings.append("Query has no LIMIT.")
        if any(isinstance(node, exp.Star) for node in tree.walk()):
            findings.append("Query uses SELECT *.")
        if join_count > 3:
            findings.append(f"Query joins {join_count} tables.")

        return findings

    # Interprets database EXPLAIN output with simple deterministic rules.
    def _explain_findings(self, sql: str) -> list[str]:
        plan = self._explain(sql)
        findings: list[str] = []
        plan_text = " ".join(plan).upper()

        if "SCAN" in plan_text:
            findings.append("EXPLAIN indicates a table scan.")
        if "TEMP" in plan_text:
            findings.append("EXPLAIN indicates temporary sorting or grouping work.")
        if not findings:
            findings.append("OK: EXPLAIN did not report obvious scan/sort risks.")

        return findings

    # Runs a database EXPLAIN statement for a SQL query.
    def _explain(self, sql: str) -> list[str]:
        explain_sql = self._explain_sql(sql)
        with self.engine.connect() as connection:
            rows = connection.execute(text(explain_sql)).all()
        return [str(tuple(row)) for row in rows]

    # Builds the dialect-specific EXPLAIN SQL statement.
    def _explain_sql(self, sql: str) -> str:
        cleaned_sql = sql.rstrip().rstrip(";")
        if self.dialect == "sqlite":
            return f"EXPLAIN QUERY PLAN {cleaned_sql}"
        return f"EXPLAIN {cleaned_sql}"
