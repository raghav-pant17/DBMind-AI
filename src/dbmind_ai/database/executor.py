"""Read-only SQL execution."""

import re

from sqlalchemy import Engine, text

from dbmind_ai.domain import QueryResult


# Executes trusted SQL through a read-only interface.
class ReadOnlyExecutor:
    """Execute trusted SQL through a read-only interface."""

    # Stores the engine and fallback row limit.
    def __init__(self, engine: Engine, default_limit: int = 100) -> None:
        self.engine = engine
        self.default_limit = default_limit

    # Executes trusted SQL and serializes the returned rows.
    def execute(self, sql: str) -> QueryResult:
        safe_sql = self._ensure_limit(sql)
        with self.engine.connect() as connection:
            rows = connection.execute(text(safe_sql)).mappings().all()
        serialized = [dict(row) for row in rows]
        columns = list(serialized[0].keys()) if serialized else []
        return QueryResult(columns=columns, rows=serialized, row_count=len(serialized))

    # Adds a fallback LIMIT when SQL does not already include one.
    def _ensure_limit(self, sql: str) -> str:
        if re.search(r"\blimit\b", sql, flags=re.IGNORECASE):
            return sql
        return f"{sql.rstrip().rstrip(';')}\nLIMIT {self.default_limit};"
