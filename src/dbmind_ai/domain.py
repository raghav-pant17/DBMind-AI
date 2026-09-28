"""Shared domain models for DBMind-AI."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


# Enumerates possible validation outcomes.
class ValidationStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


# Captures the structured meaning extracted from a user question.
class Intent(BaseModel):
    metric: str | None = None
    dimensions: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    comparison: str | None = None
    operation: str | None = None
    period: str | None = None


# Describes a database column exposed to DBMind-AI.
class ColumnInfo(BaseModel):
    name: str
    type: str = "unknown"
    description: str = ""


# Describes a known relationship between two database columns.
class RelationshipInfo(BaseModel):
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    description: str = ""


# Describes a database table and its available columns.
class TableInfo(BaseModel):
    name: str
    description: str = ""
    primary_key: list[str] = Field(default_factory=list)
    columns: list[ColumnInfo] = Field(default_factory=list)


# Groups retrieved tables and relationships for SQL generation and validation.
class SchemaContext(BaseModel):
    tables: list[TableInfo] = Field(default_factory=list)
    relationships: list[RelationshipInfo] = Field(default_factory=list)

    @property
    def table_names(self) -> set[str]:
        return {table.name for table in self.tables}

    # Returns known column names for a table.
    def columns_for(self, table_name: str) -> set[str]:
        for table in self.tables:
            if table.name == table_name:
                return {column.name for column in table.columns}
        return set()


# Represents a business definition or policy used during validation.
class BusinessRule(BaseModel):
    id: str
    name: str
    description: str
    applies_to: list[str] = Field(default_factory=list)


# Carries the outcome of one validation check.
class ValidationResult(BaseModel):
    name: str
    status: ValidationStatus
    score: int = Field(ge=0, le=100)
    messages: list[str] = Field(default_factory=list)


# Stores tabular data returned from a trusted query.
class QueryResult(BaseModel):
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0


# Represents the complete answer returned by DBMind-AI.
class DBMindResponse(BaseModel):
    question: str
    sql: str
    answer: str
    trust_score: int
    validations: list[ValidationResult]
    result: QueryResult | None = None
    corrections: int = 0
