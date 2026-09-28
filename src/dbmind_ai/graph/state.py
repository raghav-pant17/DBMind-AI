"""Workflow state."""

from typing import TypedDict

from dbmind_ai.domain import (
    BusinessRule,
    DBMindResponse,
    Intent,
    QueryResult,
    SchemaContext,
    ValidationResult,
)


# Defines the shared state shape used by graph orchestration.
class DBMindState(TypedDict, total=False):
    user_question: str
    intent: Intent
    retrieved_schema: SchemaContext
    full_schema: SchemaContext
    business_rules: list[BusinessRule]
    generated_sql: str
    sql_guardrail_result: ValidationResult
    join_validation: ValidationResult
    aggregation_validation: ValidationResult
    business_validation: ValidationResult
    performance_validation: ValidationResult
    validations: list[ValidationResult]
    failures: list[ValidationResult]
    trust_score: int
    correction_count: int
    query_result: QueryResult
    result_validation: ValidationResult
    final_answer: str
    final_response: DBMindResponse
