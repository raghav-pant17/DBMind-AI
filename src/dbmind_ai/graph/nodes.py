"""Workflow node container."""

from dataclasses import dataclass

from dbmind_ai.agents.aggregation_validator import AggregationValidator
from dbmind_ai.agents.business_validator import BusinessRuleValidator
from dbmind_ai.agents.correction_agent import CorrectionAgent
from dbmind_ai.agents.insight_agent import InsightAgent
from dbmind_ai.agents.intent_agent import IntentAgent
from dbmind_ai.agents.join_validator import JoinValidator
from dbmind_ai.agents.performance_agent import PerformanceAgent
from dbmind_ai.agents.result_validator import ResultValidator
from dbmind_ai.agents.sql_generator import SQLGenerator
from dbmind_ai.database.executor import ReadOnlyExecutor
from dbmind_ai.guardrails.sqlglot_validator import SQLGlotValidator
from dbmind_ai.rag.chroma_retriever import ChromaRetriever
from dbmind_ai.rag.schema_retriever import SchemaRetriever


# Groups concrete node implementations used by the workflow.
@dataclass
class WorkflowNodes:
    intent_agent: IntentAgent
    schema_retriever: SchemaRetriever
    rules_retriever: ChromaRetriever
    sql_generator: SQLGenerator
    sql_guardrail: SQLGlotValidator
    join_validator: JoinValidator
    aggregation_validator: AggregationValidator
    business_validator: BusinessRuleValidator
    performance_agent: PerformanceAgent
    correction_agent: CorrectionAgent
    executor: ReadOnlyExecutor
    result_validator: ResultValidator
    insight_agent: InsightAgent
