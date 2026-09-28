"""Application factory."""

from pathlib import Path

from dbmind_ai.agents.aggregation_validator import AggregationValidator
from dbmind_ai.agents.business_validator import BusinessRuleValidator
from dbmind_ai.agents.correction_agent import CorrectionAgent
from dbmind_ai.agents.insight_agent import InsightAgent
from dbmind_ai.agents.intent_agent import IntentAgent
from dbmind_ai.agents.join_validator import JoinValidator
from dbmind_ai.agents.performance_agent import PerformanceAgent
from dbmind_ai.agents.result_validator import ResultValidator
from dbmind_ai.agents.sql_generator import SQLGenerator
from dbmind_ai.config import Settings, get_settings
from dbmind_ai.database.connection import create_database_engine
from dbmind_ai.database.executor import ReadOnlyExecutor
from dbmind_ai.database.schema_introspector import SchemaIntrospector
from dbmind_ai.domain import SchemaContext
from dbmind_ai.graph.nodes import WorkflowNodes
from dbmind_ai.graph.workflow import DBMindWorkflow
from dbmind_ai.guardrails.sqlglot_validator import SQLGlotValidator
from dbmind_ai.llm_factory import build_chat_model
from dbmind_ai.rag.chroma_retriever import ChromaRetriever
from dbmind_ai.rag.schema_retriever import SchemaRetriever


# Resolves a configured path relative to the project root.
def resolve_project_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


# Resolves the configured PDF business-context directory.
def resolve_business_context_path(settings: Settings, root: Path) -> Path:
    return resolve_project_path(root, settings.business_context_path)


# Loads or introspects schema context for retriever construction.
def load_schema_context(settings: Settings, root: Path, engine) -> SchemaContext:
    schema_context_path = resolve_project_path(root, settings.schema_context_path)
    if schema_context_path.exists():
        return SchemaRetriever.from_file(schema_context_path).all()
    return SchemaIntrospector(
        engine,
        settings.database_schema,
    ).write_context(schema_context_path)


# Builds deterministic schema retrieval and Chroma PDF retrieval.
def build_retrievers(settings: Settings, root: Path, engine):
    schema_context = load_schema_context(settings, root, engine)
    chroma_path = resolve_project_path(root, settings.chroma_path)
    return (
        SchemaRetriever(schema_context),
        ChromaRetriever(
            chroma_path,
            f"{settings.chroma_collection_prefix}_documents",
        ),
    )


# Builds a fully wired DBMind-AI workflow from application settings.
def build_workflow(settings: Settings | None = None) -> DBMindWorkflow:
    settings = settings or get_settings()
    root = Path(__file__).resolve().parents[2]
    llm = build_chat_model(settings)
    generator = SQLGenerator(
        llm=llm,
        dialect=settings.sql_dialect,
    )
    engine = create_database_engine(settings.database_url, settings.database_schema)
    schema_retriever, rules_retriever = build_retrievers(settings, root, engine)
    nodes = WorkflowNodes(
        intent_agent=IntentAgent(
            llm=llm,
        ),
        schema_retriever=schema_retriever,
        rules_retriever=rules_retriever,
        sql_generator=generator,
        sql_guardrail=SQLGlotValidator(settings.sql_dialect, settings.default_limit),
        join_validator=JoinValidator(settings.sql_dialect),
        aggregation_validator=AggregationValidator(
            llm=llm,
        ),
        business_validator=BusinessRuleValidator(
            llm=llm,
        ),
        performance_agent=PerformanceAgent(engine=engine, dialect=settings.sql_dialect),
        correction_agent=CorrectionAgent(generator),
        executor=ReadOnlyExecutor(engine, settings.default_limit),
        result_validator=ResultValidator(),
        insight_agent=InsightAgent(
            llm=llm,
        ),
    )
    return DBMindWorkflow(nodes, settings.min_execution_score, settings.max_corrections)
