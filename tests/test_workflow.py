from pathlib import Path
from time import perf_counter, sleep

from langchain_core.documents import Document

from dbmind_ai.agents.correction_agent import CorrectionAgent
from dbmind_ai.agents.join_validator import JoinValidator
from dbmind_ai.agents.performance_agent import PerformanceAgent
from dbmind_ai.agents.result_validator import ResultValidator
from dbmind_ai.config import Settings
from dbmind_ai.database.connection import create_database_engine
from dbmind_ai.database.executor import ReadOnlyExecutor
from dbmind_ai.database.schema_introspector import SchemaIntrospector
from dbmind_ai.domain import (
    BusinessRule,
    Intent,
    SchemaContext,
    ValidationResult,
    ValidationStatus,
)
from dbmind_ai.graph.nodes import WorkflowNodes
from dbmind_ai.graph.workflow import DBMindWorkflow
from dbmind_ai.guardrails.sqlglot_validator import SQLGlotValidator
from dbmind_ai.rag.chroma_retriever import ChromaRetriever
from dbmind_ai.rag.document_loader import CHUNK_SIZE, split_pdf_documents
from dbmind_ai.rag.schema_retriever import SchemaRetriever


# Provides intent in tests without calling a live LLM API.
class FakeIntentAgent:
    # Returns structured intent for the revenue-by-region test question.
    def run(self, question: str) -> Intent:
        return Intent(
            metric="revenue",
            dimensions=["region"],
            operation="highest",
            period="2025",
        )


# Provides SQL in tests without calling a live LLM API.
class FakeSQLGenerator:
    # Returns trusted demo SQL for the revenue-by-region test question.
    def run(
        self,
        question: str,
        intent: Intent,
        schema: SchemaContext,
        business_rules: list[BusinessRule],
    ) -> str:
        return """
SELECT
    r.region_name,
    SUM(o.total_amount - COALESCE(o.discount_amount, 0) - COALESCE(o.refund_amount, 0)) AS revenue
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN regions r
    ON c.region_id = r.region_id
WHERE o.status NOT IN ('CANCELLED', 'REFUNDED')
AND strftime('%Y', o.order_date) = '2025'
GROUP BY r.region_name
ORDER BY revenue DESC
LIMIT 1;
""".strip()


# Provides destructive SQL in tests without calling a live LLM API.
class FakeDeleteSQLGenerator:
    # Returns a DELETE query so the workflow can verify non-SELECT rejection.
    def run(
        self,
        question: str,
        intent: Intent,
        schema: SchemaContext,
        business_rules: list[BusinessRule],
    ) -> str:
        return "DELETE FROM orders WHERE status = 'CANCELLED';"


# Provides business-rule validation in tests without calling an LLM.
class FakeBusinessRuleValidator:
    # Returns a passing semantic validation result for test SQL.
    def run(
        self,
        question: str,
        sql: str,
        intent: Intent,
        rules: list[BusinessRule],
        schema: SchemaContext,
    ) -> ValidationResult:
        return ValidationResult(
            name="business_rule_validator",
            status=ValidationStatus.PASS,
            score=100,
            messages=["Test business validation passed."],
        )


# Provides aggregation validation in tests without calling an LLM.
class FakeAggregationValidator:
    # Returns a passing semantic aggregation validation result for test SQL.
    def run(
        self,
        question: str,
        sql: str,
        intent: Intent,
        schema: SchemaContext,
        rules: list[BusinessRule],
    ) -> ValidationResult:
        return ValidationResult(
            name="aggregation_validator",
            status=ValidationStatus.PASS,
            score=100,
            messages=["Test aggregation validation passed."],
        )


# Provides insight generation in tests without calling an LLM.
class FakeInsightAgent:
    # Returns a deterministic final answer for workflow tests.
    def run(
        self,
        question: str,
        sql: str,
        result,
        trust_score: int,
        validations: list[ValidationResult],
        business_rules: list[BusinessRule],
    ) -> str:
        return f"APAC is the top region with revenue of 210,000.00. Trust Score: {trust_score}/100."


class EmptyRulesRetriever:
    def retrieve(self, question: str, top_k: int = 4) -> list[BusinessRule]:
        return []


# Provides a slow validator so tests can prove validator execution is parallel.
class SlowValidator:
    # Stores the validation name returned by this fake validator.
    def __init__(self, name: str) -> None:
        self.name = name

    # Sleeps briefly before returning a passing validation result.
    def run(self, *args, **kwargs) -> ValidationResult:
        sleep(0.2)
        return ValidationResult(
            name=self.name,
            status=ValidationStatus.PASS,
            score=100,
            messages=[f"{self.name} passed."],
        )


# Builds a workflow with a fake SQL generator for local tests.
def build_test_workflow(settings: Settings, generator=None) -> DBMindWorkflow:
    engine = create_database_engine(settings.database_url)
    generator = generator or FakeSQLGenerator()
    schema_context = SchemaIntrospector(engine).inspect()
    nodes = WorkflowNodes(
        intent_agent=FakeIntentAgent(),
        schema_retriever=SchemaRetriever(schema_context),
        rules_retriever=EmptyRulesRetriever(),
        sql_generator=generator,
        sql_guardrail=SQLGlotValidator(settings.sql_dialect, settings.default_limit),
        join_validator=JoinValidator(settings.sql_dialect),
        aggregation_validator=FakeAggregationValidator(),
        business_validator=FakeBusinessRuleValidator(),
        performance_agent=PerformanceAgent(engine, settings.sql_dialect),
        correction_agent=CorrectionAgent(generator),
        executor=ReadOnlyExecutor(engine, settings.default_limit),
        result_validator=ResultValidator(),
        insight_agent=FakeInsightAgent(),
    )
    return DBMindWorkflow(nodes, settings.min_execution_score, settings.max_corrections)


# Verifies the end-to-end revenue-by-region workflow executes successfully.
def test_revenue_by_region_workflow(demo_database_url: str) -> None:
    workflow = build_test_workflow(Settings(database_url=demo_database_url, sql_dialect="sqlite"))

    response = workflow.run("Which region generated the most revenue in 2025?")

    assert response.result is not None
    assert response.trust_score >= 90
    assert "APAC" in response.answer
    assert "CANCELLED" in response.sql


# Verifies generated destructive SQL is returned but never executed.
def test_destructive_sql_is_generated_but_not_executed(demo_database_url: str) -> None:
    workflow = build_test_workflow(
        Settings(database_url=demo_database_url, sql_dialect="sqlite"),
        FakeDeleteSQLGenerator(),
    )

    response = workflow.run("Delete all cancelled orders")

    assert response.result is None
    assert response.sql == "DELETE FROM orders WHERE status = 'CANCELLED';"
    assert "not executed" in response.answer


def test_pdf_document_chunking_preserves_metadata_and_overlap() -> None:
    text = " ".join(f"token-{index}" for index in range(500))
    chunks = split_pdf_documents(
        [Document(page_content=text, metadata={"source": "manual.pdf", "page": 2})]
    )

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= CHUNK_SIZE for chunk in chunks)
    assert all(chunk.metadata["source"] == "manual.pdf" for chunk in chunks)
    assert all(chunk.metadata["page"] == 2 for chunk in chunks)
    assert [chunk.metadata["chunk"] for chunk in chunks] == list(range(len(chunks)))
    assert set(chunks[0].page_content.split()) & set(chunks[1].page_content.split())
    assert all(chunk.id == chunk.metadata["chunk_id"] for chunk in chunks)


def test_chroma_rebuild_replaces_chunks(tmp_path: Path) -> None:
    store = ChromaRetriever(tmp_path / "chroma")
    documents = [
        Document(
            id="revenue",
            page_content="Revenue is calculated from net paid sales.",
            metadata={"source": "manual.pdf", "page": 1, "chunk": 0, "chunk_id": "revenue"},
        ),
        Document(
            id="customer",
            page_content="Customer demographics describe buyers.",
            metadata={"source": "manual.pdf", "page": 2, "chunk": 0, "chunk_id": "customer"},
        ),
    ]

    assert store.rebuild(documents) == 2
    assert store.count() == 2
    assert store.retrieve("revenue", top_k=1)[0].id == "doc-revenue"

    store.rebuild(documents[:1])
    assert store.count() == 1


# Verifies LangGraph fan-out runs independent validators concurrently.
def test_sql_validators_run_in_parallel_through_langgraph(demo_database_url: str) -> None:
    workflow = build_test_workflow(Settings(database_url=demo_database_url, sql_dialect="sqlite"))
    workflow.nodes.join_validator = SlowValidator("join_validator")
    workflow.nodes.aggregation_validator = SlowValidator("aggregation_validator")
    workflow.nodes.business_validator = SlowValidator("business_rule_validator")
    workflow.nodes.performance_agent = SlowValidator("performance_agent")

    start = perf_counter()
    response = workflow.run("Which region generated the most revenue in 2025?")
    elapsed = perf_counter() - start

    assert elapsed < 0.8
    assert [result.name for result in response.validations[:5]] == [
        "deterministic_sql_guardrail",
        "join_validator",
        "aggregation_validator",
        "business_rule_validator",
        "performance_agent",
    ]
