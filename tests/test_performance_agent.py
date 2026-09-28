from dbmind_ai.agents.performance_agent import PerformanceAgent
from dbmind_ai.database.connection import create_database_engine


# Provides a testable performance agent backed by a temporary database.
class InspectablePerformanceAgent(PerformanceAgent):
    # Stores only the fields needed for deterministic helper tests.
    def __init__(self, database_url: str, dialect: str = "sqlite") -> None:
        self.engine = create_database_engine(database_url)
        self.dialect = dialect


# Verifies static performance findings detect broad SELECT queries.
def test_static_findings_detect_select_star_and_missing_filters(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path / 'demo.db'}"
    agent = InspectablePerformanceAgent(database_url)

    findings = agent._static_findings("SELECT * FROM orders")

    assert "Query has no WHERE filter." in findings
    assert "Query has no LIMIT." in findings
    assert "Query uses SELECT *." in findings


# Verifies EXPLAIN returns a query plan from the database.
def test_explain_returns_query_plan(demo_database_url: str) -> None:
    agent = InspectablePerformanceAgent(demo_database_url)

    plan = agent._explain("SELECT order_id FROM orders LIMIT 1")

    assert plan


# Verifies the performance agent returns a warning for broad queries.
def test_run_warns_for_broad_query(demo_database_url: str) -> None:
    agent = InspectablePerformanceAgent(demo_database_url)

    result = agent.run("SELECT * FROM orders")

    assert result.status == "warn"
    assert result.score < 100
