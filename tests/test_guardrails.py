from dbmind_ai.database.connection import create_database_engine
from dbmind_ai.database.schema_introspector import SchemaIntrospector
from dbmind_ai.domain import ValidationStatus
from dbmind_ai.guardrails.sqlglot_validator import SQLGlotValidator


# Verifies unsafe SQL statements are blocked by the guardrail.
def test_blocks_write_sql(demo_database_url: str) -> None:
    engine = create_database_engine(demo_database_url)
    schema = SchemaIntrospector(engine).inspect()
    result = SQLGlotValidator().validate("DROP TABLE orders", schema)
    assert result.status == ValidationStatus.FAIL


# Verifies known SELECT statements pass deterministic validation.
def test_allows_known_select(demo_database_url: str) -> None:
    engine = create_database_engine(demo_database_url)
    schema = SchemaIntrospector(engine).inspect()
    sql = "SELECT order_id, total_amount FROM orders LIMIT 10"
    result = SQLGlotValidator().validate(sql, schema)
    assert result.status == ValidationStatus.PASS
