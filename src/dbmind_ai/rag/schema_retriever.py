"""Schema retrieval."""

import json
from pathlib import Path

from dbmind_ai.domain import SchemaContext, TableInfo
from dbmind_ai.rag.lexical_search import lexical_score


# Retrieves relevant schema context for a user question.
class SchemaRetriever:
    """Retrieve a compact schema context for a question."""

    # Stores introspected schema metadata for retrieval.
    def __init__(self, schema: SchemaContext) -> None:
        self._schema = schema

    # Loads schema context from a generated JSON file.
    @classmethod
    def from_file(cls, schema_path: Path) -> "SchemaRetriever":
        raw = json.loads(schema_path.read_text(encoding="utf-8"))
        return cls(SchemaContext.model_validate(raw))

    # Returns the most relevant tables and relationships for a question.
    def retrieve(self, question: str, top_k: int = 4) -> SchemaContext:
        scored_tables = sorted(
            self._schema.tables,
            key=lambda table: lexical_score(question, self._table_document(table)),
            reverse=True,
        )
        selected = [
            table for table in scored_tables[:top_k] if self._should_include(question, table)
        ]
        if not selected:
            selected = scored_tables[:top_k]
        selected_names = {table.name for table in selected}
        relationships = [
            relationship
            for relationship in self._schema.relationships
            if relationship.from_table in selected_names and relationship.to_table in selected_names
        ]
        return SchemaContext(tables=selected, relationships=relationships)

    # Returns the full loaded schema context.
    def all(self) -> SchemaContext:
        return self._schema

    # Converts a table definition into searchable text.
    @staticmethod
    def _table_document(table: TableInfo) -> str:
        columns = " ".join(f"{column.name} {column.description}" for column in table.columns)
        return f"{table.name} {table.description} {columns}"

    # Determines whether a table has lexical overlap with the question.
    @staticmethod
    def _should_include(question: str, table: TableInfo) -> bool:
        return lexical_score(question, SchemaRetriever._table_document(table)) > 0
