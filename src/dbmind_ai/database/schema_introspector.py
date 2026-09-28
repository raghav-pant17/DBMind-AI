"""Database schema introspection."""

import json
from pathlib import Path

from sqlalchemy import Engine, inspect

from dbmind_ai.domain import ColumnInfo, RelationshipInfo, SchemaContext, TableInfo


# Extracts schema metadata from the actual connected database.
class SchemaIntrospector:
    """Build schema context from live database metadata."""

    # Stores the database engine used for schema inspection.
    def __init__(self, engine: Engine, schema_name: str | None = None) -> None:
        self.engine = engine
        self.schema_name = schema_name or None

    # Reads tables, columns, primary keys, and foreign keys from the database.
    def inspect(self) -> SchemaContext:
        inspector = inspect(self.engine)
        tables: list[TableInfo] = []
        relationships: list[RelationshipInfo] = []

        for table_name in sorted(inspector.get_table_names(schema=self.schema_name)):
            if table_name.startswith("sqlite_"):
                continue
            primary_key = inspector.get_pk_constraint(
                table_name,
                schema=self.schema_name,
            ).get("constrained_columns", [])
            columns = [
                ColumnInfo(
                    name=column["name"],
                    type=str(column["type"]),
                    description="",
                )
                for column in inspector.get_columns(table_name, schema=self.schema_name)
            ]
            tables.append(
                TableInfo(
                    name=table_name,
                    description=f"Database table '{table_name}'.",
                    primary_key=primary_key,
                    columns=columns,
                )
            )

            for foreign_key in inspector.get_foreign_keys(table_name, schema=self.schema_name):
                referred_table = foreign_key.get("referred_table")
                local_columns = foreign_key.get("constrained_columns") or []
                remote_columns = foreign_key.get("referred_columns") or []
                if not referred_table:
                    continue
                for local_column, remote_column in zip(local_columns, remote_columns, strict=False):
                    relationships.append(
                        RelationshipInfo(
                            from_table=table_name,
                            from_column=local_column,
                            to_table=referred_table,
                            to_column=remote_column,
                            description=(
                                f"{table_name}.{local_column} references "
                                f"{referred_table}.{remote_column}."
                            ),
                        )
                    )

        return SchemaContext(tables=tables, relationships=relationships)

    # Writes the introspected schema context to a JSON file.
    def write_context(self, output_path: Path) -> SchemaContext:
        schema_context = self.inspect()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(schema_context.model_dump(), indent=2),
            encoding="utf-8",
        )
        return schema_context
