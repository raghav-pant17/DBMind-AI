"""Schema validation helpers for parsed SQL."""

from sqlglot import exp

from dbmind_ai.domain import SchemaContext


# Extracts SQL table aliases and maps them back to source table names.
def extract_table_aliases(tree: exp.Expression) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        name = table.name
        alias = table.alias_or_name
        aliases[alias] = name
        aliases[name] = name
    return aliases


# Finds table references that are not present in the retrieved schema.
def unknown_tables(tree: exp.Expression, schema: SchemaContext) -> list[str]:
    return sorted({table.name for table in tree.find_all(exp.Table)} - schema.table_names)


# Finds column references that are not present in the retrieved schema.
def unknown_columns(tree: exp.Expression, schema: SchemaContext) -> list[str]:
    aliases = extract_table_aliases(tree)
    projection_aliases = {
        expression.alias
        for expression in tree.expressions
        if isinstance(expression, exp.Alias) and expression.alias
    }
    unknown: set[str] = set()
    all_columns = set().union(*(schema.columns_for(table.name) for table in schema.tables))
    for column in tree.find_all(exp.Column):
        table = column.table
        name = column.name
        if table:
            resolved_table = aliases.get(table, table)
            if name not in schema.columns_for(resolved_table):
                unknown.add(f"{table}.{name}")
        elif name not in all_columns and name not in projection_aliases:
            unknown.add(name)
    return sorted(unknown)
