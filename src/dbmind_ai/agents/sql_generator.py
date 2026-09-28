"""LLM SQL generation agent."""

import re

from langchain_core.language_models import BaseChatModel

from dbmind_ai.agents.llm_content import content_to_text
from dbmind_ai.domain import BusinessRule, Intent, SchemaContext

SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


# Generates SQL from question intent and retrieved context using the configured LLM.
class SQLGenerator:
    """Generate SQL using a LangChain chat model."""

    def __init__(
        self,
        llm: BaseChatModel,
        dialect: str = "sqlite",
    ) -> None:
        self.llm = llm
        self.dialect = dialect

    # Sends schema, rules, and intent to the model and returns SQL only.
    def run(
        self,
        question: str,
        intent: Intent,
        schema: SchemaContext,
        business_rules: list[BusinessRule],
    ) -> str:
        response = self.llm.invoke(self._build_prompt(question, intent, schema, business_rules))
        content = content_to_text(response.content)
        return self._extract_sql(content)

    # Builds the prompt that constrains the model to schema-aware SQL.
    def _build_prompt(
        self,
        question: str,
        intent: Intent,
        schema: SchemaContext,
        business_rules: list[BusinessRule],
    ) -> str:
        return f"""
You are DBMind-AI's SQL generation agent.

Generate one {self.dialect} SQL query for the user's request.

Rules:
- Return SQL only.
- Do not wrap the answer in markdown.
- Use only the tables and columns listed in the schema.
- If the user asks for SELECT/INSERT/UPDATE/DELETE/DROP/CREATE/ALTER, generate that SQL.
- Include a LIMIT for SELECT queries when the query is not a single aggregate answer.
- Follow business rules when they apply.
- Prefer explicit joins using known relationships.
- Do not explain whether the query is safe. DBMind-AI validates safety after generation.

User question:
{question}

Structured intent:
{intent.model_dump_json(indent=2)}

Schema:
{self._format_schema(schema)}

Business rules:
{self._format_rules(business_rules)}
""".strip()

    # Formats retrieved schema as compact prompt context.
    @staticmethod
    def _format_schema(schema: SchemaContext) -> str:
        lines: list[str] = []
        for table in schema.tables:
            lines.append(f"Table: {table.name}")
            for column in table.columns:
                lines.append(f"- {column.name} ({column.type}): {column.description}")
        if schema.relationships:
            lines.append("Relationships:")
            for relationship in schema.relationships:
                lines.append(
                    "- "
                    f"{relationship.from_table}.{relationship.from_column} -> "
                    f"{relationship.to_table}.{relationship.to_column}"
                )
        return "\n".join(lines)

    # Formats retrieved business rules as compact prompt context.
    @staticmethod
    def _format_rules(business_rules: list[BusinessRule]) -> str:
        return "\n\n".join(f"{rule.name}: {rule.description}" for rule in business_rules)

    # Extracts raw SQL from model output.
    @staticmethod
    def _extract_sql(content: str) -> str:
        match = SQL_FENCE_RE.search(content)
        sql = match.group(1) if match else content
        return sql.strip().rstrip(";") + ";"
