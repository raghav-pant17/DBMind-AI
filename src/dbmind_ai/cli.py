"""Command line interface."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dbmind_ai.config import get_settings
from dbmind_ai.database.connection import create_database_engine
from dbmind_ai.database.schema_introspector import SchemaIntrospector
from dbmind_ai.factory import build_workflow, resolve_business_context_path
from dbmind_ai.rag.chroma_retriever import ChromaRetriever
from dbmind_ai.rag.document_loader import load_pdf_documents

app = typer.Typer(help="DBMind-AI command line tools.")
console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Resolves a configured path relative to the project root.
def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


# Generates schema_context.json from the configured database.
@app.command()
def introspect_schema() -> None:
    """Generate schema context from the configured database."""

    settings = get_settings()
    engine = create_database_engine(settings.database_url, settings.database_schema)
    SchemaIntrospector(
        engine,
        settings.database_schema,
    ).write_context(resolve_project_path(settings.schema_context_path))
    console.print(f"[green]Schema context written to {settings.schema_context_path}.[/green]")


# Builds or refreshes the local ChromaDB RAG index.
@app.command()
def index_rag() -> None:
    """Build the local ChromaDB index for configured PDFs."""

    settings = get_settings()
    business_context_path = resolve_business_context_path(settings, PROJECT_ROOT)
    documents = load_pdf_documents(business_context_path)
    chroma_path = resolve_project_path(settings.chroma_path)
    retriever = ChromaRetriever(
        chroma_path,
        f"{settings.chroma_collection_prefix}_documents",
    )
    chunk_count = retriever.rebuild(documents)
    console.print(
        f"[green]PDF RAG index built.[/green]\nDocument chunks: {chunk_count}\nPath: {chroma_path}"
    )


# Runs a natural-language question through the DBMind-AI workflow.
@app.command()
def ask(question: str, json_output: bool = False) -> None:
    """Ask a natural-language business question."""

    workflow = build_workflow()
    try:
        response = workflow.run(question)
    except Exception as exc:
        console.print(f"[red]DBMind-AI run failed: {exc}[/red]")
        raise typer.Exit(1) from exc
    if json_output:
        console.print(json.dumps(response.model_dump(), indent=2, default=str))
        return

    console.print(response.answer)
    console.print("\n[bold]Generated SQL for verification[/bold]")
    console.print(response.sql)
    table = Table(title="Validations")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Score")
    table.add_column("Messages")
    for validation in response.validations:
        table.add_row(
            validation.name,
            validation.status.value,
            str(validation.score),
            "\n".join(validation.messages),
        )
    console.print(table)


if __name__ == "__main__":
    app()
