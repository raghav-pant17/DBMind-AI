# DBMind-AI

DBMind-AI turns plain-English business questions into validated, read-only SQL.
It connects to PostgreSQL, retrieves the relevant schema and PDF documentation,
generates SQL with an LLM, checks the query, and executes it only when it passes
the configured trust threshold.

## What It Does

- Reads tables, columns, primary keys, and foreign keys from PostgreSQL.
- Uses local ChromaDB embeddings to retrieve relevant PDF context.
- Generates SQL with a configurable LangChain chat model.
- Blocks destructive and schema-invalid SQL before execution.
- Validates joins, aggregations, business meaning, performance, and results.
- Returns the answer, generated SQL, validation details, and trust score.

## Setup

DBMind-AI requires Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Copy `.env.example` to `.env`, then configure your PostgreSQL connection and LLM:

```env
DBMIND_DATABASE_URL=postgresql+psycopg://user:password@host:5432/database
DBMIND_DATABASE_SCHEMA=public
DBMIND_SQL_DIALECT=postgres

DBMIND_LLM_MODEL=google_genai:gemini-3.6-flash
GOOGLE_API_KEY=your_key

DBMIND_SCHEMA_CONTEXT_PATH=./data/schema_context.json
DBMIND_BUSINESS_CONTEXT_PATH=./data/internal_docs
DBMIND_CHROMA_PATH=./data/chroma
DBMIND_CHROMA_COLLECTION_PREFIX=dbmind
```

Provider-prefixed LangChain model names are supported. For example:

```env
DBMIND_LLM_MODEL=openai:o3
OPENAI_API_KEY=your_key
```

## Prepare The Context

Generate `schema_context.json` from the configured PostgreSQL database:

```powershell
dbmind-ai introspect-schema
```

Add reference PDFs to `data/internal_docs`, then build the local Chroma index:

```powershell
dbmind-ai index-rag
```

Run these commands again whenever the database schema or PDFs change.

## Ask A Question

```powershell
dbmind-ai ask "Which region generated the most revenue in 2025?"
```

For structured output:

```powershell
dbmind-ai ask "Which region generated the most revenue in 2025?" --json-output
```

The response includes the business answer, generated SQL, trust score, and every
validation result.

## How It Works

```text
Question
  -> intent extraction
  -> relevant PostgreSQL schema retrieval
  -> relevant PDF context retrieval from ChromaDB
  -> SQL generation
  -> deterministic and semantic validation
  -> correction when required
  -> read-only execution
  -> final business answer
```

Schema retrieval uses deterministic token matching against `schema_context.json`.
PDF retrieval uses ChromaDB's local embedding model and returns the four most
relevant chunks. Local embeddings do not consume LLM provider credits.

## Safety

DBMind-AI executes only validated `SELECT` or read-only CTE queries. SQLGlot checks
syntax, referenced tables and columns, forbidden operations, and query limits before
the database executor runs anything. The PostgreSQL account should still be granted
read-only permissions as an additional database-level safeguard.

## Project Layout

```text
src/dbmind_ai/
  agents/       LLM generation and validation agents
  database/     PostgreSQL connection, introspection, and execution
  graph/        LangGraph workflow
  guardrails/   deterministic SQL safety checks
  rag/          PDF loading, Chroma retrieval, and schema retrieval

data/
  internal_docs/       source PDFs
  schema_context.json  generated PostgreSQL metadata
  chroma/              generated local vector index
```

## Tests

The test suite uses a temporary SQLite database and does not modify PostgreSQL.

```powershell
python -m pytest
python -m ruff check src tests
```
