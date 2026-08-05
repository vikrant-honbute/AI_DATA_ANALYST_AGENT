# AI Data Analysis Agent

A professional LangGraph-based AI data analysis agent with deterministic execution tools and a Streamlit UI.

## Stack

- LangGraph: orchestration and conditional retry flow
- Groq + LangChain: planner/critic/insight LLM nodes using openai/gpt-oss-20b by default and openai/gpt-oss-120b for larger runs
- Pandas: declarative dataframe analysis and chart generation
- PostgreSQL + SQLAlchemy: structured read-only analysis
- MongoDB + PyMongo: lightweight, session-scoped memory store
- Streamlit: interactive app

## Project Structure

- project/main.py: CLI entry point
- project/streamlit_app.py: Streamlit entry point
- project/config.py: environment-driven settings
- project/llm.py: reusable Groq model factory
- project/graph/: workflow state, edges, nodes
- project/tools/: deterministic external tool adapters
- project/prompts/: reusable prompt templates and loader
- requirements.txt / pyproject.toml: Python dependencies
- .env.example: required environment variables
- tests/: pytest regression suite

## Workflow

START -> planner -> executor -> critic

- If retry is true and retry_count < 2, flow returns to executor.
- Otherwise flow continues to insight -> END.

## How Data Is Executed

- **Pandas actions are declarative JSON operations** (for example
  `{"operation": "groupby", "by": ["region"], "column": "sales", "function": "sum"}`),
  never Python source. The executor only implements a closed set of operations
  (head, select, filter, sort, value_counts, aggregate, groupby, correlation,
  memory_records) with strict column validation and row caps. Arbitrary code is
  not compiled or executed.
- **SQL is guarded in depth**: only single `SELECT`/`WITH` statements pass
  validation; a read-only transaction is opened, statement and lock timeouts are
  applied, the result set is capped, and dangerous functions such as `pg_sleep`
  or `nextval` are rejected. For production, grant the application role `SELECT`
  only, on an explicit schema allowlist (`ALLOWED_POSTGRES_SCHEMAS`), and rely on
  the database permissions rather than the regex as the security boundary.
- **Memory is scoped per session**: MongoDB records carry a session identifier
  and a 7-day TTL; the UI never reads another session's records.

## Privacy and Data Flow

- User queries, database/CSV metadata, selected session memory, and execution
  results are sent to the configured Groq LLM provider to plan, validate, and
  summarize analyses. Row-level values may appear in LLM prompts.
- Completed runs (query + result summary) may be stored in MongoDB for up to 7
  days, scoped to the session that produced them.
- Before deploying with real data, confirm your data agreements with the LLM
  provider and enforce database-role restrictions as described above.

## Environment Variables

Copy .env.example to .env and set:

- GROQ_API_KEY
- POSTGRES_URL
- MONGODB_URL

Optional guardrails:

- POSTGRES_STATEMENT_TIMEOUT_MS (default 10000)
- POSTGRES_MAX_ROWS (default 1000)
- ALLOWED_POSTGRES_SCHEMAS (default public, comma-separated)
- MAX_CSV_BYTES (default 10485760), MAX_CSV_ROWS (default 100000), MAX_CSV_COLUMNS (default 200)

## Run

Install dependencies:

```
pip install -r requirements.txt
```

Run the CLI (package mode, recommended):

```
python -m project.main
```

or as a plain script:

```
python project/main.py
```

Run the Streamlit app:

```
streamlit run project/streamlit_app.py
```

## Tests

```
pip install -e ".[dev]"
python -m pytest tests -q
```
