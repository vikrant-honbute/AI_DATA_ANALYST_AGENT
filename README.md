# AI Data Analysis Agent

A professional LangGraph-based AI data analysis agent with deterministic execution tools and a Streamlit UI.

## Stack

- LangGraph: orchestration and conditional retry flow
- Groq + LangChain: planner/critic/insight LLM nodes using openai/gpt-oss-20b by default and openai/gpt-oss-120b for larger runs
- Pandas: dataframe analysis and chart generation
- PostgreSQL + SQLAlchemy: structured read-only analysis
- MongoDB + PyMongo: lightweight memory store
- Streamlit: interactive app

## Project Structure

- project/main.py: CLI entry point
- project/streamlit_app.py: Streamlit entry point
- project/config.py: environment-driven settings
- project/llm.py: reusable Groq model factory
- project/graph/: workflow state, edges, nodes
- project/tools/: deterministic external tool adapters
- project/prompts/: reusable prompt templates and loader
- requirements.txt: Python dependencies
- .env.example: required environment variables

## Workflow

START -> planner -> executor -> critic

- If retry is true and retry_count < 2, flow returns to executor.
- Otherwise flow continues to insight -> END.

## Prompt System

Prompt templates are stored in project/prompts and loaded via project/prompts/loader.py.
This keeps node code clean and allows prompt tuning without logic changes.

## Environment Variables

Copy .env.example to .env and set:

- GROQ_API_KEY
- POSTGRES_URL
- MONGODB_URL

## Run

Install dependencies:

pip install -r requirements.txt

Run CLI:

python project/main.py

Run Streamlit:

streamlit run project/streamlit_app.py
