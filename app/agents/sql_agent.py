"""
TAG engine: Natural Language → SQL → Execute → Narrate
"""

import re
import ollama
from app.config import settings
from app.db import get_schema, run_query

_SQL_SYSTEM = """\
You are a SQLite expert. Given the schema below and a business question,
return ONLY a valid SQLite SELECT statement — no markdown fences, no explanation.

Rules:
- Use only tables and columns that exist in the schema.
- Always use table aliases.
- Prefer readable column aliases (e.g. AS total_revenue).
- Use strftime for date comparisons.
- Limit results to 100 rows unless the user asks for more.

Schema:
{schema}
"""

_NARRATE_SYSTEM = """\
You are an executive business analyst. Given a question, a SQL query, and its results,
write a concise, plain-English summary for a C-level executive.
- Lead with the most important number or insight.
- Use bullet points for lists of 3+ items.
- Do NOT repeat the SQL.
- Be factual; do not speculate beyond the data provided.
"""


def question_to_sql(question: str) -> str:
    schema = get_schema()
    response = ollama.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": _SQL_SYSTEM.format(schema=schema)},
            {"role": "user", "content": question},
        ],
        options={"temperature": 0.0},
    )
    raw = response["message"]["content"].strip()
    # Strip accidental markdown fences
    raw = re.sub(r"```sql\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```\s*", "", raw)
    return raw.strip()


def narrate(question: str, sql: str, rows: list[dict]) -> str:
    data_str = "\n".join(str(r) for r in rows[:50])  # cap to avoid token overflow
    prompt = (
        f"Question: {question}\n\n"
        f"SQL executed:\n{sql}\n\n"
        f"Result rows ({len(rows)} total, showing up to 50):\n{data_str}"
    )
    response = ollama.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": _NARRATE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.3},
    )
    return response["message"]["content"].strip()


def run_tag(question: str) -> dict:
    sql = question_to_sql(question)
    rows = run_query(sql)
    answer = narrate(question, sql, rows)
    return {"sql": sql, "rows": rows, "answer": answer}
