"""
Hybrid TAG + RAG orchestrator.
1. TAG  → SQL result (precise numbers)
2. RAG  → relevant free-text snippets (qualitative context)
3. LLM  → synthesise both into an executive answer
"""

import ollama
from app.agents.sql_agent import question_to_sql, narrate, run_tag
from app.agents.rag_agent import retrieve
from app.db import run_query
from app.config import settings

_SYNTH_SYSTEM = """\
You are a senior sales analyst presenting to executives.
You have been given:
  1. A structured SQL result answering the question numerically.
  2. Relevant notes and activity logs for qualitative context.

Combine both to give a crisp, insightful answer.
- Start with the key number / finding.
- Add 1-2 sentences of qualitative context where relevant.
- Flag any anomalies or risks.
- Keep the response under 200 words.
"""


def answer(question: str, use_rag: bool = True) -> dict:
    # Step 1: TAG
    sql = question_to_sql(question)
    try:
        rows = run_query(sql)
    except Exception as exc:
        return {"error": str(exc), "sql": sql, "rows": [], "answer": "", "context": []}

    # Step 2: RAG (optional)
    context_snippets: list[str] = []
    if use_rag:
        context_snippets = retrieve(question, top_k=4)

    # Step 3: Synthesis
    if context_snippets:
        data_str = "\n".join(str(r) for r in rows[:30])
        ctx_str = "\n".join(context_snippets)
        prompt = (
            f"Question: {question}\n\n"
            f"SQL result ({len(rows)} rows):\n{data_str}\n\n"
            f"Relevant activity/notes context:\n{ctx_str}"
        )
        response = ollama.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": _SYNTH_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3},
        )
        final_answer = response["message"]["content"].strip()
    else:
        final_answer = narrate(question, sql, rows)

    return {
        "sql": sql,
        "rows": rows,
        "context": context_snippets,
        "answer": final_answer,
    }
