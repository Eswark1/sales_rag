"""
FastAPI backend – exposes the hybrid TAG+RAG pipeline as a REST API.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.orchestrator import answer
from app.agents.rag_agent import build_index
from app.db import get_schema
import logging

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

app = FastAPI(title="Sales RAG API", version="0.1.0")


@app.on_event("startup")
async def startup():
    logger.info("Building/verifying RAG vector index…")
    n = build_index()
    logger.info(f"Vector index ready: {n} documents indexed.")


class QuestionRequest(BaseModel):
    question: str
    use_rag: bool = True


class AnswerResponse(BaseModel):
    question: str
    sql: str
    rows: list[dict]
    context: list[str]
    answer: str


@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    result = answer(req.question, use_rag=req.use_rag)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return AnswerResponse(question=req.question, **result)


@app.get("/schema")
def schema():
    return {"schema": get_schema()}


@app.get("/health")
def health():
    return {"status": "ok"}
