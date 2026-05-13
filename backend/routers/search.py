from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import get_current_user
from database import Concept, KnowledgeBase, Paper, User, get_db
from embeddings import semantic_search
from llm_client import answer_question

router = APIRouter(prefix="/kbs/{kb_id}/search", tags=["search"])


class HistoryTurn(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str
    history: Optional[List[HistoryTurn]] = []
    top_k: int = 5


def _check_kb(kb_id: str, user: User, db: Session) -> None:
    kb = db.query(KnowledgeBase).filter_by(id=kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found.")
    if kb.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your knowledge base.")


@router.get("/")
def search_nodes(
    kb_id: str,
    q: str = Query(..., min_length=1, description="Search query"),
    node_type: Optional[str] = Query(None, description="Filter by 'paper' or 'concept'"),
    top_k: int = Query(5, ge=1, le=20),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Semantic similarity search over all wiki nodes."""
    _check_kb(kb_id, current_user, db)
    try:
        raw = semantic_search(q, db, top_k=top_k, node_type=node_type, kb_id=kb_id)
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Embedding model not available: {exc}",
        )

    results = []
    for node_id, ntype, score in raw:
        if ntype == "paper":
            obj = db.query(Paper).filter_by(id=node_id, kb_id=kb_id).first()
            if obj:
                results.append(
                    {
                        "id": obj.id,
                        "type": "paper",
                        "name": obj.title,
                        "summary": obj.summary,
                        "score": round(score, 3),
                    }
                )
        elif ntype == "concept":
            obj = db.query(Concept).filter_by(id=node_id, kb_id=kb_id).first()
            if obj:
                results.append(
                    {
                        "id": obj.id,
                        "type": "concept",
                        "name": obj.name,
                        "definition": obj.definition,
                        "score": round(score, 3),
                    }
                )

    return {"query": q, "results": results}


@router.post("/query")
async def query_wiki(
    kb_id: str,
    body: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    RAG-style question answering with optional conversation history.
    """
    _check_kb(kb_id, current_user, db)
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    top_k = max(1, min(body.top_k, 15))

    try:
        raw = semantic_search(question, db, top_k=top_k, kb_id=kb_id)
    except ImportError:
        raw = []

    context_nodes = []
    for node_id, ntype, _score in raw:
        if ntype == "paper":
            obj = db.query(Paper).filter_by(id=node_id, kb_id=kb_id).first()
            if obj:
                content = (
                    f"Summary: {obj.summary}\n"
                    f"Contributions: {', '.join(obj.contributions or [])}\n"
                    f"Key findings: {', '.join(obj.key_findings or [])}"
                )
                context_nodes.append(
                    {"type": "paper", "name": obj.title, "content": content, "id": obj.id}
                )
        elif ntype == "concept":
            obj = db.query(Concept).filter_by(id=node_id, kb_id=kb_id).first()
            if obj:
                context_nodes.append(
                    {
                        "type": "concept",
                        "name": obj.name,
                        "content": obj.definition,
                        "id": obj.id,
                    }
                )

    if not context_nodes:
        return {
            "answer": "No relevant information found in the wiki yet.",
            "sources": [],
        }

    # Keep last 6 turns (3 exchanges) to stay within context budget
    trimmed_history = [{"role": t.role, "content": t.content} for t in (body.history or [])][-6:]

    try:
        answer = await answer_question(question, context_nodes, history=trimmed_history, api_key=current_user.openrouter_api_key)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "answer": answer,
        "sources": [
            {"id": n["id"], "type": n["type"], "name": n["name"]}
            for n in context_nodes
        ],
    }
