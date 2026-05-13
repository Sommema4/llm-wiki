from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from auth import get_current_user
from database import Concept, Edge, KnowledgeBase, Paper, User, get_db
from schemas import ConceptBase

router = APIRouter(prefix="/kbs/{kb_id}/concepts", tags=["concepts"])


def _check_kb(kb_id: str, user: User, db: Session) -> None:
    kb = db.query(KnowledgeBase).filter_by(id=kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found.")
    if kb.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your knowledge base.")


@router.get("/", response_model=list[ConceptBase])
def list_concepts(
    kb_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _check_kb(kb_id, current_user, db)
    return db.query(Concept).filter_by(kb_id=kb_id).order_by(Concept.name).all()


@router.get("/{concept_id}")
def get_concept(
    kb_id: str,
    concept_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _check_kb(kb_id, current_user, db)
    concept = db.query(Concept).filter_by(id=concept_id, kb_id=kb_id).first()
    if not concept:
        raise HTTPException(status_code=404, detail="Concept not found.")

    # Papers that use this concept
    paper_edges = (
        db.query(Edge)
        .filter(Edge.target_id == concept_id, Edge.source_type == "paper", Edge.kb_id == kb_id)
        .all()
    )
    paper_ids = [e.source_id for e in paper_edges]
    papers = db.query(Paper).filter(Paper.id.in_(paper_ids)).all()

    # Concept-to-concept edges
    cc_edges = (
        db.query(Edge)
        .filter(
            ((Edge.source_id == concept_id) | (Edge.target_id == concept_id)),
            Edge.source_type == "concept",
            Edge.kb_id == kb_id,
        )
        .all()
    )
    related_ids = {
        e.target_id if e.source_id == concept_id else e.source_id
        for e in cc_edges
    }
    related = db.query(Concept).filter(Concept.id.in_(related_ids)).all()

    return {
        "id": concept.id,
        "name": concept.name,
        "definition": concept.definition,
        "created_at": concept.created_at,
        "updated_at": concept.updated_at,
        "papers": [
            {"id": p.id, "title": p.title, "year": p.year} for p in papers
        ],
        "related_concepts": [
            {"id": c.id, "name": c.name} for c in related
        ],
    }
