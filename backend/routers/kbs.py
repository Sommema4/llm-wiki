import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import get_current_user
from database import (
    Concept, Edge, KnowledgeBase, LintReport, NodeEmbedding, Paper, User, get_db,
)

router = APIRouter(prefix="/kbs", tags=["knowledge bases"])


class KBCreate(BaseModel):
    name: str


class KBResponse(BaseModel):
    id: str
    name: str

    model_config = {"from_attributes": True}


def _owned_kb(kb_id: str, user: User, db: Session) -> KnowledgeBase:
    """Return the KB if it belongs to the current user, else 404/403."""
    kb = db.query(KnowledgeBase).filter_by(id=kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found.")
    if kb.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your knowledge base.")
    return kb


@router.get("/", response_model=List[KBResponse])
def list_kbs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return db.query(KnowledgeBase).filter_by(user_id=current_user.id).order_by(KnowledgeBase.name).all()


@router.post("/", response_model=KBResponse, status_code=201)
def create_kb(
    body: KBCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name must not be empty.")
    kb = KnowledgeBase(id=str(uuid.uuid4()), name=name, user_id=current_user.id)
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return kb


@router.delete("/{kb_id}", status_code=204)
def delete_kb(
    kb_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    kb = _owned_kb(kb_id, current_user, db)

    # Cascade: delete all uploads from disk before DB rows
    for paper in db.query(Paper).filter_by(kb_id=kb_id).all():
        import os
        if paper.file_path and os.path.exists(paper.file_path):
            os.remove(paper.file_path)

    db.delete(kb)
    db.commit()
