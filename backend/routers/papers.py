import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from auth import get_current_user
from database import Edge, KnowledgeBase, NodeEmbedding, Paper, User, get_db
from ingestion import ingest_paper
from schemas import PaperBase
from config import get_settings

router = APIRouter(prefix="/kbs/{kb_id}/papers", tags=["papers"])
settings = get_settings()


def _check_kb(kb_id: str, user: User, db: Session) -> None:
    kb = db.query(KnowledgeBase).filter_by(id=kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found.")
    if kb.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your knowledge base.")


@router.post("/upload", response_model=PaperBase, status_code=202)
async def upload_paper(
    kb_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _check_kb(kb_id, current_user, db)
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    paper_id = str(uuid.uuid4())
    file_path = str(upload_dir / f"{paper_id}.pdf")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    paper = Paper(id=paper_id, kb_id=kb_id, file_path=file_path, status="processing")
    db.add(paper)
    db.commit()
    db.refresh(paper)

    background_tasks.add_task(ingest_paper, paper_id, file_path, kb_id, current_user.openrouter_api_key)

    return paper


@router.get("/", response_model=list[PaperBase])
def list_papers(
    kb_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _check_kb(kb_id, current_user, db)
    return db.query(Paper).filter_by(kb_id=kb_id).order_by(Paper.created_at.desc()).all()


@router.get("/{paper_id}", response_model=PaperBase)
def get_paper(
    kb_id: str,
    paper_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _check_kb(kb_id, current_user, db)
    paper = db.query(Paper).filter_by(id=paper_id, kb_id=kb_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return paper


@router.delete("/{paper_id}", status_code=204)
def delete_paper(
    kb_id: str,
    paper_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _check_kb(kb_id, current_user, db)
    paper = db.query(Paper).filter_by(id=paper_id, kb_id=kb_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")

    if paper.file_path and os.path.exists(paper.file_path):
        os.remove(paper.file_path)

    db.query(Edge).filter(
        (Edge.source_id == paper_id) | (Edge.target_id == paper_id)
    ).delete(synchronize_session=False)
    db.query(NodeEmbedding).filter_by(node_id=paper_id).delete()
    db.delete(paper)
    db.commit()


@router.post("/{paper_id}/reingest", response_model=PaperBase, status_code=202)
async def reingest_paper(
    kb_id: str,
    paper_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Re-run ingestion for a paper that failed or needs refreshing."""
    _check_kb(kb_id, current_user, db)
    paper = db.query(Paper).filter_by(id=paper_id, kb_id=kb_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    if not paper.file_path or not os.path.exists(paper.file_path):
        raise HTTPException(status_code=409, detail="Original PDF file is no longer on disk.")

    paper.status = "processing"
    paper.error_message = None
    db.commit()
    db.refresh(paper)

    background_tasks.add_task(ingest_paper, paper_id, paper.file_path, kb_id, current_user.openrouter_api_key)
    return paper
