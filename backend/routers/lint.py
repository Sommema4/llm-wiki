import uuid
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from auth import get_current_user
from database import KnowledgeBase, LintReport, User, get_db
from linter import run_lint
from schemas import LintReportResponse

router = APIRouter(prefix="/kbs/{kb_id}/lint", tags=["lint"])


def _check_kb(kb_id: str, user: User, db: Session) -> None:
    kb = db.query(KnowledgeBase).filter_by(id=kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found.")
    if kb.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your knowledge base.")


@router.post("/run", response_model=LintReportResponse, status_code=202)
async def start_lint(
    kb_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Kick off a lint job. Returns immediately with the report ID to poll."""
    _check_kb(kb_id, current_user, db)
    report_id = str(uuid.uuid4())
    report = LintReport(id=report_id, kb_id=kb_id, status="running")
    db.add(report)
    db.commit()
    db.refresh(report)

    background_tasks.add_task(run_lint, report_id, kb_id, current_user.openrouter_api_key)

    return report


@router.get("/status/{report_id}", response_model=LintReportResponse)
def get_lint_status(
    kb_id: str,
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _check_kb(kb_id, current_user, db)
    report = db.query(LintReport).filter_by(id=report_id, kb_id=kb_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Lint report not found.")
    return report


@router.get("/reports", response_model=List[LintReportResponse])
def list_lint_reports(
    kb_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _check_kb(kb_id, current_user, db)
    return (
        db.query(LintReport)
        .filter_by(kb_id=kb_id)
        .order_by(LintReport.created_at.desc())
        .limit(20)
        .all()
    )
