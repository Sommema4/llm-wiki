from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class PaperBase(BaseModel):
    id: str
    title: Optional[str] = None
    authors: Optional[List[str]] = []
    year: Optional[int] = None
    venue: Optional[str] = None
    summary: Optional[str] = None
    contributions: Optional[List[str]] = []
    key_findings: Optional[List[str]] = []
    status: str
    error_message: Optional[str] = None
    pages_extracted: Optional[int] = None
    pages_total: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ConceptBase(BaseModel):
    id: str
    name: str
    concept_type: Optional[str] = None
    summary: Optional[str] = None
    explanation: Optional[str] = None
    definition: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class EdgeBase(BaseModel):
    id: str
    source_id: str
    source_type: str
    target_id: str
    target_type: str
    relation: str
    label: Optional[str] = None

    model_config = {"from_attributes": True}


class LintReportResponse(BaseModel):
    id: str
    status: str
    created_at: datetime
    report: Optional[Dict[str, Any]] = None

    model_config = {"from_attributes": True}
