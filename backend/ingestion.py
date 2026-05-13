import datetime
import logging
import uuid

from sqlalchemy.orm import Session

from database import Concept, Edge, Paper, SessionLocal
from embeddings import store_embedding
from llm_client import enrich_concept_definition, extract_paper_metadata
from pdf_extractor import extract_pdf_text
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def ingest_paper(paper_id: str, file_path: str, kb_id: str, api_key: str | None = None, base_url: str | None = None, default_model: str | None = None, chat_model: str | None = None) -> None:
    """
    Full ingestion pipeline for one paper.
    Runs as a FastAPI BackgroundTask — creates its own DB session.

    Steps:
      1. Extract text from PDF
      2. Call LLM for structured metadata + concepts + relations
      3. Persist paper record
      4. Create / enrich concept nodes
      5. Create edges (paper→concept, paper→existing concept)
      6. Generate embeddings for paper and all touched concepts
    """
    db: Session = SessionLocal()
    try:
        paper = db.query(Paper).filter_by(id=paper_id).first()
        if not paper:
            logger.error("Paper %s not found in DB — aborting ingestion.", paper_id)
            return

        # ── 1. Extract text ──────────────────────────────────────────────────
        logger.info("[%s] Extracting PDF text …", paper_id)
        text, pages_extracted, total_pages = extract_pdf_text(file_path, max_chars=settings.max_text_chars)

        if not text.strip():
            paper.status = "failed"
            paper.error_message = "Could not extract text from PDF (possibly scanned image)."
            db.commit()
            return

        # ── 2. LLM extraction ────────────────────────────────────────────────
        existing_concepts = db.query(Concept).filter_by(kb_id=kb_id).all()
        concept_name_map: dict[str, Concept] = {c.name.lower(): c for c in existing_concepts}

        logger.info("[%s] Calling LLM for metadata extraction …", paper_id)
        metadata = await extract_paper_metadata(text, list(concept_name_map.keys()), api_key=api_key, base_url=base_url, model=default_model)

        # ── 3. Persist paper ─────────────────────────────────────────────────
        paper.title = metadata.get("title") or "Unknown Title"
        paper.authors = metadata.get("authors") or []
        paper.year = metadata.get("year")
        paper.venue = metadata.get("venue")
        paper.summary = metadata.get("summary") or ""
        paper.contributions = metadata.get("contributions") or []
        paper.key_findings = metadata.get("key_findings") or []
        paper.pages_extracted = pages_extracted
        paper.pages_total = total_pages
        paper.updated_at = datetime.datetime.utcnow()

        # ── 4. Process concepts_used ─────────────────────────────────────────
        for concept_data in metadata.get("concepts_used") or []:
            cname = (concept_data.get("name") or "").strip().lower()
            cdefinition = (concept_data.get("definition") or "").strip()

            if not cname:
                continue

            if cname in concept_name_map:
                # Enrich the existing definition
                existing = concept_name_map[cname]
                try:
                    enriched = await enrich_concept_definition(
                        cname,
                        existing.definition,
                        paper.title,
                        text[:8000],
                        api_key=api_key,
                        base_url=base_url,
                        model=default_model,
                    )
                    existing.definition = enriched
                    existing.updated_at = datetime.datetime.utcnow()
                except Exception as exc:
                    logger.warning("Could not enrich concept '%s': %s", cname, exc)
                concept_node = existing
            else:
                # Create new concept node
                concept_node = Concept(
                    id=str(uuid.uuid4()),
                    kb_id=kb_id,
                    name=cname,
                    definition=cdefinition or f"A concept introduced or used in: {paper.title}",
                )
                db.add(concept_node)
                db.flush()  # get the id without full commit
                concept_name_map[cname] = concept_node

            # Edge: paper --[uses]--> concept
            # Avoid duplicates
            already = (
                db.query(Edge)
                .filter_by(kb_id=kb_id, source_id=paper.id, target_id=concept_node.id, relation="uses")
                .first()
            )
            if not already:
                db.add(
                    Edge(
                        id=str(uuid.uuid4()),
                        kb_id=kb_id,
                        source_id=paper.id,
                        source_type="paper",
                        target_id=concept_node.id,
                        target_type="concept",
                        relation="uses",
                        label=f"{paper.title} uses {cname}",
                    )
                )

        # ── 5. Relation edges to existing concepts ───────────────────────────
        for rel in metadata.get("relations_to_existing") or []:
            rel_name = (rel.get("concept") or "").strip().lower()
            if rel_name not in concept_name_map:
                continue

            rel_concept = concept_name_map[rel_name]
            already = (
                db.query(Edge)
                .filter_by(
                    source_id=paper.id,
                    target_id=rel_concept.id,
                    relation=rel.get("relation", "related_to"),
                )
                .first()
            )
            if not already:
                db.add(
                    Edge(
                        id=str(uuid.uuid4()),
                        kb_id=kb_id,
                        source_id=paper.id,
                        source_type="paper",
                        target_id=rel_concept.id,
                        target_type="concept",
                        relation=rel.get("relation", "related_to"),
                        label=rel.get("description", ""),
                    )
                )

        paper.status = "ready"
        db.commit()
        logger.info("[%s] Ingestion complete: %s", paper_id, paper.title)

        # ── 6. Embeddings (non-critical, best-effort) ────────────────────────
        try:
            paper_content = (
                f"{paper.title}\n{paper.summary}\n"
                + " ".join(paper.contributions or [])
            )
            store_embedding(paper.id, "paper", paper_content, db, kb_id=kb_id)

            for concept in db.query(Concept).filter_by(kb_id=kb_id).all():
                store_embedding(
                    concept.id,
                    "concept",
                    f"{concept.name}: {concept.definition}",
                    db,
                    kb_id=kb_id,
                )
            db.commit()
        except Exception as exc:
            logger.warning("Embeddings could not be generated: %s", exc)

    except Exception as exc:
        logger.error("[%s] Ingestion failed: %s", paper_id, exc, exc_info=True)
        try:
            paper = db.query(Paper).filter_by(id=paper_id).first()
            if paper:
                paper.status = "failed"
                paper.error_message = str(exc)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()
