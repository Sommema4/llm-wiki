import hashlib
import logging
from typing import List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from database import NodeEmbedding

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from fastembed import TextEmbedding
            from config import get_settings
            _model = TextEmbedding(model_name=get_settings().embedding_model)
            logger.info("Embedding model loaded.")
        except ImportError as exc:
            raise ImportError(
                "fastembed is not installed. Run: pip install fastembed"
            ) from exc
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return embeddings for a list of texts as plain Python float lists."""
    model = _get_model()
    return [emb.tolist() for emb in model.embed(texts)]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a), np.array(b)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def store_embedding(
    node_id: str,
    node_type: str,
    content: str,
    db: Session,
    kb_id: str = "",
) -> None:
    """Generate and persist an embedding for a node, skipping if unchanged."""
    content_hash = hashlib.md5(content.encode()).hexdigest()

    existing = db.query(NodeEmbedding).filter_by(node_id=node_id).first()
    if existing and existing.content_hash == content_hash:
        return  # content hasn't changed — no-op

    [embedding] = embed_texts([content])

    if existing:
        existing.embedding = embedding
        existing.content_hash = content_hash
    else:
        db.add(
            NodeEmbedding(
                node_id=node_id,
                node_type=node_type,
                kb_id=kb_id,
                embedding=embedding,
                content_hash=content_hash,
            )
        )


def semantic_search(
    query: str,
    db: Session,
    top_k: int = 5,
    node_type: Optional[str] = None,
    kb_id: Optional[str] = None,
) -> List[Tuple[str, str, float]]:
    """Return [(node_id, node_type, score)] sorted by cosine similarity."""
    [query_embedding] = embed_texts([query])

    q = db.query(NodeEmbedding)
    if node_type:
        q = q.filter_by(node_type=node_type)
    if kb_id:
        q = q.filter_by(kb_id=kb_id)
    all_records = q.all()

    if not all_records:
        return []

    scores = [
        (rec.node_id, rec.node_type, cosine_similarity(query_embedding, rec.embedding))
        for rec in all_records
    ]
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_k]


def find_similar_concept_pairs(
    db: Session,
    threshold: float = 0.92,
    kb_id: Optional[str] = None,
) -> List[Tuple[str, str, float]]:
    """Find pairs of concept nodes whose embeddings exceed the similarity threshold."""
    q = db.query(NodeEmbedding).filter_by(node_type="concept")
    if kb_id:
        q = q.filter_by(kb_id=kb_id)
    records = q.all()

    if len(records) < 2:
        return []

    pairs: List[Tuple[str, str, float]] = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            sim = cosine_similarity(records[i].embedding, records[j].embedding)
            if sim >= threshold:
                pairs.append((records[i].node_id, records[j].node_id, sim))
    return pairs
