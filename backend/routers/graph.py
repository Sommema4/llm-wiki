from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from auth import get_current_user
from database import Concept, Edge, KnowledgeBase, Paper, User, get_db
from linter import build_graph, compute_clusters

router = APIRouter(prefix="/kbs/{kb_id}/graph", tags=["graph"])


@router.get("/")
def get_graph(
    kb_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns the full graph payload for frontend visualisation:
      - nodes  (papers + concepts)
      - edges
      - clusters (Louvain communities)
    """
    kb = db.query(KnowledgeBase).filter_by(id=kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found.")
    if kb.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your knowledge base.")

    papers = db.query(Paper).filter_by(status="ready", kb_id=kb_id).all()
    concepts = db.query(Concept).filter_by(kb_id=kb_id).all()
    edges = db.query(Edge).filter_by(kb_id=kb_id).all()

    nodes = []
    for p in papers:
        nodes.append(
            {
                "id": p.id,
                "type": "paper",
                "name": p.title or "Unknown",
                "year": p.year,
                "venue": p.venue,
                "summary": p.summary,
            }
        )
    for c in concepts:
        nodes.append(
            {
                "id": c.id,
                "type": "concept",
                "name": c.name,
                "definition": c.definition,
            }
        )

    edge_list = [
        {
            "id": e.id,
            "source": e.source_id,
            "target": e.target_id,
            "relation": e.relation,
            "label": e.label,
        }
        for e in edges
    ]

    # Clusters
    G = build_graph(db, kb_id)
    cluster_list = []
    try:
        for i, cluster_nodes in enumerate(compute_clusters(G)):
            cluster_list.append({"id": i, "node_ids": cluster_nodes, "size": len(cluster_nodes)})
    except Exception:
        pass

    return {"nodes": nodes, "edges": edge_list, "clusters": cluster_list}
