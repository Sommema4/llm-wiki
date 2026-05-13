import logging
import uuid
from typing import Any, Dict, List

import networkx as nx
from sqlalchemy.orm import Session

from database import Concept, Edge, LintReport, NodeEmbedding, Paper, SessionLocal
from embeddings import find_similar_concept_pairs, store_embedding
from llm_client import (
    check_concept_relation,
    confirm_concept_merge,
    score_concept_quality,
)

logger = logging.getLogger(__name__)


# ── Graph helpers ──────────────────────────────────────────────────────────────

def build_graph(db: Session, kb_id: str) -> nx.Graph:
    """Build a NetworkX graph from the current DB state for one KB."""
    G = nx.Graph()

    for p in db.query(Paper).filter_by(status="ready", kb_id=kb_id).all():
        G.add_node(p.id, type="paper", name=p.title or "Unknown")
    for c in db.query(Concept).filter_by(kb_id=kb_id).all():
        G.add_node(c.id, type="concept", name=c.name)
    for e in db.query(Edge).filter_by(kb_id=kb_id).all():
        if G.has_node(e.source_id) and G.has_node(e.target_id):
            G.add_edge(e.source_id, e.target_id, relation=e.relation, edge_id=e.id)

    return G


def compute_clusters(G: nx.Graph) -> List[List[str]]:
    """
    Cluster nodes using Louvain community detection (NetworkX ≥ 3.x).
    Falls back to connected components if the graph is empty or the
    algorithm is unavailable.
    """
    if len(G.nodes) == 0:
        return []
    try:
        communities = nx.community.louvain_communities(G, seed=42)
        return [list(c) for c in communities]
    except Exception:
        return [list(c) for c in nx.connected_components(G)]


# ── Lint pipeline ──────────────────────────────────────────────────────────────

async def run_lint(report_id: str, kb_id: str, api_key: str | None = None) -> None:
    """
    Full lint pipeline. Called as a FastAPI BackgroundTask.

    Pass 1 — Deduplication:   merge concept pairs with embedding similarity ≥ 0.92
    Pass 2 — Orphan detection: flag nodes with degree < 2
    Pass 3 — Missing edges:   discover unlabelled relations inside each cluster
    Pass 4 — Quality check:   score concept definitions, flag weak ones
    Pass 5 — Re-cluster:      recompute clusters and attach a summary
    """
    db: Session = SessionLocal()
    report: Dict[str, Any] = {
        "merges": [],
        "orphans": [],
        "new_edges": [],
        "quality_flags": [],
        "cluster_updates": [],
        "summary": {},
    }

    try:
        # ── Pass 1: Deduplication ─────────────────────────────────────────────
        logger.info("[lint %s] Pass 1 — deduplication", report_id)
        similar_pairs = find_similar_concept_pairs(db, threshold=0.92, kb_id=kb_id)
        merged_ids: set = set()

        for id_a, id_b, similarity in similar_pairs:
            if id_a in merged_ids or id_b in merged_ids:
                continue

            ca = db.query(Concept).filter_by(id=id_a).first()
            cb = db.query(Concept).filter_by(id=id_b).first()
            if not ca or not cb:
                continue

            merge_result = await confirm_concept_merge(
                {"name": ca.name, "definition": ca.definition},
                {"name": cb.name, "definition": cb.definition},
                api_key=api_key,
            )

            if merge_result:
                # Redirect all edges that point to cb → point to ca instead
                db.query(Edge).filter_by(source_id=cb.id, kb_id=kb_id).update(
                    {"source_id": ca.id}, synchronize_session=False
                )
                db.query(Edge).filter_by(target_id=cb.id, kb_id=kb_id).update(
                    {"target_id": ca.id}, synchronize_session=False
                )

                # Remove self-loops that may have been created
                self_loops = (
                    db.query(Edge)
                    .filter(Edge.source_id == ca.id, Edge.target_id == ca.id)
                    .all()
                )
                for e in self_loops:
                    db.delete(e)

                # Redirect embedding
                db.query(NodeEmbedding).filter_by(node_id=cb.id).delete()

                ca.name = merge_result["canonical_name"]
                ca.definition = merge_result["merged_definition"]
                db.delete(cb)
                merged_ids.add(id_b)

                report["merges"].append(
                    {
                        "from": cb.name,
                        "to": merge_result["canonical_name"],
                        "similarity": round(similarity, 3),
                    }
                )

        db.commit()

        # ── Pass 2: Orphan detection ──────────────────────────────────────────
        logger.info("[lint %s] Pass 2 — orphan detection", report_id)
        G = build_graph(db, kb_id)

        for node_id in list(G.nodes):
            if G.degree(node_id) < 2 and G.nodes[node_id].get("type") == "concept":
                report["orphans"].append(
                    {
                        "id": node_id,
                        "name": G.nodes[node_id].get("name"),
                        "degree": G.degree(node_id),
                        "action": "flagged",
                    }
                )

        # ── Pass 3: Missing-edge discovery ───────────────────────────────────
        logger.info("[lint %s] Pass 3 — missing-edge discovery", report_id)
        G = build_graph(db, kb_id)  # rebuild after merges
        concept_id_set = {n for n in G.nodes if G.nodes[n].get("type") == "concept"}
        clusters = compute_clusters(G)

        for cluster in clusters:
            cluster_concepts = [n for n in cluster if n in concept_id_set]
            if len(cluster_concepts) < 2:
                continue

            checks_this_cluster = 0
            for i in range(len(cluster_concepts)):
                for j in range(i + 1, len(cluster_concepts)):
                    if checks_this_cluster >= 20:
                        break

                    id_a, id_b = cluster_concepts[i], cluster_concepts[j]
                    if G.has_edge(id_a, id_b):
                        continue

                    ca = db.query(Concept).filter_by(id=id_a).first()
                    cb = db.query(Concept).filter_by(id=id_b).first()
                    if not ca or not cb:
                        continue

                    relation = await check_concept_relation(
                        {"name": ca.name, "definition": ca.definition},
                        {"name": cb.name, "definition": cb.definition},
                        api_key=api_key,
                    )

                    if relation:
                        db.add(
                            Edge(
                                id=str(uuid.uuid4()),
                                kb_id=kb_id,
                                source_id=id_a,
                                source_type="concept",
                                target_id=id_b,
                                target_type="concept",
                                relation=relation["relation"],
                                label=relation["description"],
                            )
                        )
                        report["new_edges"].append(
                            {
                                "from": ca.name,
                                "to": cb.name,
                                "relation": relation["relation"],
                            }
                        )
                        checks_this_cluster += 1

                if checks_this_cluster >= 20:
                    break

        db.commit()

        # ── Pass 4: Quality check ─────────────────────────────────────────────
        logger.info("[lint %s] Pass 4 — quality check", report_id)
        all_concepts = db.query(Concept).filter_by(kb_id=kb_id).all()
        batch_size = 10

        for i in range(0, len(all_concepts), batch_size):
            batch = all_concepts[i : i + batch_size]
            batch_data = [{"name": c.name, "definition": c.definition} for c in batch]
            try:
                scores = await score_concept_quality(batch_data, api_key=api_key)
                for item in scores:
                    if item.get("score", 1.0) < 0.6:
                        concept = next(
                            (c for c in batch if c.name == item["name"]), None
                        )
                        if concept:
                            report["quality_flags"].append(
                                {
                                    "id": concept.id,
                                    "name": concept.name,
                                    "score": item["score"],
                                    "issue": item.get("issue"),
                                }
                            )
            except Exception as exc:
                logger.warning("Quality check batch failed: %s", exc)

        # ── Pass 5: Re-cluster ────────────────────────────────────────────────
        logger.info("[lint %s] Pass 5 — re-clustering", report_id)
        G = build_graph(db, kb_id)
        cluster_summary: List[Dict] = []
        try:
            for cluster_nodes in compute_clusters(G):
                sample = [
                    G.nodes[n]["name"]
                    for n in cluster_nodes
                    if G.nodes[n].get("type") == "concept"
                ][:5]
                cluster_summary.append(
                    {"size": len(cluster_nodes), "sample_concepts": sample}
                )
        except Exception as exc:
            logger.warning("Re-clustering failed: %s", exc)
        report["cluster_updates"] = cluster_summary

        # ── Finalise ──────────────────────────────────────────────────────────
        report["summary"] = {
            "merges": len(report["merges"]),
            "orphans_flagged": len(report["orphans"]),
            "new_edges": len(report["new_edges"]),
            "quality_flags": len(report["quality_flags"]),
            "clusters": len(cluster_summary),
        }

        lint_report = db.query(LintReport).filter_by(id=report_id).first()
        if lint_report:
            lint_report.status = "completed"
            lint_report.report = report
            db.commit()

        logger.info("[lint %s] Done — %s", report_id, report["summary"])

    except Exception as exc:
        logger.error("[lint %s] Failed: %s", report_id, exc, exc_info=True)
        try:
            lint_report = db.query(LintReport).filter_by(id=report_id).first()
            if lint_report:
                lint_report.status = "failed"
                lint_report.report = {"error": str(exc)}
                db.commit()
        except Exception:
            pass
    finally:
        db.close()
