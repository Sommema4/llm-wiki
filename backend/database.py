import datetime
import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ── Auth / multi-tenancy ───────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    openrouter_api_key = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


# ── Wiki content ───────────────────────────────────────────────────────────────

class Paper(Base):
    __tablename__ = "papers"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    kb_id = Column(String, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String, nullable=True)
    authors = Column(JSON, default=list)
    year = Column(Integer, nullable=True)
    venue = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    contributions = Column(JSON, default=list)
    key_findings = Column(JSON, default=list)
    file_path = Column(String)
    status = Column(String, default="processing")  # processing | ready | failed
    error_message = Column(Text, nullable=True)
    pages_extracted = Column(Integer, nullable=True)
    pages_total = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )


class Concept(Base):
    __tablename__ = "concepts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    kb_id = Column(String, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String, nullable=False)
    definition = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )


class Edge(Base):
    __tablename__ = "edges"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    kb_id = Column(String, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True)
    source_id = Column(String, nullable=False)
    source_type = Column(String, nullable=False)  # paper | concept
    target_id = Column(String, nullable=False)
    target_type = Column(String, nullable=False)  # paper | concept
    relation = Column(String)  # uses | extends | contradicts | improves | related_to
    label = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class NodeEmbedding(Base):
    __tablename__ = "node_embeddings"

    node_id = Column(String, primary_key=True)
    node_type = Column(String)
    kb_id = Column(String, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True)
    embedding = Column(JSON)        # list[float]
    content_hash = Column(String)   # md5 of the text — skip re-embed if unchanged


class LintReport(Base):
    __tablename__ = "lint_reports"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    kb_id = Column(String, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String, default="running")  # running | completed | failed
    report = Column(JSON, nullable=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    # Migrate: add columns introduced after initial schema creation
    with engine.connect() as conn:
        papers_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(papers)")).fetchall()}
        for col_name, col_def in [("pages_extracted", "INTEGER"), ("pages_total", "INTEGER")]:
            if col_name not in papers_cols:
                conn.execute(text(f"ALTER TABLE papers ADD COLUMN {col_name} {col_def}"))

        users_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(users)")).fetchall()}
        if "openrouter_api_key" not in users_cols:
            conn.execute(text("ALTER TABLE users ADD COLUMN openrouter_api_key TEXT"))

        conn.commit()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
