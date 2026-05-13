# LLM Wiki

A personal knowledge base for scientific papers. Upload PDFs, let an LLM extract structured metadata, concepts, and relations, then explore everything as an interactive knowledge graph and ask questions across your whole library using semantic search.

Inspired by Andrej Karpathy's idea of building tools to help researchers navigate the ever-growing landscape of literature — see his [arxiv-sanity](https://github.com/karpathy/arxiv-sanity-preserver) and his talks on the importance of understanding foundational concepts deeply.

---

## What it does

- **Ingest PDFs** — upload papers and the app extracts text, then calls an LLM to pull out title, authors, year, venue, summary, key findings, and contributions
- **Concept graph** — the LLM identifies concepts in each paper, builds definitions, and links them to concepts from other papers, forming a growing knowledge graph
- **Graph visualisation** — explore papers and concepts as an interactive graph with Louvain community clustering
- **Semantic search** — find papers and concepts by meaning using local ONNX embeddings ([BGE-small](https://huggingface.co/BAAI/bge-small-en-v1.5))
- **Q&A** — ask natural-language questions across your knowledge base; the app retrieves relevant nodes and answers using an LLM with conversation history
- **Linting** — detect duplicate or near-duplicate concepts, score concept quality, and suggest merges
- **Multi-user / multi-KB** — each user has their own knowledge bases; auth via JWT

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| Database | [SQLite](https://www.sqlite.org/) via [SQLAlchemy](https://www.sqlalchemy.org/) |
| LLM | [OpenRouter](https://openrouter.ai/) (OpenAI-compatible API) — default models: Gemini 2.0 Flash + Claude Sonnet |
| Embeddings | [FastEmbed](https://github.com/qdrant/fastembed) — local ONNX, no GPU required |
| Graph algorithms | [NetworkX](https://networkx.org/) with Louvain community detection |
| PDF extraction | [PyMuPDF](https://pymupdf.readthedocs.io/) |
| Auth | JWT ([python-jose](https://github.com/mpdavis/python-jose)) + bcrypt ([passlib](https://passlib.readthedocs.io/)) |
| Frontend | Vanilla HTML/JS (single `index.html`, served by FastAPI) |

---

## Prerequisites

- An [OpenRouter](https://openrouter.ai/) API key
- **For local run:** Python 3.11 or 3.12
- **For Docker:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)

---

## Running locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/llm_wiki.git
cd llm_wiki
```

### 2. Create and activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv backend/.venv
source backend/.venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Create your `.env` file

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

### 5. Start the server

```bash
cd backend
uvicorn main:app --reload
```

The app will be available at **http://localhost:8000** — open that in your browser to use the app.

> The SQLite database (`llm_wiki.db`) and uploaded files (`uploads/`) are created automatically in the `backend/` directory on first run.

---

## Running with Docker

### 1. Install Docker Desktop

Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/). It includes everything needed — no separate WSL setup required (it configures WSL 2 automatically on Windows).

After installing, launch Docker Desktop and wait until the status shows **"Engine running"**.

### 2. Clone the repo

```bash
git clone https://github.com/your-username/llm_wiki.git
cd llm_wiki
```

### 3. Create your `.env` file

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

### 4. Build and start

```bash
docker compose up --build
```

The app will be available at **http://localhost:8000** — open that in your browser to use the app.

The `--build` flag is only needed the first time (or after changing `requirements.txt`). For subsequent starts just run:

```bash
docker compose up
```

### 5. Stop

```bash
# Stop containers (data is preserved)
docker compose down

# Stop AND wipe all data (database, uploads, model cache)
docker compose down -v
```

> All data is stored in a Docker named volume (`app_data`), so it persists across restarts. The embedding model is also cached in the volume so it is only downloaded once.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | API base URL |
| `DEFAULT_MODEL` | `google/gemini-2.0-flash-001` | Model used for metadata extraction |
| `CHAT_MODEL` | `anthropic/claude-sonnet-4-5` | Model used for Q&A |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `SECRET_KEY` | random default | JWT signing secret — set a strong value in production |

---

## API docs

FastAPI generates interactive docs automatically:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Inspiration & further reading

- Andrej Karpathy — [arxiv-sanity-preserver](https://github.com/karpathy/arxiv-sanity-preserver) — the original inspiration for building tools to navigate ML papers
- Andrej Karpathy — [arxiv-sanity-lite](https://github.com/karpathy/arxiv-sanity-lite) — the leaner successor
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) — the local embedding model used for semantic search
- [OpenRouter](https://openrouter.ai/) — unified API for accessing many LLMs
- [FastEmbed](https://github.com/qdrant/fastembed) — lightweight ONNX-based embeddings, no PyTorch required
- [NetworkX — Louvain communities](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html)
