FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy source
COPY backend/ backend/
COPY frontend/ frontend/

WORKDIR /app/backend

# Data lives in a mounted volume so it persists across restarts
ENV DATABASE_URL=sqlite:////data/llm_wiki.db
ENV UPLOAD_DIR=/data/uploads
# fastembed caches downloaded embedding models here
ENV FASTEMBED_CACHE_PATH=/data/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
