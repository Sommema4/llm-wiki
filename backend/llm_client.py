import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from config import get_settings

settings = get_settings()

# Limit concurrent outgoing LLM requests so we never exceed provider
# parallel-request quotas (e.g. MetaCentrum allows 4; we stay at 3).
_llm_semaphore = asyncio.Semaphore(3)


def _get_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> AsyncOpenAI:
    resolved_key = api_key or settings.openrouter_api_key
    if not resolved_key:
        raise ValueError(
            "No API key available. Set your key via the API Key button in the app, "
            "or ask the server administrator to set OPENROUTER_API_KEY in .env."
        )
    return AsyncOpenAI(
        api_key=resolved_key,
        base_url=base_url or settings.openrouter_base_url,
        max_retries=6,  # exponential backoff: ~1s, 2s, 4s, 8s, 16s, 32s
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "LLM Wiki",
        },
    )


# ── Pydantic response models ──────────────────────────────────────────────────

class _ConceptExtract(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    concept_type: str = "METHOD"  # PROBLEM|CONTRIBUTION|METHOD|MODEL|DATASET|RESULT|METRIC|LIMITATION
    summary: str = ""             # L0: one sentence — what it is
    explanation: str = ""         # L1: 2-3 sentences — how it works and why it matters
    definition: str = ""          # L2: technical detail, equations, paper-specific usage


class _RelationToExisting(BaseModel):
    model_config = ConfigDict(extra="ignore")
    concept: str
    relation: str
    description: str = ""


class _PaperExtract(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str
    authors: List[str] = []
    year: Optional[int] = None
    venue: Optional[str] = None
    summary: str = ""
    contributions: List[str] = []
    key_findings: List[str] = []
    concepts_used: List[_ConceptExtract] = []
    relations_to_existing: List[_RelationToExisting] = []

    @field_validator("year", mode="before")
    @classmethod
    def _coerce_year(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    @field_validator("authors", mode="before")
    @classmethod
    def _coerce_authors(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [v]
        return v or []


class _UpdatedConcept(BaseModel):
    model_config = ConfigDict(extra="ignore")
    summary: str
    explanation: str
    definition: str


class _MergeDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")
    should_merge: bool
    canonical_name: Optional[str] = None
    merged_definition: Optional[str] = None


class _RelationCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    related: bool
    relation: Optional[str] = None
    description: Optional[str] = None


class _ConceptScore(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    score: float
    issue: Optional[str] = None


class _ScoreBatch(BaseModel):
    model_config = ConfigDict(extra="ignore")
    scores: List[_ConceptScore] = []


# ─────────────────────────────────────────────────────────────────────────────


def _parse_json(content: str) -> Dict[str, Any]:
    """Parse JSON from an LLM response, handling markdown code fences."""
    # Direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Strip ```json ... ``` fences
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Find first {...} blob
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {content[:300]}")


def _validate(content: str, model: type[BaseModel]) -> BaseModel:
    """Parse JSON from an LLM response and validate it against a Pydantic model."""
    try:
        return model.model_validate(_parse_json(content))
    except ValidationError as exc:
        raise ValueError(
            f"LLM response did not match expected schema ({model.__name__}): {exc}"
        ) from exc


async def extract_paper_metadata(
    text: str,
    existing_concepts: List[str],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Single LLM call that extracts all structured metadata from a paper."""
    client = _get_client(api_key, base_url)
    concept_list = ", ".join(existing_concepts[:500]) if existing_concepts else "none yet"

    prompt = f"""You are analyzing a scientific paper to build a semantic knowledge graph. Extract typed concept nodes that have standalone semantic value.

Paper text (may be truncated):
{text[:400000]}

Existing concepts already in the knowledge base: {concept_list}

Return a JSON object with EXACTLY this structure:
{{
  "title": "full paper title",
  "authors": ["Author Name"],
  "year": 2024,
  "venue": "conference or journal name, or null",
  "summary": "4-6 sentence summary covering the problem, approach, key mechanism, and results",
  "contributions": ["specific contribution 1", "specific contribution 2"],
  "key_findings": ["precise quantitative or qualitative finding 1", "finding 2"],
  "concepts_used": [
    {{
      "name": "concept name in lowercase",
      "concept_type": "METHOD",
      "summary": "one sentence — what this concept is",
      "explanation": "2-3 sentences — how it works and why it matters",
      "definition": "technical detail: formal definition, mathematical formulation where applicable, how this paper specifically uses or extends it"
    }}
  ],
  "relations_to_existing": [
    {{"concept": "existing concept name", "relation": "extends|uses|contradicts|improves|applies", "description": "how this paper relates"}}
  ]
}}

Concept types and their meaning:
- PROBLEM: the research problem or challenge this paper addresses (1-3 total)
- CONTRIBUTION: novel ideas, techniques, or findings specific to this paper (1-5 total)
- METHOD: algorithms, procedures, or techniques central to the paper's approach (1-8 total)
- MODEL: specific model architectures, variants, or systems (1-4 total)
- DATASET: datasets used for training or evaluation (1-6 total)
- RESULT: key quantitative or qualitative findings worth referencing independently (1-5 total)
- METRIC: evaluation metrics — only if non-standard or specially defined (1-4 total)
- LIMITATION: stated limitations, failure modes, or open questions (1-3 total)

Anti-fragmentation rules — DO NOT extract a concept if:
- It is standard domain knowledge any expert would know (e.g. "gradient descent", "relu activation")
- It is mentioned only once without meaningful discussion
- It is a minor implementation detail (specific hyperparameter value, line of code)
- It is an elaboration of another concept already in your list
- It cannot be independently referenced or reused across different papers

A concept earns its place only if a researcher would plausibly search for it by name.
- relations_to_existing: only for concepts already in the knowledge base list above
- Return ONLY the JSON object, no other text"""

    async with _llm_semaphore:
        response = await client.chat.completions.create(
            model=model or settings.default_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30000,
            temperature=0.1,
        )
    return _validate(response.choices[0].message.content, _PaperExtract).model_dump()


async def enrich_concept_definition(
    concept_name: str,
    current_definition: str,
    paper_title: str,
    context_excerpt: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, str]:
    """Update a concept's multi-resolution descriptions with insights from a new paper."""
    client = _get_client(api_key, base_url)

    prompt = f"""Update the descriptions of a scientific concept using new information from a paper.

Concept: "{concept_name}"
Current definition: {current_definition}

New context from "{paper_title}":
{context_excerpt[:4000]}

Return JSON with three levels of description:
{{
  "summary": "one sentence — what this concept is",
  "explanation": "2-3 sentences — how it works and why it matters",
  "definition": "technical detail: formal definition, mathematical formulation where applicable, and how this paper uses or extends the concept"
}}"""

    async with _llm_semaphore:
        response = await client.chat.completions.create(
            model=model or settings.default_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1,
        )
    result = _validate(response.choices[0].message.content, _UpdatedConcept)
    return {"summary": result.summary, "explanation": result.explanation, "definition": result.definition}


async def confirm_concept_merge(
    concept_a: Dict[str, str],
    concept_b: Dict[str, str],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Check if two similar-sounding concepts should be merged into one."""
    client = _get_client(api_key, base_url)

    prompt = f"""Should these two concept entries be merged into one?

Concept A: "{concept_a['name']}" — {concept_a['definition']}
Concept B: "{concept_b['name']}" — {concept_b['definition']}

Return JSON:
{{
  "should_merge": true or false,
  "canonical_name": "best name to use (only if merging)",
  "merged_definition": "combined definition (only if merging)"
}}"""

    async with _llm_semaphore:
        response = await client.chat.completions.create(
            model=model or settings.default_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.1,
        )
    decision = _validate(response.choices[0].message.content, _MergeDecision)
    if decision.should_merge:
        return {
            "canonical_name": decision.canonical_name or concept_a["name"],
            "merged_definition": decision.merged_definition or concept_a["definition"],
        }
    return None


async def check_concept_relation(
    concept_a: Dict[str, str],
    concept_b: Dict[str, str],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Check if two unconnected concepts have a meaningful relation."""
    client = _get_client(api_key, base_url)

    prompt = f"""Are these two scientific concepts meaningfully related?

Concept A: "{concept_a['name']}" — {concept_a['definition']}
Concept B: "{concept_b['name']}" — {concept_b['definition']}

Return JSON:
{{
  "related": true or false,
  "relation": "extends|uses|part_of|contrasts_with|related_to (only if related)",
  "description": "one sentence explanation (only if related)"
}}"""

    async with _llm_semaphore:
        response = await client.chat.completions.create(
            model=model or settings.default_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.1,
        )
    check = _validate(response.choices[0].message.content, _RelationCheck)
    if check.related:
        return {
            "relation": check.relation or "related_to",
            "description": check.description or "",
        }
    return None


async def score_concept_quality(
    concepts: List[Dict[str, str]],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Score a batch of concept definitions for quality (completeness, clarity)."""
    client = _get_client(api_key, base_url)
    concept_list = "\n".join(
        [f'- "{c["name"]}": {c["definition"]}' for c in concepts]
    )

    prompt = f"""Rate the quality of these concept definitions in a scientific knowledge base.

{concept_list}

For each concept, provide a quality score (0.0-1.0) and describe any issues.
Score < 0.6 means the definition is too vague, too short, or misleading.

Return JSON:
{{
  "scores": [
    {{"name": "concept name", "score": 0.85, "issue": null}},
    {{"name": "weak concept", "score": 0.4, "issue": "definition is too vague and lacks technical detail"}}
  ]
}}"""

    async with _llm_semaphore:
        response = await client.chat.completions.create(
            model=model or settings.default_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1,
        )
    batch = _validate(response.choices[0].message.content, _ScoreBatch)
    return [s.model_dump() for s in batch.scores]


async def answer_question(
    question: str,
    context_nodes: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Generate a grounded answer from retrieved wiki nodes, with optional conversation history."""
    client = _get_client(api_key, base_url)

    context = "\n\n".join(
        f"[{node['type'].upper()}: {node['name']}]\n{node['content']}"
        for node in context_nodes
    )

    system_prompt = f"""You are a research assistant with access to a personal scientific paper knowledge base.
Answer questions using ONLY the context provided below. Cite the specific papers or concepts you draw from.
If the available information is insufficient, say so explicitly.

Knowledge base context:
{context}"""

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Inject conversation history (last N turns already trimmed by caller)
    for turn in (history or []):
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Current question
    messages.append({"role": "user", "content": question})

    async with _llm_semaphore:
        response = await client.chat.completions.create(
            model=model or settings.chat_model,
            messages=messages,
            temperature=0.3,
        )
    return response.choices[0].message.content
