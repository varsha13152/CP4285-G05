# Stage 1 — LLM Intent Understanding

Owned by **Varsha**. Parses natural-language search queries into structured
`QueryIntent` using SOCLaaS (NUS OpenAI-compatible LLM API).

## Output

Produces a `QueryIntent` dataclass (defined in `recsys.contracts.models`):

```python
from recsys.contracts import QueryIntent, QueryIntentType, ContentStyle

# QueryIntent has:
#   raw_query: str                     — the original search text
#   topics: list[CategoryMatch]        — matched KuaiRec categories (ID, Chinese/English name, confidence)
#   style: ContentStyle                — desired tone (FUNNY, SERIOUS, RELAXING, etc.)
#   duration: DurationPreference       — SHORT / MEDIUM / LONG / ANY
#   intent_type: QueryIntentType       — ENTERTAINMENT, LEARNING, INSPIRATION, SOCIAL, SPECIFIC_SEARCH, EXPLORATION
#   negative_constraints: list[str]    — things the user explicitly wants to exclude
#   semantic_summary: str              — one-line description for retrieval
#   query_embedding: list[float]|None  — optional dense vector (pluggable embedder)
#   confidence: float                  — 0.0-1.0
```

## Usage

```python
from recsys.intent import QueryParser
from recsys.routing import QueryRouter

parser = QueryParser()                                  # reads .env for SOCLaaS creds
intent = parser.parse("fun cooking videos for students")

print(intent.topics)       # [CategoryMatch(12, "美食", "Food", 0.9)]
print(intent.style)        # ContentStyle.FUNNY
print(intent.intent_type)  # QueryIntentType.ENTERTAINMENT

router = QueryRouter()
request = router.route(intent, user_id=42)
# → QueryRetrievalRequest with strategies, category_filter, duration_range
```

## CLI

```bash
python scripts/run_pipeline.py --query "fun cooking videos for students"
python scripts/run_pipeline.py --query "cute cat videos" --user_id 14
```

## Files

- `query_parser.py` — `QueryParser` class: sends query to SOCLaaS LLM, parses JSON response into `QueryIntent`
- `embeddings.py` — Pluggable embedder protocol (`QueryEmbedder`) + `NullEmbedder` default

## Evaluation

```bash
python scripts/evaluate_stage1.py --verbose
```

Runs 30 synthetic queries from `tests/fixtures/synthetic_queries.py` and reports
topic-match accuracy (Jaccard), style/duration/intent accuracy, and latency.

## Dependencies

- `openai>=1.0` — SOCLaaS is OpenAI-compatible
- `python-dotenv` — loads `.env` with `SOCLAAS_BASE_URL`, `SOCLAAS_API_KEY`, `SOCLAAS_MODEL`
