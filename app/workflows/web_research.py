"""WebResearchWorkflow — web search and source curation for a trade thesis.

Runs 3-5 targeted web searches based on the thesis content, collects up to
10 candidate sources, ranks them (cited sources first, then by primary source
quality), and stores the top 3-5 as FurtherReading candidates with
agent-written one-sentence annotations.

Two Anthropic API calls are made per execution:
  1. Query generation — produces 3-5 targeted search queries from the thesis.
  2. Source annotation — selects the top 3-5 sources and annotates each.

Web searches use WebSearchClient (Anthropic web_search tool). All web sources
are cited with full URL and retrieval date.

NOTE (future improvement): _N_QUERIES_MAX, _MAX_CANDIDATES, and
_N_FURTHER_READING_MAX are currently module-level constants. They should be
moved into a workflow_configs PG table and loaded via a WorkflowSettings
Pydantic model (see future_improvements.md).
"""

from __future__ import annotations

import json
import logging
import time
import uuid

from app.integrations.anthropic_client import AnthropicClient
from app.integrations.web_search_client import (
    SearchResult,
    WebSearchClient,
    WebSearchClientError,
)
from app.models.workflow import FurtherReading
from app.workflows.base import (
    BaseWorkflow,
    Citation,
    CitationSourceType,
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)

# Maximum queries to execute. Capped at 3 to stay within Tier 1 token rate
# limits (30k input tokens/min) — each search drives a multi-turn tool_use
# loop that saturates the budget; fewer calls is the right fix over sleeping.
# TODO: move to workflow_configs table — see future_improvements.md
_N_QUERIES_MAX = 3

# Maximum candidate sources to collect before LLM selection.
# TODO: move to workflow_configs table — see future_improvements.md
_MAX_CANDIDATES = 10

# Maximum FurtherReading entries to store.
# TODO: move to workflow_configs table — see future_improvements.md
_N_FURTHER_READING_MAX = 5

# ---------------------------------------------------------------------------
# Source classification helpers
# ---------------------------------------------------------------------------

_FRED_URL_FRAGMENTS = frozenset(
    {
        "fred.stlouisfed.org",
        "api.stlouisfed.org",
        "research.stlouisfed.org",
    }
)

_ACADEMIC_URL_FRAGMENTS = frozenset(
    {
        "nber.org",
        "ssrn.com",
        ".edu",
        ".ac.uk",
        "brookings.edu",
        "piie.com",
        "iie.com",
    }
)

# Primary source domains for quality-tier ranking.
# Mirrors WebSearchClient._PRIMARY_DOMAIN_FRAGMENTS — duplicated here to avoid
# importing a private symbol from another module.
_PRIMARY_DOMAIN_FRAGMENTS = frozenset(
    {
        ".gov",
        "federalreserve.gov",
        "stlouisfed.org",
        "bls.gov",
        "bea.gov",
        "treasury.gov",
        "whitehouse.gov",
        "sec.gov",
        "cftc.gov",
        "imf.org",
        "worldbank.org",
        "bis.org",
        "ecb.europa.eu",
        "oecd.org",
        "un.org",
        "wto.org",
        "bankofengland.co.uk",
        "boj.or.jp",
        ".edu",
        ".ac.uk",
        "nber.org",
        "ssrn.com",
        "brookings.edu",
        "piie.com",
        "iie.com",
    }
)


def _classify_source_type(url: str) -> str:
    """Return 'FRED series', 'academic paper', or 'web' based on URL."""
    url_lower = url.lower()
    if any(f in url_lower for f in _FRED_URL_FRAGMENTS):
        return "FRED series"
    if any(f in url_lower for f in _ACADEMIC_URL_FRAGMENTS):
        return "academic paper"
    return "web"


def _is_primary_source(url: str) -> bool:
    """Return True if *url* belongs to a primary source domain."""
    url_lower = url.lower()
    return any(f in url_lower for f in _PRIMARY_DOMAIN_FRAGMENTS)


def _deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    """Deduplicate *results* by URL, preserving first-occurrence order."""
    seen: set[str] = set()
    deduped: list[SearchResult] = []
    for r in results:
        key = r.url.lower().rstrip("/")
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def _rank_sources(sources: list[dict]) -> list[dict]:
    """Rank sources: cited first, then by primary source quality within each group.

    Sorting priority (ascending):
      1. is_cited=True  + primary source
      2. is_cited=True  + secondary source
      3. is_cited=False + primary source
      4. is_cited=False + secondary source

    Returns a new list with a 'rank' key (1-based) added. Input dicts are not
    mutated.
    """

    def _key(s: dict) -> tuple[int, int]:
        cited = 0 if s.get("is_cited", False) else 1
        primary = 0 if _is_primary_source(s.get("url", "")) else 1
        return (cited, primary)

    sorted_sources = sorted(sources, key=_key)
    return [{**s, "rank": i} for i, s in enumerate(sorted_sources, 1)]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_QUERY_GENERATION_SYSTEM_PROMPT = """\
You are a macro research analyst preparing to research a trade thesis.

You will be given a trade thesis. Generate 3-5 targeted web search queries \
that would surface the most relevant and high-quality sources for evaluating \
this thesis.

Focus queries on:
- Recent news and data directly relevant to the thesis mechanism
- Central bank communications and policy signals
- Academic or institutional research on the thesis's core assumptions
- Analyst commentary or economic reports on the thesis instruments

Respond with a JSON object with exactly one key:
- "queries": a list of 3-5 search query strings

Respond only with the JSON object. No markdown fences, no preamble.\
"""

_ANNOTATION_SYSTEM_PROMPT = """\
You are a macro research analyst curating sources for a trade thesis research brief.

You will be given a trade thesis and a list of candidate web sources \
(title, URL, and snippet). Your task:
1. Select the top 3-5 most relevant and high-quality sources.
2. For each selected source, write a one-sentence annotation explaining \
why it is relevant to the thesis.
3. Mark each source as "cited" (true) if it directly supports or informs a \
core claim about the thesis mechanism, or false if it provides useful context \
but is not a primary reference.

Respond with a JSON object with exactly one key:
- "sources": a list of selected source objects, each containing:
  - "url": the exact URL from the candidate list
  - "annotation": one sentence explaining relevance to the thesis
  - "is_cited": true if directly cited in understanding the thesis, else false

Return between 3 and 5 sources. Respond only with the JSON object. \
No markdown fences, no preamble.\
"""


def _build_query_generation_message(thesis) -> str:
    direction = (
        thesis.direction.value
        if hasattr(thesis.direction, "value")
        else str(thesis.direction)
    )
    return (
        f"Thesis: {thesis.title}\n"
        f"Direction: {direction}\n"
        f"Time Horizon: {thesis.time_horizon}\n"
        f"Notes: {thesis.notes or '(none)'}"
    )


def _build_annotation_message(thesis, candidates: list[SearchResult]) -> str:
    direction = (
        thesis.direction.value
        if hasattr(thesis.direction, "value")
        else str(thesis.direction)
    )
    sources_lines = []
    for i, r in enumerate(candidates, 1):
        sources_lines.append(
            f"{i}. Title: {r.title}\n"
            f"   URL: {r.url}\n"
            f"   Snippet: {r.snippet[:200]}"
        )
    return (
        f"Thesis: {thesis.title}\n"
        f"Direction: {direction}\n"
        f"Time Horizon: {thesis.time_horizon}\n"
        f"Notes: {thesis.notes or '(none)'}\n\n"
        "Candidate sources:\n" + "\n".join(sources_lines)
    )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class WebResearchWorkflow(BaseWorkflow):
    """Run targeted web searches and curate FurtherReading sources.

    Generates 3-5 search queries via LLM, executes them via WebSearchClient,
    deduplicates and caps candidates at 10, then asks the LLM to select and
    annotate the top 3-5 sources. Ranked results are persisted to the
    further_reading table and returned in structured_output.

    Outputs (structured_output):
        search_queries (list[str]): queries that were executed.
        sources (list[dict]): ranked sources, each with title, url,
            source_type, annotation, is_cited, rank.
    """

    name = "WebResearchWorkflow"
    description = (
        "Runs 3-5 targeted web searches based on thesis content; collects up to "
        "10 candidate sources; annotates and stores the top 3-5 as FurtherReading."
    )
    required_inputs = ["title", "direction", "time_horizon"]
    model: str = "claude-sonnet-4-6"

    def __init__(
        self,
        web_search_client: WebSearchClient | None = None,
        anthropic_client: AnthropicClient | None = None,
    ) -> None:
        self._web_search = web_search_client
        self._anthropic = anthropic_client

    def execute(self, thesis, context: WorkflowContext) -> WorkflowResult:
        # --- Resolve clients ---
        if self._web_search is None or self._anthropic is None:
            from app.core.settings import get_settings

            settings = get_settings()
        else:
            settings = None  # type: ignore[assignment]

        anthropic = self._anthropic or AnthropicClient(
            api_key=settings.anthropic_api_key,
            db=context.db,
        )
        web_search = self._web_search or WebSearchClient(
            api_key=settings.anthropic_api_key,
            model=self.model,
        )

        citations: list[Citation] = []
        agent_inferences: list[str] = []

        # --- Step 1: Generate search queries ---
        query_response = anthropic.complete(
            messages=[
                {"role": "user", "content": _build_query_generation_message(thesis)}
            ],
            model=self.model,
            task_type="web_research_queries",
            workflow_run_id=context.current_workflow_run_id,
            thesis_id=thesis.id,
            pod_id=getattr(thesis, "pod_id", None),
            max_tokens=512,
            system=_QUERY_GENERATION_SYSTEM_PROMPT,
        )

        try:
            parsed_queries = json.loads(query_response.content)
            queries: list[str] = parsed_queries.get("queries", [])
        except (json.JSONDecodeError, AttributeError):
            logger.warning(
                "WebResearchWorkflow: query generation response was not valid JSON"
            )
            agent_inferences.append(
                "[Agent inference] Query generation returned malformed JSON; "
                "falling back to thesis title as search query."
            )
            queries = []

        queries = [q for q in queries if isinstance(q, str) and q.strip()]
        queries = queries[:_N_QUERIES_MAX]
        if not queries:
            queries = [thesis.title]

        # --- Step 2: Execute searches and collect candidates ---
        all_results: list[SearchResult] = []
        for query in queries:
            try:
                results = web_search.search(query)
                all_results.extend(results)
            except WebSearchClientError as exc:
                logger.warning(
                    "WebResearchWorkflow: search failed for query %r: %s", query, exc
                )
                agent_inferences.append(
                    f"[Agent inference] Web search failed for query '{query}' — "
                    "results from this query are unavailable."
                )

        candidates = _deduplicate(all_results)[:_MAX_CANDIDATES]

        if not candidates:
            agent_inferences.append(
                "[Agent inference] No web search results were retrieved — "
                "FurtherReading could not be populated."
            )
            return WorkflowResult(
                workflow_name=self.name,
                status=WorkflowStatus.PARTIAL,
                structured_output={"search_queries": queries, "sources": []},
                citations=[],
                agent_inferences=agent_inferences,
                raw_output=query_response.content,
            )

        # Web search calls consume most of the Tier 1 token rate limit (30k
        # input tokens/min). Sleep 65s to let the window reset before the
        # annotation call, which would otherwise fail with 429 immediately.
        logger.debug(
            "WebResearchWorkflow: sleeping 65s to reset rate limit window"
            " before annotation"
        )
        time.sleep(65)

        # --- Step 3: LLM annotation and selection ---
        annotation_response = anthropic.complete(
            messages=[
                {
                    "role": "user",
                    "content": _build_annotation_message(thesis, candidates),
                }
            ],
            model=self.model,
            task_type="web_research_annotation",
            workflow_run_id=context.current_workflow_run_id,
            thesis_id=thesis.id,
            pod_id=getattr(thesis, "pod_id", None),
            max_tokens=1024,
            system=_ANNOTATION_SYSTEM_PROMPT,
        )

        try:
            parsed_annotations = json.loads(annotation_response.content)
            llm_sources: list[dict] = parsed_annotations.get("sources", [])
        except (json.JSONDecodeError, AttributeError):
            logger.warning(
                "WebResearchWorkflow: annotation response was not valid JSON"
            )
            agent_inferences.append(
                "[Agent inference] Source annotation returned malformed JSON; "
                "no FurtherReading entries were generated."
            )
            llm_sources = []

        llm_sources = llm_sources[:_N_FURTHER_READING_MAX]

        # --- Step 4: Enrich LLM selection with candidate metadata ---
        candidate_lookup: dict[str, SearchResult] = {r.url: r for r in candidates}
        enriched: list[dict] = []
        for source_dict in llm_sources:
            url: str = source_dict.get("url", "")
            candidate = candidate_lookup.get(url)
            if candidate is None:
                agent_inferences.append(
                    f"[Agent inference] LLM returned URL not in candidate set: "
                    f"{url!r} — skipped."
                )
                continue
            enriched.append(
                {
                    "url": url,
                    "annotation": str(source_dict.get("annotation", "")),
                    "is_cited": bool(source_dict.get("is_cited", False)),
                    "title": candidate.title or url,
                    "source_type": _classify_source_type(url),
                    "_retrieval_date": candidate.retrieval_date.date(),
                }
            )

        # --- Step 5: Apply deterministic ranking ---
        ranked = _rank_sources(enriched)

        # --- Step 6: Build output and persist FurtherReading rows ---
        sources_out: list[dict] = []
        further_reading_rows: list[FurtherReading] = []

        for s in ranked:
            sources_out.append(
                {
                    "title": s["title"],
                    "url": s["url"],
                    "source_type": s["source_type"],
                    "annotation": s["annotation"],
                    "is_cited": s["is_cited"],
                    "rank": s["rank"],
                }
            )
            citations.append(
                Citation(
                    source_type=CitationSourceType.WEB,
                    label=s["url"],
                    url=s["url"],
                    retrieval_date=s["_retrieval_date"],
                )
            )
            further_reading_rows.append(
                FurtherReading(
                    id=uuid.uuid4(),
                    thesis_id=thesis.id,
                    title=s["title"],
                    url=s["url"],
                    source_type=s["source_type"],
                    annotation=s["annotation"],
                    rank=s["rank"],
                    is_cited=s["is_cited"],
                )
            )

        if context.db is not None and further_reading_rows:
            try:
                for row in further_reading_rows:
                    context.db.add(row)
                context.db.commit()
            except Exception:
                logger.exception(
                    "WebResearchWorkflow: failed to persist FurtherReading rows"
                )

        status = WorkflowStatus.COMPLETED if sources_out else WorkflowStatus.PARTIAL

        return WorkflowResult(
            workflow_name=self.name,
            status=status,
            structured_output={
                "search_queries": queries,
                "sources": sources_out,
            },
            citations=citations,
            agent_inferences=agent_inferences,
            raw_output=annotation_response.content,
        )
