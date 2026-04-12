"""Tests for WebResearchWorkflow."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

from app.integrations.anthropic_client import AnthropicResponse
from app.integrations.web_search_client import SearchResult, WebSearchClientError
from app.workflows.base import CitationSourceType, WorkflowContext, WorkflowStatus
from app.workflows.web_research import (
    WebResearchWorkflow,
    _classify_source_type,
    _deduplicate,
    _is_primary_source,
    _rank_sources,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RETRIEVAL_DT = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


def _make_thesis(title: str = "Yield Curve Steepener") -> MagicMock:
    thesis = MagicMock()
    thesis.id = uuid.uuid4()
    thesis.pod_id = uuid.uuid4()
    thesis.title = title
    thesis.direction = MagicMock()
    thesis.direction.value = "long"
    thesis.time_horizon = "6 months"
    thesis.notes = "Long TLT as yield curve steepens."
    return thesis


def _make_search_result(
    url: str = "https://example.com/article",
    title: str = "Example Article",
    snippet: str = "This is a relevant snippet.",
    retrieval_date: datetime = _RETRIEVAL_DT,
) -> SearchResult:
    return SearchResult(
        url=url, title=title, snippet=snippet, retrieval_date=retrieval_date
    )


def _make_search_results(
    n: int = 3, base_url: str = "https://example.com/article"
) -> list[SearchResult]:
    return [
        _make_search_result(
            url=f"{base_url}-{i}",
            title=f"Article {i}",
            snippet=f"Snippet {i}",
        )
        for i in range(n)
    ]


def _make_query_response(queries: list[str] | None = None) -> AnthropicResponse:
    qs = queries or [
        "TLT yield curve research",
        "Fed policy duration",
        "TLT ETF analysis",
    ]
    return AnthropicResponse(
        content=json.dumps({"queries": qs}),
        model="claude-sonnet-4-6",
        input_tokens=100,
        output_tokens=80,
        stop_reason="end_turn",
    )


def _make_annotation_response(sources: list[dict] | None = None) -> AnthropicResponse:
    default_sources = [
        {
            "url": "https://example.com/article-0",
            "annotation": "Directly relevant to yield curve thesis.",
            "is_cited": True,
        },
        {
            "url": "https://example.com/article-1",
            "annotation": "Provides useful macro context.",
            "is_cited": False,
        },
        {
            "url": "https://example.com/article-2",
            "annotation": "Background on TLT instrument.",
            "is_cited": False,
        },
    ]
    return AnthropicResponse(
        content=json.dumps({"sources": sources or default_sources}),
        model="claude-sonnet-4-6",
        input_tokens=300,
        output_tokens=200,
        stop_reason="end_turn",
    )


def _make_anthropic_client(
    query_response: AnthropicResponse | None = None,
    annotation_response: AnthropicResponse | None = None,
) -> MagicMock:
    mock = MagicMock()
    mock.complete.side_effect = [
        query_response or _make_query_response(),
        annotation_response or _make_annotation_response(),
    ]
    return mock


def _make_web_search_client(results: list[SearchResult] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.search.return_value = (
        results if results is not None else _make_search_results()
    )
    return mock


def _make_context() -> WorkflowContext:
    ctx = WorkflowContext(thesis=_make_thesis(), db=MagicMock())
    return ctx


# ---------------------------------------------------------------------------
# _classify_source_type
# ---------------------------------------------------------------------------


class TestClassifySourceType:
    def test_fred_url_classified_as_fred_series(self):
        assert (
            _classify_source_type("https://fred.stlouisfed.org/series/T10Y2Y")
            == "FRED series"
        )

    def test_api_stlouisfed_classified_as_fred_series(self):
        assert (
            _classify_source_type("https://api.stlouisfed.org/fred/series")
            == "FRED series"
        )

    def test_nber_classified_as_academic_paper(self):
        assert (
            _classify_source_type("https://www.nber.org/papers/w12345")
            == "academic paper"
        )

    def test_ssrn_classified_as_academic_paper(self):
        assert (
            _classify_source_type("https://ssrn.com/abstract=12345") == "academic paper"
        )

    def test_edu_domain_classified_as_academic_paper(self):
        assert (
            _classify_source_type("https://economics.mit.edu/research")
            == "academic paper"
        )

    def test_generic_news_classified_as_web(self):
        assert _classify_source_type("https://www.reuters.com/markets/story") == "web"

    def test_bloomberg_classified_as_web(self):
        assert (
            _classify_source_type("https://www.bloomberg.com/news/articles/2024")
            == "web"
        )

    def test_classification_is_case_insensitive(self):
        assert (
            _classify_source_type("HTTPS://FRED.STLOUISFED.ORG/SERIES/T10Y2Y")
            == "FRED series"
        )


# ---------------------------------------------------------------------------
# _is_primary_source
# ---------------------------------------------------------------------------


class TestIsPrimarySource:
    def test_fed_gov_is_primary(self):
        assert _is_primary_source("https://www.federalreserve.gov/monetarypolicy")

    def test_imf_is_primary(self):
        assert _is_primary_source("https://www.imf.org/en/Publications")

    def test_ecb_is_primary(self):
        assert _is_primary_source("https://www.ecb.europa.eu/pub")

    def test_nber_is_primary(self):
        assert _is_primary_source("https://www.nber.org/papers/w12345")

    def test_news_site_is_not_primary(self):
        assert not _is_primary_source("https://www.cnbc.com/markets")

    def test_blog_is_not_primary(self):
        assert not _is_primary_source("https://www.marketwatch.com/story")


# ---------------------------------------------------------------------------
# _deduplicate
# ---------------------------------------------------------------------------


class TestDeduplicate:
    def test_removes_exact_duplicate_urls(self):
        results = [
            _make_search_result(url="https://example.com/a"),
            _make_search_result(url="https://example.com/a"),
            _make_search_result(url="https://example.com/b"),
        ]
        deduped = _deduplicate(results)
        assert len(deduped) == 2

    def test_preserves_order_of_first_occurrence(self):
        results = [
            _make_search_result(url="https://example.com/b"),
            _make_search_result(url="https://example.com/a"),
            _make_search_result(url="https://example.com/b"),
        ]
        deduped = _deduplicate(results)
        assert deduped[0].url == "https://example.com/b"
        assert deduped[1].url == "https://example.com/a"

    def test_deduplicates_urls_with_trailing_slash(self):
        results = [
            _make_search_result(url="https://example.com/a/"),
            _make_search_result(url="https://example.com/a"),
        ]
        deduped = _deduplicate(results)
        assert len(deduped) == 1

    def test_deduplication_is_case_insensitive(self):
        results = [
            _make_search_result(url="https://EXAMPLE.COM/a"),
            _make_search_result(url="https://example.com/a"),
        ]
        deduped = _deduplicate(results)
        assert len(deduped) == 1

    def test_empty_list_returns_empty(self):
        assert _deduplicate([]) == []

    def test_unique_urls_all_preserved(self):
        results = _make_search_results(n=5)
        assert len(_deduplicate(results)) == 5


# ---------------------------------------------------------------------------
# _rank_sources
# ---------------------------------------------------------------------------


class TestRankSources:
    def _cited_primary(self, url: str = "https://www.federalreserve.gov/x") -> dict:
        return {"url": url, "is_cited": True, "annotation": "cited primary"}

    def _cited_secondary(self, url: str = "https://www.cnbc.com/x") -> dict:
        return {"url": url, "is_cited": True, "annotation": "cited secondary"}

    def _uncited_primary(self, url: str = "https://www.imf.org/x") -> dict:
        return {"url": url, "is_cited": False, "annotation": "uncited primary"}

    def _uncited_secondary(self, url: str = "https://www.reuters.com/x") -> dict:
        return {"url": url, "is_cited": False, "annotation": "uncited secondary"}

    def test_cited_primary_ranked_first(self):
        sources = [self._uncited_secondary(), self._cited_primary()]
        ranked = _rank_sources(sources)
        assert ranked[0]["annotation"] == "cited primary"

    def test_cited_before_uncited(self):
        sources = [self._uncited_secondary(), self._cited_secondary()]
        ranked = _rank_sources(sources)
        assert ranked[0]["is_cited"] is True
        assert ranked[1]["is_cited"] is False

    def test_primary_before_secondary_within_cited(self):
        sources = [self._cited_secondary(), self._cited_primary()]
        ranked = _rank_sources(sources)
        assert "federalreserve.gov" in ranked[0]["url"]

    def test_primary_before_secondary_within_uncited(self):
        sources = [self._uncited_secondary(), self._uncited_primary()]
        ranked = _rank_sources(sources)
        assert "imf.org" in ranked[0]["url"]

    def test_full_ordering(self):
        sources = [
            self._uncited_secondary(),
            self._uncited_primary(),
            self._cited_secondary(),
            self._cited_primary(),
        ]
        ranked = _rank_sources(sources)
        assert ranked[0]["annotation"] == "cited primary"
        assert ranked[1]["annotation"] == "cited secondary"
        assert ranked[2]["annotation"] == "uncited primary"
        assert ranked[3]["annotation"] == "uncited secondary"

    def test_rank_values_are_one_based_sequential(self):
        sources = [self._cited_primary(), self._uncited_secondary()]
        ranked = _rank_sources(sources)
        assert [s["rank"] for s in ranked] == [1, 2]

    def test_input_dicts_not_mutated(self):
        source = self._cited_primary()
        _rank_sources([source])
        assert "rank" not in source

    def test_empty_list_returns_empty(self):
        assert _rank_sources([]) == []


# ---------------------------------------------------------------------------
# WebResearchWorkflow — search execution
# ---------------------------------------------------------------------------


class TestWebResearchWorkflowSearchExecution:
    def test_search_called_for_each_query(self):
        web_search = _make_web_search_client()
        anthropic = _make_anthropic_client(
            query_response=_make_query_response(["query A", "query B", "query C"])
        )
        workflow = WebResearchWorkflow(
            web_search_client=web_search, anthropic_client=anthropic
        )
        workflow.execute(_make_thesis(), _make_context())
        assert web_search.search.call_count == 3

    def test_search_called_with_generated_queries(self):
        queries = ["TLT yield curve", "Fed rate path 2024"]
        web_search = _make_web_search_client()
        anthropic = _make_anthropic_client(query_response=_make_query_response(queries))
        workflow = WebResearchWorkflow(
            web_search_client=web_search, anthropic_client=anthropic
        )
        workflow.execute(_make_thesis(), _make_context())
        called_queries = [c.args[0] for c in web_search.search.call_args_list]
        assert "TLT yield curve" in called_queries
        assert "Fed rate path 2024" in called_queries

    def test_fallback_to_thesis_title_when_query_generation_fails(self):
        bad_response = AnthropicResponse(
            content="not json",
            model="claude-sonnet-4-6",
            input_tokens=10,
            output_tokens=5,
            stop_reason="end_turn",
        )
        web_search = _make_web_search_client()
        # annotation response still needed for second call
        anthropic = _make_anthropic_client(query_response=bad_response)
        thesis = _make_thesis(title="Duration Trade")
        workflow = WebResearchWorkflow(
            web_search_client=web_search, anthropic_client=anthropic
        )
        workflow.execute(thesis, _make_context())
        called_queries = [c.args[0] for c in web_search.search.call_args_list]
        assert "Duration Trade" in called_queries

    def test_queries_capped_at_five(self):
        six_queries = [f"query {i}" for i in range(6)]
        web_search = _make_web_search_client()
        anthropic = _make_anthropic_client(
            query_response=_make_query_response(six_queries)
        )
        workflow = WebResearchWorkflow(
            web_search_client=web_search, anthropic_client=anthropic
        )
        workflow.execute(_make_thesis(), _make_context())
        assert web_search.search.call_count == 5

    def test_failed_search_does_not_raise(self):
        web_search = MagicMock()
        web_search.search.side_effect = WebSearchClientError("network error")
        anthropic = _make_anthropic_client()
        workflow = WebResearchWorkflow(
            web_search_client=web_search, anthropic_client=anthropic
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.status == WorkflowStatus.PARTIAL

    def test_failed_search_flagged_in_agent_inferences(self):
        web_search = MagicMock()
        web_search.search.side_effect = WebSearchClientError("network error")
        anthropic = _make_anthropic_client()
        workflow = WebResearchWorkflow(
            web_search_client=web_search, anthropic_client=anthropic
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert any("[Agent inference]" in i for i in result.agent_inferences)


# ---------------------------------------------------------------------------
# WebResearchWorkflow — deduplication and candidate cap
# ---------------------------------------------------------------------------


class TestWebResearchWorkflowCandidateHandling:
    def test_duplicate_urls_across_searches_deduplicated(self):
        shared_url = "https://example.com/shared"
        results = [
            _make_search_result(url=shared_url, title="Shared"),
            _make_search_result(url="https://example.com/other", title="Other"),
        ]
        web_search = MagicMock()
        # Both searches return the same results (with duplicates)
        web_search.search.return_value = results
        annotation_sources = [
            {"url": shared_url, "annotation": "A", "is_cited": True},
            {"url": "https://example.com/other", "annotation": "B", "is_cited": False},
        ]
        anthropic = _make_anthropic_client(
            query_response=_make_query_response(["q1", "q2"]),
            annotation_response=_make_annotation_response(annotation_sources),
        )
        workflow = WebResearchWorkflow(
            web_search_client=web_search, anthropic_client=anthropic
        )
        result = workflow.execute(_make_thesis(), _make_context())
        urls = [s["url"] for s in result.structured_output["sources"]]
        assert urls.count(shared_url) == 1

    def test_candidates_capped_at_ten(self):
        # 3 queries × 5 results = 15, should be capped at 10 before annotation call
        web_search = MagicMock()
        web_search.search.side_effect = [
            _make_search_results(n=5, base_url="https://q1.com/a"),
            _make_search_results(n=5, base_url="https://q2.com/b"),
            _make_search_results(n=5, base_url="https://q3.com/c"),
        ]
        annotation_sources = [
            {
                "url": f"https://q1.com/a-{i}",
                "annotation": f"Note {i}",
                "is_cited": i == 0,
            }
            for i in range(3)
        ]
        anthropic = _make_anthropic_client(
            query_response=_make_query_response(["q1", "q2", "q3"]),
            annotation_response=_make_annotation_response(annotation_sources),
        )
        workflow = WebResearchWorkflow(
            web_search_client=web_search, anthropic_client=anthropic
        )
        workflow.execute(_make_thesis(), _make_context())
        # Verify annotation message contained at most 10 candidates
        annotation_msg = anthropic.complete.call_args_list[1][1]["messages"][0][
            "content"
        ]
        # URLs in annotation message should not exceed 10
        assert annotation_msg.count("URL:") <= 10


# ---------------------------------------------------------------------------
# WebResearchWorkflow — ranking logic
# ---------------------------------------------------------------------------


class TestWebResearchWorkflowRanking:
    def test_cited_sources_appear_before_uncited_in_output(self):
        results = _make_search_results(n=3)
        annotation_sources = [
            {
                "url": results[0].url,
                "annotation": "Uncited context.",
                "is_cited": False,
            },
            {"url": results[1].url, "annotation": "Directly cited.", "is_cited": True},
            {"url": results[2].url, "annotation": "More context.", "is_cited": False},
        ]
        anthropic = _make_anthropic_client(
            annotation_response=_make_annotation_response(annotation_sources)
        )
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(results),
            anthropic_client=anthropic,
        )
        result = workflow.execute(_make_thesis(), _make_context())
        sources = result.structured_output["sources"]
        cited = [s for s in sources if s["is_cited"]]
        uncited = [s for s in sources if not s["is_cited"]]
        if cited and uncited:
            assert cited[0]["rank"] < uncited[0]["rank"]

    def test_rank_values_are_sequential_from_one(self):
        results = _make_search_results(n=3)
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(results),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        ranks = sorted(s["rank"] for s in result.structured_output["sources"])
        assert ranks == list(range(1, len(ranks) + 1))

    def test_primary_source_ranked_ahead_of_secondary_within_uncited(self):
        fed_url = "https://www.federalreserve.gov/monetarypolicy/fomc.htm"
        blog_url = "https://www.marketwatch.com/story/rates"
        results = [
            _make_search_result(url=fed_url, title="Fed"),
            _make_search_result(url=blog_url, title="MarketWatch"),
        ]
        annotation_sources = [
            {"url": blog_url, "annotation": "Blog context.", "is_cited": False},
            {"url": fed_url, "annotation": "Fed statement.", "is_cited": False},
        ]
        anthropic = _make_anthropic_client(
            annotation_response=_make_annotation_response(annotation_sources)
        )
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(results),
            anthropic_client=anthropic,
        )
        result = workflow.execute(_make_thesis(), _make_context())
        sources = result.structured_output["sources"]
        fed_rank = next(s["rank"] for s in sources if s["url"] == fed_url)
        blog_rank = next(s["rank"] for s in sources if s["url"] == blog_url)
        assert fed_rank < blog_rank


# ---------------------------------------------------------------------------
# WebResearchWorkflow — structured output
# ---------------------------------------------------------------------------


class TestWebResearchWorkflowOutput:
    def test_result_status_is_completed_on_success(self):
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.status == WorkflowStatus.COMPLETED

    def test_structured_output_has_required_keys(self):
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert "search_queries" in result.structured_output
        assert "sources" in result.structured_output

    def test_each_source_has_required_fields(self):
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        for source in result.structured_output["sources"]:
            for field in [
                "title",
                "url",
                "source_type",
                "annotation",
                "is_cited",
                "rank",
            ]:
                assert field in source, f"Missing field: {field}"

    def test_source_type_values_are_valid(self):
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        valid_types = {"web", "FRED series", "academic paper"}
        for source in result.structured_output["sources"]:
            assert source["source_type"] in valid_types

    def test_search_queries_in_output(self):
        queries = ["rate path 2024", "TLT ETF performance"]
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=_make_anthropic_client(
                query_response=_make_query_response(queries)
            ),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.structured_output["search_queries"] == queries

    def test_workflow_name_is_correct(self):
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.workflow_name == "WebResearchWorkflow"

    def test_raw_output_is_annotation_response_content(self):
        annotation = _make_annotation_response()
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=_make_anthropic_client(annotation_response=annotation),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.raw_output == annotation.content

    def test_partial_status_when_no_candidates(self):
        web_search = MagicMock()
        web_search.search.return_value = []
        anthropic = _make_anthropic_client()
        workflow = WebResearchWorkflow(
            web_search_client=web_search, anthropic_client=anthropic
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.status == WorkflowStatus.PARTIAL
        assert result.structured_output["sources"] == []

    def test_partial_status_when_annotation_json_malformed(self):
        bad_annotation = AnthropicResponse(
            content="not json at all",
            model="claude-sonnet-4-6",
            input_tokens=10,
            output_tokens=5,
            stop_reason="end_turn",
        )
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=_make_anthropic_client(annotation_response=bad_annotation),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.status == WorkflowStatus.PARTIAL

    def test_llm_url_not_in_candidates_flagged_in_inferences(self):
        annotation_sources = [
            {
                "url": "https://not-a-candidate.com/page",
                "annotation": "Hallucinated URL.",
                "is_cited": True,
            }
        ]
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=_make_anthropic_client(
                annotation_response=_make_annotation_response(annotation_sources)
            ),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert any("not in candidate set" in i.lower() for i in result.agent_inferences)


# ---------------------------------------------------------------------------
# WebResearchWorkflow — citations
# ---------------------------------------------------------------------------


class TestWebResearchWorkflowCitations:
    def test_one_citation_per_selected_source(self):
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert len(result.citations) == len(result.structured_output["sources"])

    def test_citations_are_web_source_type(self):
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        for c in result.citations:
            assert c.source_type == CitationSourceType.WEB

    def test_citation_label_is_url(self):
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        for c in result.citations:
            assert c.label == c.url

    def test_citation_retrieval_date_is_set(self):
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        for c in result.citations:
            assert c.retrieval_date is not None

    def test_no_citations_when_no_candidates(self):
        web_search = MagicMock()
        web_search.search.return_value = []
        workflow = WebResearchWorkflow(
            web_search_client=web_search, anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.citations == []


# ---------------------------------------------------------------------------
# WebResearchWorkflow — annotation generation
# ---------------------------------------------------------------------------


class TestWebResearchWorkflowAnnotation:
    def test_annotation_stored_in_source_output(self):
        annotation_sources = [
            {
                "url": "https://example.com/article-0",
                "annotation": "Exactly this annotation text.",
                "is_cited": True,
            }
        ]
        results = [_make_search_result(url="https://example.com/article-0")]
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(results),
            anthropic_client=_make_anthropic_client(
                annotation_response=_make_annotation_response(annotation_sources)
            ),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert (
            result.structured_output["sources"][0]["annotation"]
            == "Exactly this annotation text."
        )

    def test_thesis_title_in_annotation_prompt(self):
        anthropic = _make_anthropic_client()
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=anthropic,
        )
        thesis = _make_thesis(title="EM Currency Devaluation Play")
        workflow.execute(thesis, _make_context())
        annotation_msg = anthropic.complete.call_args_list[1][1]["messages"][0][
            "content"
        ]
        assert "EM Currency Devaluation Play" in annotation_msg

    def test_candidate_urls_in_annotation_prompt(self):
        results = [_make_search_result(url="https://example.com/unique-url")]
        anthropic = _make_anthropic_client()
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(results),
            anthropic_client=anthropic,
        )
        workflow.execute(_make_thesis(), _make_context())
        annotation_msg = anthropic.complete.call_args_list[1][1]["messages"][0][
            "content"
        ]
        assert "https://example.com/unique-url" in annotation_msg


# ---------------------------------------------------------------------------
# WebResearchWorkflow — LLM calls
# ---------------------------------------------------------------------------


class TestWebResearchWorkflowLLMCalls:
    def test_two_llm_calls_made(self):
        anthropic = _make_anthropic_client()
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=anthropic,
        )
        workflow.execute(_make_thesis(), _make_context())
        assert anthropic.complete.call_count == 2

    def test_query_generation_uses_correct_task_type(self):
        anthropic = _make_anthropic_client()
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=anthropic,
        )
        workflow.execute(_make_thesis(), _make_context())
        first_call_kwargs = anthropic.complete.call_args_list[0][1]
        assert first_call_kwargs["task_type"] == "web_research_queries"

    def test_annotation_uses_correct_task_type(self):
        anthropic = _make_anthropic_client()
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=anthropic,
        )
        workflow.execute(_make_thesis(), _make_context())
        second_call_kwargs = anthropic.complete.call_args_list[1][1]
        assert second_call_kwargs["task_type"] == "web_research_annotation"

    def test_both_calls_use_workflow_model(self):
        anthropic = _make_anthropic_client()
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=anthropic,
        )
        workflow.execute(_make_thesis(), _make_context())
        for call_kwargs in [c[1] for c in anthropic.complete.call_args_list]:
            assert call_kwargs["model"] == "claude-sonnet-4-6"

    def test_workflow_run_id_passed_to_both_calls(self):
        anthropic = _make_anthropic_client()
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=anthropic,
        )
        run_id = uuid.uuid4()
        ctx = _make_context()
        ctx.current_workflow_run_id = run_id
        workflow.execute(_make_thesis(), ctx)
        for call_kwargs in [c[1] for c in anthropic.complete.call_args_list]:
            assert call_kwargs["workflow_run_id"] == run_id

    def test_thesis_title_in_query_generation_prompt(self):
        anthropic = _make_anthropic_client()
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=anthropic,
        )
        thesis = _make_thesis(title="Stagflation Hedge")
        workflow.execute(thesis, _make_context())
        query_msg = anthropic.complete.call_args_list[0][1]["messages"][0]["content"]
        assert "Stagflation Hedge" in query_msg


# ---------------------------------------------------------------------------
# WebResearchWorkflow — DB persistence
# ---------------------------------------------------------------------------


class TestWebResearchWorkflowPersistence:
    def test_further_reading_rows_added_to_db(self):
        db = MagicMock()
        ctx = WorkflowContext(thesis=_make_thesis(), db=db)
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=_make_anthropic_client(),
        )
        workflow.execute(_make_thesis(), ctx)
        assert db.add.called
        assert db.commit.called

    def test_one_db_add_per_source(self):
        db = MagicMock()
        ctx = WorkflowContext(thesis=_make_thesis(), db=db)
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), ctx)
        assert db.add.call_count == len(result.structured_output["sources"])

    def test_no_db_write_when_context_db_is_none(self):
        ctx = WorkflowContext(thesis=_make_thesis(), db=None)
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=_make_anthropic_client(),
        )
        # Should not raise even with no db session
        result = workflow.execute(_make_thesis(), ctx)
        assert result.status == WorkflowStatus.COMPLETED

    def test_db_failure_does_not_raise(self):
        db = MagicMock()
        db.commit.side_effect = Exception("DB unavailable")
        ctx = WorkflowContext(thesis=_make_thesis(), db=db)
        workflow = WebResearchWorkflow(
            web_search_client=_make_web_search_client(),
            anthropic_client=_make_anthropic_client(),
        )
        # Should not propagate the DB exception
        result = workflow.execute(_make_thesis(), ctx)
        assert result is not None
