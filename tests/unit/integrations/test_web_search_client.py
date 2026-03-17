"""Unit tests for WebSearchClient.

All tests mock the underlying anthropic.Anthropic client so no real network
calls are made.  Tests cover result parsing from web_search_tool_result
blocks and source quality ranking logic.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from app.integrations.web_search_client import (
    SearchResult,
    WebSearchClient,
    WebSearchClientError,
    _is_primary_source,
    _rank_results,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_result_item(
    url: str, title: str = "Title", content: str = "Snippet text."
) -> MagicMock:
    """Build a mock web search result item (the leaf object inside a block)."""
    item = MagicMock()
    item.url = url
    item.title = title
    item.encrypted_content = content
    return item


def _make_search_block(items: list[MagicMock]) -> MagicMock:
    """Build a mock web_search_tool_result content block."""
    block = MagicMock()
    block.type = "web_search_tool_result"
    block.content = items
    return block


def _make_text_block(text: str = "Some response text.") -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_response(content_blocks: list[MagicMock]) -> MagicMock:
    response = MagicMock()
    response.content = content_blocks
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_anthropic():
    with patch("app.integrations.web_search_client.anthropic.Anthropic") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def client(mock_anthropic) -> WebSearchClient:
    return WebSearchClient(api_key="test-key")


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


class TestResultParsing:
    def test_returns_search_result_objects(self, client, mock_anthropic):
        items = [_make_result_item("https://example.com", "Example")]
        mock_anthropic.messages.create.return_value = _make_response(
            [_make_search_block(items)]
        )

        results = client.search("test query")

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)

    def test_url_and_title_preserved(self, client, mock_anthropic):
        items = [_make_result_item("https://example.com/page", "Example Page")]
        mock_anthropic.messages.create.return_value = _make_response(
            [_make_search_block(items)]
        )

        results = client.search("test")

        assert results[0].url == "https://example.com/page"
        assert results[0].title == "Example Page"

    def test_snippet_comes_from_encrypted_content(self, client, mock_anthropic):
        items = [_make_result_item("https://example.com", "T", "Page content here.")]
        mock_anthropic.messages.create.return_value = _make_response(
            [_make_search_block(items)]
        )

        results = client.search("test")

        assert results[0].snippet == "Page content here."

    def test_snippet_truncated_to_300_chars(self, client, mock_anthropic):
        long_content = "x" * 500
        items = [_make_result_item("https://example.com", "T", long_content)]
        mock_anthropic.messages.create.return_value = _make_response(
            [_make_search_block(items)]
        )

        results = client.search("test")

        assert len(results[0].snippet) <= 300

    def test_retrieval_date_is_utc_datetime(self, client, mock_anthropic):
        items = [_make_result_item("https://example.com", "T")]
        mock_anthropic.messages.create.return_value = _make_response(
            [_make_search_block(items)]
        )

        results = client.search("test")

        assert isinstance(results[0].retrieval_date, datetime)
        assert results[0].retrieval_date.tzinfo is not None
        assert results[0].retrieval_date.tzinfo == UTC

    def test_ignores_non_search_blocks(self, client, mock_anthropic):
        text_block = _make_text_block()
        items = [_make_result_item("https://fed.gov/page", "Fed")]
        search_block = _make_search_block(items)

        mock_anthropic.messages.create.return_value = _make_response(
            [text_block, search_block]
        )

        results = client.search("test")

        assert len(results) == 1
        assert "fed.gov" in results[0].url

    def test_returns_empty_list_when_no_search_blocks(self, client, mock_anthropic):
        mock_anthropic.messages.create.return_value = _make_response(
            [_make_text_block()]
        )

        results = client.search("test")

        assert results == []

    def test_skips_items_with_empty_url(self, client, mock_anthropic):
        items = [
            _make_result_item("", "No URL item"),
            _make_result_item("https://example.com", "Has URL"),
        ]
        mock_anthropic.messages.create.return_value = _make_response(
            [_make_search_block(items)]
        )

        results = client.search("test")

        assert len(results) == 1
        assert results[0].url == "https://example.com"

    def test_handles_multiple_search_blocks(self, client, mock_anthropic):
        block1 = _make_search_block([_make_result_item("https://a.com", "A")])
        block2 = _make_search_block([_make_result_item("https://b.com", "B")])

        mock_anthropic.messages.create.return_value = _make_response([block1, block2])

        results = client.search("test")

        assert len(results) == 2

    def test_raises_web_search_client_error_on_api_failure(
        self, client, mock_anthropic
    ):
        mock_anthropic.messages.create.side_effect = RuntimeError("connection refused")

        with pytest.raises(WebSearchClientError, match="Web search failed"):
            client.search("test")

    def test_error_message_includes_query(self, client, mock_anthropic):
        mock_anthropic.messages.create.side_effect = RuntimeError("timeout")

        with pytest.raises(WebSearchClientError, match="my specific query"):
            client.search("my specific query")

    def test_passes_query_to_anthropic_api(self, client, mock_anthropic):
        mock_anthropic.messages.create.return_value = _make_response([])

        client.search("FOMC decision 2025")

        call_kwargs = mock_anthropic.messages.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
        assert any("FOMC decision 2025" in str(m) for m in messages)

    def test_uses_web_search_tool_in_api_call(self, client, mock_anthropic):
        mock_anthropic.messages.create.return_value = _make_response([])

        client.search("test")

        call_kwargs = mock_anthropic.messages.create.call_args
        tools = call_kwargs.kwargs.get("tools") or []
        tool_types = [t.get("type") for t in tools if isinstance(t, dict)]
        assert "web_search_20250305" in tool_types


# ---------------------------------------------------------------------------
# Source quality ranking
# ---------------------------------------------------------------------------


class TestSourceQualityRanking:
    def test_primary_sources_ranked_before_aggregators(self, client, mock_anthropic):
        items = [
            _make_result_item("https://news-blog.com/article", "Blog"),
            _make_result_item("https://federalreserve.gov/press", "Fed Press"),
            _make_result_item("https://another-site.com/post", "Post"),
        ]
        mock_anthropic.messages.create.return_value = _make_response(
            [_make_search_block(items)]
        )

        results = client.search("test")

        assert "federalreserve.gov" in results[0].url

    def test_relative_order_preserved_within_tiers(self, client, mock_anthropic):
        items = [
            _make_result_item("https://news1.com", "News 1"),
            _make_result_item("https://imf.org/report", "IMF"),
            _make_result_item("https://news2.com", "News 2"),
            _make_result_item("https://nber.org/paper", "NBER"),
        ]
        mock_anthropic.messages.create.return_value = _make_response(
            [_make_search_block(items)]
        )

        results = client.search("test")

        # Primary tier: IMF then NBER (original relative order)
        assert "imf.org" in results[0].url
        assert "nber.org" in results[1].url
        # Secondary tier: news1 then news2
        assert "news1.com" in results[2].url
        assert "news2.com" in results[3].url

    def test_all_primary_sources_sorted_before_any_secondary(
        self, client, mock_anthropic
    ):
        items = [
            _make_result_item("https://bloomberg.com", "Bloomberg"),
            _make_result_item("https://bls.gov/data", "BLS"),
            _make_result_item("https://seekingalpha.com", "SA"),
            _make_result_item("https://brookings.edu/report", "Brookings"),
        ]
        mock_anthropic.messages.create.return_value = _make_response(
            [_make_search_block(items)]
        )

        results = client.search("test")

        primary_urls = {r.url for r in results[:2]}
        assert any("bls.gov" in u for u in primary_urls)
        assert any("brookings.edu" in u for u in primary_urls)

        secondary_urls = {r.url for r in results[2:]}
        assert any("bloomberg.com" in u for u in secondary_urls)
        assert any("seekingalpha.com" in u for u in secondary_urls)

    def test_all_aggregators_returns_unchanged_order(self, client, mock_anthropic):
        items = [
            _make_result_item("https://bloomberg.com", "A"),
            _make_result_item("https://reuters.com", "B"),
            _make_result_item("https://cnbc.com", "C"),
        ]
        mock_anthropic.messages.create.return_value = _make_response(
            [_make_search_block(items)]
        )

        results = client.search("test")

        urls = [r.url for r in results]
        assert urls == [
            "https://bloomberg.com",
            "https://reuters.com",
            "https://cnbc.com",
        ]


# ---------------------------------------------------------------------------
# _is_primary_source unit tests
# ---------------------------------------------------------------------------


class TestIsPrimarySource:
    @pytest.mark.parametrize(
        "url",
        [
            "https://federalreserve.gov/releases/h15",
            "https://www.bls.gov/cpi/",
            "https://www.bea.gov/data",
            "https://home.treasury.gov/policy",
            "https://www.imf.org/en/Publications/WEO",
            "https://www.worldbank.org/en/topic",
            "https://www.bis.org/statistics",
            "https://www.ecb.europa.eu/stats",
            "https://stats.oecd.org/index",
            "https://nber.org/papers/w12345",
            "https://papers.ssrn.com/abstract=123",
            "https://www.brookings.edu/research",
            "https://data.stlouisfed.org/series/DGS10",
            "https://university.edu/economics/paper.pdf",
            "https://lse.ac.uk/research",
            "https://whitehouse.gov/briefing",
        ],
    )
    def test_returns_true_for_primary_sources(self, url):
        assert _is_primary_source(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://bloomberg.com/markets/rates",
            "https://www.reuters.com/markets",
            "https://seekingalpha.com/article/123",
            "https://reddit.com/r/economics",
            "https://medium.com/@economist",
            "https://cnbc.com/markets",
            "https://tradingeconomics.com/data",
        ],
    )
    def test_returns_false_for_aggregators_and_news(self, url):
        assert _is_primary_source(url) is False


# ---------------------------------------------------------------------------
# _rank_results unit tests
# ---------------------------------------------------------------------------


class TestRankResults:
    def test_empty_list_returns_empty(self):
        assert _rank_results([]) == []

    def test_only_primary_sources_unchanged_order(self):
        results = [
            SearchResult(
                url="https://imf.org/a",
                title="A",
                snippet="",
                retrieval_date=datetime.now(tz=UTC),
            ),
            SearchResult(
                url="https://bls.gov/b",
                title="B",
                snippet="",
                retrieval_date=datetime.now(tz=UTC),
            ),
        ]
        ranked = _rank_results(results)
        assert [r.url for r in ranked] == ["https://imf.org/a", "https://bls.gov/b"]

    def test_only_secondary_sources_unchanged_order(self):
        results = [
            SearchResult(
                url="https://news1.com",
                title="N1",
                snippet="",
                retrieval_date=datetime.now(tz=UTC),
            ),
            SearchResult(
                url="https://news2.com",
                title="N2",
                snippet="",
                retrieval_date=datetime.now(tz=UTC),
            ),
        ]
        ranked = _rank_results(results)
        assert [r.url for r in ranked] == ["https://news1.com", "https://news2.com"]
