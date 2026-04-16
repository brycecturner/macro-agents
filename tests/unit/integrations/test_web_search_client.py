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


def _make_response(
    content_blocks: list[MagicMock], stop_reason: str = "end_turn"
) -> MagicMock:
    response = MagicMock()
    response.content = content_blocks
    response.stop_reason = stop_reason
    return response


def _make_tool_use_block(tool_use_id: str = "tu_123") -> MagicMock:
    """Build a mock tool_use content block (signals another search turn)."""
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_use_id
    return block


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


# ---------------------------------------------------------------------------
# tool_use continuation loop
# ---------------------------------------------------------------------------


class TestToolUseContinuationLoop:
    def test_single_turn_end_turn_returns_results(self, client, mock_anthropic):
        """Normal path: stop_reason=end_turn on first call."""
        items = [_make_result_item("https://fed.gov/page", "Fed")]
        mock_anthropic.messages.create.return_value = _make_response(
            [_make_search_block(items)], stop_reason="end_turn"
        )

        results = client.search("test")

        assert len(results) == 1
        assert mock_anthropic.messages.create.call_count == 1

    def test_loops_when_stop_reason_is_tool_use(self, client, mock_anthropic):
        """Second API call is made when first response has stop_reason=tool_use."""
        turn1 = _make_response(
            [
                _make_search_block([_make_result_item("https://a.com", "A")]),
                _make_tool_use_block("tu_1"),
            ],
            stop_reason="tool_use",
        )
        turn2 = _make_response(
            [_make_search_block([_make_result_item("https://b.com", "B")])],
            stop_reason="end_turn",
        )
        mock_anthropic.messages.create.side_effect = [turn1, turn2]

        results = client.search("test")

        assert mock_anthropic.messages.create.call_count == 2
        assert len(results) == 2

    def test_results_accumulated_across_turns(self, client, mock_anthropic):
        """Results from all turns are combined and returned together."""
        turn1 = _make_response(
            [
                _make_search_block([_make_result_item("https://imf.org/a", "IMF")]),
                _make_tool_use_block("tu_1"),
            ],
            stop_reason="tool_use",
        )
        turn2 = _make_response(
            [_make_search_block([_make_result_item("https://news.com/b", "News")])],
            stop_reason="end_turn",
        )
        mock_anthropic.messages.create.side_effect = [turn1, turn2]

        results = client.search("test")

        urls = {r.url for r in results}
        assert "https://imf.org/a" in urls
        assert "https://news.com/b" in urls

    def test_continuation_message_contains_tool_result_block(
        self, client, mock_anthropic
    ):
        """Second call includes an assistant turn and a tool_result user turn."""
        turn1_content = [
            _make_search_block([_make_result_item("https://a.com", "A")]),
            _make_tool_use_block("tu_abc"),
        ]
        turn1 = _make_response(turn1_content, stop_reason="tool_use")
        turn2 = _make_response([], stop_reason="end_turn")
        mock_anthropic.messages.create.side_effect = [turn1, turn2]

        client.search("test query")

        second_call = mock_anthropic.messages.create.call_args_list[1]
        messages = second_call.kwargs.get("messages") or second_call.args[0]

        # messages[0] = original user query
        # messages[1] = assistant turn with turn1 content
        # messages[2] = user turn with tool_result block
        assert len(messages) == 3
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] is turn1_content

        user_turn = messages[2]
        assert user_turn["role"] == "user"
        tool_result = user_turn["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "tu_abc"

    def test_stops_when_no_tool_use_blocks_in_non_end_turn_response(
        self, client, mock_anthropic
    ):
        """Stops after one call: non-end_turn stop_reason with no tool_use blocks."""
        # Response has stop_reason="max_tokens" (not end_turn, not tool_use) with
        # no tool_use blocks — should still stop after one call.
        response = _make_response(
            [_make_search_block([_make_result_item("https://example.com", "E")])],
            stop_reason="max_tokens",
        )
        mock_anthropic.messages.create.return_value = response

        results = client.search("test")

        assert mock_anthropic.messages.create.call_count == 1
        assert len(results) == 1

    def test_api_error_on_second_turn_raises_web_search_client_error(
        self, client, mock_anthropic
    ):
        """WebSearchClientError is raised if the API fails on a continuation turn."""
        turn1 = _make_response([_make_tool_use_block("tu_1")], stop_reason="tool_use")
        mock_anthropic.messages.create.side_effect = [turn1, RuntimeError("timeout")]

        with pytest.raises(WebSearchClientError, match="Web search failed"):
            client.search("test")

    def test_multiple_tool_use_blocks_all_acknowledged(self, client, mock_anthropic):
        """Every tool_use block in a response gets a tool_result in the continuation."""
        turn1_content = [
            _make_tool_use_block("tu_1"),
            _make_tool_use_block("tu_2"),
        ]
        turn1 = _make_response(turn1_content, stop_reason="tool_use")
        turn2 = _make_response([], stop_reason="end_turn")
        mock_anthropic.messages.create.side_effect = [turn1, turn2]

        client.search("test")

        second_call = mock_anthropic.messages.create.call_args_list[1]
        messages = second_call.kwargs.get("messages") or second_call.args[0]
        tool_result_turn = messages[2]
        tool_result_ids = {b["tool_use_id"] for b in tool_result_turn["content"]}
        assert tool_result_ids == {"tu_1", "tu_2"}
