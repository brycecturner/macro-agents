import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import anthropic

logger = logging.getLogger(__name__)

# Domains and TLD patterns considered primary sources for quality ranking.
# Primary sources: government, academic institutions, central banks, and
# major international financial/economic organisations.
_PRIMARY_DOMAIN_FRAGMENTS = frozenset(
    {
        # US government
        ".gov",
        "federalreserve.gov",
        "stlouisfed.org",
        "bls.gov",
        "bea.gov",
        "treasury.gov",
        "whitehouse.gov",
        "sec.gov",
        "cftc.gov",
        # International organisations / central banks
        "imf.org",
        "worldbank.org",
        "bis.org",
        "ecb.europa.eu",
        "oecd.org",
        "un.org",
        "wto.org",
        "bankofengland.co.uk",
        "boj.or.jp",
        # Academic
        ".edu",
        ".ac.uk",
        "nber.org",
        "ssrn.com",
        "brookings.edu",
        "piie.com",
        "iie.com",
    }
)


class WebSearchClientError(Exception):
    """Raised when a web search call fails."""


@dataclass
class SearchResult:
    """A single web search result, with citation metadata.

    Citation format: ``{url}, retrieved {retrieval_date.date()}``
    """

    url: str
    title: str
    snippet: str
    retrieval_date: datetime


def _is_primary_source(url: str) -> bool:
    """Return True if *url* belongs to a primary source domain.

    Primary sources include US and international government sites, central
    banks, major economic organisations, and academic institutions.
    """
    url_lower = url.lower()
    return any(fragment in url_lower for fragment in _PRIMARY_DOMAIN_FRAGMENTS)


def _rank_results(results: list[SearchResult]) -> list[SearchResult]:
    """Return *results* with primary sources first.

    Relative order within each tier is preserved so the caller can rely on
    the original search-relevance ordering within tiers.
    """
    primary = [r for r in results if _is_primary_source(r.url)]
    secondary = [r for r in results if not _is_primary_source(r.url)]
    return primary + secondary


def _extract_results(content: list, retrieved_at: datetime) -> list[SearchResult]:
    """Parse SearchResult objects out of API response content blocks.

    The Anthropic web search tool returns results in content blocks of type
    ``web_search_tool_result``.  Each item within such a block has ``url``,
    ``title``, and ``encrypted_content`` attributes.
    """
    results: list[SearchResult] = []

    for block in content:
        if getattr(block, "type", None) != "web_search_tool_result":
            continue

        for item in getattr(block, "content", []):
            url = getattr(item, "url", "") or ""
            title = getattr(item, "title", "") or ""
            raw = getattr(item, "encrypted_content", "") or ""
            snippet = raw[:300].strip()

            if not url:
                continue

            results.append(
                SearchResult(
                    url=url,
                    title=title,
                    snippet=snippet,
                    retrieval_date=retrieved_at,
                )
            )

    return results


class WebSearchClient:
    """Wraps the Anthropic web search tool with source quality ranking.

    Calling :meth:`search` executes a web search via the Anthropic
    ``web_search_20250305`` built-in tool and returns a ranked list of
    :class:`SearchResult` objects.  Primary sources (government, academic,
    central bank) are ranked ahead of aggregators and news sites.

    All results carry a ``retrieval_date`` for citation purposes.

    Instantiate once and reuse — there is no per-instance state beyond the
    underlying Anthropic client.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def search(self, query: str) -> list[SearchResult]:
        """Execute a web search and return quality-ranked results.

        Implements the tool_use continuation loop required by the Anthropic
        Messages API.  When ``stop_reason == "tool_use"`` the model wants to
        invoke the search tool again; the assistant turn is appended to the
        conversation along with a ``tool_result`` acknowledgement for each
        ``tool_use`` block, and the API is called again.  The loop exits when
        ``stop_reason == "end_turn"``.

        For the built-in ``web_search_20250305`` server-side tool, Anthropic
        executes the search and embeds the results in the assistant response as
        ``web_search_tool_result`` blocks before the response reaches the
        client.  The ``tool_result`` sent back in the continuation turn is an
        empty acknowledgement — it signals that the client has received the
        tool output so the model can proceed.

        Args:
            query: The search query string.

        Returns:
            List of :class:`SearchResult` objects, with primary sources
            ranked before aggregators.  Results from all continuation turns
            are combined.  The list may be empty if no results are found.

        Raises:
            WebSearchClientError: If the Anthropic API call fails for any
                reason.
        """
        retrieved_at = datetime.now(tz=UTC)
        messages: list[dict] = [{"role": "user", "content": query}]
        all_results: list[SearchResult] = []

        while True:
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    messages=messages,
                )
            except Exception as exc:
                raise WebSearchClientError(
                    f"Web search failed for query '{query}': "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            all_results.extend(_extract_results(response.content, retrieved_at))

            if response.stop_reason == "end_turn":
                break

            # Build continuation: append the assistant turn, then a user turn
            # containing empty tool_result acknowledgements for every tool_use
            # block in the response.
            messages.append({"role": "assistant", "content": response.content})
            tool_results = [
                {"type": "tool_result", "tool_use_id": block.id, "content": ""}
                for block in response.content
                if getattr(block, "type", None) == "tool_use"
            ]
            if not tool_results:
                # stop_reason was not "end_turn" but no tool_use blocks are
                # present — cannot build a valid continuation; stop here.
                logger.warning(
                    "WebSearchClient: stop_reason=%r but no tool_use blocks "
                    "found in response for query %r — stopping.",
                    response.stop_reason,
                    query,
                )
                break
            messages.append({"role": "user", "content": tool_results})

        ranked = _rank_results(all_results)
        logger.debug(
            "Web search completed: query=%r results=%d primary=%d",
            query,
            len(ranked),
            sum(1 for r in ranked if _is_primary_source(r.url)),
        )
        return ranked
