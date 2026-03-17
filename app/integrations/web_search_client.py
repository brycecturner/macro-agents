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

        Args:
            query: The search query string.

        Returns:
            List of :class:`SearchResult` objects, with primary sources
            ranked before aggregators.  The list may be empty if no results
            are found.

        Raises:
            WebSearchClientError: If the Anthropic API call fails for any
                reason.
        """
        retrieved_at = datetime.now(tz=UTC)

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[{"role": "user", "content": query}],
            )
        except Exception as exc:
            raise WebSearchClientError(
                f"Web search failed for query '{query}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        results = _extract_results(response.content, retrieved_at)
        ranked = _rank_results(results)

        logger.debug(
            "Web search completed: query=%r results=%d primary=%d",
            query,
            len(ranked),
            sum(1 for r in ranked if _is_primary_source(r.url)),
        )
        return ranked
