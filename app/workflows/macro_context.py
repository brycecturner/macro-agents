"""MacroContextWorkflow — pulls FRED (and optionally OECD) series and summarizes
the current macro backdrop.

This is the first workflow in the research chain. The structured_output includes
full historical data per series so that HistoricalAnalogWorkflow can search for
genuinely comparable macro regimes across decades, not just the recent past.

The LLM prompt receives only a condensed summary of current conditions to keep
cost predictable. The full time-series data lives in structured_output.series.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from app.integrations.anthropic_client import AnthropicClient
from app.integrations.fred_client import FREDClient, FREDClientError
from app.integrations.oecd_client import OECDClient, OECDClientError, OECDSeriesSpec
from app.workflows.base import (
    BaseWorkflow,
    Citation,
    CitationSourceType,
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a macro research analyst. Summarize the current macro economic backdrop \
as it relates to a specific investment thesis.

Respond with a JSON object containing exactly two keys:
- "summary": A single paragraph (3-5 sentences) summarizing the macro backdrop \
and its relevance to the thesis. Write factually — reference the data values provided.
- "agent_inferences": A list of strings for any reasoning or interpretation that \
goes beyond the raw data (e.g. trend judgments, regime assessments, historical \
parallels). Each string MUST start with "[Agent inference]".

Respond only with the JSON object. No markdown fences, no preamble.\
"""


def _build_user_message(thesis, data_lines: list[str]) -> str:
    notes = thesis.notes or "(none)"
    direction = (
        thesis.direction.value
        if hasattr(thesis.direction, "value")
        else str(thesis.direction)
    )
    data_block = "\n".join(data_lines) or "(no data available)"
    return (
        f"Thesis: {thesis.title}\n"
        f"Direction: {direction}\n"
        f"Time Horizon: {thesis.time_horizon}\n"
        f"Notes: {notes}\n\n"
        f"Current macro data:\n{data_block}"
    )


# ---------------------------------------------------------------------------
# Series configuration
# ---------------------------------------------------------------------------

# Core FRED series fetched for every thesis.
_FRED_CORE_SERIES = ["T10Y2Y", "CPIAUCSL", "FEDFUNDS", "UNRATE"]

# Core OECD series fetched when an OECDClient is provided.
# The old stats.oecd.org SDMX-JSON API was shut down 2024-07-01. These specs
# target the new sdmx.oecd.org REST API. To verify or update any dimension_key,
# open https://data-explorer.oecd.org, find the indicator, and click the
# "Developer API" button above the data table to get the exact URL.
_OECD_CORE_SERIES: list[OECDSeriesSpec] = [
    OECDSeriesSpec(
        agency="OECD.SDD.STES",
        dataflow="DSD_STES@DF_FINMARK",
        version="4.0",
        # DSD_STES has 9 dimensions: REF_AREA.FREQ.MEASURE.UNIT_MEASURE.ACTIVITY.
        # ADJUSTMENT.TRANSFORMATION.TIME_HORIZ.METHODOLOGY — unused slots use _Z.
        # IRSTCB does not exist in this dataflow; IRSTCI is the ECB call/overnight
        # rate (nearest proxy for the ECB policy rate available here).
        # EA19 still has data in this series; EA20 is also valid post-2023.
        dimension_key="EA19.M.IRSTCI.PA._Z._Z._Z._Z.N",
        label="ECB Policy Rate (Short-term Call Rate)",
    ),
    OECDSeriesSpec(
        agency="OECD.SDD.TPS",
        dataflow="DSD_PRICES@DF_PRICES_ALL",
        version="1.0",
        # DSD_PRICES has 8 dimensions: REF_AREA.FREQ.METHODOLOGY.MEASURE.
        # UNIT_MEASURE.EXPENDITURE.ADJUSTMENT.TRANSFORMATION.
        # EA19 aggregate was discontinued; EA20 covers the current eurozone
        # (Croatia joined Jan 2023). TRANSFORMATION=GY = year-on-year % change.
        dimension_key="EA20.A.N.CPI.PA._T.N.GY",
        label="Eurozone CPI",
    ),
    OECDSeriesSpec(
        agency="OECD.SDD.STES",
        dataflow="DSD_STES@DF_CLI",
        version="4.1",
        # DSD_STES v4.1 dimensions same as v4.0. Measure is CCICP (Composite
        # Leading Indicator, amplitude-adjusted) — LI was not a valid code.
        # ADJUSTMENT=AA (amplitude-adjusted), TRANSFORMATION=IX (index),
        # METHODOLOGY=H (OECD CLI methodology).
        dimension_key="OECD.M.CCICP.IX._Z.AA.IX._Z.H",
        label="OECD Composite Leading Indicator",
    ),
]

# Long lookback so HistoricalAnalogWorkflow has decades of data to search.
_LOOKBACK_YEARS = 30
_LOOKBACK_DAYS = 365 * _LOOKBACK_YEARS

# Number of recent monthly snapshots included in the LLM prompt.
_PROMPT_RECENT_MONTHS = 6


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@dataclass
class _SeriesSnapshot:
    """Processed FRED/OECD series data for structured_output and LLM prompt."""

    series_id: str
    label: str
    current_value: float
    current_date: date
    year_ago_value: float | None
    yoy_change: float | None
    # Full history as list of {date, value} dicts — for HistoricalAnalogWorkflow
    historical_data: list[dict]
    # Condensed recent data for LLM prompt (last _PROMPT_RECENT_MONTHS months)
    recent_prompt_data: list[dict]


def _process_series(raw: pd.Series) -> _SeriesSnapshot | None:
    """Build a _SeriesSnapshot from a raw pandas Series. Returns None if empty."""
    clean = raw.dropna()
    if clean.empty:
        return None

    latest_val = float(clean.iloc[-1])
    latest_date = clean.index[-1].date()

    # Year-ago: closest observation within the prior year
    target_ago = latest_date - timedelta(days=365)
    ago_slice = clean[clean.index.date <= target_ago]  # type: ignore[attr-defined]
    year_ago_val = float(ago_slice.iloc[-1]) if not ago_slice.empty else None
    yoy_change = (
        round(latest_val - year_ago_val, 6) if year_ago_val is not None else None
    )

    # Full monthly history for downstream workflows
    try:
        monthly_full = clean.resample("ME").last().dropna()
        historical_data = [
            {"date": idx.date().isoformat(), "value": round(float(v), 6)}
            for idx, v in monthly_full.items()
        ]
    except Exception:
        historical_data = []

    # Condensed recent data for LLM prompt
    recent_prompt_data = historical_data[-_PROMPT_RECENT_MONTHS:]

    return _SeriesSnapshot(
        series_id="",  # caller fills this in
        label="",
        current_value=round(latest_val, 6),
        current_date=latest_date,
        year_ago_value=round(year_ago_val, 6) if year_ago_val is not None else None,
        yoy_change=yoy_change,
        historical_data=historical_data,
        recent_prompt_data=recent_prompt_data,
    )


def _snapshot_to_output(snap: _SeriesSnapshot) -> dict:
    """Serialize a _SeriesSnapshot to the structured_output dict format."""
    return {
        "label": snap.label,
        "current_value": snap.current_value,
        "current_date": snap.current_date.isoformat(),
        "year_ago_value": snap.year_ago_value,
        "yoy_change": snap.yoy_change,
        "historical_data": snap.historical_data,
    }


def _prompt_line(snap: _SeriesSnapshot) -> str:
    """Build a single human-readable line for the LLM prompt."""
    yoy = f", YoY change: {snap.yoy_change:+.4f}" if snap.yoy_change is not None else ""
    return (
        f"- {snap.series_id} ({snap.label}): "
        f"{snap.current_value:.4f} ({snap.current_date}){yoy}"
    )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class MacroContextWorkflow(BaseWorkflow):
    """Pull FRED (and optionally OECD) macro series and summarize current conditions.

    Fetches up to 30 years of history per series. The full historical data is
    stored in structured_output for use by HistoricalAnalogWorkflow. The LLM
    prompt receives only a condensed current-conditions summary.

    Outputs (structured_output):
        summary (str): one-paragraph macro backdrop narrative from the LLM
        series (dict[str, dict]): per-series snapshot including current value,
            YoY change, and full monthly historical_data list
    """

    name = "MacroContextWorkflow"
    description = (
        "Pulls core FRED macro series (yield curve, CPI, Fed Funds, unemployment) "
        "and optional OECD series; summarizes the current macro backdrop "
        "for the thesis."
    )
    required_inputs = ["title", "direction", "time_horizon", "notes"]
    model: str = "claude-sonnet-4-6"

    def __init__(
        self,
        fred_client: FREDClient | None = None,
        oecd_client: OECDClient | None = None,
        anthropic_client: AnthropicClient | None = None,
    ) -> None:
        # Clients default to None and are created lazily from settings in execute().
        # Inject explicitly in tests to use mocks.
        self._fred = fred_client
        self._oecd = oecd_client
        self._anthropic = anthropic_client

    def execute(self, thesis, context: WorkflowContext) -> WorkflowResult:
        if self._fred is None or self._anthropic is None:
            from app.core.settings import get_settings

            settings = get_settings()
        else:
            settings = None  # type: ignore[assignment]

        fred = self._fred or FREDClient(api_key=settings.fred_api_key)
        anthropic = self._anthropic or AnthropicClient(
            api_key=settings.anthropic_api_key,
            db=context.db,
        )

        end_date = date.today()
        start_date = end_date - timedelta(days=_LOOKBACK_DAYS)

        citations: list[Citation] = []
        series_output: dict[str, dict] = {}
        prompt_lines: list[str] = []

        # --- FRED series ---
        for series_id in _FRED_CORE_SERIES:
            try:
                result = fred.get_series(
                    series_id, start_date=start_date, end_date=end_date
                )
            except FREDClientError:
                logger.warning("Failed to fetch FRED series %s — skipping", series_id)
                continue

            snap = _process_series(result.data)
            if snap is None:
                logger.warning("FRED series %s returned no usable data", series_id)
                continue

            snap.series_id = series_id
            snap.label = series_id  # FRED series IDs are self-describing

            series_output[series_id] = _snapshot_to_output(snap)
            prompt_lines.append(_prompt_line(snap))
            citations.append(
                Citation(
                    source_type=CitationSourceType.FRED,
                    label=f"FRED:{series_id}, retrieved {result.retrieved_at.date()}",
                    url=None,
                    retrieval_date=result.retrieved_at.date(),
                )
            )

        # --- OECD series (optional) ---
        oecd = self._oecd
        if oecd is not None:
            for spec in _OECD_CORE_SERIES:
                try:
                    result = oecd.get_series(
                        spec=spec,
                        start_date=start_date,
                        end_date=end_date,
                    )
                except OECDClientError:
                    logger.warning(
                        "Failed to fetch OECD series %r — skipping", spec.label
                    )
                    continue

                series_key = f"OECD:{spec.label}"
                snap = _process_series(result.data)
                if snap is None:
                    continue

                snap.series_id = series_key
                snap.label = spec.label

                series_output[series_key] = _snapshot_to_output(snap)
                prompt_lines.append(_prompt_line(snap))
                citations.append(
                    Citation(
                        source_type=CitationSourceType.OECD,
                        label=(
                            f"OECD:{spec.dataflow}/{spec.dimension_key}, "
                            f"retrieved {result.retrieved_at.date()}"
                        ),
                        url=None,
                        retrieval_date=result.retrieved_at.date(),
                    )
                )

        response = anthropic.complete(
            messages=[
                {
                    "role": "user",
                    "content": _build_user_message(thesis, prompt_lines),
                }
            ],
            model=self.model,
            task_type="macro_context",
            workflow_run_id=context.current_workflow_run_id,
            thesis_id=thesis.id,
            pod_id=getattr(thesis, "pod_id", None),
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
        )

        try:
            parsed = json.loads(response.content)
            summary_text = parsed.get("summary", response.content)
            agent_inferences = parsed.get("agent_inferences", [])
        except (json.JSONDecodeError, AttributeError):
            logger.warning("MacroContextWorkflow: LLM response was not valid JSON")
            summary_text = response.content
            agent_inferences = []

        return WorkflowResult(
            workflow_name=self.name,
            status=WorkflowStatus.COMPLETED,
            structured_output={
                "summary": summary_text,
                "series": series_output,
            },
            citations=citations,
            agent_inferences=agent_inferences,
            raw_output=response.content,
        )
