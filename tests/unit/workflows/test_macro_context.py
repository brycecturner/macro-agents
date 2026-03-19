"""Tests for MacroContextWorkflow."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.integrations.anthropic_client import AnthropicResponse
from app.integrations.fred_client import FREDClientError, FREDSeriesResult
from app.integrations.oecd_client import OECDClientError, OECDSeriesResult
from app.workflows.base import CitationSourceType, WorkflowContext, WorkflowStatus
from app.workflows.macro_context import (
    _FRED_CORE_SERIES,
    MacroContextWorkflow,
    _process_series,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_series(
    values: list[float],
    start: str = "2020-01-01",
    freq: str = "ME",
) -> pd.Series:
    index = pd.date_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=index)


def _fred_result(
    series_id: str, values: list[float], start: str = "2020-01-01"
) -> FREDSeriesResult:
    return FREDSeriesResult(
        series_id=series_id,
        data=_make_series(values, start=start),
        retrieved_at=datetime(2024, 6, 1, tzinfo=UTC),
    )


def _oecd_result(
    dataset: str, subject: str, country: str, values: list[float]
) -> OECDSeriesResult:
    return OECDSeriesResult(
        dataset=dataset,
        subject=subject,
        country=country,
        data=_make_series(values),
        retrieved_at=datetime(2024, 6, 1, tzinfo=UTC),
    )


def _make_anthropic_response(
    summary: str = "Macro backdrop is restrictive.",
    inferences: list[str] | None = None,
) -> AnthropicResponse:
    inferences = inferences or ["[Agent inference] Yield curve likely to steepen."]
    content = json.dumps({"summary": summary, "agent_inferences": inferences})
    return AnthropicResponse(
        content=content,
        model="claude-sonnet-4-6",
        input_tokens=200,
        output_tokens=100,
        stop_reason="end_turn",
    )


def _make_fred_client(series_values: dict[str, list[float]] | None = None) -> MagicMock:
    """Build a mock FREDClient that returns sensible data for each core series."""
    defaults = {s: [1.0] * 25 for s in _FRED_CORE_SERIES}
    if series_values:
        defaults.update(series_values)

    mock = MagicMock()
    mock.get_series.side_effect = lambda series_id, **_: _fred_result(
        series_id, defaults.get(series_id, [1.0] * 25)
    )
    return mock


def _make_anthropic_client(response: AnthropicResponse | None = None) -> MagicMock:
    mock = MagicMock()
    mock.complete.return_value = response or _make_anthropic_response()
    return mock


def _make_context() -> WorkflowContext:
    thesis = _make_thesis()
    return WorkflowContext(thesis=thesis, db=MagicMock())


# ---------------------------------------------------------------------------
# _process_series
# ---------------------------------------------------------------------------


class TestProcessSeries:
    def test_returns_none_for_empty_series(self):
        assert _process_series(pd.Series([], dtype=float)) is None

    def test_returns_none_for_all_nan(self):
        s = pd.Series([float("nan"), float("nan")])
        s.index = pd.date_range("2024-01-01", periods=2, freq="ME")
        assert _process_series(s) is None

    def test_current_value_is_latest(self):
        s = _make_series([1.0, 2.0, 3.0])
        snap = _process_series(s)
        assert snap.current_value == pytest.approx(3.0)

    def test_current_date_is_latest_date(self):
        s = _make_series([1.0, 2.0], start="2024-01-31", freq="ME")
        snap = _process_series(s)
        assert snap.current_date == date(2024, 2, 29)

    def test_yoy_change_computed(self):
        # 25 months: year-ago value at index ~12, latest at index ~24
        values = [float(i) for i in range(25)]
        s = _make_series(values, start="2022-01-31", freq="ME")
        snap = _process_series(s)
        assert snap.yoy_change is not None
        assert snap.year_ago_value is not None

    def test_historical_data_is_list_of_dicts(self):
        s = _make_series([1.0, 2.0, 3.0])
        snap = _process_series(s)
        assert isinstance(snap.historical_data, list)
        assert all("date" in d and "value" in d for d in snap.historical_data)

    def test_historical_data_covers_full_range(self):
        values = [float(i) for i in range(25)]
        s = _make_series(values)
        snap = _process_series(s)
        assert len(snap.historical_data) == 25

    def test_recent_prompt_data_capped_at_six_months(self):
        values = [float(i) for i in range(25)]
        s = _make_series(values)
        snap = _process_series(s)
        assert len(snap.recent_prompt_data) <= 6


# ---------------------------------------------------------------------------
# MacroContextWorkflow — FRED series fetching
# ---------------------------------------------------------------------------


class TestMacroContextWorkflowFREDFetching:
    def test_fetches_all_four_core_series(self):
        fred = _make_fred_client()
        workflow = MacroContextWorkflow(
            fred_client=fred,
            anthropic_client=_make_anthropic_client(),
        )
        workflow.execute(_make_thesis(), _make_context())

        fetched = {call.args[0] for call in fred.get_series.call_args_list}
        assert fetched == {"T10Y2Y", "CPIAUCSL", "FEDFUNDS", "UNRATE"}

    def test_structured_output_contains_all_core_series(self):
        workflow = MacroContextWorkflow(
            fred_client=_make_fred_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())

        for series_id in _FRED_CORE_SERIES:
            assert series_id in result.structured_output["series"]

    def test_series_output_has_required_fields(self):
        workflow = MacroContextWorkflow(
            fred_client=_make_fred_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())

        series_data = result.structured_output["series"]["T10Y2Y"]
        assert "current_value" in series_data
        assert "current_date" in series_data
        assert "historical_data" in series_data

    def test_historical_data_is_list(self):
        workflow = MacroContextWorkflow(
            fred_client=_make_fred_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())

        assert isinstance(
            result.structured_output["series"]["T10Y2Y"]["historical_data"], list
        )

    def test_continues_when_one_series_fails(self):
        fred = MagicMock()

        def side_effect(series_id, **_):
            if series_id == "T10Y2Y":
                raise FREDClientError("unavailable")
            return _fred_result(series_id, [1.0] * 25)

        fred.get_series.side_effect = side_effect

        workflow = MacroContextWorkflow(
            fred_client=fred,
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())

        # Workflow completes; failed series absent from output
        assert result.status == WorkflowStatus.COMPLETED
        assert "T10Y2Y" not in result.structured_output["series"]
        assert "CPIAUCSL" in result.structured_output["series"]

    def test_uses_thirty_year_lookback(self):
        fred = _make_fred_client()
        workflow = MacroContextWorkflow(
            fred_client=fred,
            anthropic_client=_make_anthropic_client(),
        )
        workflow.execute(_make_thesis(), _make_context())

        _, call_kwargs = fred.get_series.call_args
        start = call_kwargs.get("start_date") or fred.get_series.call_args_list[0][
            1
        ].get("start_date")
        assert start is not None
        assert (date.today() - start).days >= 365 * 29  # at least 29 years back


# ---------------------------------------------------------------------------
# MacroContextWorkflow — citations
# ---------------------------------------------------------------------------


class TestMacroContextWorkflowCitations:
    def test_one_citation_per_fetched_series(self):
        workflow = MacroContextWorkflow(
            fred_client=_make_fred_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())

        assert len(result.citations) == len(_FRED_CORE_SERIES)

    def test_citation_label_format(self):
        workflow = MacroContextWorkflow(
            fred_client=_make_fred_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())

        labels = [c.label for c in result.citations]
        assert any("FRED:T10Y2Y" in label for label in labels)
        assert any("retrieved" in label for label in labels)

    def test_citation_source_type_is_fred(self):
        workflow = MacroContextWorkflow(
            fred_client=_make_fred_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())

        for citation in result.citations:
            assert citation.source_type == CitationSourceType.FRED

    def test_no_citation_for_failed_series(self):
        fred = MagicMock()

        def side_effect(series_id, **_):
            if series_id == "FEDFUNDS":
                raise FREDClientError("down")
            return _fred_result(series_id, [1.0] * 25)

        fred.get_series.side_effect = side_effect

        workflow = MacroContextWorkflow(
            fred_client=fred,
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())

        citation_labels = [c.label for c in result.citations]
        assert not any("FEDFUNDS" in label for label in citation_labels)


# ---------------------------------------------------------------------------
# MacroContextWorkflow — OECD integration
# ---------------------------------------------------------------------------


class TestMacroContextWorkflowOECD:
    def test_oecd_not_called_when_client_not_provided(self):
        workflow = MacroContextWorkflow(
            fred_client=_make_fred_client(),
            anthropic_client=_make_anthropic_client(),
            oecd_client=None,
        )
        result = workflow.execute(_make_thesis(), _make_context())

        oecd_keys = [
            k for k in result.structured_output["series"] if k.startswith("OECD:")
        ]
        assert oecd_keys == []

    def test_oecd_series_included_when_client_provided(self):
        oecd = MagicMock()
        oecd.get_series.return_value = _oecd_result(
            "MEI_FIN", "IRSTCB01", "EA19", [1.0] * 25
        )

        workflow = MacroContextWorkflow(
            fred_client=_make_fred_client(),
            anthropic_client=_make_anthropic_client(),
            oecd_client=oecd,
        )
        result = workflow.execute(_make_thesis(), _make_context())

        oecd_keys = [
            k for k in result.structured_output["series"] if k.startswith("OECD:")
        ]
        assert len(oecd_keys) > 0

    def test_oecd_citations_have_oecd_source_type(self):
        oecd = MagicMock()
        oecd.get_series.return_value = _oecd_result(
            "MEI_FIN", "IRSTCB01", "EA19", [1.0] * 25
        )

        workflow = MacroContextWorkflow(
            fred_client=_make_fred_client(),
            anthropic_client=_make_anthropic_client(),
            oecd_client=oecd,
        )
        result = workflow.execute(_make_thesis(), _make_context())

        oecd_citations = [
            c for c in result.citations if c.source_type == CitationSourceType.OECD
        ]
        assert len(oecd_citations) > 0

    def test_oecd_failure_does_not_fail_workflow(self):
        oecd = MagicMock()
        oecd.get_series.side_effect = OECDClientError("timeout")

        workflow = MacroContextWorkflow(
            fred_client=_make_fred_client(),
            anthropic_client=_make_anthropic_client(),
            oecd_client=oecd,
        )
        result = workflow.execute(_make_thesis(), _make_context())

        assert result.status == WorkflowStatus.COMPLETED


# ---------------------------------------------------------------------------
# MacroContextWorkflow — LLM call and structured output
# ---------------------------------------------------------------------------


class TestMacroContextWorkflowLLM:
    def test_result_status_is_completed(self):
        workflow = MacroContextWorkflow(
            fred_client=_make_fred_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.status == WorkflowStatus.COMPLETED

    def test_structured_output_has_summary(self):
        workflow = MacroContextWorkflow(
            fred_client=_make_fred_client(),
            anthropic_client=_make_anthropic_client(
                _make_anthropic_response("Tight conditions.")
            ),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.structured_output["summary"] == "Tight conditions."

    def test_agent_inferences_extracted(self):
        response = _make_anthropic_response(
            inferences=["[Agent inference] Steepening likely."]
        )
        workflow = MacroContextWorkflow(
            fred_client=_make_fred_client(),
            anthropic_client=_make_anthropic_client(response),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert "[Agent inference] Steepening likely." in result.agent_inferences

    def test_fallback_on_invalid_json_response(self):
        bad_response = AnthropicResponse(
            content="not json at all",
            model="claude-sonnet-4-6",
            input_tokens=10,
            output_tokens=5,
            stop_reason="end_turn",
        )
        workflow = MacroContextWorkflow(
            fred_client=_make_fred_client(),
            anthropic_client=_make_anthropic_client(bad_response),
        )
        result = workflow.execute(_make_thesis(), _make_context())

        assert result.status == WorkflowStatus.COMPLETED
        assert result.structured_output["summary"] == "not json at all"
        assert result.agent_inferences == []

    def test_llm_called_with_correct_task_type(self):
        anthropic = _make_anthropic_client()
        workflow = MacroContextWorkflow(
            fred_client=_make_fred_client(),
            anthropic_client=anthropic,
        )
        workflow.execute(_make_thesis(), _make_context())

        call_kwargs = anthropic.complete.call_args[1]
        assert call_kwargs["task_type"] == "macro_context"

    def test_llm_called_with_configured_model(self):
        anthropic = _make_anthropic_client()
        workflow = MacroContextWorkflow(
            fred_client=_make_fred_client(),
            anthropic_client=anthropic,
        )
        workflow.execute(_make_thesis(), _make_context())

        call_kwargs = anthropic.complete.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"

    def test_workflow_run_id_passed_to_llm(self):
        anthropic = _make_anthropic_client()
        workflow = MacroContextWorkflow(
            fred_client=_make_fred_client(),
            anthropic_client=anthropic,
        )
        run_id = uuid.uuid4()
        context = _make_context()
        context.current_workflow_run_id = run_id

        workflow.execute(_make_thesis(), context)

        call_kwargs = anthropic.complete.call_args[1]
        assert call_kwargs["workflow_run_id"] == run_id

    def test_raw_output_contains_llm_response(self):
        response = _make_anthropic_response("Macro is tight.")
        workflow = MacroContextWorkflow(
            fred_client=_make_fred_client(),
            anthropic_client=_make_anthropic_client(response),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert "Macro is tight." in result.raw_output

    def test_workflow_name_is_correct(self):
        workflow = MacroContextWorkflow(
            fred_client=_make_fred_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.workflow_name == "MacroContextWorkflow"
