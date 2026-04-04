"""Tests for HistoricalAnalogWorkflow."""

from __future__ import annotations

import json
import uuid
from datetime import date
from unittest.mock import MagicMock

import pandas as pd

from app.integrations.anthropic_client import AnthropicResponse
from app.workflows.base import (
    Citation,
    CitationSourceType,
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
)
from app.workflows.historical_analog import (
    HistoricalAnalogWorkflow,
    _build_history_table,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRED_SERIES = ["T10Y2Y", "CPIAUCSL", "FEDFUNDS", "UNRATE"]


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


def _make_series_data(n: int = 36) -> dict[str, dict]:
    """Build a minimal series_data dict matching MacroContextWorkflow output shape."""
    index = pd.date_range("2021-01-31", periods=n, freq="ME")
    result = {}
    for i, sid in enumerate(_FRED_SERIES):
        values = [float(i + j * 0.1) for j in range(n)]
        result[sid] = {
            "label": sid,
            "current_value": values[-1],
            "current_date": index[-1].date().isoformat(),
            "year_ago_value": values[-13] if n >= 13 else None,
            "yoy_change": None,
            "historical_data": [
                {"date": idx.date().isoformat(), "value": v}
                for idx, v in zip(index, values, strict=True)
            ],
        }
    return result


def _make_macro_result(
    series_data: dict | None = None,
    summary: str = "Macro is restrictive.",
) -> WorkflowResult:
    """Build a fake MacroContextWorkflow WorkflowResult for context injection."""
    sd = series_data if series_data is not None else _make_series_data()
    citations = [
        Citation(
            source_type=CitationSourceType.FRED,
            label=f"FRED:{sid}, retrieved {date(2024, 6, 1)}",
            url=None,
            retrieval_date=date(2024, 6, 1),
        )
        for sid in _FRED_SERIES
    ]
    return WorkflowResult(
        workflow_name="MacroContextWorkflow",
        status=WorkflowStatus.COMPLETED,
        structured_output={"summary": summary, "series": sd},
        citations=citations,
        agent_inferences=[],
        raw_output="{}",
    )


def _make_context(with_macro_result: bool = True) -> WorkflowContext:
    thesis = _make_thesis()
    ctx = WorkflowContext(thesis=thesis, db=MagicMock())
    if with_macro_result:
        ctx.prior_results.append(_make_macro_result())
    return ctx


def _make_analog_response(analogs: list[dict] | None = None) -> AnthropicResponse:
    if analogs is None:
        analogs = [
            {
                "start_date": "2006-06",
                "end_date": "2007-06",
                "label": "2006 Fed pause",
                "macro_conditions": {"T10Y2Y": -0.1, "FEDFUNDS": 5.25, "UNRATE": 4.6},
                "similarity_rationale": "[Agent inference] Similar inversion depth.",
                "outcome_summary": "[Agent inference] Curve steepened into recession.",
            },
            {
                "start_date": "2018-10",
                "end_date": "2019-08",
                "label": "2018 late-cycle tightening",
                "macro_conditions": {"T10Y2Y": 0.1, "FEDFUNDS": 2.25, "UNRATE": 3.7},
                "similarity_rationale": "[Agent inference] Fed near peak, labor tight.",
                "outcome_summary": "[Agent inference] Pivot followed Fed pause.",
            },
        ]
    content = json.dumps({"analogs": analogs, "agent_inferences": []})
    return AnthropicResponse(
        content=content,
        model="claude-sonnet-4-6",
        input_tokens=500,
        output_tokens=300,
        stop_reason="end_turn",
    )


def _make_anthropic_client(response: AnthropicResponse | None = None) -> MagicMock:
    mock = MagicMock()
    mock.complete.return_value = response or _make_analog_response()
    return mock


# ---------------------------------------------------------------------------
# _build_history_table
# ---------------------------------------------------------------------------


class TestBuildHistoryTable:
    def test_returns_string(self):
        sd = _make_series_data()
        result = _build_history_table(sd)
        assert isinstance(result, str)

    def test_empty_series_data_returns_placeholder(self):
        result = _build_history_table({})
        assert "no historical data" in result

    def test_cpiaucsl_converted_to_yoy(self):
        sd = _make_series_data(n=36)
        result = _build_history_table(sd)
        assert "CPI_YoY%" in result
        assert "CPIAUCSL" not in result

    def test_all_series_present_in_output(self):
        sd = _make_series_data()
        result = _build_history_table(sd)
        # T10Y2Y, FEDFUNDS, UNRATE should be column headers
        assert "T10Y2Y" in result
        assert "FEDFUNDS" in result
        assert "UNRATE" in result

    def test_rows_sorted_chronologically(self):
        sd = _make_series_data(n=12)
        csv = _build_history_table(sd)
        lines = [row for row in csv.strip().splitlines() if row.strip()]
        # First data line should be earlier than last
        first_date = lines[1].split(",")[0]
        last_date = lines[-1].split(",")[0]
        assert first_date < last_date

    def test_oecd_series_included_with_cleaned_name(self):
        sd = _make_series_data()
        sd["OECD:ECB Policy Rate (Short-term Call Rate)"] = {
            "label": "ECB Policy Rate",
            "current_value": 3.5,
            "current_date": "2024-06-30",
            "year_ago_value": 2.0,
            "yoy_change": 1.5,
            "historical_data": [
                {"date": "2024-01-31", "value": 3.5},
                {"date": "2024-02-29", "value": 3.5},
            ],
        }
        result = _build_history_table(sd)
        # OECD: prefix stripped; should appear without the prefix
        assert "OECD:" not in result

    def test_series_with_no_historical_data_skipped(self):
        sd = _make_series_data()
        sd["EMPTY_SERIES"] = {
            "label": "Empty",
            "current_value": 0.0,
            "current_date": "2024-06-30",
            "year_ago_value": None,
            "yoy_change": None,
            "historical_data": [],
        }
        # Should not raise
        result = _build_history_table(sd)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# HistoricalAnalogWorkflow — context handling
# ---------------------------------------------------------------------------


class TestHistoricalAnalogWorkflowContext:
    def test_runs_without_macro_result_in_context(self):
        """Workflow must not raise when MacroContextWorkflow hasn't run."""
        workflow = HistoricalAnalogWorkflow(anthropic_client=_make_anthropic_client())
        ctx = _make_context(with_macro_result=False)
        result = workflow.execute(_make_thesis(), ctx)
        assert result.status == WorkflowStatus.COMPLETED

    def test_flags_missing_context_in_agent_inferences(self):
        workflow = HistoricalAnalogWorkflow(anthropic_client=_make_anthropic_client())
        ctx = _make_context(with_macro_result=False)
        result = workflow.execute(_make_thesis(), ctx)

        combined = " ".join(result.agent_inferences)
        assert "MacroContextWorkflow" in combined
        assert "[Agent inference]" in combined

    def test_consumes_series_data_from_context(self):
        """LLM prompt must include historical data when context is available."""
        anthropic = _make_anthropic_client()
        workflow = HistoricalAnalogWorkflow(anthropic_client=anthropic)
        workflow.execute(_make_thesis(), _make_context())

        call_content = anthropic.complete.call_args[1]["messages"][0]["content"]
        # Historical data table should be present
        assert "Historical macro data" in call_content

    def test_macro_summary_included_in_prompt(self):
        anthropic = _make_anthropic_client()
        workflow = HistoricalAnalogWorkflow(anthropic_client=anthropic)
        ctx = WorkflowContext(thesis=_make_thesis(), db=MagicMock())
        ctx.prior_results.append(_make_macro_result(summary="Conditions are tight."))

        workflow.execute(_make_thesis(), ctx)

        call_content = anthropic.complete.call_args[1]["messages"][0]["content"]
        assert "Conditions are tight." in call_content


# ---------------------------------------------------------------------------
# HistoricalAnalogWorkflow — structured output
# ---------------------------------------------------------------------------


class TestHistoricalAnalogWorkflowOutput:
    def test_result_status_is_completed(self):
        workflow = HistoricalAnalogWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.status == WorkflowStatus.COMPLETED

    def test_structured_output_has_analogs_key(self):
        workflow = HistoricalAnalogWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_context())
        assert "analogs" in result.structured_output

    def test_analogs_is_a_list(self):
        workflow = HistoricalAnalogWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_context())
        assert isinstance(result.structured_output["analogs"], list)

    def test_analog_has_required_fields(self):
        workflow = HistoricalAnalogWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_context())
        analog = result.structured_output["analogs"][0]

        assert "start_date" in analog
        assert "end_date" in analog
        assert "label" in analog
        assert "similarity_rationale" in analog
        assert "outcome_summary" in analog

    def test_similarity_rationale_flagged_as_inference(self):
        workflow = HistoricalAnalogWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_context())
        for analog in result.structured_output["analogs"]:
            assert "[Agent inference]" in analog["similarity_rationale"]

    def test_outcome_summary_flagged_as_inference(self):
        workflow = HistoricalAnalogWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_context())
        for analog in result.structured_output["analogs"]:
            assert "[Agent inference]" in analog["outcome_summary"]

    def test_workflow_name_is_correct(self):
        workflow = HistoricalAnalogWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.workflow_name == "HistoricalAnalogWorkflow"

    def test_agent_inferences_extracted_from_llm_response(self):
        response = AnthropicResponse(
            content=json.dumps(
                {
                    "analogs": [],
                    "agent_inferences": [
                        "[Agent inference] Only one weak analog found."
                    ],
                }
            ),
            model="claude-sonnet-4-6",
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
        )
        workflow = HistoricalAnalogWorkflow(
            anthropic_client=_make_anthropic_client(response)
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert any("Only one weak analog" in inf for inf in result.agent_inferences)

    def test_fallback_on_invalid_json_response(self):
        bad = AnthropicResponse(
            content="not json",
            model="claude-sonnet-4-6",
            input_tokens=10,
            output_tokens=5,
            stop_reason="end_turn",
        )
        workflow = HistoricalAnalogWorkflow(
            anthropic_client=_make_anthropic_client(bad)
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.status == WorkflowStatus.COMPLETED
        assert result.structured_output["analogs"] == []

    def test_raw_output_contains_llm_response(self):
        response = _make_analog_response()
        workflow = HistoricalAnalogWorkflow(
            anthropic_client=_make_anthropic_client(response)
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert "2006" in result.raw_output


# ---------------------------------------------------------------------------
# HistoricalAnalogWorkflow — citations
# ---------------------------------------------------------------------------


class TestHistoricalAnalogWorkflowCitations:
    def test_one_citation_per_fred_series(self):
        workflow = HistoricalAnalogWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_context())
        assert len(result.citations) == len(_FRED_SERIES)

    def test_citation_source_type_is_fred(self):
        workflow = HistoricalAnalogWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_context())
        for c in result.citations:
            assert c.source_type == CitationSourceType.FRED

    def test_citation_label_format(self):
        workflow = HistoricalAnalogWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_context())
        labels = [c.label for c in result.citations]
        assert any("FRED:T10Y2Y" in label for label in labels)
        assert any("retrieved" in label for label in labels)

    def test_retrieval_date_matches_macro_result_citations(self):
        """Retrieval dates should come from MacroContextWorkflow citations."""
        workflow = HistoricalAnalogWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_context())
        for c in result.citations:
            assert c.retrieval_date == date(2024, 6, 1)

    def test_no_citations_when_no_context(self):
        workflow = HistoricalAnalogWorkflow(anthropic_client=_make_anthropic_client())
        ctx = _make_context(with_macro_result=False)
        result = workflow.execute(_make_thesis(), ctx)
        assert result.citations == []

    def test_oecd_series_not_cited_as_fred(self):
        sd = _make_series_data()
        sd["OECD:ECB Policy Rate (Short-term Call Rate)"] = {
            "label": "ECB Policy Rate",
            "current_value": 3.5,
            "current_date": "2024-06-30",
            "year_ago_value": None,
            "yoy_change": None,
            "historical_data": [{"date": "2024-01-31", "value": 3.5}],
        }
        workflow = HistoricalAnalogWorkflow(anthropic_client=_make_anthropic_client())
        ctx = WorkflowContext(thesis=_make_thesis(), db=MagicMock())
        ctx.prior_results.append(_make_macro_result(series_data=sd))
        result = workflow.execute(_make_thesis(), ctx)

        fred_labels = [
            c.label
            for c in result.citations
            if c.source_type == CitationSourceType.FRED
        ]
        assert not any("OECD:" in label for label in fred_labels)


# ---------------------------------------------------------------------------
# HistoricalAnalogWorkflow — LLM call
# ---------------------------------------------------------------------------


class TestHistoricalAnalogWorkflowLLM:
    def test_llm_called_with_correct_task_type(self):
        anthropic = _make_anthropic_client()
        workflow = HistoricalAnalogWorkflow(anthropic_client=anthropic)
        workflow.execute(_make_thesis(), _make_context())

        call_kwargs = anthropic.complete.call_args[1]
        assert call_kwargs["task_type"] == "historical_analog"

    def test_llm_called_with_configured_model(self):
        anthropic = _make_anthropic_client()
        workflow = HistoricalAnalogWorkflow(anthropic_client=anthropic)
        workflow.execute(_make_thesis(), _make_context())

        call_kwargs = anthropic.complete.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"

    def test_workflow_run_id_passed_to_llm(self):
        anthropic = _make_anthropic_client()
        workflow = HistoricalAnalogWorkflow(anthropic_client=anthropic)
        run_id = uuid.uuid4()
        ctx = _make_context()
        ctx.current_workflow_run_id = run_id

        workflow.execute(_make_thesis(), ctx)

        call_kwargs = anthropic.complete.call_args[1]
        assert call_kwargs["workflow_run_id"] == run_id

    def test_thesis_title_in_prompt(self):
        anthropic = _make_anthropic_client()
        workflow = HistoricalAnalogWorkflow(anthropic_client=anthropic)
        thesis = _make_thesis("Inflation Breakout Trade")
        workflow.execute(thesis, _make_context())

        content = anthropic.complete.call_args[1]["messages"][0]["content"]
        assert "Inflation Breakout Trade" in content
