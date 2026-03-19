"""Tests for AnthropicClient — API calls, usage logging, and cost computation."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.integrations.anthropic_client import (
    AnthropicClient,
    AnthropicClientError,
    AnthropicResponse,
    _compute_cost,
    _extract_text,
)
from app.models.log import LLMUsageLog

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db() -> MagicMock:
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    return db


def _make_api_response(
    text: str = "Hello",
    input_tokens: int = 100,
    output_tokens: int = 50,
    stop_reason: str = "end_turn",
) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text

    response = MagicMock()
    response.content = [block]
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    response.stop_reason = stop_reason
    return response


def _make_client(db=None) -> tuple[AnthropicClient, MagicMock]:
    """Return (AnthropicClient, mock_sdk_client) with SDK patched."""
    db = db or _make_db()
    with patch("app.integrations.anthropic_client.anthropic.Anthropic") as mock_cls:
        mock_sdk = MagicMock()
        mock_cls.return_value = mock_sdk
        client = AnthropicClient(api_key="test-key", db=db)
    client._client = mock_sdk
    return client, mock_sdk


# ---------------------------------------------------------------------------
# _compute_cost
# ---------------------------------------------------------------------------


class TestComputeCost:
    def test_opus_cost(self):
        cost = _compute_cost("claude-opus-4-6", input_tokens=1000, output_tokens=1000)
        # 1000/1000 * 0.000015 + 1000/1000 * 0.000075
        assert cost == pytest.approx(0.000015 + 0.000075)

    def test_sonnet_cost(self):
        cost = _compute_cost("claude-sonnet-4-6", input_tokens=1000, output_tokens=1000)
        assert cost == pytest.approx(0.000003 + 0.000015)

    def test_unknown_model_returns_zero(self):
        cost = _compute_cost("unknown-model", input_tokens=1000, output_tokens=1000)
        assert cost == 0.0

    def test_zero_tokens_returns_zero(self):
        assert _compute_cost("claude-sonnet-4-6", 0, 0) == 0.0

    def test_proportional_to_token_count(self):
        cost_1k = _compute_cost("claude-sonnet-4-6", 1000, 0)
        cost_2k = _compute_cost("claude-sonnet-4-6", 2000, 0)
        assert cost_2k == pytest.approx(cost_1k * 2)


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_extracts_single_text_block(self):
        block = MagicMock()
        block.type = "text"
        block.text = "hello world"
        assert _extract_text([block]) == "hello world"

    def test_concatenates_multiple_text_blocks(self):
        b1, b2 = MagicMock(), MagicMock()
        b1.type = "text"
        b1.text = "foo"
        b2.type = "text"
        b2.text = "bar"
        assert _extract_text([b1, b2]) == "foobar"

    def test_skips_non_text_blocks(self):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "result"
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        assert _extract_text([tool_block, text_block]) == "result"

    def test_empty_content_returns_empty_string(self):
        assert _extract_text([]) == ""


# ---------------------------------------------------------------------------
# AnthropicClient instantiation
# ---------------------------------------------------------------------------


class TestAnthropicClientInit:
    def test_raises_on_empty_api_key(self):
        with pytest.raises(AnthropicClientError, match="ANTHROPIC_API_KEY"):
            AnthropicClient(api_key="", db=_make_db())

    def test_raises_on_none_api_key(self):
        with pytest.raises(AnthropicClientError, match="ANTHROPIC_API_KEY"):
            AnthropicClient(api_key=None, db=_make_db())  # type: ignore[arg-type]

    def test_valid_key_instantiates(self):
        with patch("app.integrations.anthropic_client.anthropic.Anthropic"):
            client = AnthropicClient(api_key="sk-test", db=_make_db())
        assert client is not None


# ---------------------------------------------------------------------------
# AnthropicClient.complete — happy path
# ---------------------------------------------------------------------------


class TestAnthropicClientComplete:
    def test_returns_anthropic_response(self):
        client, mock_sdk = _make_client()
        mock_sdk.messages.create.return_value = _make_api_response("The answer.")

        result = client.complete(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-sonnet-4-6",
            task_type="test",
            max_tokens=100,
        )

        assert isinstance(result, AnthropicResponse)

    def test_response_content_extracted(self):
        client, mock_sdk = _make_client()
        mock_sdk.messages.create.return_value = _make_api_response("macro summary")

        result = client.complete(
            messages=[{"role": "user", "content": "q"}],
            model="claude-sonnet-4-6",
            task_type="test",
            max_tokens=100,
        )

        assert result.content == "macro summary"

    def test_response_token_counts(self):
        client, mock_sdk = _make_client()
        mock_sdk.messages.create.return_value = _make_api_response(
            input_tokens=200, output_tokens=80
        )

        result = client.complete(
            messages=[{"role": "user", "content": "q"}],
            model="claude-sonnet-4-6",
            task_type="test",
            max_tokens=100,
        )

        assert result.input_tokens == 200
        assert result.output_tokens == 80

    def test_kwargs_passed_to_sdk(self):
        client, mock_sdk = _make_client()
        mock_sdk.messages.create.return_value = _make_api_response()

        client.complete(
            messages=[{"role": "user", "content": "q"}],
            model="claude-sonnet-4-6",
            task_type="test",
            max_tokens=512,
            system="You are helpful.",
        )

        call_kwargs = mock_sdk.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 512
        assert call_kwargs["system"] == "You are helpful."
        assert call_kwargs["model"] == "claude-sonnet-4-6"

    def test_stop_reason_passed_through(self):
        client, mock_sdk = _make_client()
        mock_sdk.messages.create.return_value = _make_api_response(
            stop_reason="max_tokens"
        )

        result = client.complete(
            messages=[{"role": "user", "content": "q"}],
            model="claude-sonnet-4-6",
            task_type="test",
            max_tokens=10,
        )

        assert result.stop_reason == "max_tokens"


# ---------------------------------------------------------------------------
# AnthropicClient.complete — usage logging
# ---------------------------------------------------------------------------


class TestAnthropicClientUsageLogging:
    def test_logs_usage_on_success(self):
        db = _make_db()
        added: list[object] = []
        db.add.side_effect = added.append
        client, mock_sdk = _make_client(db=db)
        mock_sdk.messages.create.return_value = _make_api_response(
            input_tokens=100, output_tokens=50
        )

        client.complete(
            messages=[{"role": "user", "content": "q"}],
            model="claude-sonnet-4-6",
            task_type="macro_context",
            max_tokens=100,
        )

        log_rows = [r for r in added if isinstance(r, LLMUsageLog)]
        assert len(log_rows) == 1

    def test_log_row_has_correct_token_counts(self):
        db = _make_db()
        added: list[LLMUsageLog] = []
        db.add.side_effect = added.append
        client, mock_sdk = _make_client(db=db)
        mock_sdk.messages.create.return_value = _make_api_response(
            input_tokens=300, output_tokens=120
        )

        client.complete(
            messages=[{"role": "user", "content": "q"}],
            model="claude-sonnet-4-6",
            task_type="macro_context",
            max_tokens=200,
        )

        row = next(r for r in added if isinstance(r, LLMUsageLog))
        assert row.input_tokens == 300
        assert row.output_tokens == 120

    def test_log_row_has_estimated_cost(self):
        db = _make_db()
        added: list[LLMUsageLog] = []
        db.add.side_effect = added.append
        client, mock_sdk = _make_client(db=db)
        mock_sdk.messages.create.return_value = _make_api_response(
            input_tokens=1000, output_tokens=1000
        )

        client.complete(
            messages=[{"role": "user", "content": "q"}],
            model="claude-sonnet-4-6",
            task_type="test",
            max_tokens=100,
        )

        row = next(r for r in added if isinstance(r, LLMUsageLog))
        assert row.estimated_cost_usd == pytest.approx(0.000003 + 0.000015)

    def test_log_row_records_task_type_and_model(self):
        db = _make_db()
        added: list[LLMUsageLog] = []
        db.add.side_effect = added.append
        client, mock_sdk = _make_client(db=db)
        mock_sdk.messages.create.return_value = _make_api_response()

        client.complete(
            messages=[{"role": "user", "content": "q"}],
            model="claude-opus-4-6",
            task_type="recommendation",
            max_tokens=100,
        )

        row = next(r for r in added if isinstance(r, LLMUsageLog))
        assert row.model == "claude-opus-4-6"
        assert row.task_type == "recommendation"

    def test_log_row_records_optional_fks(self):
        db = _make_db()
        added: list[LLMUsageLog] = []
        db.add.side_effect = added.append
        client, mock_sdk = _make_client(db=db)
        mock_sdk.messages.create.return_value = _make_api_response()

        thesis_id = uuid.uuid4()
        workflow_run_id = uuid.uuid4()
        pod_id = uuid.uuid4()

        client.complete(
            messages=[{"role": "user", "content": "q"}],
            model="claude-sonnet-4-6",
            task_type="test",
            thesis_id=thesis_id,
            workflow_run_id=workflow_run_id,
            pod_id=pod_id,
            max_tokens=100,
        )

        row = next(r for r in added if isinstance(r, LLMUsageLog))
        assert row.thesis_id == thesis_id
        assert row.workflow_run_id == workflow_run_id
        assert row.pod_id == pod_id

    def test_log_row_has_uuid(self):
        db = _make_db()
        added: list[LLMUsageLog] = []
        db.add.side_effect = added.append
        client, mock_sdk = _make_client(db=db)
        mock_sdk.messages.create.return_value = _make_api_response()

        client.complete(
            messages=[{"role": "user", "content": "q"}],
            model="claude-sonnet-4-6",
            task_type="test",
            max_tokens=100,
        )

        row = next(r for r in added if isinstance(r, LLMUsageLog))
        assert isinstance(row.id, uuid.UUID)

    def test_commits_after_logging(self):
        db = _make_db()
        client, mock_sdk = _make_client(db=db)
        mock_sdk.messages.create.return_value = _make_api_response()

        client.complete(
            messages=[{"role": "user", "content": "q"}],
            model="claude-sonnet-4-6",
            task_type="test",
            max_tokens=100,
        )

        db.commit.assert_called()


# ---------------------------------------------------------------------------
# AnthropicClient.complete — failure handling
# ---------------------------------------------------------------------------


class TestAnthropicClientFailureHandling:
    def test_raises_anthropic_client_error_on_api_failure(self):
        client, mock_sdk = _make_client()
        mock_sdk.messages.create.side_effect = RuntimeError("network error")

        with pytest.raises(AnthropicClientError, match="network error"):
            client.complete(
                messages=[{"role": "user", "content": "q"}],
                model="claude-sonnet-4-6",
                task_type="test",
                max_tokens=100,
            )

    def test_logs_usage_on_failure_with_zero_tokens(self):
        db = _make_db()
        added: list[object] = []
        db.add.side_effect = added.append
        client, mock_sdk = _make_client(db=db)
        mock_sdk.messages.create.side_effect = RuntimeError("boom")

        with pytest.raises(AnthropicClientError):
            client.complete(
                messages=[{"role": "user", "content": "q"}],
                model="claude-sonnet-4-6",
                task_type="macro_context",
                max_tokens=100,
            )

        log_rows = [r for r in added if isinstance(r, LLMUsageLog)]
        assert len(log_rows) == 1
        assert log_rows[0].input_tokens == 0
        assert log_rows[0].output_tokens == 0

    def test_failure_log_row_has_zero_cost(self):
        db = _make_db()
        added: list[LLMUsageLog] = []
        db.add.side_effect = added.append
        client, mock_sdk = _make_client(db=db)
        mock_sdk.messages.create.side_effect = ValueError("bad request")

        with pytest.raises(AnthropicClientError):
            client.complete(
                messages=[{"role": "user", "content": "q"}],
                model="claude-sonnet-4-6",
                task_type="test",
                max_tokens=100,
            )

        row = next(r for r in added if isinstance(r, LLMUsageLog))
        assert row.estimated_cost_usd == 0.0

    def test_original_exception_chained(self):
        client, mock_sdk = _make_client()
        original = ConnectionError("timeout")
        mock_sdk.messages.create.side_effect = original

        with pytest.raises(AnthropicClientError) as exc_info:
            client.complete(
                messages=[{"role": "user", "content": "q"}],
                model="claude-sonnet-4-6",
                task_type="test",
                max_tokens=100,
            )

        assert exc_info.value.__cause__ is original
