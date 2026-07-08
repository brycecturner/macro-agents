"""Tests for EmailClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.integrations.email_client import EmailClient, EmailClientError


@pytest.fixture
def mock_smtp():
    with patch("app.integrations.email_client.smtplib.SMTP") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value.__enter__.return_value = instance
        yield mock_cls, instance


@pytest.fixture
def client() -> EmailClient:
    return EmailClient(
        host="smtp.example.com",
        port=587,
        username="user@example.com",
        password="secret",
        from_addr="noreply@example.com",
    )


class TestEmailClientSend:
    def test_connects_to_configured_host_and_port(self, client, mock_smtp):
        mock_cls, _instance = mock_smtp
        client.send(to="pm@example.com", subject="Subject", body="Body")
        mock_cls.assert_called_once_with("smtp.example.com", 587, timeout=10)

    def test_starts_tls(self, client, mock_smtp):
        _mock_cls, instance = mock_smtp
        client.send(to="pm@example.com", subject="Subject", body="Body")
        instance.starttls.assert_called_once()

    def test_logs_in_with_credentials(self, client, mock_smtp):
        _mock_cls, instance = mock_smtp
        client.send(to="pm@example.com", subject="Subject", body="Body")
        instance.login.assert_called_once_with("user@example.com", "secret")

    def test_skips_login_when_no_credentials(self, mock_smtp):
        _mock_cls, instance = mock_smtp
        no_auth_client = EmailClient(
            host="smtp.example.com",
            port=587,
            username=None,
            password=None,
            from_addr="noreply@example.com",
        )
        no_auth_client.send(to="pm@example.com", subject="Subject", body="Body")
        instance.login.assert_not_called()

    def test_sends_message_with_correct_headers(self, client, mock_smtp):
        _mock_cls, instance = mock_smtp
        client.send(to="pm@example.com", subject="Brief ready", body="Body text")

        sent_message = instance.send_message.call_args[0][0]
        assert sent_message["To"] == "pm@example.com"
        assert sent_message["From"] == "noreply@example.com"
        assert sent_message["Subject"] == "Brief ready"

    def test_body_included_in_message(self, client, mock_smtp):
        _mock_cls, instance = mock_smtp
        client.send(to="pm@example.com", subject="Subject", body="Hello world")

        sent_message = instance.send_message.call_args[0][0]
        assert "Hello world" in sent_message.get_content()

    def test_raises_email_client_error_on_smtp_failure(self, client, mock_smtp):
        _mock_cls, instance = mock_smtp
        instance.send_message.side_effect = RuntimeError("connection refused")

        with pytest.raises(EmailClientError, match="connection refused"):
            client.send(to="pm@example.com", subject="Subject", body="Body")

    def test_raises_email_client_error_on_login_failure(self, client, mock_smtp):
        _mock_cls, instance = mock_smtp
        instance.login.side_effect = RuntimeError("bad credentials")

        with pytest.raises(EmailClientError, match="bad credentials"):
            client.send(to="pm@example.com", subject="Subject", body="Body")
