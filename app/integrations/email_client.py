"""EmailClient — the single gateway for outbound SMTP email delivery.

Email in this system is outbound-only: recipients never reply to take action.
All actions (acknowledging alerts, responding to intake, closing trades) go
through the web UI. No workflow, service, or route may call smtplib directly.
"""

from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage

logger = logging.getLogger(__name__)


class EmailClientError(Exception):
    """Raised when sending an email fails for any reason."""


class EmailClient:
    """Thin wrapper around smtplib for outbound-only email delivery."""

    def __init__(
        self,
        host: str,
        port: int,
        username: str | None,
        password: str | None,
        from_addr: str,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._from_addr = from_addr

    def send(self, to: str, subject: str, body: str) -> None:
        """Send a plaintext email.

        Args:
            to: Destination email address.
            subject: Email subject line.
            body: Plaintext email body.

        Raises:
            EmailClientError: If the SMTP call fails for any reason.
        """
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self._from_addr
        message["To"] = to
        message.set_content(body)

        try:
            with smtplib.SMTP(self._host, self._port, timeout=10) as smtp:
                smtp.starttls()
                if self._username and self._password:
                    smtp.login(self._username, self._password)
                smtp.send_message(message)
        except Exception as exc:
            raise EmailClientError(
                f"Failed to send email to {to!r} (subject={subject!r}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        logger.info("Email sent to %s: %s", to, subject)
