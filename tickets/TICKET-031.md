# TICKET 031 — Alert Email Delivery

**Section:** 7 — Kill Authority & Alerts

## Acceptance Criteria

- AlertEmailService sends email via SMTP (SendGrid or AWS SES)
- Email contains: thesis title, condition violated, triggering data point
  with citation, kill_authority setting, action taken, and direct link to
  brief
- All sent alerts logged to alerts table with delivery status and timestamp
- Failed deliveries logged and retried once
- AlertEmailService implements a pluggable delivery interface (AlertChannel
  abstract class) so future channels (Slack, SMS) can be added without
  changing core alert logic
- Tests mock SMTP and confirm email content, logging, and retry behavior
