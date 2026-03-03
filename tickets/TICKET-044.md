# TICKET 044 — End-to-End Integration Test

**Section:** 11 — System Hardening

## Acceptance Criteria

- A full integration test runs the complete idea pipeline from submission
  to brief generation using mocked external APIs
- Test confirms: intake generated, research workflows run in order,
  brief assembled with all required fields, further reading populated,
  falsification conditions stored
- A separate integration test runs the daily monitoring job and confirms
  a falsified condition triggers an alert
- A separate integration test runs the weekly rebalance and confirms orders
  are submitted and logged correctly
