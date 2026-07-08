"""Tests for ThesisDecisionService — human Go/No-Go/Hold decision recording."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from app.models.enums import ThesisStatus
from app.models.log import AuditLog
from app.services.thesis_decision_service import InvalidDecisionError, record_decision


def _make_thesis(**kwargs) -> MagicMock:
    t = MagicMock()
    t.id = kwargs.get("id", uuid.uuid4())
    t.pod_id = kwargs.get("pod_id", uuid.uuid4())
    t.status = kwargs.get("status", ThesisStatus.researched)
    return t


class TestRecordDecision:
    def test_go_sets_status_approved(self):
        thesis = _make_thesis()
        db = MagicMock()
        record_decision(thesis, "go", db)
        assert thesis.status == ThesisStatus.approved

    def test_no_go_sets_status_rejected(self):
        thesis = _make_thesis()
        db = MagicMock()
        record_decision(thesis, "no_go", db)
        assert thesis.status == ThesisStatus.rejected

    def test_hold_leaves_status_unchanged(self):
        thesis = _make_thesis(status=ThesisStatus.researched)
        db = MagicMock()
        record_decision(thesis, "hold", db)
        assert thesis.status == ThesisStatus.researched

    def test_invalid_decision_raises(self):
        thesis = _make_thesis()
        db = MagicMock()
        with pytest.raises(InvalidDecisionError):
            record_decision(thesis, "maybe", db)

    def test_invalid_decision_does_not_change_status(self):
        thesis = _make_thesis(status=ThesisStatus.researched)
        db = MagicMock()
        with pytest.raises(InvalidDecisionError):
            record_decision(thesis, "maybe", db)
        assert thesis.status == ThesisStatus.researched

    def test_writes_audit_log_entry_for_go(self):
        thesis = _make_thesis()
        db = MagicMock()
        added = []
        db.add.side_effect = added.append
        record_decision(thesis, "go", db)

        audit_rows = [a for a in added if isinstance(a, AuditLog)]
        assert len(audit_rows) == 1
        assert audit_rows[0].entity_type == "thesis"
        assert audit_rows[0].entity_id == thesis.id
        assert audit_rows[0].pod_id == thesis.pod_id
        assert audit_rows[0].action == "thesis_decision"
        assert audit_rows[0].previous_value == {"status": "researched"}
        assert audit_rows[0].new_value == {"status": "approved", "decision": "go"}
        assert audit_rows[0].changed_by == "user"

    def test_writes_audit_log_entry_for_hold(self):
        thesis = _make_thesis()
        db = MagicMock()
        added = []
        db.add.side_effect = added.append
        record_decision(thesis, "hold", db)

        audit_row = next(a for a in added if isinstance(a, AuditLog))
        assert audit_row.previous_value == {"status": "researched"}
        assert audit_row.new_value == {"status": "researched", "decision": "hold"}

    def test_does_not_commit(self):
        thesis = _make_thesis()
        db = MagicMock()
        record_decision(thesis, "go", db)
        db.commit.assert_not_called()
