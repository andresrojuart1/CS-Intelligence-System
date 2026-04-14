"""
Tests for tech_touch_agent.py
-------------------------------
All OpenAI calls are mocked — no real API keys required.

Run with:  pytest tests/test_tech_touch_agent.py -v
"""

import json
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agents.tech_touch_agent import run_tech_touch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_openai_mock(subject: str, body: str) -> MagicMock:
    """
    Builds a mock that mimics the openai.OpenAI() client interface.

    Patches openai.OpenAI() so that calling .chat.completions.create(...)
    returns a fake response with the given subject and body as JSON content.
    The same mock works for both json_object and plain-text response modes.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    # Mimic response.choices[0].message.content
    mock_response.choices[0].message.content = json.dumps(
        {"subject": subject, "body": body}
    )
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# Account used for check-in test (14+ days no contact — YELLOW lane)
ACCOUNT_CHECK_IN = {
    "account_id": "ACC-010",
    "company_name": "NovaBuild SA",
    "client_name": "Sofia Torres",
    "csm_name": "Andrés",
    "template_type": "check_in",
    "lane": "YELLOW",
    "context": {
        "days_since_last_contact": 18,
        "usage_trend": "active contractors dropped from 12 to 8 over the past month",
    },
}

# Account used for re-engagement test (40% usage drop — YELLOW lane)
ACCOUNT_RE_ENGAGEMENT = {
    "account_id": "ACC-011",
    "company_name": "Pacifico Logistics",
    "client_name": "Carlos Mendez",
    "csm_name": "Andrés",
    "template_type": "re_engagement",
    "lane": "YELLOW",
    "context": {
        "previous_contractors": 25,
        "active_contractors": 15,
        "usage_drop_pct": 40,
    },
}

# RED account — should be rejected by the agent
ACCOUNT_RED = {
    "account_id": "ACC-001",
    "company_name": "ChurnRisk Corp",
    "client_name": "James Wu",
    "csm_name": "Andrés",
    "template_type": "check_in",
    "lane": "RED",
    "context": {
        "days_since_last_contact": 30,
        "usage_trend": "no logins in 30 days",
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTechTouchAgent:

    def test_check_in_email_yellow_account(self, tmp_path):
        """
        A YELLOW account with 14+ days no contact should generate a
        check-in email and write a pending_approval record to disk.
        """
        approvals_file = tmp_path / "approvals.json"
        mock_subject = "Checking in on your Ontop setup, Sofia"
        mock_body = (
            "Hi Sofia, I noticed your active contractor count shifted recently "
            "and wanted to reach out. Happy to chat through any changes in your "
            "team's workflow — would a quick 15-min call work this week?"
        )

        with patch("agents.tech_touch_agent.openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = _make_openai_mock(mock_subject, mock_body)
            result = run_tech_touch(ACCOUNT_CHECK_IN, approvals_file=approvals_file)

        # Generated content
        assert result["generated_subject"] == mock_subject
        assert result["generated_body"] == mock_body
        assert result["error"] == ""

        # Approval record structure
        record = result["approval_record"]
        assert record["account_id"] == "ACC-010"
        assert record["template_type"] == "check_in"
        assert record["status"] == "pending_approval"
        assert "generated_at" in record

        # Persisted to disk
        assert approvals_file.exists()
        saved = json.loads(approvals_file.read_text())
        assert len(saved) == 1
        assert saved[0]["account_id"] == "ACC-010"
        assert saved[0]["status"] == "pending_approval"

    def test_re_engagement_email_usage_drop(self, tmp_path):
        """
        A YELLOW account with a 40% usage drop should generate a
        re-engagement email referencing the contractor count change.
        """
        approvals_file = tmp_path / "approvals.json"
        mock_subject = "A feature that might help your team, Carlos"
        mock_body = (
            "Hi Carlos, your team has gone from 25 to 15 active contractors — "
            "curious if there's a project shift we can support. "
            "We recently shipped bulk onboarding which could help if you scale back up. "
            "Would a quick call be useful?"
        )

        with patch("agents.tech_touch_agent.openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = _make_openai_mock(mock_subject, mock_body)
            result = run_tech_touch(ACCOUNT_RE_ENGAGEMENT, approvals_file=approvals_file)

        assert result["lane"] == "YELLOW"
        assert result["generated_subject"] == mock_subject
        assert result["generated_body"] == mock_body
        assert result["error"] == ""
        assert result["approval_record"]["template_type"] == "re_engagement"

    def test_red_account_is_rejected(self, tmp_path):
        """
        A RED-lane account must be rejected without generating any message.
        No approval record should be written to disk.
        RED accounts require direct CSM engagement — not automated outreach.
        """
        approvals_file = tmp_path / "approvals.json"

        # No OpenAI mock needed — the agent should exit before calling the LLM
        result = run_tech_touch(ACCOUNT_RED, approvals_file=approvals_file)

        # Error is set, no content generated
        assert result["error"] != ""
        assert "RED" in result["error"]
        assert result["generated_subject"] == ""
        assert result["generated_body"] == ""
        assert result["approval_record"] == {}

        # Nothing written to disk
        assert not approvals_file.exists()

    def test_green_account_is_rejected(self, tmp_path):
        """GREEN accounts don't need outreach — agent should reject them too."""
        approvals_file = tmp_path / "approvals.json"
        green_account = {**ACCOUNT_CHECK_IN, "account_id": "ACC-GREEN", "lane": "GREEN"}

        result = run_tech_touch(green_account, approvals_file=approvals_file)

        assert result["error"] != ""
        assert "GREEN" in result["error"]
        assert not approvals_file.exists()

    def test_multiple_approvals_accumulate_in_file(self, tmp_path):
        """
        Running the agent twice should append to approvals.json,
        not overwrite — so the CSM sees the full pending queue.
        """
        approvals_file = tmp_path / "approvals.json"
        mock_subject = "Following up, Sofia"
        mock_body = "Hi Sofia, just wanted to check in quickly."

        with patch("agents.tech_touch_agent.openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = _make_openai_mock(mock_subject, mock_body)
            run_tech_touch(ACCOUNT_CHECK_IN, approvals_file=approvals_file)

        second_account = {**ACCOUNT_RE_ENGAGEMENT, "account_id": "ACC-012"}
        with patch("agents.tech_touch_agent.openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = _make_openai_mock("Subject 2", "Body 2")
            run_tech_touch(second_account, approvals_file=approvals_file)

        saved = json.loads(approvals_file.read_text())
        assert len(saved) == 2
        assert saved[0]["account_id"] == "ACC-010"
        assert saved[1]["account_id"] == "ACC-012"

    def test_unknown_template_type_is_rejected(self, tmp_path):
        """An unrecognised template_type should set error without calling OpenAI."""
        approvals_file = tmp_path / "approvals.json"
        bad_account = {**ACCOUNT_CHECK_IN, "template_type": "magic_email"}

        result = run_tech_touch(bad_account, approvals_file=approvals_file)

        assert result["error"] != ""
        assert "magic_email" in result["error"]
        assert not approvals_file.exists()

    def test_approval_record_contains_all_required_keys(self, tmp_path):
        """Every approval record must have the full set of keys the UI expects."""
        approvals_file = tmp_path / "approvals.json"
        required_keys = {
            "account_id", "company_name", "client_name", "csm_name",
            "template_type", "subject", "body", "status", "generated_at",
        }

        with patch("agents.tech_touch_agent.openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = _make_openai_mock("Subj", "Body text here.")
            result = run_tech_touch(ACCOUNT_CHECK_IN, approvals_file=approvals_file)

        assert required_keys.issubset(result["approval_record"].keys())
