"""
Tests for sprint_review_engine.py
-----------------------------------
All OpenAI calls are mocked — no real API keys required.
Each test writes to an isolated tmp_path so no test pollutes another.

Run with:  pytest tests/test_sprint_review_engine.py -v
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agents.sprint_review_engine import run_sprint_review, _classify_change


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_accounts(path: Path, accounts: list[dict]) -> None:
    """Writes a list of account dicts to the given path as JSON."""
    path.write_text(json.dumps(accounts, indent=2), encoding="utf-8")


def _mock_openai(action_items: Optional[list[str]] = None, narrative: str = "") -> MagicMock:
    """
    Returns a mock openai.OpenAI() client whose chat.completions.create()
    yields a fake JSON response with the given qualitative fields.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps({
        "csm_action_items": action_items or ["Action 1", "Action 2"],
        "narrative_summary": narrative or "Portfolio is stable with one escalation to address.",
    })
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# Minimal account fixtures
# ---------------------------------------------------------------------------

# YELLOW account whose current signal values now qualify for GREEN
ACCOUNT_YELLOW_GRADUATING = {
    "account_id": "TEST-001",
    "company_name": "GreenCandidate Co",
    "current_lane": "YELLOW",
    "lane_assigned_date": "2026-03-17T00:00:00+00:00",
    "churn_score": 20,            # below 40 — no YELLOW rule fires
    "sentiment": "positive",       # not negative — no RED rule fires
    "arr": 50000.0,
    "days_since_last_contact": 8,  # below 14 — no YELLOW rule fires
    "open_tickets": 0,
}

# GREEN account whose current signal values now qualify for YELLOW
ACCOUNT_GREEN_ESCALATING = {
    "account_id": "TEST-002",
    "company_name": "YellowCandidate Inc",
    "current_lane": "GREEN",
    "lane_assigned_date": "2026-03-17T00:00:00+00:00",
    "churn_score": 50,             # in [40, 70] — YELLOW rule fires
    "sentiment": "neutral",
    "arr": 30000.0,
    "days_since_last_contact": 10,
    "open_tickets": 0,
}

# RED account whose scores now qualify for YELLOW (RED→YELLOW graduation)
ACCOUNT_RED_GRADUATING = {
    "account_id": "TEST-003",
    "company_name": "RecoveringLtd",
    "current_lane": "RED",
    "lane_assigned_date": "2026-03-17T00:00:00+00:00",
    "churn_score": 55,             # in [40, 70] — routes to YELLOW
    "sentiment": "neutral",
    "arr": 70000.0,
    "days_since_last_contact": 12,
    "open_tickets": 1,
}

# Clearly stable GREEN — no changes expected
ACCOUNT_GREEN_STABLE = {
    "account_id": "TEST-004",
    "company_name": "RockSolid SA",
    "current_lane": "GREEN",
    "lane_assigned_date": "2026-03-17T00:00:00+00:00",
    "churn_score": 10,
    "sentiment": "positive",
    "arr": 200000.0,
    "days_since_last_contact": 3,
    "open_tickets": 0,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSprintReviewEngine:

    # --- Lane change classification ---

    def test_classify_change_stable(self):
        """Same lane → STABLE."""
        assert _classify_change("RED", "RED") == "STABLE"
        assert _classify_change("YELLOW", "YELLOW") == "STABLE"
        assert _classify_change("GREEN", "GREEN") == "STABLE"

    def test_classify_change_graduated(self):
        """Moving to a lower-risk lane → GRADUATED."""
        assert _classify_change("RED", "YELLOW") == "GRADUATED"
        assert _classify_change("RED", "GREEN") == "GRADUATED"
        assert _classify_change("YELLOW", "GREEN") == "GRADUATED"

    def test_classify_change_escalated(self):
        """Moving to a higher-risk lane → ESCALATED."""
        assert _classify_change("GREEN", "YELLOW") == "ESCALATED"
        assert _classify_change("GREEN", "RED") == "ESCALATED"
        assert _classify_change("YELLOW", "RED") == "ESCALATED"

    # --- YELLOW → GREEN graduation ---

    def test_yellow_account_graduates_to_green(self, tmp_path):
        """
        A YELLOW account whose signal values now fall below all YELLOW
        thresholds should appear in report['graduated'] with from=YELLOW, to=GREEN.
        Scores used: churn_score=20 (<40), sentiment=positive, days_since_last_contact=8 (<14).
        """
        accounts_file = tmp_path / "accounts.json"
        _write_accounts(accounts_file, [ACCOUNT_YELLOW_GRADUATING])

        with patch("agents.sprint_review_engine.openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = _mock_openai()
            result = run_sprint_review(accounts_file=accounts_file, reports_dir=tmp_path / "reports")

        assert len(result["report"]["graduated"]) == 1
        grad = result["report"]["graduated"][0]
        assert grad["account_id"] == "TEST-001"
        assert grad["from"] == "YELLOW"
        assert grad["to"] == "GREEN"

        # accounts.json should now reflect the new lane
        updated = json.loads(accounts_file.read_text())
        assert updated[0]["current_lane"] == "GREEN"
        assert updated[0]["previous_lane"] == "YELLOW"

    # --- GREEN → YELLOW escalation ---

    def test_green_account_escalates_to_yellow(self, tmp_path):
        """
        A GREEN account with churn_score in [40, 70] should appear in
        report['escalated'] with from=GREEN, to=YELLOW.
        """
        accounts_file = tmp_path / "accounts.json"
        _write_accounts(accounts_file, [ACCOUNT_GREEN_ESCALATING])

        with patch("agents.sprint_review_engine.openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = _mock_openai()
            result = run_sprint_review(accounts_file=accounts_file, reports_dir=tmp_path / "reports")

        assert len(result["report"]["escalated"]) == 1
        esc = result["report"]["escalated"][0]
        assert esc["account_id"] == "TEST-002"
        assert esc["from"] == "GREEN"
        assert esc["to"] == "YELLOW"

        updated = json.loads(accounts_file.read_text())
        assert updated[0]["current_lane"] == "YELLOW"
        assert updated[0]["previous_lane"] == "GREEN"

    # --- Full report structure ---

    def test_summary_report_has_required_keys(self, tmp_path):
        """
        Running the engine with mixed accounts should produce a report
        that contains every key the downstream consumers (Slack, n8n) expect.
        """
        accounts_file = tmp_path / "accounts.json"
        _write_accounts(accounts_file, [
            ACCOUNT_YELLOW_GRADUATING,
            ACCOUNT_GREEN_ESCALATING,
            ACCOUNT_GREEN_STABLE,
        ])

        mock_actions = ["Follow up with YellowCandidate Inc", "Monitor GreenCandidate Co"]
        mock_narrative = "Two accounts changed lanes this sprint, one requiring immediate CSM attention."

        with patch("agents.sprint_review_engine.openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = _mock_openai(mock_actions, mock_narrative)
            result = run_sprint_review(accounts_file=accounts_file, reports_dir=tmp_path / "reports")

        report = result["report"]
        required_keys = {
            "report_date", "total_accounts",
            "graduated", "escalated",
            "stable_red", "stable_yellow", "stable_green",
            "csm_action_items", "narrative_summary",
        }
        assert required_keys.issubset(report.keys())

        # Counts
        assert report["total_accounts"] == 3
        assert len(report["graduated"]) == 1     # YELLOW_GRADUATING
        assert len(report["escalated"]) == 1     # GREEN_ESCALATING
        assert report["stable_green"] == 1       # GREEN_STABLE
        assert report["stable_yellow"] == 0
        assert report["stable_red"] == 0

        # LLM fields populated
        assert report["csm_action_items"] == mock_actions
        assert report["narrative_summary"] == mock_narrative

    # --- Persistence ---

    def test_report_written_to_disk(self, tmp_path):
        """The engine must write reports/sprint_review_{date}.json to disk."""
        accounts_file = tmp_path / "accounts.json"
        reports_dir = tmp_path / "reports"
        _write_accounts(accounts_file, [ACCOUNT_GREEN_STABLE])

        with patch("agents.sprint_review_engine.openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = _mock_openai()
            result = run_sprint_review(accounts_file=accounts_file, reports_dir=reports_dir)

        report_path = Path(result["report_path"])
        assert report_path.exists()
        assert report_path.name.startswith("sprint_review_")
        assert report_path.suffix == ".json"

        saved_report = json.loads(report_path.read_text())
        assert saved_report["total_accounts"] == 1

    # --- Stable account not mutated on disk ---

    def test_stable_accounts_preserve_lane_assigned_date(self, tmp_path):
        """
        Accounts with no lane change should have their lane_assigned_date
        unchanged in accounts.json after the review runs.
        """
        original_date = "2026-03-17T00:00:00+00:00"
        accounts_file = tmp_path / "accounts.json"
        _write_accounts(accounts_file, [{**ACCOUNT_GREEN_STABLE, "lane_assigned_date": original_date}])

        with patch("agents.sprint_review_engine.openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = _mock_openai()
            run_sprint_review(accounts_file=accounts_file, reports_dir=tmp_path / "reports")

        updated = json.loads(accounts_file.read_text())
        assert updated[0]["lane_assigned_date"] == original_date

    # --- Multiple changes in one run ---

    def test_multiple_changes_detected_in_single_run(self, tmp_path):
        """
        Running the engine over a batch with both graduating and escalating
        accounts should correctly populate both lists in the report.
        """
        accounts_file = tmp_path / "accounts.json"
        _write_accounts(accounts_file, [
            ACCOUNT_YELLOW_GRADUATING,   # YELLOW → GREEN (graduated)
            ACCOUNT_GREEN_ESCALATING,    # GREEN  → YELLOW (escalated)
            ACCOUNT_RED_GRADUATING,      # RED    → YELLOW (graduated)
        ])

        with patch("agents.sprint_review_engine.openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = _mock_openai()
            result = run_sprint_review(accounts_file=accounts_file, reports_dir=tmp_path / "reports")

        report = result["report"]
        assert len(report["graduated"]) == 2
        assert len(report["escalated"]) == 1

        graduated_ids = {a["account_id"] for a in report["graduated"]}
        assert graduated_ids == {"TEST-001", "TEST-003"}
        assert report["escalated"][0]["account_id"] == "TEST-002"

    # --- LLM failure degrades gracefully ---

    def test_llm_failure_does_not_crash_engine(self, tmp_path):
        """
        If GPT-4o is unavailable, the engine should still produce a
        complete report — with a fallback message in the LLM fields.
        """
        accounts_file = tmp_path / "accounts.json"
        _write_accounts(accounts_file, [ACCOUNT_GREEN_STABLE])

        with patch("agents.sprint_review_engine.openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = RuntimeError("API timeout")
            MockOpenAI.return_value = mock_client

            result = run_sprint_review(accounts_file=accounts_file, reports_dir=tmp_path / "reports")

        report = result["report"]
        # Structured data should still be intact
        assert report["total_accounts"] == 1
        assert isinstance(report["graduated"], list)
        # Qualitative fields should contain the fallback
        assert "unavailable" in report["csm_action_items"][0].lower()
        assert "failed" in report["narrative_summary"].lower()
