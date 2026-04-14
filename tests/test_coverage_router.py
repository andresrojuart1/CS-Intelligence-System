"""
Tests for coverage_router.py
-----------------------------
Uses 5 mock accounts that exercise every lane and every trigger rule.
No external API calls — pure unit tests.

Run with:  pytest tests/test_coverage_router.py -v
"""

import sys
import os

# Allow imports from repo root regardless of how pytest is invoked
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agents.coverage_router import route_account


# ---------------------------------------------------------------------------
# Mock account fixtures
# ---------------------------------------------------------------------------

# 1. Clearly RED — high churn score, negative sentiment, many open tickets
ACCOUNT_RED_ALL_RULES = {
    "account_id": "ACC-001",
    "account_name": "ChurnRisk Corp",
    "churn_score": 85,
    "sentiment": "negative",
    "arr": 120000.0,
    "days_since_last_contact": 30,
    "open_tickets": 7,
}

# 2. RED via a single rule — negative sentiment only, everything else healthy
ACCOUNT_RED_SENTIMENT_ONLY = {
    "account_id": "ACC-002",
    "account_name": "Frustrated LLC",
    "churn_score": 25,
    "sentiment": "negative",
    "arr": 45000.0,
    "days_since_last_contact": 5,
    "open_tickets": 1,
}

# 3. YELLOW — churn score in the 40-70 band, contact is recent
ACCOUNT_YELLOW_CHURN_BAND = {
    "account_id": "ACC-003",
    "account_name": "Wavering Inc",
    "churn_score": 55,
    "sentiment": "neutral",
    "arr": 80000.0,
    "days_since_last_contact": 10,
    "open_tickets": 2,
}

# 4. YELLOW — healthy churn but contact has gone stale
ACCOUNT_YELLOW_STALE_CONTACT = {
    "account_id": "ACC-004",
    "account_name": "Silent Partners",
    "churn_score": 20,
    "sentiment": "positive",
    "arr": 32000.0,
    "days_since_last_contact": 21,
    "open_tickets": 0,
}

# 5. GREEN — everything healthy
ACCOUNT_GREEN_HEALTHY = {
    "account_id": "ACC-005",
    "account_name": "Happy Customer Co",
    "churn_score": 12,
    "sentiment": "positive",
    "arr": 200000.0,
    "days_since_last_contact": 7,
    "open_tickets": 0,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCoverageRouter:

    def test_red_all_rules_fires(self):
        """Account with high churn, negative sentiment, and many tickets → RED."""
        result = route_account(ACCOUNT_RED_ALL_RULES)
        assert result["lane"] == "RED"
        # All three RED rules should be present in triggered_rules
        rules_text = " ".join(result["triggered_rules"])
        assert "churn_score" in rules_text
        assert "sentiment=negative" in rules_text
        assert "open_tickets" in rules_text

    def test_red_triggered_by_sentiment_alone(self):
        """Negative sentiment alone is enough to land in RED regardless of other signals."""
        result = route_account(ACCOUNT_RED_SENTIMENT_ONLY)
        assert result["lane"] == "RED"
        assert any("sentiment=negative" in r for r in result["triggered_rules"])

    def test_yellow_churn_band(self):
        """churn_score in [40, 70] with recent contact → YELLOW."""
        result = route_account(ACCOUNT_YELLOW_CHURN_BAND)
        assert result["lane"] == "YELLOW"
        assert any("churn_score" in r for r in result["triggered_rules"])

    def test_yellow_stale_contact(self):
        """Low churn but days_since_last_contact > 14 → YELLOW."""
        result = route_account(ACCOUNT_YELLOW_STALE_CONTACT)
        assert result["lane"] == "YELLOW"
        assert any("days_since_last_contact" in r for r in result["triggered_rules"])

    def test_green_healthy_account(self):
        """Healthy account with all signals in range → GREEN."""
        result = route_account(ACCOUNT_GREEN_HEALTHY)
        assert result["lane"] == "GREEN"
        assert result["triggered_rules"] == ["No risk rules triggered"]

    def test_result_always_contains_required_keys(self):
        """Every routed account must return lane and triggered_rules."""
        for account in [
            ACCOUNT_RED_ALL_RULES,
            ACCOUNT_RED_SENTIMENT_ONLY,
            ACCOUNT_YELLOW_CHURN_BAND,
            ACCOUNT_YELLOW_STALE_CONTACT,
            ACCOUNT_GREEN_HEALTHY,
        ]:
            result = route_account(account)
            assert "lane" in result
            assert "triggered_rules" in result
            assert result["lane"] in ("RED", "YELLOW", "GREEN")
            assert isinstance(result["triggered_rules"], list)
            assert len(result["triggered_rules"]) > 0

    def test_red_takes_priority_over_yellow(self):
        """
        An account in the churn [40-70] band that also has negative sentiment
        should be RED, not YELLOW — RED rules are evaluated first.
        """
        borderline = {
            "account_id": "ACC-BORDER",
            "account_name": "Borderline Ltd",
            "churn_score": 55,   # would trigger YELLOW
            "sentiment": "negative",  # should elevate to RED
            "arr": 60000.0,
            "days_since_last_contact": 8,
            "open_tickets": 1,
        }
        result = route_account(borderline)
        assert result["lane"] == "RED"

    def test_ingest_clamps_out_of_range_churn_score(self):
        """churn_score > 100 should be clamped to 100 without raising an error."""
        account = {**ACCOUNT_GREEN_HEALTHY, "account_id": "ACC-CLAMP", "churn_score": 999}
        result = route_account(account)
        assert result["churn_score"] == 100
        assert result["lane"] == "RED"  # clamped to 100 → RED

    def test_unknown_sentiment_defaults_to_neutral(self):
        """Unexpected sentiment values are normalised to 'neutral' by ingest node."""
        account = {**ACCOUNT_GREEN_HEALTHY, "account_id": "ACC-UNK", "sentiment": "confused"}
        result = route_account(account)
        assert result["sentiment"] == "neutral"
