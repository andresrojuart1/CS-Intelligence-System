"""
Coverage Router Agent
---------------------
Classifies customer accounts into RED / YELLOW / GREEN coverage lanes
based on churn risk, sentiment, and engagement signals.

Lane definitions:
  RED    (Human-Touch)  — High-risk accounts requiring CSM intervention
  YELLOW (Tech-Touch)   — Medium-risk accounts handled via automated outreach
  GREEN  (Monitor)      — Healthy accounts on passive monitoring

LangGraph pattern: StateGraph with three sequential nodes:
  ingest_account → classify_account → emit_result
"""

from __future__ import annotations

import json
import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AccountState(TypedDict):
    """Shared state threaded through every node in the graph."""

    # --- Raw input fields ---
    account_id: str
    account_name: str
    churn_score: int          # 0–100; higher = more likely to churn
    sentiment: str            # "positive" | "neutral" | "negative"
    arr: float                # Annual Recurring Revenue in USD
    days_since_last_contact: int
    open_tickets: int         # Unresolved Zendesk tickets

    # --- Derived fields (populated by classify node) ---
    lane: str                 # "RED" | "YELLOW" | "GREEN"
    triggered_rules: list[str]  # Which rules fired for this classification


# ---------------------------------------------------------------------------
# Node 1 — Ingest
# ---------------------------------------------------------------------------

def ingest_account(state: AccountState) -> AccountState:
    """
    Validates and normalises raw account data before classification.

    In Sprint 1 this node receives a pre-populated dict from mock data.
    In Sprint 2 it will pull live records from Salesforce and Zendesk,
    so all normalisation logic lives here to keep the classify node clean.
    """
    logger.info("INGEST  | account_id=%s  name=%s", state["account_id"], state["account_name"])

    # Clamp numeric fields to expected ranges so the classifier is deterministic
    state["churn_score"] = max(0, min(100, int(state["churn_score"])))
    state["arr"] = max(0.0, float(state["arr"]))
    state["days_since_last_contact"] = max(0, int(state["days_since_last_contact"]))
    state["open_tickets"] = max(0, int(state["open_tickets"]))

    # Normalise sentiment to lower-case canonical value
    state["sentiment"] = state["sentiment"].strip().lower()
    if state["sentiment"] not in ("positive", "neutral", "negative"):
        logger.warning("Unknown sentiment '%s' — defaulting to 'neutral'", state["sentiment"])
        state["sentiment"] = "neutral"

    # Initialise derived fields so downstream nodes always find them
    state.setdefault("lane", "")
    state.setdefault("triggered_rules", [])

    return state


# ---------------------------------------------------------------------------
# Node 2 — Classify
# ---------------------------------------------------------------------------

def classify_account(state: AccountState) -> AccountState:
    """
    Applies lane classification rules in priority order.

    Rules are evaluated independently so multiple triggers can be logged,
    but the *highest-priority* lane wins (RED > YELLOW > GREEN).

    Priority order matches business intent: we never under-serve a
    high-risk account just because a lower-risk rule also fired.
    """
    triggered: list[str] = []

    # --- RED rules (any one is sufficient) ---
    if state["churn_score"] > 70:
        triggered.append(f"churn_score={state['churn_score']} > 70")
    if state["sentiment"] == "negative":
        triggered.append("sentiment=negative")
    if state["open_tickets"] > 3:
        triggered.append(f"open_tickets={state['open_tickets']} > 3")

    if triggered:
        state["lane"] = "RED"
        state["triggered_rules"] = triggered
        return state

    # --- YELLOW rules (any one is sufficient) ---
    if 40 <= state["churn_score"] <= 70:
        triggered.append(f"churn_score={state['churn_score']} in [40, 70]")
    if state["days_since_last_contact"] > 14:
        triggered.append(f"days_since_last_contact={state['days_since_last_contact']} > 14")

    if triggered:
        state["lane"] = "YELLOW"
        state["triggered_rules"] = triggered
        return state

    # --- GREEN (default — all clear) ---
    state["lane"] = "GREEN"
    state["triggered_rules"] = ["No risk rules triggered"]
    return state


# ---------------------------------------------------------------------------
# Node 3 — Emit result
# ---------------------------------------------------------------------------

def emit_result(state: AccountState) -> AccountState:
    """
    Logs the final classification and would dispatch downstream actions
    in later sprints (e.g. write to Supabase, trigger n8n webhook).
    """
    lane_emoji = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢"}.get(state["lane"], "⚪")

    logger.info(
        "RESULT  | %s %s  account=%s  arr=$%.0f  lane=%s",
        lane_emoji,
        state["account_name"],
        state["account_id"],
        state["arr"],
        state["lane"],
    )
    logger.info("  Rules fired: %s", " | ".join(state["triggered_rules"]))

    # Sprint 2: write classification to Supabase
    # Sprint 2: POST to n8n webhook to trigger outreach workflow

    return state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_coverage_router() -> StateGraph:
    """Constructs and compiles the LangGraph StateGraph."""
    graph = StateGraph(AccountState)

    graph.add_node("ingest_account", ingest_account)
    graph.add_node("classify_account", classify_account)
    graph.add_node("emit_result", emit_result)

    graph.set_entry_point("ingest_account")
    graph.add_edge("ingest_account", "classify_account")
    graph.add_edge("classify_account", "emit_result")
    graph.add_edge("emit_result", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def route_account(account: dict) -> dict:
    """
    Convenience wrapper — runs the router graph for a single account dict
    and returns the final state (including lane and triggered_rules).

    Args:
        account: Dict matching the AccountState schema.

    Returns:
        Final AccountState dict with 'lane' and 'triggered_rules' populated.
    """
    router = build_coverage_router()
    result = router.invoke(account)
    return result


# ---------------------------------------------------------------------------
# CLI entry point (run a quick smoke-test with one hardcoded account)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = {
        "account_id": "ACC-001",
        "account_name": "Acme Corp",
        "churn_score": 75,
        "sentiment": "negative",
        "arr": 48000.0,
        "days_since_last_contact": 20,
        "open_tickets": 5,
    }
    result = route_account(sample)
    print(json.dumps({k: result[k] for k in ("account_id", "lane", "triggered_rules")}, indent=2))
