"""
Sprint Review Engine
--------------------
Bi-weekly account rotation engine. Re-evaluates every account using the
Coverage Router, detects lane changes, generates a CSM Summary Report,
and persists the updated lane assignments.

Triggered every 2 weeks via n8n cron (Sprint 2). Run manually in Sprint 1:
    python agents/sprint_review_engine.py

LangGraph pattern: StateGraph with 5 sequential nodes:
    load_accounts → re_evaluate → update_lanes → generate_summary → save_report

Lane change vocabulary:
    GRADUATED — account moved to a healthier (lower-risk) lane  e.g. RED→YELLOW, YELLOW→GREEN
    ESCALATED — account moved to a riskier lane                 e.g. GREEN→YELLOW, YELLOW→RED
    STABLE    — no lane change this sprint
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import openai
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from agents.coverage_router import route_account

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_ACCOUNTS_FILE = REPO_ROOT / "data" / "accounts.json"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Numeric ordering of lanes: higher value = higher risk
_LANE_RISK: dict[str, int] = {"GREEN": 0, "YELLOW": 1, "RED": 2}


def _classify_change(old_lane: str, new_lane: str) -> str:
    """
    Returns GRADUATED, ESCALATED, or STABLE for a lane transition.

    Direction is determined by _LANE_RISK ordering:
      lower risk than before → GRADUATED
      higher risk than before → ESCALATED
      same lane → STABLE
    """
    if old_lane == new_lane:
        return "STABLE"
    return "GRADUATED" if _LANE_RISK[new_lane] < _LANE_RISK[old_lane] else "ESCALATED"


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class SprintReviewState(TypedDict):
    """Shared state threaded through every node in the graph."""

    # --- Input / config (injected via invoke) ---
    accounts_file: str      # Absolute path to accounts.json
    reports_dir: str        # Absolute path to the reports output directory

    # --- Populated by load_accounts ---
    accounts: list          # Raw account records loaded from accounts.json

    # --- Populated by re_evaluate ---
    # Each entry = original account dict + {new_lane, change_type, triggered_rules}
    evaluated: list

    # --- Populated by generate_summary ---
    report: dict            # Full CSM summary report

    # --- Populated by save_report ---
    report_path: str        # Absolute path of the saved report file


# ---------------------------------------------------------------------------
# Node 1 — Load accounts
# ---------------------------------------------------------------------------


def load_accounts(state: SprintReviewState) -> SprintReviewState:
    """
    Reads all accounts from accounts.json and initialises derived state.

    Expected account shape (one entry per account):
        account_id, company_name, current_lane, lane_assigned_date,
        churn_score, sentiment, arr, days_since_last_contact, open_tickets

    Sprint 2: Replace this node body with a Supabase query.
    """
    accounts_file = Path(state["accounts_file"])

    # Initialise all derived fields so downstream nodes always find them
    state["accounts"] = []
    state["evaluated"] = []
    state["report"] = {}
    state["report_path"] = ""

    logger.info("[SPRINT-REVIEW] LOAD | file=%s", accounts_file)

    if not accounts_file.exists():
        logger.error("[SPRINT-REVIEW] LOAD | accounts file not found: %s", accounts_file)
        return state

    state["accounts"] = json.loads(accounts_file.read_text(encoding="utf-8"))
    logger.info("[SPRINT-REVIEW] LOAD | %d accounts loaded", len(state["accounts"]))
    return state


# ---------------------------------------------------------------------------
# Node 2 — Re-evaluate
# ---------------------------------------------------------------------------


def re_evaluate(state: SprintReviewState) -> SprintReviewState:
    """
    Runs the Coverage Router on every account and classifies lane changes.

    For each account the router receives the current signal values
    (churn_score, sentiment, etc.) and returns the lane those signals
    justify today — which may differ from the stored current_lane if
    conditions have changed since the last sprint review.

    The Coverage Router uses account_name; accounts.json stores company_name.
    The field is mapped here so the router contract stays stable.
    """
    evaluated: list[dict] = []

    for account in state["accounts"]:
        # Map accounts.json fields → Coverage Router input contract
        router_input = {
            "account_id": account["account_id"],
            "account_name": account["company_name"],   # router calls this account_name
            "churn_score": account["churn_score"],
            "sentiment": account["sentiment"],
            "arr": account["arr"],
            "days_since_last_contact": account["days_since_last_contact"],
            "open_tickets": account["open_tickets"],
        }

        result = route_account(router_input)

        old_lane = account["current_lane"]
        new_lane = result["lane"]
        change_type = _classify_change(old_lane, new_lane)

        if change_type != "STABLE":
            icon = "📈" if change_type == "GRADUATED" else "📉"
            logger.info(
                "[SPRINT-REVIEW] %s %s | %s → %s (%s)",
                icon, account["company_name"], old_lane, new_lane, change_type,
            )

        evaluated.append({
            **account,
            "new_lane": new_lane,
            "change_type": change_type,
            "triggered_rules": result["triggered_rules"],
        })

    changed = sum(1 for a in evaluated if a["change_type"] != "STABLE")
    logger.info("[SPRINT-REVIEW] RE-EVALUATE | %d of %d accounts changed lane",
                changed, len(evaluated))

    state["evaluated"] = evaluated
    return state


# ---------------------------------------------------------------------------
# Node 3 — Update lanes
# ---------------------------------------------------------------------------


def update_lanes(state: SprintReviewState) -> SprintReviewState:
    """
    Persists updated lane assignments to accounts.json.

    For accounts that changed lanes:
        - Sets current_lane to the new lane
        - Sets lane_assigned_date to now (UTC ISO-8601)
        - Preserves previous_lane for audit history

    Accounts with STABLE classification are written back unchanged
    (stripped of the temporary derived fields added by re_evaluate).

    Sprint 2: Write to Supabase instead of the flat JSON file.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    # Fields added by re_evaluate that should not be persisted to disk
    _derived = {"new_lane", "change_type", "triggered_rules"}

    persisted: list[dict] = []

    for account in state["evaluated"]:
        # Strip derived fields before writing back
        base = {k: v for k, v in account.items() if k not in _derived}

        if account["change_type"] != "STABLE":
            base["previous_lane"] = account["current_lane"]
            base["current_lane"] = account["new_lane"]
            base["lane_assigned_date"] = now_iso
            lane_history = list(base.get("lane_history", []))
            lane_history.append({
                "from": account["current_lane"],
                "to": account["new_lane"],
                "changed_at": now_iso,
                "source": "sprint_review",
                "reason": " | ".join(account["triggered_rules"]),
            })
            base["lane_history"] = lane_history

        persisted.append(base)

    accounts_file = Path(state["accounts_file"])
    accounts_file.write_text(
        json.dumps(persisted, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("[SPRINT-REVIEW] UPDATE | %d accounts written to %s",
                len(persisted), accounts_file)
    return state


# ---------------------------------------------------------------------------
# Node 4 — Generate summary
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM_PROMPT = (
    "You are generating a bi-weekly account review summary for a Customer Success Manager "
    "at Ontop, a global payroll and contractor management platform. "
    "Be concise and actionable. Focus on what needs immediate human attention. "
    "Refer to 'account health' rather than churn scores in your output."
)


def generate_summary(state: SprintReviewState) -> SprintReviewState:
    """
    Builds the CSM Summary Report.

    Responsibility split:
        CODE  → all deterministic fields: counts, graduated/escalated lists,
                report_date, total_accounts
        GPT-4o → qualitative fields only: csm_action_items, narrative_summary

    This split keeps structured data reliable and testable while letting
    the LLM generate the parts it's genuinely good at.
    """
    evaluated = state["evaluated"]
    now = datetime.now(timezone.utc)

    # --- Build deterministic sections ---

    graduated = [
        {
            "account_id": a["account_id"],
            "company": a["company_name"],
            "from": a["current_lane"],       # original lane (before this sprint)
            "to": a["new_lane"],             # new lane (after re-evaluation)
            "reason": " | ".join(a["triggered_rules"]),
        }
        for a in evaluated if a["change_type"] == "GRADUATED"
    ]

    escalated = [
        {
            "account_id": a["account_id"],
            "company": a["company_name"],
            "from": a["current_lane"],
            "to": a["new_lane"],
            "reason": " | ".join(a["triggered_rules"]),
        }
        for a in evaluated if a["change_type"] == "ESCALATED"
    ]

    # Stable counts use new_lane (equals current_lane for STABLE accounts)
    stable_red    = sum(1 for a in evaluated if a["change_type"] == "STABLE" and a["new_lane"] == "RED")
    stable_yellow = sum(1 for a in evaluated if a["change_type"] == "STABLE" and a["new_lane"] == "YELLOW")
    stable_green  = sum(1 for a in evaluated if a["change_type"] == "STABLE" and a["new_lane"] == "GREEN")

    report: dict = {
        "report_date": now.strftime("%Y-%m-%d"),
        "total_accounts": len(evaluated),
        "graduated": graduated,
        "escalated": escalated,
        "stable_red": stable_red,
        "stable_yellow": stable_yellow,
        "stable_green": stable_green,
        "csm_action_items": [],       # filled by LLM below
        "narrative_summary": "",      # filled by LLM below
    }

    # --- Build context string for GPT-4o ---

    change_lines: list[str] = []
    for a in graduated:
        change_lines.append(f"  GRADUATED: {a['company']} ({a['from']} → {a['to']}) — {a['reason']}")
    for a in escalated:
        change_lines.append(f"  ESCALATED: {a['company']} ({a['from']} → {a['to']}) — {a['reason']}")
    change_text = "\n".join(change_lines) if change_lines else "  No lane changes this sprint."

    new_red    = sum(1 for a in evaluated if a["new_lane"] == "RED")
    new_yellow = sum(1 for a in evaluated if a["new_lane"] == "YELLOW")
    new_green  = sum(1 for a in evaluated if a["new_lane"] == "GREEN")

    llm_prompt = (
        f"Sprint Review Date: {report['report_date']}\n"
        f"Total accounts reviewed: {report['total_accounts']}\n\n"
        f"Lane changes this sprint:\n{change_text}\n\n"
        f"New portfolio distribution: {new_red} RED, {new_yellow} YELLOW, {new_green} GREEN\n\n"
        "Generate two things:\n"
        "1. csm_action_items: a list of 3–5 specific, actionable items the CSM should "
        "address this sprint, in priority order.\n"
        "2. narrative_summary: 2–3 sentences summarising overall portfolio health "
        "and the most important priorities.\n\n"
        'Respond in JSON format only:\n'
        '{"csm_action_items": ["...", "..."], "narrative_summary": "..."}'
    )

    # --- Call GPT-4o for qualitative sections ---
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": llm_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,   # low temp for actionable internal docs
        )
        qualitative = json.loads(response.choices[0].message.content)
        report["csm_action_items"] = qualitative.get("csm_action_items", [])
        report["narrative_summary"] = qualitative.get("narrative_summary", "")

    except Exception as exc:  # noqa: BLE001
        # Degrade gracefully — structured data is already complete
        logger.error("[SPRINT-REVIEW] LLM ERROR | %s", exc)
        report["csm_action_items"] = ["[LLM unavailable — review escalated accounts manually]"]
        report["narrative_summary"] = "[LLM generation failed]"

    state["report"] = report
    return state


# ---------------------------------------------------------------------------
# Node 5 — Save report
# ---------------------------------------------------------------------------


def save_report(state: SprintReviewState) -> SprintReviewState:
    """
    Writes the report to reports/sprint_review_{YYYY-MM-DD}.json
    and logs a formatted summary to the console.

    The reports directory is created if it doesn't already exist.

    Sprint 2: Also POST the report payload to the n8n webhook that
    sends a Slack digest to the #cs-intelligence channel.
    """
    if not state["report"]:
        logger.warning("[SPRINT-REVIEW] SAVE | nothing to save (report is empty)")
        return state

    reports_dir = Path(state["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_date = state["report"].get("report_date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    report_file = reports_dir / f"sprint_review_{report_date}.json"

    report_file.write_text(
        json.dumps(state["report"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    state["report_path"] = str(report_file)

    # Formatted console summary
    r = state["report"]
    logger.info("=" * 60)
    logger.info("[SPRINT-REVIEW] REPORT SAVED → %s", report_file.name)
    logger.info("  Total accounts  : %d", r["total_accounts"])
    logger.info("  Graduated       : %d  %s",
                len(r["graduated"]),
                [f"{a['company']} ({a['from']}→{a['to']})" for a in r["graduated"]])
    logger.info("  Escalated       : %d  %s",
                len(r["escalated"]),
                [f"{a['company']} ({a['from']}→{a['to']})" for a in r["escalated"]])
    logger.info("  Stable RED      : %d", r["stable_red"])
    logger.info("  Stable YELLOW   : %d", r["stable_yellow"])
    logger.info("  Stable GREEN    : %d", r["stable_green"])
    if r["narrative_summary"]:
        logger.info("  Summary: %s", r["narrative_summary"])
    logger.info("=" * 60)

    # Sprint 2: POST state["report"] to n8n webhook for Slack notification
    return state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_sprint_review_engine() -> StateGraph:
    """Constructs and compiles the LangGraph StateGraph."""
    graph = StateGraph(SprintReviewState)

    graph.add_node("load_accounts", load_accounts)
    graph.add_node("re_evaluate", re_evaluate)
    graph.add_node("update_lanes", update_lanes)
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("save_report", save_report)

    graph.set_entry_point("load_accounts")
    graph.add_edge("load_accounts", "re_evaluate")
    graph.add_edge("re_evaluate", "update_lanes")
    graph.add_edge("update_lanes", "generate_summary")
    graph.add_edge("generate_summary", "save_report")
    graph.add_edge("save_report", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def run_sprint_review(
    accounts_file: Path = DEFAULT_ACCOUNTS_FILE,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> dict:
    """
    Convenience wrapper — runs the full sprint review cycle.

    Args:
        accounts_file: Path to accounts.json (override in tests).
        reports_dir:   Directory to write report files (override in tests).

    Returns:
        Final SprintReviewState dict including the generated report.
    """
    engine = build_sprint_review_engine()
    return engine.invoke({
        "accounts_file": str(accounts_file),
        "reports_dir": str(reports_dir),
    })


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_sprint_review()
    print(json.dumps(result["report"], indent=2))
