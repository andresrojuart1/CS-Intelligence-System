"""
CS Intelligence System — Streamlit Dashboard
---------------------------------------------
Operational UI for the three LangGraph agents:
  Tab 1 — Account Overview   (Coverage Router)
  Tab 2 — Tech-Touch Agent   (Email generation + CSM approval queue)
  Tab 3 — Sprint Review      (Bi-weekly account rotation)

Run locally:
    streamlit run streamlit_app.py

Set USE_MOCK_DATA=true in .env to bypass real OpenAI calls (Sprint 1 default).
"""

from __future__ import annotations

import json
import os
import sys
from html import escape
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup — allow imports from repo root regardless of cwd
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from agents.coverage_router import route_account
from agents.tech_touch_agent import run_tech_touch, DEFAULT_APPROVALS_FILE
from agents.sprint_review_engine import run_sprint_review, DEFAULT_ACCOUNTS_FILE, DEFAULT_REPORTS_DIR

load_dotenv()

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CS Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "true").lower() in ("true", "1", "yes")

LANE_COLOR = {"RED": "#FF4B4B", "YELLOW": "#FFA500", "GREEN": "#00C853"}
LANE_EMOJI = {"RED": "🔴",      "YELLOW": "🟡",       "GREEN": "🟢"}
LANE_BG    = {"RED": "#FF4B4B18", "YELLOW": "#FFA50018", "GREEN": "#00C85318"}

ACCOUNTS_FILE  = DEFAULT_ACCOUNTS_FILE
APPROVALS_FILE = DEFAULT_APPROVALS_FILE
REPORTS_DIR    = DEFAULT_REPORTS_DIR

# ---------------------------------------------------------------------------
# Mock responses (used when USE_MOCK_DATA=true)
# ---------------------------------------------------------------------------

# Pre-written emails returned instead of calling GPT-4o
_MOCK_EMAILS: dict[str, dict] = {
    "check_in": {
        "subject": "Checking in on your Ontop experience",
        "body": (
            "Hi there — I noticed some changes in your team's contractor activity "
            "recently and wanted to reach out before too much time passed.\n\n"
            "A few teams similar to yours have found our new bulk-update workflow "
            "helpful for staying on top of contractor changes. Happy to walk you "
            "through it in 15 minutes.\n\n"
            "Would a quick call this week work for you?"
        ),
    },
    "re_engagement": {
        "subject": "Something useful for your contractor pipeline",
        "body": (
            "Hope things are going well! I wanted to share a workflow improvement "
            "that's been making a real difference for teams managing contractor "
            "volume changes on Ontop.\n\n"
            "Curious what's been driving the shift in your team's headcount recently. "
            "Has anything changed in how you're planning your contractor pipeline?\n\n"
            "Would a 15-minute call this week be useful?"
        ),
    },
    "collections": {
        "subject": "Outstanding balance — let's resolve this together",
        "body": (
            "Hi — I hope your team is well. I wanted to follow up on an outstanding "
            "balance on your Ontop account and find the best way to resolve it.\n\n"
            "You can settle directly via the payment link in your dashboard, or if "
            "you'd prefer to talk through options, I'm happy to arrange a brief call. "
            "Please let me know how you'd like to proceed by the deadline we discussed."
        ),
    },
    "milestone": {
        "subject": "Celebrating a big milestone with your team 🎉",
        "body": (
            "I just wanted to take a moment to celebrate something genuinely impressive — "
            "your team has hit a milestone that reflects real operational excellence.\n\n"
            "Managing global talent at this scale is no small feat, and it's been "
            "exciting to watch your team build this capability with Ontop.\n\n"
            "Looking forward to supporting your next chapter of growth."
        ),
    },
}

# Summary returned instead of calling GPT-4o in sprint review
_MOCK_SPRINT_SUMMARY = {
    "csm_action_items": [
        "Schedule urgent call with Crestline Operations — escalated to RED this sprint",
        "Send celebration message to Lumina People — graduated to GREEN",
        "Follow up with Meridian Staffing — improving signals, confirm upward trajectory",
        "Set check-in for Navara Consulting — newly entered YELLOW lane",
        "Update Salesforce with new lane assignments for all changed accounts",
    ],
    "narrative_summary": (
        "This sprint saw 4 accounts change lanes — 2 improvements and 2 escalations. "
        "Immediate CSM attention is required for Crestline Operations (YELLOW→RED). "
        "Lumina People's graduation to GREEN is a win worth acknowledging with a "
        "milestone email. Overall portfolio health is mixed with the RED bucket "
        "requiring active intervention this week."
    ),
}

# ---------------------------------------------------------------------------
# Session state — initialise once per session
# ---------------------------------------------------------------------------

_SESSION_DEFAULTS: dict = {
    "lane_filter":      "ALL",      # Tab 1 active filter
    "portfolio_search": "",         # Portfolio search query
    "router_changes":   None,       # None=never run, []=ran+no changes, [...]=changes
    "last_router_run":  None,       # ISO timestamp string
    "tt_generated":     None,       # Approval record dict from last Tech-Touch run
    "tt_status":        None,       # None | "pending_approval" | "approved" | "rejecting" | "rejected"
    "sprint_report":    None,       # Report dict from last in-session sprint review run
    "last_sprint_run":  None,       # ISO timestamp string
}

for _k, _v in _SESSION_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# ---------------------------------------------------------------------------
# Visual system
# ---------------------------------------------------------------------------

def inject_theme() -> None:
    """Applies the shared Ontop-inspired theme from cx-sales-agent."""
    st.markdown(
        """
        <style>
        :root {
            --ontop-purple: #261C94;
            --ontop-coral: #E35276;
            --bg-primary: #000000;
            --bg-card: #060609;
            --bg-input: #1A1A24;
            --text-primary: #FFFFFF;
            --text-secondary: #B8B8C8;
            --text-muted: #6B6B7E;
            --border-color: #2A2A3E;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(38, 28, 148, 0.35), transparent 32%),
                radial-gradient(circle at top right, rgba(227, 82, 118, 0.20), transparent 28%),
                linear-gradient(180deg, #050507 0%, #000000 100%);
            color: var(--text-primary);
        }

        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(38, 28, 148, 0.18), rgba(6, 6, 9, 0.94) 26%),
                #060609;
            border-right: 1px solid var(--border-color);
        }

        [data-testid="stSidebar"] * {
            color: var(--text-primary);
        }

        [data-testid="stSidebarNav"],
        [data-testid="stSidebarNavSeparator"] {
            display: none;
        }

        .block-container {
            max-width: 1480px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3 {
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }

        p, li, label, .stMarkdown, .stCaption, [data-testid="stMarkdownContainer"] {
            color: var(--text-secondary);
        }

        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        textarea {
            background: var(--bg-input) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
            border-radius: 14px !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            background: linear-gradient(135deg, var(--ontop-purple), var(--ontop-coral));
            color: var(--text-primary);
            border: 0;
            border-radius: 999px;
            font-weight: 600;
            box-shadow: 0 10px 24px rgba(38, 28, 148, 0.28);
            min-height: 3rem;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            filter: brightness(1.06);
            color: var(--text-primary);
        }

        .stButton > button[kind="secondary"],
        button[data-testid="stBaseButton-secondary"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(124, 115, 247, 0.28);
            box-shadow: none;
        }

        .stButton > button[kind="tertiary"],
        button[data-testid="stBaseButton-tertiary"] {
            background: transparent;
            border: 1px solid rgba(255, 255, 255, 0.10);
            box-shadow: none;
        }

        .stDataFrame,
        [data-testid="stDataFrame"],
        div[data-testid="stMetric"],
        div[data-testid="stExpander"],
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(6, 6, 9, 0.82);
            border: 1px solid var(--border-color);
            border-radius: 18px;
        }

        div[data-testid="stExpander"] {
            margin-top: 0.85rem;
            overflow: hidden;
        }

        div[data-testid="stMetric"] {
            padding: 1rem;
        }

        .stAlert {
            border-radius: 16px;
            border: 1px solid var(--border-color);
        }

        .ontop-hero {
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.5rem;
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background:
                radial-gradient(circle at top right, rgba(227, 82, 118, 0.25), transparent 35%),
                linear-gradient(135deg, rgba(38, 28, 148, 0.92), rgba(6, 6, 9, 0.94));
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
        }

        .ontop-hero h1,
        .ontop-hero h2,
        .ontop-hero p {
            color: #FFFFFF;
            margin: 0;
        }

        .ontop-hero h2 {
            font-size: clamp(2rem, 3.1vw, 3.2rem);
            line-height: 1.18;
            max-width: 1180px;
        }

        .ontop-hero p {
            margin-top: 1.4rem;
            max-width: 980px;
            font-size: 1.08rem;
            line-height: 1.5;
        }

        .ontop-eyebrow {
            display: inline-block;
            margin-bottom: 0.7rem;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.10);
            color: #FFFFFF;
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .ontop-mini-stats {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.65rem 0 1.35rem;
        }

        .ontop-mini-stat {
            padding: 0.85rem 0.95rem;
            border-radius: 20px;
            border: 1px solid var(--border-color);
            background:
                radial-gradient(circle at top right, rgba(227, 82, 118, 0.10), transparent 35%),
                linear-gradient(180deg, rgba(26, 26, 36, 0.95), rgba(6, 6, 9, 0.95));
        }

        .ontop-mini-stat-purple {
            background:
                radial-gradient(circle at top right, rgba(124, 115, 247, 0.18), transparent 35%),
                linear-gradient(180deg, rgba(27, 24, 57, 0.96), rgba(10, 10, 22, 0.96));
            border-color: rgba(124, 115, 247, 0.22);
        }

        .ontop-mini-stat-green {
            background:
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.18), transparent 35%),
                linear-gradient(180deg, rgba(12, 40, 26, 0.96), rgba(6, 18, 11, 0.96));
            border-color: rgba(34, 197, 94, 0.22);
        }

        .ontop-mini-stat-coral {
            background:
                radial-gradient(circle at top right, rgba(227, 82, 118, 0.18), transparent 35%),
                linear-gradient(180deg, rgba(40, 15, 25, 0.96), rgba(16, 7, 12, 0.96));
            border-color: rgba(227, 82, 118, 0.22);
        }

        .ontop-mini-stat-amber {
            background:
                radial-gradient(circle at top right, rgba(245, 158, 11, 0.18), transparent 35%),
                linear-gradient(180deg, rgba(45, 27, 9, 0.96), rgba(18, 11, 5, 0.96));
            border-color: rgba(245, 158, 11, 0.22);
        }

        .ontop-mini-stat-red {
            background:
                radial-gradient(circle at top right, rgba(239, 68, 68, 0.18), transparent 35%),
                linear-gradient(180deg, rgba(45, 14, 14, 0.96), rgba(20, 7, 7, 0.96));
            border-color: rgba(239, 68, 68, 0.22);
        }

        .ontop-mini-stat span {
            display: block;
            color: var(--text-muted);
            font-size: 0.78rem;
            margin-bottom: 0.3rem;
        }

        .ontop-mini-stat strong {
            display: block;
            color: #FFFFFF;
            font-size: 1.9rem;
            line-height: 1;
        }

        .ontop-section-head {
            margin: 0.25rem 0 0.75rem;
        }

        .ontop-section-head h3 {
            margin: 0;
        }

        .ontop-section-head p {
            margin: 0.25rem 0 0;
            color: var(--text-secondary);
        }

        .ontop-table-shell,
        .ontop-chart-shell {
            padding: 0;
            border-radius: 20px;
            border: 1px solid var(--border-color);
            background:
                radial-gradient(circle at top right, rgba(38, 28, 148, 0.12), transparent 28%),
                linear-gradient(180deg, rgba(6, 6, 9, 0.96), rgba(26, 26, 36, 0.92));
            margin-bottom: 1rem;
            overflow: hidden;
        }

        .ontop-sidebar-brand {
            padding: 0.1rem 0 0.75rem;
        }

        .ontop-sidebar-brand h1 {
            margin: 0.15rem 0 0;
            font-size: 1.55rem;
            line-height: 1;
        }

        .ontop-sidebar-brand p {
            margin: 0.35rem 0 0;
            color: var(--text-muted);
            font-size: 0.82rem;
        }

        .ontop-sidebar-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.28rem 0.55rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #FFFFFF;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .ontop-sidebar-section-label {
            margin: 0.9rem 0 0.4rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            font-weight: 700;
        }

        .ontop-sidebar-user {
            padding: 0.95rem;
            border-radius: 18px;
            border: 1px solid var(--border-color);
            background:
                radial-gradient(circle at top right, rgba(227, 82, 118, 0.14), transparent 36%),
                linear-gradient(180deg, rgba(26, 26, 36, 0.96), rgba(6, 6, 9, 0.96));
            margin-top: 1rem;
            margin-bottom: 0.65rem;
            display: grid;
            grid-template-columns: 2.5rem 1fr;
            gap: 0.75rem;
            align-items: center;
        }

        .ontop-sidebar-avatar {
            width: 2.5rem;
            height: 2.5rem;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, rgba(38, 28, 148, 0.95), rgba(227, 82, 118, 0.8));
            color: #FFFFFF;
            font-size: 0.92rem;
            font-weight: 800;
        }

        .ontop-sidebar-user strong {
            display: block;
            color: #FFFFFF;
            font-size: 1rem;
            margin-bottom: 0.2rem;
        }

        .ontop-sidebar-user span {
            color: var(--text-secondary);
            font-size: 0.85rem;
            word-break: break-word;
        }

        .ontop-sidebar-user-label {
            display: block;
            color: var(--text-muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.22rem;
        }

        [data-testid="stTabs"] [role="tablist"] {
            gap: 0.55rem;
            border-bottom: 1px solid var(--border-color);
        }

        [data-testid="stTabs"] [role="tab"] {
            border-radius: 16px 16px 0 0;
            color: var(--text-secondary);
            font-weight: 600;
        }

        [data-testid="stTabs"] [aria-selected="true"] {
            color: #FFFFFF;
            background: linear-gradient(135deg, rgba(38, 28, 148, 0.46), rgba(227, 82, 118, 0.26));
        }

        hr {
            border-color: var(--border-color);
            margin: 2rem 0;
        }

        code {
            background: rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            color: #FFFFFF;
        }

        @media (max-width: 700px) {
            .ontop-mini-stats {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_hero(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <section class="ontop-hero">
            <span class="ontop-eyebrow">{escape(kicker)}</span>
            <h2>{escape(title)}</h2>
            <p>{escape(subtitle)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_portfolio_cards(accounts: list[dict]) -> None:
    counts = {lane: sum(1 for a in accounts if a["current_lane"] == lane)
              for lane in ("RED", "YELLOW", "GREEN")}
    stale = sum(1 for a in accounts if int(a.get("days_since_last_contact", 0)) > 14)
    arr_total = sum(float(a.get("arr", 0)) for a in accounts)

    cards = [
        ("Total Accounts", f"{len(accounts)}", ""),
        ("Human Touch", f"{counts['RED']}", "ontop-mini-stat-red"),
        ("Tech Touch", f"{counts['YELLOW']}", "ontop-mini-stat-amber"),
        ("Monitor", f"{counts['GREEN']}", "ontop-mini-stat-green"),
        ("Stale Contact", f"{stale}", "ontop-mini-stat-purple"),
        ("Total ARR", f"${arr_total / 1000:.0f}K", "ontop-mini-stat-coral"),
    ]
    html = ['<div class="ontop-mini-stats">']
    for label, value, tone in cards:
        html.append(
            f'<div class="ontop-mini-stat {tone}"><span>{escape(label)}</span>'
            f'<strong>{escape(value)}</strong></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_section_head(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="ontop-section-head">
            <h3>{escape(title)}</h3>
            <p>{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_review_cards(report: dict) -> None:
    total_stable = report["stable_red"] + report["stable_yellow"] + report["stable_green"]
    cards = [
        ("Accounts", str(report["total_accounts"]), ""),
        ("Graduated", str(len(report["graduated"])), "ontop-mini-stat-green"),
        ("Escalated", str(len(report["escalated"])), "ontop-mini-stat-red"),
        ("Stable", str(total_stable), "ontop-mini-stat-purple"),
        ("Stable RED", str(report["stable_red"]), "ontop-mini-stat-coral"),
        ("Stable GREEN", str(report["stable_green"]), "ontop-mini-stat-amber"),
    ]
    html = ['<div class="ontop-mini-stats">']
    for label, value, tone in cards:
        html.append(
            f'<div class="ontop-mini-stat {tone}"><span>{escape(label)}</span>'
            f'<strong>{escape(value)}</strong></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def account_priority_score(account: dict) -> float:
    """Ranks accounts for the Command Center work queue."""
    lane_weight = {"RED": 1000, "YELLOW": 550, "GREEN": 0}.get(account.get("current_lane"), 0)
    stale_bonus = 180 if int(account.get("days_since_last_contact", 0)) > 14 else 0
    ticket_bonus = min(int(account.get("open_tickets", 0)), 8) * 55
    churn_bonus = float(account.get("churn_score", 0)) * 4
    arr_bonus = min(float(account.get("arr", 0)) / 1000, 250)
    return lane_weight + stale_bonus + ticket_bonus + churn_bonus + arr_bonus


def get_priority_accounts(accounts: list[dict], limit: int = 8) -> list[dict]:
    return sorted(accounts, key=account_priority_score, reverse=True)[:limit]


def priority_reason(account: dict) -> str:
    reasons: list[str] = []
    if account.get("current_lane") == "RED":
        reasons.append("human touch")
    elif account.get("current_lane") == "YELLOW":
        reasons.append("tech touch")
    if int(account.get("days_since_last_contact", 0)) > 14:
        reasons.append(f"{account['days_since_last_contact']}d no contact")
    if int(account.get("open_tickets", 0)) > 0:
        reasons.append(f"{account['open_tickets']} tickets")
    if int(account.get("churn_score", 0)) >= 70:
        reasons.append("high health risk")
    return " · ".join(reasons) or "monitor"


def priority_accounts_df(accounts: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Account": a["company_name"],
            "Lane": f"{LANE_EMOJI.get(a['current_lane'], '⚪')} {a['current_lane']}",
            "ARR": a["arr"],
            "Health": a["churn_score"],
            "Reason": priority_reason(a),
        }
        for a in accounts
    ])

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_accounts() -> list[dict]:
    if not ACCOUNTS_FILE.exists():
        return []
    return json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))


def save_accounts(accounts: list[dict]) -> None:
    ACCOUNTS_FILE.write_text(
        json.dumps(accounts, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_approvals() -> list[dict]:
    if not APPROVALS_FILE.exists():
        return []
    try:
        return json.loads(APPROVALS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_approvals(approvals: list[dict]) -> None:
    APPROVALS_FILE.write_text(
        json.dumps(approvals, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def update_approval_status(generated_at: str, status: str, notes: str = "") -> bool:
    """
    Finds the approval record with the given generated_at timestamp and
    updates its status.  Uses generated_at as a unique record identifier.
    """
    approvals = load_approvals()
    for record in approvals:
        if record.get("generated_at") == generated_at:
            record["status"] = status
            record[f"{status}_at"] = datetime.now(timezone.utc).isoformat()
            if notes:
                record["rejection_notes"] = notes
            save_approvals(approvals)
            return True
    return False


def get_latest_report() -> Optional[dict]:
    """Returns the most recent sprint_review_*.json from the reports dir."""
    if not REPORTS_DIR.exists():
        return None
    reports = sorted(REPORTS_DIR.glob("sprint_review_*.json"), reverse=True)
    if not reports:
        return None
    try:
        return json.loads(reports[0].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def accounts_to_df(accounts: list[dict]) -> pd.DataFrame:
    """Converts the accounts list into a display-ready DataFrame."""
    return pd.DataFrame([
        {
            "Company":        a["company_name"],
            "Lane":           f"{LANE_EMOJI.get(a['current_lane'], '⚪')} {a['current_lane']}",
            "Churn Score":    a["churn_score"],
            "Sentiment":      a["sentiment"].capitalize(),
            "ARR":            a["arr"],
            "Days No Contact":a["days_since_last_contact"],
            "Open Tickets":   a["open_tickets"],
            "_lane":          a["current_lane"],  # used for row colouring, hidden below
        }
        for a in accounts
    ])


# ---------------------------------------------------------------------------
# Mock OpenAI context manager
# ---------------------------------------------------------------------------

@contextmanager
def _maybe_mock_openai(mock_content: str):
    """
    When USE_MOCK_DATA=True, patches openai.OpenAI globally so that any
    agent calling openai.OpenAI().chat.completions.create() receives
    mock_content as the response instead of hitting the real API.

    Both tech_touch_agent.py and sprint_review_engine.py access OpenAI
    via `openai.OpenAI()` (attribute access through the module), so a
    single global patch is sufficient for both.
    """
    if not USE_MOCK_DATA:
        yield
        return

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = mock_content
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("openai.OpenAI", return_value=mock_client):
        yield


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="ontop-sidebar-brand">
                <span class="ontop-sidebar-badge">Workspace</span>
                <h1>CS Intelligence</h1>
                <p>Customer success coverage workspace</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if USE_MOCK_DATA:
            st.info("🔧 Mock mode — no real API calls", icon="ℹ️")

        st.divider()

        # Lane distribution metrics
        accounts = load_accounts()
        counts = {lane: sum(1 for a in accounts if a["current_lane"] == lane)
                  for lane in ("RED", "YELLOW", "GREEN")}

        st.markdown('<div class="ontop-sidebar-section-label">Portfolio</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴", counts["RED"])
        c2.metric("🟡", counts["YELLOW"])
        c3.metric("🟢", counts["GREEN"])
        st.caption(f"{len(accounts)} total accounts")

        st.divider()

        # Pending approvals
        pending = sum(1 for a in load_approvals() if a["status"] == "pending_approval")
        approved = sum(1 for a in load_approvals() if a["status"] == "approved")
        st.markdown('<div class="ontop-sidebar-section-label">Email Queue</div>', unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        a1.metric("⏳ Pending", pending)
        a2.metric("✅ Approved", approved)

        st.divider()

        # Agent last-run status
        st.markdown('<div class="ontop-sidebar-section-label">Agent Status</div>', unsafe_allow_html=True)
        router_ts = st.session_state.last_router_run or "—"
        st.caption(f"Coverage Router:  {router_ts}")

        latest = get_latest_report()
        sprint_ts = latest["report_date"] if latest else "—"
        st.caption(f"Sprint Review:    {sprint_ts}")

        st.divider()
        st.caption("Sprint 1 · Mock Data Mode")
        st.markdown(
            """
            <div class="ontop-sidebar-user">
                <div class="ontop-sidebar-avatar">AR</div>
                <div>
                    <span class="ontop-sidebar-user-label">Signed in</span>
                    <strong>Andres Rojas</strong>
                    <span>CS Intelligence Admin</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Command Center
# ---------------------------------------------------------------------------

def render_tab_command_center() -> None:
    render_page_hero(
        "Command Center",
        "Start with the accounts, drafts, and movements that need a decision.",
        "Built for daily CSM execution and leadership review: prioritize the work, inspect risk, and move into the right queue.",
    )

    accounts = load_accounts()
    if not accounts:
        st.warning("No accounts found — check `data/accounts.json`.")
        return

    approvals = load_approvals()
    pending_approvals = [a for a in approvals if a.get("status") == "pending_approval"]
    latest_report = st.session_state.sprint_report or get_latest_report()

    render_portfolio_cards(accounts)

    work_col, queue_col, movement_col = st.columns([1.55, 1, 1], gap="large")

    with work_col:
        render_section_head(
            "Priority Work Queue",
            "Accounts ranked by lane, stale contact, tickets, health score, and ARR.",
        )
        priority_df = priority_accounts_df(get_priority_accounts(accounts))
        st.dataframe(
            priority_df,
            column_config={
                "ARR": st.column_config.NumberColumn("ARR", format="$%,.0f"),
                "Health": st.column_config.ProgressColumn(
                    "Health", min_value=0, max_value=100, format="%d"
                ),
            },
            use_container_width=True,
            hide_index=True,
            height=340,
        )

    with queue_col:
        render_section_head(
            "Approval Queue",
            "Drafts waiting for CSM review before anything goes out.",
        )
        if pending_approvals:
            for rec in pending_approvals[:5]:
                with st.container(border=True):
                    st.markdown(f"**{rec.get('company_name', 'Unknown account')}**")
                    st.caption(f"{rec.get('template_type', 'draft')} · {rec.get('generated_at', '')[:10]}")
                    subject = rec.get("subject", "No subject")
                    st.write(subject)
        else:
            st.info("No pending drafts. Generate tech-touch outreach when YELLOW accounts need a nudge.")

    with movement_col:
        render_section_head(
            "Recent Movement",
            "Latest sprint review signal for leadership and CSM planning.",
        )
        if latest_report:
            st.metric("Graduated", len(latest_report.get("graduated", [])))
            st.metric("Escalated", len(latest_report.get("escalated", [])))
            st.caption(f"Last review: {latest_report.get('report_date', '—')}")
            escalated = latest_report.get("escalated", [])
            if escalated:
                st.markdown("**Top escalation**")
                st.write(escalated[0]["company"])
                st.caption(escalated[0]["reason"])
        else:
            st.info("No sprint review yet. Run one to populate account movement.")

    st.divider()

    csm_col, vp_col, exec_col = st.columns(3, gap="large")
    with csm_col:
        render_section_head("CSM View", "What should I do today?")
        st.markdown(
            "- Work RED accounts first\n"
            "- Approve or reject pending drafts\n"
            "- Follow up on stale YELLOW accounts"
        )
    with vp_col:
        render_section_head("VP View", "Where is the team exposed?")
        st.markdown(
            "- Watch escalations and ARR concentration\n"
            "- Compare portfolio movement by sprint\n"
            "- Track approval queue quality and volume"
        )
    with exec_col:
        render_section_head("C-Level View", "What changed in customer health?")
        st.markdown(
            "- Monitor ARR at risk\n"
            "- Review net movement in customer health\n"
            "- Identify strategic accounts needing intervention"
        )


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

def render_tab_accounts() -> None:
    render_page_hero(
        "Portfolio",
        "Monitor account health, risk lanes, and coverage movement.",
        "Use this view to spot human-touch accounts, route tech-touch coverage, and keep the book of business current.",
    )

    accounts = load_accounts()
    if not accounts:
        st.warning("No accounts found — check `data/accounts.json`.")
        return

    render_portfolio_cards(accounts)

    # --- Coverage Router re-classify ---
    control_copy, control_action = st.columns([2.8, 1], vertical_alignment="bottom")
    with control_copy:
        render_section_head(
            "Account Health",
            "Filter the portfolio and re-run lane assignment when account signals change.",
        )
    with control_action:
        run_router_clicked = st.button("Run Coverage Router", type="primary", use_container_width=True)

    # --- Lane filter buttons ---
    col_all, col_red, col_yel, col_grn, _spacer = st.columns([1, 1, 1, 1, 5])

    if col_all.button("All",       use_container_width=True):
        st.session_state.lane_filter = "ALL"
    if col_red.button("🔴 RED",    use_container_width=True):
        st.session_state.lane_filter = "RED"
    if col_yel.button("🟡 YELLOW", use_container_width=True):
        st.session_state.lane_filter = "YELLOW"
    if col_grn.button("🟢 GREEN",  use_container_width=True):
        st.session_state.lane_filter = "GREEN"

    lane_filter = st.session_state.lane_filter
    filtered = accounts if lane_filter == "ALL" else [
        a for a in accounts if a["current_lane"] == lane_filter
    ]

    search_col, sort_col = st.columns([2.2, 1], gap="large")
    search_query = search_col.text_input(
        "Search accounts",
        value=st.session_state.portfolio_search,
        placeholder="Company name or account id",
    ).strip()
    st.session_state.portfolio_search = search_query
    sort_mode = sort_col.selectbox(
        "Sort by",
        options=[
            "Priority",
            "ARR",
            "Churn Score",
            "Days Since Contact",
            "Open Tickets",
            "Company",
        ],
    )

    if search_query:
        q = search_query.lower()
        filtered = [
            a for a in filtered
            if q in a["company_name"].lower() or q in a["account_id"].lower()
        ]

    sorters = {
        "Priority": lambda a: account_priority_score(a),
        "ARR": lambda a: float(a.get("arr", 0)),
        "Churn Score": lambda a: int(a.get("churn_score", 0)),
        "Days Since Contact": lambda a: int(a.get("days_since_last_contact", 0)),
        "Open Tickets": lambda a: int(a.get("open_tickets", 0)),
        "Company": lambda a: a.get("company_name", "").lower(),
    }
    filtered = sorted(
        filtered,
        key=sorters[sort_mode],
        reverse=sort_mode != "Company",
    )

    if run_router_clicked:
        changes: list[dict] = []
        updated: list[dict] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        with st.spinner("Running Coverage Router on all accounts…"):
            for acct in accounts:
                result = route_account({
                    "account_id":             acct["account_id"],
                    "account_name":           acct["company_name"],
                    "churn_score":            acct["churn_score"],
                    "sentiment":              acct["sentiment"],
                    "arr":                    acct["arr"],
                    "days_since_last_contact":acct["days_since_last_contact"],
                    "open_tickets":           acct["open_tickets"],
                })

                new_lane = result["lane"]
                old_lane = acct["current_lane"]

                row = {**acct}
                if new_lane != old_lane:
                    changes.append({
                        "company":  acct["company_name"],
                        "id":       acct["account_id"],
                        "from":     old_lane,
                        "to":       new_lane,
                        "rules":    result["triggered_rules"],
                    })
                    row["previous_lane"]     = old_lane
                    row["current_lane"]      = new_lane
                    row["lane_assigned_date"]= now_iso

                updated.append(row)

        save_accounts(updated)
        st.session_state.router_changes  = changes
        st.session_state.last_router_run = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        st.rerun()

    # --- Styled dataframe + router result ---
    table_col, router_col = st.columns([2.2, 1], gap="large")

    with table_col:
        if not filtered:
            st.info("No accounts match the current filter and search.")
            df = pd.DataFrame()
        else:
            df = accounts_to_df(filtered)

        def _row_style(row: pd.Series) -> list[str]:
            bg = LANE_BG.get(row["_lane"], "")
            return [f"background-color: {bg}"] * len(row)

        if not df.empty:
            styled = (
                df.style
                .apply(_row_style, axis=1)
                .hide(subset=["_lane"], axis="columns")
                .format({"ARR": "${:,.0f}"})
            )

            st.dataframe(
                styled,
                column_config={
                    "Churn Score": st.column_config.ProgressColumn(
                        "Churn Score", min_value=0, max_value=100, format="%d"
                    ),
                    "ARR": st.column_config.NumberColumn("ARR", format="$%,.0f"),
                },
                use_container_width=True,
                hide_index=True,
                height=430,
            )
        st.caption(f"Showing {len(filtered)} of {len(accounts)} accounts  ·  "
                   f"Active filter: **{lane_filter}**")

    with router_col:
        render_section_head(
            "Router Run",
            "Latest account movements from deterministic lane rules.",
        )

        if st.session_state.router_changes is None:
            st.info("Run the router to compare stored lanes with current account signals.")
        elif not st.session_state.router_changes:
            st.success("Coverage Router complete. All accounts are already in the correct lane.")
        else:
            changes = st.session_state.router_changes
            st.success(f"{len(changes)} account(s) changed lane")

            for ch in changes:
                old_e = LANE_EMOJI.get(ch["from"], "⚪")
                new_e = LANE_EMOJI.get(ch["to"], "⚪")
                with st.container(border=True):
                    st.markdown(f"**{ch['company']}**")
                    st.caption(f"`{ch['id']}`")
                    st.markdown(f"{old_e} {ch['from']} → {new_e} {ch['to']}")
                    st.caption("Rules fired: " + "  |  ".join(ch["rules"]))


# ---------------------------------------------------------------------------
# Tab 2 — Tech-Touch Agent
# ---------------------------------------------------------------------------

# Context fields rendered per template type
# Each field: key, display label, input type, default value, extra options
_TEMPLATE_FIELDS: dict[str, list[dict]] = {
    "check_in": [
        {"key": "days_since_last_contact", "label": "Days since last contact", "type": "number", "default": 18},
        {"key": "usage_trend",             "label": "Usage trend note",         "type": "text",
         "default": "active contractors dropped from 12 to 8 over the past month"},
    ],
    "re_engagement": [
        {"key": "previous_contractors", "label": "Contractors active previously", "type": "number", "default": 20},
        {"key": "active_contractors",   "label": "Contractors active now",        "type": "number", "default": 12},
        {"key": "usage_drop_pct",       "label": "Usage drop (%)",                "type": "number", "default": 40},
    ],
    "collections": [
        {"key": "days_past_due",      "label": "Days past due",            "type": "number", "default": 30},
        {"key": "amount_overdue",     "label": "Amount overdue ($)",       "type": "number", "default": 5000},
        {"key": "previous_attempts",  "label": "Previous contact attempts","type": "number", "default": 2},
        {"key": "response_deadline",  "label": "Response deadline",        "type": "text",   "default": "within 5 business days"},
        {"key": "escalation_level",   "label": "Escalation level",         "type": "select",
         "default": "first", "options": ["first", "second", "final"]},
    ],
    "milestone": [
        {"key": "milestone_type",   "label": "Milestone type",   "type": "text",
         "default": "1-year anniversary with Ontop"},
        {"key": "milestone_detail", "label": "Milestone detail", "type": "text",
         "default": "100 contractors successfully onboarded"},
    ],
}

_TEMPLATE_LABELS = {
    "check_in":     "Check-in (14+ days no contact)",
    "re_engagement":"Re-engagement (usage drop)",
    "collections":  "Collections (overdue invoice)",
    "milestone":    "Milestone celebration",
}


def _render_context_fields(template_type: str) -> dict:
    """
    Renders the dynamic input fields for the selected template and
    returns a dict of {field_key: value} to pass as agent context.
    """
    fields = _TEMPLATE_FIELDS.get(template_type, [])
    context: dict = {}
    cols = st.columns(2)

    for i, field in enumerate(fields):
        col = cols[i % 2]
        widget_key = f"ctx_{template_type}_{field['key']}"

        if field["type"] == "number":
            context[field["key"]] = col.number_input(
                field["label"], value=int(field["default"]), min_value=0, key=widget_key
            )
        elif field["type"] == "select":
            opts = field["options"]
            context[field["key"]] = col.selectbox(
                field["label"], options=opts,
                index=opts.index(field["default"]),
                key=widget_key,
            )
        else:
            context[field["key"]] = col.text_input(
                field["label"], value=field["default"], key=widget_key
            )

    return context


def render_tab_tech_touch() -> None:
    render_page_hero(
        "Tech-Touch Queue",
        "Approve drafts and compose outreach for accounts that need a light human nudge.",
        "Use the queue as the daily review surface, then generate new YELLOW-lane messages when coverage needs automation.",
    )

    accounts = load_accounts()
    yellow_accounts = [a for a in accounts if a["current_lane"] == "YELLOW"]

    if not yellow_accounts:
        st.info(
            "No YELLOW-lane accounts found.  "
            "Run the Coverage Router in the Account Overview tab to classify accounts."
        )
        return

    approvals = load_approvals()
    pending = sum(1 for a in approvals if a["status"] == "pending_approval")
    approved = sum(1 for a in approvals if a["status"] == "approved")
    st.markdown(
        f"""
        <div class="ontop-mini-stats" style="grid-template-columns: repeat(4, minmax(0, 1fr));">
            <div class="ontop-mini-stat ontop-mini-stat-amber"><span>YELLOW Accounts</span><strong>{len(yellow_accounts)}</strong></div>
            <div class="ontop-mini-stat ontop-mini-stat-purple"><span>Templates</span><strong>{len(_TEMPLATE_FIELDS)}</strong></div>
            <div class="ontop-mini-stat ontop-mini-stat-coral"><span>Pending Drafts</span><strong>{pending}</strong></div>
            <div class="ontop-mini-stat ontop-mini-stat-green"><span>Approved</span><strong>{approved}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if pending:
        with st.expander("Pending approval queue", expanded=True):
            pending_rows = pd.DataFrame([
                {
                    "Account": rec.get("company_name", ""),
                    "Template": rec.get("template_type", ""),
                    "Subject": rec.get("subject", ""),
                    "Generated": rec.get("generated_at", "")[:16].replace("T", " "),
                }
                for rec in approvals if rec.get("status") == "pending_approval"
            ])
            st.dataframe(pending_rows, use_container_width=True, hide_index=True, height=220)

    col_form, col_preview = st.columns([0.95, 1.35], gap="large")

    with col_form:
        render_section_head(
            "Compose",
            "Choose the account, outreach play, and context for the draft.",
        )

        # Account selector (YELLOW only)
        account_map = {a["company_name"]: a for a in yellow_accounts}
        company = st.selectbox("Account (YELLOW lane only)", options=list(account_map.keys()))
        acct = account_map[company]

        # Template type
        template_type = st.selectbox(
            "Template type",
            options=list(_TEMPLATE_FIELDS.keys()),
            format_func=lambda t: _TEMPLATE_LABELS.get(t, t),
        )

        # Contact fields
        c1, c2 = st.columns(2)
        client_name = c1.text_input("Client contact name", value="Maria Lopez")
        csm_name    = c2.text_input("CSM name",            value="Andrés")

        st.markdown("**Template context**")
        context = _render_context_fields(template_type)

        generate_clicked = st.button("✉️ Generate Email", type="primary", use_container_width=True)

    # --- Handle generation ---
    if generate_clicked:
        mock_content = json.dumps(_MOCK_EMAILS.get(template_type, _MOCK_EMAILS["check_in"]))
        with col_form:
            with st.spinner("Generating…"):
                try:
                    with _maybe_mock_openai(mock_content):
                        result = run_tech_touch({
                            "account_id":   acct["account_id"],
                            "company_name": acct["company_name"],
                            "client_name":  client_name,
                            "csm_name":     csm_name,
                            "template_type":template_type,
                            "lane":         "YELLOW",
                            "context":      context,
                        })

                    if result.get("error"):
                        st.error(f"Agent error: {result['error']}")
                    else:
                        st.session_state.tt_generated = result["approval_record"]
                        st.session_state.tt_status    = "pending_approval"
                        st.rerun()

                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

    # --- Preview pane ---
    with col_preview:
        render_section_head(
            "Approval Preview",
            "Review the generated message before it enters the approval queue.",
        )

        if not st.session_state.tt_generated:
            st.markdown(
                "<div class='ontop-table-shell' style='padding:3rem 1.25rem;"
                "text-align:center;color:#B8B8C8;font-size:0.95rem'>"
                "Configure and generate an email to see the preview here."
                "</div>",
                unsafe_allow_html=True,
            )
            return

        rec    = st.session_state.tt_generated
        status = st.session_state.tt_status

        # Status badge
        badge = {
            "pending_approval": "🟡 Pending Approval",
            "approved":         "✅ Approved",
            "rejecting":        "⏸️ Awaiting rejection notes",
            "rejected":         "❌ Rejected",
        }.get(status, status)
        st.markdown(f"**Status:** {badge}")

        # Email preview card
        with st.container(border=True):
            st.markdown(
                f"**To:** {rec.get('client_name', '')} — {rec.get('company_name', '')}"
            )
            st.markdown(f"**From:** {rec.get('csm_name', '')}, Ontop CS")
            st.markdown(f"**Subject:** {rec.get('subject', '')}")
            st.divider()
            st.write(rec.get("body", ""))
            st.caption(f"Generated: {rec.get('generated_at', '')[:16].replace('T', ' ')} UTC")

        if status == "pending_approval":
            b1, b2 = st.columns(2)
            if b1.button("✅ Approve", type="primary", use_container_width=True):
                ok = update_approval_status(rec["generated_at"], "approved")
                if ok:
                    st.session_state.tt_status = "approved"
                    st.rerun()
                else:
                    st.error("Could not find record in approvals.json")

            if b2.button("❌ Reject", use_container_width=True):
                st.session_state.tt_status = "rejecting"
                st.rerun()

        elif status == "rejecting":
            notes = st.text_area("Rejection notes (optional)",
                                 placeholder="e.g. Tone is too casual, regenerate")
            c_confirm, c_cancel = st.columns(2)
            if c_confirm.button("Confirm Rejection", type="secondary", use_container_width=True):
                update_approval_status(rec["generated_at"], "rejected", notes)
                st.session_state.tt_status = "rejected"
                st.rerun()
            if c_cancel.button("Cancel", use_container_width=True):
                st.session_state.tt_status = "pending_approval"
                st.rerun()

        elif status == "approved":
            st.success("Email approved and queued.  Sprint 2: will be sent automatically via n8n.")
            if st.button("✉️ Generate another"):
                st.session_state.tt_generated = None
                st.session_state.tt_status    = None
                st.rerun()

        elif status == "rejected":
            st.warning("Email rejected.  Adjust the context fields and generate a new version.")
            if st.button("↩️ Start over"):
                st.session_state.tt_generated = None
                st.session_state.tt_status    = None
                st.rerun()


# ---------------------------------------------------------------------------
# Tab 3 — Sprint Review
# ---------------------------------------------------------------------------

def render_tab_sprint_review() -> None:
    render_page_hero(
        "Sprint Review",
        "Review account movement and prioritize the next CSM actions.",
        "Run the bi-weekly rotation to detect graduations, escalations, and the accounts that need attention now.",
    )

    latest_report = get_latest_report()

    # Header row
    hdr_col, btn_col = st.columns([3, 1])
    with hdr_col:
        render_section_head(
            "Review Control",
            "Run a portfolio rotation and compare current health signals against stored lanes.",
        )
        if latest_report:
            n_grad = len(latest_report["graduated"])
            n_esc  = len(latest_report["escalated"])
            st.markdown(
                f"**Last review:** {latest_report['report_date']}  ·  "
                f"**Accounts:** {latest_report['total_accounts']}  ·  "
                f"🟢 Graduated: **{n_grad}**  ·  🔴 Escalated: **{n_esc}**"
            )
        else:
            st.info("No sprint reviews found.  Run the engine to generate the first report.")

    with btn_col:
        run_clicked = st.button("🚀 Run Sprint Review", type="primary", use_container_width=True)

    # --- Handle sprint review run ---
    if run_clicked:
        mock_content = json.dumps(_MOCK_SPRINT_SUMMARY)
        with st.spinner("Running sprint review across all accounts…"):
            try:
                with _maybe_mock_openai(mock_content):
                    state = run_sprint_review()
                st.session_state.sprint_report   = state["report"]
                st.session_state.last_sprint_run = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                st.rerun()

            except Exception as exc:
                st.error(f"Sprint Review failed: {exc}")
                return

    # Prefer the in-session result (just ran) over the on-disk result (prior run)
    report = st.session_state.sprint_report or latest_report
    if not report:
        return

    render_review_cards(report)

    # --- Narrative + action items ---
    narr_col, items_col = st.columns([1.3, 1], gap="large")

    with narr_col:
        render_section_head(
            "Narrative Summary",
            "Portfolio health and the priorities for the current review.",
        )
        with st.container(border=True):
            st.write(report.get("narrative_summary", ""))

    with items_col:
        render_section_head(
            "CSM Action Items",
            "Check off the highest priority follow-ups for this sprint.",
        )
        items = report.get("csm_action_items", [])
        if items:
            for i, item in enumerate(items):
                # Use report_date in key to avoid key collisions across runs
                st.checkbox(item, key=f"action_{report.get('report_date','x')}_{i}")
        else:
            st.caption("No action items generated")

    st.divider()
    render_section_head(
        "Lane Movement",
        "Graduations, escalations, and stable accounts from the latest review.",
    )

    # --- Three-column lane change results ---
    col_g, col_e, col_s = st.columns(3, gap="medium")

    with col_g:
        count = len(report["graduated"])
        st.markdown(f"### Graduated  ({count})")
        st.caption("Accounts moving to a healthier lane")
        if report["graduated"]:
            for a in report["graduated"]:
                with st.container(border=True):
                    st.markdown(f"**{a['company']}**")
                    st.markdown(
                        f"{LANE_EMOJI.get(a['from'],'⚪')} {a['from']} → "
                        f"{LANE_EMOJI.get(a['to'],'⚪')} {a['to']}"
                    )
                    st.caption(a["reason"])
        else:
            st.caption("No graduations this sprint")

    with col_e:
        count = len(report["escalated"])
        st.markdown(f"### Escalated  ({count})")
        st.caption("Accounts requiring CSM attention")
        if report["escalated"]:
            for a in report["escalated"]:
                with st.container(border=True):
                    st.markdown(f"**{a['company']}**")
                    st.markdown(
                        f"{LANE_EMOJI.get(a['from'],'⚪')} {a['from']} → "
                        f"{LANE_EMOJI.get(a['to'],'⚪')} {a['to']}"
                    )
                    st.caption(a["reason"])
        else:
            st.caption("No escalations this sprint")

    with col_s:
        total_stable = report["stable_red"] + report["stable_yellow"] + report["stable_green"]
        st.markdown(f"### Stable  ({total_stable})")
        st.caption("No lane change this sprint")
        st.metric("🔴 Stable RED",    report["stable_red"])
        st.metric("🟡 Stable YELLOW", report["stable_yellow"])
        st.metric("🟢 Stable GREEN",  report["stable_green"])


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

inject_theme()
render_sidebar()

tab1, tab2, tab3, tab4 = st.tabs([
    "Command Center",
    "Portfolio",
    "Tech-Touch Queue",
    "Sprint Review",
])

with tab1:
    render_tab_command_center()
with tab2:
    render_tab_accounts()
with tab3:
    render_tab_tech_touch()
with tab4:
    render_tab_sprint_review()
