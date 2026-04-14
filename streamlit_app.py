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
    """Applies the dark workspace styling used across the app."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #030306;
            --panel: #0b0b12;
            --panel-2: #11111b;
            --panel-3: #171623;
            --line: rgba(181, 176, 255, 0.22);
            --line-strong: rgba(181, 176, 255, 0.36);
            --text: #f4f1ff;
            --muted: #aaa7bb;
            --dim: #727081;
            --violet: #5d43ff;
            --pink: #ff5f9f;
            --green: #20d47b;
            --amber: #f4ae32;
            --red: #ff4b5f;
        }

        .stApp {
            background:
                radial-gradient(circle at 80% 0%, rgba(255, 95, 159, 0.18), transparent 30rem),
                radial-gradient(circle at 18% 4%, rgba(93, 67, 255, 0.24), transparent 34rem),
                linear-gradient(180deg, #060611 0%, var(--bg) 42%, #000000 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1480px;
            padding-top: 3.3rem;
            padding-bottom: 4rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #090912 0%, #050507 56%, #030304 100%);
            border-right: 1px solid var(--line-strong);
        }

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 1rem;
        }

        [data-testid="stSidebar"] h1 {
            font-size: 1.62rem;
            letter-spacing: 0;
            margin-top: 1rem;
        }

        [data-testid="stSidebar"] h3 {
            color: var(--dim);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        [data-testid="stSidebar"] [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid rgba(181, 176, 255, 0.14);
            border-radius: 8px;
            padding: 0.7rem 0.75rem;
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--text);
            letter-spacing: 0;
        }

        p, li, label, [data-testid="stMarkdownContainer"] {
            color: var(--muted);
        }

        .cs-hero {
            margin: 0.4rem 0 2.1rem;
            padding: 2.35rem 2.55rem;
            border: 1px solid rgba(255, 95, 159, 0.28);
            border-radius: 8px;
            background:
                linear-gradient(115deg, rgba(93, 67, 255, 0.62) 0%, rgba(33, 24, 91, 0.52) 48%, rgba(255, 95, 159, 0.16) 100%),
                rgba(13, 13, 22, 0.92);
            box-shadow: 0 22px 70px rgba(0, 0, 0, 0.36), inset 0 1px 0 rgba(255,255,255,0.08);
        }

        .cs-kicker {
            display: inline-flex;
            align-items: center;
            min-height: 2rem;
            padding: 0.28rem 0.9rem;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.12);
            color: #ffffff;
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .cs-hero h1 {
            max-width: 1100px;
            margin: 1.65rem 0 1.25rem;
            color: #ffffff;
            font-size: clamp(2.2rem, 4.2vw, 4rem);
            line-height: 1.16;
            font-weight: 850;
            letter-spacing: 0;
        }

        .cs-hero p {
            max-width: 980px;
            margin: 0;
            color: rgba(255, 255, 255, 0.82);
            font-size: 1.08rem;
            line-height: 1.55;
            font-weight: 600;
        }

        .cs-metric-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 1.25rem 0 2rem;
        }

        .cs-card {
            min-height: 6.25rem;
            padding: 1.1rem 1.25rem;
            border-radius: 8px;
            border: 1px solid rgba(181, 176, 255, 0.2);
            background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.015));
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
        }

        .cs-card.red { border-color: rgba(255, 75, 95, 0.42); background-color: rgba(255, 75, 95, 0.10); }
        .cs-card.yellow { border-color: rgba(244, 174, 50, 0.42); background-color: rgba(244, 174, 50, 0.10); }
        .cs-card.green { border-color: rgba(32, 212, 123, 0.42); background-color: rgba(32, 212, 123, 0.10); }
        .cs-card.violet { border-color: rgba(93, 67, 255, 0.46); background-color: rgba(93, 67, 255, 0.12); }
        .cs-card.pink { border-color: rgba(255, 95, 159, 0.44); background-color: rgba(255, 95, 159, 0.10); }
        .cs-card.neutral { border-color: rgba(181, 176, 255, 0.32); }

        .cs-card span {
            display: block;
            color: var(--dim);
            font-size: 0.82rem;
            font-weight: 750;
            letter-spacing: 0.02em;
        }

        .cs-card strong {
            display: block;
            margin-top: 0.62rem;
            color: #ffffff;
            font-size: 2rem;
            line-height: 1;
            font-weight: 850;
            letter-spacing: 0;
        }

        div[data-testid="stButton"] > button {
            min-height: 3.2rem;
            border-radius: 8px;
            border: 1px solid rgba(181, 176, 255, 0.28);
            background: rgba(17, 17, 27, 0.82);
            color: #f5f2ff;
            font-weight: 750;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
        }

        div[data-testid="stButton"] > button:hover {
            border-color: rgba(255, 95, 159, 0.62);
            background: rgba(24, 21, 38, 0.95);
            color: #ffffff;
        }

        div[data-testid="stButton"] > button[kind="primary"],
        button[data-testid="stBaseButton-primary"] {
            background: linear-gradient(90deg, rgba(93, 67, 255, 0.88), rgba(255, 95, 159, 0.66));
            border-color: rgba(255, 255, 255, 0.18);
            color: #ffffff;
        }

        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        textarea {
            background-color: rgba(8, 8, 15, 0.92) !important;
            border: 1px solid rgba(181, 176, 255, 0.2) !important;
            border-radius: 8px !important;
            color: #ffffff !important;
        }

        [data-testid="stTabs"] [role="tablist"] {
            gap: 0.7rem;
            border-bottom: 1px solid rgba(181, 176, 255, 0.18);
        }

        [data-testid="stTabs"] [role="tab"] {
            min-height: 3rem;
            padding: 0 1.15rem;
            border-radius: 8px 8px 0 0;
            color: var(--muted);
            font-weight: 800;
        }

        [data-testid="stTabs"] [aria-selected="true"] {
            color: #ffffff;
            background: rgba(93, 67, 255, 0.18);
            border: 1px solid rgba(181, 176, 255, 0.22);
            border-bottom-color: transparent;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid rgba(181, 176, 255, 0.18);
            border-radius: 8px;
            overflow: hidden;
            background: rgba(6, 6, 12, 0.82);
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(181, 176, 255, 0.16);
            border-radius: 8px;
            padding: 1rem 1.05rem;
        }

        [data-testid="stExpander"], [data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: rgba(181, 176, 255, 0.2) !important;
            border-radius: 8px !important;
            background: rgba(8, 8, 15, 0.68) !important;
        }

        hr {
            border-color: rgba(181, 176, 255, 0.18);
            margin: 2rem 0;
        }

        .cs-user-card {
            margin-top: 1rem;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid rgba(255, 95, 159, 0.24);
            background: linear-gradient(120deg, rgba(93, 67, 255, 0.15), rgba(255, 95, 159, 0.10));
        }

        .cs-user-card .label {
            color: var(--dim);
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .cs-user-card .name {
            margin-top: 0.7rem;
            color: #ffffff;
            font-size: 1rem;
            font-weight: 850;
        }

        .cs-user-card .email {
            margin-top: 0.25rem;
            color: var(--muted);
            font-size: 0.86rem;
        }

        @media (max-width: 1100px) {
            .cs-metric-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
            .cs-hero { padding: 2rem; }
        }

        @media (max-width: 720px) {
            .block-container { padding-top: 1.4rem; }
            .cs-metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .cs-hero { padding: 1.5rem; }
            .cs-hero h1 { font-size: 2rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_hero(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <section class="cs-hero">
            <div class="cs-kicker">{escape(kicker)}</div>
            <h1>{escape(title)}</h1>
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
        ("Total Accounts", f"{len(accounts)}", "neutral"),
        ("Human Touch", f"{counts['RED']}", "red"),
        ("Tech Touch", f"{counts['YELLOW']}", "yellow"),
        ("Monitor", f"{counts['GREEN']}", "green"),
        ("Stale Contact", f"{stale}", "violet"),
        ("Total ARR", f"${arr_total / 1000:.0f}K", "pink"),
    ]
    html = ['<div class="cs-metric-grid">']
    for label, value, tone in cards:
        html.append(
            f'<div class="cs-card {tone}"><span>{escape(label)}</span>'
            f'<strong>{escape(value)}</strong></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

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
        st.markdown("### WORKSPACE")
        st.title("CS Intelligence")
        st.caption("Customer success coverage workspace")

        if USE_MOCK_DATA:
            st.info("🔧 Mock mode — no real API calls", icon="ℹ️")

        st.divider()

        # Lane distribution metrics
        accounts = load_accounts()
        counts = {lane: sum(1 for a in accounts if a["current_lane"] == lane)
                  for lane in ("RED", "YELLOW", "GREEN")}

        st.subheader("Portfolio")
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴", counts["RED"])
        c2.metric("🟡", counts["YELLOW"])
        c3.metric("🟢", counts["GREEN"])
        st.caption(f"{len(accounts)} total accounts")

        st.divider()

        # Pending approvals
        pending = sum(1 for a in load_approvals() if a["status"] == "pending_approval")
        approved = sum(1 for a in load_approvals() if a["status"] == "approved")
        st.subheader("Email Queue")
        a1, a2 = st.columns(2)
        a1.metric("⏳ Pending", pending)
        a2.metric("✅ Approved", approved)

        st.divider()

        # Agent last-run status
        st.subheader("Agent Status")
        router_ts = st.session_state.last_router_run or "—"
        st.caption(f"Coverage Router:  {router_ts}")

        latest = get_latest_report()
        sprint_ts = latest["report_date"] if latest else "—"
        st.caption(f"Sprint Review:    {sprint_ts}")

        st.divider()
        st.caption("Sprint 1 · Mock Data Mode")
        st.markdown(
            """
            <div class="cs-user-card">
                <div class="label">Signed in</div>
                <div class="name">Andres Rojas</div>
                <div class="email">CS Intelligence Admin</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 1 — Account Overview
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

    # --- Styled dataframe ---
    df = accounts_to_df(filtered)

    def _row_style(row: pd.Series) -> list[str]:
        bg = LANE_BG.get(row["_lane"], "")
        return [f"background-color: {bg}"] * len(row)

    visible_cols = ["Company", "Lane", "Churn Score", "Sentiment", "ARR",
                    "Days No Contact", "Open Tickets"]

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
    )
    st.caption(f"Showing {len(filtered)} of {len(accounts)} accounts  ·  "
               f"Active filter: **{lane_filter}**")

    st.divider()

    # --- Coverage Router re-classify ---
    st.subheader("Re-classify Accounts")
    st.write("Re-run the Coverage Router against current signal values and update lane assignments in `data/accounts.json`.")

    if st.button("🔄 Run Coverage Router", type="primary"):
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

    # Show diff from last run
    if st.session_state.router_changes is None:
        pass  # Never run — show nothing
    elif not st.session_state.router_changes:
        st.info("✅ Coverage Router ran — all accounts are already in the correct lane.")
    else:
        changes = st.session_state.router_changes
        st.success(f"✅ Coverage Router complete — **{len(changes)}** account(s) changed lane")

        for ch in changes:
            old_e = LANE_EMOJI.get(ch["from"], "⚪")
            new_e = LANE_EMOJI.get(ch["to"], "⚪")
            with st.container(border=True):
                col_name, col_change = st.columns([2, 1])
                col_name.markdown(f"**{ch['company']}** · `{ch['id']}`")
                col_change.markdown(f"{old_e} {ch['from']} → {new_e} {ch['to']}")
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
        "Tech Touch",
        "Draft customer outreach for accounts that need a light human nudge.",
        "Generate tailored emails for YELLOW-lane accounts and keep every message in the CSM approval queue.",
    )

    accounts = load_accounts()
    yellow_accounts = [a for a in accounts if a["current_lane"] == "YELLOW"]

    if not yellow_accounts:
        st.info(
            "No YELLOW-lane accounts found.  "
            "Run the Coverage Router in the Account Overview tab to classify accounts."
        )
        return

    col_form, col_preview = st.columns([1, 1], gap="large")

    with col_form:
        st.subheader("Configure")

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
        st.subheader("Preview")

        if not st.session_state.tt_generated:
            st.markdown(
                "<div style='border:1px dashed #444;border-radius:8px;padding:48px;"
                "text-align:center;color:#777;font-size:0.9rem'>"
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

    st.divider()

    # --- Three-column lane change results ---
    col_g, col_e, col_s = st.columns(3, gap="medium")

    with col_g:
        count = len(report["graduated"])
        st.markdown(f"### 🟢 Graduated  ({count})")
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
        st.markdown(f"### 🔴 Escalated  ({count})")
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
        st.markdown(f"### ➡️ Stable  ({total_stable})")
        st.caption("No lane change this sprint")
        st.metric("🔴 Stable RED",    report["stable_red"])
        st.metric("🟡 Stable YELLOW", report["stable_yellow"])
        st.metric("🟢 Stable GREEN",  report["stable_green"])

    st.divider()

    # --- Narrative + action items ---
    narr_col, items_col = st.columns([3, 2], gap="large")

    with narr_col:
        st.subheader("📋 Narrative Summary")
        with st.container(border=True):
            st.write(report.get("narrative_summary", ""))

    with items_col:
        st.subheader("✅ CSM Action Items")
        items = report.get("csm_action_items", [])
        if items:
            for i, item in enumerate(items):
                # Use report_date in key to avoid key collisions across runs
                st.checkbox(item, key=f"action_{report.get('report_date','x')}_{i}")
        else:
            st.caption("No action items generated")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

inject_theme()
render_sidebar()

tab1, tab2, tab3 = st.tabs([
    "📊 Account Overview",
    "✉️ Tech-Touch Agent",
    "🔄 Sprint Review",
])

with tab1:
    render_tab_accounts()
with tab2:
    render_tab_tech_touch()
with tab3:
    render_tab_sprint_review()
