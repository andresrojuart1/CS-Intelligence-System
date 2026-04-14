# CS Intelligence System — Full Project Context

> This document is the single source of truth for an LLM continuing work on
> this codebase. It captures architecture, data models, agent contracts,
> design decisions, and the Sprint 2 roadmap. Read this before touching any file.

---

## 1. What This System Does

Ontop is a B2B SaaS company offering global payroll and contractor management.
One CSM manages ~600 customer accounts. This system automates coverage by:

1. **Classifying** every account into a risk lane (RED / YELLOW / GREEN)
2. **Generating** personalised outreach emails for medium-risk accounts
3. **Running bi-weekly reviews** to detect accounts improving or deteriorating
4. **Surfacing everything** in a Streamlit dashboard with a CSM approval queue

The system is intentionally non-autonomous in Sprint 1. Every generated
email requires explicit CSM approval before sending. Autonomous sending
is planned for Sprint 2 once quality is validated.

---

## 2. Repository Structure

```
CS Intelligence System/
├── agents/
│   ├── coverage_router.py        # LangGraph — classifies accounts RED/YELLOW/GREEN
│   ├── tech_touch_agent.py       # LangGraph — generates emails + approval queue
│   └── sprint_review_engine.py   # LangGraph — bi-weekly rotation engine
├── integrations/
│   ├── salesforce.py             # Stub (Sprint 2)
│   ├── zendesk.py                # Stub (Sprint 2)
│   └── morpheus.py               # Stub (Sprint 2)
├── prompts/
│   ├── check_in.txt              # 14+ days no contact
│   ├── re_engagement.txt         # Usage drop
│   ├── collections.txt           # Overdue invoice
│   ├── milestone.txt             # Milestone celebration
│   └── save_brief.txt            # Internal CSM action brief (not an email)
├── data/
│   └── accounts.json             # 10 mock accounts (source of truth for Sprint 1)
├── reports/                      # Auto-created; sprint_review_{date}.json files saved here
├── tests/
│   ├── test_coverage_router.py   # 9 tests
│   ├── test_tech_touch_agent.py  # 7 tests
│   └── test_sprint_review_engine.py  # 10 tests
├── streamlit_app.py              # Operational dashboard (3 tabs + sidebar)
├── approvals.json                # Auto-created; email approval queue
├── requirements.txt
└── .env.example
```

---

## 3. Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Agent framework | LangGraph | 0.2.74 |
| LLM (email generation) | OpenAI GPT-4o | via `openai==1.77.0` |
| Dashboard | Streamlit | ≥1.32.0 |
| Data validation | Python TypedDict | stdlib |
| Data persistence (Sprint 1) | JSON files | flat files |
| Data persistence (Sprint 2) | Supabase | — |
| Integrations (Sprint 2) | Salesforce, Zendesk, Morpheus | — |
| Workflow triggers (Sprint 2) | n8n | — |
| Python version | 3.9+ (3.11+ recommended) | — |

**Environment variables** (see `.env.example`):
```
USE_MOCK_DATA=true          # Set true in Sprint 1 to bypass real API calls
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
```

---

## 4. Lane Classification Rules

These rules are implemented in `agents/coverage_router.py:classify_account()`.
They are **confirmed provisional** — the CS team needs to validate thresholds.

```
RED   (Human-Touch):  churn_score > 70
                   OR sentiment == "negative"
                   OR open_tickets > 3

YELLOW (Tech-Touch):  churn_score in [40, 70]
                   OR days_since_last_contact > 14

GREEN  (Monitor):     churn_score < 40
                  AND sentiment != "negative"
                  AND days_since_last_contact < 14
```

**Priority:** RED rules are evaluated first. If any RED rule fires, the account
is RED regardless of YELLOW signals. YELLOW is evaluated only if no RED rule fires.

---

## 5. Agent 1 — Coverage Router

**File:** `agents/coverage_router.py`
**Public API:** `route_account(account: dict) -> dict`
**No LLM calls.** Pure deterministic rule engine.

### LangGraph graph
```
ingest_account → classify_account → emit_result → END
```

### State schema: `AccountState`
```python
{
    # Input fields
    "account_id":               str,    # e.g. "ACC-001"
    "account_name":             str,    # company name
    "churn_score":              int,    # 0–100
    "sentiment":                str,    # "positive" | "neutral" | "negative"
    "arr":                      float,  # USD
    "days_since_last_contact":  int,
    "open_tickets":             int,

    # Derived fields (set by nodes)
    "lane":                     str,    # "RED" | "YELLOW" | "GREEN"
    "triggered_rules":          list[str],  # e.g. ["churn_score=75 > 70", "sentiment=negative"]
}
```

### Node responsibilities
- **ingest_account**: clamps numerics to valid ranges, normalises sentiment to lowercase.
  Unknown sentiment values default to "neutral" with a warning log.
- **classify_account**: evaluates rules in RED → YELLOW → GREEN priority order.
  All triggered rules are collected (not just the first), so you can see
  multiple signals even though only the highest-priority lane wins.
- **emit_result**: logs the result. Sprint 2 hook: write to Supabase + POST to n8n.

### Important field name difference
`accounts.json` uses `company_name` but the Coverage Router expects `account_name`.
All callers (sprint_review_engine.py, streamlit_app.py) perform this mapping:
```python
router_input = {
    "account_name": account["company_name"],  # ← mapping required
    ...
}
```

---

## 6. Agent 2 — Tech-Touch Agent

**File:** `agents/tech_touch_agent.py`
**Public API:** `run_tech_touch(account: dict, approvals_file: Path = ...) -> dict`
**LLM:** OpenAI GPT-4o (mocked in tests and when `USE_MOCK_DATA=true`).

### LangGraph graph
```
validate_input ──► (lane == YELLOW) ──► load_template ──► generate_message ──► prepare_approval ──► log_result ──► END
               │
               └──► (lane != YELLOW) ──► reject_account ──► END
```

### State schema: `TechTouchState`
```python
{
    # Input fields
    "account_id":    str,
    "company_name":  str,
    "client_name":   str,    # primary contact at the account
    "csm_name":      str,    # Ontop CSM name
    "template_type": str,    # "check_in" | "re_engagement" | "collections" | "milestone" | "save_brief"
    "context":       dict,   # template-specific variables (see section 8)
    "lane":          str,    # must be "YELLOW" — agent rejects RED and GREEN

    # Derived fields
    "prompt_text":       str,   # loaded template with variables filled in
    "generated_subject": str,   # email subject (empty for save_brief)
    "generated_body":    str,   # email body or brief content
    "approval_record":   dict,  # the full approval object written to approvals.json
    "error":             str,   # non-empty signals terminal validation failure
}
```

### Template variable injection
Templates use `{variable_name}` placeholders. The agent fills them using
`_fill_template()` — a **regex-based replacer**, NOT `str.format_map()`.

**Why regex instead of str.format_map?**
The prompt files contain `{"subject": "...", "body": "..."}` JSON at the end
(the response format instruction). `str.format_map()` interprets `{"subject"`
as a format placeholder and crashes. The regex `\{([A-Za-z_]\w*)\}` only matches
pure word-character patterns and ignores JSON syntax like `{"key":`.

### Approval record schema (written to `approvals.json`)
```json
{
    "account_id":    "ACC-003",
    "company_name":  "Axiom Workforce",
    "client_name":   "Maria Lopez",
    "csm_name":      "Andrés",
    "template_type": "check_in",
    "subject":       "...",
    "body":          "...",
    "status":        "pending_approval",
    "generated_at":  "2026-04-14T10:23:00+00:00"
}
```

After CSM action, `status` changes to `"approved"` or `"rejected"`.
`generated_at` is used as the **unique record identifier** for updates.

### What validate_input rejects
- `lane != "YELLOW"` — RED accounts need human CSM; GREEN accounts don't need outreach
- Unknown `template_type` values

### What generate_message does
- `save_brief`: calls GPT-4o with `temperature=0.4`, expects free-form markdown
- All email templates: calls GPT-4o with `response_format={"type": "json_object"}`,
  expects `{"subject": "...", "body": "..."}`, `temperature=0.7`

### approvals_file is injectable
`run_tech_touch(account, approvals_file=Path("/tmp/test.json"))` — used in tests
to write to a temp file instead of the real `approvals.json`.

---

## 7. Agent 3 — Sprint Review Engine

**File:** `agents/sprint_review_engine.py`
**Public API:** `run_sprint_review(accounts_file: Path = ..., reports_dir: Path = ...) -> dict`
**LLM:** OpenAI GPT-4o (mocked in tests and when `USE_MOCK_DATA=true`).

### LangGraph graph
```
load_accounts → re_evaluate → update_lanes → generate_summary → save_report → END
```
No branching. Every sprint review always runs to completion.

### State schema: `SprintReviewState`
```python
{
    # Input / config
    "accounts_file": str,   # absolute path to accounts.json
    "reports_dir":   str,   # absolute path to reports/ directory

    # Populated by load_accounts
    "accounts":  list,      # raw account records

    # Populated by re_evaluate
    # Each entry = original account dict + {new_lane, change_type, triggered_rules}
    "evaluated": list,

    # Populated by generate_summary
    "report":    dict,      # full CSM summary report

    # Populated by save_report
    "report_path": str,     # absolute path of saved report file
}
```

### Lane change vocabulary
```python
_LANE_RISK = {"GREEN": 0, "YELLOW": 1, "RED": 2}  # higher = worse

GRADUATED  — new_lane has lower risk than current_lane  (RED→YELLOW, YELLOW→GREEN)
ESCALATED  — new_lane has higher risk than current_lane (GREEN→YELLOW, YELLOW→RED)
STABLE     — no lane change
```

### Node responsibilities
- **load_accounts**: reads `accounts.json`. Sprint 2: replace with Supabase query.
- **re_evaluate**: calls `route_account()` for every account. Maps `company_name → account_name`.
  Adds `new_lane`, `change_type`, `triggered_rules` to each account dict.
  Does NOT modify `state["accounts"]` — augmented data goes in `state["evaluated"]`.
- **update_lanes**: writes back to `accounts.json`.
  For changed accounts: updates `current_lane`, `lane_assigned_date`, adds `previous_lane`.
  Strips the temporary derived fields (`new_lane`, `change_type`, `triggered_rules`) before writing.
  Does NOT modify `state["evaluated"]` — the change info stays available for `generate_summary`.
- **generate_summary**: builds the report.
  **CODE builds deterministic fields** (counts, lists, dates).
  **GPT-4o builds qualitative fields only** (`csm_action_items`, `narrative_summary`).
  If GPT-4o fails, graceful fallback — structured data is preserved.
- **save_report**: writes `reports/sprint_review_{YYYY-MM-DD}.json`.
  Creates `reports/` if it doesn't exist.
  Sprint 2: POST to n8n webhook for Slack digest.

### Report schema
```json
{
    "report_date":      "2026-04-14",
    "total_accounts":   10,
    "graduated": [
        {"account_id": "ACC-005", "company": "Lumina People",
         "from": "YELLOW", "to": "GREEN", "reason": "No risk rules triggered"}
    ],
    "escalated": [
        {"account_id": "ACC-006", "company": "Crestline Operations",
         "from": "YELLOW", "to": "RED", "reason": "churn_score=80 > 70 | sentiment=negative | open_tickets=5 > 3"}
    ],
    "stable_red":    1,
    "stable_yellow": 2,
    "stable_green":  3,
    "csm_action_items": ["..."],
    "narrative_summary": "..."
}
```

---

## 8. Prompt Templates

All files in `prompts/`. Templates use `{variable_name}` placeholders filled
by `_fill_template()` in `tech_touch_agent.py`.

Three variables are **always available** (from agent state, not `context` dict):
- `{company_name}`, `{client_name}`, `{csm_name}`

Template-specific `context` dict keys:

| Template | Required context keys |
|----------|----------------------|
| `check_in` | `days_since_last_contact`, `usage_trend` |
| `re_engagement` | `previous_contractors`, `active_contractors`, `usage_drop_pct` |
| `collections` | `days_past_due`, `amount_overdue`, `previous_attempts`, `response_deadline`, `escalation_level` (first\|second\|final) |
| `milestone` | `milestone_type`, `milestone_detail` |
| `save_brief` | `arr`, `churn_score`, `open_tickets`, `recent_interactions`, `renewal_date` |

Email templates (`check_in`, `re_engagement`, `collections`, `milestone`) end with:
```
Respond in JSON format only:
{"subject": "...", "body": "..."}
```
This is why the regex-based template filler is required (see section 6).

`save_brief` is **not an email** — it returns structured markdown for CSM internal use.
It does not have a JSON response instruction.

---

## 9. Data: accounts.json

**Path:** `data/accounts.json`
**10 mock accounts** designed so running the Coverage Router produces meaningful
lane changes on the first run.

### Account schema
```json
{
    "account_id":              "ACC-001",
    "company_name":            "Redstone Global",
    "current_lane":            "RED",
    "lane_assigned_date":      "2026-03-17T00:00:00+00:00",
    "churn_score":             85,
    "sentiment":               "negative",
    "arr":                     120000.0,
    "days_since_last_contact": 30,
    "open_tickets":            6,
    "note":                    "Stable RED — high churn...",

    // Added when lane changes (optional fields):
    "previous_lane":           "YELLOW"
}
```

### Expected first-run Coverage Router results
| Account | current_lane | Expected new_lane | Change |
|---------|-------------|-------------------|--------|
| ACC-001 Redstone Global | RED | RED | STABLE |
| ACC-002 Meridian Staffing | RED | YELLOW | GRADUATED |
| ACC-003 Axiom Workforce | YELLOW | YELLOW | STABLE |
| ACC-004 Pulsar Contracts | YELLOW | YELLOW | STABLE |
| ACC-005 Lumina People | YELLOW | GREEN | GRADUATED |
| ACC-006 Crestline Operations | YELLOW | RED | ESCALATED |
| ACC-007 Helix Talent | GREEN | GREEN | STABLE |
| ACC-008 Tempo Hire | GREEN | GREEN | STABLE |
| ACC-009 Navara Consulting | GREEN | YELLOW | ESCALATED |
| ACC-010 Brightpath HR | GREEN | GREEN | STABLE |

**Note:** After running the router or sprint review, `accounts.json` is mutated.
The `note` field is purely for human readability and is ignored by all agents.

---

## 10. Streamlit Dashboard

**File:** `streamlit_app.py`
**Run:** `streamlit run streamlit_app.py`

### Architecture
Single-file app. Layout: sidebar + 3 tabs rendered via named functions.

```
render_sidebar()
tab1 → render_tab_accounts()
tab2 → render_tab_tech_touch()
tab3 → render_tab_sprint_review()
```

### Mock mode
`USE_MOCK_DATA=true` (default) prevents any real OpenAI calls.
Implemented via `_maybe_mock_openai(mock_content)` context manager that patches
`openai.OpenAI` globally using `unittest.mock.patch`.

This works because both agents access OpenAI as `openai.OpenAI()` (attribute
access through the module object), not via `from openai import OpenAI`.
A single `patch("openai.OpenAI", ...)` is sufficient for both agents.

Pre-written mock responses are in `_MOCK_EMAILS` (dict by template_type) and
`_MOCK_SPRINT_SUMMARY` (dict with `csm_action_items` and `narrative_summary`).

### Session state keys
```python
"lane_filter"      # Tab 1 active filter: "ALL" | "RED" | "YELLOW" | "GREEN"
"router_changes"   # None=never run | []=no changes | [...]= list of change dicts
"last_router_run"  # ISO timestamp string
"tt_generated"     # Approval record dict from last Tech-Touch run
"tt_status"        # None | "pending_approval" | "approved" | "rejecting" | "rejected"
"sprint_report"    # Report dict from last in-session sprint review run
"last_sprint_run"  # ISO timestamp string
```

### Tab 1 — Account Overview
- Styled dataframe using `df.style.apply()` with per-row background colours.
  `_lane` column holds the raw lane value for styling, hidden via
  `Styler.hide(subset=["_lane"], axis="columns")`.
- Churn Score uses `st.column_config.ProgressColumn`.
- "🔄 Run Coverage Router" calls `route_account()` directly (no LLM),
  shows a diff card per changed account, persists to `accounts.json`.
- Stores result in `st.session_state.router_changes` before `st.rerun()`.

### Tab 2 — Tech-Touch Agent
- Account dropdown limited to YELLOW-lane accounts.
- `_TEMPLATE_FIELDS` dict defines per-template dynamic form fields
  (type: number | text | select, with defaults).
- On generation: calls `run_tech_touch()` inside `_maybe_mock_openai()`.
  Stores `result["approval_record"]` in `st.session_state.tt_generated`.
- Approval flow states: `pending_approval → approved` or `pending_approval → rejecting → rejected`.
- `update_approval_status(generated_at, status, notes)` uses `generated_at`
  as the unique record key — robust against multiple drafts for same account.

### Tab 3 — Sprint Review
- Reads latest on-disk report via `get_latest_report()` (globs `reports/sprint_review_*.json`).
- "🚀 Run Sprint Review" calls `run_sprint_review()` inside `_maybe_mock_openai()`.
- Results shown in 3 columns: Graduated / Escalated / Stable.
- Narrative summary + action items checklist (`st.checkbox` per item).
- Prefers in-session result over on-disk result so freshly generated report
  is shown immediately without needing to reload from disk.

### Sidebar
- Re-reads `accounts.json` on every render for live counts.
- Shows pending/approved email queue counts from `approvals.json`.
- Shows last Coverage Router run time (from session state) and last Sprint Review
  date (from most recent report file).

---

## 11. Test Suite

**Run all:** `python3 -m pytest tests/ -v`
**Total:** 26 tests, 0 failures

| File | Tests | What's covered |
|------|-------|----------------|
| `test_coverage_router.py` | 9 | All 3 lanes, each trigger rule, RED priority over YELLOW, churn_score clamping, unknown sentiment fallback |
| `test_tech_touch_agent.py` | 7 | Check-in email, re-engagement email, RED rejection, GREEN rejection, approval queue accumulation, unknown template type, required approval record keys |
| `test_sprint_review_engine.py` | 10 | `_classify_change` (all 3 cases), YELLOW→GREEN graduation, GREEN→YELLOW escalation, full report key contract, file persistence, stable account immutability, multi-change batch, LLM failure graceful degradation |

**Test isolation:** All tests that write to disk use `pytest`'s `tmp_path` fixture.
Real `accounts.json` and `approvals.json` are never touched by tests.

**Mocking pattern:**
```python
with patch("agents.tech_touch_agent.openai.OpenAI") as MockOpenAI:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps({"subject": "...", "body": "..."})
    mock_client.chat.completions.create.return_value = mock_response
    MockOpenAI.return_value = mock_client
    result = run_tech_touch(account, approvals_file=approvals_file)
```

For sprint review tests, patch `agents.sprint_review_engine.openai.OpenAI` instead.
The Streamlit app patches `openai.OpenAI` globally (not module-scoped).

---

## 12. Key Design Decisions

### Why LangGraph StateGraph (not plain functions)?
Each agent is a graph so nodes can be independently tested, replaced, or extended
without touching other nodes. Sprint 2 will swap `load_accounts` for a Supabase
call and `save_report` for an n8n webhook — single-node replacements.

### Why TypedDict for state (not Pydantic)?
LangGraph's native pattern. Pydantic adds validation at the cost of
serialisation complexity with LangGraph's internal state management.

### Why `_fill_template()` instead of `str.format_map()`?
Prompt files contain JSON like `{"subject": "...", "body": "..."}` at the end
(the LLM response format instruction). `str.format_map()` treats `{subject}` as
a format placeholder and raises `KeyError`. The regex `\{([A-Za-z_]\w*)\}` only
matches pure identifier patterns and ignores JSON brace syntax.

### Why code builds the deterministic report fields, not the LLM?
LLMs are unreliable at counting and listing. The `generate_summary` node builds
all structured fields (counts, graduated/escalated lists, dates) in Python, then
only asks GPT-4o for `csm_action_items` and `narrative_summary`. This makes the
structured data reliable and testable without mocking.

### Why `generated_at` as unique approval record ID?
Multiple drafts for the same account+template can exist in `approvals.json`.
Using `account_id + template_type` would match the wrong record. `generated_at`
is an ISO timestamp set at generation time — effectively unique.

### Why paths in state (not injected via closures)?
`SprintReviewState` carries `accounts_file` and `reports_dir` as strings.
Every node reads paths from state. This avoids lambda wrappers in the graph
and makes the graph definition clean. Compare to `TechTouchState` which uses
`DEFAULT_APPROVALS_FILE` as a parameter to `build_tech_touch_agent()` — a
slightly different pattern that works but is less clean.

### Why mock with `patch("openai.OpenAI")` globally in the app?
Both agents use `import openai; client = openai.OpenAI()`. Patching
`openai.OpenAI` in the `openai` module namespace intercepts both calls.
If either agent used `from openai import OpenAI; client = OpenAI()`, you'd
need per-module patches.

---

## 13. Sprint 2 Roadmap

### Integration nodes (replace stubs in `integrations/`)
```python
# integrations/salesforce.py
get_account(account_id) -> AccountState dict
list_accounts_by_segment(segment) -> list[AccountState]
update_account_lane(account_id, lane) -> None

# integrations/zendesk.py
get_open_ticket_count(account_id) -> int
get_recent_ticket_sentiment(account_id, days=30) -> str

# integrations/morpheus.py  (internal Ontop API)
get_churn_score(account_id) -> int
get_arr(account_id) -> float
```

### Node replacements
| Current (Sprint 1) | Sprint 2 replacement |
|-------------------|---------------------|
| `load_accounts` reads `accounts.json` | Reads from Supabase |
| `update_lanes` writes to `accounts.json` | Writes to Supabase |
| `save_report` writes to `reports/*.json` | Also POSTs to n8n webhook → Slack |
| `prepare_approval` writes to `approvals.json` | Writes to Supabase |
| `log_result` logs to console | Also POSTs to n8n webhook → Slack notification to CSM |

### Autonomous sending
`tech_touch_agent.py` currently always sets `status = "pending_approval"`.
Sprint 2: add a configurable `autonomous_mode` flag. When enabled and after
quality validation, approved emails are sent automatically via SendGrid/Gmail
through an n8n workflow instead of waiting for CSM click.

### n8n cron trigger
Sprint Review Engine should run every other Monday at 08:00 CST.
The n8n cron payload calls a webhook that triggers `run_sprint_review()`.

### Threshold confirmation
The CS team needs to confirm the lane classification thresholds:
- Is `churn_score > 70` for RED correct, or should it be `> 65`?
- Is `days_since_last_contact > 14` for YELLOW correct?
- Should direct GREEN→RED jumps use a different label than `ESCALATED`?

---

## 14. How to Run Everything

```bash
# Setup
cd "CS Intelligence System"
pip install -r requirements.txt
cp .env.example .env          # USE_MOCK_DATA=true is already set

# Run tests
python3 -m pytest tests/ -v

# Run individual agents (CLI)
python3 agents/coverage_router.py     # smoke test on one hardcoded account
python3 agents/sprint_review_engine.py  # full sprint review, prints report JSON

# Run the dashboard
streamlit run streamlit_app.py
```

**Dashboard flow for a first-time run:**
1. Tab 1 → click "🔄 Run Coverage Router" to classify accounts
2. Tab 2 → select a YELLOW account, pick a template, click "✉️ Generate Email"
3. Tab 2 → click "✅ Approve" to queue the email
4. Tab 3 → click "🚀 Run Sprint Review" to generate the bi-weekly report

---

## 15. Files That Are Auto-Generated (Do Not Edit Manually)

| File | Created by | Contains |
|------|-----------|----------|
| `approvals.json` | `tech_touch_agent.py` | Email approval queue (appended, never overwritten) |
| `reports/sprint_review_*.json` | `sprint_review_engine.py` | One report file per sprint review run |
| `data/accounts.json` | Manually created; mutated by `coverage_router` and `sprint_review_engine` | Current lane assignments |

---

*Last updated: 2026-04-14. 26 tests passing. Sprint 1 complete.*
