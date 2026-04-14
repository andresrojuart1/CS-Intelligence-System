"""
Tech-Touch Agent
----------------
Generates personalised outreach emails for YELLOW-lane accounts and
queues them for CSM approval before any sending takes place.

This agent is intentionally non-autonomous in Sprint 1: every generated
message lands in approvals.json with status "pending_approval".
Autonomous sending is gated on quality validation in a later sprint.

LangGraph pattern: StateGraph with a validation branch:

  validate_input ──► (YELLOW) ──► load_template
                │                       │
                └──► (not YELLOW) ──►   ▼
                     reject_account  generate_message
                          │               │
                          ▼           prepare_approval
                         END               │
                                       log_result
                                           │
                                          END
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import openai
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

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

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
DEFAULT_APPROVALS_FILE = Path(__file__).parent.parent / "approvals.json"

VALID_TEMPLATE_TYPES = {"check_in", "re_engagement", "collections", "milestone", "save_brief"}

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class TechTouchState(TypedDict):
    """Shared state threaded through every node in the graph."""

    # --- Input fields ---
    account_id: str
    company_name: str
    client_name: str        # Primary contact name at the account
    csm_name: str           # Ontop CSM owning this account
    template_type: str      # "check_in" | "re_engagement" | "collections" | "milestone" | "save_brief"
    context: dict           # Template-specific variables (see prompts/*.txt for expected keys)
    lane: str               # Must be "YELLOW" — agent rejects RED and GREEN

    # --- Derived fields (initialised by validate_input) ---
    prompt_text: str        # Loaded template with variables filled in
    generated_subject: str  # Email subject line (empty for save_brief)
    generated_body: str     # Email body or brief content
    approval_record: dict   # The structured approval/brief object written to disk
    error: str              # Non-empty string signals a terminal validation failure


# ---------------------------------------------------------------------------
# Node 1 — Validate input
# ---------------------------------------------------------------------------


def validate_input(state: TechTouchState) -> TechTouchState:
    """
    Guards the agent against invalid inputs.

    Rejects:
    - Non-YELLOW lanes (RED accounts need human CSM attention;
      GREEN accounts don't need proactive outreach)
    - Unknown template types

    Also initialises all derived fields so downstream nodes always
    find them in state regardless of branch taken.
    """
    logger.info(
        "[TECH-TOUCH] VALIDATE | account=%s  lane=%s  template=%s",
        state["account_id"],
        state["lane"],
        state["template_type"],
    )

    # Initialise derived fields with safe defaults
    state["prompt_text"] = ""
    state["generated_subject"] = ""
    state["generated_body"] = ""
    state["approval_record"] = {}
    state["error"] = ""

    if state["template_type"] not in VALID_TEMPLATE_TYPES:
        state["error"] = (
            f"Unknown template_type='{state['template_type']}'. "
            f"Must be one of: {sorted(VALID_TEMPLATE_TYPES)}."
        )
        return state

    if state["template_type"] == "save_brief":
        if state["lane"] not in {"RED", "YELLOW"}:
            state["error"] = (
                f"save_brief only handles RED or YELLOW lane accounts. "
                f"Received lane='{state['lane']}' for account {state['account_id']}."
            )
        return state

    if state["lane"] != "YELLOW":
        state["error"] = (
            f"Tech-Touch email templates only handle YELLOW lane accounts. "
            f"Received lane='{state['lane']}' for account {state['account_id']}."
        )
        return state

    return state


# ---------------------------------------------------------------------------
# Routing function (used after validate_input)
# ---------------------------------------------------------------------------


def _route_after_validation(state: TechTouchState) -> str:
    """Returns 'proceed' if validation passed, 'reject' otherwise."""
    return "reject" if state["error"] else "proceed"


# ---------------------------------------------------------------------------
# Node 2a — Reject (terminal branch for invalid inputs)
# ---------------------------------------------------------------------------


def reject_account(state: TechTouchState) -> TechTouchState:
    """Logs the rejection reason and exits the graph cleanly."""
    logger.warning(
        "[TECH-TOUCH] REJECTED | account=%s | reason: %s",
        state["account_id"],
        state["error"],
    )
    return state


# ---------------------------------------------------------------------------
# Template filling helper
# ---------------------------------------------------------------------------

def _fill_template(template: str, context: dict) -> str:
    """
    Replaces {variable_name} placeholders in a template string using values
    from context.  Uses a regex replacer instead of str.format_map so that
    literal JSON curly-braces in the prompt (e.g. {"subject": "..."}) are
    left completely untouched.

    Only patterns matching `{word_chars_only}` are treated as placeholders.
    A missing key raises KeyError so the caller can surface it as a config
    error rather than silently sending an unfilled prompt to the LLM.
    """
    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in context:
            raise KeyError(key)
        return str(context[key])

    return re.sub(r"\{([A-Za-z_]\w*)\}", _replace, template)


# ---------------------------------------------------------------------------
# Node 2b — Load template
# ---------------------------------------------------------------------------


def load_template(state: TechTouchState) -> TechTouchState:
    """
    Reads the correct prompt file from /prompts/ and fills in dynamic
    variables from state using Python str.format_map().

    The combined context available to the template is:
      - company_name, client_name, csm_name  (always present)
      - All keys from state["context"]        (template-specific)

    If the template references a variable not present in context,
    format_map raises a KeyError — this surfaces missing context fields
    early rather than silently sending a broken prompt to the LLM.
    """
    template_path = PROMPTS_DIR / f"{state['template_type']}.txt"

    if not template_path.exists():
        state["error"] = f"Prompt file not found: {template_path}"
        logger.error("[TECH-TOUCH] TEMPLATE MISSING | %s", template_path)
        return state

    raw_template = template_path.read_text(encoding="utf-8")

    # Build the full variable map the template can reference
    combined_context = {
        "company_name": state["company_name"],
        "client_name": state["client_name"],
        "csm_name": state["csm_name"],
        **state["context"],
    }

    try:
        state["prompt_text"] = _fill_template(raw_template, combined_context)
    except KeyError as exc:
        state["error"] = f"Missing context variable for template '{state['template_type']}': {exc}"
        logger.error("[TECH-TOUCH] CONTEXT ERROR | %s", state["error"])

    return state


# ---------------------------------------------------------------------------
# Node 3 — Generate message (GPT-4o)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a Customer Success Manager at Ontop, a global payroll and "
    "contractor management platform. Write professional, warm, and concise "
    "outreach. Never mention churn, churn risk, or churn scores. "
    "Never use aggressive or pressuring language. "
    "Email body must be 150 words or fewer."
)


def generate_message(state: TechTouchState) -> TechTouchState:
    """
    Calls OpenAI GPT-4o to generate the email or brief.

    Email templates (all except save_brief):
      - LLM is instructed to respond as JSON: {"subject": "...", "body": "..."}
      - response_format=json_object enforces valid JSON output

    save_brief template:
      - Returns free-form markdown — no subject line
      - Structured sections are defined in the prompt itself
    """
    if state.get("error"):
        # Short-circuit if an upstream node already failed
        return state

    client = openai.OpenAI()  # reads OPENAI_API_KEY from env

    try:
        if state["template_type"] == "save_brief":
            # save_brief is a CSM action brief, not an email
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": state["prompt_text"]},
                ],
                temperature=0.4,  # lower temp for structured internal docs
            )
            state["generated_subject"] = ""
            state["generated_body"] = response.choices[0].message.content.strip()

        else:
            # All email templates — expect JSON back
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": state["prompt_text"]},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            parsed = json.loads(response.choices[0].message.content)
            state["generated_subject"] = parsed.get("subject", "")
            state["generated_body"] = parsed.get("body", "")

    except Exception as exc:  # noqa: BLE001
        state["error"] = f"OpenAI generation failed: {exc}"
        logger.error("[TECH-TOUCH] GENERATION ERROR | account=%s | %s", state["account_id"], exc)

    return state


# ---------------------------------------------------------------------------
# Node 4 — Prepare approval record
# ---------------------------------------------------------------------------


def prepare_approval(
    state: TechTouchState,
    approvals_file: Path = DEFAULT_APPROVALS_FILE,
) -> TechTouchState:
    """
    Writes the generated message to a local approvals.json queue.

    Every record lands with status="pending_approval". The CSM reviews
    and approves/rejects via the approval UI (built in Sprint 2).
    Supabase persistence replaces this file in Sprint 2.
    """
    if state.get("error"):
        return state

    record = {
        "account_id": state["account_id"],
        "company_name": state["company_name"],
        "client_name": state["client_name"],
        "csm_name": state["csm_name"],
        "template_type": state["template_type"],
        "subject": state["generated_subject"],
        "body": state["generated_body"],
        "status": "pending_approval",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Load existing approvals list (or start a new one)
    existing: list[dict] = []
    if approvals_file.exists():
        try:
            existing = json.loads(approvals_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = []

    for pending in existing:
        if (
            pending.get("status") == "pending_approval"
            and pending.get("account_id") == state["account_id"]
            and pending.get("template_type") == state["template_type"]
        ):
            logger.info(
                "[TECH-TOUCH] DEDUPE | account=%s template=%s existing=%s",
                state["account_id"],
                state["template_type"],
                pending.get("generated_at", ""),
            )
            state["generated_subject"] = pending.get("subject", "")
            state["generated_body"] = pending.get("body", "")
            state["approval_record"] = pending
            return state

    existing.append(record)
    approvals_file.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    state["approval_record"] = record
    return state


# ---------------------------------------------------------------------------
# Node 5 — Log result
# ---------------------------------------------------------------------------


def log_result(state: TechTouchState) -> TechTouchState:
    """
    Emits the canonical [TECH-TOUCH] log line for monitoring / n8n triggers.

    Format:  [TECH-TOUCH] account_id | template_type | status | timestamp
    """
    record = state.get("approval_record", {})
    status = record.get("status", "error" if state.get("error") else "unknown")
    timestamp = record.get("generated_at", datetime.now(timezone.utc).isoformat())

    logger.info(
        "[TECH-TOUCH] %s | %s | %s | %s",
        state["account_id"],
        state["template_type"],
        status,
        timestamp,
    )

    if state["generated_subject"]:
        logger.info("  Subject : %s", state["generated_subject"])
    logger.info("  Preview : %.120s...", state["generated_body"])

    # Sprint 2: POST approval_record to n8n webhook for Slack notification to CSM

    return state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_tech_touch_agent(
    approvals_file: Path = DEFAULT_APPROVALS_FILE,
) -> StateGraph:
    """Constructs and compiles the LangGraph StateGraph."""
    graph = StateGraph(TechTouchState)

    # Register nodes — wrap prepare_approval to inject the configurable file path
    graph.add_node("validate_input", validate_input)
    graph.add_node("load_template", load_template)
    graph.add_node("generate_message", generate_message)
    graph.add_node(
        "prepare_approval",
        lambda s: prepare_approval(s, approvals_file=approvals_file),
    )
    graph.add_node("log_result", log_result)
    graph.add_node("reject_account", reject_account)

    # Entry point
    graph.set_entry_point("validate_input")

    # Conditional branch after validation
    graph.add_conditional_edges(
        "validate_input",
        _route_after_validation,
        {
            "proceed": "load_template",
            "reject": "reject_account",
        },
    )

    # Happy path
    graph.add_edge("load_template", "generate_message")
    graph.add_edge("generate_message", "prepare_approval")
    graph.add_edge("prepare_approval", "log_result")
    graph.add_edge("log_result", END)

    # Rejection path
    graph.add_edge("reject_account", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def run_tech_touch(
    account: dict,
    approvals_file: Path = DEFAULT_APPROVALS_FILE,
) -> dict:
    """
    Convenience wrapper — runs the Tech-Touch agent for a single account
    and returns the final state.

    Args:
        account:        Dict matching TechTouchState input fields.
        approvals_file: Path to the approvals JSON file (override in tests).

    Returns:
        Final TechTouchState dict with generated content and approval record.
    """
    agent = build_tech_touch_agent(approvals_file=approvals_file)
    return agent.invoke(account)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = {
        "account_id": "ACC-007",
        "company_name": "Globex Staffing",
        "client_name": "Maria Lopez",
        "csm_name": "Andrés",
        "template_type": "check_in",
        "lane": "YELLOW",
        "context": {
            "days_since_last_contact": 18,
            "usage_trend": "active contractors dropped from 12 to 8 over the past month",
        },
    }
    result = run_tech_touch(sample)
    print(json.dumps(
        {k: result[k] for k in ("account_id", "generated_subject", "generated_body", "error")},
        indent=2,
    ))
