"""
Zendesk Integration (Sprint 2)
---------------------------------
Fetches open ticket counts and recent ticket sentiment for each account.

Planned functions:
  get_open_ticket_count(account_id) → int
  get_recent_ticket_sentiment(account_id, days=30) → str  # "positive"|"neutral"|"negative"

TODO Sprint 2:
  - Use Zendesk REST API with API token auth
  - Aggregate ticket CSAT scores into sentiment signal
"""
