"""
Salesforce Integration (Sprint 2)
-----------------------------------
Pulls account health data from Salesforce CRM using simple-salesforce.

Planned functions:
  get_account(account_id) → AccountState dict
  list_accounts_by_segment(segment) → list[AccountState]
  update_account_lane(account_id, lane) → None

TODO Sprint 2:
  - Authenticate via OAuth2 using env vars from .env
  - Map Salesforce object fields to AccountState schema
"""
