"""
Morpheus Integration (Sprint 2)
---------------------------------
Client for Ontop's internal Morpheus API — source of churn_score and ARR.

Planned functions:
  get_churn_score(account_id) → int          # 0–100
  get_arr(account_id) → float                # USD
  get_account_health(account_id) → dict      # combined snapshot

TODO Sprint 2:
  - Confirm endpoint paths with the data team
  - Add retry logic with exponential backoff
"""
