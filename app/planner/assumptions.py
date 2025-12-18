# backend/app/planner/assumptions.py

# Phase-1 placeholders. We'll refine these in Phase-2.
DEFAULT_POST_TAX_RETURN_BY_RISK = {
    "Conservative": 0.08,
    "Balanced": 0.10,
    "Aggressive": 0.12,
}

DEFAULT_EQUITY_DEBT_SPLIT_BY_RISK = {
    "Conservative": {"equity": 0.30, "debt": 0.70},
    "Balanced": {"equity": 0.60, "debt": 0.40},
    "Aggressive": {"equity": 0.80, "debt": 0.20},
}

EMERGENCY_FUND_MONTHS_BY_RISK = {
    "Conservative": 9,
    "Balanced": 6,
    "Aggressive": 6,
}
