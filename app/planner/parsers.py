# backend/app/planner/parsers.py
from __future__ import annotations

import re
from typing import Any, Optional


_CLEAN_RE = re.compile(r"[,\s]")
_RUPEE_RE = re.compile(r"[₹₹]|rs\.?|inr", re.IGNORECASE)

def parse_inr(value: Any) -> Optional[float]:
    """
    Robust INR parser.
    Handles:
      - 100000, "100000", "1,00,000"
      - "₹1,00,000", "Rs. 1,00,000"
      - "25.4L", "50 Lakhs", "1 Crore", "1 Cr"
    Returns float rupees or None.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if not s:
        return None

    s = _RUPEE_RE.sub("", s).strip().lower()
    s = _CLEAN_RE.sub("", s)

    # Common words
    s = s.replace("lakhs", "lakh").replace("lacs", "lakh").replace("crores", "crore")

    # Examples: "25.4l", "50lakh", "1crore", "1cr"
    m = re.match(r"^([0-9]*\.?[0-9]+)(l|lakh|crore|cr)$", s)
    if m:
        num = float(m.group(1))
        unit = m.group(2)
        if unit in ("l", "lakh"):
            return num * 100000.0
        if unit in ("crore", "cr"):
            return num * 10000000.0

    # Plain number after cleaning
    try:
        return float(s)
    except ValueError:
        return None


def is_annual_key(key: str) -> bool:
    k = key.lower()
    return any(x in k for x in ["annual", "yearly", "peryear", "per_year", "pa", "p.a", "bonus"])


def is_emi_key(key: str) -> bool:
    k = key.lower()
    return "emi" in k


def is_total_key(key: str) -> bool:
    k = key.lower()
    return any(x in k for x in ["total", "sum", "grand"])


def monthly_rate_from_annual(annual_return: float) -> float:
    # Effective monthly rate
    return (1.0 + annual_return) ** (1.0 / 12.0) - 1.0


def sip_required_for_goal(goal_amount: float, months: int, annual_return: float) -> Optional[float]:
    """
    SIP needed to reach goal_amount in months at annual_return.
    FV = P * (( (1+r)^n - 1 ) / r)
    P = FV * r / ( (1+r)^n - 1 )
    """
    if goal_amount <= 0 or months <= 0:
        return None
    r = monthly_rate_from_annual(annual_return)
    if r <= 0:
        return goal_amount / months
    denom = (1.0 + r) ** months - 1.0
    if denom <= 0:
        return None
    return goal_amount * r / denom
