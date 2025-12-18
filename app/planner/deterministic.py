# backend/app/planner/deterministic.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

def _safe_float(v: Any) -> float:
    try:
        if v is None:
            return 0.0
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str) and v.strip() == "":
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def compute_surplus(profile: Dict[str, Any]) -> Dict[str, float]:
    """
    Uses your CURRENT Firestore key expectations from app/planner/engine.py:
      - personal.monthlyIncomeInHand
      - personal.annualBonus
      - personal.passiveIncome
      - personal.otherIncome1 / otherIncome2
      - cashFlow.totalMonthlyExpense
      - liabilities.<emi fields>
    """
    personal = profile.get("personal", {}) or {}
    cash_flow = profile.get("cashFlow", {}) or {}
    liabilities = profile.get("liabilities", {}) or {}

    monthly_income_in_hand = _safe_float(personal.get("monthlyIncomeInHand"))
    annual_bonus = _safe_float(personal.get("annualBonus"))
    passive_income = _safe_float(personal.get("passiveIncome"))
    other1 = _safe_float(personal.get("otherIncome1"))
    other2 = _safe_float(personal.get("otherIncome2"))

    annual_income = (
        12 * monthly_income_in_hand
        + annual_bonus
        + 12 * passive_income
        + 12 * other1
        + 12 * other2
    )
    monthly_income_total = annual_income / 12.0

    total_monthly_expense = _safe_float(cash_flow.get("totalMonthlyExpense"))

    emi_fields = [
        "homeLoan1Emi",
        "homeLoan2Emi",
        "personalLoan1Emi",
        "personalLoan2Emi",
        "vehicleLoanEmi",
        "loanAgainstSharesEmi",
        "creditCard1Emi",
        "creditCard2Emi",
        "otherLoanEmi",
    ]
    total_monthly_emi = sum(_safe_float(liabilities.get(f, 0)) for f in emi_fields)

    monthly_surplus = monthly_income_total - total_monthly_expense - total_monthly_emi

    return {
        "annualIncome": annual_income,
        "monthlyIncomeTotal": monthly_income_total,
        "totalMonthlyExpense": total_monthly_expense,
        "totalMonthlyEmi": total_monthly_emi,
        "monthlySurplus": monthly_surplus,
    }


def extract_risk_quiz(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Risk quiz is currently saved under profile.riskQuiz (as per your recent work).
    """
    rq = profile.get("riskQuiz") or {}
    return {
        "riskLabel": rq.get("riskLabel"),
        "totalScore": rq.get("totalScore"),
        "answers": rq.get("answers") or {},
    }


def extract_goals_placeholder(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    IMPORTANT:
    We cannot assume your goals schema yet (need onboarding.js to confirm).
    So Phase-1 stores whatever exists, and marks if mapping is needed.
    """
    goals = profile.get("goals")
    if goals is None:
        return {"exists": False, "needsMapping": True, "data": None}
    return {"exists": True, "needsMapping": False, "data": goals}


def build_financial_blueprint(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic plan blueprint (Phase-1).
    This is NOT the final PDF plan. It’s the math/summary layer that Phase-2 LLM will narrate.
    """
    personal = profile.get("personal", {}) or {}

    risk_quiz = extract_risk_quiz(profile)
    surplus = compute_surplus(profile)
    goals = extract_goals_placeholder(profile)

    out = {
        "meta": {
            "engineVersion": "phase1-blueprint-v1",
            "generatedAt": _now_iso(),
        },
        "profileSummary": {
            "name": personal.get("firstName") or personal.get("name"),
            "age": personal.get("age"),
            "city": personal.get("city"),
            "professionType": personal.get("professionType"),
        },
        "risk": risk_quiz,
        "money": surplus,
        "goals": goals,  # placeholder until we confirm schema in Phase-0
        "notes": [
            "Phase-1 blueprint uses current Firestore keys used by existing planner engine.",
            "Goals are stored as-is until we confirm onboarding goals schema (Phase-0).",
        ],
    }
    return out
